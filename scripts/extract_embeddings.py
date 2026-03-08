"""Pre-extract REVE embeddings from raw EEG tensors (offline).

Loads REVE base model + optional FiLM checkpoint, then extracts
per-channel embeddings for 9 occipital channels.

With --film_ckpt: applies FiLM modulation (FBCCA → γ*tokens+β) during extraction,
  embedding frequency-aware features into the pre-extracted representations.
Without: plain REVE tokens (optionally with LoRA merge).

Input:  data/eeg_tensors/{split}_eeg.pt  — (N, 62, 600) float32
Output: data/embeddings/{split}_embeddings.pt — {embeddings: (N, 9, 512), labels, subject_ids, ...}

Usage:
    # Plain REVE extraction
    uv run python scripts/extract_embeddings.py --reve_dir models --trial_pts 200

    # FiLM-modulated extraction (recommended)
    uv run python scripts/extract_embeddings.py \
        --film_ckpt output_film/film_200_unfreeze4_randoff_60ep/best_model.pt \
        --trial_pts 200

    # With LoRA checkpoint (no FiLM)
    uv run python scripts/extract_embeddings.py --reve_ckpt checkpoints/reve_ssvep_lora16_1s.pt
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel

from src.fbcca import OCCIPITAL_CHANNELS, SSVEP_LATENCY_S, resolve_channel_indices
from src.model_e2e import LoRALinear, REVEWithUnfreeze
from src.preprocess import VALID_CHANNEL_NAMES


def load_reve(reve_dir, reve_ckpt=None, channel_names=None):
    """Load REVE model, optionally with merged SSVEP LoRA checkpoint."""
    reve_dir = Path(reve_dir)

    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    reve_model = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )

    if channel_names is None:
        channel_names = OCCIPITAL_CHANNELS
    wrapper = REVEWithUnfreeze(reve_model, pos_bank, channel_names=channel_names, unfreeze_last_n=0)

    if reve_ckpt is not None:
        print(f"Loading SSVEP checkpoint: {reve_ckpt}")
        ckpt = torch.load(reve_ckpt, map_location="cpu", weights_only=True)
        rank = ckpt["lora_rank"]
        alpha = ckpt["lora_alpha"]
        sd = ckpt["state_dict"]

        # Inject LoRA → load weights → merge back to dense
        wrapper.inject_lora(rank=rank, alpha=alpha)

        # Map checkpoint keys to wrapper keys (prefix with "reve.")
        mapped = {}
        for k, v in sd.items():
            mapped[f"reve.{k}"] = v
        missing, unexpected = wrapper.load_state_dict(mapped, strict=False)
        print(f"  Loaded: {len(mapped)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")

        wrapper.merge_lora()

    wrapper.requires_grad_(False)
    return wrapper


def load_film_model(film_ckpt, reve_dir, trial_pts, film_scale=0.1,
                    gamma_mode="tanh", unfreeze_last_n=4,
                    lora_rank=0, lora_alpha=16,
                    token_gate=False, backbone_channels=None):
    """Load FiLMClassifier from checkpoint for FiLM-modulated extraction.

    Args:
        film_ckpt: path to FiLMClassifier state_dict checkpoint
        reve_dir: directory with reve-base/ and reve-positions/
        trial_pts: timepoints per trial (must match training config)
        film_scale: FiLM amplitude constraint (must match training config)
        gamma_mode: "tanh" or "sigmoid" (must match training config)
        unfreeze_last_n: REVE layers unfrozen during training (structure must match)
        lora_rank: LoRA rank used during training (0 = no LoRA, unfreeze mode)
        lora_alpha: LoRA alpha (only used when lora_rank > 0)
        token_gate: whether token gate was used during training
        backbone_channels: channel names for REVE backbone (None = default 9 occipital)
    """
    from src.film_classifier import build_film_classifier

    model = build_film_classifier(
        reve_dir=reve_dir,
        trial_pts=trial_pts,
        use_film=True,
        unfreeze_last_n=unfreeze_last_n,
        film_scale=film_scale,
        gamma_mode=gamma_mode,
        use_token_gate=token_gate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        backbone_channels=backbone_channels.split(",") if backbone_channels else None,
    )

    ckpt = torch.load(film_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    model.requires_grad_(False)
    model.eval()

    print(f"Loaded FiLM checkpoint: {film_ckpt}")
    print(f"  FiLM scale={model.film_scale}, gamma_mode={model.gamma_mode}")
    return model


@torch.no_grad()
def extract_split(model, eeg_data, ch_idx, trial_pts, latency_pts,
                  batch_size, device, film_model=None):
    """Extract per-channel REVE embeddings for one split.

    Args:
        model: REVEWithUnfreeze (used when film_model is None)
        eeg_data: (N, 62, 600) raw EEG
        ch_idx: list of 9 occipital channel indices into 62-ch array
        trial_pts: timepoints to use (e.g. 200 for 1s)
        latency_pts: SSVEP transient skip (28 pts @ 200Hz)
        batch_size: extraction batch size
        device: cuda/cpu
        film_model: FiLMClassifier instance (None = plain REVE extraction)

    Returns:
        embeddings: (N, C*H, 512) float32 — C=channels, H=patches per channel
                    200pts → H=1 → (N, 9, 512); 400pts → H=2 → (N, 18, 512)
    """
    N = eeg_data.shape[0]
    all_embeddings = []

    for start in tqdm(range(0, N, batch_size), desc="Extracting"):
        batch_eeg = eeg_data[start:start + batch_size]  # (B, 62, 600)

        if film_model is not None:
            # FiLM path: full 62ch → channel selection + FBCCA + modulation
            batch_eeg = batch_eeg[:, :, latency_pts:latency_pts + trial_pts].to(device)

            # REVE backbone: select channels → 4D tokens
            reve_input = batch_eeg[:, film_model.backbone_ch_idx, :]  # (B, C, T)
            tokens_4d = film_model.reve(reve_input, pool="4d")  # (B, C, H, 512)

            # FBCCA: select channels → frequency features
            fbcca_input = batch_eeg[:, film_model.fbcca_ch_idx, :]  # (B, C, T)
            fbcca_out = film_model.fbcca(fbcca_input)  # (B, 200)

            # FiLM modulation: γ * tokens + β
            h = film_model.film_ln(fbcca_out)
            raw_gamma = film_model.film_gamma_proj(h)
            raw_beta = film_model.film_beta_proj(h)

            if film_model.gamma_mode == "sigmoid":
                gamma = (1 - film_model.film_scale) + 2 * film_model.film_scale * torch.sigmoid(raw_gamma)
                beta = film_model.film_scale * torch.tanh(raw_beta)
            else:
                gamma = 1 + film_model.film_scale * torch.tanh(raw_gamma)
                beta = film_model.film_scale * torch.tanh(raw_beta)

            modulated = gamma[:, None, None, :] * tokens_4d + beta[:, None, None, :]

            # Token gate: per-channel importance weighting
            if hasattr(film_model, 'use_token_gate') and film_model.use_token_gate:
                gate = torch.sigmoid(film_model.token_gate_proj(h))  # (B, n_ch)
                modulated = gate[:, :, None, None] * modulated  # (B, C, H, E)

            # Flatten C×H → single token dimension
            B, C, H, E = modulated.shape
            tokens = modulated.reshape(B, C * H, E)  # (B, C*H, 512)
        else:
            # Plain REVE path
            batch_eeg = batch_eeg[:, ch_idx, latency_pts:latency_pts + trial_pts].to(device)
            tokens_4d = model(batch_eeg, pool="4d")  # (B, C, H, 512)
            B, C, H, E = tokens_4d.shape
            tokens = tokens_4d.reshape(B, C * H, E)  # (B, C*H, 512)

        all_embeddings.append(tokens.cpu())

    return torch.cat(all_embeddings, dim=0)  # (N, C*H, 512)


def main():
    parser = argparse.ArgumentParser(description="Pre-extract REVE embeddings")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors",
                        help="Directory with {train,val}_eeg.pt")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--reve_dir", type=str, default="models",
                        help="Directory containing reve-base/ and reve-positions/")
    parser.add_argument("--reve_ckpt", type=str, default=None,
                        help="SSVEP-tuned LoRA checkpoint (None = base REVE)")
    parser.add_argument("--film_ckpt", type=str, default=None,
                        help="FiLM classifier checkpoint for modulated extraction")
    parser.add_argument("--film_scale", type=float, default=0.1,
                        help="FiLM amplitude constraint (must match training)")
    parser.add_argument("--gamma_mode", type=str, default="tanh",
                        choices=["tanh", "sigmoid"],
                        help="FiLM gamma activation (must match training)")
    parser.add_argument("--unfreeze_last_n", type=int, default=4,
                        help="REVE layers unfrozen during FiLM training")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank used during FiLM training (0=unfreeze mode)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (only used when lora_rank > 0)")
    parser.add_argument("--token_gate", action="store_true", default=False,
                        help="Token gate was used during FiLM training")
    parser.add_argument("--backbone_channels", type=str, default=None,
                        help="REVE backbone channels (default: 9 occipital PO7/PO8)")
    parser.add_argument("--trial_pts", type=int, default=200,
                        help="Trial duration in timepoints (200=1s at 200Hz)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Extraction batch size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--splits", type=str, default="train,val",
                        help="Comma-separated splits to process")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve 9 occipital channel indices
    ch_idx = resolve_channel_indices(VALID_CHANNEL_NAMES, OCCIPITAL_CHANNELS)
    ch_names = [VALID_CHANNEL_NAMES[i] for i in ch_idx]
    print(f"Channels ({len(ch_idx)}): {ch_names}")

    latency_pts = int(SSVEP_LATENCY_S * 200)  # 28 pts
    print(f"Trial: {args.trial_pts}pts ({args.trial_pts/200:.1f}s), latency skip: {latency_pts}pts")

    # Load model(s)
    film_model = None
    reve_model = None

    if args.film_ckpt:
        # FiLM-modulated extraction
        film_model = load_film_model(
            args.film_ckpt, args.reve_dir, args.trial_pts,
            film_scale=args.film_scale,
            gamma_mode=args.gamma_mode,
            unfreeze_last_n=args.unfreeze_last_n,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            token_gate=args.token_gate,
            backbone_channels=args.backbone_channels,
        )
        film_model = film_model.to(args.device)
        total_params = sum(p.numel() for p in film_model.parameters()) / 1e6
        print(f"FiLMClassifier on {args.device}, {total_params:.1f}M params")
    else:
        # Plain REVE extraction
        reve_model = load_reve(args.reve_dir, args.reve_ckpt, channel_names=ch_names)
        reve_model = reve_model.to(args.device)
        total_params = sum(p.numel() for p in reve_model.parameters()) / 1e6
        print(f"REVE on {args.device}, {total_params:.1f}M params")

    # Process each split
    for split in args.splits.split(","):
        split = split.strip()
        eeg_path = eeg_dir / f"{split}_eeg.pt"
        if not eeg_path.exists():
            print(f"Skipping {split}: {eeg_path} not found")
            continue

        print(f"\nProcessing {split}...")
        data = torch.load(eeg_path, map_location="cpu", weights_only=True)
        eeg_data = data["eeg_data"]  # (N, 62, 600)
        print(f"  EEG: {eeg_data.shape}")

        embeddings = extract_split(
            reve_model, eeg_data, ch_idx, args.trial_pts, latency_pts,
            args.batch_size, args.device, film_model=film_model,
        )
        n_eeg_tokens = embeddings.shape[1]  # C*H: 9 for 200pts, 18 for 400pts
        print(f"  Embeddings: {embeddings.shape} (n_eeg_tokens={n_eeg_tokens})")

        # Save with metadata
        output = {
            "embeddings": embeddings,
            "labels": data["labels"],
            "subject_ids": data["subject_ids"],
            "block_ids": data["block_ids"],
            "channel_names": ch_names,
            "trial_pts": args.trial_pts,
            "n_eeg_tokens": n_eeg_tokens,
            "film_ckpt": args.film_ckpt,
            "reve_ckpt": args.reve_ckpt,
        }
        if "valid_pts" in data:
            output["valid_pts"] = data["valid_pts"]

        out_path = output_dir / f"{split}_embeddings.pt"
        torch.save(output, out_path)
        print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
