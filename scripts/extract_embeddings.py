"""Pre-extract REVE embeddings from raw EEG tensors (offline).

Loads REVE base model + optional SSVEP-tuned LoRA checkpoint, then extracts
per-channel embeddings for 9 occipital channels.

Input:  data/eeg_tensors/{split}_eeg.pt  — (N, 62, 600) float32
Output: data/embeddings/{split}_embeddings.pt — {embeddings: (N, 9, 512), labels, subject_ids, ...}

Usage:
    uv run python scripts/extract_embeddings.py --reve_dir models --trial_pts 200
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


@torch.no_grad()
def extract_split(model, eeg_data, ch_idx, trial_pts, latency_pts, batch_size, device):
    """Extract per-channel REVE embeddings for one split.

    Args:
        model: REVEWithUnfreeze (frozen, merged)
        eeg_data: (N, 62, 600) raw EEG
        ch_idx: list of 9 occipital channel indices into 62-ch array
        trial_pts: timepoints to use (e.g. 200 for 1s)
        latency_pts: SSVEP transient skip (28 pts @ 200Hz)
        batch_size: extraction batch size
        device: cuda/cpu

    Returns:
        embeddings: (N, 9, 512) float32
    """
    model.eval()
    N = eeg_data.shape[0]
    all_embeddings = []

    for start in tqdm(range(0, N, batch_size), desc="Extracting"):
        batch_eeg = eeg_data[start:start + batch_size]  # (B, 62, 600)
        # Select 9 occipital channels + crop to trial window
        batch_eeg = batch_eeg[:, ch_idx, latency_pts:latency_pts + trial_pts]  # (B, 9, trial_pts)
        batch_eeg = batch_eeg.to(device)

        # REVE forward: pool="4d" → (B, 9, H, 512), H=1 for 200pts
        tokens_4d = model(batch_eeg, pool="4d")  # (B, 9, 1, 512)
        tokens = tokens_4d.squeeze(2)  # (B, 9, 512)
        all_embeddings.append(tokens.cpu())

    return torch.cat(all_embeddings, dim=0)  # (N, 9, 512)


def main():
    parser = argparse.ArgumentParser(description="Pre-extract REVE embeddings")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors",
                        help="Directory with {train,val}_eeg.pt")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--reve_dir", type=str, default="models",
                        help="Directory containing reve-base/ and reve-positions/")
    parser.add_argument("--reve_ckpt", type=str, default="checkpoints/reve_ssvep_lora16_1s.pt",
                        help="SSVEP-tuned LoRA checkpoint (None = base REVE)")
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

    # Load model
    model = load_reve(args.reve_dir, args.reve_ckpt, channel_names=ch_names)
    model = model.to(args.device)
    print(f"REVE on {args.device}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

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
            model, eeg_data, ch_idx, args.trial_pts, latency_pts,
            args.batch_size, args.device,
        )
        print(f"  Embeddings: {embeddings.shape}")  # (N, 9, 512)

        # Save with metadata
        output = {
            "embeddings": embeddings,
            "labels": data["labels"],
            "subject_ids": data["subject_ids"],
            "block_ids": data["block_ids"],
            "channel_names": ch_names,
            "trial_pts": args.trial_pts,
            "reve_ckpt": args.reve_ckpt,
        }
        if "valid_pts" in data:
            output["valid_pts"] = data["valid_pts"]

        out_path = output_dir / f"{split}_embeddings.pt"
        torch.save(output, out_path)
        print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
