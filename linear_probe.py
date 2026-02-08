"""Quick linear probe to test REVE feature quality on 40-class SSVEP.

Tests whether frozen REVE representations can separate 40 targets,
bypassing Qwen entirely. This diagnoses whether the bottleneck is
in REVE features or in the LLM decoding pipeline.

Usage:
    uv run python linear_probe.py
    uv run python linear_probe.py --batch_size 32  # if OOM
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import AutoModel

from src.preprocess import VALID_CHANNEL_NAMES


def build_reve(reve_dir, device):
    """Load REVE model and position bank, return (model, positions)."""
    reve_dir = Path(reve_dir)
    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    reve = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )
    positions = pos_bank(VALID_CHANNEL_NAMES)  # (62, 3)
    reve = reve.to(device)
    reve.requires_grad_(False)
    positions = positions.to(device)
    return reve, positions


def reve_forward(reve, positions, eeg, pool=True):
    """Run REVE forward pass, return features.

    Args:
        eeg: (B, 62, T) tensor
        pool: if True, use attention_pooling -> (B, 512);
              if False, return all tokens -> (B, N, 512)
    """
    B = eeg.shape[0]
    pos = positions.unsqueeze(0).expand(B, -1, -1)  # (B, 62, 3)

    patches = eeg.unfold(
        dimension=2, size=reve.patch_size,
        step=reve.patch_size - reve.overlap_size,
    )
    _b, c, h, _p = patches.shape

    # 4D positional encoding
    pos_rep = pos.unsqueeze(2).repeat(1, 1, h, 1)
    t = torch.arange(h, device=pos.device, dtype=pos.dtype)
    t = t.view(1, 1, h, 1).expand(B, c, h, 1)
    pos4d = torch.cat([pos_rep, t], dim=-1).view(B, c * h, 4)

    pos_embed = reve.ln(reve.fourier4d(pos4d) + reve.mlp4d(pos4d))
    x = rearrange(
        reve.to_patch_embedding(patches),
        "b c h e -> b (c h) e", c=c, h=h, e=reve.embed_dim,
    ) + pos_embed
    x = reve.transformer(x, False)
    x = rearrange(x, "b (c h) e -> b c h e", b=_b, c=c, h=h, e=reve.embed_dim)
    x = reve.final_layer(x)

    if pool:
        return reve.attention_pooling(x)  # (B, 512)
    else:
        return rearrange(x, "b c h e -> b (c h) e")  # (B, C*h, 512)


@torch.no_grad()
def extract_features(reve, positions, eeg_data, batch_size=64, pool=True, mean_tokens=False):
    """Extract REVE features for all trials."""
    features = []
    n = len(eeg_data)
    device = positions.device

    for i in range(0, n, batch_size):
        batch = eeg_data[i:i + batch_size].float().to(device)
        feat = reve_forward(reve, positions, batch, pool=pool)
        if mean_tokens and not pool:
            feat = feat.mean(dim=1)  # (B, 512)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def run_probe(name, train_feat, train_labels, val_feat, val_labels):
    """Fit LogisticRegression and report results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Feature shape: {train_feat.shape}")

    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1,
    )
    clf.fit(train_feat, train_labels)

    train_acc = accuracy_score(train_labels, clf.predict(train_feat))
    val_acc = accuracy_score(val_labels, clf.predict(val_feat))
    val_bal = balanced_accuracy_score(val_labels, clf.predict(val_feat))

    print(f"  Train accuracy:       {train_acc:.4f} ({train_acc * 100:.1f}%)")
    print(f"  Val accuracy:         {val_acc:.4f} ({val_acc * 100:.1f}%)")
    print(f"  Val balanced acc:     {val_bal:.4f} ({val_bal * 100:.1f}%)")
    print(f"  Random baseline:      {1 / 40:.4f} (2.5%)")
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reve_dir", default="models")
    parser.add_argument("--eeg_dir", default="data/eeg_tensors")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)

    # Load data
    print("Loading data...")
    train_data = torch.load(eeg_dir / "train_eeg.pt", weights_only=True)
    val_data = torch.load(eeg_dir / "val_eeg.pt", weights_only=True)

    train_eeg = train_data["eeg_data"]     # (N, 62, 600)
    train_labels = train_data["labels"].numpy()
    val_eeg = val_data["eeg_data"]
    val_labels = val_data["labels"].numpy()

    n_classes = len(np.unique(train_labels))
    print(f"Train: {len(train_eeg)} trials, Val: {len(val_eeg)} trials")
    print(f"Classes: {n_classes}, EEG shape: {tuple(train_eeg.shape)}")

    # Load REVE
    print("\nLoading REVE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reve, positions = build_reve(args.reve_dir, device)
    n_params = sum(p.numel() for p in reve.parameters())
    print(f"REVE: {n_params:,} params on {device}")

    # === Experiment 1: attention_pooling (REVE's designed representation) ===
    print("\nExtracting features (attention_pooling, 3s full trial)...")
    train_f1 = extract_features(reve, positions, train_eeg, args.batch_size, pool=True)
    val_f1 = extract_features(reve, positions, val_eeg, args.batch_size, pool=True)
    run_probe("Exp 1: attention_pooling, 3s trial -> (512,)", train_f1, train_labels, val_f1, val_labels)

    # === Experiment 2: mean over 186 tokens ===
    print("\nExtracting features (mean over 186 tokens, 3s full trial)...")
    train_f2 = extract_features(reve, positions, train_eeg, args.batch_size, pool=False, mean_tokens=True)
    val_f2 = extract_features(reve, positions, val_eeg, args.batch_size, pool=False, mean_tokens=True)
    run_probe("Exp 2: mean(186 tokens), 3s trial -> (512,)", train_f2, train_labels, val_f2, val_labels)

    # === Experiment 3: 1.5s window (first 300pts), 62 tokens, mean ===
    print("\nExtracting features (1.5s window [0:300], mean over 62 tokens)...")
    train_win = train_eeg[:, :, :300]  # first 1.5s
    val_win = val_eeg[:, :, :300]
    train_f3 = extract_features(reve, positions, train_win, args.batch_size, pool=False, mean_tokens=True)
    val_f3 = extract_features(reve, positions, val_win, args.batch_size, pool=False, mean_tokens=True)
    run_probe("Exp 3: 1.5s window, mean(62 tokens) -> (512,)", train_f3, train_labels, val_f3, val_labels)

    # === Experiment 4: 1.5s window, attention_pooling ===
    print("\nExtracting features (1.5s window [0:300], attention_pooling)...")
    train_f4 = extract_features(reve, positions, train_win, args.batch_size, pool=True)
    val_f4 = extract_features(reve, positions, val_win, args.batch_size, pool=True)
    run_probe("Exp 4: 1.5s window, attention_pooling -> (512,)", train_f4, train_labels, val_f4, val_labels)

    print("\n" + "=" * 60)
    print("Done! If val accuracy >> 2.5%, REVE features are useful for SSVEP.")
    print("If val accuracy ~ 2.5%, REVE may not capture SSVEP frequency features.")


if __name__ == "__main__":
    main()
