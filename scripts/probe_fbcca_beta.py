"""FBCCA validation on BETA dataset: cross-subject and per-subject accuracy.

Tests:
  1. Pure FBCCA argmax (no learning) — cross-subject and per-subject
  2. Linear probe — cross-subject and per-subject breakdown

Usage:
    uv run python scripts/probe_fbcca_beta.py
    uv run python scripts/probe_fbcca_beta.py --eeg_dir data/eeg_tensors_beta
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fbcca import FBCCAFeatureExtractor


def log(msg=""):
    print(msg, flush=True)


@torch.no_grad()
def extract_fbcca_features(fbcca, eeg_data, batch_size=2048):
    """Extract FBCCA features. Returns (N, 200) on GPU."""
    device = next(fbcca.buffers()).device
    features = []
    for i in range(0, len(eeg_data), batch_size):
        batch = eeg_data[i:i + batch_size].float().to(device)
        features.append(fbcca(batch))
    return torch.cat(features, dim=0)


@torch.no_grad()
def fbcca_argmax_predict(fbcca, features):
    """Weighted FBCCA argmax prediction (no learning).

    Args:
        fbcca: FBCCAFeatureExtractor (for band_weights, n_freqs)
        features: (N, 200) raw FBCCA correlation features

    Returns:
        predictions: (N,) numpy array of predicted labels
    """
    n_bands = len(fbcca.band_weights)
    n_freqs = fbcca.n_freqs
    reshaped = features.reshape(-1, n_bands, n_freqs)  # (N, 5, 40)
    weights = fbcca.band_weights.unsqueeze(0).unsqueeze(-1)  # (1, 5, 1)
    weighted = (reshaped ** 2 * weights).sum(dim=1)  # (N, 40)
    return weighted.argmax(dim=1).cpu().numpy()


def per_subject_accuracy(predictions, labels, subject_ids):
    """Compute accuracy per unique subject. Returns dict {subj_id: (acc, n_trials)}."""
    results = {}
    for subj in sorted(np.unique(subject_ids)):
        mask = subject_ids == subj
        n = mask.sum()
        acc = np.mean(predictions[mask] == labels[mask])
        results[int(subj)] = (float(acc), int(n))
    return results


def train_linear_probe(train_feat, train_labels, n_classes=40, epochs=100, lr=0.01, batch_size=1024):
    """Train a linear probe and return (model, mean, std) for inference."""
    device = train_feat.device
    feat_dim = train_feat.shape[1]

    mean = train_feat.mean(dim=0, keepdim=True)
    std = train_feat.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_feat - mean) / std

    train_labels_t = torch.tensor(train_labels, dtype=torch.long, device=device)

    linear = nn.Linear(feat_dim, n_classes).to(device)
    optimizer = torch.optim.SGD(linear.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_norm, train_labels_t)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        linear.train()
        for feat_batch, label_batch in loader:
            logits = linear(feat_batch)
            loss = criterion(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            linear.eval()
            with torch.no_grad():
                acc = (linear(train_norm).argmax(1) == train_labels_t).float().mean().item()
            log(f"    Epoch {epoch+1:3d}/{epochs}: train_acc={acc:.4f}")

    linear.eval()
    return linear, mean, std


@torch.no_grad()
def probe_predict(linear, features, mean, std):
    """Predict with trained linear probe."""
    normed = (features - mean) / std
    return linear(normed).argmax(1).cpu().numpy()


def print_per_subject_table(per_subj, title):
    """Print a formatted per-subject accuracy table."""
    log(f"\n  {title}")
    log(f"  {'Subject':>8s}  {'Acc':>7s}  {'Trials':>6s}")
    log(f"  {'─' * 8}  {'─' * 7}  {'─' * 6}")
    accs = []
    for subj, (acc, n) in per_subj.items():
        log(f"  S{subj:02d}      {acc*100:6.1f}%  {n:6d}")
        accs.append(acc)
    log(f"  {'─' * 8}  {'─' * 7}  {'─' * 6}")
    log(f"  {'Mean':>8s}  {np.mean(accs)*100:6.1f}%")
    log(f"  {'Std':>8s}  {np.std(accs)*100:6.1f}%")
    log(f"  {'Min':>8s}  {np.min(accs)*100:6.1f}%")
    log(f"  {'Max':>8s}  {np.max(accs)*100:6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_dir", default="data/eeg_tensors_beta")
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)

    log("=" * 60)
    log("  FBCCA Validation on BETA Dataset")
    log("=" * 60)

    # Load data
    train_data = torch.load(eeg_dir / "train_eeg.pt", weights_only=True)
    val_data = torch.load(eeg_dir / "val_eeg.pt", weights_only=True)

    train_eeg = train_data["eeg_data"]
    train_labels = train_data["labels"].numpy()
    train_subjects = train_data["subject_ids"].numpy()
    val_eeg = val_data["eeg_data"]
    val_labels = val_data["labels"].numpy()
    val_subjects = val_data["subject_ids"].numpy()

    train_subj_range = np.unique(train_subjects)
    val_subj_range = np.unique(val_subjects)
    log(f"  Train: {len(train_eeg)} trials, subjects S{train_subj_range[0]:02d}-S{train_subj_range[-1]:02d}")
    log(f"  Val:   {len(val_eeg)} trials, subjects S{val_subj_range[0]:02d}-S{val_subj_range[-1]:02d}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"  Device: {torch.cuda.get_device_name(0)}")

    # Extract FBCCA features
    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=600).to(device)

    log("\nExtracting FBCCA features (3s trial)...")
    t0 = time.time()
    train_feat = extract_fbcca_features(fbcca, train_eeg, args.batch_size)
    val_feat = extract_fbcca_features(fbcca, val_eeg, args.batch_size)
    log(f"  Done in {time.time()-t0:.1f}s, feature shape: {tuple(val_feat.shape)}")

    # ========================================
    # Exp 1: Pure FBCCA argmax (no learning)
    # ========================================
    log("\n" + "=" * 60)
    log("  Exp 1: Pure FBCCA Argmax (no learning)")
    log("=" * 60)

    # Cross-subject
    val_pred = fbcca_argmax_predict(fbcca, val_feat)
    cross_acc = np.mean(val_pred == val_labels)
    log(f"\n  Cross-subject val accuracy: {cross_acc*100:.1f}%  (random: 2.5%)")

    # Per-subject (val)
    val_per_subj = per_subject_accuracy(val_pred, val_labels, val_subjects)
    print_per_subject_table(val_per_subj, "Per-Subject Val (S61-S70) — FBCCA Argmax")

    # Per-subject (train) — to see full range
    train_pred = fbcca_argmax_predict(fbcca, train_feat)
    train_per_subj = per_subject_accuracy(train_pred, train_labels, train_subjects)
    print_per_subject_table(train_per_subj, "Per-Subject Train (S01-S60) — FBCCA Argmax")

    # ========================================
    # Exp 2: Linear probe (cross-subject)
    # ========================================
    log("\n" + "=" * 60)
    log("  Exp 2: Linear Probe (train on S01-S60, eval on S61-S70)")
    log("=" * 60)

    log("\n  Training linear probe...")
    t0 = time.time()
    linear, mean, std = train_linear_probe(train_feat, train_labels)
    log(f"  Trained in {time.time()-t0:.1f}s")

    # Cross-subject
    val_probe_pred = probe_predict(linear, val_feat, mean, std)
    probe_cross_acc = np.mean(val_probe_pred == val_labels)
    log(f"\n  Cross-subject val accuracy: {probe_cross_acc*100:.1f}%  (random: 2.5%)")

    # Per-subject (val)
    val_probe_per_subj = per_subject_accuracy(val_probe_pred, val_labels, val_subjects)
    print_per_subject_table(val_probe_per_subj, "Per-Subject Val (S61-S70) — Linear Probe")

    # ========================================
    # Summary
    # ========================================
    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    val_argmax_accs = [a for a, _ in val_per_subj.values()]
    val_probe_accs = [a for a, _ in val_probe_per_subj.values()]
    log(f"  {'Method':<20s}  {'Cross-Subj':>10s}  {'Mean+-Std':>14s}  {'Range':>14s}")
    log(f"  {'─'*20}  {'─'*10}  {'─'*14}  {'─'*14}")
    log(f"  {'FBCCA Argmax':<20s}  {cross_acc*100:9.1f}%  "
        f"{np.mean(val_argmax_accs)*100:5.1f}+-{np.std(val_argmax_accs)*100:4.1f}%  "
        f"{np.min(val_argmax_accs)*100:5.1f}-{np.max(val_argmax_accs)*100:4.1f}%")
    log(f"  {'Linear Probe':<20s}  {probe_cross_acc*100:9.1f}%  "
        f"{np.mean(val_probe_accs)*100:5.1f}+-{np.std(val_probe_accs)*100:4.1f}%  "
        f"{np.min(val_probe_accs)*100:5.1f}-{np.max(val_probe_accs)*100:4.1f}%")
    log(f"  {'Random':<20s}  {'2.5%':>10s}")
    log("=" * 60)


if __name__ == "__main__":
    main()
