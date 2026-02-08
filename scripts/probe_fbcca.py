"""Validate FBCCA features with GPU linear probe on 40-class SSVEP.

Tests whether FBCCA frequency-domain features can separate 40 targets.
Expected: 40-70% on 3s trials, 20-50% on 1.5s windows (FBCCA literature).

Usage:
    uv run python scripts/probe_fbcca.py
    uv run python scripts/probe_fbcca.py --window_size 300  # 1.5s windows only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fbcca import FBCCAFeatureExtractor

LOG_FILE = None


def log(msg=""):
    print(msg, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


@torch.no_grad()
def extract_fbcca_features(fbcca, eeg_data, batch_size=256):
    """Extract FBCCA features for all trials.

    Args:
        fbcca: FBCCAFeatureExtractor module
        eeg_data: (N, 62, T) EEG data
        batch_size: extraction batch size (FBCCA is lightweight, can use large batches)

    Returns:
        features: (N, 200) on GPU
    """
    device = next(fbcca.buffers()).device
    features = []
    n = len(eeg_data)
    n_batches = (n + batch_size - 1) // batch_size

    t0 = time.time()
    for idx, i in enumerate(range(0, n, batch_size)):
        batch = eeg_data[i:i + batch_size].float().to(device)
        feat = fbcca(batch)  # (B, 200)
        features.append(feat)

        if (idx + 1) % 10 == 0 or idx == n_batches - 1:
            elapsed = time.time() - t0
            speed = (idx + 1) * batch_size / elapsed
            log(f"    [{idx + 1}/{n_batches}] {speed:.0f} samples/s, elapsed {elapsed:.1f}s")

    total_time = time.time() - t0
    log(f"    Done: {n} samples in {total_time:.1f}s ({n / total_time:.0f} samples/s)")
    return torch.cat(features, dim=0)


def run_probe(name, train_feat, train_labels, val_feat, val_labels,
              n_classes=40, epochs=100, lr=0.01, batch_size=1024):
    """Train a linear probe on GPU and report results."""
    log(f"\n{'=' * 60}")
    log(f"  {name}")
    log(f"{'=' * 60}")
    log(f"  Feature shape: {tuple(train_feat.shape)}")
    log(f"  Feature stats: mean={train_feat.mean():.4f}, std={train_feat.std():.4f}, "
        f"min={train_feat.min():.4f}, max={train_feat.max():.4f}")

    device = train_feat.device
    feat_dim = train_feat.shape[1]

    # Normalize features
    mean = train_feat.mean(dim=0, keepdim=True)
    std = train_feat.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_feat - mean) / std
    val_norm = (val_feat - mean) / std

    train_labels_t = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long, device=device)

    linear = nn.Linear(feat_dim, n_classes).to(device)
    optimizer = torch.optim.SGD(linear.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_norm, train_labels_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    log(f"  Training: {feat_dim} -> {n_classes}, epochs={epochs}, lr={lr}")

    t0 = time.time()
    best_val_acc = 0.0

    for epoch in range(epochs):
        linear.train()
        total_loss = 0.0
        for feat_batch, label_batch in train_loader:
            logits = linear(feat_batch)
            loss = criterion(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            linear.eval()
            with torch.no_grad():
                train_logits = linear(train_norm)
                val_logits = linear(val_norm)
                train_acc = (train_logits.argmax(1) == train_labels_t).float().mean().item()
                val_acc = (val_logits.argmax(1) == val_labels_t).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            avg_loss = total_loss / len(train_loader)
            log(f"    Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.4f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    fit_time = time.time() - t0

    linear.eval()
    with torch.no_grad():
        val_pred = linear(val_norm).argmax(1)
        train_pred = linear(train_norm).argmax(1)
        train_acc = (train_pred == train_labels_t).float().mean().item()
        val_acc = (val_pred == val_labels_t).float().mean().item()
        per_class_acc = []
        for c in range(n_classes):
            mask = val_labels_t == c
            if mask.sum() > 0:
                per_class_acc.append((val_pred[mask] == c).float().mean().item())
        val_bal = np.mean(per_class_acc)

    log(f"  ----------------------------------------")
    log(f"  Train accuracy:       {train_acc:.4f} ({train_acc * 100:.1f}%)")
    log(f"  Val accuracy:         {val_acc:.4f} ({val_acc * 100:.1f}%)")
    log(f"  Val balanced acc:     {val_bal:.4f} ({val_bal * 100:.1f}%)")
    log(f"  Best val accuracy:    {best_val_acc:.4f} ({best_val_acc * 100:.1f}%)")
    log(f"  Random baseline:      {1 / 40:.4f} (2.5%)")
    log(f"  Time: {fit_time:.1f}s")

    return {
        "name": name,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "best_val_acc": float(best_val_acc),
        "val_balanced_acc": float(val_bal),
        "fit_time_s": round(fit_time, 1),
        "feature_shape": list(train_feat.shape),
    }


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_dir", default="data/eeg_tensors")
    parser.add_argument("--output_dir", default="output_probe")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=None,
                        help="If set, test only this window size (e.g. 300 for 1.5s)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "fbcca_probe.log"
    LOG_FILE = open(log_path, "w")

    eeg_dir = Path(args.eeg_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log("FBCCA Linear Probe - Frequency Feature Validation")
    log(f"Timestamp: {timestamp}")
    log(f"Args: {vars(args)}")
    log()

    # Load data
    log("Loading data...")
    train_data = torch.load(eeg_dir / "train_eeg.pt", weights_only=True)
    val_data = torch.load(eeg_dir / "val_eeg.pt", weights_only=True)

    train_eeg = train_data["eeg_data"]  # (N, 62, 600)
    train_labels = train_data["labels"].numpy()
    val_eeg = val_data["eeg_data"]
    val_labels = val_data["labels"].numpy()

    n_classes = len(np.unique(train_labels))
    log(f"Train: {len(train_eeg)} trials, Val: {len(val_eeg)} trials")
    log(f"Classes: {n_classes}, EEG shape: {tuple(train_eeg.shape)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []

    # === Experiment 1: Full 3s trial (600 pts @ 200Hz) ===
    if args.window_size is None or args.window_size == 600:
        log("\n[Exp 1] FBCCA features on full 3s trial (600 pts)...")
        fbcca_3s = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=600).to(device)
        train_f1 = extract_fbcca_features(fbcca_3s, train_eeg, args.batch_size)
        val_f1 = extract_fbcca_features(fbcca_3s, val_eeg, args.batch_size)
        r1 = run_probe("FBCCA 3s trial -> (200,)", train_f1, train_labels, val_f1, val_labels)
        all_results.append(r1)

    # === Experiment 2: 1.5s window (300 pts, first window) ===
    if args.window_size is None or args.window_size == 300:
        log("\n[Exp 2] FBCCA features on 1.5s window (300 pts, first window)...")
        fbcca_15s = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=300).to(device)
        train_win = train_eeg[:, :, :300]
        val_win = val_eeg[:, :, :300]
        train_f2 = extract_fbcca_features(fbcca_15s, train_win, args.batch_size)
        val_f2 = extract_fbcca_features(fbcca_15s, val_win, args.batch_size)
        r2 = run_probe("FBCCA 1.5s window -> (200,)", train_f2, train_labels, val_f2, val_labels)
        all_results.append(r2)

    # === Experiment 3: FBCCA-only argmax (no linear probe, pure CCA) ===
    if args.window_size is None:
        log("\n[Exp 3] Pure FBCCA argmax (no learning, 3s trial)...")
        fbcca_3s = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=600).to(device)
        train_f3 = extract_fbcca_features(fbcca_3s, train_eeg, args.batch_size)
        val_f3 = extract_fbcca_features(fbcca_3s, val_eeg, args.batch_size)

        n_bands = len(fbcca_3s.band_weights)
        n_freqs = fbcca_3s.n_freqs

        # Weighted sum across sub-bands: w_k * rho_k^2
        val_reshaped = val_f3.reshape(-1, n_bands, n_freqs)  # (B, 5, 40)
        weights = fbcca_3s.band_weights.unsqueeze(0).unsqueeze(-1)  # (1, 5, 1)
        weighted = (val_reshaped ** 2 * weights).sum(dim=1)  # (B, 40)
        val_pred = weighted.argmax(dim=1).cpu().numpy()

        val_labels_np = val_data["labels"].numpy()
        val_acc = np.mean(val_pred == val_labels_np)

        per_class_acc = []
        for c in range(40):
            mask = val_labels_np == c
            if mask.sum() > 0:
                per_class_acc.append(np.mean(val_pred[mask] == c))
        val_bal = np.mean(per_class_acc)

        log(f"  Pure FBCCA argmax (3s):")
        log(f"    Val accuracy:     {val_acc:.4f} ({val_acc * 100:.1f}%)")
        log(f"    Val balanced acc: {val_bal:.4f} ({val_bal * 100:.1f}%)")
        log(f"    Random baseline:  {1 / 40:.4f} (2.5%)")

        all_results.append({
            "name": "Pure FBCCA argmax, 3s trial",
            "val_acc": float(val_acc),
            "val_balanced_acc": float(val_bal),
        })

    # Summary
    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    for r in all_results:
        log(f"  {r['name']}")
        log(f"    Val acc: {r['val_acc'] * 100:.1f}%")
    log(f"\n  Random baseline: 2.5%")
    log("=" * 60)

    results_path = output_dir / "fbcca_probe_results.json"
    with open(results_path, "w") as f:
        json.dump({"timestamp": timestamp, "args": vars(args), "results": all_results}, f, indent=2)
    log(f"\nResults saved to {results_path}")
    log(f"Log saved to {log_path}")

    LOG_FILE.close()


if __name__ == "__main__":
    main()
