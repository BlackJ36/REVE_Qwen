"""Quick linear probe to test REVE feature quality on 40-class SSVEP.

Tests whether frozen REVE representations can separate 40 targets,
bypassing Qwen entirely. This diagnoses whether the bottleneck is
in REVE features or in the LLM decoding pipeline.

Usage:
    uv run python linear_probe.py
    uv run python linear_probe.py --batch_size 32  # if OOM
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import AutoModel

from src.preprocess import VALID_CHANNEL_NAMES

# All output goes to both console and log file
LOG_FILE = None


def log(msg=""):
    """Print to console and write to log file."""
    print(msg, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


def build_reve(reve_dir, device):
    """Load REVE model and position bank, return (model, positions)."""
    reve_dir = Path(reve_dir)
    log(f"  Loading position bank from {reve_dir / 'reve-positions'}...")
    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    log(f"  Loading REVE model from {reve_dir / 'reve-base'}...")
    reve = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )
    positions = pos_bank(VALID_CHANNEL_NAMES)  # (62, 3)
    reve = reve.to(device)
    reve.requires_grad_(False)
    positions = positions.to(device)
    log(f"  Position shape: {tuple(positions.shape)}")
    log(f"  REVE patch_size={reve.patch_size}, overlap={reve.overlap_size}, "
        f"step={reve.patch_size - reve.overlap_size}")
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
    n_batches = (n + batch_size - 1) // batch_size

    t0 = time.time()
    for idx, i in enumerate(range(0, n, batch_size)):
        batch = eeg_data[i:i + batch_size].float().to(device)
        feat = reve_forward(reve, positions, batch, pool=pool)
        if mean_tokens and not pool:
            feat = feat.mean(dim=1)  # (B, 512)
        features.append(feat.cpu().numpy())

        if (idx + 1) % 10 == 0 or idx == n_batches - 1:
            elapsed = time.time() - t0
            speed = (idx + 1) * batch_size / elapsed
            log(f"    [{idx + 1}/{n_batches}] {speed:.0f} samples/s, "
                f"elapsed {elapsed:.1f}s")

    total_time = time.time() - t0
    log(f"    Done: {n} samples in {total_time:.1f}s ({n / total_time:.0f} samples/s)")
    return np.concatenate(features, axis=0)


def run_probe(name, train_feat, train_labels, val_feat, val_labels):
    """Fit LogisticRegression and report results."""
    log(f"\n{'=' * 60}")
    log(f"  {name}")
    log(f"{'=' * 60}")
    log(f"  Feature shape: {train_feat.shape}")
    log(f"  Feature stats: mean={train_feat.mean():.4f}, std={train_feat.std():.4f}, "
        f"min={train_feat.min():.4f}, max={train_feat.max():.4f}")

    log(f"  Fitting LogisticRegression (max_iter=5000)...")
    t0 = time.time()
    clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
    clf.fit(train_feat, train_labels)
    fit_time = time.time() - t0
    log(f"  Fit completed in {fit_time:.1f}s")

    train_pred = clf.predict(train_feat)
    val_pred = clf.predict(val_feat)

    train_acc = accuracy_score(train_labels, train_pred)
    val_acc = accuracy_score(val_labels, val_pred)
    val_bal = balanced_accuracy_score(val_labels, val_pred)

    log(f"  ----------------------------------------")
    log(f"  Train accuracy:       {train_acc:.4f} ({train_acc * 100:.1f}%)")
    log(f"  Val accuracy:         {val_acc:.4f} ({val_acc * 100:.1f}%)")
    log(f"  Val balanced acc:     {val_bal:.4f} ({val_bal * 100:.1f}%)")
    log(f"  Random baseline:      {1 / 40:.4f} (2.5%)")
    log(f"  ----------------------------------------")

    return {
        "name": name,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "val_balanced_acc": float(val_bal),
        "fit_time_s": round(fit_time, 1),
        "feature_shape": list(train_feat.shape),
    }


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--reve_dir", default="models")
    parser.add_argument("--eeg_dir", default="data/eeg_tensors")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", default="output_probe")
    args = parser.parse_args()

    # Setup output dir and log file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "linear_probe.log"
    LOG_FILE = open(log_path, "w")

    eeg_dir = Path(args.eeg_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log(f"Linear Probe - REVE Feature Quality Diagnostic")
    log(f"Timestamp: {timestamp}")
    log(f"Args: {vars(args)}")
    log(f"Log file: {log_path}")
    log()

    # Load data
    log("Loading data...")
    train_data = torch.load(eeg_dir / "train_eeg.pt", weights_only=True)
    val_data = torch.load(eeg_dir / "val_eeg.pt", weights_only=True)

    train_eeg = train_data["eeg_data"]     # (N, 62, 600)
    train_labels = train_data["labels"].numpy()
    val_eeg = val_data["eeg_data"]
    val_labels = val_data["labels"].numpy()

    n_classes = len(np.unique(train_labels))
    log(f"Train: {len(train_eeg)} trials, Val: {len(val_eeg)} trials")
    log(f"Classes: {n_classes}, EEG shape: {tuple(train_eeg.shape)}")
    log(f"Label distribution (train): {np.bincount(train_labels.astype(int))[:5]}... (first 5)")

    # Load REVE
    log("\nLoading REVE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    reve, positions = build_reve(args.reve_dir, device)
    n_params = sum(p.numel() for p in reve.parameters())
    log(f"  REVE: {n_params:,} params on {device}")

    all_results = []

    # === Experiment 1: attention_pooling (REVE's designed representation) ===
    log("\n[Exp 1/4] Extracting features (attention_pooling, 3s full trial)...")
    train_f1 = extract_features(reve, positions, train_eeg, args.batch_size, pool=True)
    val_f1 = extract_features(reve, positions, val_eeg, args.batch_size, pool=True)
    r1 = run_probe("Exp 1: attention_pooling, 3s trial -> (512,)", train_f1, train_labels, val_f1, val_labels)
    all_results.append(r1)

    # === Experiment 2: mean over 186 tokens ===
    log("\n[Exp 2/4] Extracting features (mean over 186 tokens, 3s full trial)...")
    train_f2 = extract_features(reve, positions, train_eeg, args.batch_size, pool=False, mean_tokens=True)
    val_f2 = extract_features(reve, positions, val_eeg, args.batch_size, pool=False, mean_tokens=True)
    r2 = run_probe("Exp 2: mean(186 tokens), 3s trial -> (512,)", train_f2, train_labels, val_f2, val_labels)
    all_results.append(r2)

    # === Experiment 3: 1.5s window (first 300pts), 62 tokens, mean ===
    log("\n[Exp 3/4] Extracting features (1.5s window [0:300], mean over 62 tokens)...")
    train_win = train_eeg[:, :, :300]  # first 1.5s
    val_win = val_eeg[:, :, :300]
    train_f3 = extract_features(reve, positions, train_win, args.batch_size, pool=False, mean_tokens=True)
    val_f3 = extract_features(reve, positions, val_win, args.batch_size, pool=False, mean_tokens=True)
    r3 = run_probe("Exp 3: 1.5s window, mean(62 tokens) -> (512,)", train_f3, train_labels, val_f3, val_labels)
    all_results.append(r3)

    # === Experiment 4: 1.5s window, attention_pooling ===
    log("\n[Exp 4/4] Extracting features (1.5s window [0:300], attention_pooling)...")
    train_f4 = extract_features(reve, positions, train_win, args.batch_size, pool=True)
    val_f4 = extract_features(reve, positions, val_win, args.batch_size, pool=True)
    r4 = run_probe("Exp 4: 1.5s window, attention_pooling -> (512,)", train_f4, train_labels, val_f4, val_labels)
    all_results.append(r4)

    # === Summary ===
    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    for r in all_results:
        log(f"  {r['name']}")
        log(f"    Val acc: {r['val_acc'] * 100:.1f}%  |  Train acc: {r['train_acc'] * 100:.1f}%")
    log(f"\n  Random baseline: 2.5%")
    log("=" * 60)

    if all(r["val_acc"] < 0.05 for r in all_results):
        log("\n  REVE features show near-random performance on SSVEP.")
        log("  Bottleneck is likely in REVE representations, not LLM decoding.")
    elif any(r["val_acc"] > 0.10 for r in all_results):
        log(f"\n  REVE features are useful for SSVEP!")
        log("  Bottleneck is likely in the LLM decoding pipeline.")

    # Save results JSON
    results_path = output_dir / "linear_probe_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "args": vars(args),
            "results": all_results,
        }, f, indent=2)
    log(f"\nResults saved to {results_path}")
    log(f"Full log saved to {log_path}")

    LOG_FILE.close()


if __name__ == "__main__":
    main()
