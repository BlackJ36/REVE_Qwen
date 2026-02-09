"""Precompute FBCCA top-3 predictions for all trials.

FBCCA requires GPU but dataset workers run on CPU. This script precomputes
results once so training can do a simple index lookup.

FBCCA is computed on the **full 600-point trial** (3s @ 200Hz) for maximum
frequency resolution, then the result is broadcast across all sliding window
offsets. This is valid because SSVEP is a steady-state response — the
frequency content is stable across the trial.

Usage:
    python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors
    python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors --window_size 300 --window_step 100

Output per split:
    {eeg_dir}/{split}_fbcca.pt = {
        "top3_indices": (N, num_offsets, 3)  int64,
        "top3_scores":  (N, num_offsets, 3)  float32,
    }
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fbcca import FBCCAFeatureExtractor, BAND_WEIGHTS


def compute_weighted_correlations(fbcca_module, eeg_batch):
    """Run FBCCA and return weighted per-frequency correlations.

    The FBCCAFeatureExtractor.forward() returns (B, 200) = 5 bands × 40 freqs
    flattened. We need to aggregate across bands to get (B, 40) scores,
    then take top-3 per trial.

    Args:
        fbcca_module: FBCCAFeatureExtractor on GPU
        eeg_batch: (B, 62, T) on GPU

    Returns:
        top3_indices: (B, 3) int64 — top-3 target label indices
        top3_scores:  (B, 3) float32 — corresponding weighted correlations
    """
    # (B, 200) = (B, 5*40)
    raw = fbcca_module(eeg_batch)
    B = raw.shape[0]
    n_bands = fbcca_module.n_bands
    n_freqs = fbcca_module.n_freqs

    # Reshape to (B, n_bands, n_freqs) and weight across bands
    corr = raw.reshape(B, n_bands, n_freqs)
    weights = fbcca_module.band_weights.to(corr.device)  # (n_bands,)
    weighted = (corr * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, 40)

    # Top-3 per trial
    top3_scores, top3_indices = weighted.topk(3, dim=-1)
    return top3_indices.cpu().to(torch.int64), top3_scores.cpu().to(torch.float32)


def precompute_split(eeg_dir, split, window_size, window_step, batch_size, device):
    """Precompute FBCCA top-3 for one split.

    FBCCA is computed on the full trial (all 600 timepoints) for maximum
    frequency resolution. The result is then broadcast across all sliding
    window offsets so the dataset can index by [trial_idx, offset_idx].
    """
    eeg_path = eeg_dir / f"{split}_eeg.pt"
    if not eeg_path.exists():
        print(f"  Skipping {split}: {eeg_path} not found")
        return

    data = torch.load(eeg_path, weights_only=True)
    eeg_data = data["eeg_data"]  # (N, 62, 600)
    N, C, total_T = eeg_data.shape

    # Compute window offsets (for output shape compatibility with dataset)
    offsets = []
    start = 0
    while start + window_size <= total_T:
        offsets.append(start)
        start += window_step
    num_offsets = len(offsets)

    print(f"  {split}: {N} trials, FBCCA on full {total_T}pts, broadcast to {num_offsets} offsets")

    # FBCCA on full trial length for maximum frequency resolution
    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=total_T).to(device)

    # Compute once per trial: (N, 3)
    trial_top3_indices = torch.zeros(N, 3, dtype=torch.int64)
    trial_top3_scores = torch.zeros(N, 3, dtype=torch.float32)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch = eeg_data[start_idx:end_idx].to(device)  # (B, 62, 600)
        top3_idx, top3_sc = compute_weighted_correlations(fbcca, batch)
        trial_top3_indices[start_idx:end_idx] = top3_idx
        trial_top3_scores[start_idx:end_idx] = top3_sc

    # Broadcast to (N, num_offsets, 3) for dataset compatibility
    all_top3_indices = trial_top3_indices.unsqueeze(1).expand(-1, num_offsets, -1).contiguous()
    all_top3_scores = trial_top3_scores.unsqueeze(1).expand(-1, num_offsets, -1).contiguous()

    out_path = eeg_dir / f"{split}_fbcca.pt"
    torch.save({
        "top3_indices": all_top3_indices,
        "top3_scores": all_top3_scores,
    }, out_path)
    print(f"  Saved: {out_path} ({all_top3_indices.shape})")

    # Quick stats
    top1_labels = data["labels"]
    top1_preds = trial_top3_indices[:, 0]  # top-1 prediction
    acc = (top1_preds == top1_labels).float().mean().item()
    print(f"  Sanity check — full-trial top-1 accuracy: {acc:.1%}")

    # Top-3 accuracy
    top3_match = (trial_top3_indices == top1_labels.unsqueeze(1)).any(dim=1)
    top3_acc = top3_match.float().mean().item()
    print(f"  Top-3 accuracy: {top3_acc:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Precompute FBCCA top-3 for all trials")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--window_step", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Precomputing FBCCA top-3 (device={device})")
    print(f"  eeg_dir: {eeg_dir}")
    print(f"  FBCCA on full trial (600pts), broadcast to window offsets")
    print(f"  window: {args.window_size}pts, step={args.window_step}pts (for offset count)")

    for split in ["train", "val"]:
        precompute_split(
            eeg_dir, split,
            window_size=args.window_size,
            window_step=args.window_step,
            batch_size=args.batch_size,
            device=device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
