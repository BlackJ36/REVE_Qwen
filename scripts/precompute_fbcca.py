"""Precompute FBCCA top-3 predictions for all trials and window offsets.

FBCCA requires GPU but dataset workers run on CPU. This script precomputes
results once so training can do a simple index lookup.

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
    """Precompute FBCCA top-3 for one split."""
    eeg_path = eeg_dir / f"{split}_eeg.pt"
    if not eeg_path.exists():
        print(f"  Skipping {split}: {eeg_path} not found")
        return

    data = torch.load(eeg_path, weights_only=True)
    eeg_data = data["eeg_data"]  # (N, 62, 600)
    N, C, total_T = eeg_data.shape

    # Compute window offsets (same logic as BCIAgentStage1Dataset)
    offsets = []
    start = 0
    while start + window_size <= total_T:
        offsets.append(start)
        start += window_step
    num_offsets = len(offsets)

    print(f"  {split}: {N} trials × {num_offsets} offsets = {N * num_offsets} windows")

    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=window_size).to(device)

    all_top3_indices = torch.zeros(N, num_offsets, 3, dtype=torch.int64)
    all_top3_scores = torch.zeros(N, num_offsets, 3, dtype=torch.float32)

    for oi, offset in enumerate(offsets):
        # Extract windows for this offset: (N, 62, window_size)
        windows = eeg_data[:, :, offset:offset + window_size]

        # Process in batches
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch = windows[start_idx:end_idx].to(device)
            top3_idx, top3_sc = compute_weighted_correlations(fbcca, batch)
            all_top3_indices[start_idx:end_idx, oi] = top3_idx
            all_top3_scores[start_idx:end_idx, oi] = top3_sc

    out_path = eeg_dir / f"{split}_fbcca.pt"
    torch.save({
        "top3_indices": all_top3_indices,
        "top3_scores": all_top3_scores,
    }, out_path)
    print(f"  Saved: {out_path} ({all_top3_indices.shape})")

    # Quick stats
    top1_labels = data["labels"]
    top1_preds = all_top3_indices[:, 0, 0]  # first offset, top-1
    acc = (top1_preds == top1_labels).float().mean().item()
    print(f"  Sanity check — offset=0 top-1 accuracy: {acc:.1%}")


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
    print(f"  window: {args.window_size}pts, step={args.window_step}pts")

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
