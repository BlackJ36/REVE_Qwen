"""Precompute FBCCA top-3 predictions for all trials.

FBCCA requires GPU but dataset workers run on CPU. This script precomputes
results once so training can do a simple index lookup.

Uses 9 occipital channels + 0.14s latency skip for optimal SSVEP detection.
62ch→9ch + latency skip improves accuracy dramatically (e.g. 1s: 15%→50%).

Usage:
    python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors
    python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors --trial_duration 2.0
    python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors --durations 1.0 1.5 2.0 3.0
    python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors --all_channels  # disable 9ch

Output per split:
    {eeg_dir}/{split}_fbcca.pt       (600pt, backward compatible)
    {eeg_dir}/{split}_fbcca_{N}pt.pt (other durations)
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fbcca import (
    FBCCAFeatureExtractor, BAND_WEIGHTS,
    SSVEP_LATENCY_S, resolve_channel_indices,
)


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


def fbcca_output_filename(split, trial_duration_pts):
    """Return the FBCCA output filename for a given duration.

    600pt → {split}_fbcca.pt (backward compatible)
    Other → {split}_fbcca_{N}pt.pt
    """
    if trial_duration_pts == 600:
        return f"{split}_fbcca.pt"
    return f"{split}_fbcca_{trial_duration_pts}pt.pt"


def precompute_split(eeg_dir, split, window_size, window_step, batch_size, device,
                     trial_duration_pts=600, channel_indices=None, latency_pts=0):
    """Precompute FBCCA top-3 for one split.

    FBCCA is computed on the (possibly truncated) trial data with optional
    channel selection and latency skip. The result is then broadcast across
    all sliding window offsets so the dataset can index by [trial_idx, offset_idx].

    Args:
        trial_duration_pts: number of timepoints to use (e.g. 600 for 3s,
            400 for 2s, 300 for 1.5s, 200 for 1s @ 200Hz).
        channel_indices: list of channel indices for FBCCA (None = all channels).
        latency_pts: skip first N timepoints (SSVEP transient response).
    """
    eeg_path = eeg_dir / f"{split}_eeg.pt"
    if not eeg_path.exists():
        print(f"  Skipping {split}: {eeg_path} not found")
        return

    data = torch.load(eeg_path, weights_only=True)
    eeg_data = data["eeg_data"]  # (N, 62, total_T)
    N, C, total_T = eeg_data.shape

    # Select channels for FBCCA (e.g. 9 occipital channels)
    if channel_indices is not None:
        eeg_data = eeg_data[:, channel_indices, :]
        n_ch = len(channel_indices)
    else:
        n_ch = C

    # Apply latency skip + truncate to requested duration
    t_start = latency_pts
    t_end = t_start + trial_duration_pts
    if t_end > total_T:
        t_end = total_T
    eeg_data = eeg_data[:, :, t_start:t_end]
    effective_T = t_end - t_start

    # Compute window offsets based on the ORIGINAL trial length (for dataset compat)
    # Offsets are computed on the full trial_duration_pts, not the latency-skipped version
    trunc_T = min(trial_duration_pts, total_T)
    effective_window_size = min(window_size, trunc_T)
    offsets = []
    start = 0
    while start + effective_window_size <= trunc_T:
        offsets.append(start)
        start += window_step
    if not offsets:
        offsets = [0]
    num_offsets = len(offsets)

    freq_res = 200.0 / effective_T
    ch_info = f"{n_ch}ch" if channel_indices else f"{C}ch(all)"
    lat_info = f"+{latency_pts}pt latency skip" if latency_pts else ""
    print(f"  {split}: {N} trials, FBCCA on {effective_T}pts "
          f"({effective_T/200:.1f}s, Δf={freq_res:.2f}Hz, {ch_info}{lat_info}), "
          f"broadcast to {num_offsets} offsets")

    # FBCCA on the effective trial length
    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=effective_T).to(device)

    # Compute once per trial: (N, 3)
    trial_top3_indices = torch.zeros(N, 3, dtype=torch.int64)
    trial_top3_scores = torch.zeros(N, 3, dtype=torch.float32)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch = eeg_data[start_idx:end_idx].to(device)
        top3_idx, top3_sc = compute_weighted_correlations(fbcca, batch)
        trial_top3_indices[start_idx:end_idx] = top3_idx
        trial_top3_scores[start_idx:end_idx] = top3_sc

    # Broadcast to (N, num_offsets, 3) for dataset compatibility
    all_top3_indices = trial_top3_indices.unsqueeze(1).expand(-1, num_offsets, -1).contiguous()
    all_top3_scores = trial_top3_scores.unsqueeze(1).expand(-1, num_offsets, -1).contiguous()

    out_filename = fbcca_output_filename(split, trial_duration_pts)
    out_path = eeg_dir / out_filename
    torch.save({
        "top3_indices": all_top3_indices,
        "top3_scores": all_top3_scores,
    }, out_path)
    print(f"  Saved: {out_path} ({all_top3_indices.shape})")

    # Quick stats
    top1_labels = data["labels"]
    top1_preds = trial_top3_indices[:, 0]  # top-1 prediction
    acc = (top1_preds == top1_labels).float().mean().item()
    print(f"  Sanity check — top-1 accuracy: {acc:.1%}")

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
    parser.add_argument("--trial_duration", type=float, default=3.0,
                        help="Trial duration in seconds (default: 3.0)")
    parser.add_argument("--durations", type=float, nargs="*",
                        help="Batch-precompute multiple durations, e.g. 1.0 1.5 2.0 3.0")
    parser.add_argument("--all_channels", action="store_true",
                        help="Use all channels instead of 9 occipital (default: 9ch)")
    parser.add_argument("--no_latency_skip", action="store_true",
                        help="Disable 0.14s SSVEP latency skip (default: enabled)")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve occipital channel indices from meta.json
    channel_indices = None
    if not args.all_channels:
        meta_path = eeg_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            channel_names = meta.get("channel_names", [])
            if channel_names:
                channel_indices = resolve_channel_indices(channel_names)
                print(f"Using {len(channel_indices)} occipital channels: "
                      f"{[channel_names[i] for i in channel_indices]}")
            else:
                print("WARNING: No channel_names in meta.json, using all channels")
        else:
            print(f"WARNING: {meta_path} not found, using all channels")

    # Latency skip
    latency_pts = 0
    if not args.no_latency_skip:
        latency_pts = int(SSVEP_LATENCY_S * 200)
        print(f"Latency skip: {latency_pts}pts ({SSVEP_LATENCY_S}s)")

    # Determine which durations to compute
    if args.durations:
        durations = args.durations
    else:
        durations = [args.trial_duration]

    for duration in durations:
        trial_duration_pts = int(duration * 200)
        print(f"\nPrecomputing FBCCA top-3 (device={device})")
        print(f"  eeg_dir: {eeg_dir}")
        print(f"  duration: {duration}s ({trial_duration_pts}pts)")
        print(f"  window: {args.window_size}pts, step={args.window_step}pts (for offset count)")

        for split in ["train", "val"]:
            precompute_split(
                eeg_dir, split,
                window_size=args.window_size,
                window_step=args.window_step,
                batch_size=args.batch_size,
                device=device,
                trial_duration_pts=trial_duration_pts,
                channel_indices=channel_indices,
                latency_pts=latency_pts,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
