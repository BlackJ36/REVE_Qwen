"""Precompute TRCA top-3 predictions for all trials.

Uses leave-one-block-out evaluation: for each subject, each block is held out
while remaining blocks provide calibration data for TRCA spatial filters.

Output format is identical to precompute_fbcca.py so TRCA predictions can be
used as drop-in replacements in the candidate injection pipeline.

Usage:
    python scripts/precompute_trca.py --eeg_dir data/eeg_tensors
    python scripts/precompute_trca.py --eeg_dir data/eeg_tensors --trial_duration 2.0
    python scripts/precompute_trca.py --eeg_dir data/eeg_tensors --durations 1.0 1.5 2.0 3.0
    python scripts/precompute_trca.py --eeg_dir data/eeg_tensors --ensemble  # use eTRCA
    python scripts/precompute_trca.py --ensemble --save_full_scores  # for KD

Output per split:
    {eeg_dir}/{split}_trca.pt       (600pt, backward compatible)
    {eeg_dir}/{split}_trca_{N}pt.pt (other durations)
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trca import leave_one_block_out_trca


def trca_output_filename(split, trial_duration_pts, ensemble=False):
    """Return the TRCA output filename for a given duration."""
    method = "etrca" if ensemble else "trca"
    if trial_duration_pts == 600:
        return f"{split}_{method}.pt"
    return f"{split}_{method}_{trial_duration_pts}pt.pt"


def precompute_split(eeg_dir, split, window_size, window_step, device,
                     trial_duration_pts=600, ensemble=False, batch_size=256,
                     save_full_scores=False):
    """Precompute TRCA top-3 for one split via leave-one-block-out."""
    eeg_path = eeg_dir / f"{split}_eeg.pt"
    if not eeg_path.exists():
        print(f"  Skipping {split}: {eeg_path} not found")
        return

    data = torch.load(eeg_path, weights_only=True)
    eeg_data = data["eeg_data"]      # (N, 62, total_T)
    labels = data["labels"]           # (N,)
    subject_ids = data["subject_ids"] # (N,)
    block_ids = data["block_ids"]     # (N,)
    N, C, total_T = eeg_data.shape

    effective_T = min(trial_duration_pts, total_T)
    method_name = "eTRCA" if ensemble else "TRCA"
    freq_res = 200.0 / effective_T
    print(f"  {split}: {N} trials, {method_name} on {effective_T}pts "
          f"({effective_T/200:.1f}s, Δf={freq_res:.2f}Hz)")

    # Run leave-one-block-out TRCA
    trca_result = leave_one_block_out_trca(
        eeg_data, labels, subject_ids, block_ids,
        trial_duration_pts=trial_duration_pts,
        sfreq=200.0,
        ensemble=ensemble,
        device=device,
        batch_size=batch_size,
        return_full_scores=save_full_scores,
    )
    if save_full_scores:
        all_preds, all_scores, all_full_scores = trca_result
    else:
        all_preds, all_scores = trca_result

    # Compute window offsets for dataset compatibility
    effective_window_size = min(window_size, effective_T)
    offsets = []
    start = 0
    while start + effective_window_size <= effective_T:
        offsets.append(start)
        start += window_step
    if not offsets:
        offsets = [0]
    num_offsets = len(offsets)

    # Broadcast to (N, num_offsets, 3)
    all_top3_indices = all_preds.unsqueeze(1).expand(-1, num_offsets, -1).contiguous()
    all_top3_scores = all_scores.unsqueeze(1).expand(-1, num_offsets, -1).contiguous()

    out_filename = trca_output_filename(split, trial_duration_pts, ensemble)
    out_path = eeg_dir / out_filename
    torch.save({
        "top3_indices": all_top3_indices,
        "top3_scores": all_top3_scores,
    }, out_path)
    print(f"  Saved: {out_path} ({all_top3_indices.shape})")

    # Save full scores for knowledge distillation (per-trial, no offset broadcast)
    if save_full_scores:
        full_filename = out_filename.replace(".pt", "_full.pt")
        full_path = eeg_dir / full_filename
        torch.save({
            "full_scores": all_full_scores,       # (N, 40)
            "top3_indices": all_preds,             # (N, 3)
            "top3_scores": all_scores,             # (N, 3)
        }, full_path)
        print(f"  Saved full scores: {full_path} ({all_full_scores.shape})")

    # Accuracy stats
    valid_mask = all_preds[:, 0] >= 0
    if valid_mask.sum() > 0:
        top1_acc = (all_preds[valid_mask, 0] == labels[valid_mask]).float().mean().item()
        top3_match = (all_preds[valid_mask] == labels[valid_mask].unsqueeze(1)).any(dim=1)
        top3_acc = top3_match.float().mean().item()
        n_valid = valid_mask.sum().item()
        n_skipped = N - n_valid
        print(f"  Top-1 accuracy: {top1_acc:.1%} ({n_valid} valid, {n_skipped} skipped)")
        print(f"  Top-3 accuracy: {top3_acc:.1%}")
    else:
        print(f"  WARNING: no valid predictions")


def main():
    parser = argparse.ArgumentParser(description="Precompute TRCA top-3 for all trials")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--window_step", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trial_duration", type=float, default=3.0,
                        help="Trial duration in seconds (default: 3.0)")
    parser.add_argument("--durations", type=float, nargs="*",
                        help="Batch-precompute multiple durations, e.g. 1.0 1.5 2.0 3.0")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble TRCA (eTRCA) instead of standard TRCA")
    parser.add_argument("--save_full_scores", action="store_true",
                        help="Save full 40-dim correlation scores for knowledge distillation")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    durations = args.durations if args.durations else [args.trial_duration]
    method = "eTRCA" if args.ensemble else "TRCA"

    for duration in durations:
        trial_duration_pts = int(duration * 200)
        print(f"\nPrecomputing {method} top-3 (device={device})")
        print(f"  eeg_dir: {eeg_dir}")
        print(f"  duration: {duration}s ({trial_duration_pts}pts)")
        print(f"  window: {args.window_size}pts, step={args.window_step}pts (for offset count)")

        for split in ["train", "val"]:
            precompute_split(
                eeg_dir, split,
                window_size=args.window_size,
                window_step=args.window_step,
                device=device,
                trial_duration_pts=trial_duration_pts,
                ensemble=args.ensemble,
                batch_size=args.batch_size,
                save_full_scores=args.save_full_scores,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
