"""Compare FBCCA vs TRCA vs eTRCA across multiple trial durations.

Standalone script that requires no trained model — purely signal-processing
decoder comparison. Useful for understanding the upper bound of each method
at different decoding windows.

Usage:
    python scripts/compare_decoders.py --eeg_dir data/eeg_tensors
    python scripts/compare_decoders.py --eeg_dir data/eeg_tensors --split train
    python scripts/compare_decoders.py --eeg_dir data/eeg_tensors --durations 1.0 1.5 2.0 3.0
    python scripts/compare_decoders.py --eeg_dir data/eeg_tensors --methods fbcca trca
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fbcca import FBCCAFeatureExtractor, BAND_WEIGHTS
from src.trca import FBTRCAClassifier, leave_one_block_out_trca


def compute_fbcca_predictions(eeg_data, labels, trial_duration_pts, device,
                              batch_size=256, latency_pts=0):
    """Run FBCCA on all trials and return top-3 predictions.

    Args:
        eeg_data: (N, C, T) raw EEG
        labels: (N,) true labels
        trial_duration_pts: timepoints to use
        device: torch device
        batch_size: batch size
        latency_pts: skip first N timepoints (SSVEP transient response)

    Returns:
        top3_indices: (N, 3) int64
        top3_scores: (N, 3) float32
    """
    N, C, total_T = eeg_data.shape
    t_end = min(latency_pts + trial_duration_pts, total_T)
    effective_T = t_end - latency_pts
    eeg = eeg_data[:, :, latency_pts:t_end]

    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=effective_T).to(device)
    band_weights = torch.tensor(BAND_WEIGHTS, dtype=torch.float32, device=device)

    all_top3_idx = torch.zeros(N, 3, dtype=torch.int64)
    all_top3_sc = torch.zeros(N, 3, dtype=torch.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = eeg[start:end].to(device)

        raw = fbcca(batch)  # (B, 200)
        B = raw.shape[0]
        corr = raw.reshape(B, fbcca.n_bands, fbcca.n_freqs)
        weighted = (corr * band_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        top3_sc, top3_idx = weighted.topk(3, dim=-1)

        all_top3_idx[start:end] = top3_idx.cpu()
        all_top3_sc[start:end] = top3_sc.cpu().float()

    return all_top3_idx, all_top3_sc


def compute_accuracy(preds, labels, name=""):
    """Compute top-1 and top-3 accuracy."""
    valid = preds[:, 0] >= 0
    if valid.sum() == 0:
        return {"top1": 0.0, "top3": 0.0, "n_valid": 0}

    top1 = (preds[valid, 0] == labels[valid]).float().mean().item()
    top3 = (preds[valid] == labels[valid].unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1": top1, "top3": top3, "n_valid": valid.sum().item()}


def compute_per_subject_accuracy(preds, labels, subject_ids):
    """Compute per-subject top-1 accuracy."""
    results = {}
    for sid in subject_ids.unique().sort().values:
        mask = subject_ids == sid
        valid = (preds[mask, 0] >= 0)
        if valid.sum() > 0:
            acc = (preds[mask][valid, 0] == labels[mask][valid]).float().mean().item()
            results[sid.item()] = {"acc": acc, "n": mask.sum().item()}
    return results


def print_comparison_table(all_results, durations, methods):
    """Print a formatted comparison table."""
    # Header
    header = f"{'Duration':>10}"
    for method in methods:
        header += f" | {method + ' Top1':>12} {method + ' Top3':>12}"
    print("\n" + "=" * len(header))
    print("DECODER COMPARISON: FBCCA vs TRCA vs eTRCA")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for dur in durations:
        row = f"  {dur:5.1f}s   "
        for method in methods:
            key = (dur, method)
            if key in all_results:
                r = all_results[key]
                row += f" | {r['top1']:>11.1%} {r['top3']:>11.1%}"
            else:
                row += f" | {'N/A':>11} {'N/A':>11}"
        print(row)

    print("=" * len(header))


def print_per_subject_table(all_results, durations, methods, subject_ids):
    """Print per-subject breakdown for each duration."""
    unique_subjects = sorted(subject_ids.unique().tolist())

    for dur in durations:
        print(f"\n{'─' * 60}")
        print(f"PER-SUBJECT ACCURACY @ {dur}s ({int(dur*200)}pts)")
        print(f"{'─' * 60}")

        header = f"{'Subject':>8}"
        for method in methods:
            header += f"  {method:>8}"
        print(header)

        for sid in unique_subjects:
            row = f"  S{sid:02d}  "
            for method in methods:
                key = (dur, method, sid)
                if key in all_results:
                    row += f"  {all_results[key]:>7.1%}"
                else:
                    row += f"  {'N/A':>7}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Compare FBCCA vs TRCA vs eTRCA at multiple durations")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--durations", type=float, nargs="*",
                        default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--methods", type=str, nargs="*",
                        default=["fbcca", "trca", "etrca"],
                        choices=["fbcca", "trca", "etrca"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--per_subject", action="store_true",
                        help="Print per-subject breakdown")
    parser.add_argument("--no_latency_skip", action="store_true",
                        help="Disable 0.14s SSVEP latency skip (default: enabled)")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    latency_pts = 0 if args.no_latency_skip else 28
    if latency_pts:
        print(f"Latency skip: {latency_pts}pts ({latency_pts/200:.2f}s)")

    # Load data
    data_path = eeg_dir / f"{args.split}_eeg.pt"
    print(f"Loading {data_path}...")
    data = torch.load(data_path, weights_only=True)
    eeg_data = data["eeg_data"]       # (N, 62, 600)
    labels = data["labels"]            # (N,)
    subject_ids = data["subject_ids"]  # (N,)
    block_ids = data["block_ids"]      # (N,)
    N, C, T = eeg_data.shape
    print(f"  {N} trials, {C} channels, {T} timepoints, "
          f"{len(subject_ids.unique())} subjects")

    all_results = {}      # (dur, method) -> {top1, top3, n_valid}
    subject_results = {}  # (dur, method, sid) -> acc

    for dur in args.durations:
        pts = int(dur * 200)
        print(f"\n{'=' * 60}")
        print(f"Duration: {dur}s ({pts}pts, Δf={200.0/pts:.2f}Hz)")
        print(f"{'=' * 60}")

        for method in args.methods:
            t0 = time.time()
            print(f"\n  [{method.upper()}]")

            if method == "fbcca":
                preds, scores = compute_fbcca_predictions(
                    eeg_data, labels, pts, device, args.batch_size,
                    latency_pts=latency_pts)
            elif method == "trca":
                preds, scores = leave_one_block_out_trca(
                    eeg_data, labels, subject_ids, block_ids,
                    trial_duration_pts=pts, sfreq=200.0, ensemble=False,
                    device=device, batch_size=args.batch_size,
                    latency_pts=latency_pts)
            elif method == "etrca":
                preds, scores = leave_one_block_out_trca(
                    eeg_data, labels, subject_ids, block_ids,
                    trial_duration_pts=pts, sfreq=200.0, ensemble=True,
                    device=device, batch_size=args.batch_size,
                    latency_pts=latency_pts)

            elapsed = time.time() - t0
            acc = compute_accuracy(preds, labels)
            all_results[(dur, method)] = acc
            print(f"  → Top-1: {acc['top1']:.1%}, Top-3: {acc['top3']:.1%} "
                  f"({elapsed:.1f}s)")

            # Per-subject
            if args.per_subject:
                per_subj = compute_per_subject_accuracy(preds, labels, subject_ids)
                for sid, info in per_subj.items():
                    subject_results[(dur, method, sid)] = info["acc"]

    # Summary table
    print_comparison_table(all_results, args.durations, args.methods)

    if args.per_subject:
        print_per_subject_table(subject_results, args.durations,
                                args.methods, subject_ids)

    # Save results
    out_path = eeg_dir / f"decoder_comparison_{args.split}.pt"
    torch.save({
        "results": {str(k): v for k, v in all_results.items()},
        "durations": args.durations,
        "methods": args.methods,
        "split": args.split,
    }, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
