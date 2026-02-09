"""Calibrate FBCCA confidence thresholds from precomputed data.

Analyzes the relationship between FBCCA score gap (top1 - top2) and
actual prediction accuracy, then recommends threshold values for
CONF_HIGH / CONF_MID / CONF_LOW tokens.

Usage:
    python scripts/calibrate_fbcca_conf.py --eeg_dir data/eeg_tensors
    python scripts/calibrate_fbcca_conf.py --eeg_dir data/eeg_tensors --write

The --write flag updates src/tokens.py with calibrated thresholds.

Requires: precomputed FBCCA data (run scripts/precompute_fbcca.py first).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze_split(eeg_dir, split):
    """Analyze gap vs accuracy for one split. Returns (gaps, corrects) arrays."""
    eeg_path = eeg_dir / f"{split}_eeg.pt"
    fbcca_path = eeg_dir / f"{split}_fbcca.pt"

    if not fbcca_path.exists():
        print(f"  Skipping {split}: {fbcca_path} not found")
        return None, None

    data = torch.load(eeg_path, weights_only=True)
    labels = data["labels"].numpy()  # (N,)

    fbcca = torch.load(fbcca_path, weights_only=True)
    top3_indices = fbcca["top3_indices"].numpy()  # (N, num_offsets, 3)
    top3_scores = fbcca["top3_scores"].numpy()    # (N, num_offsets, 3)

    N, num_offsets, _ = top3_indices.shape

    # Flatten across all trials × offsets
    top1_pred = top3_indices[:, :, 0].ravel()     # (N*O,)
    top1_score = top3_scores[:, :, 0].ravel()     # (N*O,)
    top2_score = top3_scores[:, :, 1].ravel()     # (N*O,)
    true_labels = np.repeat(labels, num_offsets)   # (N*O,)

    gaps = top1_score - top2_score
    corrects = (top1_pred == true_labels).astype(np.float32)

    print(f"  {split}: {N} trials × {num_offsets} offsets = {len(gaps)} samples")
    print(f"  Overall FBCCA top-1 accuracy: {corrects.mean():.1%}")
    print(f"  Gap range: [{gaps.min():.4f}, {gaps.max():.4f}], "
          f"mean={gaps.mean():.4f}, median={np.median(gaps):.4f}")

    return gaps, corrects


def find_thresholds(gaps, corrects):
    """Find optimal thresholds by analyzing accuracy at different gap levels."""
    # Sort by gap
    order = np.argsort(gaps)
    gaps_sorted = gaps[order]
    corrects_sorted = corrects[order]

    # --- Binned analysis ---
    print("\n  === Gap vs Accuracy (10 equal-count bins) ===")
    print(f"  {'Bin':>12s}  {'Count':>6s}  {'Accuracy':>8s}  {'Cum.Acc':>8s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*8}  {'-'*8}")

    n_bins = 10
    bin_size = len(gaps) // n_bins
    bin_stats = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(gaps)
        bin_gaps = gaps_sorted[start:end]
        bin_correct = corrects_sorted[start:end]
        acc = bin_correct.mean()
        cum_acc = corrects_sorted[start:].mean()  # accuracy from this bin onward
        lo, hi = bin_gaps[0], bin_gaps[-1]
        bin_stats.append((lo, hi, len(bin_correct), acc, cum_acc))
        print(f"  [{lo:>5.3f}, {hi:>5.3f}]  {len(bin_correct):>6d}  {acc:>8.1%}  {cum_acc:>8.1%}")

    # --- Find thresholds based on accuracy breakpoints ---
    # Strategy: find gap values where accuracy crosses 80% and 50%
    # Use rolling window for smoothness
    window = max(len(gaps) // 50, 100)
    rolling_acc = np.convolve(corrects_sorted, np.ones(window)/window, mode='valid')
    rolling_gaps = gaps_sorted[:len(rolling_acc)]

    # Find gap where accuracy first exceeds 80% (HIGH threshold)
    high_mask = rolling_acc >= 0.80
    if high_mask.any():
        high_threshold = rolling_gaps[np.argmax(high_mask)]
    else:
        high_threshold = np.percentile(gaps, 67)

    # Find gap where accuracy first exceeds 50% (MID threshold)
    mid_mask = rolling_acc >= 0.50
    if mid_mask.any():
        mid_threshold = rolling_gaps[np.argmax(mid_mask)]
    else:
        mid_threshold = np.percentile(gaps, 33)

    # --- Also show percentile-based thresholds for comparison ---
    p33 = np.percentile(gaps, 33)
    p50 = np.percentile(gaps, 50)
    p67 = np.percentile(gaps, 67)

    print(f"\n  === Percentiles ===")
    print(f"  P33={p33:.4f}  P50={p50:.4f}  P67={p67:.4f}")

    # Accuracy in each confidence band
    print(f"\n  === Accuracy-based thresholds ===")
    print(f"  HIGH threshold (≥80% acc): gap > {high_threshold:.4f}")
    print(f"  MID  threshold (≥50% acc): gap > {mid_threshold:.4f}")
    print(f"  LOW  threshold (below):    gap ≤ {mid_threshold:.4f}")

    # Verify with these thresholds
    high_mask_data = gaps > high_threshold
    mid_mask_data = (gaps > mid_threshold) & (~high_mask_data)
    low_mask_data = gaps <= mid_threshold

    for name, mask in [("HIGH", high_mask_data), ("MID", mid_mask_data), ("LOW", low_mask_data)]:
        if mask.sum() > 0:
            acc = corrects[mask].mean()
            pct = mask.mean()
            print(f"    {name}: {mask.sum():>6d} samples ({pct:.0%}), accuracy={acc:.1%}")

    return high_threshold, mid_threshold


def main():
    parser = argparse.ArgumentParser(description="Calibrate FBCCA confidence thresholds")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--write", action="store_true",
                        help="Write calibrated thresholds to src/tokens.py")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    print("Calibrating FBCCA confidence thresholds")
    print(f"  eeg_dir: {eeg_dir}\n")

    all_gaps = []
    all_corrects = []

    for split in ["train", "val"]:
        gaps, corrects = analyze_split(eeg_dir, split)
        if gaps is not None:
            all_gaps.append(gaps)
            all_corrects.append(corrects)

    if not all_gaps:
        print("No data found. Run scripts/precompute_fbcca.py first.")
        sys.exit(1)

    gaps = np.concatenate(all_gaps)
    corrects = np.concatenate(all_corrects)
    print(f"\n  Combined: {len(gaps)} total samples")

    high_thresh, mid_thresh = find_thresholds(gaps, corrects)

    # Round to 2 decimal places for cleanliness
    high_thresh = round(float(high_thresh), 2)
    mid_thresh = round(float(mid_thresh), 2)

    print(f"\n{'='*50}")
    print(f"  Recommended thresholds:")
    print(f"    CONF_HIGH: gap > {high_thresh}")
    print(f"    CONF_MID:  {mid_thresh} < gap ≤ {high_thresh}")
    print(f"    CONF_LOW:  gap ≤ {mid_thresh}")
    print(f"{'='*50}")

    if args.write:
        tokens_path = Path("src/tokens.py")
        content = tokens_path.read_text()

        # Replace the thresholds in score_gap_to_conf_token
        import re
        # Match: if gap > 0.15:
        old_high = re.search(r'if gap > ([\d.]+):', content)
        old_mid = re.search(r'elif gap > ([\d.]+):', content)

        if old_high and old_mid:
            old_h = old_high.group(1)
            old_m = old_mid.group(1)
            content = content.replace(f"if gap > {old_h}:", f"if gap > {high_thresh}:")
            content = content.replace(f"elif gap > {old_m}:", f"elif gap > {mid_thresh}:")
            tokens_path.write_text(content)
            print(f"\n  Updated src/tokens.py: {old_h} → {high_thresh}, {old_m} → {mid_thresh}")
        else:
            print("\n  ERROR: Could not find threshold patterns in src/tokens.py")
            sys.exit(1)
    else:
        print(f"\n  To apply: python scripts/calibrate_fbcca_conf.py --eeg_dir {eeg_dir} --write")
        print(f"  Or manually edit src/tokens.py: score_gap_to_conf_token()")


if __name__ == "__main__":
    main()
