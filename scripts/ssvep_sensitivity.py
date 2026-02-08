"""Identify SSVEP-insensitive subjects across Benchmark and BETA datasets.

Runs pure FBCCA argmax (zero-parameter) per subject and ranks them.
Subjects with accuracy near chance (2.5%) have weak SSVEP responses.

Usage:
    uv run python scripts/ssvep_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fbcca import FBCCAFeatureExtractor


def analyze_dataset(name, eeg_dir, fbcca, device, batch_size=2048):
    """Run FBCCA argmax on a dataset and return per-subject accuracy."""
    eeg_dir = Path(eeg_dir)
    results = []

    for split in ["train", "val"]:
        path = eeg_dir / f"{split}_eeg.pt"
        if not path.exists():
            continue
        data = torch.load(path, weights_only=True)
        eeg = data["eeg_data"]
        labels = data["labels"].numpy()
        subjects = data["subject_ids"].numpy()

        # Extract FBCCA features
        features = []
        with torch.no_grad():
            for i in range(0, len(eeg), batch_size):
                batch = eeg[i:i + batch_size].float().to(device)
                features.append(fbcca(batch))
        features = torch.cat(features, dim=0)

        # FBCCA argmax prediction
        n_bands = len(fbcca.band_weights)
        n_freqs = fbcca.n_freqs
        reshaped = features.reshape(-1, n_bands, n_freqs)
        weights = fbcca.band_weights.unsqueeze(0).unsqueeze(-1)
        weighted = (reshaped ** 2 * weights).sum(dim=1)
        preds = weighted.argmax(dim=1).cpu().numpy()

        for subj in np.unique(subjects):
            mask = subjects == subj
            acc = np.mean(preds[mask] == labels[mask])
            n = int(mask.sum())
            results.append({
                "dataset": name,
                "subject": int(subj),
                "split": split,
                "acc": float(acc),
                "n_trials": n,
            })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=600).to(device)

    all_results = []

    bm_dir = Path("data/eeg_tensors_benchmark")
    if bm_dir.exists():
        print("Analyzing Benchmark...", flush=True)
        all_results.extend(analyze_dataset("BM", bm_dir, fbcca, device))

    beta_dir = Path("data/eeg_tensors_beta")
    if beta_dir.exists():
        print("Analyzing BETA...", flush=True)
        all_results.extend(analyze_dataset("BETA", beta_dir, fbcca, device))

    # Sort by accuracy
    all_results.sort(key=lambda x: x["acc"])

    # Print full table
    print(f"\n{'='*65}")
    print(f"  SSVEP Sensitivity: All Subjects (FBCCA Argmax, 3s)")
    print(f"{'='*65}")
    print(f"  {'Dataset':<6s}  {'Subject':>8s}  {'Split':<6s}  {'Acc':>7s}  {'Trials':>6s}  {'Flag'}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}")

    low_subjects = []
    for r in all_results:
        flag = ""
        if r["acc"] < 0.30:
            flag = "** BAD"
            low_subjects.append(r)
        elif r["acc"] < 0.50:
            flag = "* WEAK"
        print(f"  {r['dataset']:<6s}  S{r['subject']:02d}      {r['split']:<6s}  "
              f"{r['acc']*100:6.1f}%  {r['n_trials']:6d}  {flag}")

    # Summary
    bm_accs = [r["acc"] for r in all_results if r["dataset"] == "BM"]
    beta_accs = [r["acc"] for r in all_results if r["dataset"] == "BETA"]

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    if bm_accs:
        print(f"  Benchmark: {len(bm_accs)} subjects, "
              f"mean={np.mean(bm_accs)*100:.1f}%, "
              f"median={np.median(bm_accs)*100:.1f}%, "
              f"range={np.min(bm_accs)*100:.1f}-{np.max(bm_accs)*100:.1f}%")
    if beta_accs:
        print(f"  BETA:      {len(beta_accs)} subjects, "
              f"mean={np.mean(beta_accs)*100:.1f}%, "
              f"median={np.median(beta_accs)*100:.1f}%, "
              f"range={np.min(beta_accs)*100:.1f}-{np.max(beta_accs)*100:.1f}%")

    # Exclusion recommendations
    bad = [r for r in all_results if r["acc"] < 0.30]
    weak = [r for r in all_results if 0.30 <= r["acc"] < 0.50]

    print(f"\n  Thresholds: ** BAD < 30%,  * WEAK < 50%,  random = 2.5%")
    print(f"\n  BAD  (<30%):  {len(bad)} subjects")
    for r in bad:
        print(f"    {r['dataset']} S{r['subject']:02d} = {r['acc']*100:.1f}%")
    print(f"\n  WEAK (<50%):  {len(weak)} subjects")
    for r in weak:
        print(f"    {r['dataset']} S{r['subject']:02d} = {r['acc']*100:.1f}%")

    # Output exclusion lists for easy copy-paste
    bm_exclude = sorted(set(r["subject"] for r in bad + weak if r["dataset"] == "BM"))
    beta_exclude = sorted(set(r["subject"] for r in bad + weak if r["dataset"] == "BETA"))
    print(f"\n  Suggested exclusion (BAD+WEAK < 50%):")
    if bm_exclude:
        print(f"    Benchmark: {','.join(str(s) for s in bm_exclude)}")
    if beta_exclude:
        print(f"    BETA:      {','.join(str(s) for s in beta_exclude)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
