"""Analyze evaluation results for alphabet characters only (A-Z, classes 0-25).

Usage:
    python scripts/analyze_alpha_only.py output_ablation_reve_candidate_s2/final/eval_predictions.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics_bci_agent import compute_fbcca_correction_metrics
from src.templates_zh import KEYBOARD_CHARS


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_alpha_only.py <eval_predictions.npz>")
        sys.exit(1)

    data = np.load(sys.argv[1])
    model_preds = data["model_preds"]
    model_probs = data["model_probs"]
    true_labels = data["true_labels"]
    fbcca_top1 = data["fbcca_top1"]
    subject_ids = data["subject_ids"]

    # Filter to A-Z only (classes 0-25)
    alpha_mask = true_labels < 26
    mp = model_preds[alpha_mask]
    mprob = model_probs[alpha_mask]
    tl = true_labels[alpha_mask]
    fb = fbcca_top1[alpha_mask]
    sids = subject_ids[alpha_mask]

    N = len(tl)
    correction = compute_fbcca_correction_metrics(mp, tl, fb)

    print("=" * 60)
    print("ALPHA ONLY (A-Z, classes 0-25)")
    print("=" * 60)
    print(f"  Trials:           {N}")
    print(f"  FBCCA accuracy:   {correction['fbcca_acc']:.1%}")
    print(f"  Model accuracy:   {correction['model_acc']:.1%}")
    print(f"  Override rate:    {correction['override_rate']:.1%}")
    print(f"  Correction rate:  {correction['correction_rate']:.1%}  (FBCCA wrong -> model right)")
    print(f"  Trust rate:       {correction['trust_rate']:.1%}  (FBCCA right -> model agrees)")
    print(f"  FBCCA errors:     {correction['correction_count']}")

    # Top-5
    top5_preds = np.argsort(-mprob, axis=1)[:, :5]
    top5_hit = np.any(top5_preds == tl[:, None], axis=1)
    print(f"  Model top-5 acc:  {top5_hit.mean():.1%}")

    # Per-subject
    print(f"\n{'-' * 60}")
    print("PER-SUBJECT (alpha only)")
    print(f"{'-' * 60}")
    print(f"{'Subject':>8} {'Trials':>7} {'FBCCA':>7} {'Model':>7} {'Corrn':>7} {'Trust':>7}")

    for sid in sorted(np.unique(sids)):
        m = sids == sid
        n = m.sum()
        if n == 0:
            continue
        s_mp, s_tl, s_fb = mp[m], tl[m], fb[m]
        s_fbcca_acc = (s_fb == s_tl).mean()
        s_model_acc = (s_mp == s_tl).mean()

        fbcca_wrong = s_fb != s_tl
        s_corr = (s_mp[fbcca_wrong] == s_tl[fbcca_wrong]).mean() if fbcca_wrong.sum() > 0 else float('nan')
        fbcca_right = s_fb == s_tl
        s_trust = (s_mp[fbcca_right] == s_fb[fbcca_right]).mean() if fbcca_right.sum() > 0 else float('nan')

        print(f"  S{sid:02d}   {n:>6}  {s_fbcca_acc:>6.1%}  {s_model_acc:>6.1%}  "
              f"{s_corr:>6.1%}  {s_trust:>6.1%}")

    # Per-class (worst 10 among alpha)
    print(f"\n{'-' * 60}")
    print("PER-CLASS (alpha only, worst 10)")
    print(f"{'-' * 60}")
    per_class = []
    for c in range(26):
        m = tl == c
        if m.sum() > 0:
            acc = (mp[m] == c).mean()
            per_class.append((c, KEYBOARD_CHARS[c], acc, m.sum()))
    per_class.sort(key=lambda x: x[2])
    for c, char, acc, n in per_class[:10]:
        print(f"  Class {c:2d} ({char}): {acc:.1%}  ({n} trials)")

    # Also show non-alpha summary
    non_alpha_mask = true_labels >= 26
    if non_alpha_mask.sum() > 0:
        na_mp = model_preds[non_alpha_mask]
        na_tl = true_labels[non_alpha_mask]
        na_fb = fbcca_top1[non_alpha_mask]
        print(f"\n{'-' * 60}")
        print("NON-ALPHA (classes 26-39) for comparison")
        print(f"{'-' * 60}")
        print(f"  Trials:         {non_alpha_mask.sum()}")
        print(f"  FBCCA accuracy: {(na_fb == na_tl).mean():.1%}")
        print(f"  Model accuracy: {(na_mp == na_tl).mean():.1%}")


if __name__ == "__main__":
    main()
