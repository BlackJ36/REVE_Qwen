"""Run FiLM ablation: 4 configs on selected LOSO folds.

Usage:
  python scripts/run_film_ablation.py                    # Run all configs on fold 1
  python scripts/run_film_ablation.py --folds 1 3 8 20 33  # Multiple folds
  python scripts/run_film_ablation.py --summary          # Print results comparison
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONFIGS = {
    "A_baseline":  {"film_scale": 0.1, "film_reg_weight": 0.01, "gamma_mode": "tanh"},
    "B_scale02":   {"film_scale": 0.2, "film_reg_weight": 1e-4, "gamma_mode": "tanh"},
    "C_sigmoid":   {"film_scale": 0.2, "film_reg_weight": 1e-4, "gamma_mode": "sigmoid"},
    "D_tokengate": {"film_scale": 0.2, "film_reg_weight": 1e-4, "gamma_mode": "tanh", "token_gate": True},
}


def run_ablation(base_dir, folds, common_args):
    for config_name, cfg in CONFIGS.items():
        ckpt_dir = base_dir / config_name
        print(f"\n{'='*60}")
        print(f"Config: {config_name} | {cfg}")
        print(f"{'='*60}")

        for fold in folds:
            cmd = [
                sys.executable, "scripts/loso_film.py",
                "--checkpoint_dir", str(ckpt_dir),
                "--dataset", "benchmark",
                "--force",
                "--film_scale", str(cfg["film_scale"]),
                "--film_reg_weight", str(cfg["film_reg_weight"]),
                "--gamma_mode", cfg["gamma_mode"],
                "--start_fold", str(fold),
                "--end_fold", str(fold),
            ]
            if cfg.get("token_gate"):
                cmd.append("--token_gate")
            cmd += common_args

            subprocess.run(cmd, check=False)


def print_summary(base_dir, folds):
    print(f"\n{'='*70}")
    print(f"FiLM Ablation Results (fold 1 = BM_S01)")
    print(f"{'='*70}")

    # Header
    header = f"{'Fold':<8}"
    for name in CONFIGS:
        header += f"{name:<16}"
    print(header)
    print("-" * 70)

    # Per-fold results
    all_accs = {name: [] for name in CONFIGS}
    for fold in folds:
        row = f"S{fold:02d}     "
        for name in CONFIGS:
            summary_path = base_dir / name / f"fold_{fold:03d}" / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    s = json.load(f)
                acc = s["val_acc"]
                all_accs[name].append(acc)
                row += f"{acc:>6.1%}{'':10}"
            else:
                row += f"{'--':<16}"
        print(row)

    # Mean
    print("-" * 70)
    row = f"{'Mean':<8}"
    for name in CONFIGS:
        accs = all_accs[name]
        if accs:
            mean = sum(accs) / len(accs)
            row += f"{mean:>6.1%}{'':10}"
        else:
            row += f"{'--':<16}"
    print(row)
    print(f"{'='*70}")

    # Also print train/val details from last epoch of each config
    print(f"\nDetailed last-epoch stats:")
    for name in CONFIGS:
        summary_path = base_dir / name / f"fold_{folds[0]:03d}" / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            print(f"  {name}: val_acc={s['val_acc']:.1%}, val_top5={s['val_top5']:.1%}, "
                  f"epochs={s['epochs']}, val_loss={s['val_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="FiLM ablation experiment")
    parser.add_argument("--base_dir", type=str, default="/tmp/film_ablation")
    parser.add_argument("--folds", type=int, nargs="+", default=[1])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--summary", action="store_true", help="Print results only")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.summary:
        print_summary(base_dir, args.folds)
        return

    common_args = [
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]

    run_ablation(base_dir, args.folds, common_args)
    print_summary(base_dir, args.folds)


if __name__ == "__main__":
    main()
