"""Evaluate a trained FiLM checkpoint per-subject on all data.

Loads the 65.9% checkpoint and reports accuracy for every subject,
without retraining. Much faster than LOSO.

Usage:
  python scripts/eval_per_subject.py --checkpoint output_film/film_200_unfreeze4_randoff_60ep/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_bci_agent import BETA_BAD_SUBJECTS
from src.dataset_reve_finetune import LOSODataset, reve_finetune_collate_fn
from src.film_classifier import build_film_classifier

BETA_BAD_REMAPPED = {s + 100 for s in BETA_BAD_SUBJECTS}


def main():
    parser = argparse.ArgumentParser(description="Per-subject evaluation of FiLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bm_dir", type=str, default="data/eeg_tensors_benchmark")
    parser.add_argument("--beta_dir", type=str, default="data/eeg_tensors_beta")
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--trial_pts", type=int, default=200)
    parser.add_argument("--unfreeze_last_n", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "benchmark", "beta"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model and load checkpoint
    print("Building model...")
    model = build_film_classifier(
        reve_dir=args.reve_dir,
        trial_pts=args.trial_pts,
        use_film=True,
        unfreeze_last_n=args.unfreeze_last_n,
    )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    all_subjects = LOSODataset.get_all_subjects(
        dataset_filter=args.dataset,
        exclude_subjects=BETA_BAD_REMAPPED,
    )

    results = []

    for sid in all_subjects:
        ds = LOSODataset(
            bm_dir=args.bm_dir, beta_dir=args.beta_dir,
            leave_out_subject=sid, is_train=False,
            trial_duration_pts=args.trial_pts,
            exclude_subjects=BETA_BAD_REMAPPED,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=reve_finetune_collate_fn,
        )

        correct, top5_correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in loader:
                eeg = batch["eeg"].to(device)
                labels = batch["labels"].to(device)
                logits = model(eeg)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                top5 = logits.topk(5, dim=-1).indices
                top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        acc = correct / total
        top5_acc = top5_correct / total
        dataset_name = "benchmark" if sid <= 35 else "beta"
        label = f"BM_S{sid:02d}" if sid <= 35 else f"BETA_S{sid-100:02d}"

        results.append({
            "subject_id": sid,
            "subject_label": label,
            "dataset": dataset_name,
            "acc": acc,
            "top5": top5_acc,
            "n_trials": total,
        })

    # Print table
    print(f"\n{'=' * 65}")
    print(f"Per-Subject Eval ({len(results)} subjects)")
    print(f"{'=' * 65}")
    print(f"{'Subject':<12} {'Dataset':<10} {'Acc':>7} {'Top5':>7} {'Trials':>7}")
    print(f"{'-' * 65}")

    bm_accs, beta_accs = [], []
    for r in results:
        print(f"{r['subject_label']:<12} {r['dataset']:<10} "
              f"{r['acc']:>6.1%} {r['top5']:>6.1%} {r['n_trials']:>7}")
        if r["dataset"] == "benchmark":
            bm_accs.append(r["acc"])
        else:
            beta_accs.append(r["acc"])

    all_accs = torch.tensor([r["acc"] for r in results])
    all_top5 = torch.tensor([r["top5"] for r in results])
    print(f"{'-' * 65}")
    print(f"{'Overall':<12} {'':10} {all_accs.mean().item():>6.1%} {all_top5.mean().item():>6.1%}")
    if bm_accs:
        bm_t = torch.tensor(bm_accs)
        print(f"{'Benchmark':<12} {'':10} {bm_t.mean().item():>6.1%} (std={bm_t.std().item():.1%}, n={len(bm_accs)})")
    if beta_accs:
        bt_t = torch.tensor(beta_accs)
        print(f"{'BETA':<12} {'':10} {bt_t.mean().item():>6.1%} (std={bt_t.std().item():.1%}, n={len(beta_accs)})")
    print(f"{'=' * 65}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "overall_acc": all_accs.mean().item(),
                "overall_top5": all_top5.mean().item(),
                "per_subject": results,
            }, f, indent=2)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
