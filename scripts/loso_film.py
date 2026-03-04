"""LOSO (Leave-One-Subject-Out) cross-validation for FiLM classifier.

Trains one model per subject, using all other subjects for training.
Aggregates per-subject accuracy for proper generalization metrics.

Usage:
  # Quick test: single fold
  python scripts/loso_film.py --checkpoint_dir /data/zjj/loso_film --start_fold 1 --end_fold 1

  # Full run (95 subjects, ~95 * 15min on single GPU)
  python scripts/loso_film.py --checkpoint_dir /data/zjj/loso_film

  # Resume from fold 35
  python scripts/loso_film.py --checkpoint_dir /data/zjj/loso_film --start_fold 35
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_bci_agent import BETA_BAD_SUBJECTS
from src.dataset_reve_finetune import LOSODataset, reve_finetune_collate_fn
from src.film_classifier import build_film_classifier


# Remapped bad subjects (BETA +100)
BETA_BAD_REMAPPED = {s + 100 for s in BETA_BAD_SUBJECTS}


def subject_label(sid):
    """Human-readable subject label."""
    if sid <= 35:
        return f"BM_S{sid:02d}"
    return f"BETA_S{sid - 100:02d}"


def train_one_fold(subject_id, args, device):
    """Train and test one LOSO fold. Returns summary dict or None on error."""
    fold_dir = Path(args.checkpoint_dir) / f"fold_{subject_id:03d}"
    summary_path = fold_dir / "summary.json"

    # Skip if already completed
    if summary_path.exists() and not args.force:
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  Fold {subject_id} ({subject_label(subject_id)}) already done: "
              f"acc={summary['val_acc']:.1%}")
        return summary

    fold_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    try:
        train_ds = LOSODataset(
            bm_dir=args.bm_dir, beta_dir=args.beta_dir,
            leave_out_subject=subject_id, is_train=True,
            trial_duration_pts=args.trial_pts,
            exclude_subjects=BETA_BAD_REMAPPED,
            random_offset=args.random_offset,
        )
        test_ds = LOSODataset(
            bm_dir=args.bm_dir, beta_dir=args.beta_dir,
            leave_out_subject=subject_id, is_train=False,
            trial_duration_pts=args.trial_pts,
            exclude_subjects=BETA_BAD_REMAPPED,
        )
    except ValueError as e:
        print(f"  SKIP fold {subject_id}: {e}")
        return None

    actual_pts = train_ds.trial_duration_pts

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )

    # Build model (fresh for each fold)
    model = build_film_classifier(
        reve_dir=args.reve_dir,
        trial_pts=actual_pts,
        use_film=args.use_film,
        unfreeze_last_n=args.unfreeze_last_n,
        film_scale=args.film_scale,
    )
    model = model.to(device)

    # Param groups
    reve_params, film_params, head_params, pool_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(param)
        elif name.startswith("film_") or name.startswith("fbcca."):
            film_params.append(param)
        elif "cls_query_token" in name or ".ln." in name:
            pool_params.append(param)
        elif name.startswith("reve."):
            reve_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if reve_params:
        param_groups.append({"params": reve_params, "lr": args.lr_reve})
    if film_params:
        param_groups.append({"params": film_params, "lr": args.lr_film})
    if pool_params:
        param_groups.append({"params": pool_params, "lr": args.lr_film})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})

    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP: bf16 on Ampere+, fp16 fallback
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_top5 = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        for batch in train_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast("cuda", dtype=amp_dtype):
                logits = model(eeg)
                loss = model.compute_loss(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Test on held-out subject
        model.eval()
        total_loss, correct, top5_correct, total = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                eeg = batch["eeg"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with autocast("cuda", dtype=amp_dtype):
                    logits = model(eeg)
                    loss = model.compute_loss(logits, labels)
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                top5 = logits.topk(5, dim=-1).indices
                top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        val_loss = total_loss / total
        val_acc = correct / total
        val_top5 = top5_correct / total

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_top5 = val_top5
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), fold_dir / "best_model.pt")
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"  E{epoch+1:02d} val_loss={val_loss:.4f} acc={val_acc:.1%} "
              f"top5={val_top5:.1%} pat={patience_counter}{marker}", flush=True)

        if patience_counter >= args.patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    # Reload best and get final predictions
    model.load_state_dict(
        torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast("cuda", dtype=amp_dtype):
                logits = model(eeg)
            logits = logits.float()  # back to fp32 for metrics
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    logits = torch.cat(all_logits)
    top5_preds = logits.topk(5, dim=-1).indices

    # Recompute metrics from final predictions
    final_acc = (preds == labels).float().mean().item()
    final_top5 = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Save predictions
    torch.save({
        "preds": preds,
        "labels": labels,
        "top5_preds": top5_preds,
        "logits": logits,
        "subject_id": subject_id,
    }, fold_dir / "predictions.pt")

    # Save summary
    dataset_name = "benchmark" if subject_id <= 35 else "beta"
    summary = {
        "subject_id": subject_id,
        "subject_label": subject_label(subject_id),
        "dataset": dataset_name,
        "val_acc": final_acc,
        "val_top5": final_top5,
        "val_loss": best_val_loss,
        "n_trials": len(labels),
        "epochs": best_epoch,
        "trial_pts": actual_pts,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def aggregate_results(checkpoint_dir, all_subjects):
    """Aggregate per-fold results into a summary."""
    results = []
    for sid in all_subjects:
        summary_path = Path(checkpoint_dir) / f"fold_{sid:03d}" / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results.append(json.load(f))

    if not results:
        print("No completed folds found.")
        return

    accs = [r["val_acc"] for r in results]
    top5s = [r["val_top5"] for r in results]
    acc_t = torch.tensor(accs)
    top5_t = torch.tensor(top5s)

    bm_results = [r for r in results if r["dataset"] == "benchmark"]
    beta_results = [r for r in results if r["dataset"] == "beta"]

    agg = {
        "n_subjects": len(results),
        "overall": {
            "mean_acc": acc_t.mean().item(),
            "std_acc": acc_t.std().item(),
            "median_acc": acc_t.median().item(),
            "min_acc": acc_t.min().item(),
            "max_acc": acc_t.max().item(),
            "mean_top5": top5_t.mean().item(),
        },
        "per_subject": sorted(results, key=lambda r: r["val_acc"]),
    }

    if bm_results:
        bm_accs = torch.tensor([r["val_acc"] for r in bm_results])
        agg["benchmark"] = {
            "n_subjects": len(bm_results),
            "mean_acc": bm_accs.mean().item(),
            "std_acc": bm_accs.std().item(),
            "median_acc": bm_accs.median().item(),
        }

    if beta_results:
        bt_accs = torch.tensor([r["val_acc"] for r in beta_results])
        agg["beta"] = {
            "n_subjects": len(beta_results),
            "mean_acc": bt_accs.mean().item(),
            "std_acc": bt_accs.std().item(),
            "median_acc": bt_accs.median().item(),
        }

    agg_path = Path(checkpoint_dir) / "aggregate_results.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 75}")
    print(f"LOSO Results: {len(results)} subjects")
    print(f"{'=' * 75}")
    print(f"{'Subject':<12} {'Dataset':<10} {'Acc':>7} {'Top5':>7} {'Trials':>7} {'Epochs':>7}")
    print(f"{'-' * 75}")
    for r in sorted(results, key=lambda r: r["subject_id"]):
        print(f"{r['subject_label']:<12} {r['dataset']:<10} "
              f"{r['val_acc']:>6.1%} {r['val_top5']:>6.1%} "
              f"{r['n_trials']:>7} {r['epochs']:>7}")
    print(f"{'-' * 75}")
    print(f"{'Overall':<12} {'':10} {acc_t.mean().item():>6.1%} {top5_t.mean().item():>6.1%}")
    if bm_results:
        bm_accs = torch.tensor([r["val_acc"] for r in bm_results])
        print(f"{'Benchmark':<12} {'':10} {bm_accs.mean().item():>6.1%}")
    if beta_results:
        bt_accs = torch.tensor([r["val_acc"] for r in beta_results])
        print(f"{'BETA':<12} {'':10} {bt_accs.mean().item():>6.1%}")
    print(f"{'=' * 75}")
    print(f"Saved: {agg_path}")


def main():
    parser = argparse.ArgumentParser(description="LOSO cross-validation for FiLM classifier")

    # Paths
    parser.add_argument("--bm_dir", type=str, default="data/eeg_tensors_benchmark")
    parser.add_argument("--beta_dir", type=str, default="data/eeg_tensors_beta")
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/zjj/loso_film")

    # Trial config
    parser.add_argument("--trial_pts", type=int, default=200, help="200=1s, 300=1.5s, 600=3s")
    parser.add_argument("--unfreeze_last_n", type=int, default=4)
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--no_film", dest="use_film", action="store_false")
    parser.add_argument("--film_scale", type=float, default=0.1)
    parser.add_argument("--random_offset", action="store_true", default=True)
    parser.add_argument("--no_random_offset", dest="random_offset", action="store_false")

    # Training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr_reve", type=float, default=1e-5)
    parser.add_argument("--lr_film", type=float, default=3e-4)
    parser.add_argument("--lr_head", type=float, default=3e-4)

    # Fold control
    parser.add_argument("--start_fold", type=int, default=None,
                        help="Start from this subject ID (for resume)")
    parser.add_argument("--end_fold", type=int, default=None,
                        help="End at this subject ID (inclusive)")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "benchmark", "beta"])
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-run even if summary.json exists")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_subjects = LOSODataset.get_all_subjects(
        dataset_filter=args.dataset,
        exclude_subjects=BETA_BAD_REMAPPED,
    )

    # Filter by start/end fold
    if args.start_fold is not None:
        all_subjects = [s for s in all_subjects if s >= args.start_fold]
    if args.end_fold is not None:
        all_subjects = [s for s in all_subjects if s <= args.end_fold]

    print(f"LOSO: {len(all_subjects)} folds, trial_pts={args.trial_pts}, "
          f"unfreeze={args.unfreeze_last_n}, film={args.use_film}")
    print(f"Output: {args.checkpoint_dir}")

    t_start = time.time()
    completed = 0

    for i, sid in enumerate(all_subjects):
        print(f"\n[{i+1}/{len(all_subjects)}] Fold {sid} ({subject_label(sid)})")
        t0 = time.time()

        summary = train_one_fold(sid, args, device)

        if summary is not None:
            dt = time.time() - t0
            print(f"  -> {subject_label(sid)}: acc={summary['val_acc']:.1%}, "
                  f"top5={summary['val_top5']:.1%}, {summary['n_trials']} trials, "
                  f"{summary['epochs']} epochs, {dt:.0f}s")
            completed += 1

    total_time = time.time() - t_start
    print(f"\nCompleted {completed}/{len(all_subjects)} folds in {total_time/60:.1f}min")

    # Aggregate all results (including previously completed folds)
    all_subjects_full = LOSODataset.get_all_subjects(
        dataset_filter=args.dataset,
        exclude_subjects=BETA_BAD_REMAPPED,
    )
    aggregate_results(args.checkpoint_dir, all_subjects_full)


if __name__ == "__main__":
    main()
