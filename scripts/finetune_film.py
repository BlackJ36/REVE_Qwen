"""Standalone FiLM verification: REVE(9ch) + FiLM(FBCCA) -> Linear(512->40).

Verifies whether FBCCA frequency prior via FiLM improves SSVEP classification,
especially at short windows. No Qwen/LLM involved -- pure EEG classification.

Usage:
  # Baseline (no FiLM, frozen REVE)
  python scripts/finetune_film.py --no_film --trial_pts 600

  # FiLM (frozen REVE)
  python scripts/finetune_film.py --trial_pts 600

  # FiLM + unfreeze last 2 REVE layers
  python scripts/finetune_film.py --trial_pts 600 --unfreeze_last_n 2

  # Sweep durations
  for pts in 200 300 400 600; do
    python scripts/finetune_film.py --trial_pts $pts --output_dir output_film/film_${pts}
    python scripts/finetune_film.py --trial_pts $pts --no_film --output_dir output_film/base_${pts}
  done
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_bci_agent import BETA_BAD_SUBJECTS
from src.dataset_reve_finetune import REVEFinetuneDataset, reve_finetune_collate_fn
from src.film_classifier import build_film_classifier


def run_validation(model, dataloader, device):
    """Returns (loss, top1_acc, top5_acc)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch["eeg"].to(device)
            labels = batch["labels"].to(device)
            etrca = batch.get("etrca_scores")
            if etrca is not None:
                etrca = etrca.to(device)

            logits = model(eeg)
            loss = model.compute_loss(logits, labels, etrca)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            top5 = logits.topk(5, dim=-1).indices
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total, top5_correct / total


def build_param_groups(model, args):
    """Create parameter groups with separate learning rates.

    Groups:
      - REVE unfrozen layers: lr_reve (small, 1e-5)
      - FiLM parameters: lr_film (medium, 3e-4)
      - Classifier head: lr_head (medium, 3e-4)
      - Attention pooling (cls_query_token + ln): lr_film
    """
    reve_params = []
    film_params = []
    head_params = []
    pooling_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(param)
        elif name.startswith("film_") or name.startswith("fbcca."):
            film_params.append(param)
        elif "cls_query_token" in name or ".ln." in name:
            pooling_params.append(param)
        elif name.startswith("reve."):
            reve_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if reve_params:
        groups.append({"params": reve_params, "lr": args.lr_reve, "name": "reve"})
    if film_params:
        groups.append({"params": film_params, "lr": args.lr_film, "name": "film"})
    if pooling_params:
        groups.append({"params": pooling_params, "lr": args.lr_film, "name": "pooling"})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head, "name": "head"})

    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {n_params:,} params, lr={g['lr']:.1e}")

    return groups


def train(model, train_loader, val_loader, device, args):
    """Main training loop with early stopping."""
    param_groups = build_param_groups(model, args)
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log training config
    config = vars(args).copy()
    config["actual_trial_pts"] = train_loader.dataset.trial_duration_pts
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}")
    mode = "FiLM" if args.use_film else "Baseline"
    print(f"Training: {mode}, trial_pts={train_loader.dataset.trial_duration_pts}, "
          f"unfreeze={args.unfreeze_last_n}")
    print(f"  epochs={args.epochs}, patience={args.patience}")
    print(f"{'=' * 70}\n")

    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            labels = batch["labels"].to(device)
            etrca = batch.get("etrca_scores")
            if etrca is not None:
                etrca = etrca.to(device)

            logits = model(eeg)
            loss = model.compute_loss(logits, labels, etrca)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += labels.size(0)

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc, val_top5 = run_validation(model, val_loader, device)
        dt = time.time() - t0

        # FiLM stats
        film_str = ""
        if model.use_film and model._last_gamma is not None:
            gamma_std = (model._last_gamma - 1).std().item()
            beta_std = model._last_beta.std().item()
            film_str = f" | g_std={gamma_std:.4f} b_std={beta_std:.4f}"

        lrs = [f"{g['name']}={scheduler.get_last_lr()[i]:.1e}"
               for i, g in enumerate(param_groups)]

        print(f"Epoch {epoch+1:3d}/{args.epochs} ({dt:.1f}s) | "
              f"train loss={train_loss:.4f} acc={train_acc:.1%} | "
              f"val loss={val_loss:.4f} acc={val_acc:.1%} top5={val_top5:.1%}"
              f"{film_str} | {', '.join(lrs)}")

        record = {
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_top5": val_top5,
        }
        history.append(record)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> New best (loss={val_loss:.4f}, acc={val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Reload best model
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True)
    )

    # Final validation
    val_loss, val_acc, val_top5 = run_validation(model, val_loader, device)
    print(f"\nFinal: val_loss={val_loss:.4f}, val_acc={val_acc:.1%}, val_top5={val_top5:.1%}")

    summary = {
        "mode": "film" if args.use_film else "baseline",
        "trial_pts": train_loader.dataset.trial_duration_pts,
        "unfreeze_last_n": args.unfreeze_last_n,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_top5": val_top5,
        "total_epochs": len(history),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="FiLM verification: REVE(9ch) + FBCCA")

    # Data
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="output_film")
    parser.add_argument("--trial_pts", type=int, default=600,
                        help="Trial duration in timepoints (200=1s, 300=1.5s, 600=3s)")

    # Model
    parser.add_argument("--use_film", action="store_true", default=True)
    parser.add_argument("--no_film", dest="use_film", action="store_false")
    parser.add_argument("--unfreeze_last_n", type=int, default=0,
                        help="Unfreeze last N REVE transformer layers (0=frozen)")
    parser.add_argument("--film_scale", type=float, default=0.1,
                        help="FiLM amplitude constraint (0.1 = +/-10%%)")
    parser.add_argument("--film_reg_weight", type=float, default=0.01,
                        help="FiLM regularization weight")

    # Training
    parser.add_argument("--lr_reve", type=float, default=1e-5,
                        help="LR for unfrozen REVE layers")
    parser.add_argument("--lr_film", type=float, default=3e-4,
                        help="LR for FiLM parameters")
    parser.add_argument("--lr_head", type=float, default=3e-4,
                        help="LR for classifier head")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=7)

    # Distillation
    parser.add_argument("--distill_alpha", type=float, default=1.0,
                        help="CE weight (1.0 = no distillation, 0.5 = 50%% CE + 50%% KL)")
    parser.add_argument("--distill_temp", type=float, default=2.0)

    # Augmentation
    parser.add_argument("--random_offset", action="store_true", default=False,
                        help="Random start offset within trial (train only)")

    # Data quality
    parser.add_argument("--exclude_bad_subjects", action="store_true", default=True)
    parser.add_argument("--no_exclude", dest="exclude_bad_subjects", action="store_false")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exclude = BETA_BAD_SUBJECTS if args.exclude_bad_subjects else None
    use_etrca = args.distill_alpha < 1.0

    # Load datasets
    print("Loading datasets...")
    train_ds = REVEFinetuneDataset(
        args.eeg_dir, split="train",
        trial_duration_pts=args.trial_pts,
        exclude_subjects=exclude,
        use_etrca=use_etrca,
        random_offset=args.random_offset,
    )
    val_ds = REVEFinetuneDataset(
        args.eeg_dir, split="val",
        trial_duration_pts=args.trial_pts,
        exclude_subjects=exclude,
        use_etrca=use_etrca,
    )

    actual_pts = train_ds.trial_duration_pts
    if actual_pts != args.trial_pts:
        print(f"  NOTE: requested {args.trial_pts}pts but capped to {actual_pts}pts "
              f"(shortest valid trial after latency skip)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )

    # Build model (use actual_pts for FBCCA template size)
    print("\nBuilding model...")
    model = build_film_classifier(
        reve_dir=args.reve_dir,
        trial_pts=actual_pts,
        use_film=args.use_film,
        unfreeze_last_n=args.unfreeze_last_n,
        film_scale=args.film_scale,
        film_reg_weight=args.film_reg_weight,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
    )
    model = model.to(device)

    best_acc = train(model, train_loader, val_loader, device, args)

    print(f"\n{'=' * 70}")
    mode = "FiLM" if args.use_film else "Baseline"
    print(f"Done: {mode} | trial_pts={actual_pts} | "
          f"unfreeze={args.unfreeze_last_n} | best_val_acc={best_acc:.1%}")
    print(f"Output: {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
