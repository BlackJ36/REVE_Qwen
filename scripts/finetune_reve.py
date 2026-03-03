"""Standalone REVE fine-tuning for SSVEP classification with eTRCA distillation.

Two-phase training (REVE paper recommended procedure):
  Phase A ("head"): REVE frozen, train linear head + attention pooling (~21K params)
  Phase B ("lora"): REVE LoRA on attention QKV/output + head (~561K params)

Usage:
  # Phase A: train classification head (~2 minutes)
  python scripts/finetune_reve.py --phase head

  # Phase B: LoRA fine-tuning on Phase A checkpoint (~5-10 minutes)
  python scripts/finetune_reve.py --phase lora --head_checkpoint output_reve_finetune

  # Both phases sequentially
  python scripts/finetune_reve.py --phase both

  # Without distillation
  python scripts/finetune_reve.py --phase both --distill_alpha 1.0
"""

import argparse
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
from src.reve_classifier import build_reve_classifier


def run_evaluation(model, dataloader, device):
    """Run model on validation set.

    Returns:
        val_loss: average loss
        val_acc: top-1 accuracy
        val_top5: top-5 accuracy
    """
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


def train_phase(model, train_loader, val_loader, device, args, phase_name):
    """Run one training phase (A or B).

    Returns:
        best_val_acc: best validation accuracy achieved
    """
    lr = args.lr_head if phase_name == "head" else args.lr_lora
    epochs = args.epochs_head if phase_name == "head" else args.epochs_lora

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    output_dir = Path(args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"Phase {'A' if phase_name == 'head' else 'B'}: {phase_name} training")
    print(f"  LR={lr}, epochs={epochs}, patience={args.patience}")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += labels.size(0)

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc, val_top5 = run_evaluation(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{epochs} ({dt:.1f}s) | "
              f"train loss={train_loss:.4f} acc={train_acc:.1%} | "
              f"val loss={val_loss:.4f} acc={val_acc:.1%} top5={val_top5:.1%} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            _save_checkpoint(model, output_dir, phase_name)
            print(f"  -> New best (loss={val_loss:.4f}, acc={val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
                break

    # Reload best checkpoint
    _load_checkpoint(model, output_dir, phase_name, device)
    print(f"\nPhase {'A' if phase_name == 'head' else 'B'} complete: "
          f"best val_acc={best_val_acc:.1%}")
    return best_val_acc


def _save_checkpoint(model, output_dir, phase_name):
    """Save phase-specific checkpoint files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save head weights
    torch.save(model.head.state_dict(), output_dir / "head.pt")

    # Save REVE pooling weights (cls_query_token + ln)
    reve = model.reve.reve
    # Handle PEFT-wrapped models
    if hasattr(reve, "base_model"):
        base_reve = reve.base_model.model
    else:
        base_reve = reve
    pooling_state = {}
    if hasattr(base_reve, "cls_query_token"):
        pooling_state["cls_query_token"] = base_reve.cls_query_token.data
    if hasattr(base_reve, "ln"):
        for name, param in base_reve.ln.named_parameters():
            pooling_state[f"ln.{name}"] = param.data
    torch.save(pooling_state, output_dir / "reve_pooling.pt")

    # Phase B: save PEFT LoRA adapter
    if phase_name == "lora" and hasattr(reve, "save_pretrained"):
        lora_dir = output_dir / "reve_lora"
        reve.save_pretrained(str(lora_dir))
        print(f"  Saved LoRA adapter to {lora_dir}")


def _load_checkpoint(model, output_dir, phase_name, device):
    """Load phase-specific checkpoint files."""
    # Load head
    head_path = output_dir / "head.pt"
    if head_path.exists():
        model.head.load_state_dict(
            torch.load(head_path, map_location=device, weights_only=True)
        )

    # Load pooling
    pooling_path = output_dir / "reve_pooling.pt"
    if pooling_path.exists():
        pooling_state = torch.load(pooling_path, map_location=device, weights_only=True)
        reve = model.reve.reve
        if hasattr(reve, "base_model"):
            base_reve = reve.base_model.model
        else:
            base_reve = reve
        for name, param in pooling_state.items():
            parts = name.split(".")
            obj = base_reve
            for part in parts:
                obj = getattr(obj, part)
            obj.data.copy_(param)

    # Phase B: load LoRA
    if phase_name == "lora":
        lora_dir = output_dir / "reve_lora"
        if lora_dir.exists() and hasattr(model.reve.reve, "load_adapter"):
            model.reve.reve.load_adapter(str(lora_dir), "default")


def main():
    parser = argparse.ArgumentParser(description="REVE two-phase fine-tuning for SSVEP")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["head", "lora", "both"],
                        help="Phase: 'head' (A), 'lora' (B), or 'both'")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="output_reve_finetune")
    parser.add_argument("--head_checkpoint", type=str, default=None,
                        help="Phase A checkpoint directory (for Phase B)")

    # Hyperparameters
    parser.add_argument("--lr_head", type=float, default=1e-3,
                        help="Learning rate for Phase A (default: 1e-3)")
    parser.add_argument("--lr_lora", type=float, default=2e-4,
                        help="Learning rate for Phase B (default: 2e-4)")
    parser.add_argument("--epochs_head", type=int, default=20,
                        help="Max epochs for Phase A (default: 20)")
    parser.add_argument("--epochs_lora", type=int, default=30,
                        help="Max epochs for Phase B (default: 30)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                        help="CE weight in distillation loss (1-alpha for KL)")
    parser.add_argument("--distill_temp", type=float, default=2.0,
                        help="Temperature for knowledge distillation")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--trial_duration", type=float, default=3.0)

    # Data quality
    parser.add_argument("--exclude_bad_subjects", action="store_true", default=True)
    parser.add_argument("--no_exclude", dest="exclude_bad_subjects", action="store_false")

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trial_duration_pts = int(args.trial_duration * 200)
    exclude = BETA_BAD_SUBJECTS if args.exclude_bad_subjects else None

    # Load datasets
    print("Loading datasets...")
    use_etrca = args.distill_alpha < 1.0  # skip eTRCA if pure CE
    train_ds = REVEFinetuneDataset(
        args.eeg_dir, split="train",
        trial_duration_pts=trial_duration_pts,
        exclude_subjects=exclude,
        use_etrca=use_etrca,
    )
    val_ds = REVEFinetuneDataset(
        args.eeg_dir, split="val",
        trial_duration_pts=trial_duration_pts,
        exclude_subjects=exclude,
        use_etrca=use_etrca,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )

    phases = []
    if args.phase in ("head", "both"):
        phases.append("head")
    if args.phase in ("lora", "both"):
        phases.append("lora")

    for phase in phases:
        head_ckpt = args.head_checkpoint or args.output_dir if phase == "lora" else None

        print(f"\nBuilding model (phase={phase})...")
        classifier = build_reve_classifier(
            phase=phase,
            reve_dir=args.reve_dir,
            distill_alpha=args.distill_alpha,
            distill_temp=args.distill_temp,
            lora_rank=args.lora_rank,
            head_checkpoint=head_ckpt,
        )
        classifier = classifier.to(device)

        best_acc = train_phase(classifier, train_loader, val_loader, device, args, phase)
        print(f"\n{'=' * 60}")
        print(f"Phase {'A' if phase == 'head' else 'B'} finished: best val_acc = {best_acc:.1%}")
        print(f"Checkpoint: {args.output_dir}/")
        print(f"{'=' * 60}")

        # Free GPU memory between phases
        del classifier
        torch.cuda.empty_cache()

    print("\nDone. Output files:")
    out = Path(args.output_dir)
    for f in sorted(out.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.relative_to(out)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
