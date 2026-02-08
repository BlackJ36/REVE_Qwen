"""Entry point for Plan A: Hybrid Transformer decoder training."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan A: HybridEncoder (REVE+FBCCA) + custom Transformer decoder"
    )

    # Data
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--output_dir", type=str, default="output_hybrid")

    # REVE
    parser.add_argument("--reve_dir", type=str, default="models")

    # Decoder architecture
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4, help="Decoder learning rate")
    parser.add_argument("--projector_lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Multi-spell
    parser.add_argument("--min_spells", type=int, default=5)
    parser.add_argument("--max_spells", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--window_step", type=int, default=100)
    parser.add_argument("--num_eeg_tokens", type=int, default=62)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def main():
    args = parse_args()

    from src.train_hybrid import run_hybrid_training

    run_hybrid_training(
        eeg_dir=args.eeg_dir,
        output_dir=args.output_dir,
        reve_dir=args.reve_dir,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        projector_lr=args.projector_lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        min_spells=args.min_spells,
        max_spells=args.max_spells,
        window_size=args.window_size,
        window_step=args.window_step,
        num_eeg_tokens=args.num_eeg_tokens,
        deepspeed_config=args.deepspeed,
    )


if __name__ == "__main__":
    main()
