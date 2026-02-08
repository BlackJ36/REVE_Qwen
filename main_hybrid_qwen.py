"""Entry point for Plan B: Hybrid Qwen3-0.6B full fine-tune training."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan B: HybridEncoder (REVE+FBCCA) + Qwen3-0.6B full fine-tune"
    )

    # Data
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--output_dir", type=str, default="output_hybrid_qwen")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")

    # REVE
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--num_eeg_tokens", type=int, default=62)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="Qwen learning rate")
    parser.add_argument("--projector_lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Multi-spell
    parser.add_argument("--min_spells", type=int, default=5)
    parser.add_argument("--max_spells", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--window_step", type=int, default=100)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def main():
    args = parse_args()

    from src.train_hybrid_qwen import run_hybrid_qwen_training

    run_hybrid_qwen_training(
        eeg_dir=args.eeg_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        from_modelscope=args.from_modelscope,
        reve_dir=args.reve_dir,
        num_eeg_tokens=args.num_eeg_tokens,
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
        deepspeed_config=args.deepspeed,
    )


if __name__ == "__main__":
    main()
