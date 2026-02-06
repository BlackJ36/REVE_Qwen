"""Entry point for BCI-Qwen training with Unsloth."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BCI-Qwen with Unsloth")

    # Data
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings")
    parser.add_argument("--output_dir", type=str, default="output")

    # Model
    parser.add_argument(
        "--model_name", type=str, default="unsloth/Qwen3-VL-4B-Instruct",
        help="Unsloth model name. Options: unsloth/Qwen3-VL-2B-Instruct, unsloth/Qwen3-VL-4B-Instruct"
    )

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()

    from src.train_unsloth import run_training_unsloth

    run_training_unsloth(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
