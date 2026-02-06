"""Entry point for BCI-Qwen training."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BCI-Qwen: REVE + Qwen3-VL LoRA Training")

    # Data
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings")
    parser.add_argument("--output_dir", type=str, default="output")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Quantization (for single GPU with limited VRAM)
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (NF4). Reduces VRAM from ~16GB to ~6GB.")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2.json")

    return parser.parse_args()


def main():
    args = parse_args()

    from src.train import run_training

    run_training(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        from_modelscope=args.from_modelscope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        deepspeed_config=args.deepspeed if not args.use_4bit else None,
    )


if __name__ == "__main__":
    main()
