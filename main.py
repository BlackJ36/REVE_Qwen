"""Entry point for BCI-Qwen training (pre-extracted embeddings).

Stage 1: Single-trial EEG classification (projector + new token embeddings)
Stage 2: Multi-char spelling + NL dialogues (LoRA + projector)
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BCI-Qwen Training")

    # Data
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--s2_train", type=str, default="data/s2_train.jsonl",
                        help="S2 train dialogue JSONL (stage 2 only)")
    parser.add_argument("--s2_val", type=str, default="data/s2_val.jsonl",
                        help="S2 val dialogue JSONL (stage 2 only)")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")

    # Stage
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to S1 checkpoint dir (stage 2 only)")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--projector_lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    from src.train import run_training

    run_training(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        from_modelscope=args.from_modelscope,
        stage=args.stage,
        stage1_checkpoint=args.stage1_checkpoint,
        s2_train_path=args.s2_train if args.stage == 2 else None,
        s2_val_path=args.s2_val if args.stage == 2 else None,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        projector_lr=args.projector_lr,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        warmup_ratio=args.warmup_ratio,
        deepspeed_config=args.deepspeed,
    )


if __name__ == "__main__":
    main()
