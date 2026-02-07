"""Entry point for end-to-end BCI-Qwen training (REVE unfrozen)."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BCI-Qwen E2E: REVE fine-tuning + Qwen LoRA")

    # Data
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--output_dir", type=str, default="output_e2e")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")

    # REVE
    parser.add_argument("--reve_dir", type=str, default="models",
                        help="Local directory containing reve-base/ and reve-positions/")
    parser.add_argument("--unfreeze_last_n", type=int, default=4,
                        help="Number of REVE transformer layers to unfreeze (from the end)")
    parser.add_argument("--num_eeg_tokens", type=int, default=62,
                        help="EEG tokens per window (62 channels × 1 patch for 1.5s windows)")
    parser.add_argument("--reve_lr", type=float, default=3e-5,
                        help="Learning rate for REVE unfrozen layers")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Quantization
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization for Qwen (single GPU)")

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4, help="LoRA learning rate")
    parser.add_argument("--projector_lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Multi-spell
    parser.add_argument("--min_spells", type=int, default=5,
                        help="Min spells per training sequence")
    parser.add_argument("--max_spells", type=int, default=10,
                        help="Max spells per training sequence")
    parser.add_argument("--window_size", type=int, default=300,
                        help="Sliding window size in timepoints (300 = 1.5s @ 200Hz)")
    parser.add_argument("--window_step", type=int, default=100,
                        help="Sliding window step in timepoints (100 = 0.5s @ 200Hz)")

    # Checkpoint
    parser.add_argument("--checkpoint_mode", type=str, default="weights_only",
                        choices=["weights_only", "full"],
                        help="weights_only (~100-200MB) for inference, full (~20-30GB) for resume")

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank passed by distributed launcher")

    return parser.parse_args()


def main():
    args = parse_args()

    from src.train_e2e import run_e2e_training

    run_e2e_training(
        eeg_dir=args.eeg_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        from_modelscope=args.from_modelscope,
        reve_dir=args.reve_dir,
        unfreeze_last_n=args.unfreeze_last_n,
        num_eeg_tokens=args.num_eeg_tokens,
        reve_lr=args.reve_lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        projector_lr=args.projector_lr,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        warmup_ratio=args.warmup_ratio,
        checkpoint_mode=args.checkpoint_mode,
        deepspeed_config=args.deepspeed if not args.use_4bit else None,
        min_spells=args.min_spells,
        max_spells=args.max_spells,
        window_size=args.window_size,
        window_step=args.window_step,
    )


if __name__ == "__main__":
    main()
