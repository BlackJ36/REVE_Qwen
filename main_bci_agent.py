"""Entry point for BCI Agent training (Stage 1 and Stage 2)."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BCI Agent: REVE+FBCCA FiLM + Qwen3-4B")

    # Stage selection
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage: 1 (alignment) or 2 (instruction tuning)")

    # Data
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output_bci_agent_s{stage})")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")
    parser.add_argument("--reve_dir", type=str, default="models")

    # Encoder backbone
    parser.add_argument("--encoder_type", type=str, default="reve", choices=["reve", "labram"],
                        help="EEG encoder backbone: 'reve' (512d) or 'labram' (200d)")
    parser.add_argument("--use_fbcca", action="store_true", default=True,
                        help="Enable FBCCA FiLM modulation (default: True)")
    parser.add_argument("--no_fbcca", dest="use_fbcca", action="store_false",
                        help="Disable FBCCA, use backbone-only")

    # Stage 2 specific
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 best checkpoint directory")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--nl_data_path", type=str, default=None,
                        help="Path to pure NL JSONL data for Stage 2 Type C")

    # Training
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device batch size (default: 64 for S1, 32 for S2)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation (default: 2 for S1, 4 for S2)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 5e-4 for S1, 2e-5 for S2)")
    parser.add_argument("--encoder_lr", type=float, default=None,
                        help="Encoder learning rate (default: 1e-3 for S1, 5e-4 for S2)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: 10 for S1, 5 for S2)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Data quality
    parser.add_argument("--exclude_bad_subjects", action="store_true", default=False,
                        help="Exclude BETA subjects with <30%% FBCCA accuracy (S11,S41,S55,S59,S64)")

    # Multi-spell
    parser.add_argument("--min_spells", type=int, default=None)
    parser.add_argument("--max_spells", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--window_step", type=int, default=100)
    parser.add_argument("--num_eeg_tokens", type=int, default=62)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero2.json")
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve exclude list
    if args.exclude_bad_subjects:
        from src.dataset_bci_agent import BETA_BAD_SUBJECTS
        exclude_subjects = BETA_BAD_SUBJECTS
    else:
        exclude_subjects = None

    # Set stage-specific defaults
    if args.output_dir is None:
        args.output_dir = f"output_bci_agent_s{args.stage}"

    if args.stage == 1:
        batch_size = args.batch_size or 64
        grad_accum = args.grad_accum or 2
        lr = args.lr or 5e-4
        encoder_lr = args.encoder_lr or 1e-3
        epochs = args.epochs or 10
        min_spells = args.min_spells or 5
        max_spells = args.max_spells or 10

        from src.train_bci_agent import run_stage1_training

        run_stage1_training(
            eeg_dir=args.eeg_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            from_modelscope=args.from_modelscope,
            reve_dir=args.reve_dir,
            encoder_type=args.encoder_type,
            use_fbcca=args.use_fbcca,
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            encoder_lr=encoder_lr,
            num_epochs=epochs,
            warmup_ratio=args.warmup_ratio,
            min_spells=min_spells,
            max_spells=max_spells,
            window_size=args.window_size,
            window_step=args.window_step,
            num_eeg_tokens=args.num_eeg_tokens,
            exclude_subjects=exclude_subjects,
            deepspeed_config=args.deepspeed,
        )

    elif args.stage == 2:
        batch_size = args.batch_size or 32
        grad_accum = args.grad_accum or 4
        lr = args.lr or 2e-5
        encoder_lr = args.encoder_lr or 5e-4
        epochs = args.epochs or 5
        min_spells = args.min_spells or 3
        max_spells = args.max_spells or 8

        from src.train_bci_agent import run_stage2_training

        run_stage2_training(
            eeg_dir=args.eeg_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            from_modelscope=args.from_modelscope,
            reve_dir=args.reve_dir,
            encoder_type=args.encoder_type,
            use_fbcca=args.use_fbcca,
            stage1_checkpoint=args.stage1_checkpoint,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            encoder_lr=encoder_lr,
            num_epochs=epochs,
            warmup_ratio=args.warmup_ratio,
            nl_data_path=args.nl_data_path,
            min_spells=min_spells,
            max_spells=max_spells,
            window_size=args.window_size,
            window_step=args.window_step,
            num_eeg_tokens=args.num_eeg_tokens,
            exclude_subjects=exclude_subjects,
            deepspeed_config=args.deepspeed,
        )


if __name__ == "__main__":
    main()
