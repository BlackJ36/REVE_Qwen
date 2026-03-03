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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")
    parser.add_argument("--reve_dir", type=str, default="models")

    # Encoder backbone
    parser.add_argument("--encoder_type", type=str, default="reve", choices=["reve", "labram"],
                        help="EEG encoder backbone: 'reve' (512d) or 'labram' (200d)")
    parser.add_argument("--unfreeze_last_n", type=int, default=0,
                        help="Unfreeze last N transformer layers of REVE/LaBraM backbone (default: 0 = frozen)")
    parser.add_argument("--reve_finetune_dir", type=str, default=None,
                        help="Directory with fine-tuned REVE LoRA + pooling (from finetune_reve.py). "
                             "Merges LoRA into base weights at load time (zero runtime overhead).")
    parser.add_argument("--fbcca_mode", type=str, default=None,
                        choices=["film", "candidate", "none"],
                        help="FBCCA integration: 'film' (FiLM modulation), "
                             "'candidate' (inject as tokens), 'none' (backbone-only)")
    parser.add_argument("--decoder_type", type=str, default="fbcca",
                        choices=["fbcca", "trca", "etrca"],
                        help="Decoder type for candidate injection: 'fbcca', 'trca', or 'etrca' "
                             "(only used when --fbcca_mode=candidate)")
    parser.add_argument("--cand_dropout", type=float, default=0.0,
                        help="Candidate dropout rate (0.0-1.0). Randomly replaces decoder "
                             "candidates with noise during training to prevent shortcut learning. "
                             "Only used when --fbcca_mode=candidate. Recommended: 0.3")
    parser.add_argument("--echo_dropout", type=float, default=0.0,
                        help="Echo character dropout rate (0.0-1.0). Skips character echo "
                             "after target to prevent LM prior from overriding EEG evidence. "
                             "Only used when --fbcca_mode=candidate. Recommended: 0.3-0.5")
    parser.add_argument("--eeg_loss_weight", type=float, default=1.0,
                        help="Loss weight multiplier for EEG-only prediction positions. "
                             "Values >1.0 upweight pure EEG decoding loss. Recommended: 2.0")
    # Legacy flags (backward compat)
    parser.add_argument("--use_fbcca", action="store_true", default=True,
                        help="Enable FBCCA FiLM modulation (default: True). "
                             "Prefer --fbcca_mode instead.")
    parser.add_argument("--no_fbcca", dest="use_fbcca", action="store_false",
                        help="Disable FBCCA. Prefer --fbcca_mode none instead.")

    # S1 LoRA (enables Qwen attention to adapt to EEG tokens in Stage 1)
    parser.add_argument("--s1_lora_rank", type=int, default=0,
                        help="LoRA rank for Stage 1 (0 = no LoRA, >0 = apply LoRA). "
                             "Recommended: 16 for candidate mode.")
    parser.add_argument("--s1_lora_alpha", type=int, default=32,
                        help="LoRA alpha for Stage 1 (default: 32)")

    # Stage 2 specific
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 best checkpoint directory")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--nl_data_path", type=str, default=None,
                        help="Path to pure NL JSONL data for Stage 2 Type C")
    parser.add_argument("--word_vocab", type=str, default=None,
                        help="Path to word vocab JSON for Stage 2 candidate mode "
                             "(default: built-in 198 common + 48 BCI words)")

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
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping patience (eval rounds without improvement)")

    # Data quality
    parser.add_argument("--exclude_bad_subjects", action="store_true", default=False,
                        help="Exclude BETA subjects with <30%% FBCCA accuracy (S11,S41,S55,S59,S64)")

    # Multi-spell
    parser.add_argument("--min_spells", type=int, default=None)
    parser.add_argument("--max_spells", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--window_step", type=int, default=100)
    parser.add_argument("--num_eeg_tokens", type=int, default=62)

    # Variable trial duration
    parser.add_argument("--trial_duration", type=float, default=3.0,
                        help="Trial duration in seconds (1.0/1.5/2.0/3.0). "
                             "Shorter = faster spelling but lower accuracy.")

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

    # Resolve fbcca_mode: explicit --fbcca_mode takes priority over --use_fbcca/--no_fbcca
    fbcca_mode = args.fbcca_mode
    if fbcca_mode is None:
        fbcca_mode = "film" if args.use_fbcca else "none"

    # Convert trial duration (seconds) to timepoints (@ 200Hz)
    trial_duration_pts = int(args.trial_duration * 200)

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
            fbcca_mode=fbcca_mode,
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
            trial_duration_pts=trial_duration_pts,
            exclude_subjects=exclude_subjects,
            decoder_type=args.decoder_type,
            cand_dropout=args.cand_dropout,
            echo_dropout=args.echo_dropout,
            eeg_loss_weight=args.eeg_loss_weight,
            early_stopping_patience=args.early_stopping_patience,
            unfreeze_last_n=args.unfreeze_last_n,
            lora_rank=args.s1_lora_rank,
            lora_alpha=args.s1_lora_alpha,
            reve_finetune_dir=args.reve_finetune_dir,
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
            fbcca_mode=fbcca_mode,
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
            word_vocab_path=args.word_vocab,
            min_spells=min_spells,
            max_spells=max_spells,
            window_size=args.window_size,
            window_step=args.window_step,
            num_eeg_tokens=args.num_eeg_tokens,
            trial_duration_pts=trial_duration_pts,
            exclude_subjects=exclude_subjects,
            decoder_type=args.decoder_type,
            cand_dropout=args.cand_dropout,
            echo_dropout=args.echo_dropout,
            eeg_loss_weight=args.eeg_loss_weight,
            early_stopping_patience=args.early_stopping_patience,
            unfreeze_last_n=args.unfreeze_last_n,
            reve_finetune_dir=args.reve_finetune_dir,
            deepspeed_config=args.deepspeed,
        )


if __name__ == "__main__":
    main()
