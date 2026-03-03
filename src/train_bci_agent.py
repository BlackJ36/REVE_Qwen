"""Two-stage training for BCI agent.

Stage 1 (alignment): Qwen frozen, train FiLM encoder + projector + embeddings
Stage 2 (instruction tuning): Qwen LoRA, train encoder + LoRA, mixed data
"""

import json
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .dataset_bci_agent import (
    BCIAgentCollator,
    BCIAgentStage1Dataset,
    BCIAgentStage2Dataset,
)
from .dataset_bci_candidate import CandidateStage1Dataset, CandidateStage2Dataset
from .metrics_bci_agent import build_metrics_fn
from .model_bci_agent import build_bci_agent_model
from .tokens import get_target_token_ids


class BCIAgentTrainer(Trainer):
    """Trainer with separate LR groups for encoder and LLM components.

    LR groups vary by stage:
      Stage 1: encoder (FiLM + projector), embeddings
      Stage 2: encoder (FiLM + projector), LoRA, embeddings
    """

    def __init__(self, encoder_lr=1e-3, best_model_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_lr = encoder_lr
        self.best_model_dir = best_model_dir
        self.best_eval_loss = float("inf")

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        encoder_params = []
        embed_params = []
        lora_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder." in name:
                encoder_params.append(param)
            elif "embed_tokens" in name or "lm_head" in name:
                embed_params.append(param)
            else:
                lora_params.append(param)

        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.encoder_lr, "name": "encoder"})
        if embed_params:
            groups.append({"params": embed_params, "lr": self.args.learning_rate, "name": "embed"})
        if lora_params:
            groups.append({"params": lora_params, "lr": self.args.learning_rate, "name": "lora"})

        from torch.optim import AdamW
        self.optimizer = AdamW(
            groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )

        print(f"BCIAgent Optimizer ({len(groups)} groups):")
        print(f"  Encoder (FiLM+proj): {len(encoder_params)} params, lr={self.encoder_lr}")
        print(f"  Embeddings:          {len(embed_params)} params, lr={self.args.learning_rate}")
        if lora_params:
            print(f"  LoRA:                {len(lora_params)} params, lr={self.args.learning_rate}")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        eeg_windows = inputs.pop("eeg_windows", None)
        window_counts = inputs.pop("window_counts", None)
        loss_weights = inputs.pop("loss_weights", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            eeg_windows=eeg_windows if eeg_windows is not None and eeg_windows.numel() > 0 else None,
            window_counts=window_counts,
            loss_weights=loss_weights,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial):
        warmup_steps = self.args.get_warmup_steps(self.state.max_steps)
        if self.state.global_step <= warmup_steps:
            return

        if self.best_model_dir and self.state.log_history:
            for entry in reversed(self.state.log_history):
                if "eval_loss" in entry:
                    current_loss = entry["eval_loss"]
                    if current_loss < self.best_eval_loss:
                        self.best_eval_loss = current_loss
                        print(f"  New best: {current_loss:.4f}, saving to {self.best_model_dir}")
                        self._save_weights(self.best_model_dir)
                    break

    def log(self, logs, *args, **kwargs):
        """Append per-group learning rates to TensorBoard logs."""
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                name = group.get("name", "unknown")
                logs[f"lr_{name}"] = group["lr"]
        super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self._save_weights(output_dir)

    def _save_weights(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))


def run_stage1_training(
    eeg_dir="data/eeg_tensors",
    output_dir="output_bci_agent_s1",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    from_modelscope=True,
    reve_dir="models",
    encoder_type="reve",
    use_fbcca=True,
    fbcca_mode=None,
    # Training
    per_device_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    encoder_lr=1e-3,
    num_epochs=15,
    warmup_ratio=0.1,
    # Multi-spell
    min_spells=5,
    max_spells=10,
    window_size=300,
    window_step=100,
    num_eeg_tokens=62,
    # Variable duration
    trial_duration_pts=600,
    # Data quality
    exclude_subjects=None,
    # Decoder type for candidate mode
    decoder_type="fbcca",
    # Candidate dropout (0.0 = off, 0.3 = 30% random candidates)
    cand_dropout=0.0,
    # Echo dropout + EEG loss weighting (anti-LM-prior)
    echo_dropout=0.0,
    eeg_loss_weight=1.0,
    # Early stopping
    early_stopping_patience=5,
    # REVE unfreezing
    unfreeze_last_n=0,
    # S1 LoRA (0 = no LoRA, >0 = apply LoRA so Qwen attention adapts to EEG tokens)
    lora_rank=0,
    lora_alpha=32,
    # Fine-tuned REVE
    reve_finetune_dir=None,
    # DeepSpeed
    deepspeed_config="configs/ds_zero2.json",
):
    """Stage 1: Alignment training.

    Trains FiLM encoder + projector + embed_tokens/lm_head.
    When lora_rank > 0, also trains LoRA adapters on Qwen attention layers.
    """
    print("=" * 60)
    print("Stage 1: Alignment Training")
    print("=" * 60)

    # Resolve fbcca_mode
    if fbcca_mode is None:
        fbcca_mode = "film" if use_fbcca else "none"

    # Clamp window_size to trial duration
    effective_window_size = min(window_size, trial_duration_pts)
    if effective_window_size != window_size:
        print(f"  window_size clamped: {window_size} -> {effective_window_size} (trial={trial_duration_pts}pts)")

    lora_str = f", lora_rank={lora_rank}" if lora_rank > 0 else ""
    ft_str = f", reve_ft={reve_finetune_dir}" if reve_finetune_dir else ""
    print(f"\nBuilding model (Stage 1, encoder={encoder_type}, fbcca_mode={fbcca_mode}, unfreeze={unfreeze_last_n}{lora_str}{ft_str})...")
    model, tokenizer = build_bci_agent_model(
        model_name=model_name,
        from_modelscope=from_modelscope,
        reve_dir=reve_dir,
        stage=1,
        encoder_type=encoder_type,
        fbcca_mode=fbcca_mode,
        window_size=effective_window_size,
        unfreeze_last_n=unfreeze_last_n,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        reve_finetune_dir=reve_finetune_dir,
    )

    print("\nLoading datasets...")
    if fbcca_mode == "candidate":
        DatasetClass = CandidateStage1Dataset
        extra_kwargs = {"decoder_type": decoder_type, "cand_dropout": cand_dropout,
                        "echo_dropout": echo_dropout, "eeg_loss_weight": eeg_loss_weight,
                        "eeg_only_labels": True}
    else:
        DatasetClass = BCIAgentStage1Dataset
        extra_kwargs = {}
    train_dataset = DatasetClass(
        eeg_dir, tokenizer, split="train",
        num_eeg_tokens=num_eeg_tokens,
        min_spells=min_spells, max_spells=max_spells,
        window_size=effective_window_size, window_step=window_step,
        exclude_subjects=exclude_subjects,
        trial_duration_pts=trial_duration_pts,
        **extra_kwargs,
    )
    val_dataset = DatasetClass(
        eeg_dir, tokenizer, split="val",
        num_eeg_tokens=num_eeg_tokens,
        min_spells=min_spells, max_spells=max_spells,
        window_size=effective_window_size, window_step=window_step,
        exclude_subjects=exclude_subjects,
        trial_duration_pts=trial_duration_pts,
        **{k: v for k, v in extra_kwargs.items() if k not in ("cand_dropout", "echo_dropout", "eeg_loss_weight")},
    )
    collator = BCIAgentCollator(tokenizer)

    # Eval every ~3 epochs worth of steps (avoids excessive eval overhead)
    import math
    n_gpus = max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(train_dataset) / (per_device_batch_size * gradient_accumulation_steps * n_gpus))
    eval_steps = max(1, steps_per_epoch * 3)
    print(f"Eval/save every {eval_steps} steps (~3 epochs, {steps_per_epoch} steps/epoch)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
    )

    best_dir = Path(output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Build evaluation metrics
    # Stage 1 candidate: eeg_only_labels → 1 target/spell, no two_step split needed
    target_ids = get_target_token_ids(tokenizer)
    compute_metrics, preprocess_logits = build_metrics_fn(target_ids, two_step=False)

    trainer = BCIAgentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        encoder_lr=encoder_lr,
        best_model_dir=str(best_dir),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Print parameter summary
    enc_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "encoder." in n)
    # embed_tokens has requires_grad=True on the full tensor, but gradient hook
    # masks original rows — only new BCI token rows actually update
    from .tokens import ALL_SPECIAL_TOKENS
    llm_dim = model.qwen.config.hidden_size
    emb_effective = len(ALL_SPECIAL_TOKENS) * llm_dim
    print(f"\nTrainable parameters (Stage 1):")
    print(f"  Encoder (FiLM+proj): {enc_p:>12,}  (lr={encoder_lr})")
    print(f"  New token embeddings:{emb_effective:>12,}  (lr={learning_rate}, {len(ALL_SPECIAL_TOKENS)} tokens × {llm_dim}d)")
    print(f"  Total effective:     {enc_p + emb_effective:>12,}\n")

    print("Starting Stage 1 training...")
    trainer.train()

    # Save final
    final_dir = Path(output_dir) / "final"
    trainer._save_weights(str(final_dir))
    print(f"\nStage 1 complete. Model saved to {final_dir}")

    return trainer


def run_stage2_training(
    eeg_dir="data/eeg_tensors",
    output_dir="output_bci_agent_s2",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    from_modelscope=True,
    reve_dir="models",
    encoder_type="reve",
    use_fbcca=True,
    fbcca_mode=None,
    # Stage 1 checkpoint
    stage1_checkpoint=None,
    # LoRA
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05,
    # Training
    per_device_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    encoder_lr=5e-4,
    num_epochs=10,
    warmup_ratio=0.1,
    # Data
    nl_data_path=None,
    type_weights=None,
    word_vocab_path=None,
    # Multi-spell
    min_spells=3,
    max_spells=8,
    window_size=300,
    window_step=100,
    num_eeg_tokens=62,
    # Variable duration
    trial_duration_pts=600,
    # Data quality
    exclude_subjects=None,
    # Decoder type for candidate mode
    decoder_type="fbcca",
    # Candidate dropout (0.0 = off, 0.3 = 30% random candidates)
    cand_dropout=0.0,
    # Echo dropout + EEG loss weighting (anti-LM-prior)
    echo_dropout=0.0,
    eeg_loss_weight=1.0,
    # Early stopping
    early_stopping_patience=5,
    # REVE unfreezing
    unfreeze_last_n=0,
    # Fine-tuned REVE
    reve_finetune_dir=None,
    # DeepSpeed
    deepspeed_config="configs/ds_zero2.json",
):
    """Stage 2: Instruction tuning with LoRA.

    Loads encoder from Stage 1, applies LoRA to Qwen, trains with mixed data.
    """
    print("=" * 60)
    print("Stage 2: Instruction Tuning")
    print("=" * 60)

    # Resolve fbcca_mode
    if fbcca_mode is None:
        fbcca_mode = "film" if use_fbcca else "none"

    if stage1_checkpoint is None:
        print("WARNING: No Stage 1 checkpoint specified. Training encoder from scratch.")

    # Clamp window_size to trial duration
    effective_window_size = min(window_size, trial_duration_pts)
    if effective_window_size != window_size:
        print(f"  window_size clamped: {window_size} -> {effective_window_size} (trial={trial_duration_pts}pts)")

    ft_str = f", reve_ft={reve_finetune_dir}" if reve_finetune_dir else ""
    print(f"\nBuilding model (Stage 2, encoder={encoder_type}, fbcca_mode={fbcca_mode}, unfreeze={unfreeze_last_n}{ft_str})...")
    model, tokenizer = build_bci_agent_model(
        model_name=model_name,
        from_modelscope=from_modelscope,
        reve_dir=reve_dir,
        stage=2,
        encoder_type=encoder_type,
        fbcca_mode=fbcca_mode,
        stage1_checkpoint=stage1_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        window_size=effective_window_size,
        unfreeze_last_n=unfreeze_last_n,
        reve_finetune_dir=reve_finetune_dir,
    )

    # Build word vocab for candidate mode
    word_vocab = None
    if fbcca_mode == "candidate":
        from .word_vocab import WordVocab
        word_vocab = WordVocab(word_vocab_path)
        DatasetClass = CandidateStage2Dataset
        extra_kwargs = {"word_vocab": word_vocab, "decoder_type": decoder_type,
                        "cand_dropout": cand_dropout,
                        "echo_dropout": echo_dropout, "eeg_loss_weight": eeg_loss_weight}
    else:
        DatasetClass = BCIAgentStage2Dataset
        extra_kwargs = {}

    print("\nLoading datasets...")
    train_dataset = DatasetClass(
        eeg_dir, tokenizer, split="train",
        nl_data_path=nl_data_path,
        weights=type_weights,
        num_eeg_tokens=num_eeg_tokens,
        min_spells=min_spells, max_spells=max_spells,
        window_size=effective_window_size, window_step=window_step,
        exclude_subjects=exclude_subjects,
        trial_duration_pts=trial_duration_pts,
        **extra_kwargs,
    )
    val_dataset = DatasetClass(
        eeg_dir, tokenizer, split="val",
        nl_data_path=nl_data_path,
        weights=type_weights,
        num_eeg_tokens=num_eeg_tokens,
        min_spells=min_spells, max_spells=max_spells,
        window_size=effective_window_size, window_step=window_step,
        exclude_subjects=exclude_subjects,
        trial_duration_pts=trial_duration_pts,
        **{k: v for k, v in extra_kwargs.items() if k not in ("cand_dropout", "echo_dropout", "eeg_loss_weight")},
    )
    collator = BCIAgentCollator(tokenizer)

    # Eval every ~3 epochs worth of steps
    import math
    n_gpus = max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(train_dataset) / (per_device_batch_size * gradient_accumulation_steps * n_gpus))
    eval_steps = max(1, steps_per_epoch * 3)
    print(f"Eval/save every {eval_steps} steps (~3 epochs, {steps_per_epoch} steps/epoch)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
    )

    best_dir = Path(output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Build evaluation metrics (two_step for candidate mode)
    target_ids = get_target_token_ids(tokenizer)
    use_two_step = (fbcca_mode == "candidate")
    compute_metrics, preprocess_logits = build_metrics_fn(target_ids, two_step=use_two_step)

    trainer = BCIAgentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        encoder_lr=encoder_lr,
        best_model_dir=str(best_dir),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Print parameter summary
    enc_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "encoder." in n)
    lora_p = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "encoder." not in n and "embed_tokens" not in n and "lm_head" not in n
    )
    emb_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("embed_tokens" in n or "lm_head" in n))
    print(f"\nTrainable parameters (Stage 2):")
    print(f"  Encoder (FiLM+proj): {enc_p:>12,}  (lr={encoder_lr})")
    print(f"  LoRA:                {lora_p:>12,}  (lr={learning_rate})")
    print(f"  Embeddings:          {emb_p:>12,}  (lr={learning_rate})")
    print(f"  Total:               {enc_p + lora_p + emb_p:>12,}\n")

    print("Starting Stage 2 training...")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer._save_weights(str(final_dir))
    print(f"\nStage 2 complete. Model saved to {final_dir}")

    return trainer
