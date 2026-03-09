"""Training logic for BCI-Qwen (pre-extracted embeddings approach).

Stage 1: Train projector + new token embeddings (Qwen frozen)
Stage 2: Train LoRA + projector (with S1 weights loaded)
"""

from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .dataset import BCIDataCollator, BCIEEGDataset
from .dataset_s2 import BCIS2Collator, BCIS2Dataset
from .metrics_bci_agent import build_metrics_fn
from .metrics_s2 import build_s2_metrics_fn
from .model import build_model
from .tokens import get_target_token_ids


class BCITrainer(Trainer):
    """Custom Trainer that passes eeg_embeddings to model forward."""

    def __init__(self, projector_lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.projector_lr = projector_lr

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        projector_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "projector" in name:
                projector_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if projector_params:
            groups.append({"params": projector_params, "lr": self.projector_lr})
        if other_params:
            groups.append({"params": other_params, "lr": self.args.learning_rate})

        from torch.optim import AdamW
        self.optimizer = AdamW(
            groups, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=self.args.weight_decay,
        )

        print(f"Optimizer: projector={len(projector_params)} params (lr={self.projector_lr}), "
              f"other={len(other_params)} params (lr={self.args.learning_rate})")
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        eeg_embeddings = inputs.pop("eeg_embeddings", None)
        eeg_char_counts = inputs.pop("eeg_char_counts", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            eeg_embeddings=eeg_embeddings,
            eeg_char_counts=eeg_char_counts,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.model.save_pretrained(output_dir)

    def _load_best_model(self):
        """Load best checkpoint using our custom save format (projector.pt + LoRA adapter)."""
        from pathlib import Path
        from peft import PeftModel

        best_path = self.state.best_model_checkpoint
        if best_path is None:
            return

        print(f"Loading best model from {best_path}")

        # Load projector
        proj_path = Path(best_path) / "projector.pt"
        if proj_path.exists():
            self.model.projector.load_state_dict(
                torch.load(proj_path, map_location="cpu", weights_only=True)
            )

        # Load LoRA adapter (load to CPU first to avoid DDP OOM)
        if isinstance(self.model.qwen, PeftModel) and (Path(best_path) / "adapter_config.json").exists():
            self.model.qwen.load_adapter(str(best_path), adapter_name="default",
                                         is_trainable=True, torch_device="cpu")

        # Load new token embeddings (S1 without modules_to_save)
        qwen_path = Path(best_path) / "qwen_trainable.pt"
        if qwen_path.exists():
            ovs = self.model.original_vocab_size
            qwen_state = torch.load(qwen_path, map_location="cpu", weights_only=True)
            if "embed_tokens.new_rows" in qwen_state and ovs is not None:
                self.model.qwen.get_input_embeddings().weight.data[ovs:] = qwen_state["embed_tokens.new_rows"]
            if "lm_head.new_rows" in qwen_state and ovs is not None:
                self.model.qwen.lm_head.weight.data[ovs:] = qwen_state["lm_head.new_rows"]

        print(f"Best model loaded from {best_path}")


def run_training(
    embedding_dir="data/embeddings",
    output_dir="output",
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    from_modelscope=True,
    stage=1,
    stage1_checkpoint=None,
    # S2 data
    s2_train_path=None,
    s2_val_path=None,
    # LoRA
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # Training
    per_device_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    projector_lr=1e-3,
    num_epochs=30,
    early_stopping_patience=5,
    warmup_ratio=0.1,
    # DeepSpeed
    deepspeed_config=None,
):
    """Main training entry point.

    Stage 1: Single-trial classification (BCIEEGDataset)
    Stage 2: Multi-char spelling + NL dialogues (BCIS2Dataset)
    """
    embedding_dir = Path(embedding_dir)

    # Build model
    print(f"Building model (Stage {stage})...")
    model, tokenizer = build_model(
        model_name=model_name,
        from_modelscope=from_modelscope,
        reve_dim=512,
        stage=stage,
        stage1_checkpoint=stage1_checkpoint,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Build datasets
    print("Loading datasets...")
    if stage == 2 and s2_train_path is not None:
        # S2: dialogue data + embedding bank for label-indexed lookup
        # Train uses train subjects' embeddings, val uses val subjects'
        # for honest cross-subject evaluation
        train_emb_path = str(embedding_dir / "train_embeddings.pt")
        val_emb_path = str(embedding_dir / "val_embeddings.pt")
        train_dataset = BCIS2Dataset(s2_train_path, train_emb_path, tokenizer, split="train")
        val_dataset = BCIS2Dataset(s2_val_path, val_emb_path, tokenizer, split="val")
        collator = BCIS2Collator(tokenizer)
    else:
        # S1: single-trial classification
        train_dataset = BCIEEGDataset(embedding_dir, tokenizer, split="train")
        val_dataset = BCIEEGDataset(embedding_dir, tokenizer, split="val")
        collator = BCIDataCollator(tokenizer)

    # Build metrics
    if stage == 2 and s2_train_path is not None:
        compute_metrics, preprocess_logits = build_s2_metrics_fn()
    else:
        target_token_ids = get_target_token_ids(tokenizer)
        compute_metrics, preprocess_logits = build_metrics_fn(target_token_ids)

    # Best model selection: S1 uses bci_acc (classification), S2 uses eval_loss (generation)
    is_s2 = stage == 2 and s2_train_path is not None
    best_metric = "eval_loss" if is_s2 else "eval_bci_acc"
    best_greater = False if is_s2 else True

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=best_greater,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
        gradient_checkpointing=(stage == 2),  # S1 doesn't need it (Qwen frozen)
        gradient_checkpointing_kwargs={"use_reentrant": False} if stage == 2 else None,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    trainer = BCITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
        projector_lr=projector_lr,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
    )

    print("Starting training...")
    trainer.train()

    # Save final
    final_dir = Path(output_dir) / "final"
    model.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")

    return trainer
