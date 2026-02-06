"""End-to-end training: REVE (partially unfrozen) → projector → Qwen (LoRA)."""

import json
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .dataset_e2e import BCIE2EDataCollator, BCIE2EDataset
from .model_e2e import build_e2e_model


class BCIE2ETrainer(Trainer):
    """Trainer with 3 LR groups and dual checkpoint modes.

    LR groups:
        - REVE unfrozen layers: reve_lr (default 1e-5)
        - Projector: projector_lr (default 1e-3)
        - LoRA adapter: args.learning_rate (default 2e-4)

    Checkpoint modes:
        - "weights_only": saves reve_unfrozen.pt + projector.pt + LoRA adapter (~100-200MB)
        - "full": delegates to Trainer's default save (all weights + optimizer + scheduler)
    """

    def __init__(self, reve_lr=1e-5, projector_lr=1e-3, checkpoint_mode="weights_only", **kwargs):
        super().__init__(**kwargs)
        self.reve_lr = reve_lr
        self.projector_lr = projector_lr
        self.checkpoint_mode = checkpoint_mode

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        reve_params = []
        projector_params = []
        lora_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("reve."):
                reve_params.append(param)
            elif "projector" in name:
                projector_params.append(param)
            else:
                lora_params.append(param)

        optimizer_grouped_parameters = [
            {"params": reve_params, "lr": self.reve_lr},
            {"params": projector_params, "lr": self.projector_lr},
            {"params": lora_params, "lr": self.args.learning_rate},
        ]

        from torch.optim import AdamW
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )

        print(f"E2E Optimizer:")
        print(f"  REVE params: {len(reve_params)}, lr={self.reve_lr}")
        print(f"  Projector params: {len(projector_params)}, lr={self.projector_lr}")
        print(f"  LoRA params: {len(lora_params)}, lr={self.args.learning_rate}")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        eeg_tensor = inputs.pop("eeg_tensor", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            eeg_tensor=eeg_tensor,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if self.checkpoint_mode == "full":
            self._save_full(output_dir, state_dict)
        else:
            self._save_weights_only(output_dir)

    def _save_weights_only(self, output_dir):
        """Save only the trainable weights (~100-200MB). For inference/sharing."""
        output_dir = Path(output_dir)

        # 1. REVE unfrozen parameters
        reve_state = {
            name: param.data
            for name, param in self.model.reve.named_parameters()
            if param.requires_grad
        }
        if reve_state:
            torch.save(reve_state, output_dir / "reve_unfrozen.pt")

        # 2. Projector
        torch.save(self.model.projector.state_dict(), output_dir / "projector.pt")

        # 3. LoRA adapter
        self.model.qwen.save_pretrained(str(output_dir), safe_serialization=True)

        # 4. Tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(output_dir))

        # 5. Checkpoint mode marker
        meta = {"checkpoint_mode": "weights_only"}
        with open(output_dir / "e2e_meta.json", "w") as f:
            json.dump(meta, f)

    def _save_full(self, output_dir, state_dict=None):
        """Full checkpoint for training resumption. Delegates heavy lifting to Trainer."""
        output_dir = Path(output_dir)

        # Save all model components
        self._save_weights_only(output_dir)

        # Save optimizer and scheduler (Trainer handles this via save_checkpoint)
        # The parent Trainer._save_checkpoint calls us for model weights,
        # and separately saves optimizer/scheduler/rng states.
        meta = {"checkpoint_mode": "full"}
        with open(output_dir / "e2e_meta.json", "w") as f:
            json.dump(meta, f)


def run_e2e_training(
    eeg_dir="data/eeg_tensors",
    output_dir="output_e2e",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    from_modelscope=True,
    # REVE
    unfreeze_last_n=4,
    reve_lr=3e-5,
    # LoRA
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.05,
    # Quantization
    use_4bit=False,
    # Training
    per_device_batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    projector_lr=3e-3,
    num_epochs=30,
    early_stopping_patience=5,
    warmup_ratio=0.1,
    checkpoint_mode="weights_only",
    # DeepSpeed
    deepspeed_config="configs/ds_zero2.json",
):
    """End-to-end training entry point."""
    eeg_dir = Path(eeg_dir)

    print("Building E2E model...")
    model, tokenizer = build_e2e_model(
        model_name=model_name,
        from_modelscope=from_modelscope,
        unfreeze_last_n=unfreeze_last_n,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_4bit=use_4bit,
    )

    print("Loading E2E datasets...")
    train_dataset = BCIE2EDataset(eeg_dir, tokenizer, split="train")
    val_dataset = BCIE2EDataset(eeg_dir, tokenizer, split="val")
    collator = BCIE2EDataCollator(tokenizer)

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
        eval_strategy="epoch",
        save_strategy="epoch",
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
        gradient_checkpointing=not use_4bit,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not use_4bit else None,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    trainer = BCIE2ETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
        reve_lr=reve_lr,
        projector_lr=projector_lr,
        checkpoint_mode=checkpoint_mode,
    )

    # Print parameter summary
    reve_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("reve."))
    proj_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "projector" in n)
    lora_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("reve.") is False and "projector" not in n)
    print(f"\nTrainable parameters:")
    print(f"  REVE unfrozen:  {reve_p:>10,}  (lr={reve_lr})")
    print(f"  Projector:      {proj_p:>10,}  (lr={projector_lr})")
    print(f"  LoRA:           {lora_p:>10,}  (lr={learning_rate})")
    print(f"  Total:          {reve_p + proj_p + lora_p:>10,}")
    print(f"  Checkpoint mode: {checkpoint_mode}\n")

    print("Starting E2E training...")
    trainer.train()

    # Save final model
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer._save_weights_only(final_dir)
    print(f"\nFinal model saved to {final_dir}")

    return trainer
