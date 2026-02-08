"""Plan A training: HybridEncoder + custom Transformer decoder.

Simpler than E2E pipeline: no LoRA, no REVE unfreezing.
2 LR groups: encoder projector (1e-3), decoder (5e-4).
"""

import json
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .dataset_hybrid import HybridDataCollator, HybridDataset
from .model_hybrid import build_hybrid_model


class HybridTrainer(Trainer):
    """Trainer with 2 LR groups for hybrid model.

    LR groups:
        - Encoder projector: projector_lr (default 1e-3)
        - Decoder: args.learning_rate (default 5e-4)
    """

    def __init__(self, projector_lr=1e-3, best_model_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.projector_lr = projector_lr
        self.best_model_dir = best_model_dir
        self.best_eval_loss = float("inf")

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        projector_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder.projector" in name:
                projector_params.append(param)
            else:
                decoder_params.append(param)

        optimizer_grouped_parameters = []
        if projector_params:
            optimizer_grouped_parameters.append(
                {"params": projector_params, "lr": self.projector_lr}
            )
        if decoder_params:
            optimizer_grouped_parameters.append(
                {"params": decoder_params, "lr": self.args.learning_rate}
            )

        from torch.optim import AdamW
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )

        print(f"Hybrid Optimizer ({len(optimizer_grouped_parameters)} groups):")
        print(f"  Projector: {len(projector_params)} params, lr={self.projector_lr}")
        print(f"  Decoder:   {len(decoder_params)} params, lr={self.args.learning_rate}")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        eeg_windows = inputs.pop("eeg_windows", None)
        window_counts = inputs.pop("window_counts", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            eeg_windows=eeg_windows,
            window_counts=window_counts,
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

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self._save_weights(output_dir)

    def _save_weights(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "encoder_projector": self.model.encoder.projector.state_dict(),
            "decoder": self.model.decoder.state_dict(),
        }, output_dir / "hybrid_model.pt")

        meta = {"model_type": "hybrid_transformer", "vocab_size": self.model.decoder.vocab_size}
        with open(output_dir / "hybrid_meta.json", "w") as f:
            json.dump(meta, f)


def run_hybrid_training(
    eeg_dir="data/eeg_tensors",
    output_dir="output_hybrid",
    reve_dir="models",
    # Decoder architecture
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    # Training
    per_device_batch_size=64,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    projector_lr=1e-3,
    num_epochs=50,
    warmup_ratio=0.1,
    # Multi-spell
    min_spells=5,
    max_spells=10,
    window_size=300,
    window_step=100,
    num_eeg_tokens=62,
    # DeepSpeed
    deepspeed_config="configs/ds_zero2.json",
):
    """Plan A training entry point."""
    print("Building Hybrid Transformer model...")
    model = build_hybrid_model(
        reve_dir=reve_dir,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        window_size=window_size,
    )

    print("Loading datasets...")
    train_dataset = HybridDataset(
        eeg_dir, split="train", num_eeg_tokens=num_eeg_tokens,
        min_spells=min_spells, max_spells=max_spells,
        window_size=window_size, window_step=window_step,
    )
    val_dataset = HybridDataset(
        eeg_dir, split="val", num_eeg_tokens=num_eeg_tokens,
        min_spells=min_spells, max_spells=max_spells,
        window_size=window_size, window_step=window_step,
    )
    collator = HybridDataCollator()

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
        load_best_model_at_end=False,
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

    trainer = HybridTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        projector_lr=projector_lr,
        best_model_dir=str(best_dir),
    )

    # Print summary
    proj_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "encoder.projector" in n)
    dec_p = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "encoder.projector" not in n)
    print(f"\nTrainable parameters:")
    print(f"  Encoder projector: {proj_p:>10,}  (lr={projector_lr})")
    print(f"  Decoder:           {dec_p:>10,}  (lr={learning_rate})")
    print(f"  Total:             {proj_p + dec_p:>10,}\n")

    print("Starting training...")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    trainer._save_weights(str(final_dir))
    print(f"\nFinal model saved to {final_dir}")

    return trainer
