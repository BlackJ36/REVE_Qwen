"""Training with Unsloth for efficient fine-tuning."""

import json
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .dataset import BCIDataCollator, BCIEEGDataset
from .model_unsloth import build_model_unsloth


class BCITrainer(Trainer):
    """Custom Trainer that passes eeg_embeddings to model forward."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        eeg_embeddings = inputs.pop("eeg_embeddings", None)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            eeg_embeddings=eeg_embeddings,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def run_training_unsloth(
    embedding_dir="data/embeddings",
    output_dir="output",
    model_name="unsloth/Qwen3-VL-4B-Instruct",
    # LoRA
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.0,
    # Training
    per_device_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_epochs=10,
    warmup_ratio=0.05,
    max_seq_length=512,
):
    """Training with Unsloth."""
    embedding_dir = Path(embedding_dir)

    # Load metadata
    meta_path = embedding_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        reve_dim = meta["reve_dim"]
        print(f"REVE embedding dim: {reve_dim}")
    else:
        reve_dim = 512

    # Build model with Unsloth
    print("Building model with Unsloth...")
    model, tokenizer = build_model_unsloth(
        model_name=model_name,
        reve_dim=reve_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
    )

    # Build datasets
    print("Loading datasets...")
    train_dataset = BCIEEGDataset(embedding_dir, tokenizer, split="train")
    val_dataset = BCIEEGDataset(embedding_dir, tokenizer, split="val")
    collator = BCIDataCollator(tokenizer)

    # Training arguments (no DeepSpeed, Unsloth handles optimization)
    # Note: save_strategy="no" to avoid safetensors tied weights issue
    # We save manually at the end using PEFT's save method
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
        save_strategy="no",  # Save manually to avoid tied weights error
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_8bit",  # Unsloth recommended
    )

    # Create trainer
    trainer = BCITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    print("Starting training with Unsloth...")
    trainer.train()

    # Save using Unsloth's method (handles tied weights properly)
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter only (recommended for PEFT models)
    inner_model = model.model  # Get the underlying PEFT model
    inner_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save projector separately
    torch.save(model.projector.state_dict(), str(final_dir / "projector.pt"))

    print(f"Model saved to {final_dir}")
    print(f"  - LoRA adapter: {final_dir}/adapter_model.safetensors")
    print(f"  - Projector: {final_dir}/projector.pt")

    return trainer
