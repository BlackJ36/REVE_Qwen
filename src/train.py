"""Training logic for BCI-Qwen."""

import json
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .dataset import BCIDataCollator, BCIEEGDataset
from .model import build_model


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

    def _save(self, output_dir=None, state_dict=None):
        """Override to save projector separately and avoid tied weights issue."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save the PEFT model (LoRA adapter)
        self.model.qwen.save_pretrained(output_dir, safe_serialization=True)

        # Save projector separately
        projector_path = Path(output_dir) / "projector.pt"
        torch.save(self.model.projector.state_dict(), projector_path)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


def run_training(
    embedding_dir="data/embeddings",
    output_dir="output",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    from_modelscope=True,
    # LoRA
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.05,
    # Quantization
    use_4bit=False,
    # Training
    per_device_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_epochs=10,
    warmup_ratio=0.05,
    # DeepSpeed
    deepspeed_config="configs/ds_zero2.json",
):
    """Main training entry point."""
    embedding_dir = Path(embedding_dir)

    # Load metadata to get REVE dim
    meta_path = embedding_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        reve_dim = meta["reve_dim"]
        print(f"REVE embedding dim from metadata: {reve_dim}")
    else:
        reve_dim = 512
        print(f"No metadata found, using default REVE dim: {reve_dim}")

    # Build model
    print("Building model...")
    model, tokenizer = build_model(
        model_name=model_name,
        from_modelscope=from_modelscope,
        reve_dim=reve_dim,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_4bit=use_4bit,
    )

    # Build datasets
    print("Loading datasets...")
    train_dataset = BCIEEGDataset(embedding_dir, tokenizer, split="train")
    val_dataset = BCIEEGDataset(embedding_dir, tokenizer, split="val")
    collator = BCIDataCollator(tokenizer)

    # Set up different learning rates for projector vs LoRA
    projector_params = [
        p for n, p in model.named_parameters()
        if "projector" in n and p.requires_grad
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if "projector" not in n and p.requires_grad
    ]

    # Training arguments
    # Note: gradient_checkpointing is handled by prepare_model_for_kbit_training when use_4bit=True
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
        dataloader_num_workers=8,  # 增加数据加载并行度
        dataloader_pin_memory=True,  # 加速 CPU->GPU 传输
        dataloader_prefetch_factor=4,  # 预加载更多 batch
        dataloader_persistent_workers=True,  # 保持 worker 进程存活
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=not use_4bit,  # Already handled in prepare_model_for_kbit_training
        gradient_checkpointing_kwargs={"use_reentrant": False} if not use_4bit else None,
    )

    # Create trainer
    trainer = BCITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Manually set higher LR for projector via param groups
    # (Trainer handles this through optimizers if we override create_optimizer)
    print(f"Projector params: {sum(p.numel() for p in projector_params)}")
    print(f"Other trainable params: {sum(p.numel() for p in other_params)}")

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model (using our custom _save method)
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter
    model.qwen.save_pretrained(str(final_dir), safe_serialization=True)

    # Save projector
    torch.save(model.projector.state_dict(), str(final_dir / "projector.pt"))

    # Save tokenizer
    tokenizer.save_pretrained(str(final_dir))

    print(f"Model saved to {final_dir}")
    print(f"  - LoRA adapter: {final_dir}/adapter_model.safetensors")
    print(f"  - Projector: {final_dir}/projector.pt")

    return trainer
