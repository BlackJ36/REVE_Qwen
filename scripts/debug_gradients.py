"""Debug script to check if gradients are flowing correctly."""

import torch
from pathlib import Path

# Load a small batch and check gradients
def debug_training():
    from src.model import build_model
    from src.dataset import BCIEEGDataset, BCIDataCollator

    print("=== Gradient Debug ===\n")

    # Build model (use smaller settings for quick test)
    print("Loading model...")
    model, tokenizer = build_model(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        from_modelscope=True,
        lora_rank=16,
        lora_alpha=32,
    )
    model = model.cuda()

    # Check projector requires_grad
    print("\n1. Projector parameters:")
    for name, param in model.projector.named_parameters():
        print(f"   {name}: requires_grad={param.requires_grad}, shape={param.shape}")

    # Load a batch
    print("\n2. Loading test batch...")
    dataset = BCIEEGDataset(Path("data/embeddings"), tokenizer, split="train")
    collator = BCIDataCollator(tokenizer)
    batch = collator([dataset[i] for i in range(4)])

    # Move to GPU
    batch = {k: v.cuda() for k, v in batch.items()}

    # Check input shapes
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   eeg_embeddings shape: {batch['eeg_embeddings'].shape}")

    # Check bci_pad_id positions
    bci_pad_id = model.bci_pad_id
    pad_mask = batch['input_ids'] == bci_pad_id
    print(f"   bci_pad_id: {bci_pad_id}")
    print(f"   pad positions per sample: {pad_mask.sum(dim=1).tolist()}")

    # Forward pass
    print("\n3. Forward pass...")
    eeg_emb = batch.pop("eeg_embeddings")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        eeg_embeddings=eeg_emb,
    )

    loss = outputs.loss
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Random baseline (ln(40)): {torch.log(torch.tensor(40.0)).item():.4f}")

    # Backward pass
    print("\n4. Backward pass...")
    loss.backward()

    # Check projector gradients
    print("\n5. Projector gradients:")
    has_grad = False
    for name, param in model.projector.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"   {name}: grad_norm={grad_norm:.6f}")
            has_grad = True
        else:
            print(f"   {name}: grad=None ❌")

    if not has_grad:
        print("\n   ❌ PROBLEM: Projector has no gradients!")
        print("   This means the EEG embeddings are not contributing to the loss.")
    else:
        print("\n   ✅ Projector gradients are flowing correctly.")

    # Check LoRA gradients
    print("\n6. Sample LoRA gradients:")
    lora_grad_count = 0
    for name, param in model.qwen.named_parameters():
        if param.grad is not None and "lora" in name.lower():
            lora_grad_count += 1
            if lora_grad_count <= 3:
                print(f"   {name}: grad_norm={param.grad.norm().item():.6f}")
    print(f"   ... and {lora_grad_count - 3} more LoRA params with gradients")


if __name__ == "__main__":
    import os
    os.environ["ALL_PROXY"] = ""
    os.environ["all_proxy"] = ""
    debug_training()
