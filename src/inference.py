"""Inference utilities for BCI-Qwen classification."""

import torch
import torch.nn.functional as F
from typing import List, Tuple

from .tokens import TARGET_TOKENS, TARGET_INDEX_TO_TOKEN, BCI_START, BCI_END, BCI_PAD


def get_target_token_ids(tokenizer) -> torch.Tensor:
    """Get token IDs for all 40 target tokens."""
    target_ids = []
    for i in range(40):
        token = TARGET_INDEX_TO_TOKEN[i]
        token_id = tokenizer.convert_tokens_to_ids(token)
        target_ids.append(token_id)
    return torch.tensor(target_ids)


def classify_eeg(
    model,
    tokenizer,
    eeg_embedding: torch.Tensor,
    device: str = "cuda",
) -> Tuple[int, torch.Tensor]:
    """
    Classify a single EEG trial using constrained logits.

    Instead of using model.generate() which might produce arbitrary text,
    we do a single forward pass and look at the logits for the position
    where the target token should be generated, then pick the highest
    probability among the 40 valid target tokens.

    Args:
        model: BCIQwenForCausalLM model
        tokenizer: Tokenizer with BCI special tokens
        eeg_embedding: (reve_dim,) tensor, single trial embedding
        device: Device to run inference on

    Returns:
        (predicted_class, probabilities) where:
        - predicted_class: int 0-39
        - probabilities: (40,) tensor of probabilities for each class
    """
    model.eval()

    # Build input sequence (without target token)
    prompt = (
        f"<|im_start|>system\n你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。<|im_end|>\n"
        f"<|im_start|>user\n{BCI_START}{BCI_PAD}{BCI_END}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    # Get target token IDs for classification
    target_token_ids = get_target_token_ids(tokenizer).to(device)

    # Prepare EEG embedding
    eeg_embedding = eeg_embedding.unsqueeze(0).to(device)  # (1, reve_dim)

    with torch.no_grad():
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eeg_embeddings=eeg_embedding,
        )

        # Get logits for the last position (where target should be generated)
        # Shape: (1, seq_len, vocab_size)
        logits = outputs.logits

        # Get logits for the last token position
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Extract logits only for the 40 target tokens
        target_logits = last_logits[target_token_ids]  # (40,)

        # Convert to probabilities
        probabilities = F.softmax(target_logits, dim=0)

        # Get prediction
        predicted_class = probabilities.argmax().item()

    return predicted_class, probabilities


def classify_batch(
    model,
    tokenizer,
    eeg_embeddings: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classify a batch of EEG trials.

    Args:
        model: BCIQwenForCausalLM model
        tokenizer: Tokenizer with BCI special tokens
        eeg_embeddings: (B, reve_dim) tensor
        device: Device to run inference on

    Returns:
        (predictions, probabilities) where:
        - predictions: (B,) tensor of predicted classes (0-39)
        - probabilities: (B, 40) tensor of class probabilities
    """
    model.eval()
    batch_size = eeg_embeddings.size(0)

    # Build input sequence
    prompt = (
        f"<|im_start|>system\n你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。<|im_end|>\n"
        f"<|im_start|>user\n{BCI_START}{BCI_PAD}{BCI_END}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Tokenize once and repeat for batch
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.expand(batch_size, -1).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Get target token IDs
    target_token_ids = get_target_token_ids(tokenizer).to(device)

    # Move embeddings to device
    eeg_embeddings = eeg_embeddings.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eeg_embeddings=eeg_embeddings,
        )

        # Get logits for last position: (B, vocab_size)
        last_logits = outputs.logits[:, -1, :]

        # Extract target token logits: (B, 40)
        target_logits = last_logits[:, target_token_ids]

        # Probabilities and predictions
        probabilities = F.softmax(target_logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return predictions, probabilities


def compute_accuracy(
    model,
    tokenizer,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict:
    """
    Compute classification accuracy on a dataset.

    Args:
        model: BCIQwenForCausalLM model
        tokenizer: Tokenizer
        embeddings: (N, reve_dim) tensor
        labels: (N,) tensor of ground truth labels (0-39)
        device: Device
        batch_size: Batch size for inference

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    all_preds = []
    all_probs = []

    n_samples = embeddings.size(0)

    for i in range(0, n_samples, batch_size):
        batch_emb = embeddings[i:i + batch_size]
        preds, probs = classify_batch(model, tokenizer, batch_emb, device)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    predictions = torch.cat(all_preds)
    probabilities = torch.cat(all_probs)
    labels = labels.cpu()

    # Compute metrics
    correct = (predictions == labels).sum().item()
    accuracy = correct / n_samples

    # Top-5 accuracy
    _, top5_indices = probabilities.topk(5, dim=1)
    top5_correct = (top5_indices == labels.unsqueeze(1)).any(dim=1).sum().item()
    top5_accuracy = top5_correct / n_samples

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "correct": correct,
        "total": n_samples,
        "predictions": predictions,
        "probabilities": probabilities,
    }
