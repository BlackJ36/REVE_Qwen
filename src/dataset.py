"""BCI-EEG dataset for Qwen3-VL fine-tuning."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokens import BCI_END, BCI_PAD, BCI_START, TARGET_INDEX_TO_TOKEN

SYSTEM_PROMPT = "你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。"


class BCIEEGDataset(Dataset):
    """Loads pre-extracted REVE embeddings and creates training samples."""

    def __init__(self, embedding_dir, tokenizer, split="train"):
        """
        Args:
            embedding_dir: path containing {split}_embeddings.pt and {split}_labels.pt
            tokenizer: Qwen tokenizer with BCI special tokens registered
            split: 'train' or 'val'
        """
        self.tokenizer = tokenizer
        self.embedding_dir = Path(embedding_dir)

        # Load pre-extracted embeddings
        data = torch.load(
            self.embedding_dir / f"{split}_embeddings.pt", weights_only=True
        )
        self.embeddings = data["embeddings"]  # (N, reve_dim)
        self.labels = data["labels"]  # (N,) int, 0-39

        # Pre-build the chat template tokens
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self._build_template()

        print(f"[{split}] Loaded {len(self)} samples, embedding dim={self.embeddings.shape[-1]}")

    def _build_template(self):
        """Pre-tokenize the fixed parts of the chat template."""
        # Build the input text (without the target token)
        chat_prefix = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{BCI_START}{BCI_PAD}{BCI_END}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prefix_ids = self.tokenizer.encode(chat_prefix, add_special_tokens=False)
        self.suffix_text = "<|im_end|>"
        self.suffix_ids = self.tokenizer.encode(
            self.suffix_text, add_special_tokens=False
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_emb = self.embeddings[idx]  # (reve_dim,)
        target_idx = int(self.labels[idx])
        target_token = TARGET_INDEX_TO_TOKEN[target_idx]

        # Build full input_ids: prefix + target_token + suffix
        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
        input_ids = self.prefix_ids + target_ids + self.suffix_ids

        # Labels: -100 for everything except the target token position
        labels = [-100] * len(self.prefix_ids) + target_ids + self.suffix_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_embeddings": eeg_emb.float(),
        }


class BCIDataCollator:
    """Pads batch and handles EEG embeddings."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        eeg_embeddings = torch.stack([f["eeg_embeddings"] for f in features])

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            # Left-pad (Qwen uses left padding)
            offset = max_len - seq_len
            input_ids[i, offset:] = f["input_ids"]
            labels[i, offset:] = f["labels"]
            attention_mask[i, offset:] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "eeg_embeddings": eeg_embeddings,
        }
