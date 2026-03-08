"""BCI-EEG dataset for Qwen3 fine-tuning with pre-extracted REVE embeddings.

Each trial has N EEG token embeddings (N × 512d), represented as
N <|bci_pad|> tokens in the chat template. At forward time, each pad position
is replaced with its corresponding REVE embedding via a projector.

N depends on extraction config:
  200pts (1s): 9 channels × 1 patch = 9 tokens
  400pts (2s): 9 channels × 2 patches = 18 tokens

Sequence format:
  system: 你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。
  user: <|bci_start|><|bci_pad|>×N<|bci_end|>
  assistant: <|tXX|><|im_end|>
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokens import BCI_END, BCI_PAD, BCI_START, TARGET_INDEX_TO_TOKEN

SYSTEM_PROMPT = "你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。"

# Default number of EEG tokens (overridden by embedding file metadata)
N_EEG_TOKENS = 9


class BCIEEGDataset(Dataset):
    """Loads pre-extracted REVE embeddings (N, T, 512) and creates training samples."""

    def __init__(self, embedding_dir, tokenizer, split="train", n_eeg_tokens=None):
        """
        Args:
            embedding_dir: path containing {split}_embeddings.pt
            tokenizer: Qwen tokenizer with BCI special tokens registered
            split: 'train' or 'val'
            n_eeg_tokens: override pad token count (None = auto-detect from embedding file)
        """
        self.tokenizer = tokenizer
        self.embedding_dir = Path(embedding_dir)

        # Load pre-extracted embeddings
        data = torch.load(
            self.embedding_dir / f"{split}_embeddings.pt", weights_only=True
        )
        self.embeddings = data["embeddings"]  # (N, T, 512)
        self.labels = data["labels"]  # (N,) int, 0-39

        # Auto-detect n_eeg_tokens from file metadata or embedding shape
        if n_eeg_tokens is not None:
            self.n_eeg_tokens = n_eeg_tokens
        elif "n_eeg_tokens" in data:
            self.n_eeg_tokens = int(data["n_eeg_tokens"])
        else:
            self.n_eeg_tokens = self.embeddings.shape[1]  # backward compat

        # Pre-build the chat template tokens
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self._build_template()

        print(f"[{split}] Loaded {len(self)} samples, "
              f"embedding shape={list(self.embeddings.shape)}")

    def _build_template(self):
        """Pre-tokenize the fixed parts of the chat template."""
        # 9 pads between BCI_START and BCI_END
        pads = BCI_PAD * self.n_eeg_tokens
        chat_prefix = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{BCI_START}{pads}{BCI_END}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prefix_ids = self.tokenizer.encode(chat_prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

        # Pre-compute all 40 target sequences (avoid tokenizing in __getitem__)
        self.target_sequences = {}
        for target_idx, target_token in TARGET_INDEX_TO_TOKEN.items():
            target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
            input_ids = self.prefix_ids + target_ids + self.suffix_ids
            labels = [-100] * len(self.prefix_ids) + target_ids + self.suffix_ids
            self.target_sequences[target_idx] = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_emb = self.embeddings[idx]  # (9, 512)
        target_idx = int(self.labels[idx])

        seq = self.target_sequences[target_idx]

        return {
            "input_ids": seq["input_ids"],
            "labels": seq["labels"],
            "eeg_embeddings": eeg_emb.float(),
        }


class BCIDataCollator:
    """Pads batch and stacks EEG embeddings."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        eeg_embeddings = torch.stack([f["eeg_embeddings"] for f in features])  # (B, 9, 512)

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
