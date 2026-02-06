"""BCI-EEG dataset for end-to-end training with raw EEG tensors."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokens import BCI_END, BCI_PAD, BCI_START, TARGET_INDEX_TO_TOKEN

SYSTEM_PROMPT = "你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。"


class BCIE2EDataset(Dataset):
    """Loads raw preprocessed EEG tensors for end-to-end training."""

    def __init__(self, eeg_dir, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.eeg_dir = Path(eeg_dir)

        data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
        self.eeg_data = data["eeg_data"]  # (N, 62, 600)
        self.labels = data["labels"]  # (N,)
        self.channel_names = data["channel_names"]  # list of 62 names

        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self._build_template()

        print(f"[{split}] Loaded {len(self)} E2E samples, shape={self.eeg_data.shape[1:]}")

    def _build_template(self):
        """Pre-tokenize the fixed parts of the chat template."""
        chat_prefix = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{BCI_START}{BCI_PAD}{BCI_END}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prefix_ids = self.tokenizer.encode(chat_prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

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
        eeg_tensor = self.eeg_data[idx]  # (62, 600)
        target_idx = int(self.labels[idx])
        seq = self.target_sequences[target_idx]

        return {
            "input_ids": seq["input_ids"],
            "labels": seq["labels"],
            "eeg_tensor": eeg_tensor.float(),
        }


class BCIE2EDataCollator:
    """Pads text sequences and stacks raw EEG tensors."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        eeg_tensors = torch.stack([f["eeg_tensor"] for f in features])  # (B, 62, 600)

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            offset = max_len - seq_len  # left-pad for Qwen
            input_ids[i, offset:] = f["input_ids"]
            labels[i, offset:] = f["labels"]
            attention_mask[i, offset:] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "eeg_tensor": eeg_tensors,
        }
