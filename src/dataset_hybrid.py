"""Dataset for Plan A: custom Transformer decoder with 43-token vocab.

Simplified version of BCIE2EDataset that uses a tiny vocabulary instead of
the Qwen tokenizer. Same grouping/windowing logic, but no chat template.

Sequence format:
  <bos> [62 pads] target_0 <trans> [62 pads] target_1 ... [62 pads] target_K

Labels: only target tokens are supervised; everything else is -100.
"""

import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .model_hybrid import VOCAB_BOS, VOCAB_EEG_PAD, VOCAB_PAD, VOCAB_TARGETS_START, VOCAB_TRANS


class HybridDataset(Dataset):
    """Multi-spell sequence dataset for Plan A hybrid model.

    Each sample contains K spells from the same subject+block.
    Uses a tiny 43-token vocabulary instead of the Qwen tokenizer.

    Args:
        eeg_dir: directory containing {split}_eeg.pt files
        split: "train" or "val"
        num_eeg_tokens: number of EEG pad tokens per window (62)
        min_spells: minimum spells per sequence
        max_spells: maximum spells per sequence
        window_size: sliding window size in timepoints
        window_step: sliding window step in timepoints
    """

    def __init__(
        self,
        eeg_dir,
        split="train",
        num_eeg_tokens=62,
        min_spells=5,
        max_spells=10,
        window_size=300,
        window_step=100,
    ):
        self.eeg_dir = Path(eeg_dir)
        self.num_eeg_tokens = num_eeg_tokens
        self.min_spells = min_spells
        self.max_spells = max_spells
        self.window_size = window_size
        self.window_step = window_step

        data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
        self.eeg_data = data["eeg_data"]  # (N, 62, 600)
        self.labels = data["labels"]  # (N,)
        self.subject_ids = data["subject_ids"]  # (N,)
        self.block_ids = data["block_ids"]  # (N,)

        # Sliding window offsets
        total_timepoints = self.eeg_data.shape[2]
        self.window_offsets = []
        start = 0
        while start + window_size <= total_timepoints:
            self.window_offsets.append(start)
            start += window_step

        # Group by (subject, block)
        self.groups = defaultdict(list)
        for idx in range(len(self.labels)):
            key = (int(self.subject_ids[idx]), int(self.block_ids[idx]))
            self.groups[key].append(idx)
        self.group_keys = list(self.groups.keys())

        print(
            f"[{split}] HybridDataset: {len(self.eeg_data)} trials, "
            f"{len(self.group_keys)} groups, "
            f"{len(self.window_offsets)} windows/trial, "
            f"spells={min_spells}-{max_spells}"
        )

    def __len__(self):
        avg_spells = (self.min_spells + self.max_spells) / 2
        return int(len(self.eeg_data) / avg_spells)

    def __getitem__(self, idx):
        group_key = self.group_keys[idx % len(self.group_keys)]
        group_indices = self.groups[group_key]

        K = random.randint(self.min_spells, self.max_spells)
        chosen_indices = random.choices(group_indices, k=K)

        eeg_windows = []
        target_indices = []
        for trial_idx in chosen_indices:
            target_indices.append(int(self.labels[trial_idx]))
            offset = random.choice(self.window_offsets)
            window = self.eeg_data[trial_idx, :, offset:offset + self.window_size]
            eeg_windows.append(window)

        eeg_windows = torch.stack(eeg_windows)  # (K, 62, window_size)

        input_ids, labels = self._build_sequence(target_indices)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": K,
        }

    def _build_sequence(self, target_indices):
        """Build sequence with 43-token vocab.

        Format: <bos> [62 pads] target_0 <trans> [62 pads] target_1 ... target_K
        Labels: -100 for everything except target tokens.
        """
        K = len(target_indices)
        n = self.num_eeg_tokens

        input_ids = [VOCAB_BOS]
        labels = [-100]

        for i in range(K):
            # EEG placeholder pads
            input_ids.extend([VOCAB_EEG_PAD] * n)
            labels.extend([-100] * n)

            # Target token (supervised)
            tid = VOCAB_TARGETS_START + target_indices[i]  # 0-39 -> 3-42
            input_ids.append(tid)
            labels.append(tid)

            # Transition separator (except after last spell)
            if i < K - 1:
                input_ids.append(VOCAB_TRANS)
                labels.append(-100)

        return input_ids, labels


class HybridDataCollator:
    """Pads sequences and concatenates EEG windows for Plan A model."""

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), VOCAB_PAD, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        all_windows = []
        window_counts = []

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            offset = max_len - seq_len  # left-pad
            input_ids[i, offset:] = f["input_ids"]
            labels[i, offset:] = f["labels"]
            attention_mask[i, offset:] = 1

            all_windows.append(f["eeg_windows"])
            window_counts.append(f["num_spells"])

        eeg_windows = torch.cat(all_windows, dim=0)
        window_counts = torch.tensor(window_counts, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "eeg_windows": eeg_windows,
            "window_counts": window_counts,
        }
