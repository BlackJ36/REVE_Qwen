"""BCI-EEG dataset for end-to-end multi-spell sequence training."""

import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokens import (
    BCI_PAD,
    BCI_TRANS,
    TARGET_INDEX_TO_TOKEN,
)

SYSTEM_PROMPT = "你是一个脑机接口解码器。根据EEG信号，实时解码对应的目标。"
USER_PROMPT = "请依次解码以下EEG信号。"


class BCIE2EDataset(Dataset):
    """Multi-spell sequence dataset for end-to-end BCI training.

    Each sample contains K (min_spells to max_spells) consecutive SSVEP "spells"
    from the same subject+block, with sliding windows and transition tokens.
    """

    def __init__(
        self,
        eeg_dir,
        tokenizer,
        split="train",
        num_eeg_tokens=62,
        min_spells=5,
        max_spells=10,
        window_size=300,
        window_step=100,
    ):
        self.tokenizer = tokenizer
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
        self.channel_names = data["channel_names"]

        # Compute sliding window offsets: how many 1.5s windows fit in 3s signal
        total_timepoints = self.eeg_data.shape[2]  # 600
        self.window_offsets = []
        start = 0
        while start + window_size <= total_timepoints:
            self.window_offsets.append(start)
            start += window_step
        # e.g. [0, 100, 200, 300] for 600 pts, window=300, step=100

        # Group trial indices by (subject_id, block_id)
        self.groups = defaultdict(list)
        for idx in range(len(self.labels)):
            key = (int(self.subject_ids[idx]), int(self.block_ids[idx]))
            self.groups[key].append(idx)
        self.group_keys = list(self.groups.keys())

        # Pre-tokenize fixed token IDs
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.bci_trans_id = tokenizer.convert_tokens_to_ids(BCI_TRANS)

        # Target token IDs: map target index → single token ID
        self.target_token_ids = {}
        for target_idx, target_token in TARGET_INDEX_TO_TOKEN.items():
            ids = tokenizer.encode(target_token, add_special_tokens=False)
            assert len(ids) == 1, f"Target token {target_token} encoded to {len(ids)} tokens"
            self.target_token_ids[target_idx] = ids[0]

        # Pre-tokenize fixed text parts
        self._tokenize_parts()

        n_windows = len(self.window_offsets)
        print(
            f"[{split}] Loaded {len(self.eeg_data)} trials, "
            f"{len(self.group_keys)} groups, "
            f"{n_windows} windows/trial, "
            f"spells={min_spells}-{max_spells}"
        )

    def _tokenize_parts(self):
        """Pre-tokenize the fixed text parts of the chat template.

        Interleaved format (EEG and targets alternate in assistant turn):
          system: ...
          user: 请依次解码以下EEG信号。
          assistant: [62 pads]<|t05|><|bci_trans|>[62 pads]<|t12|>...[62 pads]<|tNN|><|im_end|>

        This ensures causal correctness: when predicting target_i,
        the model can only attend to EEG_1 through EEG_i (not future EEGs).
        """
        # System + user trigger → prefix (fixed for all samples)
        prefix_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{USER_PROMPT}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)

        # im_end for assistant
        self.im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        group_key = self.group_keys[idx % len(self.group_keys)]
        group_indices = self.groups[group_key]

        K = random.randint(self.min_spells, self.max_spells)
        # Sample K trials with replacement from this group
        chosen_indices = random.choices(group_indices, k=K)

        # Extract windows and targets
        eeg_windows = []
        target_indices = []
        for trial_idx in chosen_indices:
            target_indices.append(int(self.labels[trial_idx]))
            # Random window from available sliding windows
            offset = random.choice(self.window_offsets)
            window = self.eeg_data[trial_idx, :, offset:offset + self.window_size]  # (62, 300)
            eeg_windows.append(window)

        eeg_windows = torch.stack(eeg_windows)  # (K, 62, 300)

        # Build token sequence
        input_ids, labels = self._build_sequence(target_indices)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": K,
        }

    def _build_sequence(self, target_indices):
        """Build interleaved input_ids and labels for streaming-compatible training.

        Format (all in assistant turn, causal mask ensures streaming correctness):
          prefix: system + user + assistant_start
          assistant: [62 pads]<|t05|><|bci_trans|>[62 pads]<|t12|>...[62 pads]<|tNN|><|im_end|>

        Labels: only target tokens are supervised; pads, trans, im_end are -100.
        """
        K = len(target_indices)
        n = self.num_eeg_tokens  # 62

        # === Interleaved section (within assistant turn) ===
        interleaved_ids = []
        interleaved_labels = []

        for i in range(K):
            # EEG pads (replaced with projected embeddings during forward)
            interleaved_ids.extend([self.bci_pad_id] * n)
            interleaved_labels.extend([-100] * n)

            # Target token (supervised)
            tid = self.target_token_ids[target_indices[i]]
            interleaved_ids.append(tid)
            interleaved_labels.append(tid)

            # Transition separator (except after last spell)
            if i < K - 1:
                interleaved_ids.append(self.bci_trans_id)
                interleaved_labels.append(-100)

        # End of sequence
        interleaved_ids.extend(self.im_end_ids)
        interleaved_labels.extend([-100] * len(self.im_end_ids))

        # === Combine ===
        input_ids = self.prefix_ids + interleaved_ids
        labels = [-100] * len(self.prefix_ids) + interleaved_labels

        return input_ids, labels


class BCIE2EDataCollator:
    """Pads text sequences and concatenates EEG windows from variable-K samples."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        # Concatenate all EEG windows: (sum(K_i), 62, window_size)
        all_windows = []
        window_counts = []

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            offset = max_len - seq_len  # left-pad for Qwen
            input_ids[i, offset:] = f["input_ids"]
            labels[i, offset:] = f["labels"]
            attention_mask[i, offset:] = 1

            all_windows.append(f["eeg_windows"])  # (K_i, 62, 300)
            window_counts.append(f["num_spells"])

        eeg_windows = torch.cat(all_windows, dim=0)  # (total_K, 62, 300)
        window_counts = torch.tensor(window_counts, dtype=torch.long)  # (B,)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "eeg_windows": eeg_windows,
            "window_counts": window_counts,
        }
