"""Dataset for BCI agent: Stage 1 (classification) and Stage 2 (mixed instruction).

Stage 1: Multi-spell EEG -> target tokens, formatted with Qwen chat template.
Stage 2: Mixed data types (single decode, streaming, batch, error handling, pure NL).

Both stages use Qwen tokenizer and chat template for consistency.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .templates_zh import (
    SYSTEM_PROMPT,
    make_batch_messages,
    make_error_messages,
    make_single_decode_messages,
    make_streaming_messages,
)
from .tokens import BCI_PAD, BCI_TRANS, TARGET_INDEX_TO_TOKEN

# BETA subjects with <30% FBCCA accuracy — near-random SSVEP signal.
# Note: subject_id overlaps between Benchmark and BETA (both start from 1).
# S11 exists in both datasets; excluding it also removes good BM S11 (~240 trials).
# S41/S55/S59 are BETA-only (ID > 35). S64 is in BETA val set.
BETA_BAD_SUBJECTS = {11, 41, 55, 59, 64}


def _filter_by_subjects(data, exclude_subjects):
    """Filter .pt data dict, removing trials from excluded subjects.

    Returns filtered data dict (all tensor keys masked) and count removed.
    """
    mask = torch.ones(len(data["labels"]), dtype=torch.bool)
    for sid in exclude_subjects:
        mask &= data["subject_ids"] != sid
    n_removed = int((~mask).sum())
    filtered = {k: v[mask] if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
    return filtered, n_removed


class BCIAgentStage1Dataset(Dataset):
    """Stage 1: multi-spell EEG -> target classification with chat template.

    Same grouping/windowing as HybridDataset, but uses Qwen tokenizer + chat template.
    Format per sample:
        system: BCI decoder prompt
        user: (instruction)
        assistant: [62 pads]<|tXX|><|bci_trans|>[62 pads]<|tYY|>...

    Args:
        eeg_dir: directory containing {split}_eeg.pt files
        tokenizer: Qwen tokenizer with BCI special tokens registered
        split: "train" or "val"
        num_eeg_tokens: number of EEG pad tokens per window (62)
        min_spells / max_spells: spells per sequence
        window_size / window_step: sliding window parameters
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
        exclude_subjects=None,
        trial_duration_pts=600,
    ):
        self.eeg_dir = Path(eeg_dir)
        self.tokenizer = tokenizer
        self.num_eeg_tokens = num_eeg_tokens
        self.min_spells = min_spells
        self.max_spells = max_spells
        self.window_size = window_size
        self.window_step = window_step

        data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
        if exclude_subjects:
            data, n_removed = _filter_by_subjects(data, exclude_subjects)
            print(f"[{split}] Excluded subjects {exclude_subjects}: removed {n_removed} trials")
        self.eeg_data = data["eeg_data"]       # (N, 62, total_T)

        # Per-trial valid timepoints (handles zero-padded BETA S01-S19)
        if "valid_pts" in data:
            self.valid_pts = data["valid_pts"]  # (N,)
        else:
            self.valid_pts = torch.full((len(data["labels"]),), self.eeg_data.shape[2], dtype=torch.long)

        # Truncate EEG to requested duration
        if trial_duration_pts < self.eeg_data.shape[2]:
            self.eeg_data = self.eeg_data[:, :, :trial_duration_pts]
            self.valid_pts = self.valid_pts.clamp(max=trial_duration_pts)
        self.labels = data["labels"]           # (N,)
        self.subject_ids = data["subject_ids"] # (N,)
        self.block_ids = data["block_ids"]     # (N,)

        # Sliding window offsets (global, per-trial filtering in __getitem__)
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

        # Pre-tokenize special tokens
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.bci_trans_id = tokenizer.convert_tokens_to_ids(BCI_TRANS)
        self.target_ids = {
            i: tokenizer.convert_tokens_to_ids(tok)
            for i, tok in TARGET_INDEX_TO_TOKEN.items()
        }

        # Build the prefix: system + user turn (tokenize once, reuse)
        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请解码以下脑电信号。"},
        ]
        # Apply chat template for system+user, then start of assistant turn
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
        )
        self.prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        print(
            f"[{split}] Stage1 Dataset: {len(self.eeg_data)} trials, "
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
            # Filter offsets to stay within valid (non-padded) data
            vp = int(self.valid_pts[trial_idx])
            valid_offsets = [o for o in self.window_offsets if o + self.window_size <= vp]
            if not valid_offsets:
                valid_offsets = [0]  # fallback: use first window
            offset = random.choice(valid_offsets)
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
        """Build token sequence: prefix + [pads target trans]* + eos.

        Labels: -100 for everything except target token positions.
        """
        n = self.num_eeg_tokens
        K = len(target_indices)

        # Start with prefix (system + user + assistant start)
        input_ids = list(self.prefix_ids)
        labels = [-100] * len(input_ids)

        for i in range(K):
            # EEG pad tokens (will be replaced by encoder output)
            input_ids.extend([self.bci_pad_id] * n)
            labels.extend([-100] * n)

            # Target token (supervised)
            tid = self.target_ids[target_indices[i]]
            input_ids.append(tid)
            labels.append(tid)

            # Transition separator (except after last spell)
            if i < K - 1:
                input_ids.append(self.bci_trans_id)
                labels.append(-100)

        # EOS token
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids.append(eos_id)
            labels.append(eos_id)

        return input_ids, labels


class BCIAgentStage2Dataset(Dataset):
    """Stage 2: mixed instruction data for interactive BCI agent.

    Samples from 5 data types with configurable weights:
      A (30%): single EEG decode -> NL response
      B (30%): multi-turn streaming spelling
      C (20%): pure natural language (from external JSONL)
      D (10%): error handling / commands
      E (10%): batch spelling

    Args:
        eeg_dir: directory with preprocessed EEG
        tokenizer: Qwen tokenizer
        split: "train" or "val"
        nl_data_path: path to pure NL JSONL file (Type C), optional
        weights: dict of type weights, default {A:0.3, B:0.3, C:0.2, D:0.1, E:0.1}
        num_eeg_tokens: EEG pad tokens per window
        min_spells / max_spells: for streaming/batch sequences
        window_size / window_step: EEG windowing
    """

    def __init__(
        self,
        eeg_dir,
        tokenizer,
        split="train",
        nl_data_path=None,
        weights=None,
        num_eeg_tokens=62,
        min_spells=3,
        max_spells=8,
        window_size=300,
        window_step=100,
        exclude_subjects=None,
        trial_duration_pts=600,
    ):
        self.eeg_dir = Path(eeg_dir)
        self.tokenizer = tokenizer
        self.num_eeg_tokens = num_eeg_tokens
        self.min_spells = min_spells
        self.max_spells = max_spells
        self.window_size = window_size
        self.window_step = window_step
        self.split = split

        # Weights for each type
        self.weights = weights or {"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.1, "E": 0.1}
        self.types = list(self.weights.keys())
        self.type_probs = [self.weights[t] for t in self.types]

        # Load EEG data
        data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
        if exclude_subjects:
            data, n_removed = _filter_by_subjects(data, exclude_subjects)
            print(f"[{split}] Excluded subjects {exclude_subjects}: removed {n_removed} trials")
        self.eeg_data = data["eeg_data"]

        # Per-trial valid timepoints (handles zero-padded BETA S01-S19)
        if "valid_pts" in data:
            self.valid_pts = data["valid_pts"]
        else:
            self.valid_pts = torch.full((len(data["labels"]),), self.eeg_data.shape[2], dtype=torch.long)

        # Truncate EEG to requested duration
        if trial_duration_pts < self.eeg_data.shape[2]:
            self.eeg_data = self.eeg_data[:, :, :trial_duration_pts]
            self.valid_pts = self.valid_pts.clamp(max=trial_duration_pts)
        self.labels = data["labels"]
        self.subject_ids = data["subject_ids"]
        self.block_ids = data["block_ids"]

        # Sliding window offsets (global, per-trial filtering in __getitem__)
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

        # Load pure NL data (Type C)
        self.nl_data = []
        if nl_data_path and Path(nl_data_path).exists():
            with open(nl_data_path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Expect {"messages": [{"role": ..., "content": ...}, ...]}
                    if "messages" in item:
                        self.nl_data.append(item["messages"])
            print(f"[{split}] Loaded {len(self.nl_data)} pure NL samples from {nl_data_path}")
        else:
            # If no NL data, redistribute weight to other types
            if "C" in self.weights and self.weights["C"] > 0:
                print(f"[{split}] No NL data file, redistributing Type C weight to A and B")
                extra = self.weights.pop("C")
                self.weights["A"] = self.weights.get("A", 0) + extra / 2
                self.weights["B"] = self.weights.get("B", 0) + extra / 2
                self.types = list(self.weights.keys())
                self.type_probs = [self.weights[t] for t in self.types]

        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)

        eeg_types = sum(1 for t in self.types if t != "C")
        print(
            f"[{split}] Stage2 Dataset: {len(self.eeg_data)} trials, "
            f"{len(self.group_keys)} groups, "
            f"types={self.types}, weights={[f'{p:.0%}' for p in self.type_probs]}"
        )

    def __len__(self):
        # Roughly match Stage 1 size
        avg_spells = (self.min_spells + self.max_spells) / 2
        return int(len(self.eeg_data) / avg_spells)

    def __getitem__(self, idx):
        # Sample data type
        data_type = random.choices(self.types, weights=self.type_probs, k=1)[0]

        if data_type == "A":
            return self._make_type_a(idx)
        elif data_type == "B":
            return self._make_type_b(idx)
        elif data_type == "C":
            return self._make_type_c(idx)
        elif data_type == "D":
            return self._make_type_d(idx)
        elif data_type == "E":
            return self._make_type_e(idx)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _sample_trials(self, K=1):
        """Sample K trials from the same subject+block group."""
        group_key = random.choice(self.group_keys)
        group_indices = self.groups[group_key]
        chosen = random.choices(group_indices, k=K)

        eeg_windows = []
        label_indices = []
        for trial_idx in chosen:
            label_indices.append(int(self.labels[trial_idx]))
            # Filter offsets to stay within valid (non-padded) data
            vp = int(self.valid_pts[trial_idx])
            valid_offsets = [o for o in self.window_offsets if o + self.window_size <= vp]
            if not valid_offsets:
                valid_offsets = [0]
            offset = random.choice(valid_offsets)
            window = self.eeg_data[trial_idx, :, offset:offset + self.window_size]
            eeg_windows.append(window)

        return torch.stack(eeg_windows), label_indices

    def _tokenize_messages(self, messages):
        """Tokenize a list of messages and create labels.

        Labels: -100 for system + user turns, actual token ids for assistant turns.
        """
        # Replace BCI_PAD strings with actual pad token IDs after tokenization
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Build labels: only supervise assistant turns
        labels = [-100] * len(input_ids)

        # Find assistant turn boundaries by tokenizing incrementally
        current_pos = 0
        for i, msg in enumerate(messages):
            # Tokenize up to and including this message
            partial_messages = messages[:i + 1]
            partial_text = self.tokenizer.apply_chat_template(
                partial_messages, tokenize=False, add_generation_prompt=False,
            )
            partial_ids = self.tokenizer.encode(partial_text, add_special_tokens=False)
            end_pos = len(partial_ids)

            if msg["role"] == "assistant":
                # Supervise this assistant turn
                for j in range(current_pos, min(end_pos, len(labels))):
                    labels[j] = input_ids[j]

            current_pos = end_pos

        return input_ids, labels

    def _count_pads(self, input_ids):
        """Count how many BCI_PAD tokens are in the sequence."""
        return sum(1 for t in input_ids if t == self.bci_pad_id)

    def _make_type_a(self, idx):
        """Type A: single EEG decode -> NL response."""
        eeg_windows, label_indices = self._sample_trials(K=1)
        messages = make_single_decode_messages(
            label_indices[0], num_eeg_pads=self.num_eeg_tokens,
        )
        input_ids, labels = self._tokenize_messages(messages)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": 1,
        }

    def _make_type_b(self, idx):
        """Type B: multi-turn streaming spelling."""
        K = random.randint(self.min_spells, self.max_spells)
        eeg_windows, label_indices = self._sample_trials(K=K)
        messages = make_streaming_messages(
            label_indices, num_eeg_pads=self.num_eeg_tokens,
        )
        input_ids, labels = self._tokenize_messages(messages)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": K,
        }

    def _make_type_c(self, idx):
        """Type C: pure NL (no EEG)."""
        if not self.nl_data:
            # Fallback to Type A if no NL data
            return self._make_type_a(idx)

        messages = random.choice(self.nl_data)
        input_ids, labels = self._tokenize_messages(messages)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": torch.zeros(0, 62, self.window_size),  # empty
            "num_spells": 0,
        }

    def _make_type_d(self, idx):
        """Type D: error handling / commands."""
        error_type = random.choice(["low_confidence", "undo", "clear", "help"])

        if error_type == "low_confidence":
            # Use actual EEG data (model should learn to detect low quality)
            eeg_windows, label_indices = self._sample_trials(K=1)
        else:
            eeg_windows = torch.zeros(0, 62, self.window_size)

        spelled = ""
        if error_type == "undo":
            # Generate a random partial spelling for context
            n_chars = random.randint(1, 5)
            chars = [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n_chars)]
            spelled = "".join(chars[:-1]) if len(chars) > 1 else ""

        messages = make_error_messages(error_type, spelled=spelled)
        input_ids, labels = self._tokenize_messages(messages)

        num_spells = 1 if error_type == "low_confidence" else 0
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float() if eeg_windows.numel() > 0 else eeg_windows,
            "num_spells": num_spells,
        }

    def _make_type_e(self, idx):
        """Type E: single-turn batch spelling."""
        K = random.randint(self.min_spells, self.max_spells)
        eeg_windows, label_indices = self._sample_trials(K=K)
        messages = make_batch_messages(
            label_indices, num_eeg_pads=self.num_eeg_tokens,
        )
        input_ids, labels = self._tokenize_messages(messages)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": K,
        }


class BCIAgentCollator:
    """Pads sequences and concatenates EEG windows across the batch.

    Works for both Stage 1 and Stage 2 datasets.
    """

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)
        has_loss_weights = "loss_weights" in features[0]

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        if has_loss_weights:
            loss_weights = torch.zeros(batch_size, max_len, dtype=torch.float32)

        all_windows = []
        window_counts = []

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            offset = max_len - seq_len  # left-pad (Qwen uses left padding)
            input_ids[i, offset:] = f["input_ids"]
            labels[i, offset:] = f["labels"]
            attention_mask[i, offset:] = 1
            if has_loss_weights:
                loss_weights[i, offset:] = f["loss_weights"]

            if f["eeg_windows"].numel() > 0:
                all_windows.append(f["eeg_windows"])
            window_counts.append(f["num_spells"])

        # Concatenate EEG windows across batch
        if all_windows:
            eeg_windows = torch.cat(all_windows, dim=0)
        else:
            eeg_windows = torch.zeros(0, 62, features[0]["eeg_windows"].shape[-1] if features[0]["eeg_windows"].dim() > 1 else 300)
        window_counts = torch.tensor(window_counts, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "eeg_windows": eeg_windows,
            "window_counts": window_counts,
        }
        if has_loss_weights:
            batch["loss_weights"] = loss_weights
        return batch
