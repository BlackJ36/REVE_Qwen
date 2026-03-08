"""Stage 2 dataset: multi-character spelling + NL dialogues.

Loads S2 dialogue JSONL and pre-extracted REVE embeddings.
For each EEG character, randomly samples a real trial embedding with the
matching label from the embedding bank.

Sequence format (Type A/C):
  system: ...
  user: <|bci_start|>[9 pads]<|bci_sep|>[9 pads]<|bci_sep|>...[9 pads]<|bci_end|>
  assistant: WATER<|im_end|>

Type D (pure NL): no EEG tokens, standard text dialogue.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokens import BCI_END, BCI_PAD, BCI_SEP, BCI_START

N_EEG_TOKENS = 9  # channels per character


class BCIS2Dataset(Dataset):
    """Stage 2 dataset with label-indexed embedding lookup."""

    def __init__(self, dialogue_path, embedding_path, tokenizer, split="train"):
        """
        Args:
            dialogue_path: path to s2_dialogues.jsonl
            embedding_path: path to {split}_embeddings.pt (for embedding bank)
            tokenizer: Qwen tokenizer with BCI special tokens
            split: 'train' or 'val' (determines embedding source)
        """
        self.tokenizer = tokenizer
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)

        # Load dialogues
        self.dialogues = []
        with open(dialogue_path) as f:
            for line in f:
                self.dialogues.append(json.loads(line))
        print(f"[S2 {split}] Loaded {len(self.dialogues)} dialogues")

        # Load embedding bank and build label index
        emb_data = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.emb_bank = emb_data["embeddings"]  # (N, 9, 512)
        emb_labels = emb_data["labels"]  # (N,)

        self.label_to_indices = {l: [] for l in range(40)}
        for i, label in enumerate(emb_labels.tolist()):
            self.label_to_indices[label].append(i)

        counts = [len(v) for v in self.label_to_indices.values()]
        print(f"  Embedding bank: {self.emb_bank.shape[0]} trials, "
              f"per-label: {min(counts)}-{max(counts)}")

        # Pre-tokenize fixed parts
        self._cache_tokens()

    def _cache_tokens(self):
        """Pre-tokenize special token sequences."""
        self.bci_start_ids = self.tokenizer.encode(BCI_START, add_special_tokens=False)
        self.bci_end_ids = self.tokenizer.encode(BCI_END, add_special_tokens=False)
        self.bci_sep_ids = self.tokenizer.encode(BCI_SEP, add_special_tokens=False)
        self.bci_pad_ids = self.tokenizer.encode(BCI_PAD, add_special_tokens=False)
        self.im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def _build_eeg_token_ids(self, n_chars):
        """Build token IDs for EEG block: <|bci_start|>[9p]<|bci_sep|>...[9p]<|bci_end|>"""
        ids = list(self.bci_start_ids)
        for i in range(n_chars):
            ids.extend(self.bci_pad_ids * N_EEG_TOKENS)
            if i < n_chars - 1:
                ids.extend(self.bci_sep_ids)
        ids.extend(self.bci_end_ids)
        return ids

    def _tokenize_dialogue(self, dialogue):
        """Convert dialogue to input_ids + labels + eeg_embeddings.

        Returns (input_ids, labels, eeg_embeddings) or None if invalid.
        """
        messages = dialogue["messages"]
        eeg_labels = dialogue["eeg_labels"]
        has_eeg = len(eeg_labels) > 0

        # Build chat sequence
        all_ids = []
        all_labels = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                header = self.tokenizer.encode(
                    f"<|im_start|>system\n{content}<|im_end|>\n",
                    add_special_tokens=False,
                )
                all_ids.extend(header)
                all_labels.extend([-100] * len(header))

            elif role == "user":
                # User header
                user_header = self.tokenizer.encode(
                    "<|im_start|>user\n", add_special_tokens=False,
                )
                all_ids.extend(user_header)
                all_labels.extend([-100] * len(user_header))

                if content == "__EEG__" and has_eeg:
                    # Replace with EEG tokens
                    eeg_ids = self._build_eeg_token_ids(len(eeg_labels))
                    all_ids.extend(eeg_ids)
                    all_labels.extend([-100] * len(eeg_ids))
                else:
                    # Regular text
                    text_ids = self.tokenizer.encode(content, add_special_tokens=False)
                    all_ids.extend(text_ids)
                    all_labels.extend([-100] * len(text_ids))

                # User footer
                footer = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                all_ids.extend(footer)
                all_labels.extend([-100] * len(footer))

            elif role == "assistant":
                # Assistant header (no loss)
                asst_header = self.tokenizer.encode(
                    "<|im_start|>assistant\n", add_special_tokens=False,
                )
                all_ids.extend(asst_header)
                all_labels.extend([-100] * len(asst_header))

                # Assistant content (loss computed here)
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                all_ids.extend(content_ids)
                all_labels.extend(content_ids)

                # im_end (loss)
                all_ids.extend(self.im_end_ids)
                all_labels.extend(self.im_end_ids)

        # Sample EEG embeddings
        eeg_embeddings = None
        if has_eeg:
            embs = []
            for label in eeg_labels:
                indices = self.label_to_indices[label]
                idx = random.choice(indices)
                embs.append(self.emb_bank[idx])  # (9, 512)
            eeg_embeddings = torch.stack(embs)  # (n_chars, 9, 512)

        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(all_labels, dtype=torch.long),
            eeg_embeddings,
        )

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        input_ids, labels, eeg_embeddings = self._tokenize_dialogue(dialogue)

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if eeg_embeddings is not None:
            result["eeg_embeddings"] = eeg_embeddings.float()  # (n_chars, 9, 512)
        return result


class BCIS2Collator:
    """Collates S2 batches with variable-length EEG embeddings."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        # Collect EEG embeddings: flatten all characters across batch
        all_eeg = []
        eeg_char_counts = []  # number of EEG characters per sample

        for i, f in enumerate(features):
            seq_len = f["input_ids"].size(0)
            offset = max_len - seq_len  # left-pad
            input_ids[i, offset:] = f["input_ids"]
            labels[i, offset:] = f["labels"]
            attention_mask[i, offset:] = 1

            if "eeg_embeddings" in f:
                all_eeg.append(f["eeg_embeddings"])  # (n_chars_i, 9, 512)
                eeg_char_counts.append(f["eeg_embeddings"].size(0))
            else:
                eeg_char_counts.append(0)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "eeg_char_counts": torch.tensor(eeg_char_counts, dtype=torch.long),
        }

        if all_eeg:
            # (total_chars, 9, 512) — flatten across batch
            result["eeg_embeddings"] = torch.cat(all_eeg, dim=0)

        return result
