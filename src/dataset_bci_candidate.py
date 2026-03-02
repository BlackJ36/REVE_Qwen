"""Dataset with decoder candidate injection for BCI agent training.

Stage 1: Same classification task as BCIAgentStage1Dataset, but each spell
  includes decoder (FBCCA/TRCA/eTRCA) top-3 predictions as explicit context tokens.

Stage 2: Word-level spelling — assembles multi-spell sequences that spell
  real words, using label→trial mapping to find matching EEG data.

Per-spell token format:
    [62×pad] <|rank1|><|tXX|> <|rank2|><|tYY|> <|rank3|><|tZZ|> <|conf_X|> <|tNN|> <|bci_trans|>
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^
             decoder context (all -100 labels)                                supervised

Where tXX/tYY/tZZ are decoder's top-3 predictions and tNN is the true target.
Supported decoder types: fbcca, trca, etrca (set via decoder_type parameter).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset_bci_agent import BETA_BAD_SUBJECTS, BCIAgentCollator, _filter_by_subjects
from .templates_zh import KEYBOARD_CHARS, SYSTEM_PROMPT, SYSTEM_PROMPT_SPELLING
from .tokens import (
    BCI_PAD,
    BCI_TRANS,
    CONF_HIGH,
    CONF_LOW,
    CONF_MID,
    RANK1,
    RANK2,
    RANK3,
    TARGET_INDEX_TO_TOKEN,
    score_gap_to_conf_token,
    score_gap_to_conf_token_adaptive,
)
from .word_vocab import WordVocab, generate_random_sequence, sample_word, word_to_labels


class CandidateStage1Dataset(Dataset):
    """Stage 1: multi-spell EEG classification with decoder candidate context.

    Same grouping/windowing as BCIAgentStage1Dataset, but inserts decoder top-3
    candidate tokens between the EEG pads and the supervised target.

    Args:
        eeg_dir: directory containing {split}_eeg.pt and {split}_{decoder_type}.pt
        tokenizer: Qwen tokenizer with BCI special tokens registered
        split: "train" or "val"
        num_eeg_tokens: EEG pad tokens per window (62)
        min_spells / max_spells: spells per sequence
        window_size / window_step: sliding window parameters
        exclude_subjects: set of subject IDs to exclude
        decoder_type: "fbcca", "trca", or "etrca" (determines which precomputed file to load)
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
        decoder_type="fbcca",
    ):
        self.eeg_dir = Path(eeg_dir)
        self.tokenizer = tokenizer
        self.num_eeg_tokens = num_eeg_tokens
        self.min_spells = min_spells
        self.max_spells = max_spells
        self.window_size = window_size
        self.window_step = window_step
        self.duration_scale = trial_duration_pts / 600.0

        # Load EEG data
        data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
        if exclude_subjects:
            data, n_removed = _filter_by_subjects(data, exclude_subjects)
            print(f"[{split}] Excluded subjects {exclude_subjects}: removed {n_removed} trials")
        self.eeg_data = data["eeg_data"]       # (N, 62, total_T)
        self.labels = data["labels"]           # (N,)
        self.subject_ids = data["subject_ids"]
        self.block_ids = data["block_ids"]
        N = len(self.labels)

        # Truncate EEG to requested duration
        if trial_duration_pts < self.eeg_data.shape[2]:
            self.eeg_data = self.eeg_data[:, :, :trial_duration_pts]

        # Load precomputed decoder candidates (duration-aware filename)
        # Supports: fbcca, trca, etrca
        self.decoder_type = decoder_type
        if trial_duration_pts == 600:
            cand_filename = f"{split}_{decoder_type}.pt"
        else:
            cand_filename = f"{split}_{decoder_type}_{trial_duration_pts}pt.pt"
        cand_path = self.eeg_dir / cand_filename
        if not cand_path.exists():
            precompute_hint = {
                "fbcca": f"python scripts/precompute_fbcca.py --eeg_dir {self.eeg_dir} --trial_duration {trial_duration_pts / 200.0}",
                "trca":  f"python scripts/precompute_trca.py --eeg_dir {self.eeg_dir} --trial_duration {trial_duration_pts / 200.0}",
                "etrca": f"python scripts/precompute_trca.py --eeg_dir {self.eeg_dir} --trial_duration {trial_duration_pts / 200.0} --ensemble",
            }
            raise FileNotFoundError(
                f"Precomputed {decoder_type} not found: {cand_path}\n"
                f"Run: {precompute_hint.get(decoder_type, 'unknown decoder_type')}"
            )
        cand_data = torch.load(cand_path, weights_only=True)
        self.cand_top3_indices = cand_data["top3_indices"]  # (N_full, num_offsets, 3)
        self.cand_top3_scores = cand_data["top3_scores"]    # (N_full, num_offsets, 3)

        # If subjects were filtered, we need to filter candidate data too
        if exclude_subjects:
            orig_data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
            mask = torch.ones(len(orig_data["labels"]), dtype=torch.bool)
            for sid in exclude_subjects:
                mask &= orig_data["subject_ids"] != sid
            self.cand_top3_indices = self.cand_top3_indices[mask]
            self.cand_top3_scores = self.cand_top3_scores[mask]

        assert len(self.cand_top3_indices) == N, \
            f"{decoder_type} data size mismatch: {len(self.cand_top3_indices)} vs {N} trials"

        # Sliding window offsets
        total_timepoints = self.eeg_data.shape[2]
        self.window_offsets = []
        start = 0
        while start + window_size <= total_timepoints:
            self.window_offsets.append(start)
            start += window_step

        # Group by (subject, block)
        self.groups = defaultdict(list)
        for idx in range(N):
            key = (int(self.subject_ids[idx]), int(self.block_ids[idx]))
            self.groups[key].append(idx)
        self.group_keys = list(self.groups.keys())

        # Pre-tokenize special tokens
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.bci_trans_id = tokenizer.convert_tokens_to_ids(BCI_TRANS)
        self.rank_ids = [
            tokenizer.convert_tokens_to_ids(RANK1),
            tokenizer.convert_tokens_to_ids(RANK2),
            tokenizer.convert_tokens_to_ids(RANK3),
        ]
        self.conf_ids = {
            CONF_HIGH: tokenizer.convert_tokens_to_ids(CONF_HIGH),
            CONF_MID: tokenizer.convert_tokens_to_ids(CONF_MID),
            CONF_LOW: tokenizer.convert_tokens_to_ids(CONF_LOW),
        }
        self.target_ids = {
            i: tokenizer.convert_tokens_to_ids(tok)
            for i, tok in TARGET_INDEX_TO_TOKEN.items()
        }

        # Build prefix: system + user turn (tokenize once)
        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请解码以下脑电信号。"},
        ]
        prefix_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
        )
        self.prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        print(
            f"[{split}] CandidateStage1: {N} trials, "
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
        fbcca_candidates = []  # list of (top3_idx, top3_sc) per spell

        for trial_idx in chosen_indices:
            target_indices.append(int(self.labels[trial_idx]))

            # Pick random window offset
            offset_idx = random.randrange(len(self.window_offsets))
            offset = self.window_offsets[offset_idx]
            window = self.eeg_data[trial_idx, :, offset:offset + self.window_size]
            eeg_windows.append(window)

            # Look up precomputed FBCCA for this trial + offset
            top3_idx = self.cand_top3_indices[trial_idx, offset_idx].tolist()  # [3]
            top3_sc = self.cand_top3_scores[trial_idx, offset_idx].tolist()    # [3]
            fbcca_candidates.append((top3_idx, top3_sc))

        eeg_windows = torch.stack(eeg_windows)  # (K, 62, window_size)
        input_ids, labels = self._build_sequence(target_indices, fbcca_candidates)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": K,
        }

    def _build_sequence(self, target_indices, fbcca_candidates):
        """Build token sequence with FBCCA candidate injection.

        Format per spell:
            [62×pad] [rank1][tXX] [rank2][tYY] [rank3][tZZ] [conf_X] [target] [trans]
        Labels: -100 everywhere except target token positions.
        """
        n = self.num_eeg_tokens
        K = len(target_indices)

        input_ids = list(self.prefix_ids)
        labels = [-100] * len(input_ids)

        for i in range(K):
            # EEG pad tokens
            input_ids.extend([self.bci_pad_id] * n)
            labels.extend([-100] * n)

            # FBCCA candidates: [rank1][tXX] [rank2][tYY] [rank3][tZZ]
            top3_idx, top3_sc = fbcca_candidates[i]
            for rank_j in range(3):
                input_ids.append(self.rank_ids[rank_j])
                labels.append(-100)
                input_ids.append(self.target_ids[top3_idx[rank_j]])
                labels.append(-100)

            # Confidence token (duration-adaptive)
            conf_token = score_gap_to_conf_token_adaptive(
                top3_sc[0], top3_sc[1], self.duration_scale)
            input_ids.append(self.conf_ids[conf_token])
            labels.append(-100)

            # True target (supervised)
            tid = self.target_ids[target_indices[i]]
            input_ids.append(tid)
            labels.append(tid)

            # Transition separator (except after last spell)
            if i < K - 1:
                input_ids.append(self.bci_trans_id)
                labels.append(-100)

        # EOS
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids.append(eos_id)
            labels.append(eos_id)

        return input_ids, labels


class CandidateStage2Dataset(Dataset):
    """Stage 2: word-level spelling with decoder candidate injection.

    Samples from data types:
      - "word" (40%): spell a real word using EEG trials with matching labels
      - "random" (30%): random char sequence (prevents LM shortcuts)
      - "nl" (15%): pure natural language (no EEG)
      - "error" (15%): error handling / commands

    The label→trial mapping per group enables spelling any word: given "HELP",
    find trials with labels [7, 4, 11, 15] within the same subject+block.

    Args:
        eeg_dir: directory with preprocessed EEG + precomputed decoder candidates
        tokenizer: Qwen tokenizer with BCI special tokens registered
        split: "train" or "val"
        nl_data_path: path to pure NL JSONL file (optional)
        weights: dict of type weights
        decoder_type: "fbcca", "trca", or "etrca"
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
        word_vocab=None,
        trial_duration_pts=600,
        decoder_type="fbcca",
    ):
        self.eeg_dir = Path(eeg_dir)
        self.tokenizer = tokenizer
        self.num_eeg_tokens = num_eeg_tokens
        self.min_spells = min_spells
        self.max_spells = max_spells
        self.window_size = window_size
        self.window_step = window_step
        self.split = split
        self.duration_scale = trial_duration_pts / 600.0

        # Word vocabulary (custom or built-in)
        self.vocab = word_vocab if word_vocab is not None else WordVocab()

        # Weights for data types
        self.weights = weights or {"word": 0.4, "random": 0.3, "nl": 0.15, "error": 0.15}
        self.types = list(self.weights.keys())
        self.type_probs = [self.weights[t] for t in self.types]

        # Load EEG data
        data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
        if exclude_subjects:
            data, n_removed = _filter_by_subjects(data, exclude_subjects)
            print(f"[{split}] Excluded subjects {exclude_subjects}: removed {n_removed} trials")
        self.eeg_data = data["eeg_data"]
        self.labels = data["labels"]
        self.subject_ids = data["subject_ids"]
        self.block_ids = data["block_ids"]
        N = len(self.labels)

        # Truncate EEG to requested duration
        if trial_duration_pts < self.eeg_data.shape[2]:
            self.eeg_data = self.eeg_data[:, :, :trial_duration_pts]

        # Load precomputed decoder candidates (duration-aware filename)
        self.decoder_type = decoder_type
        if trial_duration_pts == 600:
            cand_filename = f"{split}_{decoder_type}.pt"
        else:
            cand_filename = f"{split}_{decoder_type}_{trial_duration_pts}pt.pt"
        cand_path = self.eeg_dir / cand_filename
        if not cand_path.exists():
            precompute_hint = {
                "fbcca": f"python scripts/precompute_fbcca.py --eeg_dir {self.eeg_dir} --trial_duration {trial_duration_pts / 200.0}",
                "trca":  f"python scripts/precompute_trca.py --eeg_dir {self.eeg_dir} --trial_duration {trial_duration_pts / 200.0}",
                "etrca": f"python scripts/precompute_trca.py --eeg_dir {self.eeg_dir} --trial_duration {trial_duration_pts / 200.0} --ensemble",
            }
            raise FileNotFoundError(
                f"Precomputed {decoder_type} not found: {cand_path}\n"
                f"Run: {precompute_hint.get(decoder_type, 'unknown decoder_type')}"
            )
        cand_data = torch.load(cand_path, weights_only=True)
        self.cand_top3_indices = cand_data["top3_indices"]
        self.cand_top3_scores = cand_data["top3_scores"]

        if exclude_subjects:
            orig_data = torch.load(self.eeg_dir / f"{split}_eeg.pt", weights_only=True)
            mask = torch.ones(len(orig_data["labels"]), dtype=torch.bool)
            for sid in exclude_subjects:
                mask &= orig_data["subject_ids"] != sid
            self.cand_top3_indices = self.cand_top3_indices[mask]
            self.cand_top3_scores = self.cand_top3_scores[mask]

        assert len(self.cand_top3_indices) == N

        # Sliding window offsets
        total_timepoints = self.eeg_data.shape[2]
        self.window_offsets = []
        start = 0
        while start + window_size <= total_timepoints:
            self.window_offsets.append(start)
            start += window_step

        # Group by (subject, block) and build label→trial index
        self.groups = defaultdict(list)
        self.label_to_trials = defaultdict(lambda: defaultdict(list))
        for idx in range(N):
            key = (int(self.subject_ids[idx]), int(self.block_ids[idx]))
            self.groups[key].append(idx)
            label = int(self.labels[idx])
            self.label_to_trials[key][label].append(idx)
        self.group_keys = list(self.groups.keys())

        # Load NL data
        self.nl_data = []
        if nl_data_path and Path(nl_data_path).exists():
            with open(nl_data_path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    if "messages" in item:
                        self.nl_data.append(item["messages"])
            print(f"[{split}] Loaded {len(self.nl_data)} pure NL samples")
        else:
            if "nl" in self.weights and self.weights["nl"] > 0:
                print(f"[{split}] No NL data, redistributing to word/random")
                extra = self.weights.pop("nl")
                self.weights["word"] = self.weights.get("word", 0) + extra / 2
                self.weights["random"] = self.weights.get("random", 0) + extra / 2
                self.types = list(self.weights.keys())
                self.type_probs = [self.weights[t] for t in self.types]

        # Pre-tokenize
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.bci_trans_id = tokenizer.convert_tokens_to_ids(BCI_TRANS)
        self.rank_ids = [
            tokenizer.convert_tokens_to_ids(RANK1),
            tokenizer.convert_tokens_to_ids(RANK2),
            tokenizer.convert_tokens_to_ids(RANK3),
        ]
        self.conf_ids = {
            CONF_HIGH: tokenizer.convert_tokens_to_ids(CONF_HIGH),
            CONF_MID: tokenizer.convert_tokens_to_ids(CONF_MID),
            CONF_LOW: tokenizer.convert_tokens_to_ids(CONF_LOW),
        }
        self.target_ids = {
            i: tokenizer.convert_tokens_to_ids(tok)
            for i, tok in TARGET_INDEX_TO_TOKEN.items()
        }

        print(
            f"[{split}] CandidateStage2: {N} trials, "
            f"{len(self.group_keys)} groups, "
            f"types={self.types}, weights={[f'{p:.0%}' for p in self.type_probs]}"
        )

    def __len__(self):
        avg_spells = (self.min_spells + self.max_spells) / 2
        return int(len(self.eeg_data) / avg_spells)

    def __getitem__(self, idx):
        data_type = random.choices(self.types, weights=self.type_probs, k=1)[0]

        if data_type == "word":
            return self._make_word_sequence()
        elif data_type == "random":
            return self._make_random_sequence()
        elif data_type == "nl":
            return self._make_nl()
        elif data_type == "error":
            return self._make_error()
        else:
            return self._make_word_sequence()

    def _get_trial_for_label(self, group_key, label):
        """Find a trial with the given label in the group. Returns (trial_idx, offset_idx, window, fbcca)."""
        trials = self.label_to_trials[group_key].get(label, [])
        if not trials:
            return None
        trial_idx = random.choice(trials)
        offset_idx = random.randrange(len(self.window_offsets))
        offset = self.window_offsets[offset_idx]
        window = self.eeg_data[trial_idx, :, offset:offset + self.window_size]
        top3_idx = self.cand_top3_indices[trial_idx, offset_idx].tolist()
        top3_sc = self.cand_top3_scores[trial_idx, offset_idx].tolist()
        return window, (top3_idx, top3_sc)

    def _make_word_sequence(self):
        """Spell a real word using EEG trials with matching labels."""
        word, label_indices = self.vocab.sample()
        if label_indices is None:
            # Fallback to random
            return self._make_random_sequence()

        # Truncate to max_spells
        if len(label_indices) > self.max_spells:
            label_indices = label_indices[:self.max_spells]
            word = word[:self.max_spells]

        # Find a group that has all needed labels
        group_key = self._find_group_for_labels(label_indices)
        if group_key is None:
            return self._make_random_sequence()

        return self._assemble_spelling_sequence(word, label_indices, group_key)

    def _make_random_sequence(self):
        """Random char sequence — forces model to rely on EEG, not LM."""
        length = random.randint(self.min_spells, self.max_spells)
        word, label_indices = self.vocab.random_sequence(length)

        group_key = self._find_group_for_labels(label_indices)
        if group_key is None:
            # Extremely unlikely but handle gracefully
            group_key = random.choice(self.group_keys)
            group_indices = self.groups[group_key]
            chosen = random.choices(group_indices, k=length)
            label_indices = [int(self.labels[t]) for t in chosen]
            word = "".join(KEYBOARD_CHARS[l] for l in label_indices)

        return self._assemble_spelling_sequence(word, label_indices, group_key)

    def _find_group_for_labels(self, label_indices):
        """Find a group that has at least one trial for each required label."""
        needed = set(label_indices)
        # Try a few random groups
        candidates = random.sample(self.group_keys, min(10, len(self.group_keys)))
        for gk in candidates:
            if all(len(self.label_to_trials[gk].get(l, [])) > 0 for l in needed):
                return gk
        # Fallback: exhaustive search
        for gk in self.group_keys:
            if all(len(self.label_to_trials[gk].get(l, [])) > 0 for l in needed):
                return gk
        return None

    def _assemble_spelling_sequence(self, word, label_indices, group_key):
        """Build a multi-turn spelling chat sequence with FBCCA candidates."""
        eeg_windows = []
        fbcca_candidates = []

        for label in label_indices:
            result = self._get_trial_for_label(group_key, label)
            if result is None:
                # Should not happen if _find_group_for_labels succeeded
                return self._make_random_sequence()
            window, fbcca = result
            eeg_windows.append(window)
            fbcca_candidates.append(fbcca)

        eeg_windows = torch.stack(eeg_windows)

        # Build chat template with streaming spelling format
        input_ids, labels = self._build_spelling_sequence(
            word, label_indices, fbcca_candidates,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": eeg_windows.float(),
            "num_spells": len(label_indices),
        }

    def _build_spelling_sequence(self, word, target_indices, fbcca_candidates):
        """Build multi-turn spelling sequence with candidate injection.

        Uses the same format as CandidateStage1Dataset._build_sequence
        but wrapped in a streaming spelling chat template.
        """
        n = self.num_eeg_tokens
        K = len(target_indices)

        # Build prefix with spelling system prompt
        prefix_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SPELLING},
            {"role": "user", "content": "开始拼写。"},
        ]
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        labels = [-100] * len(input_ids)

        # Each spell: pads + FBCCA candidates + target + trans
        spelled = ""
        for i in range(K):
            # EEG pads
            input_ids.extend([self.bci_pad_id] * n)
            labels.extend([-100] * n)

            # FBCCA candidates
            top3_idx, top3_sc = fbcca_candidates[i]
            for rank_j in range(3):
                input_ids.append(self.rank_ids[rank_j])
                labels.append(-100)
                input_ids.append(self.target_ids[top3_idx[rank_j]])
                labels.append(-100)

            # Confidence (duration-adaptive)
            conf_token = score_gap_to_conf_token_adaptive(
                top3_sc[0], top3_sc[1], self.duration_scale)
            input_ids.append(self.conf_ids[conf_token])
            labels.append(-100)

            # True target (supervised)
            tid = self.target_ids[target_indices[i]]
            input_ids.append(tid)
            labels.append(tid)

            # Transition
            if i < K - 1:
                input_ids.append(self.bci_trans_id)
                labels.append(-100)

            spelled += KEYBOARD_CHARS[target_indices[i]]

        # EOS
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids.append(eos_id)
            labels.append(eos_id)

        return input_ids, labels

    def _make_nl(self):
        """Pure NL sample (no EEG)."""
        if not self.nl_data:
            return self._make_word_sequence()

        messages = random.choice(self.nl_data)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Supervise assistant turns only
        labels = [-100] * len(input_ids)
        current_pos = 0
        for i, msg in enumerate(messages):
            partial = messages[:i + 1]
            partial_text = self.tokenizer.apply_chat_template(
                partial, tokenize=False, add_generation_prompt=False,
            )
            partial_ids = self.tokenizer.encode(partial_text, add_special_tokens=False)
            end_pos = len(partial_ids)
            if msg["role"] == "assistant":
                for j in range(current_pos, min(end_pos, len(labels))):
                    labels[j] = input_ids[j]
            current_pos = end_pos

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": torch.zeros(0, 62, self.window_size),
            "num_spells": 0,
        }

    def _make_error(self):
        """Error handling sample (undo/clear/help — no EEG)."""
        from .templates_zh import (
            TEMPLATES_CLEAR_RESPONSE,
            TEMPLATES_CLEAR_USER,
            TEMPLATES_HELP_RESPONSE,
            TEMPLATES_HELP_USER,
            TEMPLATES_UNDO_RESPONSE,
            TEMPLATES_UNDO_USER,
            make_error_messages,
        )

        error_type = random.choice(["undo", "clear", "help"])
        spelled = ""
        if error_type == "undo":
            n_chars = random.randint(1, 5)
            chars = [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n_chars)]
            spelled = "".join(chars[:-1]) if len(chars) > 1 else ""

        messages = make_error_messages(error_type, spelled=spelled)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        labels = [-100] * len(input_ids)
        current_pos = 0
        for i, msg in enumerate(messages):
            partial = messages[:i + 1]
            partial_text = self.tokenizer.apply_chat_template(
                partial, tokenize=False, add_generation_prompt=False,
            )
            partial_ids = self.tokenizer.encode(partial_text, add_special_tokens=False)
            end_pos = len(partial_ids)
            if msg["role"] == "assistant":
                for j in range(current_pos, min(end_pos, len(labels))):
                    labels[j] = input_ids[j]
            current_pos = end_pos

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eeg_windows": torch.zeros(0, 62, self.window_size),
            "num_spells": 0,
        }
