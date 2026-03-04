"""Per-trial dataset for standalone REVE fine-tuning with eTRCA distillation.

Simple dataset: no windowing, no multi-spell, no tokenization.
Each sample is a single EEG trial with its label and optional eTRCA soft targets.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset_bci_agent import BETA_BAD_SUBJECTS
from .fbcca import SSVEP_LATENCY_S


class REVEFinetuneDataset(Dataset):
    """Per-trial EEG dataset for REVE classifier fine-tuning.

    Args:
        eeg_dir: directory containing {split}_eeg.pt and {split}_etrca_full.pt
        split: "train" or "val"
        trial_duration_pts: number of timepoints per trial (default 600 = 3s @ 200Hz)
        exclude_subjects: set of subject IDs to exclude (e.g. BETA_BAD_SUBJECTS)
        use_etrca: whether to load eTRCA full scores for distillation
        latency_skip: skip SSVEP transient response (default: True, 0.14s @ 200Hz)
    """

    def __init__(
        self,
        eeg_dir,
        split="train",
        trial_duration_pts=600,
        exclude_subjects=None,
        use_etrca=True,
        latency_skip=True,
        random_offset=False,
    ):
        eeg_dir = Path(eeg_dir)
        data = torch.load(eeg_dir / f"{split}_eeg.pt", weights_only=True)

        # Filter bad subjects
        if exclude_subjects:
            mask = torch.ones(len(data["labels"]), dtype=torch.bool)
            for sid in exclude_subjects:
                mask &= data["subject_ids"] != sid
            n_removed = int((~mask).sum())
            data = {k: v[mask] for k, v in data.items() if isinstance(v, torch.Tensor)}
            print(f"  Excluded {n_removed} trials from {len(exclude_subjects)} subjects")

        self.eeg_data = data["eeg_data"]    # (N, 62, T)
        self.labels = data["labels"]         # (N,)

        # Per-trial valid timepoints (handles zero-padded BETA S01-S19)
        if "valid_pts" in data:
            self.valid_pts = data["valid_pts"]  # (N,)
        else:
            # Backward compat: assume all timepoints are valid
            self.valid_pts = torch.full((len(self.labels),), self.eeg_data.shape[2], dtype=torch.long)

        # Skip SSVEP transient response (0.14s @ 200Hz = 28 pts)
        self.latency_pts = int(SSVEP_LATENCY_S * 200) if latency_skip else 0
        # Global trial duration capped by shortest valid trial
        min_valid = int(self.valid_pts.min()) - self.latency_pts
        self.trial_duration_pts = min(trial_duration_pts, min_valid)

        # Random offset augmentation: sample random start position per __getitem__
        # Only for training; val uses fixed position for reproducible metrics
        self.random_offset = random_offset and split == "train"
        if self.random_offset:
            # Per-trial max offset: valid_pts - latency - duration
            self.max_offsets = (self.valid_pts - self.latency_pts - self.trial_duration_pts).clamp(min=0)
            print(f"  Random offset augmentation: max_offset range [{int(self.max_offsets.min())}, {int(self.max_offsets.max())}]")

        # Load eTRCA full scores for knowledge distillation
        self.etrca_scores = None
        if use_etrca:
            etrca_path = eeg_dir / f"{split}_etrca_full.pt"
            if etrca_path.exists():
                etrca_data = torch.load(etrca_path, weights_only=True)
                scores = etrca_data["full_scores"]  # (N_original, 40)
                if exclude_subjects:
                    scores = scores[mask]
                self.etrca_scores = scores
                print(f"  Loaded eTRCA full scores: {scores.shape}")
            else:
                print(f"  WARNING: {etrca_path} not found, training without distillation")

        latency_str = f", latency_skip={self.latency_pts}pts" if self.latency_pts > 0 else ""
        print(f"  {split}: {len(self)} trials, {self.trial_duration_pts}pts{latency_str} "
              f"({'with' if self.etrca_scores is not None else 'without'} eTRCA)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t0 = self.latency_pts
        if self.random_offset:
            max_off = int(self.max_offsets[idx])
            if max_off > 0:
                t0 += torch.randint(0, max_off + 1, (1,)).item()
        eeg = self.eeg_data[idx, :, t0:t0 + self.trial_duration_pts]  # (62, T)
        label = self.labels[idx].long()

        sample = {"eeg": eeg, "label": label}
        if self.etrca_scores is not None:
            sample["etrca_scores"] = self.etrca_scores[idx]
        return sample


class LOSODataset(Dataset):
    """LOSO (Leave-One-Subject-Out) dataset for within-dataset cross-validation.

    BM and BETA are trained/tested separately (no cross-dataset mixing).
    Subject ID determines which dataset to load:
      - 1-35: Benchmark only
      - 101-170: BETA only (original IDs +100 to avoid collision)

    Args:
        bm_dir: directory with Benchmark {train,val}_eeg.pt
        beta_dir: directory with BETA {train,val}_eeg.pt
        leave_out_subject: subject ID to hold out (1-35 for BM, 101-170 for BETA)
        is_train: True = training set (same dataset, other subjects), False = test set
        trial_duration_pts: timepoints per trial
        exclude_subjects: set of remapped subject IDs to exclude (e.g. {111,141,155,159,164})
        latency_skip: skip SSVEP transient (0.14s)
        random_offset: random start offset augmentation (train only)
    """

    # BETA S01-S15 have 500 valid pts (zero-padded to 600), rest have 600
    BETA_SHORT_SUBJECTS = set(range(1, 16))  # original IDs before remapping

    def __init__(
        self,
        bm_dir,
        beta_dir,
        leave_out_subject,
        is_train=True,
        trial_duration_pts=600,
        exclude_subjects=None,
        latency_skip=True,
        random_offset=False,
    ):
        bm_dir, beta_dir = Path(bm_dir), Path(beta_dir)
        all_eeg, all_labels, all_sids, all_valid = [], [], [], []

        # Determine which dataset based on subject ID
        is_beta = leave_out_subject > 35

        if not is_beta:
            # Load Benchmark only (both train/val splits)
            for split in ("train", "val"):
                p = bm_dir / f"{split}_eeg.pt"
                if not p.exists():
                    continue
                d = torch.load(p, weights_only=True)
                all_eeg.append(d["eeg_data"])
                all_labels.append(d["labels"])
                all_sids.append(d["subject_ids"])  # 1-35
                all_valid.append(torch.full((len(d["labels"]),), 600, dtype=torch.long))
        else:
            # Load BETA only (both train/val splits), remap IDs +100
            for split in ("train", "val"):
                p = beta_dir / f"{split}_eeg.pt"
                if not p.exists():
                    continue
                d = torch.load(p, weights_only=True)
                all_eeg.append(d["eeg_data"])
                all_labels.append(d["labels"])
                orig_sids = d["subject_ids"]
                all_sids.append(orig_sids + 100)  # remap
                # Compute valid_pts per trial
                vp = torch.full((len(d["labels"]),), 600, dtype=torch.long)
                for s in self.BETA_SHORT_SUBJECTS:
                    vp[orig_sids == s] = 500
                all_valid.append(vp)

        eeg_data = torch.cat(all_eeg)
        labels = torch.cat(all_labels)
        subject_ids = torch.cat(all_sids)
        valid_pts = torch.cat(all_valid)

        # Exclude bad subjects
        if exclude_subjects:
            mask = torch.ones(len(labels), dtype=torch.bool)
            for sid in exclude_subjects:
                mask &= subject_ids != sid
            n_removed = int((~mask).sum())
            eeg_data = eeg_data[mask]
            labels = labels[mask]
            subject_ids = subject_ids[mask]
            valid_pts = valid_pts[mask]
            if n_removed > 0:
                print(f"  Excluded {n_removed} trials from bad subjects")

        # Split: leave_out vs rest
        is_held = subject_ids == leave_out_subject
        if is_train:
            sel = ~is_held
        else:
            sel = is_held

        self.eeg_data = eeg_data[sel]
        self.labels = labels[sel]
        self.subject_ids = subject_ids[sel]
        self.valid_pts = valid_pts[sel]

        if len(self.labels) == 0:
            raise ValueError(f"No trials for subject {leave_out_subject} (is_train={is_train})")

        # Latency skip
        self.latency_pts = int(SSVEP_LATENCY_S * 200) if latency_skip else 0
        min_valid = int(self.valid_pts.min()) - self.latency_pts
        self.trial_duration_pts = min(trial_duration_pts, min_valid)

        # Random offset
        self.random_offset = random_offset and is_train
        if self.random_offset:
            self.max_offsets = (self.valid_pts - self.latency_pts - self.trial_duration_pts).clamp(min=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t0 = self.latency_pts
        if self.random_offset:
            max_off = int(self.max_offsets[idx])
            if max_off > 0:
                t0 += torch.randint(0, max_off + 1, (1,)).item()
        eeg = self.eeg_data[idx, :, t0:t0 + self.trial_duration_pts]
        return {
            "eeg": eeg,
            "label": self.labels[idx].long(),
            "subject_id": self.subject_ids[idx].long(),
        }

    @staticmethod
    def get_all_subjects(dataset_filter="all", exclude_subjects=None):
        """Return sorted list of all valid subject IDs."""
        subjects = []
        if dataset_filter in ("all", "benchmark"):
            subjects.extend(range(1, 36))  # BM 1-35
        if dataset_filter in ("all", "beta"):
            subjects.extend(range(101, 171))  # BETA 101-170
        if exclude_subjects:
            subjects = [s for s in subjects if s not in exclude_subjects]
        return sorted(subjects)


def reve_finetune_collate_fn(batch):
    """Collate function for REVEFinetuneDataset.

    Returns:
        eeg: (B, 62, T) float32
        labels: (B,) int64
        etrca_scores: (B, 40) float32 or None
    """
    eeg = torch.stack([s["eeg"] for s in batch])
    labels = torch.stack([s["label"] for s in batch])

    result = {"eeg": eeg, "labels": labels}
    if "etrca_scores" in batch[0]:
        result["etrca_scores"] = torch.stack([s["etrca_scores"] for s in batch])
    return result
