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

        # Skip SSVEP transient response (0.14s @ 200Hz = 28 pts)
        self.latency_pts = int(SSVEP_LATENCY_S * 200) if latency_skip else 0
        available = self.eeg_data.shape[2] - self.latency_pts
        self.trial_duration_pts = min(trial_duration_pts, available)

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
        eeg = self.eeg_data[idx, :, t0:t0 + self.trial_duration_pts]  # (62, T)
        label = self.labels[idx].long()

        sample = {"eeg": eeg, "label": label}
        if self.etrca_scores is not None:
            sample["etrca_scores"] = self.etrca_scores[idx]
        return sample


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
