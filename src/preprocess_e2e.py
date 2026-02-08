"""Preprocess EEG data and save raw tensors for end-to-end training.

Unlike preprocess.py which extracts REVE embeddings offline, this saves
the preprocessed EEG tensors directly so REVE can be fine-tuned end-to-end.

Saves subject_ids and block_ids alongside EEG data for multi-spell training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

from .preprocess import (
    VALID_CHANNEL_INDICES,
    VALID_CHANNEL_NAMES,
    preprocess_trial,
    split_by_subject,
    TARGET_LENGTH,
)


def load_benchmark_with_blocks(data_dir, sfreq=250):
    """Load Tsinghua Benchmark dataset with block metadata.

    Returns list of (subject_id, block_id, eeg_data, label) tuples.
    """
    data_dir = Path(data_dir)
    samples = []

    for subj_idx in range(1, 36):
        mat_file = data_dir / f"S{subj_idx:02d}.mat"
        if not mat_file.exists():
            print(f"  Warning: {mat_file} not found, skipping")
            continue

        mat = loadmat(str(mat_file))
        data = mat["data"]  # (64, 1500, 40, 6)

        n_channels, n_times, n_targets, n_blocks = data.shape
        # Skip 0.5s visual cue (125 samples at 250Hz)
        cue_samples = int(0.5 * sfreq)
        for block in range(n_blocks):
            for target in range(n_targets):
                trial = data[:, cue_samples:, target, block]  # (64, 1375)
                trial = trial[VALID_CHANNEL_INDICES, :]  # (62, 1375)
                samples.append((subj_idx, block, trial, target))

    print(f"Benchmark: loaded {len(samples)} trials from {data_dir}")
    return samples


def load_beta_with_blocks(data_dir, sfreq=250):
    """Load BETA dataset with block metadata.

    Returns list of (subject_id, block_id, eeg_data, label) tuples.
    """
    data_dir = Path(data_dir)
    samples = []

    for subj_idx in range(1, 71):
        mat_file = data_dir / f"S{subj_idx:02d}.mat"
        if not mat_file.exists():
            print(f"  Warning: {mat_file} not found, skipping")
            continue

        mat = loadmat(str(mat_file))
        data_struct = mat["data"]
        eeg = data_struct["EEG"][0, 0]  # (64, 750, blocks, targets)

        n_channels, n_times, n_blocks, n_targets = eeg.shape
        # Skip 0.5s visual cue (125 samples at 250Hz)
        cue_samples = int(0.5 * sfreq)
        for block in range(n_blocks):
            for target in range(n_targets):
                trial = eeg[:, cue_samples:, block, target]  # (64, 625)
                trial = trial[VALID_CHANNEL_INDICES, :]  # (62, 625)
                samples.append((subj_idx, block, trial, target))

    print(f"BETA: loaded {len(samples)} trials from {data_dir}")
    return samples


def split_by_subject_4tuple(samples, val_subjects):
    """Split 4-tuple samples (subject_id, block_id, eeg, label) by subject ID."""
    train = [s for s in samples if s[0] not in val_subjects]
    val = [s for s in samples if s[0] in val_subjects]
    return train, val


def preprocess_and_save_tensors(samples, desc="Processing"):
    """Preprocess all trials and return tensors with metadata.

    Args:
        samples: list of (subject_id, block_id, eeg_data, label) tuples

    Returns:
        eeg_data: (N, 62, TARGET_LENGTH) float32 tensor
        labels: (N,) long tensor
        subject_ids: (N,) long tensor
        block_ids: (N,) long tensor
    """
    all_eeg = []
    all_labels = []
    all_subjects = []
    all_blocks = []

    for subj_id, block_id, trial_data, label in tqdm(samples, desc=desc):
        processed = preprocess_trial(trial_data)  # (62, 600)
        all_eeg.append(processed)
        all_labels.append(label)
        all_subjects.append(subj_id)
        all_blocks.append(block_id)

    eeg_data = torch.tensor(np.stack(all_eeg), dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.long)
    subject_ids = torch.tensor(all_subjects, dtype=torch.long)
    block_ids = torch.tensor(all_blocks, dtype=torch.long)
    return eeg_data, labels, subject_ids, block_ids


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEG and save raw tensors for E2E training")
    parser.add_argument("--benchmark_dir", type=str, default="data/benchmark_raw")
    parser.add_argument("--beta_dir", type=str, default="data/beta_raw")
    parser.add_argument("--output_dir", type=str, default="data/eeg_tensors")
    parser.add_argument(
        "--benchmark_val_subjects", type=str, default="31,32,33,34,35",
        help="Comma-separated subject IDs for validation",
    )
    parser.add_argument(
        "--beta_val_subjects", type=str, default="61,62,63,64,65,66,67,68,69,70",
        help="Comma-separated subject IDs for validation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_train_samples = []
    all_val_samples = []

    benchmark_dir = Path(args.benchmark_dir)
    if benchmark_dir.exists():
        benchmark_samples = load_benchmark_with_blocks(benchmark_dir)
        val_subj = {int(x) for x in args.benchmark_val_subjects.split(",")}
        train_b, val_b = split_by_subject_4tuple(benchmark_samples, val_subj)
        all_train_samples.extend(train_b)
        all_val_samples.extend(val_b)
        print(f"Benchmark split: {len(train_b)} train, {len(val_b)} val")
    else:
        print(f"Benchmark dir {benchmark_dir} not found, skipping")

    beta_dir = Path(args.beta_dir)
    if beta_dir.exists():
        beta_samples = load_beta_with_blocks(beta_dir)
        val_subj = {int(x) for x in args.beta_val_subjects.split(",")}
        train_bt, val_bt = split_by_subject_4tuple(beta_samples, val_subj)
        all_train_samples.extend(train_bt)
        all_val_samples.extend(val_bt)
        print(f"BETA split: {len(train_bt)} train, {len(val_bt)} val")
    else:
        print(f"BETA dir {beta_dir} not found, skipping")

    if not all_train_samples:
        print("No data found! Please place datasets in:")
        print("  data/benchmark_raw/S01.mat ... S35.mat")
        print("  data/beta_raw/S01.mat ... S70.mat")
        return

    # Preprocess and save
    print(f"\nProcessing train samples ({len(all_train_samples)})...")
    train_eeg, train_labels, train_subj, train_blocks = preprocess_and_save_tensors(
        all_train_samples, "Train",
    )

    print(f"Processing val samples ({len(all_val_samples)})...")
    val_eeg, val_labels, val_subj, val_blocks = preprocess_and_save_tensors(
        all_val_samples, "Val",
    )

    torch.save(
        {
            "eeg_data": train_eeg,
            "labels": train_labels,
            "subject_ids": train_subj,
            "block_ids": train_blocks,
            "channel_names": VALID_CHANNEL_NAMES,
        },
        output_dir / "train_eeg.pt",
    )
    torch.save(
        {
            "eeg_data": val_eeg,
            "labels": val_labels,
            "subject_ids": val_subj,
            "block_ids": val_blocks,
            "channel_names": VALID_CHANNEL_NAMES,
        },
        output_dir / "val_eeg.pt",
    )

    # Count unique groups
    train_groups = len(set(zip(train_subj.tolist(), train_blocks.tolist())))
    val_groups = len(set(zip(val_subj.tolist(), val_blocks.tolist())))

    meta = {
        "n_channels": train_eeg.shape[1],
        "n_timepoints": train_eeg.shape[2],
        "n_targets": 40,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "n_train_groups": train_groups,
        "n_val_groups": val_groups,
        "channel_names": VALID_CHANNEL_NAMES,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  train_eeg.pt: {train_eeg.shape} ({train_eeg.element_size() * train_eeg.nelement() / 1e6:.1f} MB)")
    print(f"  val_eeg.pt: {val_eeg.shape} ({val_eeg.element_size() * val_eeg.nelement() / 1e6:.1f} MB)")
    print(f"  Train groups (subject, block): {train_groups}")
    print(f"  Val groups (subject, block): {val_groups}")


if __name__ == "__main__":
    main()
