"""Preprocess EEG data and save raw tensors for end-to-end training.

Unlike preprocess.py which extracts REVE embeddings offline, this saves
the preprocessed EEG tensors directly so REVE can be fine-tuned end-to-end.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .preprocess import (
    VALID_CHANNEL_NAMES,
    load_benchmark,
    load_beta,
    preprocess_trial,
    split_by_subject,
    TARGET_LENGTH,
)


def preprocess_and_save_tensors(samples, desc="Processing"):
    """Preprocess all trials and return tensors.

    Args:
        samples: list of (subject_id, eeg_data, label) tuples
        desc: progress bar description

    Returns:
        eeg_data: (N, 62, TARGET_LENGTH) float32 tensor
        labels: (N,) long tensor
    """
    all_eeg = []
    all_labels = []

    for subj_id, trial_data, label in tqdm(samples, desc=desc):
        processed = preprocess_trial(trial_data)  # (62, 600)
        all_eeg.append(processed)
        all_labels.append(label)

    eeg_data = torch.tensor(np.stack(all_eeg), dtype=torch.float32)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return eeg_data, labels


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
        benchmark_samples = load_benchmark(benchmark_dir)
        val_subj = {int(x) for x in args.benchmark_val_subjects.split(",")}
        train_b, val_b = split_by_subject(benchmark_samples, val_subj)
        all_train_samples.extend(train_b)
        all_val_samples.extend(val_b)
        print(f"Benchmark split: {len(train_b)} train, {len(val_b)} val")
    else:
        print(f"Benchmark dir {benchmark_dir} not found, skipping")

    beta_dir = Path(args.beta_dir)
    if beta_dir.exists():
        beta_samples = load_beta(beta_dir)
        val_subj = {int(x) for x in args.beta_val_subjects.split(",")}
        train_bt, val_bt = split_by_subject(beta_samples, val_subj)
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
    train_eeg, train_labels = preprocess_and_save_tensors(all_train_samples, "Train")

    print(f"Processing val samples ({len(all_val_samples)})...")
    val_eeg, val_labels = preprocess_and_save_tensors(all_val_samples, "Val")

    torch.save(
        {"eeg_data": train_eeg, "labels": train_labels, "channel_names": VALID_CHANNEL_NAMES},
        output_dir / "train_eeg.pt",
    )
    torch.save(
        {"eeg_data": val_eeg, "labels": val_labels, "channel_names": VALID_CHANNEL_NAMES},
        output_dir / "val_eeg.pt",
    )

    meta = {
        "n_channels": train_eeg.shape[1],
        "n_timepoints": train_eeg.shape[2],
        "n_targets": 40,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "channel_names": VALID_CHANNEL_NAMES,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  train_eeg.pt: {train_eeg.shape} ({train_eeg.element_size() * train_eeg.nelement() / 1e6:.1f} MB)")
    print(f"  val_eeg.pt: {val_eeg.shape} ({val_eeg.element_size() * val_eeg.nelement() / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
