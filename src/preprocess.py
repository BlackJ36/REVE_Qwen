"""Download data, preprocess EEG, and extract REVE embeddings offline."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample
from tqdm import tqdm

# Standard 64-channel names for Tsinghua Benchmark/BETA (Neuroscan layout)
CHANNELS_64 = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3",
    "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5",
    "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6",
    "PO8", "CB1", "O1", "Oz", "O2", "CB2", "M1", "M2",
]

# Channels that are NOT in REVE position bank (CB1=57, CB2=61)
EXCLUDED_CHANNEL_INDICES = [57, 61]
# Valid 62-channel indices
VALID_CHANNEL_INDICES = [i for i in range(64) if i not in EXCLUDED_CHANNEL_INDICES]
VALID_CHANNEL_NAMES = [CHANNELS_64[i] for i in VALID_CHANNEL_INDICES]

REVE_SFREQ = 200  # REVE requires 200 Hz

# BETA uses different target-to-frequency mapping than Benchmark.
# Benchmark: 5×8 grid row-major, label i → freq = 8.0 + (i%8)*1.0 + (i//8)*0.2
# BETA: [8.6, 8.8, 9.0, ..., 15.8, 8.0, 8.2, 8.4] (linear from 8.6, wraps around)
# We remap BETA targets to Benchmark's canonical label ordering.
_BETA_FREQS = [8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8,
               11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0,
               13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2,
               15.4, 15.6, 15.8, 8.0, 8.2, 8.4]
_BM_FREQ_TO_LABEL = {round(8.0 + (i % 8) * 1.0 + (i // 8) * 0.2, 1): i for i in range(40)}
BETA_LABEL_REMAP = [_BM_FREQ_TO_LABEL[round(f, 1)] for f in _BETA_FREQS]


def bandpass_filter(data, sfreq, l_freq=3.0, h_freq=90.0):
    """Apply 3-90Hz bandpass + 50Hz notch filter to EEG data. data: (channels, timepoints)."""
    import warnings
    info = mne.create_info(ch_names=data.shape[0], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.filter(l_freq, h_freq, verbose=False, filter_length='auto', phase='zero-double')
        raw.notch_filter(50.0, verbose=False)
    return raw.get_data()


def resample_data(data, orig_sfreq, target_sfreq):
    """Resample from orig_sfreq to target_sfreq. data: (channels, timepoints)."""
    if orig_sfreq == target_sfreq:
        return data
    n_samples = int(data.shape[1] * target_sfreq / orig_sfreq)
    return resample(data, n_samples, axis=1)


def load_benchmark(data_dir, sfreq=250):
    """Load Tsinghua Benchmark dataset.

    Expected structure: data_dir/S{01-35}.mat
    Each .mat contains 'data' of shape (64, 1500, 40, 6).

    Returns list of (subject_id, eeg_data, label) tuples.
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
                samples.append((subj_idx, trial, target))

    print(f"Benchmark: loaded {len(samples)} trials from {data_dir}")
    return samples


def load_beta(data_dir, sfreq=250):
    """Load BETA dataset.

    Expected structure: data_dir/S{01-70}.mat
    Each .mat has nested structure: data['EEG'].
    Recording length varies: S01-S19 have 750pts (3s), S20-S70 have 1000pts (4s).
    BETA uses different target-to-frequency mapping — labels are remapped to
    Benchmark's canonical ordering via BETA_LABEL_REMAP.

    Returns list of (subject_id, eeg_data, label) tuples.
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
        eeg = data_struct["EEG"][0, 0]  # (64, T, blocks, targets)

        n_channels, n_times, n_blocks, n_targets = eeg.shape
        # Skip 0.5s visual cue (125 samples at 250Hz)
        cue_samples = int(0.5 * sfreq)
        for block in range(n_blocks):
            for target in range(n_targets):
                trial = eeg[:, cue_samples:, block, target]
                trial = trial[VALID_CHANNEL_INDICES, :]
                canonical_label = BETA_LABEL_REMAP[target]
                samples.append((subj_idx, trial, canonical_label))

    print(f"BETA: loaded {len(samples)} trials from {data_dir}")
    return samples


TARGET_LENGTH = 600  # 3 seconds at 200Hz


def preprocess_trial(trial_data, orig_sfreq=250):
    """Filter and resample a single trial. Output: (62, TARGET_LENGTH) at 200Hz.

    Input already has visual cue period removed:
      Benchmark: 1375 @ 250Hz (5.5s stimulus) -> filter -> resample -> 1100 -> first 600
      BETA: 625 @ 250Hz (2s stimulus + 0.5s rest) -> filter -> resample -> 500 -> pad to 600
    """
    filtered = bandpass_filter(trial_data, orig_sfreq)
    resampled = resample_data(filtered, orig_sfreq, REVE_SFREQ)

    # Truncate or pad to TARGET_LENGTH
    if resampled.shape[1] > TARGET_LENGTH:
        # Take the first 600 samples (cue already removed, starts at stimulus onset)
        resampled = resampled[:, :TARGET_LENGTH]
    elif resampled.shape[1] < TARGET_LENGTH:
        # BETA: 500 pts of real data, pad to 600
        pad_width = TARGET_LENGTH - resampled.shape[1]
        resampled = np.pad(resampled, ((0, 0), (0, pad_width)), mode='constant')

    return resampled


def load_reve_model(device="cuda"):
    """Load REVE model and position bank from HuggingFace.

    Note: trust_remote_code=True is required by the REVE model architecture
    which uses custom code on HuggingFace Hub (brain-bzh organization).
    """
    from transformers import AutoModel

    print("Loading REVE position bank...")
    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions", trust_remote_code=True
    )

    print("Loading REVE-base model...")
    model = AutoModel.from_pretrained(
        "brain-bzh/reve-base", trust_remote_code=True
    )
    model = model.to(device)
    model.requires_grad_(False)

    return model, pos_bank


@torch.no_grad()
def extract_reve_embeddings(
    model, pos_bank, trials, channel_names=None, device="cuda", batch_size=64
):
    """Extract REVE embeddings for all trials.

    Args:
        model: REVE model
        pos_bank: REVE position bank for electrode coordinates
        trials: list of (subject_id, eeg_data, label) tuples
        channel_names: list of electrode names
        device: torch device
        batch_size: extraction batch size

    Returns:
        embeddings: (N, reve_dim) tensor
        labels: (N,) tensor
    """
    if channel_names is None:
        channel_names = VALID_CHANNEL_NAMES  # Use 62 valid channels (excluding CB1, CB2)

    # Get electrode positions
    positions = pos_bank(channel_names)  # (n_channels, 3)
    print(f"Electrode positions shape: {positions.shape}")

    all_embeddings = []
    all_labels = []

    for i in tqdm(range(0, len(trials), batch_size), desc="Extracting REVE embeddings"):
        batch = trials[i : i + batch_size]
        eeg_batch = []
        labels_batch = []

        for subj_id, trial_data, label in batch:
            processed = preprocess_trial(trial_data)
            eeg_batch.append(processed)
            labels_batch.append(label)

        # Stack into tensor: (B, channels, timepoints)
        eeg_tensor = torch.tensor(np.stack(eeg_batch), dtype=torch.float32).to(device)
        pos_tensor = positions.unsqueeze(0).expand(len(batch), -1, -1).to(device)

        # Extract REVE features: output shape is (B, channels, patches, embed_dim)
        output = model(eeg_tensor, pos_tensor)

        # Apply REVE's attention pooling to get (B, embed_dim)
        emb = model.attention_pooling(output)

        all_embeddings.append(emb.cpu())
        all_labels.extend(labels_batch)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)

    print(f"Extracted embeddings: {embeddings.shape}, labels: {labels.shape}")
    return embeddings, labels


def split_by_subject(samples, val_subjects):
    """Split samples by subject ID."""
    train = [s for s in samples if s[0] not in val_subjects]
    val = [s for s in samples if s[0] in val_subjects]
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEG and extract REVE embeddings")
    parser.add_argument("--benchmark_dir", type=str, default="data/benchmark_raw")
    parser.add_argument("--beta_dir", type=str, default="data/beta_raw")
    parser.add_argument("--output_dir", type=str, default="data/embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
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

    # Load REVE model
    reve_model, pos_bank = load_reve_model(args.device)

    # Load datasets
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
        print("Download from: http://bci.med.tsinghua.edu.cn/download.html")
        return

    # Extract embeddings
    print(f"\nExtracting train embeddings ({len(all_train_samples)} samples)...")
    train_emb, train_labels = extract_reve_embeddings(
        reve_model, pos_bank, all_train_samples,
        device=args.device, batch_size=args.batch_size,
    )

    print(f"\nExtracting val embeddings ({len(all_val_samples)} samples)...")
    val_emb, val_labels = extract_reve_embeddings(
        reve_model, pos_bank, all_val_samples,
        device=args.device, batch_size=args.batch_size,
    )

    # Save embeddings
    torch.save(
        {"embeddings": train_emb, "labels": train_labels},
        output_dir / "train_embeddings.pt",
    )
    torch.save(
        {"embeddings": val_emb, "labels": val_labels},
        output_dir / "val_embeddings.pt",
    )

    # Save metadata
    meta = {
        "reve_dim": int(train_emb.shape[1]),
        "n_targets": 40,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  train: {train_emb.shape}, val: {val_emb.shape}")
    print(f"  REVE embedding dim: {train_emb.shape[1]}")


if __name__ == "__main__":
    main()
