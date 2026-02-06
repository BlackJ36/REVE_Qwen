#!/bin/bash
# Preprocess EEG data into raw tensors for end-to-end training
set -e

export ALL_PROXY="" all_proxy=""

echo "=== Preprocessing EEG → raw tensors for E2E training ==="

uv run python -m src.preprocess_e2e \
    --benchmark_dir data/benchmark_raw \
    --beta_dir data/beta_raw \
    --output_dir data/eeg_tensors

echo "=== Done! Tensors saved to data/eeg_tensors/ ==="
