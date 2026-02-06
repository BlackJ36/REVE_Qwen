#!/bin/bash
# Extract REVE embeddings from raw EEG data
# Run this BEFORE training

set -e

# Fix httpx socks proxy incompatibility
export ALL_PROXY="" all_proxy=""

echo "=== REVE Embedding Extraction ==="
echo "Make sure raw data is placed in:"
echo "  data/benchmark_raw/S01.mat ... S35.mat"
echo "  data/beta_raw/S01.mat ... S70.mat"
echo ""

uv run python -m src.preprocess \
    --benchmark_dir data/benchmark_raw \
    --beta_dir data/beta_raw \
    --output_dir data/embeddings \
    --device cuda \
    --batch_size 64

echo "=== Done! Embeddings saved to data/embeddings/ ==="
