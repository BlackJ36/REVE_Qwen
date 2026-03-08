#!/bin/bash
# Stage 1: Train projector + new token embeddings (Qwen frozen)
# Single-trial EEG classification on pre-extracted REVE embeddings
#
# Prerequisites:
#   1. Extract embeddings: uv run python scripts/extract_embeddings.py
#   2. Ensure data/embeddings/{train,val}_embeddings.pt exist

set -e

OUTPUT_DIR="output/s1"

uv run python main.py \
    --stage 1 \
    --embedding_dir data/embeddings \
    --output_dir "$OUTPUT_DIR" \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --from_modelscope \
    --lora_rank 0 \
    --batch_size 16 \
    --grad_accum 1 \
    --lr 5e-4 \
    --projector_lr 1e-3 \
    --epochs 30 \
    --early_stopping 5 \
    --warmup_ratio 0.1

echo "S1 done. Best model saved to $OUTPUT_DIR/"
echo "Next: bash scripts/train_s2.sh"
