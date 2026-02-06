#!/bin/bash
# Train with Unsloth (efficient 4-bit fine-tuning)
set -e

# Fix proxy issues
export ALL_PROXY="" all_proxy=""

MODEL=${MODEL:-"unsloth/Qwen3-VL-4B-Instruct"}
EPOCHS=${EPOCHS:-10}

echo "=== BCI-Qwen Training with Unsloth ==="
echo "Model: ${MODEL}"

uv run python train_unsloth.py \
    --embedding_dir data/embeddings \
    --output_dir output \
    --model_name "${MODEL}" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --epochs "${EPOCHS}"

echo "=== Training complete! ==="
