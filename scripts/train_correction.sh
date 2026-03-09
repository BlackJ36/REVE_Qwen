#!/bin/bash
# Train text-only BCI spelling correction model
#
# Prerequisites:
#   Generate data: uv run python scripts/generate_correction_data.py
#
# Usage:
#   bash scripts/train_correction.sh              # default settings
#   bash scripts/train_correction.sh --epochs 20   # override args

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

echo "=== BCI Spelling Correction (Text-Only) ==="
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Generate data if not exists
if [ ! -f data/correction/train.jsonl ]; then
    echo "Generating correction data..."
    uv run python scripts/generate_correction_data.py \
        --n_train 10000 --n_val 2000
fi

# Train
uv run python scripts/train_correction.py \
    --data_dir data/correction \
    --output_dir output/correction \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --from_modelscope \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_length 1280 \
    --lr 2e-4 \
    --epochs 10 \
    --early_stopping 3 \
    "$@"

echo ""
echo "=== Training complete. Evaluating... ==="

# Evaluate
uv run python scripts/eval_correction.py \
    --data_dir data/correction \
    --checkpoint output/correction/final \
    --split val

echo "Done!"
