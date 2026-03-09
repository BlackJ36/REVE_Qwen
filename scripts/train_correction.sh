#!/bin/bash
# Train text-only BCI spelling correction model
#
# Prerequisites:
#   Generate data: uv run python scripts/generate_correction_data.py
#
# Usage:
#   bash scripts/train_correction.sh              # multi-GPU (3,4,5,6,7)
#   bash scripts/train_correction.sh --epochs 20   # override args

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6,7}"
N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

echo "=== BCI Spelling Correction (Text-Only) ==="
echo "Using $N_GPU GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Generate data if not exists
if [ ! -f data/correction/train.jsonl ]; then
    echo "Generating correction data..."
    uv run python scripts/generate_correction_data.py
fi

# Train with DDP
# per_device=4 × 5 GPUs × grad_accum=2 = effective batch 40
uv run torchrun --nproc_per_node=$N_GPU scripts/train_correction.py \
    --data_dir data/correction \
    --output_dir output/correction \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --from_modelscope \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --grad_accum 2 \
    --max_length 1280 \
    --lr 2e-4 \
    --epochs 10 \
    --early_stopping 3 \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Checkpoint saved to output/correction/final"
echo ""
echo "To evaluate:"
echo "  CUDA_VISIBLE_DEVICES=3 uv run python scripts/eval_correction.py \\"
echo "      --checkpoint output/correction/final --run_base"
