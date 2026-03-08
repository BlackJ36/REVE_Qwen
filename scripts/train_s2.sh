#!/bin/bash
# Stage 2: LoRA + projector on multi-char spelling + NL dialogues
#
# Prerequisites:
#   1. S1 checkpoint: output/s1/final/
#   2. S2 dialogues: data/s2_train.jsonl, data/s2_val.jsonl
#      Generate with: uv run python scripts/generate_s2_dialogues.py
#   3. Embedding bank: data/embeddings/train_embeddings.pt

set -e

S1_CKPT="${1:-output/s1/final}"
OUTPUT_DIR="output/s2"

echo "Loading S1 checkpoint from: $S1_CKPT"

uv run python main.py \
    --stage 2 \
    --stage1_checkpoint "$S1_CKPT" \
    --embedding_dir data/embeddings \
    --s2_train data/s2_train.jsonl \
    --s2_val data/s2_val.jsonl \
    --output_dir "$OUTPUT_DIR" \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --from_modelscope \
    --lora_rank 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --projector_lr 5e-4 \
    --epochs 20 \
    --early_stopping 5 \
    --warmup_ratio 0.1

echo "S2 done. Model saved to $OUTPUT_DIR/"
