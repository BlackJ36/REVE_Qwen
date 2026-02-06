#!/bin/bash
# Single GPU training with 4-bit quantization (for 16GB VRAM)
set -e

# Fix httpx socks proxy incompatibility
export ALL_PROXY="" all_proxy=""

echo "=== BCI-Qwen Training (Single GPU, 4-bit) ==="

uv run python main.py \
    --embedding_dir data/embeddings \
    --output_dir output \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --from_modelscope \
    --use_4bit \
    --lora_rank 32 \
    --lora_alpha 64 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 2e-4 \
    --epochs 10

echo "=== Training complete! Model saved to output/final/ ==="
