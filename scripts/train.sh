#!/bin/bash
# Launch 6-GPU training with DeepSpeed ZeRO-2
set -e

# Fix httpx socks proxy incompatibility
export ALL_PROXY="" all_proxy=""

NUM_GPUS=${NUM_GPUS:-6}

echo "=== BCI-Qwen Training (${NUM_GPUS} GPUs) ==="

uv run deepspeed \
    --num_gpus=${NUM_GPUS} \
    main.py \
    --embedding_dir data/embeddings \
    --output_dir output \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --from_modelscope \
    --lora_rank 64 \
    --lora_alpha 128 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr 2e-4 \
    --epochs 10 \
    --deepspeed configs/ds_zero2.json

echo "=== Training complete! Model saved to output/final/ ==="
