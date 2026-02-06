#!/bin/bash
# Launch end-to-end training with DeepSpeed ZeRO-2
set -e

# Fix httpx socks proxy incompatibility
export ALL_PROXY="" all_proxy=""
export MODELSCOPE_OFFLINE=1
export HF_HUB_OFFLINE=1

# Performance optimizations
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

NUM_GPUS=${NUM_GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-128}
GRAD_ACCUM=${GRAD_ACCUM:-1}
EPOCHS=${EPOCHS:-30}
UNFREEZE=${UNFREEZE:-4}

echo "=== BCI-Qwen E2E Training (${NUM_GPUS} GPUs) ==="
echo "Batch size per GPU: ${BATCH_SIZE}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch size: $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"
echo "REVE unfreeze last: ${UNFREEZE} layers"

uv run deepspeed \
    --include localhost:2,3,4,5 \
    main_e2e.py \
    --eeg_dir data/eeg_tensors \
    --output_dir output_e2e \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --from_modelscope \
    --unfreeze_last_n ${UNFREEZE} \
    --reve_lr 3e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr 5e-4 \
    --projector_lr 3e-3 \
    --epochs ${EPOCHS} \
    --early_stopping_patience 5 \
    --warmup_ratio 0.1 \
    --checkpoint_mode weights_only \
    --deepspeed configs/ds_zero2.json

echo "=== E2E Training complete! Model saved to output_e2e/final/ ==="
