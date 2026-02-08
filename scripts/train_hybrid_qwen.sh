#!/bin/bash
# Plan B: HybridEncoder (REVE+FBCCA) + Qwen3-0.6B full fine-tune
# ~600M trainable params
# 4x 48GB GPUs with DeepSpeed ZeRO-2

set -e

NUM_GPUS=4
BATCH_SIZE=128
GRAD_ACCUM=1
# Effective batch size: 4 * 128 * 1 = 512
EPOCHS=30
LR=2e-5
PROJECTOR_LR=1e-3

deepspeed --num_gpus $NUM_GPUS main_hybrid_qwen.py \
    --eeg_dir data/eeg_tensors \
    --output_dir output_hybrid_qwen \
    --model_name Qwen/Qwen3-0.6B \
    --reve_dir models \
    --num_eeg_tokens 62 \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LR \
    --projector_lr $PROJECTOR_LR \
    --epochs $EPOCHS \
    --warmup_ratio 0.1 \
    --min_spells 5 \
    --max_spells 10 \
    --window_size 300 \
    --window_step 100 \
    --deepspeed configs/ds_zero2.json
