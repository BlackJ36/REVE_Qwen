#!/bin/bash
# Plan A: HybridEncoder (REVE+FBCCA) + custom Transformer decoder
# ~20M trainable params, very lightweight
# 4x 48GB GPUs with DeepSpeed ZeRO-2

set -e

NUM_GPUS=4
BATCH_SIZE=64
GRAD_ACCUM=2
# Effective batch size: 4 * 64 * 2 = 512
EPOCHS=50
LR=5e-4
PROJECTOR_LR=1e-3

deepspeed --num_gpus $NUM_GPUS main_hybrid.py \
    --eeg_dir data/eeg_tensors \
    --output_dir output_hybrid \
    --reve_dir models \
    --d_model 512 \
    --nhead 8 \
    --num_layers 6 \
    --dim_feedforward 2048 \
    --dropout 0.1 \
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
