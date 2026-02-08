#!/bin/bash
# Stage 1: Alignment - FiLM encoder + projector + embeddings
# Qwen3-4B transformer layers frozen
# 6x 48GB GPUs with DeepSpeed ZeRO-2

set -e

NUM_GPUS=6

deepspeed --num_gpus $NUM_GPUS main_bci_agent.py \
    --stage 1 \
    --eeg_dir data/eeg_tensors \
    --output_dir output_bci_agent_s1 \
    --model_name Qwen/Qwen3-4B-Instruct \
    --reve_dir models \
    --batch_size 64 \
    --grad_accum 2 \
    --lr 5e-4 \
    --encoder_lr 1e-3 \
    --epochs 10 \
    --warmup_ratio 0.1 \
    --min_spells 5 \
    --max_spells 10 \
    --window_size 300 \
    --window_step 100 \
    --deepspeed configs/ds_zero2.json
