#!/bin/bash
# Stage 1: Alignment with SSVEP-tuned REVE (9ch, 1s) + candidate mode
# REVE backbone frozen (merged LoRA), trains projector + embeddings
# 5x RTX 4090 (48GB) with DeepSpeed ZeRO-2

set -e

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
NUM_GPUS=5

deepspeed --num_gpus $NUM_GPUS main_bci_agent.py \
    --stage 1 \
    --eeg_dir data/eeg_tensors \
    --output_dir output_bci_merged_s1 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --reve_dir models \
    --reve_merged_ckpt checkpoints/reve_ssvep_lora16_1s.pt \
    --fbcca_mode candidate \
    --decoder_type fbcca \
    --window_size 200 \
    --trial_duration 1.0 \
    --batch_size 64 \
    --grad_accum 2 \
    --lr 5e-4 \
    --encoder_lr 1e-3 \
    --epochs 10 \
    --warmup_ratio 0.1 \
    --min_spells 5 \
    --max_spells 10 \
    --deepspeed configs/ds_zero2.json
