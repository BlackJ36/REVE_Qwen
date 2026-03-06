#!/bin/bash
# Stage 2: Instruction tuning with SSVEP-tuned REVE (9ch) + candidate mode
# Loads encoder from Stage 1, applies LoRA to Qwen3-4B
# 6x 48GB GPUs with DeepSpeed ZeRO-2

set -e

NUM_GPUS=6
S1_CHECKPOINT="output_bci_merged_s1/best"

deepspeed --num_gpus $NUM_GPUS main_bci_agent.py \
    --stage 2 \
    --eeg_dir data/eeg_tensors \
    --output_dir output_bci_merged_s2 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --reve_dir models \
    --reve_merged_ckpt checkpoints/reve_ssvep_lora16_1s.pt \
    --fbcca_mode candidate \
    --decoder_type fbcca \
    --stage1_checkpoint $S1_CHECKPOINT \
    --lora_rank 32 \
    --lora_alpha 64 \
    --window_size 200 \
    --trial_duration 1.0 \
    --batch_size 32 \
    --grad_accum 4 \
    --lr 2e-5 \
    --encoder_lr 5e-4 \
    --epochs 5 \
    --warmup_ratio 0.1 \
    --min_spells 3 \
    --max_spells 8 \
    --deepspeed configs/ds_zero2.json
