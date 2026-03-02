#!/bin/bash
# Stage 2: Instruction tuning - LoRA + mixed data
# Loads encoder from Stage 1, applies LoRA to Qwen3-4B
# 6x 48GB GPUs with DeepSpeed ZeRO-2

set -e

NUM_GPUS=6
S1_CHECKPOINT="output_bci_agent_s1/best"

# Optional: path to pure NL JSONL data for Type C
# NL_DATA="data/nl_instructions_zh.jsonl"

deepspeed --num_gpus $NUM_GPUS main_bci_agent.py \
    --stage 2 \
    --eeg_dir data/eeg_tensors \
    --output_dir output_bci_agent_s2 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --reve_dir models \
    --stage1_checkpoint $S1_CHECKPOINT \
    --lora_rank 32 \
    --lora_alpha 64 \
    --batch_size 32 \
    --grad_accum 4 \
    --lr 2e-5 \
    --encoder_lr 5e-4 \
    --epochs 5 \
    --warmup_ratio 0.1 \
    --min_spells 3 \
    --max_spells 8 \
    --window_size 300 \
    --window_step 100 \
    --deepspeed configs/ds_zero2.json
    # --nl_data_path $NL_DATA  # Uncomment when NL data is ready
