#!/bin/bash
# Diagnostic: single-GPU training without DeepSpeed
# Tests if model + training loop works before scaling to multi-GPU
set -e

export CUDA_VISIBLE_DEVICES=3

echo "=== Environment ==="
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.get_device_name(0)}')
free, total = torch.cuda.mem_get_info(0)
print(f'GPU mem: {free/1e9:.1f} / {total/1e9:.1f} GB free')
import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
"
echo ""

echo "=== Single-GPU training (no DeepSpeed) ==="
python main_bci_agent.py \
    --stage 1 \
    --eeg_dir data/eeg_tensors \
    --output_dir output_bci_test \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --reve_dir models \
    --reve_merged_ckpt checkpoints/reve_ssvep_lora16_1s.pt \
    --fbcca_mode candidate \
    --decoder_type fbcca \
    --window_size 200 \
    --trial_duration 1.0 \
    --batch_size 16 \
    --grad_accum 8 \
    --lr 5e-4 \
    --encoder_lr 1e-3 \
    --epochs 1 \
    --warmup_ratio 0.1 \
    --min_spells 5 \
    --max_spells 10

echo ""
echo "=== Done. Check GPU peak memory above ==="
