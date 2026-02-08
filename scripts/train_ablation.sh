#!/bin/bash
# Ablation study: 4 encoder configurations × 2 stages
#
# Usage:
#   bash scripts/train_ablation.sh s1 [experiment]   # Stage 1 only
#   bash scripts/train_ablation.sh s2 [experiment]   # Stage 2 only (needs S1 checkpoint)
#   bash scripts/train_ablation.sh all [experiment]   # Stage 1 → Stage 2
#
# experiment: reve_fbcca | reve_only | labram_fbcca | labram_only | all (default)
#
# Estimated time (6x 48GB GPUs):
#   Stage 1: ~40-60 min total (4 experiments)
#   Stage 2: ~40-60 min total (4 experiments)
#   Full:    ~1.5-2 hours

set -e

NUM_GPUS=6
STAGE=${1:-all}
EXPERIMENT=${2:-all}

CONFIGS=(
    "reve_fbcca:reve:--use_fbcca"
    "reve_only:reve:--no_fbcca"
    "labram_fbcca:labram:--use_fbcca"
    "labram_only:labram:--no_fbcca"
)

run_stage1() {
    local name=$1 encoder_type=$2 fbcca_flag=$3
    local output_dir="output_ablation_${name}_s1"

    echo "============================================================"
    echo "Stage 1: ${name} (encoder=${encoder_type}, ${fbcca_flag})"
    echo "Output: ${output_dir}"
    echo "============================================================"

    deepspeed --num_gpus $NUM_GPUS main_bci_agent.py \
        --stage 1 \
        --eeg_dir data/eeg_tensors \
        --output_dir "$output_dir" \
        --model_name Qwen/Qwen3-4B-Instruct \
        --reve_dir models \
        --encoder_type "$encoder_type" \
        $fbcca_flag \
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
}

run_stage2() {
    local name=$1 encoder_type=$2 fbcca_flag=$3
    local s1_dir="output_ablation_${name}_s1"
    local s1_ckpt="${s1_dir}/best"
    local output_dir="output_ablation_${name}_s2"

    if [ ! -d "$s1_ckpt" ]; then
        echo "ERROR: Stage 1 checkpoint not found: ${s1_ckpt}"
        echo "  Run Stage 1 first: bash $0 s1 ${name}"
        return 1
    fi

    echo "============================================================"
    echo "Stage 2: ${name} (encoder=${encoder_type}, ${fbcca_flag})"
    echo "S1 checkpoint: ${s1_ckpt}"
    echo "Output: ${output_dir}"
    echo "============================================================"

    deepspeed --num_gpus $NUM_GPUS main_bci_agent.py \
        --stage 2 \
        --eeg_dir data/eeg_tensors \
        --output_dir "$output_dir" \
        --model_name Qwen/Qwen3-4B-Instruct \
        --reve_dir models \
        --encoder_type "$encoder_type" \
        $fbcca_flag \
        --stage1_checkpoint "$s1_ckpt" \
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
}

run_for_experiment() {
    local stage_fn=$1 experiment=$2

    for cfg in "${CONFIGS[@]}"; do
        IFS=: read -r name encoder_type fbcca_flag <<< "$cfg"
        if [ "$experiment" = "all" ] || [ "$experiment" = "$name" ]; then
            $stage_fn "$name" "$encoder_type" "$fbcca_flag"
        fi
    done
}

case "$STAGE" in
    s1)
        run_for_experiment run_stage1 "$EXPERIMENT"
        ;;
    s2)
        run_for_experiment run_stage2 "$EXPERIMENT"
        ;;
    all)
        echo "=== Running Stage 1 for all experiments ==="
        run_for_experiment run_stage1 "$EXPERIMENT"
        echo ""
        echo "=== Running Stage 2 for all experiments ==="
        run_for_experiment run_stage2 "$EXPERIMENT"
        ;;
    *)
        echo "Usage: $0 <s1|s2|all> [reve_fbcca|reve_only|labram_fbcca|labram_only|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. Results in output_ablation_*/"
