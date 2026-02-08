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
# GPU selection:
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_ablation.sh s1 reve_fbcca
#   NUM_GPUS=4 bash scripts/train_ablation.sh all

set -e

# --- GPU detection ---
NUM_GPUS=${NUM_GPUS:-}
if [ -z "$NUM_GPUS" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    else
        NUM_GPUS=6
    fi
fi

DEEPSPEED_ARGS=()
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEEPSPEED_ARGS=(--num_gpus "$NUM_GPUS")
fi

# --- Batch size / accumulation ---
# Target effective global batch = 32 (balanced for ~2100 sequences)
# Formula: global_batch = micro_bs × grad_accum × num_gpus
TARGET_BATCH=${TARGET_BATCH:-32}

auto_accum() {
    local micro_bs=$1
    local denom=$((micro_bs * NUM_GPUS))
    local accum=$((TARGET_BATCH / denom))
    [ "$accum" -lt 1 ] && accum=1
    echo "$accum"
}

# Stage 1: micro_bs=16 (safe for most GPUs)
S1_BS=${S1_BS:-16}
S1_ACCUM=$(auto_accum $S1_BS)
S1_EPOCHS=${S1_EPOCHS:-15}

# Stage 2: micro_bs=8 (LoRA uses more memory)
S2_BS=${S2_BS:-8}
S2_ACCUM=$(auto_accum $S2_BS)
S2_EPOCHS=${S2_EPOCHS:-10}

# --- Print config ---
echo "=== Ablation Config ==="
echo "GPUs:        ${NUM_GPUS}"
echo "Target batch: ${TARGET_BATCH}"
echo "Stage 1:     bs=${S1_BS} × accum=${S1_ACCUM} × ${NUM_GPUS}gpu = $((S1_BS * S1_ACCUM * NUM_GPUS)) eff, ${S1_EPOCHS} epochs"
echo "Stage 2:     bs=${S2_BS} × accum=${S2_ACCUM} × ${NUM_GPUS}gpu = $((S2_BS * S2_ACCUM * NUM_GPUS)) eff, ${S2_EPOCHS} epochs"
echo "========================"
echo ""

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

    deepspeed "${DEEPSPEED_ARGS[@]}" main_bci_agent.py \
        --stage 1 \
        --eeg_dir data/eeg_tensors \
        --output_dir "$output_dir" \
        --model_name Qwen/Qwen3-4B-Instruct \
        --reve_dir models \
        --encoder_type "$encoder_type" \
        $fbcca_flag \
        --exclude_bad_subjects \
        --batch_size "$S1_BS" \
        --grad_accum "$S1_ACCUM" \
        --lr 5e-4 \
        --encoder_lr 1e-3 \
        --epochs "$S1_EPOCHS" \
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

    deepspeed "${DEEPSPEED_ARGS[@]}" main_bci_agent.py \
        --stage 2 \
        --eeg_dir data/eeg_tensors \
        --output_dir "$output_dir" \
        --model_name Qwen/Qwen3-4B-Instruct \
        --reve_dir models \
        --encoder_type "$encoder_type" \
        $fbcca_flag \
        --exclude_bad_subjects \
        --stage1_checkpoint "$s1_ckpt" \
        --lora_rank 32 \
        --lora_alpha 64 \
        --batch_size "$S2_BS" \
        --grad_accum "$S2_ACCUM" \
        --lr 2e-5 \
        --encoder_lr 5e-4 \
        --epochs "$S2_EPOCHS" \
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
