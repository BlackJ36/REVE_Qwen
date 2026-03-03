#!/bin/bash
# Ablation study: 5 encoder configurations × 2 stages
#
# Usage:
#   bash scripts/train_ablation.sh s1 [experiment] [duration]   # Stage 1 only
#   bash scripts/train_ablation.sh s2 [experiment] [duration]   # Stage 2 only (needs S1 checkpoint)
#   bash scripts/train_ablation.sh all [experiment] [duration]   # Stage 1 → Stage 2
#
# experiment: reve_fbcca | reve_candidate | reve_etrca | reve_only | labram_fbcca | labram_only | reve_ft_etrca | all (default)
# duration: trial duration in seconds (default: 3.0). E.g. 1.0, 1.5, 2.0, 3.0
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
        NUM_GPUS=5
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
S1_EPOCHS=${S1_EPOCHS:-30}
S1_LR=${S1_LR:-5e-4}
S1_ENC_LR=${S1_ENC_LR:-1e-3}

# Stage 2: micro_bs=4 (LoRA + long sequences need more memory)
S2_BS=${S2_BS:-4}
S2_ACCUM=$(auto_accum $S2_BS)
S2_EPOCHS=${S2_EPOCHS:-20}
S2_LR=${S2_LR:-2e-5}
S2_ENC_LR=${S2_ENC_LR:-5e-4}
S2_LORA_RANK=${S2_LORA_RANK:-32}
S2_LORA_ALPHA=${S2_LORA_ALPHA:-64}

# Early stopping
PATIENCE=${PATIENCE:-5}

# --- Print config ---
echo "=== Ablation Config ==="
echo "GPUs:        ${NUM_GPUS}"
echo "Target batch: ${TARGET_BATCH}"
echo "Stage 1:     bs=${S1_BS} × accum=${S1_ACCUM} × ${NUM_GPUS}gpu = $((S1_BS * S1_ACCUM * NUM_GPUS)) eff, ${S1_EPOCHS} epochs, lr=${S1_LR}, enc_lr=${S1_ENC_LR}"
echo "Stage 2:     bs=${S2_BS} × accum=${S2_ACCUM} × ${NUM_GPUS}gpu = $((S2_BS * S2_ACCUM * NUM_GPUS)) eff, ${S2_EPOCHS} epochs, lr=${S2_LR}, enc_lr=${S2_ENC_LR}, rank=${S2_LORA_RANK}"
echo "Patience:    ${PATIENCE}"
echo "========================"
echo ""

STAGE=${1:-all}
EXPERIMENT=${2:-all}
DURATION=${3:-3.0}

echo "Duration:    ${DURATION}s"

CONFIGS=(
    "reve_fbcca:reve:--fbcca_mode film"
    "reve_candidate:reve:--fbcca_mode candidate --decoder_type fbcca --s1_lora_rank 16"
    "reve_etrca:reve:--fbcca_mode candidate --decoder_type etrca --s1_lora_rank 16"
    "reve_only:reve:--fbcca_mode none"
    "labram_fbcca:labram:--fbcca_mode film"
    "labram_only:labram:--fbcca_mode none"
    "reve_ft_etrca:reve:--fbcca_mode candidate --decoder_type etrca --s1_lora_rank 16 --reve_finetune_dir output_reve_finetune"
)

run_stage1() {
    local name=$1 encoder_type=$2 fbcca_flag=$3
    local output_dir="output_ablation_${name}_s1"

    # Candidate mode requires precomputed decoder predictions
    if [[ "$fbcca_flag" == *"candidate"* ]]; then
        if [[ "$fbcca_flag" == *"etrca"* ]]; then
            local needed_file="data/eeg_tensors/train_etrca.pt"
            local gen_cmd="python scripts/precompute_trca.py --eeg_dir data/eeg_tensors --ensemble"
        else
            local needed_file="data/eeg_tensors/train_fbcca.pt"
            local gen_cmd="python scripts/precompute_fbcca.py --eeg_dir data/eeg_tensors"
        fi
        if [ ! -f "$needed_file" ]; then
            echo "ERROR: Precomputed decoder data not found: ${needed_file}"
            echo "  Run: ${gen_cmd}"
            return 1
        fi
    fi

    # Fine-tuned REVE requires finetune_reve.py checkpoint
    if [[ "$fbcca_flag" == *"reve_finetune_dir"* ]]; then
        local ft_dir
        ft_dir=$(echo "$fbcca_flag" | grep -oP '(?<=--reve_finetune_dir )\S+')
        if [ -n "$ft_dir" ] && [ ! -d "$ft_dir/reve_lora" ]; then
            echo "ERROR: Fine-tuned REVE checkpoint not found: ${ft_dir}/reve_lora"
            echo "  Run: python scripts/finetune_reve.py --phase both --output_dir ${ft_dir}"
            return 1
        fi
    fi

    echo "============================================================"
    echo "Stage 1: ${name} (encoder=${encoder_type}, ${fbcca_flag})"
    echo "Output: ${output_dir}"
    echo "============================================================"

    deepspeed "${DEEPSPEED_ARGS[@]}" main_bci_agent.py \
        --stage 1 \
        --eeg_dir data/eeg_tensors \
        --output_dir "$output_dir" \
        --model_name Qwen/Qwen3-4B-Instruct-2507 \
        --reve_dir models \
        --encoder_type "$encoder_type" \
        $fbcca_flag \
        --exclude_bad_subjects \
        --batch_size "$S1_BS" \
        --grad_accum "$S1_ACCUM" \
        --lr "$S1_LR" \
        --encoder_lr "$S1_ENC_LR" \
        --epochs "$S1_EPOCHS" \
        --warmup_ratio 0.1 \
        --early_stopping_patience "$PATIENCE" \
        --min_spells 5 \
        --max_spells 10 \
        --window_size 300 \
        --window_step 100 \
        --trial_duration "$DURATION" \
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
        --model_name Qwen/Qwen3-4B-Instruct-2507 \
        --reve_dir models \
        --encoder_type "$encoder_type" \
        $fbcca_flag \
        --exclude_bad_subjects \
        --stage1_checkpoint "$s1_ckpt" \
        --lora_rank "$S2_LORA_RANK" \
        --lora_alpha "$S2_LORA_ALPHA" \
        --batch_size "$S2_BS" \
        --grad_accum "$S2_ACCUM" \
        --lr "$S2_LR" \
        --encoder_lr "$S2_ENC_LR" \
        --epochs "$S2_EPOCHS" \
        --warmup_ratio 0.1 \
        --early_stopping_patience "$PATIENCE" \
        --min_spells 3 \
        --max_spells 50 \
        --window_size 300 \
        --window_step 100 \
        --trial_duration "$DURATION" \
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
