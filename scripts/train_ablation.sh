#!/bin/bash
# Ablation study: 4 encoder configurations
# Run each experiment sequentially with Stage 1 only (alignment)
# Usage: bash scripts/train_ablation.sh [experiment_name]
#   experiment_name: reve_fbcca | reve_only | labram_fbcca | labram_only | all

set -e

NUM_GPUS=6
EXPERIMENT=${1:-all}

run_experiment() {
    local name=$1
    local encoder_type=$2
    local fbcca_flag=$3
    local output_dir="output_ablation_${name}"

    echo "============================================================"
    echo "Experiment: ${name} (encoder=${encoder_type}, fbcca=${fbcca_flag})"
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

case "$EXPERIMENT" in
    reve_fbcca)
        run_experiment "reve_fbcca" "reve" "--use_fbcca"
        ;;
    reve_only)
        run_experiment "reve_only" "reve" "--no_fbcca"
        ;;
    labram_fbcca)
        run_experiment "labram_fbcca" "labram" "--use_fbcca"
        ;;
    labram_only)
        run_experiment "labram_only" "labram" "--no_fbcca"
        ;;
    all)
        run_experiment "reve_fbcca" "reve" "--use_fbcca"
        run_experiment "reve_only" "reve" "--no_fbcca"
        run_experiment "labram_fbcca" "labram" "--use_fbcca"
        run_experiment "labram_only" "labram" "--no_fbcca"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [reve_fbcca|reve_only|labram_fbcca|labram_only|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. Results in output_ablation_*/"
