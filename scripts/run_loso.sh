#!/bin/bash
# LOSO cross-validation launcher: 5 GPUs, each running multiple folds in parallel.
#
# Usage:
#   bash scripts/run_loso.sh bm        # Run BM folds (1-35)
#   bash scripts/run_loso.sh beta      # Run BETA folds (101-170)
#   bash scripts/run_loso.sh all       # Run both
#   bash scripts/run_loso.sh status    # Check progress
#   bash scripts/run_loso.sh aggregate # Aggregate results from completed folds

set -e

CHECKPOINT_DIR="/data/zjj/loso_film"
LOG_DIR="${CHECKPOINT_DIR}/logs"
GPUS=(3 4 5 6 7)
N_GPUS=${#GPUS[@]}

# Training hyperparams (bs=512, sqrt-scaled lr, 80 epochs)
COMMON_ARGS="--checkpoint_dir ${CHECKPOINT_DIR} --batch_size 512 --epochs 80 --patience 15 --lr_reve 2e-5 --lr_film 6e-4 --lr_head 6e-4 --num_workers 2"

mkdir -p "${LOG_DIR}"

launch_folds() {
    local folds=("$@")
    local n_folds=${#folds[@]}
    local per_gpu=$(( (n_folds + N_GPUS - 1) / N_GPUS ))

    echo "Launching ${n_folds} folds across ${N_GPUS} GPUs (~${per_gpu}/GPU)"

    local idx=0
    for gpu in "${GPUS[@]}"; do
        local count=0
        while [ $count -lt $per_gpu ] && [ $idx -lt $n_folds ]; do
            local fold=${folds[$idx]}
            local log="${LOG_DIR}/fold_${fold}.log"
            CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/loso_film.py \
                ${COMMON_ARGS} --start_fold $fold --end_fold $fold \
                > "$log" 2>&1 &
            count=$((count + 1))
            idx=$((idx + 1))
        done
        echo "  GPU $gpu: ${count} folds (PIDs: $(jobs -p | tail -$count | tr '\n' ' '))"
    done

    echo ""
    echo "Total: ${idx} processes launched"
    echo "Logs:  ${LOG_DIR}/fold_*.log"
    echo "Monitor: tail -f ${LOG_DIR}/fold_<N>.log"
    echo "GPU:     nvidia-smi"
    echo "Progress: bash scripts/run_loso.sh status"
}

case "${1:-all}" in
    bm)
        echo "=== Launching BM folds (S01-S35) ==="
        folds=($(seq 1 35))
        launch_folds "${folds[@]}"
        ;;
    beta)
        echo "=== Launching BETA folds (S101-S170, excluding bad) ==="
        bad=(111 141 155 159 164)
        folds=()
        for i in $(seq 101 170); do
            skip=false
            for b in "${bad[@]}"; do
                [ "$i" -eq "$b" ] && skip=true && break
            done
            $skip || folds+=($i)
        done
        launch_folds "${folds[@]}"
        ;;
    all)
        echo "=== Launching ALL folds (BM + BETA) ==="
        bad=(111 141 155 159 164)
        folds=($(seq 1 35))
        for i in $(seq 101 170); do
            skip=false
            for b in "${bad[@]}"; do
                [ "$i" -eq "$b" ] && skip=true && break
            done
            $skip || folds+=($i)
        done
        launch_folds "${folds[@]}"
        ;;
    status)
        total_bm=$(ls -d ${CHECKPOINT_DIR}/fold_0*/summary.json 2>/dev/null | wc -l)
        total_beta=$(ls -d ${CHECKPOINT_DIR}/fold_1*/summary.json 2>/dev/null | wc -l)
        running=$(pgrep -f "loso_film.py" | wc -l)
        echo "=== LOSO Progress ==="
        echo "  BM completed:   ${total_bm}/35"
        echo "  BETA completed: ${total_beta}/65"
        echo "  Running procs:  ${running}"
        echo ""
        # Show latest completed folds
        if [ $((total_bm + total_beta)) -gt 0 ]; then
            echo "Latest completions:"
            ls -t ${CHECKPOINT_DIR}/fold_*/summary.json 2>/dev/null | head -5 | while read f; do
                fold=$(basename $(dirname $f))
                acc=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['subject_label']}: {d['val_acc']:.1%}\")")
                echo "  ${acc}"
            done
        fi
        ;;
    aggregate)
        echo "=== Aggregating results ==="
        python scripts/loso_film.py ${COMMON_ARGS} --start_fold 999 --end_fold 999
        echo "Done. See ${CHECKPOINT_DIR}/aggregate_results.json"
        ;;
    *)
        echo "Usage: bash scripts/run_loso.sh {bm|beta|all|status|aggregate}"
        exit 1
        ;;
esac
