#!/bin/bash
# FiLM ablation: 4 configs x 5 folds for quick diagnostic.
#
# Usage:
#   bash scripts/run_film_ablation.sh          # Run all 4 configs
#   bash scripts/run_film_ablation.sh status   # Check progress
#   bash scripts/run_film_ablation.sh summary  # Print results comparison

set -e

# 5 diverse folds: 2 strong + 2 medium + 1 weak (from per-subject eval)
FOLDS="1 3 8 20 33"
COMMON="--epochs 120 --patience 20 --batch_size 128 --num_workers 0 --dataset benchmark"

BASE_DIR="/tmp/film_ablation"

# Config definitions
declare -A CONFIGS
CONFIGS[A_baseline]="--film_scale 0.1 --film_reg_weight 0.01"
CONFIGS[B_scale02]="--film_scale 0.2 --film_reg_weight 1e-4"
CONFIGS[C_sigmoid]="--film_scale 0.2 --film_reg_weight 1e-4 --gamma_mode sigmoid"
CONFIGS[D_tokengate]="--film_scale 0.2 --film_reg_weight 1e-4 --token_gate"

run_all() {
    echo "=== FiLM Ablation: 4 configs x $(echo $FOLDS | wc -w) folds ==="
    echo "Folds: $FOLDS"
    echo ""

    for config_name in A_baseline B_scale02 C_sigmoid D_tokengate; do
        config_args="${CONFIGS[$config_name]}"
        ckpt_dir="${BASE_DIR}/${config_name}"
        log="${BASE_DIR}/${config_name}.log"
        mkdir -p "$ckpt_dir"

        echo "[$config_name] $config_args"
        echo "  -> $ckpt_dir"

        # Run folds sequentially (single GPU local test)
        for fold in $FOLDS; do
            echo -n "  Fold $fold..."
        done
        echo ""

        nohup python scripts/loso_film.py \
            ${COMMON} ${config_args} \
            --checkpoint_dir "$ckpt_dir" \
            --start_fold 1 --end_fold 35 \
            > "$log" 2>&1 &

        echo "  PID: $!, log: $log"
        echo ""
    done

    echo "All launched. Monitor with: bash scripts/run_film_ablation.sh status"
}

show_status() {
    echo "=== FiLM Ablation Status ==="
    for config_name in A_baseline B_scale02 C_sigmoid D_tokengate; do
        ckpt_dir="${BASE_DIR}/${config_name}"
        n_done=$(ls -d ${ckpt_dir}/fold_*/summary.json 2>/dev/null | wc -l)
        running=$(pgrep -f "checkpoint_dir.*${config_name}" 2>/dev/null | wc -l)
        echo "  ${config_name}: ${n_done}/$(echo $FOLDS | wc -w) done, ${running} running"
    done
}

show_summary() {
    echo "=== FiLM Ablation Results ==="
    echo ""
    printf "%-14s" "Fold"
    for config_name in A_baseline B_scale02 C_sigmoid D_tokengate; do
        printf "%-14s" "$config_name"
    done
    echo ""
    echo "--------------------------------------------------------------"

    for fold in $FOLDS; do
        printf "S%02d           " "$fold"
        for config_name in A_baseline B_scale02 C_sigmoid D_tokengate; do
            summary="${BASE_DIR}/${config_name}/fold_$(printf '%03d' $fold)/summary.json"
            if [ -f "$summary" ]; then
                acc=$(python3 -c "import json; print(f\"{json.load(open('$summary'))['val_acc']:.1%}\")")
                printf "%-14s" "$acc"
            else
                printf "%-14s" "-"
            fi
        done
        echo ""
    done

    echo "--------------------------------------------------------------"
    # Print means
    printf "%-14s" "Mean"
    for config_name in A_baseline B_scale02 C_sigmoid D_tokengate; do
        mean=$(python3 -c "
import json, os, statistics
accs = []
for fold in [${FOLDS// /,}]:
    p = '${BASE_DIR}/${config_name}/fold_%03d/summary.json' % fold
    if os.path.exists(p):
        accs.append(json.load(open(p))['val_acc'])
if accs:
    print(f'{statistics.mean(accs):.1%}')
else:
    print('-')
")
        printf "%-14s" "$mean"
    done
    echo ""
}

case "${1:-run}" in
    run)     run_all ;;
    status)  show_status ;;
    summary) show_summary ;;
    *)       echo "Usage: bash scripts/run_film_ablation.sh {run|status|summary}" ;;
esac
