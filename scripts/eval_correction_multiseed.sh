#!/bin/bash
# Multi-seed evaluation for FBCCA+LLM correction model.
# Generates val data with 3 different seeds, runs eval on each, averages results.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash scripts/eval_correction_multiseed.sh \
#       --checkpoint output/correction_2s/checkpoint-1500 \
#       --decoder_pts 400 --batch_size 64
#
#   # 1s model
#   CUDA_VISIBLE_DEVICES=3 bash scripts/eval_correction_multiseed.sh \
#       --checkpoint output/correction/final \
#       --decoder_pts 200 --batch_size 64

set -e

# ─── Defaults ───
CHECKPOINT="output/correction/final"
DECODER_PTS=200
BATCH_SIZE=64
EEG_DIR="data/eeg_tensors"
CORPUS="data/spelling_corpus_5k.json"
N_TYPE_A=7000
N_TYPE_C=2000
N_TYPE_D=1000
SEEDS=(42 123 456)

# ─── Parse args ───
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --decoder_pts) DECODER_PTS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --eeg_dir) EEG_DIR="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DUR_S=$((DECODER_PTS / 200))
DATA_DIR="data/correction_multiseed_${DUR_S}s"

echo "=== Multi-seed FBCCA+LLM Evaluation ==="
echo "  Checkpoint:  $CHECKPOINT"
echo "  Decoder pts: $DECODER_PTS (${DUR_S}s)"
echo "  Batch size:  $BATCH_SIZE"
echo "  Seeds:       ${SEEDS[*]}"
echo "  Data dir:    $DATA_DIR"
echo ""

# ─── Step 1: Generate val data for each seed ───
for seed in "${SEEDS[@]}"; do
    seed_dir="${DATA_DIR}/seed${seed}"
    if [ -f "${seed_dir}/val.jsonl" ]; then
        echo "[Seed $seed] val.jsonl exists, skipping generation"
    else
        echo "[Seed $seed] Generating data..."
        uv run python scripts/generate_correction_data.py \
            --eeg_dir "$EEG_DIR" \
            --corpus "$CORPUS" \
            --output_dir "$seed_dir" \
            --decoder_pts "$DECODER_PTS" \
            --seed "$seed" \
            --n_type_a $N_TYPE_A --n_type_c $N_TYPE_C --n_type_d $N_TYPE_D
    fi
done

# ─── Step 2: Run eval for each seed ───
echo ""
echo "=== Running evaluation (3 seeds) ==="

for seed in "${SEEDS[@]}"; do
    seed_dir="${DATA_DIR}/seed${seed}"
    results_file="${CHECKPOINT}/results_val.json"

    echo ""
    echo "─── Seed $seed ───"
    uv run python scripts/eval_correction.py \
        --data_dir "$seed_dir" \
        --checkpoint "$CHECKPOINT" \
        --split val \
        --batch_size "$BATCH_SIZE"

    # Move results to seed dir (eval_correction saves to checkpoint dir)
    if [ -f "$results_file" ]; then
        cp "$results_file" "${seed_dir}/results_val.json"
    fi
done

# ─── Step 3: Aggregate results ───
echo ""
echo "=== Aggregating results across ${#SEEDS[@]} seeds ==="

uv run python -c "
import json
import numpy as np

seeds = ${SEEDS[@]/%/,}
seeds = [${SEEDS[0]}, ${SEEDS[1]}, ${SEEDS[2]}]
base_dir = '${DATA_DIR}'

all_metrics = []
for seed in seeds:
    path = f'{base_dir}/seed{seed}/results_val.json'
    with open(path) as f:
        data = json.load(f)
    m = data['metrics']
    all_metrics.append(m)
    print(f'  Seed {seed}: word_acc={m.get(\"A_word_acc\", 0):.1%}, '
          f'char_acc={m.get(\"A_char_acc\", 0):.1%}, '
          f'avg_ed={m.get(\"A_avg_ed\", 0):.2f}, '
          f'C_corr={m.get(\"C_correction_acc\", 0):.1%}')

keys = all_metrics[0].keys()
print()
print('=' * 70)
print(f'  Average over {len(seeds)} seeds (mean ± std):')
print('─' * 70)
for key in sorted(keys):
    vals = [m[key] for m in all_metrics]
    if isinstance(vals[0], (int, float)):
        mean = np.mean(vals)
        std = np.std(vals)
        if key.endswith('_count'):
            print(f'  {key:>25s}: {mean:.0f}')
        else:
            print(f'  {key:>25s}: {mean:.4f} ± {std:.4f}  ({mean:.1%} ± {std:.1%})')
print('=' * 70)

# Save aggregated
agg = {}
for key in keys:
    vals = [m[key] for m in all_metrics]
    if isinstance(vals[0], (int, float)):
        agg[key] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

out_path = f'{base_dir}/results_aggregated.json'
with open(out_path, 'w') as f:
    json.dump({'seeds': seeds, 'metrics': agg}, f, indent=2, ensure_ascii=False)
print(f'\nSaved to {out_path}')
"
