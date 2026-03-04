#!/bin/bash
# Clean LOSO folds with accuracy below threshold (polluted by bad hyperparams).
#
# Usage:
#   bash scripts/clean_bad_folds.sh              # Preview only (dry run)
#   bash scripts/clean_bad_folds.sh --delete      # Actually delete bad folds
#   bash scripts/clean_bad_folds.sh --threshold 0.20  # Custom threshold (default 0.15)

CHECKPOINT_DIR="/data/zjj/loso_film"
THRESHOLD=0.15
DRY_RUN=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --delete) DRY_RUN=false; shift ;;
        --threshold) THRESHOLD=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Scanning folds (threshold: ${THRESHOLD}) ==="
echo ""

n_keep=0
n_bad=0

for f in ${CHECKPOINT_DIR}/fold_*/summary.json; do
    [ -f "$f" ] || continue
    result=$(python3 -c "
import json
d = json.load(open('$f'))
acc = d['val_acc']
label = d['subject_label']
status = 'BAD' if acc < ${THRESHOLD} else 'OK'
print(f'{status} {label} {acc:.1%}')
")
    status=$(echo "$result" | cut -d' ' -f1)
    if [ "$status" = "BAD" ]; then
        folder=$(dirname "$f")
        if $DRY_RUN; then
            echo "  [DELETE] $result  ($folder)"
        else
            rm -rf "$folder"
            echo "  Deleted: $result  ($folder)"
        fi
        n_bad=$((n_bad + 1))
    else
        echo "  [KEEP]   $result"
        n_keep=$((n_keep + 1))
    fi
done

echo ""
echo "Keep: ${n_keep}, Bad: ${n_bad}"

if $DRY_RUN && [ $n_bad -gt 0 ]; then
    echo ""
    echo "Dry run only. Run with --delete to remove bad folds."
fi
