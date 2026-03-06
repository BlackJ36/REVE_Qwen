#!/bin/bash
# Run 3 improvement experiments on 2s (400pts) FiLM with D_tokengate config
# Base: D_tokengate unfreeze4 random_offset = 87.4% val

COMMON="--trial_pts 400 --random_offset --film_scale 0.2 --film_reg_weight 1e-4 --gamma_mode tanh --token_gate --epochs 60 --patience 15 --batch_size 128"

echo "=== Exp1: Label smoothing + Dropout + Unfreeze 6 ==="
uv run python scripts/finetune_film.py $COMMON \
  --unfreeze_last_n 6 --dropout 0.2 --label_smoothing 0.1 \
  --output_dir output_film/film_400_exp1_reg

echo "=== Exp2: Extended channels (20ch) ==="
uv run python scripts/finetune_film.py $COMMON \
  --unfreeze_last_n 4 --dropout 0.1 --label_smoothing 0.1 \
  --backbone_channels "Oz,O1,O2,POz,PO3,PO4,PO5,PO6,PO7,PO8,Pz,P1,P2,P3,P4,P5,P6,P7,P8,CPz" \
  --output_dir output_film/film_400_exp2_20ch

echo "=== Exp3: Mixup ==="
uv run python scripts/finetune_film.py $COMMON \
  --unfreeze_last_n 4 --dropout 0.1 --label_smoothing 0.1 --mixup_alpha 0.3 \
  --output_dir output_film/film_400_exp3_mixup

echo "=== ALL DONE ==="
