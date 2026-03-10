"""Compute word-level spelling metrics for FiLM classifier checkpoints.

Loads trained FiLMClassifier, runs inference on val set for top-3 predictions,
then computes word-level spelling metrics (same protocol as eval_baselines.py).

Usage:
    uv run python scripts/eval_film_spelling.py \
        --checkpoint output_film/film_200_lora16/best_model.pt \
        --config output_film/film_200_lora16/config.json

    # Two checkpoints (1s + 2s)
    uv run python scripts/eval_film_spelling.py \
        --checkpoint output_film/film_200_lora16/best_model.pt \
        --checkpoint2 output_film/film_400_lora16_po56/best_model.pt
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_bci_agent import BETA_BAD_SUBJECTS
from src.dataset_reve_finetune import REVEFinetuneDataset, reve_finetune_collate_fn
from src.film_classifier import build_film_classifier

KEYBOARD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0", "_", ".", "<", ">",
]
CHAR_TO_LABEL = {ch: i for i, ch in enumerate(KEYBOARD_CHARS)}


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def get_predictions(model, val_ds, device, batch_size=256):
    """Run inference and return top-3 predictions for all val trials."""
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=reve_finetune_collate_fn,
    )
    model.eval()

    all_top3 = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            labels = batch["labels"]
            logits = model(eeg)  # (B, 40)
            top3 = logits.topk(3, dim=-1).indices.cpu()
            all_top3.append(top3)
            all_labels.append(labels)

    top3 = torch.cat(all_top3, dim=0)  # (N, 3)
    labels = torch.cat(all_labels, dim=0)

    top1_acc = (top3[:, 0] == labels).float().mean().item()
    top3_acc = (top3 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    print(f"  Per-trial top-1 acc: {top1_acc:.1%} ({int(top1_acc*len(labels))}/{len(labels)})")
    print(f"  Per-trial top-3 acc: {top3_acc:.1%}")

    return top3, labels, top1_acc, top3_acc


def compute_word_metrics(top3, labels, subject_ids, block_ids,
                         corpus_words, n_words=1000, seed=42):
    """Compute word-level spelling metrics from top-3 predictions."""
    rng = random.Random(seed)

    groups = defaultdict(lambda: defaultdict(list))
    for idx in range(len(labels)):
        key = (int(subject_ids[idx]), int(block_ids[idx]))
        groups[key][int(labels[idx])].append(idx)

    results = []
    for _ in range(n_words * 3):
        if len(results) >= n_words:
            break
        word = rng.choice(corpus_words)
        label_indices = [CHAR_TO_LABEL.get(c) for c in word]
        if None in label_indices:
            continue
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        for gk in group_keys:
            if all(len(groups[gk].get(l, [])) > 0 for l in label_indices):
                decoded = []
                for l in label_indices:
                    tidx = rng.choice(groups[gk][l])
                    pred_label = int(top3[tidx, 0])
                    decoded.append(KEYBOARD_CHARS[pred_label])
                results.append({"target": word, "noisy": "".join(decoded)})
                break

    n = len(results)
    word_correct = sum(1 for r in results if r["noisy"] == r["target"])
    char_total = sum(len(r["target"]) for r in results)
    total_ed = sum(edit_distance(r["noisy"], r["target"]) for r in results)

    return {
        "n_words": n,
        "word_acc": word_correct / max(n, 1),
        "char_acc": 1.0 - total_ed / max(char_total, 1),
        "avg_ed": total_ed / max(n, 1),
        "avg_word_len": char_total / max(n, 1),
    }


def run_checkpoint(ckpt_path, config_path, eeg_dir, corpus_words,
                   device, batch_size=256):
    """Load checkpoint, run inference, compute metrics."""
    with open(config_path) as f:
        cfg = json.load(f)

    trial_pts = cfg["trial_pts"]
    dur_s = trial_pts / 200
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"  trial_pts={trial_pts} ({dur_s:.0f}s), use_film={cfg.get('use_film', True)}")
    print(f"  unfreeze={cfg.get('unfreeze_last_n', 0)}, "
          f"lora_rank={cfg.get('lora_rank', 0)}")
    print(f"{'='*60}")

    # Load val dataset
    exclude = BETA_BAD_SUBJECTS if cfg.get("exclude_bad_subjects", True) else None
    val_ds = REVEFinetuneDataset(
        eeg_dir, split="val",
        trial_duration_pts=trial_pts,
        exclude_subjects=exclude,
        use_etrca=False,
    )

    # Build model with same config
    backbone_ch = cfg.get("backbone_channels", None)
    if isinstance(backbone_ch, str):
        backbone_ch = backbone_ch.split(",")

    model = build_film_classifier(
        reve_dir="models",
        trial_pts=val_ds.trial_duration_pts,
        use_film=cfg.get("use_film", True),
        unfreeze_last_n=cfg.get("unfreeze_last_n", 0),
        film_scale=cfg.get("film_scale", 0.1),
        film_reg_weight=cfg.get("film_reg_weight", 0.01),
        gamma_mode=cfg.get("gamma_mode", "tanh"),
        use_token_gate=cfg.get("token_gate", False),
        dropout=cfg.get("dropout", 0.0),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        backbone_channels=backbone_ch,
        lora_rank=cfg.get("lora_rank", 0),
        lora_alpha=cfg.get("lora_alpha", 16),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    print(f"  Loaded {len(ckpt)} tensors")

    # Get predictions
    top3, labels, top1_acc, top3_acc = get_predictions(
        model, val_ds, device, batch_size
    )

    # Load subject/block IDs
    eeg_data = torch.load(
        Path(eeg_dir) / "val_eeg.pt", map_location="cpu", weights_only=True
    )
    subject_ids = eeg_data["subject_ids"]
    block_ids = eeg_data["block_ids"]

    if exclude:
        mask = torch.ones(len(subject_ids), dtype=torch.bool)
        for sid in exclude:
            mask &= subject_ids != sid
        subject_ids = subject_ids[mask]
        block_ids = block_ids[mask]

    # Word-level metrics
    print("\nComputing word-level spelling metrics...")
    wm = compute_word_metrics(top3, labels, subject_ids, block_ids, corpus_words)

    print(f"\n  FiLM {dur_s:.0f}s: trial={top1_acc:.1%}, "
          f"word={wm['word_acc']:.1%}, char={wm['char_acc']:.1%}, "
          f"ed={wm['avg_ed']:.2f}")

    del model
    torch.cuda.empty_cache()

    return {
        "checkpoint": str(ckpt_path),
        "trial_pts": trial_pts,
        "duration_s": dur_s,
        "trial_acc": top1_acc,
        "trial_top3": top3_acc,
        **wm,
    }


def main():
    parser = argparse.ArgumentParser(
        description="FiLM classifier word-level spelling metrics"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Config JSON (default: same dir as checkpoint)")
    parser.add_argument("--checkpoint2", type=str, default=None)
    parser.add_argument("--config2", type=str, default=None)
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.corpus) as f:
        corpus = json.load(f)
    words = [w for w in corpus.get("sentences", []) if len(w) >= 2 and w.isalpha()]
    print(f"Corpus: {len(words)} words")

    results = []

    # Checkpoint 1
    cfg1 = args.config or str(Path(args.checkpoint).parent / "config.json")
    r1 = run_checkpoint(
        args.checkpoint, cfg1, args.eeg_dir, words, device, args.batch_size
    )
    results.append(r1)

    # Checkpoint 2
    if args.checkpoint2:
        cfg2 = args.config2 or str(Path(args.checkpoint2).parent / "config.json")
        r2 = run_checkpoint(
            args.checkpoint2, cfg2, args.eeg_dir, words, device, args.batch_size
        )
        results.append(r2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Method':>15} {'Duration':>8} | {'Trial':>6} {'Top3':>6} | "
          f"{'Word':>6} {'Char':>6} {'AvgED':>6}")
    print(f"{'-'*70}")
    for r in results:
        print(f"  {'FiLM':>13} {r['duration_s']:>5.0f}s    | "
              f"{r['trial_acc']:>5.1%} {r['trial_top3']:>5.1%} | "
              f"{r['word_acc']:>5.1%} {r['char_acc']:>5.1%} "
              f"{r['avg_ed']:>5.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
