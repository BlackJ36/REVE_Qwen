"""Evaluate CCA / FBCCA baselines on val correction data.

Computes word-level and char-level accuracy for:
  1. FBCCA 1s (200pts) — from precomputed val_fbcca_200pt.pt
  2. FBCCA 3s (full)   — from precomputed val_fbcca.pt
  3. CCA 1s (single band, no filter bank) — computed from raw EEG

Usage:
    uv run python scripts/eval_baselines.py
    uv run python scripts/eval_baselines.py --compute_cca  # also compute CCA from raw EEG
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


KEYBOARD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0", "_", ".", "<", ">",
]


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


def compute_trial_acc(top3_indices, labels, offset=0):
    """Compute per-trial top-1 and top-3 accuracy."""
    n_offsets = top3_indices.shape[1]
    oidx = min(offset, n_offsets - 1)
    top1 = top3_indices[:, oidx, 0]
    top1_acc = (top1 == labels).float().mean().item()

    top3 = top3_indices[:, oidx, :]  # (N, 3)
    top3_hit = (top3 == labels.unsqueeze(1)).any(dim=1)
    top3_acc = top3_hit.float().mean().item()

    return top1_acc, top3_acc


def compute_word_metrics_from_precomputed(eeg_dir, split, decoder_name, labels, subject_ids,
                                          block_ids, corpus_words, n_words=1000, seed=42):
    """Build words from precomputed decoder and compute word/char metrics."""
    import random
    rng = random.Random(seed)

    eeg_dir = Path(eeg_dir)
    cand_data = torch.load(eeg_dir / f"{split}_{decoder_name}.pt",
                           map_location="cpu", weights_only=True)
    top3_indices = cand_data["top3_indices"]

    # Group by (subject, block) → label → trial indices
    CHAR_TO_LABEL = {ch: i for i, ch in enumerate(KEYBOARD_CHARS)}
    groups = defaultdict(lambda: defaultdict(list))
    for idx in range(len(labels)):
        key = (int(subject_ids[idx]), int(block_ids[idx]))
        label = int(labels[idx])
        groups[key][label].append(idx)

    results = []
    for _ in range(n_words * 3):
        if len(results) >= n_words:
            break
        word = rng.choice(corpus_words)
        label_indices = [CHAR_TO_LABEL.get(c) for c in word]
        if None in label_indices:
            continue

        # Find a group with all labels
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        found = False
        for gk in group_keys:
            if all(len(groups[gk].get(l, [])) > 0 for l in label_indices):
                # Decode each char
                decoded = []
                for l in label_indices:
                    tidx = rng.choice(groups[gk][l])
                    pred_label = int(top3_indices[tidx, 0, 0])
                    decoded.append(KEYBOARD_CHARS[pred_label])
                noisy = "".join(decoded)
                results.append({"target": word, "noisy": noisy})
                found = True
                break

    # Metrics
    n = len(results)
    word_correct = sum(1 for r in results if r["noisy"] == r["target"])
    char_correct = 0
    char_total = 0
    total_ed = 0
    for r in results:
        noisy, target = r["noisy"], r["target"]
        for j in range(min(len(noisy), len(target))):
            if noisy[j] == target[j]:
                char_correct += 1
        char_total += len(target)
        total_ed += edit_distance(noisy, target)

    return {
        "n_words": n,
        "word_acc": word_correct / max(n, 1),
        "char_acc": char_correct / max(char_total, 1),
        "avg_ed": total_ed / max(n, 1),
    }


def compute_cca_from_eeg(eeg_dir, split, n_timepoints=200):
    """Compute basic CCA (no filter bank) from raw EEG."""
    from src.fbcca import FBCCAFeatureExtractor, SSVEP_FREQS, OCCIPITAL_CHANNELS

    eeg_dir = Path(eeg_dir)
    eeg_data = torch.load(eeg_dir / f"{split}_eeg.pt", map_location="cpu", weights_only=True)
    eeg = eeg_data["eeg_data"]      # (N, 62, 600)
    labels = eeg_data["labels"]
    channel_names = eeg_data["channel_names"]

    # Find occipital channel indices
    occ_indices = []
    for ch in OCCIPITAL_CHANNELS:
        if ch in channel_names:
            occ_indices.append(channel_names.index(ch))
    print(f"  CCA using {len(occ_indices)} occipital channels")

    # Truncate to n_timepoints, skip latency (28 pts = 0.14s)
    latency = 28
    eeg_crop = eeg[:, occ_indices, latency:latency + n_timepoints]  # (N, 9, T)

    # Build CCA (single band: use full spectrum, no filter bank)
    # Reuse FBCCAFeatureExtractor but only take band 0 (6-90 Hz, widest)
    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=n_timepoints)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fbcca = fbcca.to(device)
    eeg_crop = eeg_crop.to(device)

    # Process in batches
    batch_size = 256
    all_features = []
    for i in range(0, len(eeg_crop), batch_size):
        batch = eeg_crop[i:i+batch_size]
        with torch.no_grad():
            feat = fbcca(batch)  # (B, 200) = 5 bands × 40 freqs
        all_features.append(feat.cpu())
    all_features = torch.cat(all_features, dim=0)  # (N, 200)

    # FBCCA: weighted sum across bands → (N, 40)
    band_weights = fbcca.band_weights  # (5,)
    feat_5x40 = all_features.reshape(-1, 5, 40)
    fbcca_scores = (feat_5x40 * band_weights.cpu().unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (N, 40)
    fbcca_pred = fbcca_scores.argmax(dim=1)
    fbcca_acc = (fbcca_pred == labels).float().mean().item()

    # CCA: only band 0 (6-90 Hz, widest, no filter bank weighting)
    cca_scores = feat_5x40[:, 0, :]  # (N, 40) — first band only
    cca_pred = cca_scores.argmax(dim=1)
    cca_acc = (cca_pred == labels).float().mean().item()

    # Top-3
    fbcca_top3 = fbcca_scores.topk(3, dim=1).indices  # (N, 3)
    fbcca_top3_acc = (fbcca_top3 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    cca_top3 = cca_scores.topk(3, dim=1).indices
    cca_top3_acc = (cca_top3 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return {
        "cca_top1": cca_acc,
        "cca_top3": cca_top3_acc,
        "fbcca_top1": fbcca_acc,
        "fbcca_top3": fbcca_top3_acc,
        "n_trials": len(labels),
        "n_timepoints": n_timepoints,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CCA/FBCCA baselines")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--correction_dir", type=str, default="data/correction")
    parser.add_argument("--compute_cca", action="store_true",
                        help="Compute CCA from raw EEG (needs GPU)")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    split = "val"

    # Load common data
    eeg_data = torch.load(eeg_dir / f"{split}_eeg.pt", map_location="cpu", weights_only=True)
    labels = eeg_data["labels"]
    subject_ids = eeg_data["subject_ids"]
    block_ids = eeg_data["block_ids"]

    # Load corpus
    with open(args.corpus) as f:
        corpus = json.load(f)
    words = [w for w in corpus.get("sentences", []) if len(w) >= 2 and w.isalpha()]

    print(f"Val: {len(labels)} trials, {len(words)} corpus words\n")

    # ─── 1. Per-trial accuracy from precomputed ───
    print("=" * 60)
    print("Per-trial accuracy (val subjects)")
    print("=" * 60)
    for decoder_name, desc in [
        ("fbcca_200pt", "FBCCA 1s (200pts)"),
        ("fbcca_100pt", "FBCCA 0.5s (100pts)"),
        ("fbcca", "FBCCA full (~3s)"),
        ("etrca", "eTRCA"),
    ]:
        fpath = eeg_dir / f"{split}_{decoder_name}.pt"
        if not fpath.exists():
            continue
        cand = torch.load(fpath, map_location="cpu", weights_only=True)
        top1_acc, top3_acc = compute_trial_acc(cand["top3_indices"], labels)
        print(f"  {desc:25s}  top1={top1_acc:.1%}  top3={top3_acc:.1%}")

    # ─── 2. Word-level accuracy (simulated spelling) ───
    print()
    print("=" * 60)
    print("Word-level accuracy (1000 random words, val subjects)")
    print("=" * 60)
    for decoder_name, desc in [
        ("fbcca_200pt", "FBCCA 1s (200pts)"),
        ("fbcca", "FBCCA full (~3s)"),
        ("etrca", "eTRCA"),
    ]:
        fpath = eeg_dir / f"{split}_{decoder_name}.pt"
        if not fpath.exists():
            continue
        m = compute_word_metrics_from_precomputed(
            args.eeg_dir, split, decoder_name, labels, subject_ids, block_ids, words)
        print(f"  {desc:25s}  word_acc={m['word_acc']:.1%}  char_acc={m['char_acc']:.1%}  avg_ed={m['avg_ed']:.2f}")

    # ─── 3. Correction data baseline (from val.jsonl) ───
    val_jsonl = Path(args.correction_dir) / "val.jsonl"
    if val_jsonl.exists():
        print()
        print("=" * 60)
        print("Correction val.jsonl baseline (FBCCA noisy_word vs target)")
        print("=" * 60)
        samples = []
        with open(val_jsonl) as f:
            for line in f:
                samples.append(json.loads(line))

        for dtype in ["A", "C", "A+C"]:
            if dtype == "A+C":
                subset = [s for s in samples if s["type"] in ("A", "C")]
            else:
                subset = [s for s in samples if s["type"] == dtype]
            if not subset:
                continue
            n = len(subset)
            word_ok = sum(1 for s in subset if s["noisy_word"] == s["target_word"])
            char_ok = sum(
                sum(1 for a, b in zip(s["noisy_word"], s["target_word"]) if a == b)
                for s in subset
            )
            char_total = sum(len(s["target_word"]) for s in subset)
            avg_ed = sum(edit_distance(s["noisy_word"], s["target_word"]) for s in subset) / n
            print(f"  Type {dtype:5s} (n={n:4d})  word_acc={word_ok/n:.1%}  char_acc={char_ok/char_total:.1%}  avg_ed={avg_ed:.2f}")

    # ─── 4. CCA from raw EEG (optional) ───
    if args.compute_cca:
        print()
        print("=" * 60)
        print("CCA vs FBCCA computed from raw EEG (val subjects)")
        print("=" * 60)
        for pts, desc in [(200, "1s"), (400, "2s"), (600, "3s")]:
            # Check valid_pts
            valid_pts = eeg_data.get("valid_pts", None)
            if valid_pts is not None:
                min_valid = int(valid_pts.min())
                if pts > min_valid - 28:
                    print(f"  Skipping {desc} ({pts}pts): some trials only have {min_valid} valid pts")
                    continue
            m = compute_cca_from_eeg(args.eeg_dir, split, n_timepoints=pts)
            print(f"  {desc} ({pts}pts):  CCA top1={m['cca_top1']:.1%} top3={m['cca_top3']:.1%}"
                  f"  |  FBCCA top1={m['fbcca_top1']:.1%} top3={m['fbcca_top3']:.1%}")


if __name__ == "__main__":
    main()
