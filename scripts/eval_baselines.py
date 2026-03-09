"""Evaluate all decoder baselines: CCA / FBCCA / eTRCA at 1s and 2s.

Computes per-trial accuracy and word-level spelling metrics for each decoder.
CCA is computed from raw EEG (single band, no filter bank).
FBCCA and eTRCA use precomputed top-3 files.

Usage:
    uv run python scripts/eval_baselines.py
    uv run python scripts/eval_baselines.py --decoder_pts 200 400
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


def compute_cca_word_metrics(eeg_dir, split, n_timepoints, labels, subject_ids,
                             block_ids, corpus_words, n_words=1000, seed=42):
    """Compute CCA (single-band) top-3 from raw EEG and then word-level metrics."""
    import random
    from src.fbcca import FBCCAFeatureExtractor, OCCIPITAL_CHANNELS

    rng = random.Random(seed)
    eeg_dir = Path(eeg_dir)
    eeg_data = torch.load(eeg_dir / f"{split}_eeg.pt", map_location="cpu", weights_only=True)
    eeg = eeg_data["eeg_data"]
    channel_names = eeg_data["channel_names"]

    occ_indices = [channel_names.index(ch) for ch in OCCIPITAL_CHANNELS if ch in channel_names]

    latency = 28
    total_T = eeg.shape[2]
    t_end = min(latency + n_timepoints, total_T)
    effective_T = t_end - latency
    eeg_crop = eeg[:, occ_indices, latency:t_end]

    fbcca_mod = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=effective_T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fbcca_mod = fbcca_mod.to(device)

    all_cca_top3 = []
    batch_size = 256
    for i in range(0, len(eeg_crop), batch_size):
        batch = eeg_crop[i:i+batch_size].to(device)
        with torch.no_grad():
            feat = fbcca_mod(batch)  # (B, 200) = 5 bands × 40 freqs
        feat_5x40 = feat.reshape(-1, 5, 40)
        cca_scores = feat_5x40[:, 0, :]  # band 0 only
        top3 = cca_scores.topk(3, dim=1).indices.cpu()
        all_cca_top3.append(top3)

    cca_top3 = torch.cat(all_cca_top3, dim=0)  # (N, 3)

    # Per-trial accuracy
    top1_acc = (cca_top3[:, 0] == labels).float().mean().item()
    top3_acc = (cca_top3 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Word-level: reuse same logic as compute_word_metrics_from_precomputed
    CHAR_TO_LABEL = {ch: i for i, ch in enumerate(KEYBOARD_CHARS)}
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
                    pred_label = int(cca_top3[tidx, 0])
                    decoded.append(KEYBOARD_CHARS[pred_label])
                results.append({"target": word, "noisy": "".join(decoded)})
                break

    n = len(results)
    word_correct = sum(1 for r in results if r["noisy"] == r["target"])
    char_correct = sum(
        sum(1 for a, b in zip(r["noisy"], r["target"]) if a == b)
        for r in results
    )
    char_total = sum(len(r["target"]) for r in results)
    total_ed = sum(edit_distance(r["noisy"], r["target"]) for r in results)

    return {
        "top1": top1_acc,
        "top3": top3_acc,
        "n_words": n,
        "word_acc": word_correct / max(n, 1),
        "char_acc": char_correct / max(char_total, 1),
        "avg_ed": total_ed / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all decoder baselines")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--decoder_pts", type=int, nargs="*", default=[200, 400],
                        help="Timepoints to evaluate (default: 200 400 = 1s 2s)")
    args = parser.parse_args()

    eeg_dir = Path(args.eeg_dir)
    split = "val"

    eeg_data = torch.load(eeg_dir / f"{split}_eeg.pt", map_location="cpu", weights_only=True)
    labels = eeg_data["labels"]
    subject_ids = eeg_data["subject_ids"]
    block_ids = eeg_data["block_ids"]

    with open(args.corpus) as f:
        corpus = json.load(f)
    words = [w for w in corpus.get("sentences", []) if len(w) >= 2 and w.isalpha()]

    print(f"Val: {len(labels)} trials, {len(words)} corpus words\n")

    # Collect all results: (decoder, pts) → {top1, top3, word_acc, char_acc, avg_ed}
    all_results = {}

    for pts in args.decoder_pts:
        dur_s = pts / 200
        pts_suffix = f"_{pts}pt"

        # FBCCA
        fbcca_path = eeg_dir / f"{split}_fbcca{pts_suffix}.pt"
        if fbcca_path.exists():
            cand = torch.load(fbcca_path, map_location="cpu", weights_only=True)
            top1, top3 = compute_trial_acc(cand["top3_indices"], labels)
            wm = compute_word_metrics_from_precomputed(
                args.eeg_dir, split, f"fbcca{pts_suffix}", labels,
                subject_ids, block_ids, words)
            all_results[("FBCCA", pts)] = {
                "top1": top1, "top3": top3, **wm}

        # eTRCA
        etrca_path = eeg_dir / f"{split}_etrca{pts_suffix}.pt"
        if etrca_path.exists():
            cand = torch.load(etrca_path, map_location="cpu", weights_only=True)
            top1, top3 = compute_trial_acc(cand["top3_indices"], labels)
            wm = compute_word_metrics_from_precomputed(
                args.eeg_dir, split, f"etrca{pts_suffix}", labels,
                subject_ids, block_ids, words)
            all_results[("eTRCA", pts)] = {
                "top1": top1, "top3": top3, **wm}

        # CCA (from raw EEG)
        print(f"Computing CCA {dur_s:.0f}s ({pts}pts) from raw EEG...")
        cca = compute_cca_word_metrics(
            args.eeg_dir, split, pts, labels, subject_ids, block_ids, words)
        all_results[("CCA", pts)] = cca

    # ─── Print table ───
    print()
    print("=" * 80)
    print(f"{'Decoder':>10} {'Duration':>8} │ {'Trial':>6} {'Top1':>6} {'Top3':>6} │ "
          f"{'Word':>6} {'Acc':>6} {'Char':>6} {'Acc':>6} {'AvgED':>6}")
    print("─" * 80)

    for pts in args.decoder_pts:
        dur_s = pts / 200
        for decoder in ["CCA", "FBCCA", "eTRCA"]:
            key = (decoder, pts)
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"  {decoder:>8} {dur_s:>5.0f}s    │ "
                  f"top1={r['top1']:>5.1%} top3={r['top3']:>5.1%} │ "
                  f"word={r['word_acc']:>5.1%} char={r['char_acc']:>5.1%} "
                  f"ed={r['avg_ed']:>5.2f}")
        print("─" * 80)

    print()


if __name__ == "__main__":
    main()
