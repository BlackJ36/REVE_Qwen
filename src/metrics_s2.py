"""Stage 2 evaluation metrics: character accuracy + instruction retrieval.

Metrics:
  1. char_acc          - per-character accuracy (model output vs target)
  2. word_acc          - exact word match accuracy
  3. correction_acc    - Type C: corrected word matches intended word
  4. ed_retrieval_top1 - edit distance nearest neighbor is correct (primary)
  5. ed_retrieval_top5 - correct match in top-5 by edit distance
  6. emb_retrieval_*   - sentence embedding cosine similarity (secondary)

Edit distance is the primary retrieval metric for BCI spelling output
(no-space uppercase character strings with character-level errors).
Sentence embedding is secondary/complementary.
"""

import json
import re

import numpy as np
import requests


EMBED_URL = "http://localhost:8002/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
EMBED_BATCH_SIZE = 64

KEYBOARD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0", "_", ".", "<", ">",
]


# ─── Edit distance ────────────────────────────────────────────

def edit_distance(s1, s2):
    """Levenshtein distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _ed_retrieval(pred_words, target_words, corpus_words):
    """Find nearest corpus match by edit distance."""
    valid = [(p, t) for p, t in zip(pred_words, target_words) if p]
    if not valid:
        return {"ed_top1": 0.0, "ed_top5": 0.0, "ed_avg_dist": 0.0}

    preds = [v[0] for v in valid]
    targets = [v[1] for v in valid]

    top1_correct, top5_correct = 0, 0
    target_dists = []

    for i, pred in enumerate(preds):
        dists = [(edit_distance(pred, w), w) for w in corpus_words]
        dists.sort(key=lambda x: x[0])
        top5 = [w for _, w in dists[:5]]

        if dists[0][1] == targets[i]:
            top1_correct += 1
        if targets[i] in top5:
            top5_correct += 1

        # Distance to target
        if targets[i] in corpus_words:
            target_dists.append(edit_distance(pred, targets[i]))

    return {
        "ed_top1": top1_correct / len(preds),
        "ed_top5": top5_correct / len(preds),
        "ed_avg_dist": float(np.mean(target_dists)) if target_dists else 0.0,
    }


# ─── Sentence embedding ──────────────────────────────────────

def get_embeddings(texts, url=EMBED_URL, model=EMBED_MODEL, batch_size=EMBED_BATCH_SIZE):
    """Get sentence embeddings from local API."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(url, json={"input": batch, "model": model}, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        all_embs.extend([d["embedding"] for d in data])
    return np.array(all_embs, dtype=np.float32)


def cosine_similarity(a, b):
    """Cosine similarity between (N, D) and (M, D) matrices."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def _emb_retrieval(pred_words, target_words, corpus_words, embed_url, embed_model):
    """Find nearest corpus match by sentence embedding cosine similarity."""
    valid = [(p, t) for p, t in zip(pred_words, target_words) if p]
    if not valid:
        return {"emb_top1": 0.0, "emb_top5": 0.0, "emb_avg_sim": 0.0}

    preds = [v[0] for v in valid]
    targets = [v[1] for v in valid]

    print(f"  Embedding {len(preds)} predictions + {len(corpus_words)} corpus...")
    pred_embs = get_embeddings(preds, url=embed_url, model=embed_model)
    corpus_embs = get_embeddings(corpus_words, url=embed_url, model=embed_model)

    sim = cosine_similarity(pred_embs, corpus_embs)
    corpus_set = {w: i for i, w in enumerate(corpus_words)}

    top1_correct, top5_correct = 0, 0
    target_sims = []

    for i in range(len(preds)):
        top5_idx = np.argsort(sim[i])[-5:][::-1]
        top5_words = [corpus_words[j] for j in top5_idx]

        if top5_words[0] == targets[i]:
            top1_correct += 1
        if targets[i] in top5_words:
            top5_correct += 1

        if targets[i] in corpus_set:
            target_sims.append(float(sim[i, corpus_set[targets[i]]]))

    return {
        "emb_top1": top1_correct / len(preds),
        "emb_top5": top5_correct / len(preds),
        "emb_avg_sim": float(np.mean(target_sims)) if target_sims else 0.0,
    }


# ─── Helpers ──────────────────────────────────────────────────

def extract_correction(text):
    """Extract corrected word from Type C output like '... 想拼WATER'."""
    for pat in [r"想拼([A-Z]+)", r"纠正为([A-Z]+)", r"输入的是([A-Z]+)", r"修正为([A-Z]+)"]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def extract_raw_decoded(text):
    """Extract first uppercase block (raw decoded before correction)."""
    m = re.match(r"([A-Z]+)", text)
    return m.group(1) if m else None


def labels_to_word(label_indices):
    return "".join(KEYBOARD_CHARS[l] for l in label_indices)


def compute_char_accuracy(pred_word, target_word):
    """Character-level accuracy between two strings."""
    total = max(len(pred_word), len(target_word))
    if total == 0:
        return 1.0
    correct = sum(1 for a, b in zip(pred_word, target_word) if a == b)
    return correct / total


# ─── Main evaluation ─────────────────────────────────────────

def evaluate_s2(
    predictions,
    val_dialogues,
    corpus_path="data/spelling_corpus_5k.json",
    use_embedding=True,
    embed_url=EMBED_URL,
    embed_model=EMBED_MODEL,
):
    """Evaluate S2 model outputs.

    Args:
        predictions: list of str (model-generated assistant text per val sample)
        val_dialogues: list of dialogue dicts from s2_val.jsonl
        corpus_path: spelling corpus for retrieval evaluation
        use_embedding: whether to compute embedding-based retrieval (slow)

    Returns:
        dict of metrics
    """
    with open(corpus_path) as f:
        corpus_words = json.load(f)["sentences"]

    by_type = {"A": [], "C": [], "D": []}
    for pred, dial in zip(predictions, val_dialogues):
        t = dial["type"]
        if t in by_type:
            by_type[t].append((pred, dial))

    metrics = {}

    # ─── Type A: correct spelling ─────────────────────────────
    if by_type["A"]:
        char_accs, word_correct = [], 0
        pred_words, target_words = [], []

        for pred, dial in by_type["A"]:
            target = labels_to_word(dial["label_indices"])
            m = re.match(r"([A-Z0-9_.]+)", (pred or "").strip())
            pw = m.group(1) if m else ""

            char_accs.append(compute_char_accuracy(pw, target))
            if pw == target:
                word_correct += 1
            pred_words.append(pw)
            target_words.append(target)

        metrics["a_char_acc"] = float(np.mean(char_accs))
        metrics["a_word_acc"] = word_correct / len(by_type["A"])
        metrics["a_count"] = len(by_type["A"])

        # Edit distance retrieval (primary)
        print(f"  Type A: edit distance retrieval ({len(pred_words)} preds)...")
        ed = _ed_retrieval(pred_words, target_words, corpus_words)
        for k, v in ed.items():
            metrics[f"a_{k}"] = v

        # Embedding retrieval (secondary)
        if use_embedding:
            emb = _emb_retrieval(pred_words, target_words, corpus_words, embed_url, embed_model)
            for k, v in emb.items():
                metrics[f"a_{k}"] = v

    # ─── Type C: auto-correction ──────────────────────────────
    if by_type["C"]:
        raw_accs, correction_correct = [], 0
        pred_corrections, target_words = [], []

        for pred, dial in by_type["C"]:
            target = labels_to_word(dial["label_indices"])
            eeg_decoded = labels_to_word(dial["eeg_labels"])
            pred_text = (pred or "").strip()

            raw = extract_raw_decoded(pred_text)
            if raw:
                raw_accs.append(compute_char_accuracy(raw, eeg_decoded))

            corrected = extract_correction(pred_text)
            pred_corrections.append(corrected or "")
            if corrected == target:
                correction_correct += 1

            target_words.append(target)

        metrics["c_raw_char_acc"] = float(np.mean(raw_accs)) if raw_accs else 0.0
        metrics["c_correction_acc"] = correction_correct / len(by_type["C"])
        metrics["c_count"] = len(by_type["C"])

        # Edit distance retrieval on corrected words (primary)
        print(f"  Type C: edit distance retrieval ({len(pred_corrections)} preds)...")
        ed = _ed_retrieval(pred_corrections, target_words, corpus_words)
        for k, v in ed.items():
            metrics[f"c_{k}"] = v

        # Embedding retrieval (secondary)
        if use_embedding:
            emb = _emb_retrieval(pred_corrections, target_words, corpus_words, embed_url, embed_model)
            for k, v in emb.items():
                metrics[f"c_{k}"] = v

    metrics["d_count"] = len(by_type["D"])

    return metrics


def evaluate_from_files(
    predictions_path,
    val_path,
    corpus_path="data/spelling_corpus_5k.json",
    use_embedding=True,
):
    """Evaluate from saved prediction files.

    Args:
        predictions_path: JSONL with {"prediction": "..."} per line
        val_path: s2_val.jsonl
    """
    with open(predictions_path) as f:
        predictions = [json.loads(line)["prediction"] for line in f]
    with open(val_path) as f:
        val_dialogues = [json.loads(line) for line in f]

    metrics = evaluate_s2(
        predictions, val_dialogues, corpus_path,
        use_embedding=use_embedding,
    )

    print("\n=== S2 Evaluation Results ===")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics
