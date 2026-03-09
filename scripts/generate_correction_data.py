"""Generate text-only BCI correction training data (full S2 format).

Replaces EEG embeddings with FBCCA decoded text + per-position candidates.
Three dialogue types matching S2:
  A. 正确拼写 — FBCCA candidates → correct sentence
  C. 自动纠错 — FBCCA candidates (with errors) → wrong + correction
  D. 纯 NL   — no EEG, conversational text / robot commands

Usage:
    uv run python scripts/generate_correction_data.py \
        --eeg_dir data/eeg_tensors \
        --corpus data/spelling_corpus_5k.json \
        --output_dir data/correction \
        --n_type_a 5000 --n_type_c 2500 --n_type_d 2500
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

# ─── Keyboard mapping ─────────────────────────────────────────

KEYBOARD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0", "_", ".", "<", ">",
]
CHAR_TO_LABEL = {ch: i for i, ch in enumerate(KEYBOARD_CHARS)}

# ─── SSVEP confusion model (from generate_s2_dialogues.py) ───

def build_confusion_neighbors(n_classes=40, n_cols=8):
    neighbors = {}
    for i in range(n_classes):
        row, col = i // n_cols, i % n_cols
        nbrs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < (n_classes // n_cols) and 0 <= nc < n_cols:
                nbrs.append(nr * n_cols + nc)
        neighbors[i] = nbrs
    return neighbors

CONFUSION_NEIGHBORS = build_confusion_neighbors()

# Error type weights
ERROR_NEIGHBOR = 0.60
ERROR_DROP = 0.15
ERROR_REPEAT = 0.10
ERROR_RANDOM = 0.15

# ─── System prompts (matching S2) ─────────────────────────────

SYSTEM_SPELLING = (
    "你是一个脑机接口助手。用户通过注视屏幕上闪烁的字符来拼写文字。"
    "解码器会给出每个位置的候选字符及置信度。"
    "请根据候选信息和语言知识，输出解码结果。"
)

SYSTEM_CHAT = "你是一个脑机接口助手，负责帮助用户通过脑电信号进行交流。"

# ─── Correction templates (matching S2) ──────────────────────

CORRECTION_TEMPLATES = [
    "{wrong} 你是不是想拼{right}",
    "{wrong} 可能你想输入的是{right}",
    "{wrong} 这看起来像是{right}的误拼",
    "{wrong} 检测到可能的拼写错误 建议修正为{right}",
    "{wrong} 自动纠正为{right}",
]

# ─── NL dialogue templates (matching S2) ─────────────────────

NL_DIALOGUES = [
    ("你好", "你好！我是脑机接口助手，可以帮你通过EEG信号拼写文字。"),
    ("早上好", "早上好！准备好开始拼写了吗？"),
    ("你能做什么", "我可以解码你的SSVEP脑电信号，帮你拼写文字和发送消息。"),
    ("现在拼了什么", "当前没有进行中的拼写会话。"),
    ("信号质量怎么样", "请开始注视目标字符，我会实时评估信号质量。"),
    ("重新开始", "好的，已清空拼写内容。请重新注视目标字符。"),
    ("怎么使用", "注视屏幕上闪烁的字符即可输入。拼写完成后我会显示结果。"),
    ("帮助", "注视字符拼写文字。说\"结束\"完成拼写，\"重新开始\"清空内容。"),
    ("确认发送", "消息已发送！"),
    ("取消", "好的，已取消发送。"),
    ("是的", "收到！"),
    ("不是", "好的，请继续拼写。"),
    ("对", "明白了。"),
    ("错了", "请告诉我哪里不对，我来修正。"),
    ("谢谢", "不客气！随时可以继续拼写。"),
    ("太慢了", "理解你的感受。我们可以尝试缩短信号采集时间来加快速度。"),
    ("好累", "辛苦了！需要休息的话随时告诉我。"),
    ("继续", "好的，请继续注视目标字符。"),
    ("今天天气怎么样", "抱歉，我无法获取天气信息。但我可以帮你拼写消息询问他人。"),
    ("几点了", "我无法查看时间，需要我帮你拼写询问吗？"),
    ("叫护士", "好的，我帮你发送呼叫护士的消息。"),
    ("我不舒服", "请告诉我具体哪里不舒服，我帮你拼写给医护人员。"),
    ("我想喝水", "好的，我帮你发送消息：需要喝水。"),
    ("打开灯", "好的，发送指令：打开灯。"),
    ("关电视", "好的，发送指令：关闭电视。"),
    ("音量大一点", "好的，发送指令：增大音量。"),
    ("调低温度", "好的，发送指令：降低温度。"),
    ("打电话给家人", "好的，我帮你发送呼叫家人的请求。"),
]


def load_fbcca_data(eeg_dir, split, decoder_type="fbcca"):
    """Load EEG labels + precomputed decoder results."""
    eeg_dir = Path(eeg_dir)

    eeg_data = torch.load(eeg_dir / f"{split}_eeg.pt", map_location="cpu", weights_only=True)
    labels = eeg_data["labels"]
    subject_ids = eeg_data["subject_ids"]
    block_ids = eeg_data["block_ids"]

    cand_data = torch.load(eeg_dir / f"{split}_{decoder_type}.pt",
                           map_location="cpu", weights_only=True)
    top3_indices = cand_data["top3_indices"]  # (N, num_offsets, 3)
    top3_scores = cand_data["top3_scores"]    # (N, num_offsets, 3)

    # Group by (subject, block) → label → trial indices
    groups = defaultdict(lambda: defaultdict(list))
    for idx in range(len(labels)):
        key = (int(subject_ids[idx]), int(block_ids[idx]))
        label = int(labels[idx])
        groups[key][label].append(idx)

    print(f"[{split}] {len(labels)} trials, {len(groups)} groups, "
          f"candidates shape: {top3_indices.shape}")

    return labels, groups, top3_indices, top3_scores


def word_to_labels(word):
    """Convert word to label indices. Returns None if invalid."""
    labels = []
    for ch in word:
        if ch not in CHAR_TO_LABEL:
            return None
        labels.append(CHAR_TO_LABEL[ch])
    return labels


def sample_error_rate():
    """Sample per-dialogue error rate simulating subject variability."""
    tier = random.random()
    if tier < 0.3:
        return random.uniform(0.05, 0.12)
    elif tier < 0.7:
        return random.uniform(0.12, 0.25)
    else:
        return random.uniform(0.25, 0.40)


def introduce_errors(labels, error_rate=None):
    """Introduce diverse BCI decoding errors (drops, repeats, substitutions).

    Returns (corrupted_labels, error_count). Length may differ from input.
    """
    if error_rate is None:
        error_rate = sample_error_rate()

    corrupted = []
    error_count = 0

    for label in labels:
        if random.random() < error_rate:
            r = random.random()
            if r < ERROR_NEIGHBOR:
                nbrs = CONFUSION_NEIGHBORS.get(label, [])
                corrupted.append(random.choice(nbrs) if nbrs else label)
            elif r < ERROR_NEIGHBOR + ERROR_DROP:
                pass  # drop
            elif r < ERROR_NEIGHBOR + ERROR_DROP + ERROR_REPEAT:
                corrupted.append(label)
                corrupted.append(label)
            else:
                wrong = random.randrange(40)
                while wrong == label:
                    wrong = random.randrange(40)
                corrupted.append(wrong)
            error_count += 1
        else:
            corrupted.append(label)

    return corrupted, error_count


def get_fbcca_candidates_for_label(label, groups, top3_indices, top3_scores):
    """Get FBCCA top-3 candidates for one character label.

    Picks a random trial from a random group that has this label.
    Returns (top1_char, candidates_str) or None if not found.
    """
    # Find groups that have this label
    available_groups = [gk for gk in groups if label in groups[gk] and len(groups[gk][label]) > 0]
    if not available_groups:
        return None

    gk = random.choice(available_groups)
    trial_idx = random.choice(groups[gk][label])

    # Random offset
    n_offsets = top3_indices.shape[1]
    oidx = random.randint(0, n_offsets - 1)
    t3_idx = top3_indices[trial_idx, oidx].tolist()
    t3_sc = top3_scores[trial_idx, oidx].tolist()

    # Normalize to probabilities
    total = sum(abs(s) for s in t3_sc) + 1e-8
    t3_prob = [abs(s) / total for s in t3_sc]

    # Sort by probability
    ranked = sorted(zip(t3_idx, t3_prob), key=lambda x: -x[1])

    top1_char = KEYBOARD_CHARS[ranked[0][0]]
    cand_str = " ".join(f"{KEYBOARD_CHARS[idx]}({prob:.2f})" for idx, prob in ranked)

    return top1_char, cand_str


def build_candidates_text(eeg_labels, groups, top3_indices, top3_scores):
    """Build FBCCA candidates for a sequence of labels.

    Returns (noisy_word, candidates_text_lines) or None if any label not found.
    """
    noisy_chars = []
    cand_lines = []

    for i, label in enumerate(eeg_labels):
        result = get_fbcca_candidates_for_label(label, groups, top3_indices, top3_scores)
        if result is None:
            return None
        top1_char, cand_str = result
        noisy_chars.append(top1_char)
        cand_lines.append(f"  位置{i+1}: {cand_str}")

    noisy_word = "".join(noisy_chars)
    candidates_text = "\n".join(cand_lines)
    return noisy_word, candidates_text


def make_type_a(word, labels, groups, top3_indices, top3_scores):
    """Type A: correct spelling. Candidates → word."""
    result = build_candidates_text(labels, groups, top3_indices, top3_scores)
    if result is None:
        return None

    noisy_word, candidates_text = result
    user_text = f"解码结果: \"{noisy_word}\"\n候选:\n{candidates_text}"

    return {
        "type": "A",
        "messages": [
            {"role": "system", "content": SYSTEM_SPELLING},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": word},
        ],
        "target_word": word,
        "noisy_word": noisy_word,
    }


def make_type_c(word, labels, groups, top3_indices, top3_scores, error_rate=None):
    """Type C: auto-correction. Candidates for corrupted labels → wrong + suggestion."""
    corrupted, error_count = introduce_errors(labels, error_rate)

    # Must have at least 1 error and non-empty
    if error_count == 0 or len(corrupted) == 0:
        corrupted = list(labels)
        pos = random.randrange(len(labels))
        nbrs = CONFUSION_NEIGHBORS.get(labels[pos], [])
        if nbrs:
            corrupted[pos] = random.choice(nbrs)
        else:
            return None

    result = build_candidates_text(corrupted, groups, top3_indices, top3_scores)
    if result is None:
        return None

    noisy_word, candidates_text = result
    user_text = f"解码结果: \"{noisy_word}\"\n候选:\n{candidates_text}"

    # The "wrong" word in the assistant response is from the corrupted labels
    wrong_word = "".join(KEYBOARD_CHARS[l] for l in corrupted)
    template = random.choice(CORRECTION_TEMPLATES)
    assistant_text = template.format(wrong=wrong_word, right=word)

    return {
        "type": "C",
        "messages": [
            {"role": "system", "content": SYSTEM_SPELLING},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "target_word": word,
        "noisy_word": noisy_word,
        "wrong_word": wrong_word,
    }


def make_type_d():
    """Type D: pure NL dialogue (no EEG)."""
    user_text, assistant_text = random.choice(NL_DIALOGUES)
    return {
        "type": "D",
        "messages": [
            {"role": "system", "content": SYSTEM_CHAT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "target_word": "",
        "noisy_word": "",
    }


def split_by_type(dialogues, val_ratio=0.2):
    """Split dialogues into train/val, stratified by type."""
    by_type = {}
    for d in dialogues:
        by_type.setdefault(d["type"], []).append(d)

    train, val = [], []
    for t, items in by_type.items():
        random.shuffle(items)
        n_val = int(len(items) * val_ratio)
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Generate BCI correction data (full S2 format)")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--output_dir", type=str, default="data/correction")
    parser.add_argument("--decoder_type", type=str, default="fbcca",
                        choices=["fbcca", "trca", "etrca"])
    parser.add_argument("--n_type_a", type=int, default=5000)
    parser.add_argument("--n_type_c", type=int, default=2500)
    parser.add_argument("--n_type_d", type=int, default=2500)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load FBCCA data (use train split for data, cross-subject handled by FBCCA itself)
    labels, groups, t3i, t3s = load_fbcca_data(
        args.eeg_dir, "train", args.decoder_type)

    # Load corpus
    with open(args.corpus) as f:
        corpus = json.load(f)
    raw_words = corpus.get("sentences", corpus.get("words", []))

    # Filter valid words
    valid_words = []
    for w in raw_words:
        wl = word_to_labels(w)
        if wl is not None and len(wl) >= 2:
            valid_words.append((w, wl))
    print(f"Valid words: {len(valid_words)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogues = []

    # Type A: correct spelling
    print(f"Generating {args.n_type_a} Type A dialogues...")
    generated = 0
    attempts = 0
    while generated < args.n_type_a and attempts < args.n_type_a * 3:
        attempts += 1
        word, wl = random.choice(valid_words)
        d = make_type_a(word, wl, groups, t3i, t3s)
        if d is not None:
            dialogues.append(d)
            generated += 1
    print(f"  Generated {generated} (attempts: {attempts})")

    # Type C: auto-correction
    print(f"Generating {args.n_type_c} Type C dialogues...")
    generated = 0
    attempts = 0
    while generated < args.n_type_c and attempts < args.n_type_c * 5:
        attempts += 1
        word, wl = random.choice(valid_words)
        d = make_type_c(word, wl, groups, t3i, t3s)
        if d is not None:
            dialogues.append(d)
            generated += 1
    print(f"  Generated {generated} (attempts: {attempts})")

    # Type D: pure NL
    print(f"Generating {args.n_type_d} Type D dialogues...")
    for _ in range(args.n_type_d):
        dialogues.append(make_type_d())

    # Split train/val
    train, val = split_by_type(dialogues, val_ratio=args.val_ratio)

    # Save
    for name, data in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Stats
    for name, data in [("Train", train), ("Val", val)]:
        type_counts = {}
        for d in data:
            type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
        print(f"\n{name}: {len(data)} dialogues")
        for t, c in sorted(type_counts.items()):
            print(f"  Type {t}: {c}")

    # Type A stats
    type_a = [d for d in dialogues if d["type"] == "A"]
    if type_a:
        n_correct = sum(1 for d in type_a if d["noisy_word"] == d["target_word"])
        avg_len = sum(len(d["target_word"]) for d in type_a) / len(type_a)
        print(f"\nType A stats:")
        print(f"  FBCCA already correct: {n_correct}/{len(type_a)} ({n_correct/len(type_a):.1%})")
        print(f"  Avg word length: {avg_len:.1f}")

    # Type C stats
    type_c = [d for d in dialogues if d["type"] == "C"]
    if type_c:
        len_diffs = [len(d.get("wrong_word", "")) - len(d["target_word"]) for d in type_c]
        n_shorter = sum(1 for x in len_diffs if x < 0)
        n_longer = sum(1 for x in len_diffs if x > 0)
        n_same = sum(1 for x in len_diffs if x == 0)
        print(f"\nType C stats:")
        print(f"  Length: same={n_same}, shorter(drops)={n_shorter}, longer(repeats)={n_longer}")

    # Examples
    print("\n" + "=" * 60)
    print("Examples:")
    print("=" * 60)
    for t in ["A", "C", "D"]:
        samples = [d for d in val if d["type"] == t][:2]
        for s in samples:
            print(f"\n--- Type {t} ---")
            print(f"[System] {s['messages'][0]['content'][:60]}...")
            user_content = s['messages'][1]['content']
            # Truncate user content for display
            user_lines = user_content.split('\n')
            if len(user_lines) > 5:
                display = '\n'.join(user_lines[:3] + ['  ...'] + user_lines[-1:])
            else:
                display = user_content
            print(f"[User]\n{display}")
            print(f"[Assistant] {s['messages'][2]['content'][:80]}")


if __name__ == "__main__":
    main()
