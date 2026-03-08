"""Generate Stage 2 dialogue data for BCI-Qwen training.

Three types:
  A. 正确拼写 (30%) — EEG sequence → correct word
  C. 自动纠错 (15%) — EEG with errors → decoded (wrong) + correction suggestion
  D. 纯 NL   (15%) — no EEG, conversational text

Output: data/s2_dialogues.jsonl  (one JSON object per line)

Each dialogue has:
  - type: "A" | "C" | "D"
  - messages: [{role, content}, ...]   (chat format)
  - label_indices: [int, ...]          (for A/C: ground-truth labels per EEG block)
  - eeg_labels: [int, ...]             (for A/C: actual EEG labels used, may differ from label_indices for C)

Usage:
    uv run python scripts/generate_s2_dialogues.py --corpus data/spelling_corpus_5k.json
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)

# ─── SSVEP confusion model ────────────────────────────────────

def build_confusion_neighbors(n_classes=40, n_cols=8):
    """Build frequency-neighbor confusion map from 5x8 grid layout.

    Adjacent frequencies (±0.2Hz same column, ±1.0Hz same row) are most
    likely to be confused in SSVEP decoding.
    """
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

# ─── Keyboard mapping ─────────────────────────────────────────

KEYBOARD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0", "_", ".", "<", ">",
]

CHAR_TO_LABEL = {ch: i for i, ch in enumerate(KEYBOARD_CHARS)}
ALPHA_CHARS = set(KEYBOARD_CHARS[:26])  # A-Z only

# ─── System prompts ───────────────────────────────────────────

SYSTEM_SPELLING = (
    "你是一个脑机接口助手。用户通过注视屏幕上闪烁的字符来拼写文字。"
    "每次收到一组脑电信号后，输出解码的字符序列。"
)

SYSTEM_CHAT = "你是一个脑机接口助手，负责帮助用户通过脑电信号进行交流。"

# ─── Correction templates ─────────────────────────────────────

CORRECTION_TEMPLATES = [
    "{wrong} 你是不是想拼{right}",
    "{wrong} 可能你想输入的是{right}",
    "{wrong} 这看起来像是{right}的误拼",
    "{wrong} 检测到可能的拼写错误 建议修正为{right}",
    "{wrong} 自动纠正为{right}",
]

# ─── NL dialogue templates ────────────────────────────────────

NL_DIALOGUES = [
    # Greetings
    ("你好", "你好！我是脑机接口助手，可以帮你通过EEG信号拼写文字。"),
    ("早上好", "早上好！准备好开始拼写了吗？"),
    ("你能做什么", "我可以解码你的SSVEP脑电信号，帮你拼写文字和发送消息。"),
    # Status
    ("现在拼了什么", "当前没有进行中的拼写会话。"),
    ("信号质量怎么样", "请开始注视目标字符，我会实时评估信号质量。"),
    ("重新开始", "好的，已清空拼写内容。请重新注视目标字符。"),
    # Help
    ("怎么使用", "注视屏幕上闪烁的字符即可输入。拼写完成后我会显示结果。"),
    ("帮助", "注视字符拼写文字。说\"结束\"完成拼写，\"重新开始\"清空内容。"),
    # Confirmation
    ("确认发送", "消息已发送！"),
    ("取消", "好的，已取消发送。"),
    ("是的", "收到！"),
    ("不是", "好的，请继续拼写。"),
    ("对", "明白了。"),
    ("错了", "请告诉我哪里不对，我来修正。"),
    # Emotional
    ("谢谢", "不客气！随时可以继续拼写。"),
    ("太慢了", "理解你的感受。我们可以尝试缩短信号采集时间来加快速度。"),
    ("好累", "辛苦了！需要休息的话随时告诉我。"),
    ("继续", "好的，请继续注视目标字符。"),
    # Context
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


def word_to_labels(word):
    """Convert a word (A-Z only) to label indices. Returns None if invalid."""
    labels = []
    for ch in word:
        if ch not in CHAR_TO_LABEL:
            return None
        labels.append(CHAR_TO_LABEL[ch])
    return labels


def sample_error_rate():
    """Sample per-dialogue error rate simulating subject variability.

    Distribution (roughly matching real BCI population):
      30% strong subjects:  5-12% errors
      40% average subjects: 12-25% errors
      30% weak subjects:    25-40% errors
    """
    tier = random.random()
    if tier < 0.3:
        return random.uniform(0.05, 0.12)
    elif tier < 0.7:
        return random.uniform(0.12, 0.25)
    else:
        return random.uniform(0.25, 0.40)


# Error type weights: neighbor > drop > repeat > random
ERROR_NEIGHBOR = 0.60   # SSVEP frequency confusion (most realistic)
ERROR_DROP = 0.15       # character skipped (attention lapse)
ERROR_REPEAT = 0.10     # character decoded twice (gaze lingering)
ERROR_RANDOM = 0.15     # random substitution (noise/artifact)


def introduce_errors(labels, error_rate=None):
    """Introduce diverse BCI decoding errors.

    Error types:
      - neighbor:  replace with adjacent frequency on 5×8 grid (60%)
      - drop:      skip character entirely (15%)
      - repeat:    duplicate character (10%)
      - random:    replace with any of 40 classes (15%)

    Returns (corrupted_labels, error_count).
    Note: corrupted may differ in length from labels (drops/repeats).
    """
    if error_rate is None:
        error_rate = sample_error_rate()

    corrupted = []
    error_count = 0

    for i, label in enumerate(labels):
        if random.random() < error_rate:
            # Pick error type
            r = random.random()
            if r < ERROR_NEIGHBOR:
                # Frequency neighbor confusion
                nbrs = CONFUSION_NEIGHBORS.get(label, [])
                if nbrs:
                    corrupted.append(random.choice(nbrs))
                else:
                    corrupted.append(label)  # edge case: no neighbors
            elif r < ERROR_NEIGHBOR + ERROR_DROP:
                # Drop: skip this character entirely
                pass  # don't append
            elif r < ERROR_NEIGHBOR + ERROR_DROP + ERROR_REPEAT:
                # Repeat: output this character twice
                corrupted.append(label)
                corrupted.append(label)
            else:
                # Random substitution: any class except correct
                wrong = random.randrange(40)
                while wrong == label:
                    wrong = random.randrange(40)
                corrupted.append(wrong)
            error_count += 1
        else:
            corrupted.append(label)

    return corrupted, error_count


# ─── Dialogue generators ──────────────────────────────────────

def make_type_a(word, labels):
    """Type A: correct spelling. EEG → word."""
    assistant_text = word
    return {
        "type": "A",
        "messages": [
            {"role": "system", "content": SYSTEM_SPELLING},
            {"role": "user", "content": "__EEG__"},  # placeholder, replaced by dataset
            {"role": "assistant", "content": assistant_text},
        ],
        "label_indices": labels,
        "eeg_labels": labels,  # same as label_indices (no errors)
    }


def make_type_c(word, labels, error_rate=None):
    """Type C: auto-correction. EEG with errors → wrong + suggestion.

    Corrupted labels may differ in length from original (drops/repeats).
    """
    corrupted, error_count = introduce_errors(labels, error_rate)

    # Must have at least 1 error and non-empty result
    if error_count == 0 or len(corrupted) == 0:
        # Force one neighbor error
        corrupted = list(labels)
        pos = random.randrange(len(labels))
        nbrs = CONFUSION_NEIGHBORS.get(labels[pos], [])
        if nbrs:
            corrupted[pos] = random.choice(nbrs)
            error_count = 1
        else:
            return None

    wrong_word = "".join(KEYBOARD_CHARS[l] for l in corrupted)
    template = random.choice(CORRECTION_TEMPLATES)
    assistant_text = template.format(wrong=wrong_word, right=word)

    return {
        "type": "C",
        "messages": [
            {"role": "system", "content": SYSTEM_SPELLING},
            {"role": "user", "content": "__EEG__"},
            {"role": "assistant", "content": assistant_text},
        ],
        "label_indices": labels,       # ground truth (intended word)
        "eeg_labels": corrupted,        # actual EEG labels (with errors, may differ in length)
        "error_count": error_count,
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
        "label_indices": [],
        "eeg_labels": [],
    }


# ─── Main generation ──────────────────────────────────────────

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


def save_jsonl(dialogues, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def print_stats(dialogues, label):
    type_counts = {}
    total_chars = 0
    for d in dialogues:
        t = d["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        total_chars += len(d["eeg_labels"])
    print(f"  {label}: {len(dialogues)} dialogues, {total_chars} EEG chars")
    for t, c in sorted(type_counts.items()):
        print(f"    Type {t}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Generate S2 dialogue data")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json",
                        help="Path to spelling corpus JSON")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for train/val JSONL files")
    parser.add_argument("--n_type_a", type=int, default=5000,
                        help="Number of Type A (correct spelling) dialogues")
    parser.add_argument("--n_type_c", type=int, default=2500,
                        help="Number of Type C (auto-correction) dialogues")
    parser.add_argument("--n_type_d", type=int, default=2500,
                        help="Number of Type D (pure NL) dialogues")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2 = 2000 val)")
    args = parser.parse_args()

    # Load corpus
    with open(args.corpus) as f:
        corpus = json.load(f)
    words = corpus["sentences"]
    print(f"Loaded {len(words)} words from corpus")

    # Filter to valid words (A-Z only, convertible to labels)
    valid_words = []
    for w in words:
        labels = word_to_labels(w)
        if labels is not None and len(labels) >= 2:
            valid_words.append((w, labels))
    print(f"Valid words: {len(valid_words)}")

    dialogues = []

    # Type A: correct spelling
    print(f"Generating {args.n_type_a} Type A dialogues...")
    for _ in range(args.n_type_a):
        word, labels = random.choice(valid_words)
        dialogues.append(make_type_a(word, labels))

    # Type C: auto-correction
    print(f"Generating {args.n_type_c} Type C dialogues...")
    generated_c = 0
    while generated_c < args.n_type_c:
        word, labels = random.choice(valid_words)
        d = make_type_c(word, labels)
        if d is not None:
            dialogues.append(d)
            generated_c += 1

    # Type D: pure NL
    print(f"Generating {args.n_type_d} Type D dialogues...")
    for _ in range(args.n_type_d):
        dialogues.append(make_type_d())

    # Split train/val (stratified by type)
    train, val = split_by_type(dialogues, val_ratio=args.val_ratio)

    # Save
    out_dir = Path(args.output_dir)
    save_jsonl(train, out_dir / "s2_train.jsonl")
    save_jsonl(val, out_dir / "s2_val.jsonl")
    # Also save combined for backward compat
    save_jsonl(dialogues, out_dir / "s2_dialogues.jsonl")

    print(f"\nSaved to {out_dir}/")
    print_stats(train, "Train")
    print_stats(val, "Val")

    # Error statistics for Type C
    type_c = [d for d in dialogues if d["type"] == "C"]
    if type_c:
        len_diffs = [len(d["eeg_labels"]) - len(d["label_indices"]) for d in type_c]
        err_counts = [d.get("error_count", 0) for d in type_c]
        n_shorter = sum(1 for x in len_diffs if x < 0)  # drops
        n_longer = sum(1 for x in len_diffs if x > 0)   # repeats
        n_same = sum(1 for x in len_diffs if x == 0)     # substitution only
        print(f"\n  Type C error stats:")
        print(f"    Avg errors/word: {sum(err_counts)/len(err_counts):.1f}")
        print(f"    Length: same={n_same}, shorter(drops)={n_shorter}, longer(repeats)={n_longer}")

    # Sample
    print("\nSamples:")
    for t in ["A", "C", "D"]:
        samples = [d for d in val if d["type"] == t][:3]
        for s in samples:
            assistant = s["messages"][-1]["content"]
            target = "".join(KEYBOARD_CHARS[l] for l in s["label_indices"]) if s["label_indices"] else ""
            eeg = "".join(KEYBOARD_CHARS[l] for l in s["eeg_labels"]) if s["eeg_labels"] else "(none)"
            print(f"  [{t}] target={target} eeg={eeg} → {assistant[:70]}")


if __name__ == "__main__":
    main()
