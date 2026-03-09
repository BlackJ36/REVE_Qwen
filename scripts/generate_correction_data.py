"""Generate text-only BCI correction training data (full S2 format).

Replaces EEG embeddings with FBCCA decoded text + per-position candidates.
Three dialogue types matching S2:
  A. 正确拼写 — FBCCA candidates → correct sentence
  C. 自动纠错 — FBCCA candidates (with errors) → wrong + correction
  D. 纯 NL   — no EEG, conversational text / robot commands

v2 changes:
  - Ratio: A(70%) C(20%) D(10%) — more spelling, less NL
  - Short words (2-8 chars) augmented — easier samples for better learning
  - --short_ratio controls mix of short vs corpus words (default 0.3)

Usage:
    uv run python scripts/generate_correction_data.py \
        --eeg_dir data/eeg_tensors \
        --corpus data/spelling_corpus_5k.json \
        --output_dir data/correction \
        --n_type_a 7000 --n_type_c 2000 --n_type_d 1000
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

# ─── Short words for augmentation (2-8 chars) ────────────────

SHORT_WORDS = [
    "GO", "NO", "OK", "HI", "UP", "ON", "IN", "AT", "DO", "SO",
    "YES", "THE", "AND", "FOR", "NOT", "BUT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "ARE", "HAS", "HIS", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "WAY", "MAY", "DAY", "TOO",
    "ANY", "WHO", "BOY", "DID", "GET", "HIM", "LET", "SAY", "SHE",
    "USE", "DAD", "MOM", "RUN", "SET", "TRY", "ASK", "OWN", "WHY",
    "BIG", "END", "PUT", "RED", "SIT", "TOP", "CAT", "DOG", "EAT",
    "FAR", "HOT", "LOW", "SUN", "BED", "CUT", "FLY", "WIN", "AIR",
    "BACK", "COME", "DOOR", "EACH", "FIND", "GIVE", "HAND", "JUST",
    "KEEP", "LONG", "MAKE", "NAME", "OPEN", "PART", "READ", "SAME",
    "TAKE", "VERY", "WANT", "YEAR", "CALL", "DOES", "EVEN", "FOUR",
    "GOOD", "HAVE", "HELP", "HERE", "HOME", "INTO", "KNOW", "LAST",
    "LEFT", "LIFE", "LIKE", "LINE", "LOOK", "LOVE", "MANY", "MEAN",
    "MORE", "MOST", "MUCH", "MUST", "NEXT", "ONLY", "OVER", "PLAY",
    "ROOM", "SAID", "SHOW", "SIDE", "SOME", "SUCH", "SURE", "TELL",
    "THAN", "THEM", "THEN", "THEY", "THIS", "TIME", "TURN", "UPON",
    "WELL", "WERE", "WHAT", "WHEN", "WILL", "WITH", "WORD", "WORK",
    "AGAIN", "BEGIN", "BEING", "BELOW", "CARRY", "CLEAN", "CLOSE",
    "COVER", "DRINK", "EARLY", "EIGHT", "EVERY", "FIRST", "FLOOR",
    "FOUND", "GOING", "GREEN", "GROUP", "HAPPY", "HEART", "HOUSE",
    "LARGE", "LATER", "LEARN", "LIGHT", "MIGHT", "MONEY", "MUSIC",
    "NEVER", "NIGHT", "OFTEN", "ORDER", "OTHER", "PAPER", "PLACE",
    "PLANT", "POINT", "POWER", "QUICK", "QUITE", "RIGHT", "RIVER",
    "SEVEN", "SHALL", "SINCE", "SLEEP", "SMALL", "SOUTH", "SPACE",
    "STAND", "START", "STILL", "STORY", "TABLE", "THEIR", "THERE",
    "THESE", "THING", "THINK", "THREE", "TODAY", "UNDER", "UNTIL",
    "VOICE", "WATER", "WHERE", "WHICH", "WHILE", "WHITE", "WHOLE",
    "WORLD", "WOULD", "WRITE", "YOUNG",
    "ACROSS", "ALMOST", "ALWAYS", "ANIMAL", "ANSWER", "BEFORE",
    "BETTER", "CHANGE", "CHURCH", "DINNER", "DOCTOR", "DURING",
    "ENOUGH", "FAMILY", "FATHER", "FRIEND", "GARDEN", "HEALTH",
    "ISLAND", "ITSELF", "LITTLE", "LIVING", "MARKET", "MATTER",
    "MINUTE", "MOTHER", "MYSELF", "NATURE", "NUMBER", "OFFICE",
    "PEOPLE", "PERIOD", "PERSON", "PLEASE", "PUBLIC", "REASON",
    "RESULT", "RETURN", "SCHOOL", "SECOND", "SHOULD", "SIMPLE",
    "SISTER", "STRONG", "SUMMER", "SYSTEM", "TOWARD", "TRAVEL",
    "WINDOW", "WINTER", "WITHIN", "WONDER",
    "BECAUSE", "BROTHER", "CAPTAIN", "CERTAIN", "CHICKEN",
    "COMPANY", "CONTROL", "COUNTRY", "ENGLISH", "EXAMPLE",
    "GENERAL", "HIMSELF", "HISTORY", "HUNDRED", "KITCHEN",
    "MEASURE", "MILLION", "MORNING", "NOTHING", "PICTURE",
    "PROBLEM", "PRODUCT", "PROGRAM", "PROJECT", "PROVIDE",
    "WEATHER", "WORKING", "WRITING",
    "ACTUALLY", "ANYTHING", "BUILDING", "BUSINESS", "CHILDREN",
    "COMPUTER", "CONSIDER", "DAUGHTER", "EVERYONE", "EXERCISE",
    "FAMILIAR", "FINISHED", "FRIENDLY", "HAPPENED", "HOSPITAL",
    "INTEREST", "LEARNING", "NATIONAL", "POSSIBLE", "PRACTICE",
    "PRESSURE", "QUESTION", "TOGETHER",
]

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


def load_fbcca_data(eeg_dir, split, decoder_type="fbcca", decoder_pts=None):
    """Load EEG labels + precomputed decoder results."""
    eeg_dir = Path(eeg_dir)

    eeg_data = torch.load(eeg_dir / f"{split}_eeg.pt", map_location="cpu", weights_only=True)
    labels = eeg_data["labels"]
    subject_ids = eeg_data["subject_ids"]
    block_ids = eeg_data["block_ids"]

    # e.g. val_fbcca_200pt.pt for 1s data
    if decoder_pts:
        fname = f"{split}_{decoder_type}_{decoder_pts}pt.pt"
    else:
        fname = f"{split}_{decoder_type}.pt"
    cand_data = torch.load(eeg_dir / fname,
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


def main():
    parser = argparse.ArgumentParser(description="Generate BCI correction data (full S2 format)")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--output_dir", type=str, default="data/correction")
    parser.add_argument("--decoder_type", type=str, default="fbcca",
                        choices=["fbcca", "trca", "etrca"])
    parser.add_argument("--decoder_pts", type=int, default=200,
                        help="Timepoints for decoder (200=1s, None=full)")
    parser.add_argument("--n_type_a", type=int, default=7000)
    parser.add_argument("--n_type_c", type=int, default=2000)
    parser.add_argument("--n_type_d", type=int, default=1000)
    parser.add_argument("--short_ratio", type=float, default=0.3,
                        help="Fraction of Type A/C that use short words (2-8 chars)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load FBCCA data: train subjects for training, val subjects for evaluation
    pts = args.decoder_pts if args.decoder_pts > 0 else None
    train_labels, train_groups, train_t3i, train_t3s = load_fbcca_data(
        args.eeg_dir, "train", args.decoder_type, pts)
    val_labels, val_groups, val_t3i, val_t3s = load_fbcca_data(
        args.eeg_dir, "val", args.decoder_type, pts)

    # Load corpus
    with open(args.corpus) as f:
        corpus = json.load(f)
    raw_words = corpus.get("sentences", corpus.get("words", []))

    # Filter valid words (corpus = long phrases)
    corpus_words = []
    for w in raw_words:
        wl = word_to_labels(w)
        if wl is not None and len(wl) >= 2:
            corpus_words.append((w, wl))

    # Build short word pool
    short_words = []
    for w in SHORT_WORDS:
        wl = word_to_labels(w)
        if wl is not None and len(wl) >= 2:
            short_words.append((w, wl))

    print(f"Corpus words: {len(corpus_words)} (avg {sum(len(w) for w,_ in corpus_words)/len(corpus_words):.1f} chars)")
    print(f"Short words: {len(short_words)} (avg {sum(len(w) for w,_ in short_words)/len(short_words):.1f} chars)")
    print(f"Short ratio: {args.short_ratio:.0%}")

    def pick_word():
        """Pick a word: short_ratio chance of short, else corpus."""
        if short_words and random.random() < args.short_ratio:
            return random.choice(short_words)
        return random.choice(corpus_words)

    # For backward compatibility
    valid_words = corpus_words

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate train and val separately with their own FBCCA data
    n_train_a = args.n_type_a
    n_train_c = args.n_type_c
    n_train_d = args.n_type_d
    n_val_a = int(args.n_type_a * args.val_ratio)
    n_val_c = int(args.n_type_c * args.val_ratio)
    n_val_d = int(args.n_type_d * args.val_ratio)

    for split, groups, t3i, t3s, na, nc, nd in [
        ("train", train_groups, train_t3i, train_t3s, n_train_a, n_train_c, n_train_d),
        ("val", val_groups, val_t3i, val_t3s, n_val_a, n_val_c, n_val_d),
    ]:
        print(f"\n--- Generating {split} ---")
        dialogues = []

        # Type A
        print(f"  Type A: {na}...")
        generated = 0
        attempts = 0
        while generated < na and attempts < na * 3:
            attempts += 1
            word, wl = pick_word()
            d = make_type_a(word, wl, groups, t3i, t3s)
            if d is not None:
                dialogues.append(d)
                generated += 1
        print(f"    Generated {generated}")

        # Type C
        print(f"  Type C: {nc}...")
        generated = 0
        attempts = 0
        while generated < nc and attempts < nc * 5:
            attempts += 1
            word, wl = pick_word()
            d = make_type_c(word, wl, groups, t3i, t3s)
            if d is not None:
                dialogues.append(d)
                generated += 1
        print(f"    Generated {generated}")

        # Type D
        print(f"  Type D: {nd}...")
        for _ in range(nd):
            dialogues.append(make_type_d())

        random.shuffle(dialogues)

        # Save
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for d in dialogues:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        # Stats
        type_counts = {}
        for d in dialogues:
            type_counts[d["type"]] = type_counts.get(d["type"], 0) + 1
        print(f"  {split}: {len(dialogues)} dialogues — {type_counts}")

        # Type A stats
        type_a = [d for d in dialogues if d["type"] == "A"]
        if type_a:
            n_correct = sum(1 for d in type_a if d["noisy_word"] == d["target_word"])
            avg_len = sum(len(d["target_word"]) for d in type_a) / len(type_a)
            n_short = sum(1 for d in type_a if len(d["target_word"]) <= 8)
            print(f"  Type A: FBCCA correct={n_correct}/{len(type_a)} ({n_correct/len(type_a):.1%}), "
                  f"avg_len={avg_len:.1f}, short={n_short} ({n_short/len(type_a):.0%})")

        # Type C stats
        type_c = [d for d in dialogues if d["type"] == "C"]
        if type_c:
            len_diffs = [len(d.get("wrong_word", "")) - len(d["target_word"]) for d in type_c]
            n_shorter = sum(1 for x in len_diffs if x < 0)
            n_longer = sum(1 for x in len_diffs if x > 0)
            n_same = sum(1 for x in len_diffs if x == 0)
            print(f"  Type C: same={n_same}, drops={n_shorter}, repeats={n_longer}")

        # Show examples from val
        if split == "val":
            print("\n" + "=" * 60)
            print("Val Examples:")
            print("=" * 60)
            for t in ["A", "C", "D"]:
                samples = [d for d in dialogues if d["type"] == t][:2]
                for s in samples:
                    print(f"\n--- Type {t} ---")
                    print(f"[System] {s['messages'][0]['content'][:60]}...")
                    user_content = s['messages'][1]['content']
                    user_lines = user_content.split('\n')
                    if len(user_lines) > 5:
                        display = '\n'.join(user_lines[:3] + ['  ...'] + user_lines[-1:])
                    else:
                        display = user_content
                    print(f"[User]\n{display}")
                    print(f"[Assistant] {s['messages'][2]['content'][:80]}")


if __name__ == "__main__":
    main()
