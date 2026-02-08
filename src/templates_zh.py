"""Chinese response templates and keyboard layout for BCI agent.

Defines the 40-target SSVEP keyboard layout, frequency mapping,
and template strings for generating Stage 2 training data.
"""

import random

from .fbcca import SSVEP_FREQS

# ─────────────────────────────────────────────────────────────
# 40-target keyboard layout (5x8 grid, row-major)
# Matches Tsinghua Benchmark / BETA standard SSVEP speller.
# Label i -> KEYBOARD_CHARS[i], frequency -> SSVEP_FREQS[i]
# ─────────────────────────────────────────────────────────────
KEYBOARD_CHARS = [
    # Row 0 (8.0-15.0 Hz): letters A-H
    "A", "B", "C", "D", "E", "F", "G", "H",
    # Row 1 (8.2-15.2 Hz): letters I-P
    "I", "J", "K", "L", "M", "N", "O", "P",
    # Row 2 (8.4-15.4 Hz): letters Q-X
    "Q", "R", "S", "T", "U", "V", "W", "X",
    # Row 3 (8.6-15.6 Hz): Y, Z, digits 1-6
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    # Row 4 (8.8-15.8 Hz): digits 7-0, special
    "7", "8", "9", "0", "_", ".", "<", ">",
]

assert len(KEYBOARD_CHARS) == 40

# ─────────────────────────────────────────────────────────────
# Confidence level thresholds (based on FBCCA max correlation rho)
# ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLDS = {
    "高": 0.6,   # rho > 0.6
    "中": 0.3,   # 0.3 < rho <= 0.6
    "低": 0.0,   # rho <= 0.3
}


def confidence_label(rho):
    """Map FBCCA correlation rho to confidence string."""
    if rho > CONFIDENCE_THRESHOLDS["高"]:
        return "高"
    elif rho > CONFIDENCE_THRESHOLDS["中"]:
        return "中"
    else:
        return "低"


def target_info(label_idx):
    """Get (character, frequency) for a target label index (0-39)."""
    return KEYBOARD_CHARS[label_idx], SSVEP_FREQS[label_idx]


# ─────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "你是一个脑机接口助手，负责解码用户的SSVEP脑电信号并协助交流。"

SYSTEM_PROMPT_SPELLING = (
    "你是一个脑机接口助手。用户通过注视屏幕上闪烁的字符来拼写文字。"
    "每次收到脑电信号后，请报告解码结果和当前拼写进度。"
)

# ─────────────────────────────────────────────────────────────
# Type A: Single EEG decode templates
# ─────────────────────────────────────────────────────────────
TEMPLATES_SINGLE = [
    "检测到目标{char}（频率{freq:.1f}Hz），置信度：{conf}。",
    "解码结果：{char}，对应频率{freq:.1f}Hz。信号质量{conf}。",
    "识别到您正在注视{char}（{freq:.1f}Hz），置信度{conf}。",
    "目标：{char}（{freq:.1f}Hz），信号置信度：{conf}。",
    "检测结果：{char}，频率{freq:.1f}Hz，置信度{conf}。",
    "您注视的目标是{char}（{freq:.1f}Hz），当前信号质量：{conf}。",
    "解码完成：目标{char}，频率{freq:.1f}Hz。置信度：{conf}。",
    "SSVEP解码：{char}（{freq:.1f}Hz），置信度{conf}。",
]

TEMPLATES_SINGLE_USER = [
    "请解码以下脑电信号。",
    "解码这段脑电信号。",
    "请识别当前EEG信号。",
    "请解码。",
]

# ─────────────────────────────────────────────────────────────
# Type B: Streaming spelling step templates
# ─────────────────────────────────────────────────────────────
TEMPLATES_SPELL_STEP = [
    "{char}（{freq:.1f}Hz，{conf}）。已拼写：{spelled}",
    "检测到{char}（{freq:.1f}Hz），置信度{conf}。当前拼写：{spelled}",
    "{char}（{freq:.1f}Hz，置信度{conf}）。拼写进度：{spelled}",
]

TEMPLATES_SPELL_START_USER = [
    "开始拼写。",
    "开始拼写会话。",
    "请帮我拼写。",
]

TEMPLATES_SPELL_END = [
    "拼写完成：{spelled}。是否发送此消息？",
    "拼写内容：{spelled}。确认发送吗？",
    "已完成拼写：{spelled}。请确认是否发送。",
]

TEMPLATES_SPELL_END_USER = [
    "结束拼写",
    "完成",
    "结束",
]

# ─────────────────────────────────────────────────────────────
# Type D: Error handling templates
# ─────────────────────────────────────────────────────────────
TEMPLATES_LOW_CONFIDENCE = [
    "信号质量较低，无法可靠解码。请确保电极接触良好，并重新注视目标字符。",
    "检测到的信号置信度过低，建议重新注视目标。",
    "当前信号不稳定，解码结果可能不可靠。请重试。",
]

TEMPLATES_UNDO_USER = ["撤销", "撤销上一个", "删除上一个字符"]
TEMPLATES_UNDO_RESPONSE = [
    "已撤销最后一个字符。当前拼写内容：{spelled}",
    "已删除。当前拼写：{spelled}",
]

TEMPLATES_CLEAR_USER = ["清空", "全部清除", "重新开始"]
TEMPLATES_CLEAR_RESPONSE = [
    "已清空所有拼写内容。请继续注视目标字符。",
    "拼写内容已清空，可以重新开始。",
]

TEMPLATES_HELP_USER = ["帮助", "怎么用"]
TEMPLATES_HELP_RESPONSE = [
    (
        "可用操作：\n"
        "- 注视屏幕上的字符即可输入\n"
        "- \"撤销\"：删除上一个字符\n"
        "- \"清空\"：清除所有已拼写内容\n"
        "- \"结束拼写\"：完成并确认发送\n"
        "- \"重试\"：重新解码上一个信号"
    ),
]

# ─────────────────────────────────────────────────────────────
# Type E: Batch spelling templates
# ─────────────────────────────────────────────────────────────
TEMPLATES_BATCH_USER = [
    "请解码以下一组脑电信号。",
    "批量解码以下EEG信号。",
    "请一次性解码这些脑电信号。",
]


def format_batch_result(labels, confidences=None):
    """Format batch decoding result as numbered list.

    Args:
        labels: list of target label indices (0-39)
        confidences: optional list of confidence strings

    Returns:
        formatted result string
    """
    lines = ["解码结果："]
    spelled = ""
    for i, label in enumerate(labels):
        char, freq = target_info(label)
        conf = confidences[i] if confidences else "高"
        lines.append(f"{i+1}. {char}（{freq:.1f}Hz，置信度{conf}）")
        spelled += char
    lines.append(f"\n拼写内容：{spelled}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Helper: generate a single training example (messages list)
# ─────────────────────────────────────────────────────────────

def make_single_decode_messages(label_idx, confidence="高", num_eeg_pads=62):
    """Type A: single EEG -> NL decode."""
    from .tokens import BCI_PAD

    char, freq = target_info(label_idx)
    pads = BCI_PAD * num_eeg_pads

    user_text = random.choice(TEMPLATES_SINGLE_USER) + "\n" + pads
    assistant_text = random.choice(TEMPLATES_SINGLE).format(
        char=char, freq=freq, conf=confidence,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def make_streaming_messages(label_indices, confidences=None, num_eeg_pads=62):
    """Type B: multi-turn streaming spelling."""
    from .tokens import BCI_PAD

    pads = BCI_PAD * num_eeg_pads
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SPELLING},
    ]

    spelled = ""
    K = len(label_indices)
    for i, label in enumerate(label_indices):
        char, freq = target_info(label)
        conf = (confidences[i] if confidences else "高")
        spelled += char

        # User turn: EEG pads (first turn includes start prompt)
        if i == 0:
            user_text = random.choice(TEMPLATES_SPELL_START_USER) + "\n" + pads
        else:
            user_text = pads

        messages.append({"role": "user", "content": user_text})

        # Assistant turn: decode result + progress
        assistant_text = random.choice(TEMPLATES_SPELL_STEP).format(
            char=char, freq=freq, conf=conf, spelled=spelled,
        )
        messages.append({"role": "assistant", "content": assistant_text})

    # End turn
    messages.append({"role": "user", "content": random.choice(TEMPLATES_SPELL_END_USER)})
    messages.append({
        "role": "assistant",
        "content": random.choice(TEMPLATES_SPELL_END).format(spelled=spelled),
    })

    return messages


def make_batch_messages(label_indices, confidences=None, num_eeg_pads=62):
    """Type E: single-turn batch spelling."""
    from .tokens import BCI_PAD

    pads = BCI_PAD * num_eeg_pads
    all_pads = pads * len(label_indices)

    user_text = random.choice(TEMPLATES_BATCH_USER) + "\n" + all_pads
    assistant_text = format_batch_result(label_indices, confidences)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def make_error_messages(error_type="low_confidence", spelled=""):
    """Type D: error handling / command."""
    if error_type == "low_confidence":
        from .tokens import BCI_PAD
        pads = BCI_PAD * 62
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pads},
            {"role": "assistant", "content": random.choice(TEMPLATES_LOW_CONFIDENCE)},
        ]
    elif error_type == "undo":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_SPELLING},
            {"role": "user", "content": random.choice(TEMPLATES_UNDO_USER)},
            {"role": "assistant", "content": random.choice(TEMPLATES_UNDO_RESPONSE).format(spelled=spelled)},
        ]
    elif error_type == "clear":
        return [
            {"role": "system", "content": SYSTEM_PROMPT_SPELLING},
            {"role": "user", "content": random.choice(TEMPLATES_CLEAR_USER)},
            {"role": "assistant", "content": random.choice(TEMPLATES_CLEAR_RESPONSE)},
        ]
    elif error_type == "help":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": random.choice(TEMPLATES_HELP_USER)},
            {"role": "assistant", "content": TEMPLATES_HELP_RESPONSE[0]},
        ]
    else:
        raise ValueError(f"Unknown error type: {error_type}")
