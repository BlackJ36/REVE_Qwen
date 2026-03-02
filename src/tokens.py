"""Special token definitions for BCI-Qwen pipeline."""

# Control tokens: mark EEG embedding boundaries
BCI_START = "<|bci_start|>"
BCI_END = "<|bci_end|>"
BCI_SEP = "<|bci_sep|>"
BCI_PAD = "<|bci_pad|>"
BCI_TRANS = "<|bci_trans|>"

# FBCCA candidate rank markers (injected as explicit tokens per spell)
RANK1 = "<|rank1|>"
RANK2 = "<|rank2|>"
RANK3 = "<|rank3|>"

# FBCCA confidence levels (gap-based: top1_score - top2_score)
CONF_HIGH = "<|conf_high|>"  # gap > 0.15
CONF_MID = "<|conf_mid|>"   # 0.05 < gap <= 0.15
CONF_LOW = "<|conf_low|>"   # gap <= 0.05

CANDIDATE_TOKENS = [RANK1, RANK2, RANK3, CONF_HIGH, CONF_MID, CONF_LOW]

CONTROL_TOKENS = [BCI_START, BCI_END, BCI_SEP, BCI_PAD, BCI_TRANS] + CANDIDATE_TOKENS

# Target tokens: 40 SSVEP targets
TARGET_TOKENS = [f"<|t{i:02d}|>" for i in range(1, 41)]

ALL_SPECIAL_TOKENS = CONTROL_TOKENS + TARGET_TOKENS

# Mapping from target index (0-39) to token string
TARGET_INDEX_TO_TOKEN = {i: TARGET_TOKENS[i] for i in range(40)}
TOKEN_TO_TARGET_INDEX = {v: k for k, v in TARGET_INDEX_TO_TOKEN.items()}


def register_special_tokens(tokenizer):
    """Add BCI special tokens to tokenizer. Returns number of tokens added."""
    new_tokens = []
    for token in ALL_SPECIAL_TOKENS:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return len(new_tokens)


def get_target_token_ids(tokenizer):
    """Return dict mapping target index (0-39) to token ID."""
    return {
        i: tokenizer.convert_tokens_to_ids(tok)
        for i, tok in TARGET_INDEX_TO_TOKEN.items()
    }


# Calibrated confidence thresholds per decoder type (at 3.0s / 600pt).
# Each tuple: (high_threshold, mid_threshold).
# Calibrated to yield ~40% HIGH, ~25% MID, ~35% LOW on training set.
CONF_THRESHOLDS = {
    "fbcca": (0.16, 0.08),   # FBCCA: HIGH 41% (91.5% acc), MID 20% (64.1%), LOW 39% (28.9%)
    "trca":  (0.30, 0.12),   # TRCA: estimated, recalibrate when precomputed data available
    "etrca": (0.45, 0.18),   # eTRCA: much wider gaps — HIGH 40% (100%), MID 25% (99%), LOW 35% (55%)
}


def score_gap_to_conf_token(top1_score, top2_score, decoder_type="fbcca"):
    """Map decoder score gap to confidence token string.

    Gap = top1 correlation - top2 correlation.
    Thresholds are calibrated per decoder type.
    """
    gap = top1_score - top2_score
    high_th, mid_th = CONF_THRESHOLDS.get(decoder_type, CONF_THRESHOLDS["fbcca"])
    if gap > high_th:
        return CONF_HIGH
    elif gap > mid_th:
        return CONF_MID
    else:
        return CONF_LOW


def score_gap_to_conf_token_adaptive(top1_score, top2_score, duration_scale=1.0,
                                     decoder_type="fbcca"):
    """Duration-adaptive, decoder-aware confidence threshold.

    Shorter trials produce smaller score gaps. Scale thresholds proportionally
    so confidence remains calibrated across durations and decoder types.

    Args:
        duration_scale: trial_duration_pts / 600.0 (1.0 for 3s, 0.5 for 1.5s, etc.)
        decoder_type: "fbcca", "trca", or "etrca"
    """
    gap = top1_score - top2_score
    high_th, mid_th = CONF_THRESHOLDS.get(decoder_type, CONF_THRESHOLDS["fbcca"])
    if gap > high_th * duration_scale:
        return CONF_HIGH
    elif gap > mid_th * duration_scale:
        return CONF_MID
    else:
        return CONF_LOW
