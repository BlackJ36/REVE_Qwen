"""Special token definitions for BCI-Qwen pipeline."""

# Control tokens: mark EEG embedding boundaries
BCI_START = "<|bci_start|>"
BCI_END = "<|bci_end|>"
BCI_SEP = "<|bci_sep|>"
BCI_PAD = "<|bci_pad|>"
BCI_TRANS = "<|bci_trans|>"
BCI_PRED = "<|bci_pred|>"  # Two-step prediction: EEG-only supervised marker

# Candidate marker (decoder top-3 in random order, no rank/confidence info)
CAND = "<|cand|>"

CANDIDATE_TOKENS = [CAND]

CONTROL_TOKENS = [BCI_START, BCI_END, BCI_SEP, BCI_PAD, BCI_TRANS, BCI_PRED] + CANDIDATE_TOKENS

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
