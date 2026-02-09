"""Word vocabulary for BCI Agent Stage 2 word-level spelling.

Provides word lists and char→label mapping for constructing multi-spell
word sequences from real EEG data. Random sequences are included to prevent
the LLM from learning language-model shortcuts that ignore EEG signals.

All words use only A-Z characters available on the 40-target SSVEP keyboard.

To use a custom vocabulary, create a JSON file:
    {
        "common": ["THE", "AND", "FOR", ...],
        "bci": ["HELP", "YES", "NO", ...],
        "weights": {"common": 0.5, "bci": 0.3, "random": 0.2}
    }
Then pass --word_vocab path/to/vocab.json to training.
"""

import json
import random
from pathlib import Path

from .templates_zh import KEYBOARD_CHARS

# Map each keyboard character to its label index (0-39)
CHAR_TO_LABEL = {char: idx for idx, char in enumerate(KEYBOARD_CHARS)}

# Only A-Z are valid for word spelling (labels 0-25)
ALPHA_CHARS = set(KEYBOARD_CHARS[:26])


def _filter_words(words):
    """Keep only words whose characters are all on the keyboard (A-Z)."""
    return [w for w in words if all(c in ALPHA_CHARS for c in w)]


# Common English words (high-frequency, all A-Z)
COMMON_WORDS = _filter_words([
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "HAD", "HAS", "HIS", "HOW", "ITS",
    "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET",
    "LET", "SAY", "SHE", "TOO", "USE", "DAD", "MOM", "RUN", "SET", "TRY",
    "ASK", "MEN", "RAN", "BIG", "END", "PUT", "GOT", "TOP", "RED", "BAD",
    "THAT", "WITH", "HAVE", "THIS", "WILL", "YOUR", "FROM", "THEY", "BEEN",
    "CALL", "COME", "EACH", "FIND", "GOOD", "HELP", "HERE", "HOME", "JUST",
    "KEEP", "KNOW", "LAST", "LONG", "LOOK", "MADE", "MAKE", "MORE", "MUCH",
    "MUST", "NAME", "NEED", "NEXT", "ONLY", "OVER", "PART", "PLAY", "SAME",
    "SHOW", "SIDE", "TAKE", "TELL", "THEM", "THEN", "TURN", "VERY", "WANT",
    "WENT", "WHEN", "WORK", "YEAR", "BACK", "GIVE", "HAND", "HIGH", "LEFT",
    "LIFE", "LINE", "LIVE", "MOVE", "OPEN", "PLAN", "REAL", "ROOM", "STOP",
    "SURE", "TALK", "TIME", "USED", "WELL", "WORD", "ALSO", "AREA", "BOOK",
    "CITY", "DONE", "DOWN", "EVEN", "FACE", "FEEL", "FOUR", "FREE", "FULL",
    "HEAD", "KIND", "LAND", "LATE", "LEAD", "LEFT", "LIKE", "LIST", "LOVE",
    "MEAN", "MIND", "MOST", "NEAR", "NEWS", "NOTE", "ONCE", "PICK", "PLAN",
    "POINT", "AFTER", "AGAIN", "BEING", "BELOW", "CHILD", "EVERY", "FIRST",
    "FOUND", "GOING", "GREAT", "GROUP", "HOUSE", "LARGE", "LATER", "LEARN",
    "LEAVE", "LIGHT", "MIGHT", "NEVER", "NIGHT", "ORDER", "OTHER", "PLACE",
    "PLANT", "POINT", "RIGHT", "SHALL", "SINCE", "SMALL", "SOUND", "SPELL",
    "STAND", "START", "STILL", "STUDY", "THING", "THINK", "THREE", "TIMES",
    "UNDER", "UNTIL", "WATER", "WHILE", "WORLD", "WOMAN", "WOULD", "WRITE",
    "YOUNG",
])

# BCI-specific phrases — things a locked-in patient might need
BCI_PHRASES = _filter_words([
    "HELP", "YES", "NO", "PAIN", "WATER", "DOCTOR", "NURSE", "COLD", "HOT",
    "GOOD", "BAD", "MORE", "STOP", "THANKS", "LOVE", "FAMILY", "TIRED",
    "HUNGRY", "THIRSTY", "PLEASE", "SORRY", "HELLO", "BYE",
    "OK", "FINE", "SICK", "NEED", "WANT", "FEEL", "CALL", "HOME",
    "SLEEP", "LIGHT", "DARK", "OPEN", "CLOSE", "UP", "DOWN", "LEFT", "RIGHT",
    "BACK", "DONE", "WAIT", "AGAIN", "FOOD", "DRINK", "WARM", "REST",
])


def word_to_labels(word):
    """Convert word to list of label indices. Returns None if invalid chars."""
    word = word.upper()
    labels = []
    for c in word:
        if c not in CHAR_TO_LABEL:
            return None
        labels.append(CHAR_TO_LABEL[c])
    return labels


def sample_word(category_weights=None):
    """Sample a word and return (word, label_indices).

    Args:
        category_weights: dict with keys "common", "bci", "random".
            Default: {"common": 0.5, "bci": 0.3, "random": 0.2}

    Returns:
        (word_string, list_of_label_indices)
    """
    weights = category_weights or {"common": 0.5, "bci": 0.3, "random": 0.2}
    categories = list(weights.keys())
    probs = [weights[c] for c in categories]
    choice = random.choices(categories, weights=probs, k=1)[0]

    if choice == "common":
        word = random.choice(COMMON_WORDS)
    elif choice == "bci":
        word = random.choice(BCI_PHRASES)
    else:
        word, _ = generate_random_sequence(random.randint(2, 6))
        return word, word_to_labels(word)

    return word, word_to_labels(word)


def generate_random_sequence(length):
    """Generate random A-Z character sequence (no linguistic pattern).

    This prevents the LLM from learning to ignore EEG and just do
    language model completion.

    Returns:
        (sequence_string, list_of_label_indices)
    """
    chars = [random.choice(KEYBOARD_CHARS[:26]) for _ in range(length)]
    word = "".join(chars)
    return word, word_to_labels(word)


class WordVocab:
    """Configurable word vocabulary for Stage 2 spelling.

    Supports loading from a JSON file or using built-in defaults.

    JSON format:
        {
            "common": ["THE", "AND", ...],
            "bci": ["HELP", "YES", ...],
            "weights": {"common": 0.5, "bci": 0.3, "random": 0.2}
        }

    All words are auto-filtered to A-Z only. Invalid words are silently dropped.

    Usage:
        vocab = WordVocab()                          # built-in defaults
        vocab = WordVocab("path/to/vocab.json")      # from file
        word, labels = vocab.sample()                 # sample a word
        word, labels = vocab.random_sequence(4)       # random chars
    """

    def __init__(self, path=None):
        if path is not None:
            self._load_from_file(path)
        else:
            self.common = list(COMMON_WORDS)
            self.bci = list(BCI_PHRASES)
            self.weights = {"common": 0.5, "bci": 0.3, "random": 0.2}

    def _load_from_file(self, path):
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        self.common = _filter_words([w.upper() for w in data.get("common", [])])
        self.bci = _filter_words([w.upper() for w in data.get("bci", [])])
        self.weights = data.get("weights", {"common": 0.5, "bci": 0.3, "random": 0.2})
        # Validate weights keys
        for key in self.weights:
            if key not in ("common", "bci", "random"):
                raise ValueError(f"Unknown weight key: {key!r}. Must be common/bci/random.")
        if not self.common and not self.bci:
            raise ValueError(f"Vocab file {path} has no valid words after filtering to A-Z")
        print(f"WordVocab: loaded {len(self.common)} common + {len(self.bci)} bci words from {path}")

    def sample(self):
        """Sample a word. Returns (word, label_indices)."""
        categories = list(self.weights.keys())
        probs = [self.weights[c] for c in categories]
        choice = random.choices(categories, weights=probs, k=1)[0]

        if choice == "common" and self.common:
            word = random.choice(self.common)
        elif choice == "bci" and self.bci:
            word = random.choice(self.bci)
        else:
            return self.random_sequence(random.randint(2, 6))

        return word, word_to_labels(word)

    def random_sequence(self, length):
        """Generate random A-Z sequence. Returns (word, label_indices)."""
        return generate_random_sequence(length)
