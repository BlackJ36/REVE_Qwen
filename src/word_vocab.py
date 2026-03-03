"""Word vocabulary for BCI Agent Stage 2 word-level spelling.

Provides word lists and char→label mapping for constructing multi-spell
word sequences from real EEG data. Random sequences are included to prevent
the LLM from learning language-model shortcuts that ignore EEG signals.

All words use only A-Z characters available on the 40-target SSVEP keyboard.

To use a custom vocabulary, create a JSON file:
    {
        "common": ["THE", "AND", "FOR", ...],
        "bci": ["HELP", "YES", "NO", ...],
        "sentence": ["I WANT WATER", "CALL DOCTOR", ...],
        "weights": {"common": 0.3, "bci": 0.2, "sentence": 0.3, "random": 0.2}
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


def _to_alpha(s):
    """Strip everything except A-Z from an uppercase string."""
    return "".join(c for c in s.upper() if c in ALPHA_CHARS)


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

# Short sentences — household needs and daily expressions for locked-in patients.
# Spaces are stripped before spelling (e.g., "I WANT WATER" → "IWANTWATER").
# Kept short (≤15 chars after stripping) to fit within max_spells.
_RAW_SENTENCES = [
    # Basic needs
    "I WANT WATER", "I NEED HELP", "I AM COLD", "I AM HOT", "I AM TIRED",
    "I AM HUNGRY", "I FEEL SICK", "I FEEL GOOD", "I FEEL PAIN", "I WANT FOOD",
    "I NEED REST", "I AM FINE", "I WANT MORE", "I AM DONE", "I NEED SLEEP",
    # Requests
    "CALL DOCTOR", "CALL NURSE", "OPEN DOOR", "CLOSE DOOR", "TURN ON LIGHT",
    "TURN OFF LIGHT", "OPEN WINDOW", "COME HERE", "HELP ME", "HOLD MY HAND",
    "SIT ME UP", "LAY ME DOWN", "MORE PLEASE", "STOP PLEASE", "WAIT PLEASE",
    # Communication
    "THANK YOU", "I LOVE YOU", "GOOD NIGHT", "GOOD DAY", "SEE YOU LATER",
    "I AM SORRY", "I AM OK", "NOT YET", "YES PLEASE", "NO THANKS",
    "HOW ARE YOU", "MISS YOU", "COME BACK", "STAY HERE", "GO HOME",
    # Environment
    "TOO LOUD", "TOO DARK", "TOO COLD", "TOO HOT", "NEED AIR",
    "WANT MUSIC", "CHANGE SIDE", "FIX PILLOW", "NEED BLANKET", "WANT BOOK",
    # Daily activities
    "WATCH TV", "READ BOOK", "GO OUTSIDE", "TAKE WALK", "EAT NOW",
    "DRINK WATER", "BRUSH TEETH", "WASH FACE", "GET DRESSED", "TAKE BATH",
]

# Robot commands from command_corpus_40.json — household robot interactions
_ROBOT_COMMANDS = [
    # give_object commands
    "give me soap bottle", "bring me the bread", "hand me the apple",
    "pass me the toilet paper", "fetch me a fork", "get me a dish sponge",
    "I need the cup", "can you bring me the butter knife",
    "please give me a box", "I want the spray bottle",
    "could you get me a plunger", "grab me a bowl",
    "bring the remote control here", "I would like the newspaper",
    "hand me a baseball bat please", "pass the watch to me",
    "get the spatula for me", "give me the credit card please",
    "bring a tissue box over here", "fetch the pillow for me",
    # go_to_location commands
    "go to bed", "go to the bedroom", "head to the drawer",
    "move to the fridge", "walk to the bathtub",
    "navigate to the coffee table", "take me to the microwave",
    "go over to the dining table", "let's go to the sink",
    "proceed to the sofa", "head over to the armchair",
    "I want to go to the stove", "go check the kitchen",
    "go near the toilet", "walk over to the countertop",
    "can you go to the dining room", "please go to the shelf",
    "go towards the bathroom", "move over to the garbage can",
    "go find the cabinet",
]

# Household task descriptions from alfworld_task_corpus_100.json
_TASK_COMMANDS = [
    "Put a cardboard box on a table",
    "Fill a cup with water and place in the microwave",
    "Put a candle on top of a dresser",
    "Put the bowl with ladle in the cabinet",
    "Put a cold wine bottle in the cabinet",
    "Place a cup with a pencil in it on a book case shelf",
    "move soap bottle from back of toilet to cabinet",
    "Put a grey bowl with a black pen in it on the desk",
    "Putting clean sliced lettuce on a table",
    "Moving two vases from the fireplace to a side table",
    "put cooked egg inside fridge",
    "get a ring and turn on a lamp",
    "Turn the lamp on while holding the keys",
    "Take a apple and heat it put it back when finished",
    "place pan with spatula on back corner of table",
    "Examine a cushion by the light of a lamp",
    "Place a heated cup into the refrigerator",
    "To chill a pan and put it in the sink",
    "To move two spray bottles to the back of the toilet",
    "Put a heated egg on the counter",
    "Pick up a credit card and turn a lamp off",
    "place a cooled egg on the kitchen counter",
    "Wash the lettuce on the table",
    "place a clean mug in the coffee maker",
    "To heat the apple",
    "Look at a computer by lamp light",
    "Get the keys from the round table",
    "Clean the fork from the sink",
    "Cool an egg in the fridge",
    "Put a spray bottle in the bin",
    "Rinse the lettuce and place it inside the refrigerator",
    "Place a box on a couch",
    "Put a cooked apple in a trash bin",
    "Put a rinsed cloth in the bath tub",
    "Place two heads of lettuce in a fridge",
    "Place two credit cards on a chair",
    "Put the washed rag into the bath tub",
    "Move two forks to the sink",
    "place a cooled egg inside the microwave",
    "Read a book by lamp light",
    "Put a blue vase in a safe",
    "Examine a credit card by the light of the lamp on the desk",
    "Throw out a heated slice of tomato",
    "Put a heated tomato in the trash can",
    "Place a cool tomato in the bin",
    "Put a rinsed cloth in a cabinet",
    "Place two towels into the tub",
    "wash the spoon in the sink put it on the table",
    "Put a clean drinking glass in the microwave",
    "Put two yellow spray bottles in the trash can",
    "Put a cooked piece of apple in a sink",
    "Place clean lettuce in the fridge",
    "Put the credit cards on the blue sofa",
    "Place a frying pan with a tomato slice on a table",
    "Move a knife in a mug to the microwave cart",
    "Put a credit card on a couch",
    "Put cooked potato slice in the sink",
    "Microwave a chilled tomato",
    "Put the heated slice apple in the trash bin",
    "Put a cooked piece of bread on the table",
    "Place a heated potato on a counter",
    "Put a pillow from the couch on the chair",
    "Put two towels in the sink",
    "Turn the lamp on in the corner",
    "place chilled lettuce in sink",
    "Put a candle on the back of the toilet",
    "Look at keys in the light of a lamp",
    "Place both laptops on the counter",
    "Collecting newspapers from the room",
    "put two spoons on the counter by the microwave",
    "Place a clean plate in the cabinet",
    "Place a chilled pot on a stove top",
    "Put a chilled tomato in the microwave",
    "Place a laptop on the couch",
    "Inspect a bowl by lamplight",
]

# Strip to A-Z only and deduplicate
_ALL_RAW = _RAW_SENTENCES + _ROBOT_COMMANDS + _TASK_COMMANDS
SENTENCES = list(dict.fromkeys(_to_alpha(s) for s in _ALL_RAW))
SENTENCES = [s for s in SENTENCES if len(s) >= 3]
SENTENCES = _filter_words(SENTENCES)


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
        category_weights: dict with keys "common", "bci", "sentence", "random".
            Default: {"common": 0.4, "bci": 0.2, "sentence": 0.2, "random": 0.2}

    Returns:
        (word_string, list_of_label_indices)
    """
    weights = category_weights or {"common": 0.3, "bci": 0.2, "sentence": 0.4, "random": 0.1}
    categories = list(weights.keys())
    probs = [weights[c] for c in categories]
    choice = random.choices(categories, weights=probs, k=1)[0]

    if choice == "common":
        word = random.choice(COMMON_WORDS)
    elif choice == "bci":
        word = random.choice(BCI_PHRASES)
    elif choice == "sentence":
        word = random.choice(SENTENCES)
    else:
        word, _ = generate_random_sequence(random.randint(2, 6))
        return word, word_to_labels(word)

    return word, word_to_labels(word)


def generate_random_sequence(length):
    """Generate random character sequence from all 40 keyboard chars.

    Covers A-Z, 0-9, and special chars to prevent class imbalance.
    Also prevents the LLM from learning language model shortcuts.

    Returns:
        (sequence_string, list_of_label_indices)
    """
    chars = [random.choice(KEYBOARD_CHARS) for _ in range(length)]
    word = "".join(chars)
    return word, word_to_labels(word)


class WordVocab:
    """Configurable word vocabulary for Stage 2 spelling.

    Supports loading from a JSON file or using built-in defaults.
    Categories: common (words), bci (patient phrases), sentence (short sentences), random.

    JSON format:
        {
            "common": ["THE", "AND", ...],
            "bci": ["HELP", "YES", ...],
            "sentence": ["IWANTWATER", "INEEDHELP", ...],
            "weights": {"common": 0.3, "bci": 0.2, "sentence": 0.3, "random": 0.2}
        }

    All words are auto-filtered to A-Z only. Invalid words are silently dropped.

    Usage:
        vocab = WordVocab()                          # built-in defaults
        vocab = WordVocab("path/to/vocab.json")      # from file
        word, labels = vocab.sample()                 # sample a word/sentence
        word, labels = vocab.random_sequence(4)       # random chars
    """

    def __init__(self, path=None):
        if path is not None:
            self._load_from_file(path)
        else:
            self.common = list(COMMON_WORDS)
            self.bci = list(BCI_PHRASES)
            self.sentence = list(SENTENCES)
            self.weights = {"common": 0.3, "bci": 0.2, "sentence": 0.4, "random": 0.1}

    def _load_from_file(self, path):
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        self.common = _filter_words([w.upper() for w in data.get("common", [])])
        self.bci = _filter_words([w.upper() for w in data.get("bci", [])])
        self.sentence = _filter_words([w.upper().replace(" ", "") for w in data.get("sentence", [])])
        self.weights = data.get("weights", {"common": 0.3, "bci": 0.2, "sentence": 0.3, "random": 0.2})
        for key in self.weights:
            if key not in ("common", "bci", "sentence", "random"):
                raise ValueError(f"Unknown weight key: {key!r}. Must be common/bci/sentence/random.")
        if not self.common and not self.bci and not self.sentence:
            raise ValueError(f"Vocab file {path} has no valid words after filtering to A-Z")
        print(f"WordVocab: loaded {len(self.common)} common + {len(self.bci)} bci "
              f"+ {len(self.sentence)} sentence from {path}")

    def sample(self):
        """Sample a word/sentence. Returns (word, label_indices)."""
        categories = list(self.weights.keys())
        probs = [self.weights[c] for c in categories]
        choice = random.choices(categories, weights=probs, k=1)[0]

        if choice == "common" and self.common:
            word = random.choice(self.common)
        elif choice == "bci" and self.bci:
            word = random.choice(self.bci)
        elif choice == "sentence" and self.sentence:
            word = random.choice(self.sentence)
        else:
            return self.random_sequence(random.randint(2, 6))

        return word, word_to_labels(word)

    def random_sequence(self, length):
        """Generate random sequence from all 40 chars. Returns (word, label_indices)."""
        return generate_random_sequence(length)
