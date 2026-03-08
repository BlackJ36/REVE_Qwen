"""Augment spelling corpus from ~160 seed sentences to ~5000 via combinatorial expansion.

Strategy:
  1. Object x Location x Action template expansion (~2500)
  2. BCI patient need templates (~800)
  3. Paraphrase patterns on existing commands (~700)
  4. Daily activity / environment control templates (~500)
  5. Short conversational phrases (~500)

All output is filtered to A-Z only (keyboard constraint).
Deduplication at the end.

Usage:
    uv run python scripts/augment_spelling_corpus.py --output data/spelling_corpus_5k.json
"""

import argparse
import json
import random
from itertools import product

random.seed(42)

# ─── Vocabulary pools ───────────────────────────────────────

OBJECTS = [
    "apple", "bread", "cup", "bowl", "fork", "knife", "spoon", "plate",
    "mug", "pan", "pot", "box", "bottle", "can", "jar", "bag",
    "towel", "cloth", "sponge", "soap", "pillow", "blanket", "book",
    "pen", "pencil", "key", "phone", "remote", "lamp", "candle",
    "vase", "clock", "watch", "wallet", "card", "newspaper", "magazine",
    "egg", "tomato", "potato", "lettuce", "carrot", "onion", "pepper",
    "milk", "water", "juice", "coffee", "tea", "wine", "beer",
    "shirt", "hat", "shoe", "coat", "glove", "scarf", "ring",
    "ball", "bat", "toy", "doll", "game", "puzzle", "dice",
    "tissue", "paper", "tape", "string", "rope", "wire", "tool",
    "hammer", "drill", "brush", "comb", "mirror", "tray", "basket",
]

LOCATIONS = [
    "table", "desk", "counter", "shelf", "drawer", "cabinet", "closet",
    "fridge", "microwave", "oven", "stove", "sink", "bathtub", "toilet",
    "bed", "sofa", "couch", "chair", "armchair", "bench", "floor",
    "kitchen", "bedroom", "bathroom", "hallway", "garage", "garden",
    "dining room", "living room", "laundry room", "balcony", "porch",
    "trash can", "bin", "basket", "box", "tray", "rack", "hook",
    "window sill", "doorway", "corner", "nightstand", "dresser",
]

ACTIONS_PUT = [
    "put {obj} on the {loc}",
    "place {obj} on the {loc}",
    "move {obj} to the {loc}",
    "set {obj} on the {loc}",
    "lay {obj} on the {loc}",
    "drop {obj} on the {loc}",
    "leave {obj} on the {loc}",
    "put the {obj} in the {loc}",
    "place the {obj} inside the {loc}",
    "move the {obj} into the {loc}",
    "store the {obj} in the {loc}",
    "keep the {obj} in the {loc}",
]

ACTIONS_GIVE = [
    "give me the {obj}",
    "bring me the {obj}",
    "hand me the {obj}",
    "pass me the {obj}",
    "fetch me the {obj}",
    "get me the {obj}",
    "grab the {obj} for me",
    "I need the {obj}",
    "I want the {obj}",
    "can you bring me the {obj}",
    "please give me the {obj}",
    "could you get the {obj}",
    "would you hand me the {obj}",
    "bring the {obj} here",
    "get the {obj} please",
]

ACTIONS_GO = [
    "go to the {loc}",
    "head to the {loc}",
    "walk to the {loc}",
    "move to the {loc}",
    "go over to the {loc}",
    "navigate to the {loc}",
    "proceed to the {loc}",
    "please go to the {loc}",
    "take me to the {loc}",
    "go check the {loc}",
    "go near the {loc}",
    "walk over to the {loc}",
    "head over to the {loc}",
    "go towards the {loc}",
    "I want to go to the {loc}",
]

ACTIONS_COOK = [
    "heat the {obj}",
    "cook the {obj}",
    "warm up the {obj}",
    "boil the {obj}",
    "fry the {obj}",
    "bake the {obj}",
    "toast the {obj}",
    "microwave the {obj}",
    "grill the {obj}",
    "roast the {obj}",
]

ACTIONS_CLEAN = [
    "wash the {obj}",
    "clean the {obj}",
    "rinse the {obj}",
    "wipe the {obj}",
    "scrub the {obj}",
    "dry the {obj}",
    "polish the {obj}",
    "dust the {obj}",
]

ACTIONS_COOL = [
    "cool the {obj}",
    "chill the {obj}",
    "freeze the {obj}",
    "refrigerate the {obj}",
    "put the {obj} in the fridge",
    "cool down the {obj}",
]

COOKABLE = ["egg", "tomato", "potato", "bread", "apple", "lettuce", "carrot", "onion", "pepper"]
CLEANABLE = ["cup", "bowl", "fork", "knife", "spoon", "plate", "mug", "pan", "pot",
             "cloth", "towel", "sponge", "lettuce", "tomato", "potato", "apple"]
COOLABLE = ["egg", "tomato", "potato", "apple", "milk", "water", "juice", "wine", "beer", "bottle"]

# ─── BCI Patient Needs ──────────────────────────────────────

FEELINGS = ["cold", "hot", "tired", "hungry", "thirsty", "sick", "dizzy",
            "weak", "strong", "happy", "sad", "scared", "bored", "lonely",
            "warm", "sleepy", "angry", "calm", "nervous", "confused",
            "anxious", "restless", "peaceful", "grateful", "hopeful",
            "frustrated", "overwhelmed", "relaxed", "energetic", "numb",
            "itchy", "stiff", "sore", "cramped", "bloated", "nauseous"]

NEEDS = ["water", "food", "rest", "sleep", "help", "air", "light", "quiet",
         "music", "medicine", "blanket", "pillow", "drink", "company",
         "warmth", "shade", "space", "time", "break", "comfort",
         "attention", "care", "support", "patience", "silence",
         "fresh air", "cold water", "hot water", "warm milk", "ice",
         "pain relief", "eye drops", "ear plugs", "hand cream", "lip balm"]

BCI_TEMPLATES = [
    "I am {feeling}",
    "I feel {feeling}",
    "I am very {feeling}",
    "I feel so {feeling}",
    "I am a bit {feeling}",
    "I am really {feeling}",
    "still {feeling}",
    "getting {feeling}",
    "not {feeling} anymore",
    "I need {need}",
    "I want {need}",
    "please give me {need}",
    "can I have {need}",
    "more {need} please",
    "I would like {need}",
    "get me {need}",
    "bring me {need}",
    "I really need {need}",
    "can you get me {need}",
    "I am asking for {need}",
]

BCI_REQUESTS = [
    "call the doctor", "call the nurse", "call my family", "call my friend",
    "call my mother", "call my father", "call my wife", "call my husband",
    "open the door", "close the door", "open the window", "close the window",
    "turn on the light", "turn off the light", "turn on the fan", "turn off the fan",
    "turn on the tv", "turn off the tv", "turn up the volume", "turn down the volume",
    "come here please", "help me please", "hold my hand", "sit me up", "lay me down",
    "change the channel", "adjust the bed", "raise the bed", "lower the bed",
    "more please", "stop please", "wait please", "try again",
    "read to me", "talk to me", "sing to me", "play music",
    "check the time", "what day is it", "where am I", "who is here",
    "move me to the left", "move me to the right", "tilt me forward",
    "give me a hug", "scratch my nose", "fix my hair", "wipe my face",
    "I need oxygen", "check my blood", "take my pulse", "measure my temp",
    "I have a question", "I have something to say", "listen to me",
    "I want to go home", "when can I leave", "am I getting better",
    "is it morning", "is it night", "what time is it now",
    "who is visiting", "tell them I said hello", "ask them to come",
    "I am in pain", "my head hurts", "my back hurts", "my leg hurts",
    "my arm hurts", "my stomach hurts", "I feel numb", "I feel dizzy",
    "I cannot breathe well", "I feel pressure", "something is wrong",
]

# ─── Daily Activities ────────────────────────────────────────

DAILY_TEMPLATES = [
    "I want to {activity}",
    "can I {activity}",
    "please help me {activity}",
    "I need to {activity}",
    "let me {activity}",
    "time to {activity}",
]

ACTIVITIES = [
    "eat now", "drink water", "take a walk", "go outside", "watch tv",
    "read a book", "listen to music", "take a bath", "brush my teeth",
    "wash my face", "get dressed", "go to sleep", "wake up", "sit down",
    "stand up", "lie down", "roll over", "stretch", "exercise",
    "take medicine", "use the bathroom", "change clothes", "eat lunch",
    "eat dinner", "eat breakfast", "have a snack", "drink coffee",
    "drink tea", "drink juice", "have some milk", "rest now",
]

# ─── Conversational Phrases ─────────────────────────────────

GREETINGS = [
    "hello", "good morning", "good afternoon", "good evening", "good night",
    "how are you", "nice to see you", "welcome", "goodbye", "see you later",
    "take care", "have a good day", "have a nice day", "sleep well",
    "nice day today", "see you soon", "come again", "sweet dreams",
    "rise and shine", "wake up sleepy head", "hello there", "hey friend",
]

RESPONSES = [
    "yes", "no", "ok", "sure", "maybe", "not yet", "later", "now",
    "please", "thank you", "thanks", "sorry", "excuse me",
    "I agree", "I disagree", "that is fine", "not now", "go ahead",
    "wait a moment", "one more time", "say again", "I understand",
    "I do not understand", "repeat please", "speak louder",
    "yes please", "no thanks", "of course", "never mind", "forget it",
    "sounds good", "that works", "perfect", "exactly", "correct",
    "wrong", "not quite", "almost", "close enough", "try again",
    "absolutely", "definitely", "certainly", "no way", "impossible",
    "I think so", "I hope so", "I doubt it", "probably", "unlikely",
    "let me think", "give me a second", "hold on", "just a moment",
]

EMOTIONS = [
    "I love you", "I miss you", "I am sorry", "forgive me",
    "I am proud of you", "well done", "good job", "great work",
    "I am grateful", "you are kind", "be careful", "stay safe",
    "do not worry", "it is ok", "everything is fine", "I am here",
    "you are the best", "I believe in you", "keep going", "stay strong",
    "I appreciate it", "you make me happy", "I am lucky",
    "that means a lot", "I care about you", "you are important",
    "I trust you", "we can do this", "do not give up", "hang in there",
    "I am doing my best", "one step at a time", "today is a good day",
    "tomorrow will be better", "life is beautiful", "I am blessed",
]

# ─── Environment & Smart Home ───────────────────────────────

SMART_HOME = [
    "set alarm for seven", "set alarm for eight", "set timer for five minutes",
    "play my playlist", "skip this song", "pause the music", "resume playing",
    "dim the lights", "brighten the lights", "set lights to blue",
    "lock the door", "unlock the door", "check the locks",
    "what is the weather", "is it raining", "how hot is it outside",
    "order groceries", "order pizza", "order medicine",
    "send a message", "read my messages", "check my email",
    "take a photo", "record a video", "start recording",
    "make a note", "remind me later", "set a reminder",
    "search the web", "look this up", "find information about",
    "navigate home", "show me the map", "how far is it",
    "charge my phone", "connect to wifi", "turn on bluetooth",
    "increase the heat", "decrease the heat", "set temp to twenty",
    "start the vacuum", "water the plants", "feed the cat",
    "feed the dog", "check the baby", "monitor the room",
]

# ─── Complex multi-step tasks ───────────────────────────────

COMPLEX_TEMPLATES = [
    "pick up the {obj1} and put it on the {loc}",
    "move the {obj1} from the {loc1} to the {loc2}",
    "take the {obj1} to the {loc}",
    "bring the {obj1} from the {loc} to me",
    "put the {obj1} and the {obj2} on the {loc}",
    "clean the {obj1} and put it in the {loc}",
    "heat the {obj1} and place it on the {loc}",
    "wash the {obj1} and dry it",
    "find the {obj1} and bring it here",
    "grab the {obj1} from the {loc} please",
    "can you move the {obj1} next to the {obj2}",
    "swap the {obj1} and the {obj2}",
    "put the {obj1} inside the {obj2}",
    "take the {obj1} out of the {loc}",
    "check if the {obj1} is on the {loc}",
    "look for the {obj1} in the {loc}",
    "is the {obj1} on the {loc}",
    "where is the {obj1}",
    "find my {obj1}",
]


def to_alpha(s):
    """Strip everything except A-Z."""
    return "".join(c for c in s.upper() if c.isalpha())


def generate_put_commands(n=2500):
    """Generate put/place/move commands: action x object x location."""
    sentences = set()
    combos = list(product(ACTIONS_PUT, OBJECTS, LOCATIONS))
    random.shuffle(combos)
    for template, obj, loc in combos:
        s = template.format(obj=obj, loc=loc)
        alpha = to_alpha(s)
        if 5 <= len(alpha) <= 40:
            sentences.add(alpha)
        if len(sentences) >= n:
            break
    return list(sentences)


def generate_give_commands(n=500):
    """Generate give/bring/fetch commands."""
    sentences = set()
    combos = list(product(ACTIONS_GIVE, OBJECTS))
    random.shuffle(combos)
    for template, obj in combos:
        s = template.format(obj=obj)
        alpha = to_alpha(s)
        if 5 <= len(alpha) <= 35:
            sentences.add(alpha)
        if len(sentences) >= n:
            break
    return list(sentences)


def generate_go_commands(n=500):
    """Generate go/navigate commands."""
    sentences = set()
    combos = list(product(ACTIONS_GO, LOCATIONS))
    random.shuffle(combos)
    for template, loc in combos:
        s = template.format(loc=loc)
        alpha = to_alpha(s)
        if 5 <= len(alpha) <= 35:
            sentences.add(alpha)
        if len(sentences) >= n:
            break
    return list(sentences)


def generate_cook_clean_cool(n=500):
    """Generate cook/clean/cool commands."""
    sentences = set()
    for template in ACTIONS_COOK:
        for obj in COOKABLE:
            alpha = to_alpha(template.format(obj=obj))
            if 5 <= len(alpha) <= 30:
                sentences.add(alpha)
    for template in ACTIONS_CLEAN:
        for obj in CLEANABLE:
            alpha = to_alpha(template.format(obj=obj))
            if 5 <= len(alpha) <= 30:
                sentences.add(alpha)
    for template in ACTIONS_COOL:
        for obj in COOLABLE:
            alpha = to_alpha(template.format(obj=obj))
            if 5 <= len(alpha) <= 30:
                sentences.add(alpha)
    result = list(sentences)
    random.shuffle(result)
    return result[:n]


def generate_bci_needs(n=800):
    """Generate BCI patient need expressions."""
    sentences = set()
    # Feeling templates
    for template in BCI_TEMPLATES:
        if "{feeling}" in template:
            for feeling in FEELINGS:
                alpha = to_alpha(template.format(feeling=feeling))
                if 3 <= len(alpha) <= 25:
                    sentences.add(alpha)
        if "{need}" in template:
            for need in NEEDS:
                alpha = to_alpha(template.format(need=need))
                if 3 <= len(alpha) <= 25:
                    sentences.add(alpha)
    # Fixed requests
    for req in BCI_REQUESTS:
        alpha = to_alpha(req)
        if 3 <= len(alpha) <= 30:
            sentences.add(alpha)
    result = list(sentences)
    random.shuffle(result)
    return result[:n]


def generate_daily_activities(n=300):
    """Generate daily activity requests."""
    sentences = set()
    for template in DAILY_TEMPLATES:
        for activity in ACTIVITIES:
            alpha = to_alpha(template.format(activity=activity))
            if 5 <= len(alpha) <= 35:
                sentences.add(alpha)
    result = list(sentences)
    random.shuffle(result)
    return result[:n]


def generate_conversational(n=500):
    """Generate conversational phrases + smart home commands."""
    sentences = set()
    for phrase in GREETINGS + RESPONSES + EMOTIONS + SMART_HOME:
        alpha = to_alpha(phrase)
        if 2 <= len(alpha) <= 35:
            sentences.add(alpha)
    result = list(sentences)
    random.shuffle(result)
    return result[:n]


def generate_complex_tasks(n=500):
    """Generate multi-step task commands."""
    sentences = set()
    for template in COMPLEX_TEMPLATES:
        placeholders = template.count("{")
        for _ in range(50):
            objs = random.sample(OBJECTS, min(2, len(OBJECTS)))
            locs = random.sample(LOCATIONS, min(2, len(LOCATIONS)))
            try:
                s = template.format(
                    obj1=objs[0], obj2=objs[1] if len(objs) > 1 else objs[0],
                    loc=locs[0], loc1=locs[0], loc2=locs[1] if len(locs) > 1 else locs[0],
                )
            except (KeyError, IndexError):
                continue
            alpha = to_alpha(s)
            if 8 <= len(alpha) <= 45:
                sentences.add(alpha)
        if len(sentences) >= n:
            break
    result = list(sentences)
    random.shuffle(result)
    return result[:n]


def main():
    parser = argparse.ArgumentParser(description="Augment spelling corpus to ~5k")
    parser.add_argument("--output", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--target", type=int, default=5000, help="Target number of sentences")
    args = parser.parse_args()

    print("Generating augmented spelling corpus...")

    # Generate each category
    put_cmds = generate_put_commands(2300)
    print(f"  Put/place/move commands: {len(put_cmds)}")

    give_cmds = generate_give_commands(500)
    print(f"  Give/bring/fetch commands: {len(give_cmds)}")

    go_cmds = generate_go_commands(500)
    print(f"  Go/navigate commands: {len(go_cmds)}")

    cook_clean = generate_cook_clean_cool(400)
    print(f"  Cook/clean/cool commands: {len(cook_clean)}")

    bci_needs = generate_bci_needs(600)
    print(f"  BCI patient needs: {len(bci_needs)}")

    daily = generate_daily_activities(300)
    print(f"  Daily activities: {len(daily)}")

    conv = generate_conversational(500)
    print(f"  Conversational + smart home: {len(conv)}")

    complex_tasks = generate_complex_tasks(500)
    print(f"  Complex multi-step tasks: {len(complex_tasks)}")

    # Merge and deduplicate
    all_sentences = list(set(
        put_cmds + give_cmds + go_cmds + cook_clean +
        bci_needs + daily + conv + complex_tasks
    ))
    random.shuffle(all_sentences)

    # Length statistics
    lengths = [len(s) for s in all_sentences]
    avg_len = sum(lengths) / len(lengths)
    print(f"\nTotal unique sentences: {len(all_sentences)}")
    print(f"Length: min={min(lengths)}, max={max(lengths)}, avg={avg_len:.1f}")

    # Trim to target if needed
    if len(all_sentences) > args.target:
        all_sentences = all_sentences[:args.target]
        print(f"Trimmed to {args.target}")

    # Build output with category labels
    output = {
        "sentences": all_sentences,
        "stats": {
            "total": len(all_sentences),
            "avg_length": round(avg_len, 1),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "categories": {
                "put_place_move": len(put_cmds),
                "give_bring_fetch": len(give_cmds),
                "go_navigate": len(go_cmds),
                "cook_clean_cool": len(cook_clean),
                "bci_patient_needs": len(bci_needs),
                "daily_activities": len(daily),
                "conversational": len(conv),
                "complex_tasks": len(complex_tasks),
            }
        }
    }

    # Save
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Preview
    print("\nSample sentences:")
    for s in random.sample(all_sentences, min(15, len(all_sentences))):
        print(f"  {s} ({len(s)} chars)")


if __name__ == "__main__":
    main()
