"""Interactive inference demo for candidate-mode BCI agent.

Simulates real-time spelling by feeding EEG trials one-by-one
and showing model predictions with FBCCA context.

Usage:
    python scripts/demo_inference.py \
        --checkpoint output_ablation_reve_candidate_s2/final \
        --s1_checkpoint output_ablation_reve_candidate_s1/best \
        --eeg_dir data/eeg_tensors \
        --word HELLO

    # Random trials from validation set
    python scripts/demo_inference.py \
        --checkpoint output_ablation_reve_candidate_s2/final \
        --s1_checkpoint output_ablation_reve_candidate_s1/best \
        --n_trials 10
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_bci_agent import BETA_BAD_SUBJECTS
from src.dataset_bci_candidate import CandidateStage1Dataset
from src.dataset_bci_agent import BCIAgentCollator
from src.tokens import (
    TARGET_INDEX_TO_TOKEN, BCI_PAD,
    RANK1, RANK2, RANK3, CONF_HIGH, CONF_MID, CONF_LOW,
    score_gap_to_conf_token,
)
from src.templates_zh import KEYBOARD_CHARS
from src.word_vocab import CHAR_TO_LABEL

# Reuse model loading from evaluate script
from evaluate_candidate import load_model_for_inference


def spell_word(model, tokenizer, dataset, eeg_data, fbcca_indices, fbcca_scores,
               labels, word, device):
    """Simulate spelling a word character by character."""
    target_token_ids = {
        i: tokenizer.convert_tokens_to_ids(tok)
        for i, tok in TARGET_INDEX_TO_TOKEN.items()
    }
    target_id_tensor = torch.tensor(list(target_token_ids.values()), device=device)

    collator = BCIAgentCollator(tokenizer)

    print(f"\n{'=' * 50}")
    print(f"  Spelling: {word}")
    print(f"{'=' * 50}")

    spelled = ""
    correct_count = 0

    for ci, char in enumerate(word.upper()):
        if char not in CHAR_TO_LABEL:
            print(f"  [{ci+1}] '{char}' not in keyboard, skipping")
            continue

        target_label = CHAR_TO_LABEL[char]

        # Find a trial with this label
        matching = (labels == target_label).nonzero(as_tuple=True)[0]
        if len(matching) == 0:
            print(f"  [{ci+1}] No trial found for '{char}' (label={target_label})")
            continue

        trial_idx = matching[random.randrange(len(matching))].item()
        offset_idx = 0
        offset = dataset.window_offsets[offset_idx]
        window = eeg_data[trial_idx, :, offset:offset + 300]

        top3_idx = fbcca_indices[trial_idx, offset_idx].tolist()
        top3_sc = fbcca_scores[trial_idx, offset_idx].tolist()

        # Build sequence
        input_ids, label_ids = dataset._build_sequence(
            [target_label], [(top3_idx, top3_sc)]
        )

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "eeg_windows": window.unsqueeze(0).float(),
            "num_spells": 1,
        }
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            eeg_windows = batch.pop("eeg_windows", None)
            window_counts = batch.pop("window_counts", None)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                eeg_windows=eeg_windows,
                window_counts=window_counts,
            )

            logits = outputs.logits

            # Find target position
            label_row = batch["labels"][0]
            target_id_to_label = {v: k for k, v in target_token_ids.items()}
            target_positions = [
                pos for pos in range(len(label_row))
                if label_row[pos].item() in target_id_to_label
            ]

            if target_positions:
                pos = target_positions[0]
                target_logits = logits[0, pos - 1, target_id_tensor]
                probs = F.softmax(target_logits, dim=0)
                pred_label = probs.argmax().item()
                pred_char = KEYBOARD_CHARS[pred_label]
                confidence = probs.max().item()
            else:
                pred_char = "?"
                pred_label = -1
                confidence = 0

        # FBCCA info
        fbcca_chars = [KEYBOARD_CHARS[i] for i in top3_idx]
        conf_token = score_gap_to_conf_token(top3_sc[0], top3_sc[1])
        conf_str = {CONF_HIGH: "HIGH", CONF_MID: "MID", CONF_LOW: "LOW"}[conf_token]

        is_correct = pred_label == target_label
        if is_correct:
            correct_count += 1
        spelled += pred_char

        status = "OK" if is_correct else "MISS"
        fbcca_correct = top3_idx[0] == target_label
        if is_correct and not fbcca_correct:
            status = "FIXED!"

        print(f"\n  [{ci+1}] Target: '{char}'  |  FBCCA: {fbcca_chars[0]}>{fbcca_chars[1]}>{fbcca_chars[2]} ({conf_str})")
        print(f"      Model prediction: '{pred_char}' (conf={confidence:.2f})  [{status}]")
        print(f"      Spelled so far: \"{spelled}\"")

    print(f"\n{'=' * 50}")
    print(f"  Target:  {word.upper()}")
    print(f"  Spelled: {spelled}")
    print(f"  Accuracy: {correct_count}/{len(word)} ({correct_count/max(len(word),1):.0%})")
    print(f"{'=' * 50}")


def random_trials(model, tokenizer, dataset, eeg_data, fbcca_indices, fbcca_scores,
                  labels, subject_ids, n_trials, device):
    """Evaluate on random trials from validation set."""
    target_token_ids = {
        i: tokenizer.convert_tokens_to_ids(tok)
        for i, tok in TARGET_INDEX_TO_TOKEN.items()
    }
    target_id_tensor = torch.tensor(list(target_token_ids.values()), device=device)
    target_id_to_label = {v: k for k, v in target_token_ids.items()}

    collator = BCIAgentCollator(tokenizer)

    N = len(labels)
    indices = random.sample(range(N), min(n_trials, N))

    print(f"\n{'=' * 70}")
    print(f"  Random {len(indices)} trials from validation set")
    print(f"{'=' * 70}")
    print(f"{'#':>4} {'Subj':>5} {'True':>5} {'Model':>6} {'FBCCA':>6} {'Conf':>6} {'Score':>6} {'Result':>8}")

    correct = 0
    fbcca_correct_count = 0

    for rank, trial_idx in enumerate(indices):
        label = int(labels[trial_idx])
        sid = int(subject_ids[trial_idx])
        offset_idx = 0
        offset = dataset.window_offsets[offset_idx]
        window = eeg_data[trial_idx, :, offset:offset + 300]

        top3_idx = fbcca_indices[trial_idx, offset_idx].tolist()
        top3_sc = fbcca_scores[trial_idx, offset_idx].tolist()

        input_ids, label_ids = dataset._build_sequence(
            [label], [(top3_idx, top3_sc)]
        )

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "eeg_windows": window.unsqueeze(0).float(),
            "num_spells": 1,
        }
        batch = collator([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            eeg_windows = batch.pop("eeg_windows", None)
            window_counts = batch.pop("window_counts", None)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                eeg_windows=eeg_windows,
                window_counts=window_counts,
            )

            logits = outputs.logits
            label_row = batch["labels"][0]
            target_positions = [
                pos for pos in range(len(label_row))
                if label_row[pos].item() in target_id_to_label
            ]

            if target_positions:
                pos = target_positions[0]
                target_logits = logits[0, pos - 1, target_id_tensor]
                probs = F.softmax(target_logits, dim=0)
                pred_label = probs.argmax().item()
                confidence = probs.max().item()
            else:
                pred_label = -1
                confidence = 0

        true_char = KEYBOARD_CHARS[label]
        pred_char = KEYBOARD_CHARS[pred_label] if 0 <= pred_label < 40 else "?"
        fbcca_char = KEYBOARD_CHARS[top3_idx[0]]

        is_correct = pred_label == label
        fbcca_hit = top3_idx[0] == label
        if is_correct:
            correct += 1
        if fbcca_hit:
            fbcca_correct_count += 1

        result = "OK" if is_correct else "MISS"
        if is_correct and not fbcca_hit:
            result = "FIXED!"

        print(f"  {rank+1:3d}  S{sid:02d}    {true_char:>4}   {pred_char:>5}   {fbcca_char:>5}   "
              f"{confidence:>5.2f}   {top3_sc[0]:>5.2f}   {result:>7}")

    print(f"\n  Model accuracy: {correct}/{len(indices)} ({correct/len(indices):.1%})")
    print(f"  FBCCA accuracy: {fbcca_correct_count}/{len(indices)} ({fbcca_correct_count/len(indices):.1%})")


def main():
    parser = argparse.ArgumentParser(description="Inference demo for BCI agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--s1_checkpoint", type=str, required=True)
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--encoder_type", type=str, default="reve")
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--word", type=str, default=None,
                        help="Word to spell (e.g. HELLO)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random trials (when --word not set)")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", action="store_true")
    args = parser.parse_args()
    if args.no_modelscope:
        args.from_modelscope = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, tokenizer = load_model_for_inference(
        checkpoint_dir=args.checkpoint,
        s1_checkpoint=args.s1_checkpoint,
        model_name=args.model_name,
        encoder_type=args.encoder_type,
        reve_dir=args.reve_dir,
        device=device,
        from_modelscope=args.from_modelscope,
    )

    # Load validation data
    exclude_subjects = BETA_BAD_SUBJECTS
    data = torch.load(Path(args.eeg_dir) / "val_eeg.pt", weights_only=True)
    fbcca_data = torch.load(Path(args.eeg_dir) / "val_fbcca.pt", weights_only=True)

    mask = torch.ones(len(data["labels"]), dtype=torch.bool)
    for sid in exclude_subjects:
        mask &= data["subject_ids"] != sid

    eeg_data = data["eeg_data"][mask]
    labels_t = data["labels"][mask]
    subject_ids = data["subject_ids"][mask]
    fbcca_indices = fbcca_data["top3_indices"][mask]
    fbcca_scores = fbcca_data["top3_scores"][mask]

    # Build dataset just for tokenization helpers
    dataset = CandidateStage1Dataset(
        eeg_dir=args.eeg_dir,
        tokenizer=tokenizer,
        split="val",
        min_spells=1, max_spells=1,
        exclude_subjects=exclude_subjects,
    )

    if args.word:
        spell_word(model, tokenizer, dataset, eeg_data, fbcca_indices,
                   fbcca_scores, labels_t, args.word, device)
    else:
        random_trials(model, tokenizer, dataset, eeg_data, fbcca_indices,
                      fbcca_scores, labels_t, subject_ids, args.n_trials, device)


if __name__ == "__main__":
    main()
