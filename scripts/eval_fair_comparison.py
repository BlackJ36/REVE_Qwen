"""Fair comparison: FBCCA vs S2 model on multi-character spelling.

Uses val subjects' real EEG trials to construct spelling sequences,
then evaluates both FBCCA (independent per-character) and S2 model
(with language prior) on the exact same trials.

Usage:
    # FBCCA only (no GPU needed for model):
    uv run python scripts/eval_fair_comparison.py --fbcca_only

    # Full comparison:
    uv run python scripts/eval_fair_comparison.py \
        --s2_checkpoint output/s2/final \
        --trial_pts 200 --n_words 50
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fbcca import FBCCAFeatureExtractor, resolve_channel_indices
from src.metrics_s2 import edit_distance
from src.preprocess import VALID_CHANNEL_NAMES

KEYBOARD_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z", "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "0", "_", ".", "<", ">",
]
CHAR_TO_LABEL = {ch: i for i, ch in enumerate(KEYBOARD_CHARS)}
LATENCY_PTS = 28  # 0.14s cue period @ 200Hz


# ─── Data loading ────────────────────────────────────────────

def load_val_eeg(eeg_dir):
    """Load val EEG data and organize by (subject, label)."""
    path = Path(eeg_dir) / "val_eeg.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)

    eeg = data["eeg_data"]        # (N, 62, 600)
    labels = data["labels"]        # (N,)
    sids = data["subject_ids"]     # (N,)
    valid_pts = data["valid_pts"]  # (N,)

    # Index: (subject_id, label) -> list of trial indices
    trial_index = defaultdict(list)
    for i in range(len(labels)):
        trial_index[(int(sids[i]), int(labels[i]))].append(i)

    subjects = sorted(set(int(s) for s in sids.tolist()))
    print(f"Val EEG: {len(labels)} trials, {len(subjects)} subjects: {subjects}")
    return eeg, labels, sids, valid_pts, trial_index, subjects


def load_corpus_words(corpus_path, max_len=8):
    """Load spelling corpus and filter to A-Z words within max_len."""
    with open(corpus_path) as f:
        corpus = json.load(f)
    words = corpus.get("sentences", corpus.get("words", []))
    alpha_words = [w for w in words if w.isalpha() and w == w.upper()
                   and len(w) <= max_len and all(c in CHAR_TO_LABEL for c in w)]
    print(f"Corpus: {len(words)} total, {len(alpha_words)} A-Z words (len<={max_len})")
    return alpha_words


def select_eval_words(corpus_words, trial_index, subjects, n_words, seed=42):
    """Select words that can be spelled by each val subject.

    For each word, every character must have at least 1 trial available
    for the given subject.
    """
    rng = random.Random(seed)

    eval_items = []  # (subject_id, word, trial_indices)
    for sid in subjects:
        available_labels = {label for (s, label) in trial_index if s == sid}
        valid_words = [w for w in corpus_words
                       if all(CHAR_TO_LABEL[c] in available_labels for c in w)]
        if not valid_words:
            print(f"  S{sid:02d}: no valid words, skipping")
            continue

        selected = rng.sample(valid_words, min(n_words, len(valid_words)))
        for word in selected:
            indices = []
            for ch in word:
                label = CHAR_TO_LABEL[ch]
                candidates = trial_index[(sid, label)]
                indices.append(rng.choice(candidates))
            eval_items.append((sid, word, indices))

    print(f"Eval set: {len(eval_items)} word instances across {len(subjects)} subjects")
    return eval_items


# ─── FBCCA evaluation ───────────────────────────────────────

def run_fbcca(eeg_data, valid_pts_all, eval_items, trial_pts, ch_idx):
    """Run FBCCA independently on each character trial."""
    fbcca = FBCCAFeatureExtractor(sfreq=200.0, n_timepoints=trial_pts)

    results = []
    for sid, word, trial_indices in eval_items:
        pred_chars = []
        for idx in trial_indices:
            vp = int(valid_pts_all[idx])
            actual_pts = min(trial_pts, vp - LATENCY_PTS)
            if actual_pts < trial_pts:
                eeg_trial = torch.zeros(1, len(ch_idx), trial_pts)
                eeg_trial[:, :, :actual_pts] = eeg_data[idx:idx+1, ch_idx,
                                                         LATENCY_PTS:LATENCY_PTS + actual_pts]
            else:
                eeg_trial = eeg_data[idx:idx+1, ch_idx,
                                     LATENCY_PTS:LATENCY_PTS + trial_pts]

            with torch.no_grad():
                corr = fbcca(eeg_trial)  # (1, 200)
                corr_5x40 = corr.reshape(1, 5, 40)
                weights = fbcca.band_weights.unsqueeze(0).unsqueeze(-1)
                scores = (corr_5x40 * weights).sum(dim=1)  # (1, 40)
                pred_label = scores.argmax(dim=-1).item()

            pred_chars.append(KEYBOARD_CHARS[pred_label])

        results.append({
            "subject": sid,
            "target": word,
            "prediction": "".join(pred_chars),
        })

    return results


# ─── S2 model evaluation ────────────────────────────────────

def load_s2_model(checkpoint_dir, model_name, from_modelscope):
    """Load trained S2 model directly from checkpoint (no double-LoRA).

    S2 checkpoint contains: adapter_config.json + adapter_model.safetensors
    (LoRA + modules_to_save for embed_tokens/lm_head) + projector.pt.
    We load base Qwen → apply S2 adapter → load projector.
    """
    from peft import PeftModel as PeftModelClass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.model import BCIQwenModel, EEGProjector, _get_llm_dim
    from src.tokens import register_special_tokens

    ckpt_dir = Path(checkpoint_dir)

    # Load base Qwen
    if from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_name)
    else:
        model_path = model_name

    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    num_new = register_special_tokens(tokenizer)

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    qwen_model.resize_token_embeddings(len(tokenizer))

    llm_dim = _get_llm_dim(qwen_model)
    original_vocab_size = len(tokenizer) - num_new

    # Load S2 LoRA adapter (includes modules_to_save for embed_tokens/lm_head)
    if (ckpt_dir / "adapter_config.json").exists():
        print(f"Loading S2 adapter from {ckpt_dir}")
        qwen_model = PeftModelClass.from_pretrained(
            qwen_model, str(ckpt_dir), is_trainable=False,
        )

    # Load projector
    projector = EEGProjector(reve_dim=512, qwen_dim=llm_dim)
    proj_path = ckpt_dir / "projector.pt"
    if proj_path.exists():
        projector.load_state_dict(
            torch.load(proj_path, map_location="cpu", weights_only=True)
        )
        print(f"Loaded projector from {proj_path}")

    model = BCIQwenModel(qwen_model, tokenizer, projector, original_vocab_size)
    model.requires_grad_(False)
    return model, tokenizer


def run_s2_model(model, tokenizer, eval_items, val_emb_path, device):
    """Run S2 model on eval items using generate()."""
    from src.tokens import BCI_END, BCI_PAD, BCI_SEP, BCI_START

    # Load val embeddings
    emb_data = torch.load(val_emb_path, map_location="cpu", weights_only=True)
    emb_bank = emb_data["embeddings"]  # (N, T, 512)
    n_eeg_tokens = int(emb_data.get("n_eeg_tokens", emb_bank.shape[1]))
    print(f"Val embedding bank: {emb_bank.shape}, n_eeg_tokens={n_eeg_tokens}")

    # Pre-tokenize fixed parts
    bci_start = tokenizer.encode(BCI_START, add_special_tokens=False)
    bci_end = tokenizer.encode(BCI_END, add_special_tokens=False)
    bci_sep = tokenizer.encode(BCI_SEP, add_special_tokens=False)
    bci_pad = tokenizer.encode(BCI_PAD, add_special_tokens=False)
    bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    system_text = ("你是一个脑机接口助手。用户通过注视屏幕上闪烁的字符来拼写文字。"
                   "请根据脑电信号识别用户想要拼写的内容。")

    results = []
    model.eval()

    for item_idx, (sid, word, trial_indices) in enumerate(eval_items):
        n_chars = len(word)

        # Get embeddings for these specific trials
        char_embs = [emb_bank[idx] for idx in trial_indices]  # list of (T, 512)
        eeg_embeddings = torch.stack(char_embs)  # (n_chars, T, 512)

        # Build input token IDs
        system_ids = tokenizer.encode(
            f"<|im_start|>system\n{system_text}<|im_end|>\n",
            add_special_tokens=False,
        )
        user_header = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        user_footer = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        asst_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        eeg_ids = list(bci_start)
        for i in range(n_chars):
            eeg_ids.extend(bci_pad * n_eeg_tokens)
            if i < n_chars - 1:
                eeg_ids.extend(bci_sep)
        eeg_ids.extend(bci_end)

        input_ids = system_ids + user_header + eeg_ids + user_footer + asst_header
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids)

        # Embed and inject EEG
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()

        projected = model.projector(eeg_embeddings.to(device).float())
        flat_tokens = projected.reshape(-1, projected.size(-1))  # (n_chars*T, dim)

        pad_positions = (input_ids[0] == bci_pad_id).nonzero(as_tuple=True)[0]
        n = min(len(pad_positions), flat_tokens.size(0))
        inputs_embeds[0, pad_positions[:n]] = flat_tokens[:n].to(inputs_embeds.dtype)

        # Generate
        with torch.no_grad():
            output_ids = model.qwen.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=len(word) * 3 + 50,
                do_sample=False,
                eos_token_id=im_end_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract predicted word (first uppercase block)
        m = re.match(r"([A-Z0-9_.]+)", generated_text)
        pred_word = m.group(1) if m else ""

        results.append({
            "subject": sid,
            "target": word,
            "prediction": pred_word,
            "generated_text": generated_text,
        })

        if (item_idx + 1) % 50 == 0:
            print(f"  [{item_idx+1}/{len(eval_items)}] S{sid:02d} "
                  f"target={word} pred={pred_word}")

    return results


# ─── Metrics ─────────────────────────────────────────────────

def compute_metrics(results, label=""):
    """Compute char_acc, word_acc, avg edit distance."""
    if not results:
        return {}

    char_correct, char_total = 0, 0
    word_correct = 0
    total_ed = 0
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0,
                                        "words": 0, "word_correct": 0})

    for r in results:
        target = r["target"]
        pred = r["prediction"]
        sid = r["subject"]

        n = max(len(target), len(pred))
        c = sum(1 for a, b in zip(target, pred) if a == b)
        char_correct += c
        char_total += n

        if pred == target:
            word_correct += 1

        total_ed += edit_distance(pred, target)

        per_subject[sid]["correct"] += c
        per_subject[sid]["total"] += n
        per_subject[sid]["words"] += 1
        per_subject[sid]["word_correct"] += (1 if pred == target else 0)

    metrics = {
        f"{label}char_acc": char_correct / max(char_total, 1),
        f"{label}word_acc": word_correct / len(results),
        f"{label}avg_ed": total_ed / len(results),
        f"{label}count": len(results),
    }

    subject_accs = []
    for sid in sorted(per_subject):
        s = per_subject[sid]
        s_char_acc = s["correct"] / max(s["total"], 1)
        s_word_acc = s["word_correct"] / max(s["words"], 1)
        subject_accs.append(s_char_acc)
        metrics[f"{label}S{sid:02d}_char"] = s_char_acc
        metrics[f"{label}S{sid:02d}_word"] = s_word_acc

    if subject_accs:
        metrics[f"{label}char_acc_std"] = float(torch.tensor(subject_accs).std())

    return metrics


def compute_correction_rate(fbcca_results, s2_results):
    """How often S2 corrects FBCCA errors / trusts FBCCA correct answers."""
    fbcca_wrong_s2_right = 0
    fbcca_wrong_total = 0
    fbcca_right_s2_right = 0
    fbcca_right_total = 0

    for fb, s2 in zip(fbcca_results, s2_results):
        target = fb["target"]
        for fb_ch, s2_ch, tgt_ch in zip(fb["prediction"], s2["prediction"], target):
            if fb_ch != tgt_ch:
                fbcca_wrong_total += 1
                if s2_ch == tgt_ch:
                    fbcca_wrong_s2_right += 1
            else:
                fbcca_right_total += 1
                if s2_ch == tgt_ch:
                    fbcca_right_s2_right += 1

    return {
        "correction_rate": fbcca_wrong_s2_right / max(fbcca_wrong_total, 1),
        "fbcca_wrong_count": fbcca_wrong_total,
        "trust_rate": fbcca_right_s2_right / max(fbcca_right_total, 1),
        "fbcca_right_count": fbcca_right_total,
    }


# ─── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fair FBCCA vs S2 model comparison")
    parser.add_argument("--s2_checkpoint", type=str, default=None,
                        help="S2 model checkpoint dir (skip model if None)")
    parser.add_argument("--embedding_dir", type=str, default="data/embeddings")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--corpus", type=str, default="data/spelling_corpus_5k.json")
    parser.add_argument("--trial_pts", type=int, default=200)
    parser.add_argument("--n_words", type=int, default=50)
    parser.add_argument("--max_word_len", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--no_modelscope", dest="from_modelscope", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fbcca_only", action="store_true",
                        help="Only run FBCCA baseline")
    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("Loading val EEG data...")
    eeg_data, labels, sids, valid_pts, trial_index, subjects = load_val_eeg(args.eeg_dir)
    corpus_words = load_corpus_words(args.corpus, max_len=args.max_word_len)

    print("\nSelecting evaluation words...")
    eval_items = select_eval_words(corpus_words, trial_index, subjects, args.n_words)

    # ─── FBCCA ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FBCCA ({args.trial_pts}pts = {args.trial_pts/200:.1f}s)")
    print("=" * 60)
    ch_idx = resolve_channel_indices(VALID_CHANNEL_NAMES)
    fbcca_results = run_fbcca(eeg_data, valid_pts, eval_items, args.trial_pts, ch_idx)
    fbcca_metrics = compute_metrics(fbcca_results, label="fbcca_")

    print(f"\n  Char Acc:  {fbcca_metrics['fbcca_char_acc']:.1%}")
    print(f"  Word Acc:  {fbcca_metrics['fbcca_word_acc']:.1%}")
    print(f"  Avg ED:    {fbcca_metrics['fbcca_avg_ed']:.2f}")
    print(f"\n  Per-subject:")
    for sid in subjects:
        ck = f"fbcca_S{sid:02d}_char"
        wk = f"fbcca_S{sid:02d}_word"
        if ck in fbcca_metrics:
            print(f"    S{sid:02d}: char={fbcca_metrics[ck]:.1%}  word={fbcca_metrics[wk]:.1%}")

    if args.fbcca_only:
        if args.output:
            Path(args.output).write_text(json.dumps(
                {"fbcca": fbcca_metrics, "n_items": len(eval_items)},
                indent=2, ensure_ascii=False))
            print(f"\nSaved to {args.output}")
        return

    # ─── S2 Model ────────────────────────────────────────────
    if args.s2_checkpoint is None:
        print("\nNo --s2_checkpoint, skipping model evaluation.")
        return

    print(f"\n{'='*60}")
    print("S2 Model")
    print("=" * 60)
    device = torch.device(args.device)

    print("Loading model...")
    model, tokenizer = load_s2_model(args.s2_checkpoint, args.model_name,
                                      args.from_modelscope)
    model = model.to(device)

    val_emb_path = str(Path(args.embedding_dir) / "val_embeddings.pt")
    print(f"\nInference on {len(eval_items)} words...")
    s2_results = run_s2_model(model, tokenizer, eval_items, val_emb_path, device)
    s2_metrics = compute_metrics(s2_results, label="s2_")

    print(f"\n  Char Acc:  {s2_metrics['s2_char_acc']:.1%}")
    print(f"  Word Acc:  {s2_metrics['s2_word_acc']:.1%}")
    print(f"  Avg ED:    {s2_metrics['s2_avg_ed']:.2f}")
    print(f"\n  Per-subject:")
    for sid in subjects:
        ck = f"s2_S{sid:02d}_char"
        wk = f"s2_S{sid:02d}_word"
        if ck in s2_metrics:
            print(f"    S{sid:02d}: char={s2_metrics[ck]:.1%}  word={s2_metrics[wk]:.1%}")

    # ─── Correction analysis ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Correction Analysis")
    print("=" * 60)
    corr = compute_correction_rate(fbcca_results, s2_results)
    print(f"  Correction rate: {corr['correction_rate']:.1%} "
          f"(FBCCA wrong: {corr['fbcca_wrong_count']})")
    print(f"  Trust rate:      {corr['trust_rate']:.1%} "
          f"(FBCCA right: {corr['fbcca_right_count']})")

    # ─── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'':20s} {'FBCCA':>10s} {'S2 Model':>10s} {'Delta':>10s}")
    print(f"  {'─'*50}")
    for m in ["char_acc", "word_acc", "avg_ed"]:
        fb = fbcca_metrics.get(f"fbcca_{m}", 0)
        s2 = s2_metrics.get(f"s2_{m}", 0)
        d = s2 - fb
        if "acc" in m:
            print(f"  {m:20s} {fb:>9.1%} {s2:>9.1%} {d:>+9.1%}")
        else:
            print(f"  {m:20s} {fb:>10.2f} {s2:>10.2f} {d:>+10.2f}")

    # Examples
    print(f"\n{'='*60}")
    print("Examples (first 20)")
    print("=" * 60)
    for fb, s2 in list(zip(fbcca_results, s2_results))[:20]:
        t = fb["target"]
        fp = fb["prediction"]
        sp = s2["prediction"]
        print(f"  S{fb['subject']:02d} {t:>8s} | FBCCA: {fp:>8s} {'OK' if fp==t else 'XX'} "
              f"| S2: {sp:>8s} {'OK' if sp==t else 'XX'}")

    if args.output:
        output = {
            "fbcca": fbcca_metrics, "s2": s2_metrics, "correction": corr,
            "config": vars(args), "n_items": len(eval_items),
        }
        Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
