"""Offline evaluation of candidate-mode BCI agent.

Loads a trained S2 checkpoint, runs on validation set trial-by-trial,
and computes FBCCA correction/trust metrics.

Usage:
    python scripts/evaluate_candidate.py \
        --checkpoint output_ablation_reve_candidate_s2/final \
        --s1_checkpoint output_ablation_reve_candidate_s1/best \
        --eeg_dir data/eeg_tensors

    # With specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_candidate.py ...
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_bci_candidate import CandidateStage1Dataset
from src.dataset_bci_agent import BCIAgentCollator, BETA_BAD_SUBJECTS
from src.metrics_bci_agent import compute_fbcca_correction_metrics
from src.model_bci_agent import build_bci_agent_model
from src.tokens import TARGET_INDEX_TO_TOKEN
from src.templates_zh import KEYBOARD_CHARS


def load_model_for_inference(
    checkpoint_dir,
    s1_checkpoint,
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    encoder_type="reve",
    fbcca_mode="candidate",
    reve_dir="models",
    window_size=300,
    device="cuda",
    from_modelscope=True,
):
    """Load a trained S2 model for inference.

    Strategy:
      1. Build base model (stage=1) — no LoRA, just Qwen + encoder
      2. Load S1 encoder + embedding weights
      3. Apply trained LoRA from S2 checkpoint via PeftModel.from_pretrained
      4. Override encoder with S2 weights (further trained in stage 2)
      5. Merge LoRA for faster inference
    """
    from peft import PeftModel as PeftModelClass

    checkpoint_dir = Path(checkpoint_dir)
    s1_dir = Path(s1_checkpoint)

    # Step 1: Build base model (stage=1 = no LoRA)
    model, tokenizer = build_bci_agent_model(
        model_name=model_name,
        from_modelscope=from_modelscope,
        reve_dir=reve_dir,
        stage=1,  # Base model without LoRA
        encoder_type=encoder_type,
        fbcca_mode=fbcca_mode,
        window_size=window_size,
    )

    # Step 2: Load S1 encoder weights
    enc_path = s1_dir / "encoder_trainable.pt"
    if enc_path.exists():
        state_dict = torch.load(enc_path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded S1 encoder from {enc_path}")

    # Load S1 Qwen weights (LoRA or embeddings)
    if (s1_dir / "adapter_config.json").exists():
        # S1 used LoRA — load and merge
        model.qwen = PeftModelClass.from_pretrained(model.qwen, str(s1_dir))
        model.qwen = model.qwen.merge_and_unload()
        print(f"Loaded and merged S1 LoRA from {s1_dir}")
    else:
        # Legacy: load qwen_trainable.pt
        qwen_path = s1_dir / "qwen_trainable.pt"
        if qwen_path.exists():
            qwen_state = torch.load(qwen_path, map_location="cpu", weights_only=True)
            ovs = model.original_vocab_size
            if "embed_tokens.new_rows" in qwen_state:
                model.qwen.get_input_embeddings().weight.data[ovs:] = qwen_state["embed_tokens.new_rows"]
                print(f"  Restored {qwen_state['embed_tokens.new_rows'].shape[0]} new token embeddings")
            if "lm_head.new_rows" in qwen_state:
                model.qwen.lm_head.weight.data[ovs:] = qwen_state["lm_head.new_rows"]
                print(f"  Restored lm_head new token rows")

    # Step 3: Apply trained LoRA from S2 checkpoint
    if (checkpoint_dir / "adapter_config.json").exists():
        print(f"Loading S2 LoRA from {checkpoint_dir}")
        model.qwen = PeftModelClass.from_pretrained(
            model.qwen, str(checkpoint_dir),
        )
        # Step 5: Merge LoRA for faster inference
        model.qwen = model.qwen.merge_and_unload()
        print("Merged LoRA weights into base model")
    else:
        print(f"WARNING: No adapter_config.json in {checkpoint_dir}")

    # Step 4: Override encoder with S2 weights (if available)
    s2_enc_path = checkpoint_dir / "encoder_trainable.pt"
    if s2_enc_path.exists():
        state_dict = torch.load(s2_enc_path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded S2 encoder from {s2_enc_path}")

    model = model.to(device)
    model.eval()
    return model, tokenizer


def run_evaluation(model, tokenizer, eeg_dir, device, exclude_bad=True, batch_size=8,
                   trial_duration=3.0, decoder_type="fbcca"):
    """Run evaluation on validation set and collect per-trial predictions.

    Args:
        trial_duration: trial duration in seconds (affects EEG truncation and decoder file).
        decoder_type: "fbcca", "trca", or "etrca" — which precomputed candidates to use.
    """
    exclude_subjects = BETA_BAD_SUBJECTS if exclude_bad else None
    trial_duration_pts = int(trial_duration * 200)
    effective_window_size = min(300, trial_duration_pts)

    # Load dataset (single-spell for clean per-trial evaluation)
    dataset = CandidateStage1Dataset(
        eeg_dir=eeg_dir,
        tokenizer=tokenizer,
        split="val",
        min_spells=1,
        max_spells=1,
        window_size=effective_window_size,
        window_step=100,
        exclude_subjects=exclude_subjects,
        trial_duration_pts=trial_duration_pts,
        decoder_type=decoder_type,
    )

    # Load raw data for per-trial comparison
    data = torch.load(Path(eeg_dir) / "val_eeg.pt", weights_only=True)

    # Load duration-aware decoder data
    if trial_duration_pts == 600:
        cand_filename = f"val_{decoder_type}.pt"
    else:
        cand_filename = f"val_{decoder_type}_{trial_duration_pts}pt.pt"
    cand_path = Path(eeg_dir) / cand_filename
    if not cand_path.exists():
        raise FileNotFoundError(
            f"Precomputed {decoder_type} not found: {cand_path}\n"
            f"Run the appropriate precompute script first."
        )
    fbcca_data = torch.load(cand_path, weights_only=True)

    if exclude_bad:
        mask = torch.ones(len(data["labels"]), dtype=torch.bool)
        for sid in exclude_subjects:
            mask &= data["subject_ids"] != sid
        labels = data["labels"][mask]
        subject_ids = data["subject_ids"][mask]
        eeg_data = data["eeg_data"][mask]
        fbcca_indices = fbcca_data["top3_indices"][mask]
        fbcca_scores = fbcca_data["top3_scores"][mask]
    else:
        labels = data["labels"]
        subject_ids = data["subject_ids"]
        eeg_data = data["eeg_data"]
        fbcca_indices = fbcca_data["top3_indices"]
        fbcca_scores = fbcca_data["top3_scores"]

    N = len(labels)
    print(f"\nEvaluating {N} trials (duration={trial_duration}s, {trial_duration_pts}pts)...")

    # Get target token IDs
    target_token_ids = {
        i: tokenizer.convert_tokens_to_ids(tok)
        for i, tok in TARGET_INDEX_TO_TOKEN.items()
    }
    target_id_to_label = {v: k for k, v in target_token_ids.items()}
    target_id_tensor = torch.tensor(list(target_token_ids.values()), device=device)

    collator = BCIAgentCollator(tokenizer)
    # Two-step: collect both EEG-only and Final predictions
    eeg_only_preds = []
    eeg_only_probs = []
    final_preds = []
    final_probs = []
    two_step_detected = None  # Auto-detect from first batch

    # Process in batches
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_items = []

        for trial_idx in range(start_idx, end_idx):
            label = int(labels[trial_idx])
            offset_idx = 0
            offset = dataset.window_offsets[offset_idx]
            window = eeg_data[trial_idx, :, offset:offset + effective_window_size]

            top3_idx = fbcca_indices[trial_idx, offset_idx].tolist()
            top3_sc = fbcca_scores[trial_idx, offset_idx].tolist()

            input_ids, label_ids = dataset._build_sequence(
                [label], [(top3_idx, top3_sc)]
            )

            batch_items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
                "eeg_windows": window.unsqueeze(0).float(),
                "num_spells": 1,
            })

        batch = collator(batch_items)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            eeg_windows = batch.pop("eeg_windows", None)
            window_counts = batch.pop("window_counts", None)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                eeg_windows=eeg_windows if eeg_windows is not None and eeg_windows.numel() > 0 else None,
                window_counts=window_counts,
            )

            logits = outputs.logits

            B = batch["labels"].shape[0]
            for i in range(B):
                label_row = batch["labels"][i]
                target_positions = [
                    pos for pos in range(len(label_row))
                    if label_row[pos].item() in target_id_to_label
                ]

                # Auto-detect two-step mode from first trial
                if two_step_detected is None:
                    two_step_detected = len(target_positions) >= 2
                    mode_str = "two-step" if two_step_detected else "single-step"
                    print(f"  Detected {mode_str} mode ({len(target_positions)} target positions per trial)")

                if two_step_detected and len(target_positions) >= 2:
                    # EEG-only prediction (1st target position)
                    pos_eeg = target_positions[0]
                    eeg_logits = logits[i, pos_eeg - 1, target_id_tensor]
                    eeg_p = F.softmax(eeg_logits, dim=0)
                    # Use full-vocab argmax (consistent with Trainer metrics)
                    eeg_full_pred = logits[i, pos_eeg - 1].argmax().item()
                    eeg_label = target_id_to_label.get(eeg_full_pred, -1)
                    eeg_only_preds.append(eeg_label)
                    eeg_only_probs.append(eeg_p.cpu().float().numpy())

                    # Final prediction (2nd target position)
                    pos_final = target_positions[1]
                    final_logits = logits[i, pos_final - 1, target_id_tensor]
                    final_p = F.softmax(final_logits, dim=0)
                    final_full_pred = logits[i, pos_final - 1].argmax().item()
                    final_label = target_id_to_label.get(final_full_pred, -1)
                    final_preds.append(final_label)
                    final_probs.append(final_p.cpu().float().numpy())
                elif target_positions:
                    # Single-step fallback (backward compat)
                    pos = target_positions[0]
                    target_logits = logits[i, pos - 1, target_id_tensor]
                    probs = F.softmax(target_logits, dim=0)
                    full_pred = logits[i, pos - 1].argmax().item()
                    final_preds.append(target_id_to_label.get(full_pred, -1))
                    final_probs.append(probs.cpu().float().numpy())
                    eeg_only_preds.append(-1)
                    eeg_only_probs.append(np.zeros(40))
                else:
                    final_preds.append(-1)
                    final_probs.append(np.zeros(40))
                    eeg_only_preds.append(-1)
                    eeg_only_probs.append(np.zeros(40))

        if (start_idx // batch_size) % 50 == 0:
            print(f"  Processed {end_idx}/{N} trials...")

    true_labels_np = labels.numpy()
    fbcca_top1 = fbcca_indices[:, 0, 0].numpy()
    subject_ids_np = subject_ids.numpy()

    return {
        "eeg_only_preds": np.array(eeg_only_preds),
        "eeg_only_probs": np.array(eeg_only_probs),
        "final_preds": np.array(final_preds),
        "final_probs": np.array(final_probs),
        "true_labels": true_labels_np,
        "fbcca_top1": fbcca_top1,
        "subject_ids": subject_ids_np,
        "two_step": bool(two_step_detected),
    }


def filter_results_by_char_type(results, keep="letters"):
    """Filter evaluation results by character type.

    Args:
        results: dict from run_evaluation()
        keep: "letters" (A-Z, indices 0-25) or "digits" (1-0, indices 26-35)

    Returns:
        Filtered results dict (same structure, fewer trials)
    """
    true_labels = results["true_labels"]
    if keep == "letters":
        mask = true_labels <= 25
        desc = "letters only (A-Z)"
    elif keep == "digits":
        mask = (true_labels >= 26) & (true_labels <= 35)
        desc = "digits only (0-9)"
    else:
        raise ValueError(f"Unknown keep={keep!r}, must be 'letters' or 'digits'")

    n_before = len(true_labels)
    n_after = mask.sum()
    print(f"\nFiltering: {desc} → {n_after}/{n_before} trials kept")

    filtered = {
        "true_labels": true_labels[mask],
        "fbcca_top1": results["fbcca_top1"][mask],
        "subject_ids": results["subject_ids"][mask],
        "final_preds": results["final_preds"][mask],
        "final_probs": results["final_probs"][mask],
        "eeg_only_preds": results["eeg_only_preds"][mask],
        "eeg_only_probs": results["eeg_only_probs"][mask],
        "two_step": results["two_step"],
    }
    return filtered


def print_results(results, decoder_name="FBCCA"):
    """Print comprehensive evaluation results."""
    model_preds = results["final_preds"]
    model_probs = results["final_probs"]
    true_labels = results["true_labels"]
    fbcca_top1 = results["fbcca_top1"]
    subject_ids = results["subject_ids"]
    is_two_step = results["two_step"]
    N = len(true_labels)

    correction = compute_fbcca_correction_metrics(model_preds, true_labels, fbcca_top1)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Trials:           {N}")
    print(f"  {decoder_name} accuracy:   {correction['fbcca_acc']:.1%}")

    # EEG-only accuracy (two-step mode)
    if is_two_step:
        eeg_preds = results["eeg_only_preds"]
        valid_eeg = eeg_preds >= 0
        if valid_eeg.sum() > 0:
            eeg_acc = (eeg_preds[valid_eeg] == true_labels[valid_eeg]).mean()
            eeg_top5 = np.argsort(-results["eeg_only_probs"][valid_eeg], axis=1)[:, :5]
            eeg_top5_hit = np.any(eeg_top5 == true_labels[valid_eeg, None], axis=1).mean()
            print(f"  EEG-only acc:     {eeg_acc:.1%}   (pure EEG decoding, no candidates)")
            print(f"  EEG-only top-5:   {eeg_top5_hit:.1%}")

    print(f"  Final model acc:  {correction['model_acc']:.1%}")
    print(f"  Override rate:    {correction['override_rate']:.1%}  (model disagrees with {decoder_name})")
    print(f"  Correction rate:  {correction['correction_rate']:.1%}  ({decoder_name} wrong -> model right)")
    print(f"  Trust rate:       {correction['trust_rate']:.1%}  ({decoder_name} right -> model agrees)")
    print(f"  {decoder_name} errors:     {correction['correction_count']}")

    # Top-5
    top5_preds = np.argsort(-model_probs, axis=1)[:, :5]
    top5_hit = np.any(top5_preds == true_labels[:, None], axis=1)
    print(f"  Final top-5 acc:  {top5_hit.mean():.1%}")

    # Per-subject breakdown
    print("\n" + "-" * 60)
    print("PER-SUBJECT BREAKDOWN")
    print("-" * 60)
    dec_short = decoder_name[:6]
    header = f"{'Subject':>8} {'Trials':>7} {dec_short:>7}"
    if is_two_step:
        header += f" {'EEGonly':>7}"
    header += f" {'Final':>7} {'Corrn':>7} {'Trust':>7}"
    print(header)

    unique_subjects = sorted(np.unique(subject_ids))
    for sid in unique_subjects:
        mask = subject_ids == sid
        n = mask.sum()
        if n == 0:
            continue

        s_model = model_preds[mask]
        s_labels = true_labels[mask]
        s_fbcca = fbcca_top1[mask]

        s_fbcca_acc = (s_fbcca == s_labels).mean()
        s_model_acc = (s_model == s_labels).mean()

        fbcca_wrong = s_fbcca != s_labels
        if fbcca_wrong.sum() > 0:
            s_correction = (s_model[fbcca_wrong] == s_labels[fbcca_wrong]).mean()
        else:
            s_correction = float('nan')

        fbcca_right = s_fbcca == s_labels
        if fbcca_right.sum() > 0:
            s_trust = (s_model[fbcca_right] == s_fbcca[fbcca_right]).mean()
        else:
            s_trust = float('nan')

        line = f"  S{sid:02d}   {n:>6}  {s_fbcca_acc:>6.1%}"
        if is_two_step:
            s_eeg = results["eeg_only_preds"][mask]
            s_eeg_valid = s_eeg >= 0
            s_eeg_acc = (s_eeg[s_eeg_valid] == s_labels[s_eeg_valid]).mean() if s_eeg_valid.sum() > 0 else float('nan')
            line += f"  {s_eeg_acc:>6.1%}"
        line += f"  {s_model_acc:>6.1%}  {s_correction:>6.1%}  {s_trust:>6.1%}"
        print(line)

    # Per-class accuracy (worst 10)
    print("\n" + "-" * 60)
    print("PER-CLASS ACCURACY (worst 10)")
    print("-" * 60)
    per_class = []
    for c in range(40):
        mask = true_labels == c
        if mask.sum() > 0:
            acc = (model_preds[mask] == c).mean()
            char = KEYBOARD_CHARS[c]
            per_class.append((c, char, acc, mask.sum()))

    per_class.sort(key=lambda x: x[2])
    for c, char, acc, n in per_class[:10]:
        print(f"  Class {c:2d} ({char}): {acc:.1%}  ({n} trials)")

    # Sample predictions
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS (first 20)")
    print("-" * 60)
    if is_two_step:
        print(f"{'#':>4} {'True':>5} {'EEGonly':>7} {'Final':>6} {dec_short:>6} {'Conf':>6} {'Result':>8}")
    else:
        print(f"{'#':>4} {'True':>5} {'Model':>6} {dec_short:>6} {'Conf':>6} {'Result':>8}")
    for i in range(min(20, N)):
        true_char = KEYBOARD_CHARS[true_labels[i]]
        model_char = KEYBOARD_CHARS[model_preds[i]] if 0 <= model_preds[i] < 40 else "?"
        fbcca_char = KEYBOARD_CHARS[fbcca_top1[i]]
        conf = model_probs[i].max()
        result = "OK" if model_preds[i] == true_labels[i] else "MISS"
        if model_preds[i] == true_labels[i] and fbcca_top1[i] != true_labels[i]:
            result = "FIXED!"
        if is_two_step:
            eeg_pred = results["eeg_only_preds"][i]
            eeg_char = KEYBOARD_CHARS[eeg_pred] if 0 <= eeg_pred < 40 else "?"
            print(f"  {i:3d}   {true_char:>4}   {eeg_char:>6}   {model_char:>5}   {fbcca_char:>5}   {conf:>5.2f}   {result:>7}")
        else:
            print(f"  {i:3d}   {true_char:>4}   {model_char:>5}   {fbcca_char:>5}   {conf:>5.2f}   {result:>7}")


def print_duration_summary(results_by_duration, decoder_name="FBCCA"):
    """Print a comparison table across multiple trial durations."""
    has_eeg = any("eeg_only_acc" in res for res in results_by_duration.values())

    print("\n" + "=" * 90)
    print("MULTI-DURATION COMPARISON")
    print("=" * 90)
    dec_header = f"{decoder_name} Acc"
    header = f"{'Duration':>10} | {dec_header:>10} |"
    if has_eeg:
        header += f" {'EEG-only':>10} |"
    header += f" {'Final Acc':>10} | {'Correction':>10} | {'Trust':>10} | {'Top-5':>10}"
    print(header)
    print("-" * 90)

    for dur, res in sorted(results_by_duration.items()):
        line = f"  {dur:5.1f}s   | {res['fbcca_acc']:>9.1%} |"
        if has_eeg:
            eeg_acc = res.get("eeg_only_acc", float("nan"))
            line += f" {eeg_acc:>9.1%} |"
        line += (f" {res['model_acc']:>9.1%} | "
                 f"{res['correction_rate']:>9.1%} | {res['trust_rate']:>9.1%} | "
                 f"{res['top5_acc']:>9.1%}")
        print(line)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation of candidate-mode BCI agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="S2 checkpoint directory")
    parser.add_argument("--s1_checkpoint", type=str, required=True,
                        help="S1 checkpoint directory (for base model loading)")
    parser.add_argument("--eeg_dir", type=str, default="data/eeg_tensors")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--encoder_type", type=str, default="reve")
    parser.add_argument("--reve_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_exclude_bad", action="store_true")
    parser.add_argument("--from_modelscope", action="store_true", default=True,
                        help="Download model from ModelScope (default: True)")
    parser.add_argument("--no_modelscope", action="store_true",
                        help="Download from HuggingFace instead of ModelScope")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trial_duration", type=float, default=3.0,
                        help="Trial duration in seconds (default: 3.0)")
    parser.add_argument("--durations", type=float, nargs="*",
                        help="Evaluate multiple durations and print comparison table")
    parser.add_argument("--decoder_type", type=str, default="fbcca",
                        choices=["fbcca", "trca", "etrca"],
                        help="Decoder type for candidate predictions (default: fbcca)")
    parser.add_argument("--letters_only", action="store_true",
                        help="Evaluate only letter targets (A-Z, indices 0-25), excluding digits and special chars")
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

    # Determine durations to evaluate
    durations = args.durations if args.durations else [args.trial_duration]

    if len(durations) > 1:
        # Multi-duration comparison mode
        results_by_duration = {}
        for dur in durations:
            print(f"\n{'='*60}")
            print(f"Evaluating duration: {dur}s ({int(dur*200)}pts)")
            print(f"{'='*60}")

            results = run_evaluation(
                model, tokenizer,
                eeg_dir=args.eeg_dir,
                device=device,
                exclude_bad=not args.no_exclude_bad,
                batch_size=args.batch_size,
                trial_duration=dur,
                decoder_type=args.decoder_type,
            )

            if args.letters_only:
                results = filter_results_by_char_type(results, keep="letters")

            correction = compute_fbcca_correction_metrics(
                results["final_preds"], results["true_labels"], results["fbcca_top1"])
            top5_preds = np.argsort(-results["final_probs"], axis=1)[:, :5]
            top5_acc = np.any(top5_preds == results["true_labels"][:, None], axis=1).mean()

            dur_summary = {**correction, "top5_acc": float(top5_acc)}

            # Add EEG-only accuracy if two-step
            if results["two_step"]:
                eeg_p = results["eeg_only_preds"]
                valid = eeg_p >= 0
                if valid.sum() > 0:
                    dur_summary["eeg_only_acc"] = float(
                        (eeg_p[valid] == results["true_labels"][valid]).mean())

            results_by_duration[dur] = dur_summary

            # Brief per-duration summary
            dec_name = args.decoder_type.upper()
            eeg_str = ""
            if "eeg_only_acc" in dur_summary:
                eeg_str = f", EEG-only: {dur_summary['eeg_only_acc']:.1%}"
            print(f"  {dec_name}: {correction['fbcca_acc']:.1%}, "
                  f"Model: {correction['model_acc']:.1%}, "
                  f"Top-5: {top5_acc:.1%}{eeg_str}")

        print_duration_summary(results_by_duration, decoder_name=args.decoder_type.upper())
    else:
        # Single-duration mode (full output)
        dur = durations[0]
        print("Running evaluation...")
        results = run_evaluation(
            model, tokenizer,
            eeg_dir=args.eeg_dir,
            device=device,
            exclude_bad=not args.no_exclude_bad,
            batch_size=args.batch_size,
            trial_duration=dur,
            decoder_type=args.decoder_type,
        )

        if args.letters_only:
            results = filter_results_by_char_type(results, keep="letters")

        print_results(results, decoder_name=args.decoder_type.upper())

        # Save predictions
        suffix = "" if dur == 3.0 else f"_{int(dur*200)}pt"
        out_path = Path(args.checkpoint) / f"eval_predictions{suffix}.npz"
        save_dict = {
            "final_preds": results["final_preds"],
            "final_probs": results["final_probs"],
            "true_labels": results["true_labels"],
            "fbcca_top1": results["fbcca_top1"],
            "subject_ids": results["subject_ids"],
        }
        if results["two_step"]:
            save_dict["eeg_only_preds"] = results["eeg_only_preds"]
            save_dict["eeg_only_probs"] = results["eeg_only_probs"]
        np.savez(out_path, **save_dict)
        print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
