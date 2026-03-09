"""Evaluate BCI spelling correction model (full S2 format).

Evaluates on Type A (spelling) and Type C (correction) samples.
Type D (NL) is evaluated separately.

Metrics:
  - word_acc: exact match of full output
  - char_acc: character-level accuracy (for Type A)
  - correction_rate: how often model fixes FBCCA errors
  - trust_rate: how often model preserves FBCCA correct results

Usage:
    uv run python scripts/eval_correction.py \
        --data_dir data/correction \
        --checkpoint output/correction/final
"""

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def edit_distance(s1, s2):
    """Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def extract_correction(text):
    """Extract the corrected word from Type C assistant output.

    Format: '{wrong} 你是不是想拼{right}' or similar templates.
    Returns (wrong_part, right_part) or (text, text) if no correction found.
    """
    patterns = [
        r"(.+?)\s+你是不是想拼(.+)",
        r"(.+?)\s+可能你想输入的是(.+)",
        r"(.+?)\s+这看起来像是(.+?)的误拼",
        r"(.+?)\s+检测到可能的拼写错误\s+建议修正为(.+)",
        r"(.+?)\s+自动纠正为(.+)",
    ]
    for p in patterns:
        m = re.match(p, text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return text.strip(), text.strip()


def run_generation(model, tokenizer, data_path, device, max_new_tokens=80,
                   batch_size=8, max_samples=None):
    """Run batched generation and compute per-type metrics."""
    import time

    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if max_samples and max_samples < len(samples):
        # Stratified sample: keep proportions of each type
        import random
        rng = random.Random(42)
        by_type = defaultdict(list)
        for s in samples:
            by_type[s.get("type", "A")].append(s)
        total = len(samples)
        sampled = []
        for dtype, items in by_type.items():
            n = max(1, round(len(items) / total * max_samples))
            sampled.extend(rng.sample(items, min(n, len(items))))
        samples = sampled[:max_samples]
        print(f"  Subsampled to {len(samples)} samples")

    results = {"A": [], "C": [], "D": []}
    model.eval()
    t0 = time.time()

    # Prepare all prompts
    all_prompts = []
    for sample in samples:
        messages = sample["messages"][:2]  # system + user only
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        all_prompts.append(text)

    # Batch generation
    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_samples = samples[batch_start:batch_end]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]  # padded length (same for all)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        for j, sample in enumerate(batch_samples):
            new_tokens = output_ids[j][input_len:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            target = sample["messages"][2]["content"]
            dtype = sample.get("type", "A")

            result = {
                "type": dtype,
                "pred": pred,
                "target": target,
                "target_word": sample.get("target_word", ""),
                "noisy_word": sample.get("noisy_word", ""),
                "exact_match": pred == target,
            }
            results[dtype].append(result)

            idx = batch_start + j
            if idx < 5 or idx % 100 == 0:
                print(f"  [{idx}/{len(samples)}][{dtype}] pred={pred[:70]}")
                print(f"  {' ':>{len(str(len(samples)))}}       target={target[:70]}")

        elapsed = time.time() - t0
        speed = batch_end / elapsed
        eta = (len(samples) - batch_end) / speed if speed > 0 else 0
        if batch_start % (batch_size * 5) == 0 or batch_end == len(samples):
            print(f"  [{batch_end}/{len(samples)}] {speed:.1f} samples/s, "
                  f"ETA {eta:.0f}s", flush=True)

    return results


def compute_metrics(results):
    """Compute metrics per type."""
    metrics = {}

    # Type A: spelling accuracy
    type_a = results.get("A", [])
    if type_a:
        exact = sum(r["exact_match"] for r in type_a)
        # Character accuracy: 1 - (edit_distance / target_length)
        char_total = 0
        total_ed = 0
        for r in type_a:
            target = r["target_word"]
            pred = r["pred"]
            ed = edit_distance(pred, target)
            total_ed += ed
            char_total += len(target)

        metrics["A_word_acc"] = exact / len(type_a)
        metrics["A_char_acc"] = 1.0 - total_ed / max(char_total, 1)
        metrics["A_avg_ed"] = total_ed / len(type_a)
        metrics["A_count"] = len(type_a)

        # FBCCA baseline for Type A
        noisy_exact = sum(1 for r in type_a if r["noisy_word"] == r["target_word"])
        noisy_ed = sum(edit_distance(r["noisy_word"], r["target_word"]) for r in type_a)
        metrics["A_fbcca_word_acc"] = noisy_exact / len(type_a)
        metrics["A_fbcca_char_acc"] = 1.0 - noisy_ed / max(char_total, 1)
        metrics["A_fbcca_avg_ed"] = noisy_ed / len(type_a)

    # Type C: correction accuracy
    type_c = results.get("C", [])
    if type_c:
        exact = sum(r["exact_match"] for r in type_c)
        # Extract correction from pred and check if right part matches target
        correction_correct = 0
        for r in type_c:
            _, pred_right = extract_correction(r["pred"])
            if pred_right == r["target_word"]:
                correction_correct += 1

        metrics["C_exact_match"] = exact / len(type_c)
        metrics["C_correction_acc"] = correction_correct / len(type_c)
        metrics["C_count"] = len(type_c)

    # Type D: NL accuracy
    type_d = results.get("D", [])
    if type_d:
        exact = sum(r["exact_match"] for r in type_d)
        metrics["D_exact_match"] = exact / len(type_d)
        metrics["D_count"] = len(type_d)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate BCI correction model")
    parser.add_argument("--data_dir", type=str, default="data/correction")
    parser.add_argument("--checkpoint", type=str, default="output/correction/final")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--from_modelscope", action="store_true", default=True)
    parser.add_argument("--run_base", action="store_true",
                        help="Also run base model (zero-shot)")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (stratified, default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"

    # Load base model
    if args.from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(args.model_name)
    else:
        model_path = args.model_name

    print(f"Loading base model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )

    # Run base model (zero-shot) if requested
    if args.run_base:
        print("\n=== Base Model (Zero-Shot) ===")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(device)
        base_results = run_generation(base_model, tokenizer, data_path, device,
                                      batch_size=args.batch_size,
                                      max_samples=args.max_samples)
        del base_model
        torch.cuda.empty_cache()

        base_metrics = compute_metrics(base_results)
        print("\nBase model results:")
        for k, v in base_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Run fine-tuned model
    ckpt_dir = Path(args.checkpoint)
    if ckpt_dir.exists():
        print(f"\n=== Fine-tuned Model ({args.checkpoint}) ===")
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        if (ckpt_dir / "adapter_config.json").exists():
            ft_model = PeftModel.from_pretrained(ft_model, str(ckpt_dir))
        ft_model = ft_model.to(device)

        ft_results = run_generation(ft_model, tokenizer, data_path, device,
                                    batch_size=args.batch_size,
                                    max_samples=args.max_samples)
        ft_metrics = compute_metrics(ft_results)

        print("\nFine-tuned model results:")
        for k, v in ft_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save
        out_path = ckpt_dir / f"results_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump({"metrics": ft_metrics}, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print(f"Checkpoint not found: {ckpt_dir}")


if __name__ == "__main__":
    main()
