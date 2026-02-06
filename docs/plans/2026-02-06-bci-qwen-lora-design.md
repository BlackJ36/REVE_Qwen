# BCI-Qwen: REVE + Qwen3-VL LoRA Fine-tuning Design

## Overview

Use REVE (EEG foundation model) as a BCI adapter to process Tsinghua Benchmark and BETA SSVEP datasets, project EEG embeddings into Qwen3-VL-Instruct-8B's input space via a learned projection layer, and LoRA fine-tune the model to decode BCI signals into target tokens.

## Architecture

```
EEG (64-ch) → REVE Encoder (frozen) → Projection MLP (trainable) → Qwen3-VL (LoRA, bf16)
                                                                          ↓
                                                                      <|tXX|>
```

### Components
- **REVE** (frozen): Pretrained EEG foundation model from brain-bzh/reve-base
- **Projection Layer** (trainable): 2-layer MLP, REVE_dim → 3584 (Qwen3-VL hidden dim)
- **Qwen3-VL-8B-Instruct** (LoRA): From ModelScope, bf16
- **Special Tokens**: 3 control (`<|bci_start|>`, `<|bci_end|>`, `<|bci_sep|>`) + 40 target (`<|t01|>`-`<|t40|>`)

## Data

- Tsinghua Benchmark: 35 subjects, 8,400 samples
- BETA: 70 subjects, 11,200 samples
- Total: ~19,600 samples
- Split: by subject (Benchmark 5 val, BETA 10 val)
- Preprocessing: bandpass 1-40Hz, trial segmentation, normalization
- REVE embeddings extracted offline to .pt files

## Training Config

- 6x 48GB GPUs, DeepSpeed ZeRO-2
- bf16, LoRA rank 64, alpha 128
- Effective batch size: 96
- LR: 2e-4 (projection: 1e-3)
- 10 epochs, cosine scheduler
- ~60M trainable parameters (<1% of 8B)
- Estimated training time: 15-20 minutes

## Two-Stage Plan
1. **Stage 1**: Trial-level (current) - each trial → one target token
2. **Stage 2**: Sliding window - continuous EEG → streaming output
