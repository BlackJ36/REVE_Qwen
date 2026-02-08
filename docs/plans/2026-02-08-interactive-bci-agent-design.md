# Interactive BCI Agent: REVE + FBCCA + Qwen3-4B Design

> Supersedes: `2026-02-06-bci-qwen-lora-design.md` (trial-level classification only)

## Motivation

Previous design treated the model as a pure 40-class SSVEP classifier (`EEG → <|tXX|>`).
This redesign targets an **interactive BCI agent** that can:

- Decode EEG signals into target characters with confidence estimates
- Maintain spelling session state across multiple rounds
- Respond to user commands (undo, resend, switch layout)
- Generate natural language for error handling and user guidance
- Support real-time streaming: decode each spell as it arrives

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │       HybridEncoder (frozen)     │
                    │                                  │
EEG (62ch, 300pts)──┤  REVE (69.2M, frozen)            │
                    │    → (B, 62, 512) spatial tokens  │
                    │                                  │
                    │  FBCCA (zero-param)               │
                    │    → (B, 200) freq correlations   │
                    │                                  │
                    │  FiLM Modulation                  │
                    │    γ,β = MLP(fbcca)  → (B, 512)  │
                    │    out = γ * reve + β → (B,62,512)│
                    │                                  │
                    │  Projector MLP                    │
                    │    (B, 62, 512) → (B, 62, 2048)  │
                    └────────────┬────────────────────┘
                                 │ 62 EEG tokens
                                 ▼
                    ┌─────────────────────────────────┐
                    │   Qwen3-4B-Instruct (LoRA)       │
                    │                                  │
                    │   pad replacement: <|bci_pad|>   │
                    │   tokens ← EEG embeddings        │
                    │                                  │
                    │   Output: 自然语言 + target info   │
                    └─────────────────────────────────┘
```

### Component Summary

| Component | Params | Trainable | Role |
|-----------|--------|-----------|------|
| REVE | 69.2M | No | Spatial/temporal EEG features |
| FBCCA | 0 | No | Frequency-domain SSVEP correlations |
| FiLM layers | ~200K | Yes | Frequency modulates spatial features |
| Projector MLP | ~2M | Yes | Map 512d → 2048d (Qwen hidden dim) |
| Qwen3-4B | ~4B | LoRA only (~10-20M) | Language generation + sequence modeling |

Total trainable: ~12-22M (depending on LoRA rank)

## Key Design Decisions

### Why Qwen3-4B-Instruct (text-only)?

- **Not 0.6B**: Language too weak for interactive agent
- **Not 8B**: 4B is sufficient for structured BCI interaction; trains 2x faster
- **Not VL**: No vision encoder needed; `AutoModelForCausalLM` is simpler
- **Text-only Instruct**: Pre-trained instruction following + Chinese capability
- **Upgrade path**: Switch to 8B with zero code changes if 4B language quality insufficient

### Why FiLM over Concat?

Current `encoder_hybrid.py` broadcasts FBCCA (B, 200) identically across all 62 electrode tokens.
FiLM is better because:

- **Multiplicative interaction** has higher feature interaction order than concatenation
- **No dimension inflation**: output stays 512d (vs 712d with concat)
- **Semantic match**: frequency info naturally "modulates" spatial features
  (analogous to VQA: question modulates image features)
- **Minimal cost**: only 2 × Linear(200, 512) ≈ 200K params

```python
# FiLM modulation (replaces concat)
gamma = self.film_gamma(fbcca_out)  # (B, 512)
beta  = self.film_beta(fbcca_out)   # (B, 512)
fused = gamma.unsqueeze(1) * reve_out + beta.unsqueeze(1)  # (B, 62, 512)
```

### Why LLaVA-style Injection (not Q-Former or Flamingo)?

- **Token budget is fine**: 10 spells × 62 tokens = 620 tokens, only 2% of 32K context
- **Already implemented**: pad-replacement pattern exists in codebase
- **Compatible with streaming**: standard multi-turn chat inference with KV-cache
- **Simplest to debug**: EEG tokens are just regular embeddings in the sequence

## Streaming Architecture

### Real-time Inference Flow

Each SSVEP trial (3s of EEG) is processed and responded to individually:

```
Trial 1 arrives (3s) ──→ REVE+FBCCA+FiLM ──→ 62 tokens injected
                                                    │
                     Qwen generates (~220ms) ◄──────┘
                            │
                     "H（12.0Hz，高）。已拼写：H"
                            │
Trial 2 arrives (3s) ──→ ... (KV-cache reuses previous computation)
                            │
                     "E（9.0Hz，高）。已拼写：HE"
                            │
User types "撤销" ──────→ Qwen processes text input
                            │
                     "已撤销。当前：H"
```

### Per-spell Latency

| Step | Time |
|------|------|
| REVE forward | ~10ms |
| FBCCA forward | ~5ms |
| FiLM + projector | ~1ms |
| Qwen3-4B generate ~30 tokens | ~200ms |
| **Total** | **~220ms** |

Each trial lasts 3 seconds. Response is ready in 0.22s — no perceptible delay.

### Multi-turn Format (inference)

```
<|im_start|>system
你是一个脑机接口助手，负责解码用户的SSVEP脑电信号并协助交流。<|im_end|>
<|im_start|>user
[62 EEG pads]<|im_end|>
<|im_start|>assistant
H（12.0Hz，置信度高）。已拼写：H<|im_end|>
<|im_start|>user
[62 EEG pads]<|im_end|>
<|im_start|>assistant
E（9.0Hz，置信度高）。已拼写：HE<|im_end|>
<|im_start|>user
发送<|im_end|>
<|im_start|>assistant
消息"HE"已发送。<|im_end|>
```

## Training Strategy

### Stage 1: Alignment (EEG → Qwen embedding space)

**Goal**: Projector + FiLM learn to produce embeddings that frozen Qwen can decode.

| Setting | Value |
|---------|-------|
| Qwen | Frozen (no LoRA) |
| Trainable | Projector + FiLM (~2M params) |
| REVE/FBCCA | Frozen |
| Data | Pure EEG → target token (existing dataset) |
| Epochs | 5-10 |
| LR | 1e-3 (projector), 1e-3 (FiLM) |
| Batch size | 64-128 per GPU |

**Data format** (same as current E2E but with chat template):

```
<|im_start|>system
You are a BCI decoder.<|im_end|>
<|im_start|>user
Decode the EEG signal.<|im_end|>
<|im_start|>assistant
[62 pads]<|t15|><|bci_trans|>[62 pads]<|t08|><|bci_trans|>...[62 pads]<|t22|><|im_end|>
```

**Loss**: Cross-entropy, only on target token positions (all else masked -100).

```python
labels = [-100, ..., -100, t15_id, -100, ..., -100, t08_id, ...]
#         system/user/pads  ^^^^    bci_trans/pads   ^^^^
L_stage1 = CrossEntropyLoss(ignore_index=-100)(logits, labels)
```

**Expected outcome**: Classification accuracy comparable to FBCCA linear probe (~68-83%).
Model outputs only target tokens, no natural language.

### Stage 2: Instruction Tuning (unlock language + interaction)

**Goal**: Teach Qwen to generate NL responses based on EEG input while preserving language.

| Setting | Value |
|---------|-------|
| Qwen | LoRA (rank=32-64, target: q_proj, v_proj) |
| Trainable | LoRA + Projector + FiLM (~12-22M params) |
| REVE/FBCCA | Frozen |
| Data | Mixed 4 types (see below) |
| Epochs | 3-5 |
| LR | 2e-5 (LoRA), 5e-4 (projector/FiLM, lower than Stage 1) |
| Batch size | 32-64 per GPU |

**Loss**: Cross-entropy on **entire assistant turn** (not just target tokens).

```python
labels = [-100, ..., -100,  检, 测, 到, 目, 标, K, ..., 。, eos]
#         system + user      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                            full assistant response in loss
L_stage2 = CrossEntropyLoss(ignore_index=-100)(logits, labels)
```

## Training Data Construction (Stage 2)

### Type A: Single EEG Decode (30%)

Single trial → natural language description.

```
<|im_start|>system
你是一个脑机接口助手，负责解码用户的SSVEP脑电信号并协助交流。<|im_end|>
<|im_start|>user
请解码以下脑电信号。
[62 EEG pads]<|im_end|>
<|im_start|>assistant
检测到目标 K（频率 12.0Hz），置信度：高。<|im_end|>
```

**Generation**: Template-based with FBCCA ρ values for confidence.

Templates (examples, need 10-20 variants for diversity):
```
- "检测到目标 {char}（频率 {freq}Hz），置信度：{conf}。"
- "解码结果：{char}，对应频率 {freq}Hz。信号质量{quality}。"
- "识别到您正在注视 {char}（{freq}Hz）。"
```

Confidence thresholds (based on FBCCA max ρ):
- High (高): ρ > 0.6
- Medium (中): 0.3 < ρ ≤ 0.6
- Low (低): ρ ≤ 0.3

### Type B: Multi-turn Streaming Spelling (30%)

Multi-spell session in multi-turn chat format.

```
<|im_start|>system
你是一个脑机接口助手。用户通过注视屏幕上的字符来拼写。
每次收到脑电信号后，请报告解码结果和当前拼写进度。<|im_end|>
<|im_start|>user
[62 EEG pads]<|im_end|>
<|im_start|>assistant
H（12.0Hz，置信度高）。已拼写：H<|im_end|>
<|im_start|>user
[62 EEG pads]<|im_end|>
<|im_start|>assistant
E（9.0Hz，置信度高）。已拼写：HE<|im_end|>
<|im_start|>user
[62 EEG pads]<|im_end|>
<|im_start|>assistant
L（11.0Hz，置信度中）。已拼写：HEL<|im_end|>
<|im_start|>user
结束拼写<|im_end|>
<|im_start|>assistant
拼写完成：HEL。是否发送此消息？<|im_end|>
```

**Generation**: Sample K=3-8 consecutive trials from same subject.
Map target labels to characters using the 40-target keyboard layout.
Randomize session length and include occasional user commands (undo, end).

### Type C: Pure Natural Language (20%)

No EEG input. Maintains language capability, prevents catastrophic forgetting.

```
<|im_start|>user
什么是SSVEP？<|im_end|>
<|im_start|>assistant
SSVEP（稳态视觉诱发电位）是当人注视以特定频率闪烁的视觉刺激时，
大脑视觉皮层产生的与刺激频率同步的电生理响应...<|im_end|>
```

**Data sources**:
- Open-source Chinese instruction dataset subset (e.g., Alpaca-GPT4-Chinese)
- BCI domain Q&A (manually written or LLM-generated, ~200-500 samples)
- General knowledge and conversation

### Type D: Error Handling & Commands (10%)

Teaches robustness and interactive commands.

```
# Low confidence
<|im_start|>user
[62 EEG pads - noisy]<|im_end|>
<|im_start|>assistant
信号质量较低，无法可靠解码。请确保电极接触良好，并重新注视目标字符。<|im_end|>

# Undo command
<|im_start|>user
撤销上一个字符<|im_end|>
<|im_start|>assistant
已撤销最后一个字符。当前拼写内容：HEL<|im_end|>

# Help
<|im_start|>user
帮助<|im_end|>
<|im_start|>assistant
可用命令：
- 注视字符即可输入
- "撤销"：删除上一个字符
- "清空"：清除所有已拼写内容
- "发送"：发送当前消息
- "重试"：重新解码上一个信号<|im_end|>
```

### Type E: Batch Spelling (10%)

Single-turn batch format (backward compatible with offline use).

```
<|im_start|>user
请解码以下一组脑电信号。
[62 pads][62 pads][62 pads][62 pads][62 pads]<|im_end|>
<|im_start|>assistant
解码结果：
1. H（12.0Hz，置信度高）
2. E（9.0Hz，置信度高）
3. L（11.0Hz，置信度中）
4. L（11.0Hz，置信度高）
5. O（14.0Hz，置信度高）

拼写内容：HELLO<|im_end|>
```

## Data Budget Estimate

| Type | Source | Samples | Notes |
|------|--------|---------|-------|
| A: Single decode | Existing EEG dataset | ~19,600 | One per trial |
| B: Streaming spell | Grouped trials | ~3,000 sessions | K=3-8 trials each |
| C: Pure NL | Open-source + handwritten | ~5,000 | Subset of instruction data |
| D: Error/command | Template + LLM generated | ~1,000 | |
| E: Batch spell | Grouped trials | ~2,000 sessions | |
| **Total** | | **~30,000 samples** | |

## Hardware & Training Estimates

### Stage 1 (Alignment)

- Hardware: 6x 48GB GPUs, DeepSpeed ZeRO-2
- VRAM per GPU: ~10-12GB (Qwen frozen in bf16 + encoder + projector)
- Batch: 128 per GPU, grad_accum=1, effective=768
- Time estimate: ~10-15 minutes for 5 epochs
- Checkpoint: ~50MB (projector + FiLM weights only)

### Stage 2 (Instruction Tuning)

- Hardware: same 6x 48GB GPUs
- VRAM per GPU: ~14-18GB (Qwen bf16 + LoRA + mixed-length sequences)
- Batch: 32-64 per GPU, grad_accum=2, effective=384-768
- Time estimate: ~30-60 minutes for 3-5 epochs
- Checkpoint: ~100MB (LoRA adapter + projector + FiLM)

## File Structure (planned)

```
src/
  encoder_film.py          # NEW: HybridEncoder with FiLM (replaces encoder_hybrid.py concat)
  model_bci_agent.py       # NEW: LLaVA-style Qwen wrapper with pad replacement
  dataset_bci_agent.py     # NEW: Multi-type dataset with template generation
  train_bci_agent.py       # NEW: Two-stage training loop
  templates_zh.py          # NEW: Chinese response templates
main_bci_agent.py          # NEW: Entry point
scripts/
  train_bci_agent_s1.sh    # Stage 1: alignment
  train_bci_agent_s2.sh    # Stage 2: instruction tuning
configs/
  ds_zero2.json            # Reuse existing DeepSpeed config
```

## Open Questions

1. **40-target keyboard layout**: Need to define the freq → character mapping table
   (currently only freq → label index exists)
2. **Confidence calibration**: FBCCA ρ thresholds (0.3/0.6) need validation on actual data
3. **Pure NL dataset selection**: Which Chinese instruction dataset to use for Type C
4. **LoRA rank**: 32 vs 64 — experiment in Stage 2
5. **Stage 1 convergence criterion**: When is alignment "good enough" to move to Stage 2?

## Relationship to Previous Work

- **Reuses**: REVE loading (`model_e2e.py:REVEWithUnfreeze`), FBCCA (`fbcca.py`),
  preprocessing pipeline, special token definitions (`tokens.py`)
- **Supersedes**: `model_hybrid.py` (Plan A, 43-token custom decoder),
  `model_hybrid_qwen.py` (Plan B, 0.6B full fine-tune) — both lack language capability
- **Evolves from**: `2026-02-06-bci-qwen-lora-design.md` — same LLaVA-style pattern
  but now with FiLM fusion, streaming support, and instruction tuning
