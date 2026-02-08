"""Plan B: HybridEncoder + Qwen3-0.6B full fine-tune for SSVEP decoding.

Key differences from existing BCIE2EQwenForCausalLM:
  - Qwen3-0.6B is text-only: AutoModelForCausalLM (not AutoModelForImageTextToText)
  - Full fine-tune (no LoRA): all 600M params trainable
  - HybridEncoder (REVE+FBCCA) instead of REVE alone
  - Hidden dim: 1024 (vs 3584 for 8B model)
"""

import torch
import torch.nn as nn

from .encoder_hybrid import HybridEncoder
from .tokens import ALL_SPECIAL_TOKENS, BCI_PAD, register_special_tokens


class HybridQwenModel(nn.Module):
    """HybridEncoder + Qwen3-0.6B (full fine-tune).

    Same pad-replacement pattern as BCIE2EQwenForCausalLM but with:
    - HybridEncoder (REVE + FBCCA + fusion projector) as encoder
    - Qwen3-0.6B text-only model as decoder
    - No LoRA, all Qwen params trainable
    """

    def __init__(self, encoder, qwen_model, tokenizer, num_eeg_tokens=62):
        super().__init__()
        self.encoder = encoder
        self.qwen = qwen_model
        self.tokenizer = tokenizer
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.num_eeg_tokens = num_eeg_tokens

    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None,
                eeg_windows=None, window_counts=None, **kwargs):
        """Forward pass with EEG pad replacement.

        Args:
            input_ids: (B, L) token IDs with <|bci_pad|> placeholders
            attention_mask: (B, L)
            labels: (B, L) with -100 for non-target positions
            eeg_windows: (total_K, 62, T) all EEG windows concatenated
            window_counts: (B,) number of windows per sample
        """
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if eeg_windows is not None:
            eeg_windows = eeg_windows.to(device=inputs_embeds.device, dtype=torch.float32)
            # Encode: (total_K, 62, T) -> (total_K, N, qwen_dim)
            projected = self.encoder(eeg_windows, output_dtype=inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.clone()
            bci_pad_mask = input_ids == self.bci_pad_id
            B = input_ids.size(0)
            qwen_dim = projected.size(-1)

            offset = 0
            for i in range(B):
                K_i = int(window_counts[i])
                sample_tokens = projected[offset:offset + K_i].reshape(-1, qwen_dim)
                pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                n = min(len(pad_positions), sample_tokens.size(0))
                inputs_embeds[i, pad_positions[:n]] = sample_tokens[:n]
                offset += K_i

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @property
    def config(self):
        return self.qwen.config

    @property
    def device(self):
        return self.qwen.device

    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, path, **kwargs):
        from pathlib import Path as P
        P(path).mkdir(parents=True, exist_ok=True)
        self.qwen.save_pretrained(path, **kwargs)
        torch.save(self.encoder.projector.state_dict(), P(path) / "encoder_projector.pt")


def build_hybrid_qwen_model(
    model_name="Qwen/Qwen3-0.6B",
    from_modelscope=True,
    reve_dir="models",
    num_eeg_tokens=62,
    window_size=300,
    sfreq=200.0,
    dropout=0.1,
):
    """Build the full Plan B hybrid Qwen model.

    Returns (model, tokenizer).
    """
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .encoder_hybrid import build_hybrid_encoder

    # --- Load Qwen3-0.6B (text-only) ---
    if from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_name)
    else:
        model_path = model_name

    print(f"Loading Qwen3-0.6B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    num_new = register_special_tokens(tokenizer)
    print(f"Added {num_new} special tokens to tokenizer")

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    qwen_model.resize_token_embeddings(len(tokenizer))

    # Get hidden dim
    qwen_dim = qwen_model.config.hidden_size
    print(f"Qwen hidden dim: {qwen_dim}")

    # --- Build HybridEncoder with Qwen's hidden dim ---
    encoder = build_hybrid_encoder(
        reve_dir=reve_dir,
        decoder_dim=qwen_dim,
        window_size=window_size,
        sfreq=sfreq,
        dropout=dropout,
    )

    # --- Assemble ---
    model = HybridQwenModel(encoder, qwen_model, tokenizer, num_eeg_tokens=num_eeg_tokens)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHybrid Qwen Model:")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  EEG tokens/window: {num_eeg_tokens}")

    return model, tokenizer
