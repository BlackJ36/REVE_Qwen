"""Plan A: Custom Transformer decoder with HybridEncoder for SSVEP decoding.

Architecture (~20M params):
  - HybridEncoder (REVE frozen + FBCCA): projects EEG to (K, 62, 512) tokens
  - Decoder-only Transformer (6 layers, 512-dim, 8 heads): causal LM on 43-token vocab
  - Same pad-replacement pattern as BCIE2EQwenForCausalLM

Vocab (43 tokens):
  - 40 target tokens: <|t01|> ... <|t40|>
  - <|bci_trans|>: transition separator between spells
  - <|bos|>: beginning of sequence
  - <|pad|>: padding token
"""

import math

import torch
import torch.nn as nn

from .encoder_hybrid import HybridEncoder


# Tiny vocab for Plan A
VOCAB_BOS = 0
VOCAB_PAD = 1
VOCAB_TRANS = 2
VOCAB_TARGETS_START = 3  # targets occupy indices 3..42
VOCAB_SIZE = 43  # 3 control + 40 targets

# Token ID for EEG placeholder (replaced during forward)
VOCAB_EEG_PAD = 1  # reuse pad token ID for EEG placeholders


class SSVEPTransformerDecoder(nn.Module):
    """Custom decoder-only Transformer for SSVEP sequence decoding.

    Small, efficient model (~20M params) with a tiny 43-token vocabulary.
    Uses the same causal masking as standard GPT-style models.

    Args:
        vocab_size: vocabulary size (default 43)
        d_model: model dimension (default 512)
        nhead: number of attention heads (default 8)
        num_layers: number of transformer layers (default 6)
        dim_feedforward: feedforward dimension (default 2048)
        dropout: dropout rate (default 0.1)
        max_seq_len: maximum sequence length (default 1024)
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embed and lm_head share weights
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs_embeds, attention_mask=None):
        """Forward pass with pre-computed embeddings (after EEG replacement).

        Args:
            inputs_embeds: (B, L, d_model) token embeddings
            attention_mask: (B, L) 1 for valid, 0 for padding

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L, D = inputs_embeds.shape

        # Add positional embeddings
        positions = torch.arange(L, device=inputs_embeds.device)
        pos_emb = self.pos_embed(positions)  # (L, D)
        hidden = inputs_embeds + pos_emb.unsqueeze(0)

        # Create causal mask: (L, L) upper-triangle = -inf
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )

        # Padding mask: (B, L) -> True means IGNORE
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        return logits

    def get_input_embeddings(self):
        return self.embed


class HybridTransformerModel(nn.Module):
    """Full Plan A model: HybridEncoder + SSVEPTransformerDecoder.

    Uses the same pad-replacement pattern as BCIE2EQwenForCausalLM:
    EEG placeholder tokens in the input sequence are replaced with
    projected encoder outputs during forward pass.

    Args:
        encoder: HybridEncoder instance
        decoder: SSVEPTransformerDecoder instance
        eeg_pad_id: token ID for EEG placeholders in input_ids
    """

    def __init__(self, encoder, decoder, eeg_pad_id=VOCAB_EEG_PAD):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eeg_pad_id = eeg_pad_id
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None,
                eeg_windows=None, window_counts=None, **kwargs):
        """Forward pass with EEG pad replacement.

        Args:
            input_ids: (B, L) token IDs with EEG pad placeholders
            attention_mask: (B, L)
            labels: (B, L) with -100 for non-target positions
            eeg_windows: (total_K, 62, T) all EEG windows concatenated
            window_counts: (B,) number of windows per sample
        """
        embed_layer = self.decoder.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # (B, L, d_model)

        if eeg_windows is not None:
            eeg_windows = eeg_windows.to(device=inputs_embeds.device, dtype=torch.float32)
            # Encode all windows: (total_K, 62, T) -> (total_K, N, d_model)
            projected = self.encoder(eeg_windows, output_dtype=inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.clone()
            eeg_pad_mask = input_ids == self.eeg_pad_id
            B = input_ids.size(0)
            d_model = projected.size(-1)

            offset = 0
            for i in range(B):
                K_i = int(window_counts[i])
                sample_tokens = projected[offset:offset + K_i].reshape(-1, d_model)
                pad_positions = eeg_pad_mask[i].nonzero(as_tuple=True)[0]
                n = min(len(pad_positions), sample_tokens.size(0))
                inputs_embeds[i, pad_positions[:n]] = sample_tokens[:n]
                offset += K_i

        logits = self.decoder(inputs_embeds, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, self.decoder.vocab_size),
                shift_labels.view(-1),
            )

        return type("Output", (), {"loss": loss, "logits": logits})()

    @property
    def config(self):
        """Minimal config for Trainer compatibility."""
        return type("Config", (), {"is_encoder_decoder": False})()

    @property
    def device(self):
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        pass  # Small model, no need for gradient checkpointing

    def save_pretrained(self, path, **kwargs):
        """Save model weights."""
        from pathlib import Path as P
        P(path).mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder_projector": self.encoder.projector.state_dict(),
            "decoder": self.decoder.state_dict(),
        }, P(path) / "hybrid_model.pt")


def build_hybrid_model(
    reve_dir="models",
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    window_size=300,
    sfreq=200.0,
    max_seq_len=1024,
):
    """Build the full Plan A hybrid model.

    Returns:
        HybridTransformerModel instance
    """
    from .encoder_hybrid import build_hybrid_encoder

    encoder = build_hybrid_encoder(
        reve_dir=reve_dir,
        decoder_dim=d_model,
        window_size=window_size,
        sfreq=sfreq,
        dropout=dropout,
    )

    decoder = SSVEPTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )

    model = HybridTransformerModel(encoder, decoder)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHybrid Transformer Model:")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Decoder: {num_layers} layers, d_model={d_model}, heads={nhead}")
    print(f"  Vocab: {VOCAB_SIZE} tokens")

    return model
