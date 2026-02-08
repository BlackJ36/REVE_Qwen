"""Hybrid encoder: REVE (frozen) + FBCCA + fusion projector.

Combines REVE time-domain features with FBCCA frequency-domain features
for SSVEP decoding. REVE captures spatial/temporal structure from MAE
pretraining; FBCCA captures SSVEP-specific frequency correlations.

Output: (total_K, 62, decoder_dim) projected tokens per EEG window.
"""

import torch
import torch.nn as nn

from .fbcca import FBCCAFeatureExtractor
from .model_e2e import REVEWithUnfreeze


class HybridEncoder(nn.Module):
    """Combines REVE (frozen) and FBCCA features with a learned fusion projector.

    Args:
        reve_wrapper: REVEWithUnfreeze instance (frozen, reused from model_e2e.py)
        fbcca: FBCCAFeatureExtractor instance (zero parameters)
        decoder_dim: output dimension for the decoder
        reve_dim: REVE embedding dimension (default 512)
        fbcca_dim: FBCCA output dimension (default 200 = 5 bands × 40 freqs)
        dropout: dropout rate for the projector
    """

    def __init__(self, reve_wrapper, fbcca, decoder_dim, reve_dim=512, fbcca_dim=200, dropout=0.1):
        super().__init__()
        self.reve = reve_wrapper
        self.fbcca = fbcca
        self.reve_dim = reve_dim
        self.fbcca_dim = fbcca_dim

        fused_dim = reve_dim + fbcca_dim  # 712
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )

    def forward(self, eeg_windows, output_dtype=None):
        """Encode EEG windows into fused REVE+FBCCA tokens.

        Args:
            eeg_windows: (total_K, 62, T) raw EEG windows
            output_dtype: optional dtype cast (e.g. bf16 for mixed precision)

        Returns:
            (total_K, 62, decoder_dim) projected tokens
        """
        # REVE: (total_K, 62, T) -> (total_K, 62, 512) with pool=False
        # For 1.5s windows (T=300), REVE outputs (B, 62, 1, 512) -> reshape to (B, 62, 512)
        reve_out = self.reve(eeg_windows, output_dtype=output_dtype, pool=False)
        # reve_out shape: (B, N, 512) where N = 62*patches
        # For T=300: N = 62*1 = 62; For T=600: N = 62*3 = 186
        B, N, D = reve_out.shape

        # FBCCA: (total_K, 62, T) -> (total_K, 200)
        fbcca_out = self.fbcca(eeg_windows)  # (B, 200)
        if output_dtype is not None:
            fbcca_out = fbcca_out.to(dtype=output_dtype)

        # Expand FBCCA to match REVE token count: (B, 200) -> (B, N, 200)
        fbcca_expanded = fbcca_out.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: (B, N, 512+200) = (B, N, 712)
        fused = torch.cat([reve_out, fbcca_expanded], dim=-1)

        # Project to decoder dimension: (B, N, decoder_dim)
        return self.projector(fused)

    @property
    def output_dim(self):
        """Output feature dimension per token."""
        return self.projector[-1].normalized_shape[0]

    @property
    def num_tokens_per_window(self):
        """Number of tokens output per EEG window (depends on window size)."""
        # For 1.5s windows (T=300): 62 tokens; for 3s (T=600): 186 tokens
        # This is determined at runtime by REVE's patch_size and overlap
        return None  # Must be determined from forward pass


def build_hybrid_encoder(
    reve_dir="models",
    decoder_dim=512,
    window_size=300,
    sfreq=200.0,
    dropout=0.1,
):
    """Build a HybridEncoder from scratch.

    Args:
        reve_dir: directory containing reve-base/ and reve-positions/
        decoder_dim: output dimension for the decoder
        window_size: EEG window size in timepoints
        sfreq: sampling frequency
        dropout: projector dropout

    Returns:
        HybridEncoder instance
    """
    from pathlib import Path
    from transformers import AutoModel

    from .preprocess import VALID_CHANNEL_NAMES

    reve_dir = Path(reve_dir)
    print(f"Loading REVE from {reve_dir}...")
    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    reve_model = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )

    # Fully frozen REVE
    reve_wrapper = REVEWithUnfreeze(
        reve_model, pos_bank,
        channel_names=VALID_CHANNEL_NAMES,
        unfreeze_last_n=0,
    )

    fbcca = FBCCAFeatureExtractor(sfreq=sfreq, n_timepoints=window_size)
    print(f"FBCCA: {fbcca.n_bands} bands × {fbcca.n_freqs} freqs = {fbcca.output_dim} features")

    encoder = HybridEncoder(
        reve_wrapper=reve_wrapper,
        fbcca=fbcca,
        decoder_dim=decoder_dim,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"HybridEncoder: {total_params:,} total, {trainable_params:,} trainable (projector only)")

    return encoder
