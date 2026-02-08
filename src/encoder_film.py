"""FiLM hybrid encoder: REVE (frozen) + FBCCA with FiLM modulation.

Replaces the concat fusion in encoder_hybrid.py with Feature-wise Linear
Modulation: FBCCA frequency features modulate REVE spatial features via
learned scale (gamma) and shift (beta).

Output: (total_K, N, llm_dim) projected tokens per EEG window.
"""

import torch
import torch.nn as nn

from .fbcca import FBCCAFeatureExtractor
from .model_e2e import REVEWithUnfreeze


class FiLMHybridEncoder(nn.Module):
    """REVE + FBCCA with FiLM modulation and projector to LLM hidden dim.

    FiLM: out = (1 + gamma(fbcca)) * reve + beta(fbcca)
    Initialized to identity (gamma=0, beta=0) so initial output = REVE features.

    Args:
        reve_wrapper: REVEWithUnfreeze instance (frozen)
        fbcca: FBCCAFeatureExtractor instance (zero parameters)
        llm_dim: output dimension matching LLM hidden size (e.g. 2560 for Qwen3-4B)
        reve_dim: REVE embedding dimension (default 512)
        fbcca_dim: FBCCA output dimension (default 200 = 5 bands x 40 freqs)
        dropout: dropout rate for the projector
    """

    def __init__(self, reve_wrapper, fbcca, llm_dim, reve_dim=512, fbcca_dim=200, dropout=0.1):
        super().__init__()
        self.reve = reve_wrapper
        self.fbcca = fbcca
        self.reve_dim = reve_dim
        self.fbcca_dim = fbcca_dim

        # FiLM: FBCCA generates scale and shift for REVE features
        self.film_gamma = nn.Linear(fbcca_dim, reve_dim)
        self.film_beta = nn.Linear(fbcca_dim, reve_dim)

        # Initialize to identity: gamma output ~ 0 (so 1+0=1), beta output ~ 0
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Project fused features to LLM hidden dim
        self.projector = nn.Sequential(
            nn.Linear(reve_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, eeg_windows, output_dtype=None):
        """Encode EEG windows via REVE + FBCCA FiLM fusion + projection.

        Args:
            eeg_windows: (total_K, 62, T) raw EEG windows at 200Hz
            output_dtype: optional dtype cast (e.g. bf16 for mixed precision)

        Returns:
            (total_K, N, llm_dim) projected tokens. N=62 for T=300, N=186 for T=600.
        """
        # REVE: (B, 62, T) -> (B, N, 512) spatial tokens
        reve_out = self.reve(eeg_windows, output_dtype=output_dtype, pool=False)
        B, N, D = reve_out.shape

        # FBCCA: (B, 62, T) -> (B, 200) frequency correlations
        fbcca_out = self.fbcca(eeg_windows)
        if output_dtype is not None:
            fbcca_out = fbcca_out.to(dtype=output_dtype)

        # FiLM modulation: frequency info modulates spatial features
        gamma = 1 + self.film_gamma(fbcca_out)  # (B, 512), starts at 1
        beta = self.film_beta(fbcca_out)          # (B, 512), starts at 0
        fused = gamma.unsqueeze(1) * reve_out + beta.unsqueeze(1)  # (B, N, 512)

        # Project to LLM hidden dim
        return self.projector(fused)  # (B, N, llm_dim)

    @property
    def output_dim(self):
        return self.projector[-1].normalized_shape[0]

    @property
    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_film_encoder(reve_dir="models", llm_dim=2560, window_size=300, sfreq=200.0, dropout=0.1):
    """Build a FiLMHybridEncoder from scratch.

    Args:
        reve_dir: directory containing reve-base/ and reve-positions/
        llm_dim: LLM hidden dimension to project into
        window_size: EEG window size in timepoints
        sfreq: sampling frequency
        dropout: projector dropout

    Returns:
        FiLMHybridEncoder instance (REVE frozen, FBCCA zero-param, FiLM+projector trainable)
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

    # Fully frozen REVE (unfreeze_last_n=0)
    reve_wrapper = REVEWithUnfreeze(
        reve_model, pos_bank,
        channel_names=VALID_CHANNEL_NAMES,
        unfreeze_last_n=0,
    )

    fbcca = FBCCAFeatureExtractor(sfreq=sfreq, n_timepoints=window_size)
    print(f"FBCCA: {fbcca.n_bands} bands x {fbcca.n_freqs} freqs = {fbcca.output_dim} features")

    encoder = FiLMHybridEncoder(
        reve_wrapper=reve_wrapper,
        fbcca=fbcca,
        llm_dim=llm_dim,
        dropout=dropout,
    )

    total = sum(p.numel() for p in encoder.parameters())
    trainable = encoder.trainable_param_count
    print(f"FiLMHybridEncoder: {total:,} total, {trainable:,} trainable (FiLM + projector)")

    return encoder
