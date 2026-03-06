"""REVE(9ch) + FiLM(FBCCA) standalone classifier for SSVEP verification.

Verifies whether FBCCA frequency prior via FiLM modulation improves
EEG classification, especially at short windows (0.5-1.5s).

Architecture:
  EEG(62ch) → [select 9ch] → REVE(frozen/partial) → tokens (B, C, H, 512)
  EEG(62ch) → [select 9ch] → FBCCA → (B, 200) → constrained FiLM γ/β
  FiLM: γ * tokens + β → attention_pooling → Linear(512→40)

Constrained FiLM (prevents FBCCA from dominating):
  γ = 1 + scale * tanh(Wγ @ LN(fbcca))
  β = scale * tanh(Wβ @ LN(fbcca))
  scale=0.1 limits modulation amplitude to ±10%.

Without FiLM (baseline):
  EEG(62ch) → [select 9ch] → REVE → attention_pooling → Linear(512→40)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fbcca import FBCCAFeatureExtractor, OCCIPITAL_CHANNELS, resolve_channel_indices
from .model_e2e import REVEWithUnfreeze
from .preprocess import VALID_CHANNEL_NAMES


class FiLMClassifier(nn.Module):
    """REVE(9ch) + optional FiLM(FBCCA) → attention_pooling → Linear head.

    Args:
        reve_wrapper: REVEWithUnfreeze instance (9ch or 62ch)
        n_classes: number of SSVEP target classes
        fbcca: FBCCAFeatureExtractor instance (None if use_film=False)
        backbone_ch_idx: indices to select from 62ch input for REVE (None = use all)
        fbcca_ch_idx: indices to select from 62ch input for FBCCA (None = use all)
        use_film: whether to apply FiLM modulation
        film_scale: amplitude constraint for γ/β (0.1 = ±10% modulation)
        film_reg_weight: regularization weight for ||γ-1||² + ||β||²
        distill_alpha: CE weight in distillation loss (1-α for KL)
        distill_temp: temperature for KD softmax
    """

    def __init__(
        self,
        reve_wrapper,
        n_classes=40,
        fbcca=None,
        backbone_ch_idx=None,
        fbcca_ch_idx=None,
        use_film=True,
        film_scale=0.1,
        film_reg_weight=0.01,
        distill_alpha=0.5,
        distill_temp=2.0,
        gamma_mode="tanh",
        use_token_gate=False,
        n_backbone_ch=9,
        dropout=0.0,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.reve = reve_wrapper
        self.use_film = use_film and fbcca is not None
        self.gamma_mode = gamma_mode
        self.label_smoothing = label_smoothing

        if backbone_ch_idx is not None:
            self.register_buffer("backbone_ch_idx", torch.tensor(backbone_ch_idx, dtype=torch.long))
        else:
            self.backbone_ch_idx = None

        if self.use_film:
            self.fbcca = fbcca
            if fbcca_ch_idx is not None:
                self.register_buffer("fbcca_ch_idx", torch.tensor(fbcca_ch_idx, dtype=torch.long))
            else:
                self.fbcca_ch_idx = None

            fbcca_dim = fbcca.output_dim  # 200
            backbone_dim = 512
            self.film_scale = film_scale

            # Constrained FiLM: LN → Linear → activation
            self.film_ln = nn.LayerNorm(fbcca_dim)
            self.film_gamma_proj = nn.Linear(fbcca_dim, backbone_dim)
            self.film_beta_proj = nn.Linear(fbcca_dim, backbone_dim)

            # Zero init → identity modulation at start
            nn.init.zeros_(self.film_gamma_proj.weight)
            nn.init.zeros_(self.film_gamma_proj.bias)
            nn.init.zeros_(self.film_beta_proj.weight)
            nn.init.zeros_(self.film_beta_proj.bias)

            # Token gate: per-channel importance weight
            self.use_token_gate = use_token_gate
            if use_token_gate:
                self.token_gate_proj = nn.Linear(fbcca_dim, n_backbone_ch)
                nn.init.zeros_(self.token_gate_proj.weight)
                nn.init.zeros_(self.token_gate_proj.bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(512, n_classes)

        self.distill_alpha = distill_alpha
        self.distill_temp = distill_temp
        self.film_reg_weight = film_reg_weight

        # For logging
        self._last_gamma = None
        self._last_beta = None

    def forward(self, eeg):
        """Forward: EEG → REVE tokens → optional FiLM → attention_pooling → head.

        Args:
            eeg: (B, 62, T) full-channel EEG at 200Hz

        Returns:
            logits: (B, n_classes)
        """
        # Channel selection for REVE
        if self.backbone_ch_idx is not None:
            reve_input = eeg[:, self.backbone_ch_idx, :]
        else:
            reve_input = eeg

        if self.use_film:
            # Get 4D tokens for FiLM modulation
            tokens_4d = self.reve(reve_input, pool="4d")  # (B, C, H, E)

            # FBCCA channel selection
            if self.fbcca_ch_idx is not None:
                fbcca_input = eeg[:, self.fbcca_ch_idx, :]
            else:
                fbcca_input = eeg
            fbcca_out = self.fbcca(fbcca_input)  # (B, 200)

            # FiLM modulation
            h = self.film_ln(fbcca_out)
            raw_gamma = self.film_gamma_proj(h)  # (B, 512)
            raw_beta = self.film_beta_proj(h)    # (B, 512)

            if self.gamma_mode == "sigmoid":
                # gamma in [1-scale, 1+scale] via sigmoid
                gamma = (1 - self.film_scale) + 2 * self.film_scale * torch.sigmoid(raw_gamma)
                beta = self.film_scale * torch.tanh(raw_beta)
            else:
                # Default tanh: gamma in [1-scale, 1+scale]
                gamma = 1 + self.film_scale * torch.tanh(raw_gamma)
                beta = self.film_scale * torch.tanh(raw_beta)

            self._last_gamma = gamma
            self._last_beta = beta

            # Modulate: broadcast over C and H dims
            modulated = gamma[:, None, None, :] * tokens_4d + beta[:, None, None, :]

            # Token gate: per-channel importance weighting
            if self.use_token_gate:
                gate = torch.sigmoid(self.token_gate_proj(h))  # (B, n_ch)
                modulated = gate[:, :, None, None] * modulated  # (B, C, H, E)

            # Pool using REVE's attention_pooling
            pooled = self.reve.reve.attention_pooling(modulated)  # (B, 512)
        else:
            # Baseline: direct REVE pooling
            pooled = self.reve(reve_input, pool=True)  # (B, 512)

        return self.head(self.dropout(pooled))  # (B, 40)

    def compute_loss(self, logits, hard_labels, etrca_scores=None):
        """CE loss + optional eTRCA distillation + FiLM regularization."""
        ce = F.cross_entropy(logits, hard_labels, label_smoothing=self.label_smoothing)

        loss = ce
        if etrca_scores is not None and self.distill_alpha < 1.0:
            T = self.distill_temp
            alpha = self.distill_alpha
            kl = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(etrca_scores / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)
            loss = alpha * ce + (1 - alpha) * kl

        # FiLM regularization: penalize deviation from identity
        if self.use_film and self.film_reg_weight > 0 and self._last_gamma is not None:
            reg = (self._last_gamma - 1).pow(2).mean() + self._last_beta.pow(2).mean()
            loss = loss + self.film_reg_weight * reg

        return loss

    @property
    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_film_classifier(
    reve_dir="models",
    n_classes=40,
    trial_pts=600,
    sfreq=200.0,
    use_film=True,
    unfreeze_last_n=0,
    film_scale=0.1,
    film_reg_weight=0.01,
    distill_alpha=0.5,
    distill_temp=2.0,
    gamma_mode="tanh",
    use_token_gate=False,
    dropout=0.0,
    label_smoothing=0.0,
    backbone_channels=None,
    lora_rank=0,
    lora_alpha=16,
):
    """Build FiLMClassifier with REVE backbone.

    Args:
        reve_dir: directory containing reve-base/ and reve-positions/
        n_classes: number of target classes
        trial_pts: number of timepoints per trial (for FBCCA template precomputation)
        sfreq: sampling frequency
        use_film: whether to enable FiLM modulation
        unfreeze_last_n: number of REVE transformer layers to unfreeze (0 = fully frozen)
        film_scale: FiLM amplitude constraint
        film_reg_weight: FiLM regularization weight
        distill_alpha: CE weight in distillation loss
        distill_temp: KD temperature

    Returns:
        FiLMClassifier instance
    """
    from pathlib import Path
    from transformers import AutoModel

    reve_dir = Path(reve_dir)

    # Load REVE
    print(f"Loading REVE from {reve_dir}...")
    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    reve_model = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )

    # Channel selection for REVE backbone
    if backbone_channels is None:
        backbone_channels = OCCIPITAL_CHANNELS
    backbone_ch_idx = resolve_channel_indices(VALID_CHANNEL_NAMES, backbone_channels)
    reve_channel_names = [VALID_CHANNEL_NAMES[i] for i in backbone_ch_idx]
    print(f"REVE backbone: {len(reve_channel_names)} channels {reve_channel_names}")

    reve_wrapper = REVEWithUnfreeze(
        reve_model, pos_bank,
        channel_names=reve_channel_names,
        unfreeze_last_n=0 if lora_rank > 0 else unfreeze_last_n,
    )
    if lora_rank > 0:
        reve_wrapper.inject_lora(rank=lora_rank, alpha=lora_alpha)

    # FBCCA channels: use same as backbone if specified, else default occipital
    fbcca_channels = backbone_channels if backbone_channels is not None else OCCIPITAL_CHANNELS
    fbcca_ch_idx = resolve_channel_indices(VALID_CHANNEL_NAMES, fbcca_channels)

    # Build FBCCA
    fbcca = None
    if use_film:
        fbcca = FBCCAFeatureExtractor(sfreq=sfreq, n_timepoints=trial_pts)
        print(f"FBCCA: {fbcca.n_bands} bands x {fbcca.n_freqs} freqs = {fbcca.output_dim}d, T={trial_pts}")

    classifier = FiLMClassifier(
        reve_wrapper=reve_wrapper,
        n_classes=n_classes,
        fbcca=fbcca,
        backbone_ch_idx=backbone_ch_idx,
        fbcca_ch_idx=fbcca_ch_idx if use_film else None,
        use_film=use_film,
        film_scale=film_scale,
        film_reg_weight=film_reg_weight,
        distill_alpha=distill_alpha,
        distill_temp=distill_temp,
        gamma_mode=gamma_mode,
        use_token_gate=use_token_gate,
        n_backbone_ch=len(backbone_ch_idx),
        dropout=dropout,
        label_smoothing=label_smoothing,
    )

    total = sum(p.numel() for p in classifier.parameters())
    trainable = classifier.trainable_param_count
    if use_film:
        extras = []
        if gamma_mode != "tanh":
            extras.append(f"gamma={gamma_mode}")
        if use_token_gate:
            extras.append("token_gate")
        extra_str = f", {', '.join(extras)}" if extras else ""
        mode = f"FiLM(scale={film_scale}{extra_str})"
    else:
        mode = "baseline"
    unfreeze_str = f", unfreeze={unfreeze_last_n}" if unfreeze_last_n > 0 else ""
    print(f"\nFiLMClassifier ({mode}{unfreeze_str}): {total:,} total, {trainable:,} trainable")

    return classifier
