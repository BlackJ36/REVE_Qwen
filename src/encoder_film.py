"""FiLM hybrid encoder: backbone (REVE or LaBraM) + optional FBCCA FiLM modulation.

Replaces the concat fusion in encoder_hybrid.py with Feature-wise Linear
Modulation: FBCCA frequency features modulate backbone spatial features via
learned scale (gamma) and shift (beta).

Supports:
  - encoder_type="reve": REVE (512d) as backbone
  - encoder_type="labram": LaBraM (200d) as backbone
  - use_fbcca=True/False: enable/disable FBCCA FiLM modulation

Output: (total_K, N, llm_dim) projected tokens per EEG window.
"""

import torch.nn as nn

from .fbcca import FBCCAFeatureExtractor
from .model_e2e import REVEWithUnfreeze


class FiLMHybridEncoder(nn.Module):
    """Backbone + optional FBCCA FiLM modulation + projector to LLM hidden dim.

    FiLM: out = (1 + gamma(fbcca)) * backbone + beta(fbcca)
    Initialized to identity (gamma=0, beta=0) so initial output = backbone features.

    When use_fbcca=False, skips FiLM entirely: out = projector(backbone).

    Args:
        backbone: REVEWithUnfreeze or LaBraMWrapper (same forward interface)
        fbcca: FBCCAFeatureExtractor instance or None (when use_fbcca=False)
        llm_dim: output dimension matching LLM hidden size (e.g. 2560 for Qwen3-4B)
        backbone_dim: backbone embedding dimension (512 for REVE, 200 for LaBraM)
        fbcca_dim: FBCCA output dimension (default 200 = 5 bands x 40 freqs)
        dropout: dropout rate for the projector
    """

    def __init__(self, backbone, fbcca, llm_dim, backbone_dim=512, fbcca_dim=200, dropout=0.3,
                 backbone_channel_indices=None, fbcca_channel_indices=None):
        super().__init__()
        self.reve = backbone  # kept as "reve" for checkpoint compatibility
        self.fbcca = fbcca
        self.backbone_dim = backbone_dim
        self.fbcca_dim = fbcca_dim
        self.use_fbcca = fbcca is not None

        # Decoupled channel selection: REVE and FBCCA can use different channels
        # - backbone_channel_indices: channels for REVE (None = all 62)
        # - fbcca_channel_indices: channels for FBCCA (None = all, but 9ch is optimal)
        import torch
        if backbone_channel_indices is not None:
            self.register_buffer("backbone_ch_idx", torch.tensor(backbone_channel_indices, dtype=torch.long))
        else:
            self.backbone_ch_idx = None
        if fbcca_channel_indices is not None:
            self.register_buffer("fbcca_ch_idx", torch.tensor(fbcca_channel_indices, dtype=torch.long))
        else:
            self.fbcca_ch_idx = None

        if self.use_fbcca:
            # FiLM: FBCCA generates scale and shift for backbone features
            self.film_gamma = nn.Linear(fbcca_dim, backbone_dim)
            self.film_beta = nn.Linear(fbcca_dim, backbone_dim)

            # Initialize to identity: gamma output ~ 0 (so 1+0=1), beta output ~ 0
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        # Project fused features to LLM hidden dim
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, eeg_windows, output_dtype=None):
        """Encode EEG windows via backbone + optional FBCCA FiLM + projection.

        Args:
            eeg_windows: (total_K, 62, T) raw EEG windows at 200Hz (full channels)
            output_dtype: optional dtype cast (e.g. bf16 for mixed precision)

        Returns:
            (total_K, N, llm_dim) projected tokens. N=n_channels typically.
        """
        # Channel selection for backbone (e.g. 62ch -> 9ch occipital)
        if self.backbone_ch_idx is not None:
            backbone_input = eeg_windows[:, self.backbone_ch_idx, :]
        else:
            backbone_input = eeg_windows

        # Backbone: (B, C, T) -> (B, N, backbone_dim)
        backbone_out = self.reve(backbone_input, output_dtype=output_dtype, pool=False)

        if self.use_fbcca:
            # Channel selection for FBCCA (always 9ch occipital for best accuracy)
            if self.fbcca_ch_idx is not None:
                fbcca_input = eeg_windows[:, self.fbcca_ch_idx, :]
            else:
                fbcca_input = eeg_windows
            # FBCCA: (B, C, T) -> (B, 200) frequency correlations
            fbcca_out = self.fbcca(fbcca_input)
            if output_dtype is not None:
                fbcca_out = fbcca_out.to(dtype=output_dtype)

            # FiLM modulation: frequency info modulates spatial features
            gamma = 1 + self.film_gamma(fbcca_out)    # (B, backbone_dim)
            beta = self.film_beta(fbcca_out)            # (B, backbone_dim)
            fused = gamma.unsqueeze(1) * backbone_out + beta.unsqueeze(1)
        else:
            fused = backbone_out

        # Project to LLM hidden dim (match projector weight dtype for mixed precision)
        proj_dtype = next(self.projector.parameters()).dtype
        return self.projector(fused.to(dtype=proj_dtype))  # (B, N, llm_dim)

    @property
    def output_dim(self):
        return self.projector[-1].normalized_shape[0]

    @property
    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_film_encoder(
    reve_dir="models",
    llm_dim=2560,
    window_size=300,
    sfreq=200.0,
    dropout=0.1,
    encoder_type="reve",
    use_fbcca=True,
    n_chans=62,
    unfreeze_last_n=0,
    reve_finetune_dir=None,
    occipital_only=False,
    reve_merged_ckpt=None,
):
    """Build a FiLMHybridEncoder with configurable backbone and FBCCA.

    Args:
        reve_dir: directory containing reve-base/ and reve-positions/ (for encoder_type="reve")
        llm_dim: LLM hidden dimension to project into
        window_size: EEG window size in timepoints
        sfreq: sampling frequency
        dropout: projector dropout
        encoder_type: "reve" or "labram"
        use_fbcca: whether to include FBCCA FiLM modulation
        n_chans: number of EEG channels
        reve_finetune_dir: directory containing REVE LoRA + pooling from finetune_reve.py.
            When provided, loads and merges LoRA into REVE base weights (zero runtime overhead).
            Only applicable for encoder_type="reve".
        occipital_only: if True, use only 9 occipital channels for REVE (from 62ch input).
            Reduces noise for SSVEP but fewer tokens. Only for encoder_type="reve".
        reve_merged_ckpt: path to merged FiLMClassifier checkpoint (from LoRA merge).
            Loads REVE backbone weights (transformer + pooling) into the encoder.
            Forces occipital_only=True since merged checkpoint is 9ch.

    Returns:
        FiLMHybridEncoder instance
    """
    # --- Build backbone ---
    if encoder_type == "reve":
        from pathlib import Path

        from transformers import AutoModel

        from .fbcca import OCCIPITAL_CHANNELS, resolve_channel_indices
        from .preprocess import VALID_CHANNEL_NAMES

        # Merged checkpoint forces 9ch occipital
        if reve_merged_ckpt is not None:
            occipital_only = True

        reve_dir = Path(reve_dir)
        print(f"Loading REVE from {reve_dir}...")
        pos_bank = AutoModel.from_pretrained(
            str(reve_dir / "reve-positions"), trust_remote_code=True,
        )
        reve_model = AutoModel.from_pretrained(
            str(reve_dir / "reve-base"), trust_remote_code=True,
        )

        # Channel selection: 9 occipital or all 62
        channel_indices = None
        if occipital_only:
            channel_indices = resolve_channel_indices(VALID_CHANNEL_NAMES, OCCIPITAL_CHANNELS)
            reve_channel_names = [VALID_CHANNEL_NAMES[i] for i in channel_indices]
            print(f"REVE occipital-only: {len(reve_channel_names)} channels {reve_channel_names}")
        else:
            reve_channel_names = VALID_CHANNEL_NAMES

        backbone = REVEWithUnfreeze(
            reve_model, pos_bank,
            channel_names=reve_channel_names,
            unfreeze_last_n=unfreeze_last_n,
        )
        backbone_dim = 512

        # Load merged REVE weights (from FiLMClassifier LoRA merge)
        if reve_merged_ckpt is not None:
            import torch

            print(f"Loading merged REVE from {reve_merged_ckpt}")
            ckpt = torch.load(reve_merged_ckpt, map_location="cpu", weights_only=True)
            # Extract REVE keys: "reve.reve.X" → "reve.X" for REVEWithUnfreeze
            reve_state = {}
            for k, v in ckpt.items():
                if k.startswith("reve."):
                    reve_state[k[len("reve."):]] = v
            missing, unexpected = backbone.load_state_dict(reve_state, strict=False)
            loaded = len(reve_state) - len(unexpected)
            print(f"  Loaded {loaded} REVE tensors (skipped {len(unexpected)} unexpected)")
            if missing:
                print(f"  Missing: {missing}")
            backbone.reve.requires_grad_(False)
            print("  REVE re-frozen with merged weights")

        # Load and merge fine-tuned REVE LoRA (from finetune_reve.py)
        elif reve_finetune_dir is not None:
            import torch

            from peft import PeftModel as PeftModelClass

            ft_dir = Path(reve_finetune_dir)
            lora_dir = ft_dir / "reve_lora"
            pooling_path = ft_dir / "reve_pooling.pt"

            if lora_dir.exists():
                print(f"Loading fine-tuned REVE LoRA from {lora_dir}")
                backbone.reve = PeftModelClass.from_pretrained(backbone.reve, str(lora_dir))
                backbone.reve = backbone.reve.merge_and_unload()
                print("  Merged LoRA into REVE base weights")
            else:
                print(f"WARNING: REVE LoRA not found at {lora_dir}")

            if pooling_path.exists():
                print(f"Loading fine-tuned pooling from {pooling_path}")
                pooling_state = torch.load(pooling_path, map_location="cpu", weights_only=True)
                for name, param in pooling_state.items():
                    parts = name.split(".")
                    obj = backbone.reve
                    for part in parts:
                        obj = getattr(obj, part)
                    obj.data.copy_(param)
                print("  Restored cls_query_token + ln")

            # Re-freeze everything (fine-tuned weights are now baked in)
            backbone.reve.requires_grad_(False)
            print("  REVE re-frozen with fine-tuned weights (zero runtime overhead)")

    elif encoder_type == "labram":
        from .encoder_labram import build_labram_wrapper

        backbone = build_labram_wrapper(
            n_chans=n_chans,
            n_times=window_size,
            unfreeze_last_n=unfreeze_last_n,
        )
        backbone_dim = backbone.embed_dim  # 200

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type!r}, must be 'reve' or 'labram'")

    # --- Build FBCCA (optional) ---
    fbcca = None
    if use_fbcca:
        fbcca = FBCCAFeatureExtractor(sfreq=sfreq, n_timepoints=window_size)
        print(f"FBCCA: {fbcca.n_bands} bands x {fbcca.n_freqs} freqs = {fbcca.output_dim} features")
    else:
        print("FBCCA: disabled (backbone-only mode)")

    # --- Channel indices for FBCCA (always 9ch occipital for optimal accuracy) ---
    fbcca_channel_indices = None
    if encoder_type == "reve" and fbcca is not None:
        from .preprocess import VALID_CHANNEL_NAMES as _ALL_NAMES
        fbcca_channel_indices = resolve_channel_indices(_ALL_NAMES, OCCIPITAL_CHANNELS)
        print(f"FBCCA channel selection: {len(fbcca_channel_indices)} occipital channels")

    # --- Assemble ---
    encoder = FiLMHybridEncoder(
        backbone=backbone,
        fbcca=fbcca,
        llm_dim=llm_dim,
        backbone_dim=backbone_dim,
        dropout=dropout,
        backbone_channel_indices=channel_indices if encoder_type == "reve" else None,
        fbcca_channel_indices=fbcca_channel_indices,
    )

    total = sum(p.numel() for p in encoder.parameters())
    trainable = encoder.trainable_param_count
    fbcca_str = "+FBCCA FiLM" if use_fbcca else " only"
    print(f"FiLMHybridEncoder ({encoder_type}{fbcca_str}): {total:,} total, {trainable:,} trainable")

    return encoder
