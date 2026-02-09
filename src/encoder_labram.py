"""LaBraM wrapper for BCI Agent encoder ablation.

Wraps braindecode's Labram model with the same forward interface as
REVEWithUnfreeze, so it can be used as a drop-in backbone replacement
in FiLMHybridEncoder.

LaBraM (NeurIPS 2024): BEiT-v2 style EEG foundation model with VQ-NSP
pretraining. neural_tokenizer mode produces one token per channel.

Output: (B, 62, 200) per-channel tokens, matching REVE's (B, 62, 512).
"""

import torch.nn as nn


class LaBraMWrapper(nn.Module):
    """Wraps braindecode Labram for use in FiLMHybridEncoder.

    Provides the same forward signature as REVEWithUnfreeze:
        forward(eeg_windows, output_dtype=None, pool=False)
        → (B, N, D) where N=n_chans=62, D=embed_dim=200

    For T=300 with patch_size=200: SegmentPatch takes 1 patch (200pts) per
    channel, truncating the last 100pts. This gives 62 tokens matching REVE.

    Args:
        labram_model: braindecode Labram instance
        unfreeze_last_n: number of transformer blocks to unfreeze from the end
    """

    def __init__(self, labram_model, unfreeze_last_n=0):
        super().__init__()
        self.labram = labram_model
        self.patch_size = labram_model.patch_size

        # Freeze everything first
        self.labram.requires_grad_(False)

        # Optionally unfreeze last N transformer blocks
        if unfreeze_last_n > 0:
            blocks = self.labram.blocks
            n_blocks = len(blocks)
            for i in range(max(0, n_blocks - unfreeze_last_n), n_blocks):
                blocks[i].requires_grad_(True)
            if hasattr(self.labram, "norm"):
                self.labram.norm.requires_grad_(True)
            if hasattr(self.labram, "fc_norm") and self.labram.fc_norm is not None:
                self.labram.fc_norm.requires_grad_(True)

        unfrozen = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        n_blocks = len(self.labram.blocks)
        mode = (
            "fully frozen"
            if unfreeze_last_n == 0
            else f"unfroze last {unfreeze_last_n}/{n_blocks} blocks"
        )
        print(f"LaBraM: {mode}")
        print(f"  Trainable: {unfrozen:,} / {total:,} ({100*unfrozen/total:.1f}%)")

    def forward(self, eeg_windows, output_dtype=None, pool=False):
        """Encode EEG windows via LaBraM.

        Args:
            eeg_windows: (B, 62, T) raw EEG at 200Hz
            output_dtype: optional dtype cast (e.g. bf16)
            pool: if True return (B, 200) pooled; if False return (B, n_chans, 200)

        Returns:
            (B, 200) if pool=True, (B, n_chans, 200) if pool=False
        """
        # No padding: let SegmentPatch truncate to floor(T/patch_size) patches.
        # For T=300, patch_size=200: 1 patch per channel → n_chans tokens.
        # This preserves 1:1 token-per-channel mapping matching REVE's output.
        out = self.labram.forward_features(
            eeg_windows,
            return_patch_tokens=not pool,
        )

        if output_dtype is not None:
            out = out.to(dtype=output_dtype)
        return out

    @property
    def embed_dim(self):
        return self.labram.embed_dim


def build_labram_wrapper(n_chans=62, n_times=300, unfreeze_last_n=0):
    """Build a frozen LaBraMWrapper from pretrained weights.

    With n_times=300 and patch_size=200 (default): 1 patch per channel,
    producing n_chans=62 tokens of embed_dim=200. The last 100 timepoints
    (0.5s) are truncated by SegmentPatch, keeping 1s of signal per patch.

    Pretrained LaBraM has position_embedding (1, 129, 200) for 128 channels
    and temporal_embedding (1, 16, 200) for 16 patches. We create the model
    with our dimensions (62 ch, 1 patch) and slice pretrained embeddings.

    Args:
        n_chans: number of EEG channels
        n_times: window length in timepoints
        unfreeze_last_n: transformer blocks to unfreeze

    Returns:
        LaBraMWrapper instance
    """
    from braindecode.models import Labram
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from .preprocess import VALID_CHANNEL_NAMES

    chs_info = [{"ch_name": name} for name in VALID_CHANNEL_NAMES[:n_chans]]

    print(f"Loading LaBraM pretrained (n_chans={n_chans}, n_times={n_times})...")

    # Create model with our dimensions (not pretrained's 128ch/16patch)
    labram_model = Labram(
        n_chans=n_chans, n_times=n_times, n_outputs=0, chs_info=chs_info,
    )

    # Load pretrained weights, adapting mismatched embedding sizes
    weight_path = hf_hub_download("braindecode/labram-pretrained", "model.safetensors")
    pretrained = load_file(weight_path)
    model_state = labram_model.state_dict()

    for key, src_tensor in pretrained.items():
        if key not in model_state:
            continue
        dst_shape = model_state[key].shape
        if src_tensor.shape == dst_shape:
            model_state[key] = src_tensor
        else:
            # Slice pretrained embedding to fit our smaller dimensions
            # position_embedding: (1,129,200) → (1,63,200)
            # temporal_embedding: (1,16,200) → (1,2,200)
            slices = tuple(slice(0, d) for d in dst_shape)
            model_state[key] = src_tensor[slices]
            print(f"  Adapted {key}: {tuple(src_tensor.shape)} → {dst_shape}")

    labram_model.load_state_dict(model_state)

    n_patches = n_times // labram_model.patch_size
    print(f"LaBraM: patch_size={labram_model.patch_size}, n_patches={n_patches}, "
          f"tokens_per_window={n_chans * n_patches}")

    wrapper = LaBraMWrapper(labram_model, unfreeze_last_n=unfreeze_last_n)
    print(f"LaBraM embed_dim: {wrapper.embed_dim}")

    return wrapper
