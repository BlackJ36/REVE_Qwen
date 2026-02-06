"""End-to-end BCI-Qwen model with REVE fine-tuning."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

from .model import EEGProjector
from .preprocess import VALID_CHANNEL_NAMES
from .tokens import ALL_SPECIAL_TOKENS, BCI_PAD, register_special_tokens


class REVEWithUnfreeze(nn.Module):
    """Wraps REVE model with selective layer unfreezing and position caching.

    REVE architecture: 22 transformer layers at .transformer.layers,
    attention_pooling via cls_query_token, plus to_patch_embedding and fourier4d.
    """

    def __init__(self, reve_model, pos_bank, channel_names=None, unfreeze_last_n=4):
        super().__init__()
        self.reve = reve_model
        self.channel_names = channel_names or VALID_CHANNEL_NAMES

        # Cache electrode positions as buffer (moves with model, no gradient)
        positions = pos_bank(self.channel_names)  # (n_channels, 3)
        self.register_buffer("electrode_positions", positions)

        # Freeze everything first
        self.reve.requires_grad_(False)

        # Unfreeze last N transformer layers
        layers = self._get_layers()
        n_layers = len(layers)
        unfreeze_from = max(0, n_layers - unfreeze_last_n)
        for i in range(unfreeze_from, n_layers):
            layers[i].requires_grad_(True)

        # Unfreeze attention pooling components (cls_query_token, final ln)
        if hasattr(self.reve, "cls_query_token"):
            self.reve.cls_query_token.requires_grad_(True)
        if hasattr(self.reve, "ln"):
            self.reve.ln.requires_grad_(True)

        unfrozen = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"REVE: unfroze last {unfreeze_last_n}/{n_layers} layers + pooling")
        print(f"  Trainable: {unfrozen:,} / {total:,} ({100*unfrozen/total:.1f}%)")

    def _get_layers(self):
        """Discover transformer layers programmatically."""
        for path in ["transformer.layers", "layers", "encoder.layers"]:
            obj = self.reve
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        raise RuntimeError(
            f"Cannot find transformer layers in REVE. "
            f"Top-level modules: {[n for n, _ in self.reve.named_children()]}"
        )

    def forward(self, eeg_tensor):
        """
        Args:
            eeg_tensor: (B, 62, 600) raw preprocessed EEG
        Returns:
            (B, 512) pooled embedding
        """
        B = eeg_tensor.shape[0]
        # Match REVE parameter dtype (bf16 under DeepSpeed/AMP)
        dtype = next(self.reve.parameters()).dtype
        eeg_tensor = eeg_tensor.to(dtype=dtype)
        pos = self.electrode_positions.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
        output = self.reve(eeg_tensor, pos)  # (B, 62, patches, 512)
        pooled = self.reve.attention_pooling(output)  # (B, 512)
        return pooled

    @property
    def embed_dim(self):
        return self.reve.config.embed_dim


class BCIE2EQwenForCausalLM(nn.Module):
    """End-to-end model: raw EEG → REVE → projector → Qwen."""

    def __init__(self, reve_wrapper, qwen_model, tokenizer, projector):
        super().__init__()
        self.reve = reve_wrapper
        self.qwen = qwen_model
        self.tokenizer = tokenizer
        self.projector = projector
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)

    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None, eeg_tensor=None, **kwargs):
        """
        Args:
            input_ids: (B, L) token IDs with <|bci_pad|> placeholders
            attention_mask: (B, L)
            labels: (B, L) with -100 for non-target positions
            eeg_tensor: (B, 62, 600) raw EEG signals
        """
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if eeg_tensor is not None:
            eeg_tensor = eeg_tensor.to(inputs_embeds.device)
            # REVE forward: (B, 62, 600) → (B, 512)
            reve_emb = self.reve(eeg_tensor)
            # Project: (B, 512) → (B, qwen_dim)
            projected = self.projector(reve_emb.to(inputs_embeds.dtype))

            inputs_embeds = inputs_embeds.clone()
            bci_pad_mask = input_ids == self.bci_pad_id
            for i in range(input_ids.size(0)):
                pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    inputs_embeds[i, pad_positions[0]] = projected[i]

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
        self.qwen.save_pretrained(path, **kwargs)
        torch.save(self.projector.state_dict(), f"{path}/projector.pt")


def build_e2e_model(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    from_modelscope=True,
    reve_dir="models",
    reve_dim=512,
    unfreeze_last_n=4,
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.05,
    use_4bit=False,
):
    """Build the full end-to-end BCI-Qwen model.

    Returns (model, tokenizer).
    """
    from pathlib import Path

    # --- Load REVE from local directory ---
    reve_dir = Path(reve_dir)
    print(f"Loading REVE from {reve_dir}...")
    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    reve_model = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )

    reve_wrapper = REVEWithUnfreeze(
        reve_model, pos_bank, channel_names=VALID_CHANNEL_NAMES, unfreeze_last_n=unfreeze_last_n,
    )

    # --- Load Qwen ---
    if from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_name)
    else:
        model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    num_new = register_special_tokens(tokenizer)
    print(f"Added {num_new} special tokens to tokenizer")

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "sequential"
        load_kwargs["max_memory"] = {0: "14GiB", "cpu": "32GiB"}

    qwen_model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    qwen_model.resize_token_embeddings(len(tokenizer))

    if use_4bit:
        from peft import prepare_model_for_kbit_training
        qwen_model = prepare_model_for_kbit_training(
            qwen_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    config = qwen_model.config
    qwen_dim = getattr(config, "hidden_size", None)
    if qwen_dim is None:
        qwen_dim = config.text_config.hidden_size
    print(f"Qwen hidden dim: {qwen_dim}")

    # --- LoRA ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    qwen_model = get_peft_model(qwen_model, lora_config)
    qwen_model.print_trainable_parameters()

    # --- Projector ---
    projector = EEGProjector(reve_dim=reve_dim, qwen_dim=qwen_dim)

    # --- Assemble ---
    model = BCIE2EQwenForCausalLM(reve_wrapper, qwen_model, tokenizer, projector)

    if not use_4bit:
        model.gradient_checkpointing_enable()

    return model, tokenizer
