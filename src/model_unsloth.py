"""BCI-Qwen model with Unsloth for efficient fine-tuning."""

# Import unsloth first to enable optimizations
import unsloth  # noqa: F401

import torch
import torch.nn as nn

from .tokens import BCI_PAD, register_special_tokens


class EEGProjector(nn.Module):
    """Projects REVE embeddings (512-dim) into Qwen3-VL embedding space."""

    def __init__(self, reve_dim=512, qwen_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(reve_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, x):
        return self.proj(x)


class BCIQwenUnsloth(nn.Module):
    """Wrapper that injects projected EEG embeddings into Qwen3-VL (Unsloth)."""

    def __init__(self, model, tokenizer, projector):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.projector = projector
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        eeg_embeddings=None,
        **kwargs,
    ):
        embed_layer = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if eeg_embeddings is not None:
            # Move projector to same device as embeddings
            self.projector = self.projector.to(eeg_embeddings.device)
            projected = self.projector(eeg_embeddings)

            bci_pad_mask = input_ids == self.bci_pad_id
            for i in range(input_ids.size(0)):
                pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    inputs_embeds[i, pad_positions[0]] = projected[i]

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return next(self.model.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        pass  # Unsloth handles this

    def save_pretrained(self, path, **kwargs):
        self.model.save_pretrained(path, **kwargs)
        torch.save(self.projector.state_dict(), f"{path}/projector.pt")


def build_model_unsloth(
    model_name="unsloth/Qwen3-VL-4B-Instruct",
    reve_dim=512,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.0,
    max_seq_length=512,
):
    """Build BCI-Qwen model using Unsloth for efficient training.

    Args:
        model_name: Unsloth model name (e.g., unsloth/Qwen3-VL-4B-Instruct)
        reve_dim: REVE embedding dimension
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        max_seq_length: Maximum sequence length

    Returns (model, tokenizer).
    """
    from unsloth import FastVisionModel

    print(f"Loading {model_name} with Unsloth...")

    # Load model with Unsloth (handles 4-bit quantization internally)
    # Note: FastVisionModel returns (model, processor) not (model, tokenizer)
    model, processor = FastVisionModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    # Extract tokenizer from processor
    tokenizer = processor.tokenizer

    # Add special tokens
    num_new = register_special_tokens(tokenizer)
    print(f"Added {num_new} special tokens to tokenizer")

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA with Unsloth
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
        use_gradient_checkpointing="unsloth",
    )

    # Get hidden dim
    config = model.config
    qwen_dim = getattr(config, "hidden_size", None)
    if qwen_dim is None:
        qwen_dim = getattr(config, "text_config", config).hidden_size
    print(f"Qwen hidden dim: {qwen_dim}")

    # Build projector
    projector = EEGProjector(reve_dim=reve_dim, qwen_dim=qwen_dim)

    # Wrap model
    bci_model = BCIQwenUnsloth(model, tokenizer, projector)

    return bci_model, tokenizer
