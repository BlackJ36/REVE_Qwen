"""BCI-Qwen model: REVE projection + Qwen3-VL with LoRA."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from .tokens import ALL_SPECIAL_TOKENS, BCI_PAD, register_special_tokens


class EEGProjector(nn.Module):
    """Projects REVE embeddings (512-dim) into Qwen3-VL embedding space (3584-dim)."""

    def __init__(self, reve_dim=512, qwen_dim=3584):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(reve_dim, qwen_dim // 2),
            nn.GELU(),
            nn.Linear(qwen_dim // 2, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, x):
        return self.proj(x)


class BCIQwenForCausalLM(nn.Module):
    """Wrapper that injects projected EEG embeddings into Qwen3-VL."""

    def __init__(self, qwen_model, tokenizer, projector):
        super().__init__()
        self.qwen = qwen_model
        self.tokenizer = tokenizer
        self.projector = projector
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)

    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        eeg_embeddings=None,
        **kwargs,
    ):
        """
        Args:
            input_ids: (B, L) token IDs with <|bci_pad|> placeholders
            attention_mask: (B, L)
            labels: (B, L) with -100 for non-target positions
            eeg_embeddings: (B, reve_dim) pooled REVE embedding per sample
        """
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if eeg_embeddings is not None:
            # Ensure projector is on same device as embeddings
            eeg_embeddings = eeg_embeddings.to(inputs_embeds.device)
            self.projector = self.projector.to(inputs_embeds.device)
            projected = self.projector(eeg_embeddings)  # (B, qwen_dim)

            # Clone to avoid in-place modification issues with gradient flow
            inputs_embeds = inputs_embeds.clone()

            # Replace <|bci_pad|> placeholder embeddings with projected EEG
            bci_pad_mask = input_ids == self.bci_pad_id  # (B, L)
            for i in range(input_ids.size(0)):
                pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    # Stage 1: single embedding per trial → fill first pad position
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


def build_model(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    from_modelscope=True,
    reve_dim=512,
    lora_rank=64,
    lora_alpha=128,
    lora_dropout=0.05,
    use_4bit=False,
):
    """Build the full BCI-Qwen model with LoRA.

    Args:
        use_4bit: If True, load model with 4-bit quantization for memory efficiency.
                  Requires ~6GB VRAM instead of ~16GB. Suitable for single 16GB GPU.

    Returns (model, tokenizer).
    """
    # Load tokenizer
    if from_modelscope:
        from modelscope import snapshot_download

        model_path = snapshot_download(model_name)
    else:
        model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )

    # Add special tokens
    num_new = register_special_tokens(tokenizer)
    print(f"Added {num_new} special tokens to tokenizer")

    # Configure quantization if requested
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
        )
        print("Using 4-bit quantization (NF4 + double quant)")

    # Load base model (Qwen3VLForConditionalGeneration via Auto class)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        # Use sequential device map for more controlled loading
        load_kwargs["device_map"] = "sequential"
        load_kwargs["max_memory"] = {0: "14GiB", "cpu": "32GiB"}
        # Enable disk offload for extreme memory situations
        load_kwargs["offload_folder"] = "/tmp/offload"

    qwen_model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)

    # Resize embeddings for new tokens
    qwen_model.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training if using quantization
    if use_4bit:
        qwen_model = prepare_model_for_kbit_training(
            qwen_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    # Get hidden dim from model config (may be nested under text_config for VL models)
    config = qwen_model.config
    qwen_dim = getattr(config, "hidden_size", None)
    if qwen_dim is None:
        qwen_dim = config.text_config.hidden_size
    print(f"Qwen hidden dim: {qwen_dim}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",  # PEFT uses CAUSAL_LM for VL models' language head too
        bias="none",
    )
    qwen_model = get_peft_model(qwen_model, lora_config)
    qwen_model.print_trainable_parameters()

    # Build projection layer
    projector = EEGProjector(reve_dim=reve_dim, qwen_dim=qwen_dim)

    # Assemble
    model = BCIQwenForCausalLM(qwen_model, tokenizer, projector)

    # Enable gradient checkpointing (skip if already done for 4-bit)
    if not use_4bit:
        model.gradient_checkpointing_enable()

    return model, tokenizer
