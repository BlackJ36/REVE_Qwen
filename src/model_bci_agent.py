"""BCI Agent model: FiLM encoder + Qwen3-4B (text-only) with LLaVA-style injection.

Two-stage training:
  Stage 1 (alignment): Qwen frozen (except embed_tokens/lm_head), train encoder only
  Stage 2 (instruction tuning): Qwen LoRA + encoder, mixed EEG/NL data
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .encoder_film import FiLMHybridEncoder
from .tokens import BCI_PAD, register_special_tokens


class BCIAgentModel(nn.Module):
    """FiLMHybridEncoder + Qwen3 text-only with pad replacement.

    Same pad-replacement pattern as BCIE2EQwenForCausalLM: <|bci_pad|> tokens
    in the input sequence are replaced with projected EEG embeddings.

    Args:
        encoder: FiLMHybridEncoder instance
        qwen_model: Qwen3 causal LM (frozen or LoRA-wrapped)
        tokenizer: tokenizer with BCI special tokens registered
        original_vocab_size: vocab size before adding BCI tokens (for partial save)
    """

    def __init__(self, encoder, qwen_model, tokenizer, original_vocab_size=None):
        super().__init__()
        self.encoder = encoder
        self.qwen = qwen_model
        self.tokenizer = tokenizer
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.original_vocab_size = original_vocab_size

    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None,
                eeg_windows=None, window_counts=None, **kwargs):
        """Forward pass with EEG pad replacement.

        Args:
            input_ids: (B, L) token IDs with <|bci_pad|> placeholders
            attention_mask: (B, L)
            labels: (B, L) with -100 for non-target positions
            eeg_windows: (total_K, 62, T) all EEG windows concatenated across batch
            window_counts: (B,) number of EEG windows per sample
        """
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if eeg_windows is not None:
            eeg_windows = eeg_windows.to(device=inputs_embeds.device, dtype=torch.float32)
            # Encode all windows: (total_K, 62, T) -> (total_K, N, llm_dim)
            projected = self.encoder(eeg_windows, output_dtype=inputs_embeds.dtype)

            inputs_embeds = inputs_embeds.clone()
            bci_pad_mask = input_ids == self.bci_pad_id
            B = input_ids.size(0)
            dim = projected.size(-1)

            offset = 0
            for i in range(B):
                K_i = int(window_counts[i])
                # Flatten K_i windows of N tokens each -> (K_i * N, dim)
                sample_tokens = projected[offset:offset + K_i].reshape(-1, dim)
                pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                n = min(len(pad_positions), sample_tokens.size(0))
                inputs_embeds[i, pad_positions[:n]] = sample_tokens[:n].to(inputs_embeds.dtype)
                offset += K_i

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @property
    def config(self):
        return self.qwen.config

    @property
    def device(self):
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, path, **kwargs):
        """Save trainable weights only (encoder + Qwen trainable params).

        Stage 1: saves encoder_trainable.pt + qwen_trainable.pt (~100MB total)
        Stage 2: saves encoder_trainable.pt + LoRA adapter (~150MB total)
        """
        from pathlib import Path as P

        from peft import PeftModel

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save encoder trainable weights (FiLM + projector)
        encoder_state = {
            name: param.data
            for name, param in self.encoder.named_parameters()
            if param.requires_grad
        }
        torch.save(encoder_state, save_dir / "encoder_trainable.pt")

        if isinstance(self.qwen, PeftModel):
            # Stage 2: save LoRA adapter + modules_to_save (embed_tokens/lm_head)
            self.qwen.save_pretrained(str(save_dir), **kwargs)
        else:
            # Stage 1: only save NEW token embedding rows (not full 389M matrix)
            new_token_state = {}
            ovs = self.original_vocab_size
            if ovs is not None:
                embed_w = self.qwen.get_input_embeddings().weight
                new_token_state["embed_tokens.new_rows"] = embed_w.data[ovs:]
                if not getattr(self.qwen.config, "tie_word_embeddings", True):
                    lm_w = self.qwen.lm_head.weight
                    new_token_state["lm_head.new_rows"] = lm_w.data[ovs:]
            torch.save(new_token_state, save_dir / "qwen_trainable.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_dir))


def _get_llm_dim(qwen_model):
    """Extract hidden_size from Qwen model config."""
    config = qwen_model.config
    # For PEFT-wrapped models, check base_model_prefix
    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        return config.hidden_size
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size from config: {config}")


def build_bci_agent_model(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    from_modelscope=True,
    reve_dir="models",
    stage=1,
    # Stage 1 checkpoint (for Stage 2 to load encoder weights)
    stage1_checkpoint=None,
    # LoRA config (Stage 2 only)
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "v_proj"),
    # Encoder config
    encoder_type="reve",
    use_fbcca=True,
    fbcca_mode=None,
    window_size=300,
    sfreq=200.0,
    encoder_dropout=0.1,
):
    """Build the BCI Agent model for Stage 1 or Stage 2.

    Stage 1: Qwen frozen (except embed_tokens), encoder trainable
    Stage 2: Qwen with LoRA, encoder trainable (loaded from Stage 1)

    fbcca_mode controls FBCCA integration:
      - "film": FiLM modulation (FBCCA modulates backbone via gamma/beta)
      - "candidate": FBCCA info via tokens (encoder is backbone-only)
      - "none": no FBCCA at all (pure backbone)
      - None: infer from use_fbcca for backward compatibility

    Returns (model, tokenizer).
    """
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve fbcca_mode (backward compat with use_fbcca boolean)
    if fbcca_mode is None:
        fbcca_mode = "film" if use_fbcca else "none"
    # For candidate mode, encoder runs without FiLM (FBCCA comes as tokens)
    encoder_use_fbcca = (fbcca_mode == "film")

    # --- Load Qwen3-4B (text-only) ---
    if from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_name)
    else:
        model_path = model_name

    print(f"Loading {model_name} from {model_path}...")
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

    llm_dim = _get_llm_dim(qwen_model)
    print(f"Qwen hidden dim: {llm_dim}")

    # --- Freeze strategy ---
    original_vocab_size = len(tokenizer) - num_new  # vocab size before adding BCI tokens

    if stage == 1:
        # Freeze all Qwen params
        qwen_model.requires_grad_(False)
        # Unfreeze embed_tokens so new token rows can train
        embed_weight = qwen_model.get_input_embeddings().weight
        embed_weight.requires_grad_(True)
        # Gradient hook: zero out gradients for original vocab rows
        # Only the num_new new token rows (at the end) receive updates
        def _mask_original_grad(grad):
            grad[:original_vocab_size] = 0
            return grad
        embed_weight.register_hook(_mask_original_grad)

        # If lm_head is separate (not tied), do the same
        if not getattr(qwen_model.config, "tie_word_embeddings", True):
            lm_weight = qwen_model.lm_head.weight
            lm_weight.requires_grad_(True)
            lm_weight.register_hook(_mask_original_grad)

        new_token_params = num_new * llm_dim
        print(f"Stage 1: Qwen frozen, only {num_new} new token embeddings trainable ({new_token_params:,} params)")

    elif stage == 2:
        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            modules_to_save=["embed_tokens", "lm_head"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        qwen_model = get_peft_model(qwen_model, lora_config)
        qwen_model.print_trainable_parameters()
    else:
        raise ValueError(f"Invalid stage: {stage}, must be 1 or 2")

    # --- Build encoder (backbone + optional FBCCA FiLM) ---
    from .encoder_film import build_film_encoder

    encoder = build_film_encoder(
        reve_dir=reve_dir,
        llm_dim=llm_dim,
        window_size=window_size,
        sfreq=sfreq,
        dropout=encoder_dropout,
        encoder_type=encoder_type,
        use_fbcca=encoder_use_fbcca,
    )

    # --- Load Stage 1 weights for Stage 2 ---
    if stage == 2 and stage1_checkpoint is not None:
        s1_dir = Path(stage1_checkpoint)

        # Load encoder weights (FiLM + projector)
        enc_path = s1_dir / "encoder_trainable.pt"
        if enc_path.exists():
            print(f"Loading Stage 1 encoder from {enc_path}")
            state_dict = torch.load(enc_path, map_location="cpu", weights_only=True)
            missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
        else:
            print(f"WARNING: {enc_path} not found, training encoder from scratch")

        # Load trained new-token embedding rows from Stage 1
        qwen_path = s1_dir / "qwen_trainable.pt"
        if qwen_path.exists():
            print(f"Loading Stage 1 embeddings from {qwen_path}")
            qwen_state = torch.load(qwen_path, map_location="cpu", weights_only=True)
            ovs = original_vocab_size
            if "embed_tokens.new_rows" in qwen_state:
                new_rows = qwen_state["embed_tokens.new_rows"]
                qwen_model.get_input_embeddings().weight.data[ovs:] = new_rows
                print(f"  Restored {new_rows.shape[0]} new token embeddings")
            if "lm_head.new_rows" in qwen_state:
                qwen_model.lm_head.weight.data[ovs:] = qwen_state["lm_head.new_rows"]
                print(f"  Restored lm_head new token rows")
        else:
            print(f"WARNING: {qwen_path} not found, special token embeddings untrained")

    # --- Assemble ---
    model = BCIAgentModel(encoder, qwen_model, tokenizer, original_vocab_size)
    model.gradient_checkpointing_enable()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # For Stage 1, effective trainable is much less (gradient hook masks original tokens)
    effective_trainable = trainable
    if stage == 1:
        effective_trainable = trainable - original_vocab_size * llm_dim + num_new * llm_dim
    print(f"\nBCIAgentModel (Stage {stage}):")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,} (effective: {effective_trainable:,})")

    return model, tokenizer
