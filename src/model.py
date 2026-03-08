"""BCI-Qwen model: EEG projector + Qwen3-4B with LLaVA-style pad replacement.

Pre-extracted REVE embeddings (T tokens × 512d) are projected to Qwen's
hidden dimension and injected at <|bci_pad|> positions.
T depends on extraction config: 9 (200pts/1s) or 18 (400pts/2s).

Two-stage training:
  Stage 1: Qwen frozen (except new token embeddings), train projector only
  Stage 2: Qwen LoRA + projector, mixed EEG/NL data
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .tokens import BCI_PAD, register_special_tokens


class EEGProjector(nn.Module):
    """Projects REVE channel embeddings (512d) into Qwen embedding space."""

    def __init__(self, reve_dim=512, qwen_dim=2560):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(reve_dim, qwen_dim // 2),
            nn.GELU(),
            nn.Linear(qwen_dim // 2, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, x):
        """x: (..., 512) → (..., qwen_dim). Works for any batch dims."""
        return self.proj(x)


class BCIQwenModel(nn.Module):
    """EEG projector + Qwen3-4B with pad replacement.

    Each sample has T <|bci_pad|> tokens in the input sequence. These are
    replaced with projected REVE embeddings (T = n_eeg_tokens from extraction).
    """

    def __init__(self, qwen_model, tokenizer, projector, original_vocab_size=None):
        super().__init__()
        self.qwen = qwen_model
        self.tokenizer = tokenizer
        self.projector = projector
        self.bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        self.original_vocab_size = original_vocab_size

    def get_input_embeddings(self):
        return self.qwen.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None,
                eeg_embeddings=None, eeg_char_counts=None, **kwargs):
        """
        Args:
            input_ids: (B, L) token IDs with <|bci_pad|> placeholders
            attention_mask: (B, L)
            labels: (B, L) with -100 for non-target positions
            eeg_embeddings: S1: (B, T, 512) single trial per sample
                           S2: (total_chars, T, 512) flattened across batch
            eeg_char_counts: S2 only: (B,) number of EEG characters per sample
        """
        embed_layer = self.qwen.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if eeg_embeddings is not None:
            eeg_embeddings = eeg_embeddings.to(device=inputs_embeds.device)
            projected = self.projector(eeg_embeddings)  # (..., 9, qwen_dim)

            inputs_embeds = inputs_embeds.clone()
            bci_pad_mask = input_ids == self.bci_pad_id  # (B, L)
            B = input_ids.size(0)

            if eeg_char_counts is not None:
                # S2: variable-length, flattened embeddings
                # projected: (total_chars, 9, qwen_dim)
                offset = 0
                for i in range(B):
                    n_chars = int(eeg_char_counts[i])
                    if n_chars == 0:
                        continue
                    # (n_chars, 9, qwen_dim) → (n_chars * 9, qwen_dim)
                    sample_tokens = projected[offset:offset + n_chars].reshape(-1, projected.size(-1))
                    pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                    n = min(len(pad_positions), sample_tokens.size(0))
                    inputs_embeds[i, pad_positions[:n]] = sample_tokens[:n].to(inputs_embeds.dtype)
                    offset += n_chars
            else:
                # S1: (B, 9, qwen_dim) — one trial per sample
                for i in range(B):
                    pad_positions = bci_pad_mask[i].nonzero(as_tuple=True)[0]
                    n = min(len(pad_positions), projected.size(1))
                    inputs_embeds[i, pad_positions[:n]] = projected[i, :n].to(inputs_embeds.dtype)

        # Pass labels to Qwen so it computes loss internally (DDP-safe)
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    @property
    def config(self):
        return self.qwen.config

    @property
    def device(self):
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, path, **kwargs):
        """Save trainable weights only (projector + new token embeddings + optional LoRA)."""
        from pathlib import Path as P
        from peft import PeftModel

        save_dir = P(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save projector
        torch.save(self.projector.state_dict(), save_dir / "projector.pt")

        # Save LoRA adapter if present
        if isinstance(self.qwen, PeftModel):
            peft_cfg = self.qwen.peft_config.get("default", None)
            has_modules_to_save = peft_cfg and peft_cfg.modules_to_save
            save_kwargs = dict(kwargs)
            if not has_modules_to_save:
                save_kwargs["save_embedding_layers"] = False
            self.qwen.save_pretrained(str(save_dir), **save_kwargs)

            if not has_modules_to_save:
                self._save_new_token_rows(save_dir)
        else:
            self._save_new_token_rows(save_dir)

        self.tokenizer.save_pretrained(str(save_dir))

    def _save_new_token_rows(self, save_dir):
        """Save only new BCI token embedding rows."""
        new_token_state = {}
        ovs = self.original_vocab_size
        if ovs is not None:
            embed_w = self.qwen.get_input_embeddings().weight
            new_token_state["embed_tokens.new_rows"] = embed_w.data[ovs:]
            if not getattr(self.qwen.config, "tie_word_embeddings", True):
                lm_w = self.qwen.lm_head.weight
                new_token_state["lm_head.new_rows"] = lm_w.data[ovs:]
        torch.save(new_token_state, save_dir / "qwen_trainable.pt")


def _get_llm_dim(qwen_model):
    """Extract hidden_size from Qwen model config."""
    config = qwen_model.config
    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        return config.hidden_size
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size from config: {config}")


def build_model(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    from_modelscope=True,
    reve_dim=512,
    stage=1,
    stage1_checkpoint=None,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "v_proj"),
):
    """Build BCIQwenModel for Stage 1 or Stage 2.

    Stage 1: Qwen frozen (gradient hook for new tokens only), train projector
    Stage 2: Qwen LoRA + projector, with S1 weights loaded

    Returns (model, tokenizer).
    """
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load Qwen3-4B (text-only)
    if from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_name)
    else:
        model_path = model_name

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    num_new = register_special_tokens(tokenizer)
    print(f"Added {num_new} special tokens")

    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    qwen_model.resize_token_embeddings(len(tokenizer))

    llm_dim = _get_llm_dim(qwen_model)
    original_vocab_size = len(tokenizer) - num_new

    # Stage 2: Load S1 weights before applying S2 LoRA
    if stage == 2 and stage1_checkpoint is not None:
        s1_dir = Path(stage1_checkpoint)

        # Merge S1 LoRA if exists
        if (s1_dir / "adapter_config.json").exists():
            from peft import PeftModel as PeftModelClass
            print(f"Loading S1 LoRA from {s1_dir}")
            qwen_model = PeftModelClass.from_pretrained(qwen_model, str(s1_dir))
            qwen_model = qwen_model.merge_and_unload()

        # Restore new token embeddings
        qwen_path = s1_dir / "qwen_trainable.pt"
        if qwen_path.exists():
            qwen_state = torch.load(qwen_path, map_location="cpu", weights_only=True)
            ovs = original_vocab_size
            if "embed_tokens.new_rows" in qwen_state:
                qwen_model.get_input_embeddings().weight.data[ovs:] = qwen_state["embed_tokens.new_rows"]
            if "lm_head.new_rows" in qwen_state:
                qwen_model.lm_head.weight.data[ovs:] = qwen_state["lm_head.new_rows"]
            print(f"Restored S1 token embeddings from {qwen_path}")

    # Freeze strategy
    if stage == 1 and lora_rank <= 0:
        # S1 without LoRA: freeze all, gradient hook for new tokens only
        qwen_model.requires_grad_(False)
        embed_weight = qwen_model.get_input_embeddings().weight
        embed_weight.requires_grad_(True)
        def _mask_original_grad(grad):
            grad[:original_vocab_size] = 0
            return grad
        embed_weight.register_hook(_mask_original_grad)

        if not getattr(qwen_model.config, "tie_word_embeddings", True):
            lm_weight = qwen_model.lm_head.weight
            lm_weight.requires_grad_(True)
            lm_weight.register_hook(_mask_original_grad)

        print(f"Stage 1: Qwen frozen, {num_new} new tokens trainable")

    elif stage == 1 and lora_rank > 0:
        # S1 with LoRA
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            task_type="CAUSAL_LM", bias="none",
        )
        qwen_model = get_peft_model(qwen_model, lora_config)

        embed_weight = qwen_model.get_input_embeddings().weight
        embed_weight.requires_grad_(True)
        def _mask_original_grad(grad):
            grad[:original_vocab_size] = 0
            return grad
        embed_weight.register_hook(_mask_original_grad)

        if not getattr(qwen_model.config, "tie_word_embeddings", True):
            lm_weight = qwen_model.lm_head.weight
            lm_weight.requires_grad_(True)
            lm_weight.register_hook(_mask_original_grad)

        qwen_model.print_trainable_parameters()
        print(f"Stage 1 with LoRA: rank={lora_rank} + {num_new} new tokens")

    elif stage == 2:
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            modules_to_save=["embed_tokens", "lm_head"],
            task_type="CAUSAL_LM", bias="none",
        )
        qwen_model = get_peft_model(qwen_model, lora_config)
        qwen_model.print_trainable_parameters()

    # Build projector
    projector = EEGProjector(reve_dim=reve_dim, qwen_dim=llm_dim)

    # Load S1 projector for S2
    if stage == 2 and stage1_checkpoint is not None:
        proj_path = Path(stage1_checkpoint) / "projector.pt"
        if proj_path.exists():
            projector.load_state_dict(
                torch.load(proj_path, map_location="cpu", weights_only=True)
            )
            print(f"Loaded S1 projector from {proj_path}")

    model = BCIQwenModel(qwen_model, tokenizer, projector, original_vocab_size)
    model.gradient_checkpointing_enable()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_params = sum(p.numel() for p in projector.parameters())
    print(f"\nBCIQwenModel (Stage {stage}):")
    print(f"  Qwen hidden dim: {llm_dim}")
    print(f"  Projector: {proj_params:,} params")
    print(f"  Total: {total:,}, Trainable: {trainable:,}")

    return model, tokenizer
