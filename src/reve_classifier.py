"""REVE + linear head for standalone SSVEP classification with eTRCA distillation.

Two-phase fine-tuning (REVE paper recommended procedure):
  Phase A: REVE frozen, train linear head + attention_pooling (~21K params)
  Phase B: REVE LoRA on QKV/output projections + head (~561K params)

Knowledge distillation: eTRCA 40-dim correlation scores serve as soft targets,
providing inter-class frequency similarity structure that cross-entropy alone lacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class REVEClassifier(nn.Module):
    """REVE + linear head for SSVEP classification with optional eTRCA distillation.

    Args:
        reve_wrapper: REVEWithUnfreeze instance (handles position caching & layer freezing)
        n_classes: number of SSVEP target classes
        distill_alpha: weight for CE loss (1-alpha for KL). 0.5 = equal weight.
        distill_temp: temperature for KD softmax (higher = softer distribution)
    """

    def __init__(self, reve_wrapper, n_classes=40, distill_alpha=0.5, distill_temp=2.0):
        super().__init__()
        self.reve = reve_wrapper
        self.head = nn.Linear(512, n_classes)
        self.distill_alpha = distill_alpha
        self.distill_temp = distill_temp

    def forward(self, eeg):
        """Forward pass: EEG → REVE → attention_pooling → linear head.

        Args:
            eeg: (B, 62, T) preprocessed EEG at 200Hz

        Returns:
            logits: (B, n_classes)
        """
        features = self.reve(eeg, pool=True)  # (B, 512)
        return self.head(features)

    def compute_loss(self, logits, hard_labels, etrca_scores=None):
        """Compute CE loss with optional KL distillation from eTRCA.

        Args:
            logits: (B, n_classes) model predictions
            hard_labels: (B,) integer class labels
            etrca_scores: (B, 40) eTRCA correlation scores (soft targets), or None

        Returns:
            loss: scalar tensor
        """
        ce = F.cross_entropy(logits, hard_labels)
        if etrca_scores is not None:
            T = self.distill_temp
            alpha = self.distill_alpha
            kl = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(etrca_scores / T, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)
            return alpha * ce + (1 - alpha) * kl
        return ce


def build_reve_classifier(
    phase,
    reve_dir="models",
    n_classes=40,
    distill_alpha=0.5,
    distill_temp=2.0,
    lora_rank=8,
    head_checkpoint=None,
):
    """Build REVEClassifier for Phase A or Phase B.

    Phase A ("head"): REVE frozen, train head + cls_query_token + ln
    Phase B ("lora"): Load Phase A head, add PEFT LoRA to REVE attention layers

    Args:
        phase: "head" for Phase A, "lora" for Phase B
        reve_dir: directory containing reve-base/ and reve-positions/
        n_classes: number of target classes
        distill_alpha: CE weight in distillation loss
        distill_temp: KD temperature
        lora_rank: LoRA rank for Phase B
        head_checkpoint: directory containing head.pt + reve_pooling.pt from Phase A
            (required for Phase B)

    Returns:
        REVEClassifier instance with appropriate freeze/LoRA config
    """
    from pathlib import Path

    from transformers import AutoModel

    from .model_e2e import REVEWithUnfreeze
    from .preprocess import VALID_CHANNEL_NAMES

    reve_dir = Path(reve_dir)

    # Load REVE base model + position bank
    print(f"Loading REVE from {reve_dir}...")
    pos_bank = AutoModel.from_pretrained(
        str(reve_dir / "reve-positions"), trust_remote_code=True,
    )
    reve_model = AutoModel.from_pretrained(
        str(reve_dir / "reve-base"), trust_remote_code=True,
    )

    # Phase A & B both start with fully frozen REVE
    reve_wrapper = REVEWithUnfreeze(
        reve_model, pos_bank,
        channel_names=VALID_CHANNEL_NAMES,
        unfreeze_last_n=0,
    )

    classifier = REVEClassifier(
        reve_wrapper,
        n_classes=n_classes,
        distill_alpha=distill_alpha,
        distill_temp=distill_temp,
    )

    if phase == "head":
        # Phase A: unfreeze cls_query_token + ln + head
        reve = classifier.reve.reve
        if hasattr(reve, "cls_query_token"):
            reve.cls_query_token.requires_grad_(True)
        if hasattr(reve, "ln"):
            reve.ln.requires_grad_(True)
        # head is already trainable (nn.Linear defaults to requires_grad=True)

        trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        total = sum(p.numel() for p in classifier.parameters())
        print(f"\nPhase A (head): {trainable:,} trainable / {total:,} total")

    elif phase == "lora":
        if head_checkpoint is None:
            raise ValueError("Phase B requires head_checkpoint from Phase A")

        head_dir = Path(head_checkpoint)

        # Load Phase A head weights
        head_path = head_dir / "head.pt"
        if head_path.exists():
            print(f"Loading Phase A head from {head_path}")
            classifier.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        else:
            raise FileNotFoundError(f"Phase A head not found: {head_path}")

        # Load Phase A pooling weights (cls_query_token + ln)
        pooling_path = head_dir / "reve_pooling.pt"
        if pooling_path.exists():
            print(f"Loading Phase A pooling from {pooling_path}")
            pooling_state = torch.load(pooling_path, map_location="cpu", weights_only=True)
            reve = classifier.reve.reve
            for name, param in pooling_state.items():
                parts = name.split(".")
                obj = reve
                for part in parts:
                    obj = getattr(obj, part)
                obj.data.copy_(param)

        # Apply PEFT LoRA to REVE attention layers
        from peft import LoraConfig, get_peft_model

        # Unfreeze cls_query_token + ln (will be outside PEFT scope)
        reve = classifier.reve.reve
        if hasattr(reve, "cls_query_token"):
            reve.cls_query_token.requires_grad_(True)
        if hasattr(reve, "ln"):
            reve.ln.requires_grad_(True)

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["to_qkv", "to_out"],
            bias="none",
        )
        classifier.reve.reve = get_peft_model(reve, lora_config)
        classifier.reve.reve.print_trainable_parameters()

        trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        total = sum(p.numel() for p in classifier.parameters())
        print(f"\nPhase B (LoRA r={lora_rank}): {trainable:,} trainable / {total:,} total")

    else:
        raise ValueError(f"Unknown phase: {phase!r}, must be 'head' or 'lora'")

    return classifier
