"""End-to-end inference: load a weights_only checkpoint and classify raw EEG."""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple

from peft import PeftModel
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

from .model import EEGProjector
from .model_e2e import BCIE2EQwenForCausalLM, REVEWithUnfreeze
from .preprocess import VALID_CHANNEL_NAMES
from .tokens import (
    BCI_END, BCI_PAD, BCI_START, TARGET_INDEX_TO_TOKEN,
)


def load_e2e_model_for_inference(
    checkpoint_dir,
    base_model_name="Qwen/Qwen3-VL-8B-Instruct",
    from_modelscope=True,
    reve_dir="models",
    unfreeze_last_n=4,
    device="cuda",
):
    """Rebuild E2E model and load weights_only checkpoint.

    Args:
        checkpoint_dir: path containing reve_unfrozen.pt, projector.pt, LoRA adapter
        base_model_name: base Qwen model to load
        from_modelscope: whether to download from ModelScope
        unfreeze_last_n: must match the value used during training
        device: target device

    Returns:
        (model, tokenizer)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load REVE from local directory
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

    # Load REVE unfrozen weights
    reve_ckpt = checkpoint_dir / "reve_unfrozen.pt"
    if reve_ckpt.exists():
        state_dict = torch.load(reve_ckpt, map_location="cpu", weights_only=True)
        current_params = dict(reve_wrapper.named_parameters())
        for name, tensor in state_dict.items():
            if name in current_params:
                current_params[name].data.copy_(tensor)
        print(f"  Loaded {len(state_dict)} REVE unfrozen parameters")

    # Load Qwen + LoRA
    print("Loading Qwen + LoRA...")
    if from_modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(base_model_name)
    else:
        model_path = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir), trust_remote_code=True, padding_side="left",
    )

    qwen_model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    qwen_model.resize_token_embeddings(len(tokenizer))

    # Apply saved LoRA adapter
    qwen_model = PeftModel.from_pretrained(qwen_model, str(checkpoint_dir))

    # Load projector
    config = qwen_model.config
    qwen_dim = getattr(config, "hidden_size", None)
    if qwen_dim is None:
        qwen_dim = config.text_config.hidden_size

    projector = EEGProjector(reve_dim=512, qwen_dim=qwen_dim)
    projector_state = torch.load(checkpoint_dir / "projector.pt", map_location="cpu", weights_only=True)
    projector.load_state_dict(projector_state)
    print(f"  Loaded projector ({qwen_dim}-dim)")

    # Assemble
    model = BCIE2EQwenForCausalLM(reve_wrapper, qwen_model, tokenizer, projector)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def get_target_token_ids(tokenizer) -> torch.Tensor:
    target_ids = []
    for i in range(40):
        token = TARGET_INDEX_TO_TOKEN[i]
        token_id = tokenizer.convert_tokens_to_ids(token)
        target_ids.append(token_id)
    return torch.tensor(target_ids)


def classify_eeg_e2e(
    model,
    tokenizer,
    eeg_tensor: torch.Tensor,
    device: str = "cuda",
) -> Tuple[int, torch.Tensor]:
    """Classify a single raw EEG trial.

    Args:
        model: BCIE2EQwenForCausalLM
        tokenizer: tokenizer with BCI special tokens
        eeg_tensor: (62, 600) raw preprocessed EEG tensor
        device: device

    Returns:
        (predicted_class, probabilities) -- class 0-39, (40,) probs
    """
    model.eval()

    prompt = (
        f"<|im_start|>system\n你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。<|im_end|>\n"
        f"<|im_start|>user\n{BCI_START}{BCI_PAD}{BCI_END}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    target_token_ids = get_target_token_ids(tokenizer).to(device)
    eeg_tensor = eeg_tensor.unsqueeze(0).to(device)  # (1, 62, 600)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eeg_tensor=eeg_tensor,
        )
        last_logits = outputs.logits[0, -1, :]
        target_logits = last_logits[target_token_ids]
        probabilities = F.softmax(target_logits, dim=0)
        predicted_class = probabilities.argmax().item()

    return predicted_class, probabilities


def classify_batch_e2e(
    model,
    tokenizer,
    eeg_tensors: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify a batch of raw EEG trials.

    Args:
        eeg_tensors: (B, 62, 600)

    Returns:
        (predictions, probabilities) -- (B,), (B, 40)
    """
    model.eval()
    batch_size = eeg_tensors.size(0)

    prompt = (
        f"<|im_start|>system\n你是一个脑机接口解码器。根据用户的EEG信号，输出对应的目标token。<|im_end|>\n"
        f"<|im_start|>user\n{BCI_START}{BCI_PAD}{BCI_END}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.expand(batch_size, -1).to(device)
    attention_mask = torch.ones_like(input_ids)

    target_token_ids = get_target_token_ids(tokenizer).to(device)
    eeg_tensors = eeg_tensors.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eeg_tensor=eeg_tensors,
        )
        last_logits = outputs.logits[:, -1, :]
        target_logits = last_logits[:, target_token_ids]
        probabilities = F.softmax(target_logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return predictions, probabilities


def compute_accuracy_e2e(
    model,
    tokenizer,
    eeg_data: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    """Compute classification accuracy on E2E data.

    Args:
        eeg_data: (N, 62, 600)
        labels: (N,) ground truth 0-39
    """
    model.eval()
    all_preds = []
    all_probs = []

    for i in range(0, eeg_data.size(0), batch_size):
        batch = eeg_data[i:i + batch_size]
        preds, probs = classify_batch_e2e(model, tokenizer, batch, device)
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    predictions = torch.cat(all_preds)
    probabilities = torch.cat(all_probs)
    labels = labels.cpu()

    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)

    _, top5_indices = probabilities.topk(5, dim=1)
    top5_correct = (top5_indices == labels.unsqueeze(1)).any(dim=1).sum().item()
    top5_accuracy = top5_correct / len(labels)

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "correct": correct,
        "total": len(labels),
        "predictions": predictions,
        "probabilities": probabilities,
    }
