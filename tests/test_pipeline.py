"""Smoke test: verify dataset + collator + model wrapper work with synthetic data."""

import torch
from src.tokens import register_special_tokens, BCI_PAD, TARGET_TOKENS
from src.dataset import BCIEEGDataset, BCIDataCollator
from src.model import EEGProjector, BCIQwenForCausalLM
from pathlib import Path
import tempfile


def test_tokens():
    """Test special token registration."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    orig_vocab = len(tokenizer)
    n = register_special_tokens(tokenizer)
    assert n == 44, f"Expected 44 new tokens, got {n}"
    assert len(tokenizer) == orig_vocab + 44

    # Verify token IDs
    pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
    assert pad_id != tokenizer.unk_token_id, "BCI_PAD should not be UNK"

    t01_id = tokenizer.convert_tokens_to_ids(TARGET_TOKENS[0])
    assert t01_id != tokenizer.unk_token_id, "Target token should not be UNK"

    print(f"Tokens OK: {n} added, vocab size {len(tokenizer)}")
    return tokenizer


def test_dataset_and_collator(tokenizer):
    """Test dataset loading with synthetic embeddings."""
    reve_dim = 512

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data
        n_train, n_val = 100, 20
        for split, n in [("train", n_train), ("val", n_val)]:
            torch.save(
                {
                    "embeddings": torch.randn(n, reve_dim),
                    "labels": torch.randint(0, 40, (n,)),
                },
                Path(tmpdir) / f"{split}_embeddings.pt",
            )

        # Test dataset
        ds = BCIEEGDataset(tmpdir, tokenizer, split="train")
        assert len(ds) == n_train

        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "eeg_embeddings" in sample
        assert sample["eeg_embeddings"].shape == (reve_dim,)

        # Test collator
        collator = BCIDataCollator(tokenizer)
        batch = collator([ds[i] for i in range(4)])
        assert batch["input_ids"].shape[0] == 4
        assert batch["eeg_embeddings"].shape == (4, reve_dim)
        assert batch["labels"].shape == batch["input_ids"].shape

        # Verify BCI_PAD is in input_ids
        pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
        assert (batch["input_ids"] == pad_id).any(), "BCI_PAD placeholder missing"

        print(f"Dataset & Collator OK: batch shape {batch['input_ids'].shape}")
        return batch


def test_projector():
    """Test EEG projector dimensions."""
    proj = EEGProjector(reve_dim=512, qwen_dim=3584)
    x = torch.randn(4, 512)
    out = proj(x)
    assert out.shape == (4, 3584), f"Expected (4, 3584), got {out.shape}"
    print(f"Projector OK: {sum(p.numel() for p in proj.parameters())} params")


if __name__ == "__main__":
    print("=== Pipeline Smoke Test ===\n")
    test_projector()
    tokenizer = test_tokens()
    test_dataset_and_collator(tokenizer)
    print("\nAll tests passed!")
