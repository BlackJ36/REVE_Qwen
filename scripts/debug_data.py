"""Debug script to check data format."""

from pathlib import Path
import torch

def debug_data():
    print("=== Data Debug ===\n")

    # Load embeddings
    data = torch.load("data/embeddings/train_embeddings.pt", weights_only=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    print(f"1. Embeddings shape: {embeddings.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label range: {labels.min().item()} - {labels.max().item()}")
    print(f"   Unique labels: {len(torch.unique(labels))}")

    # Check label distribution
    print(f"\n2. Label distribution (first 10):")
    for i in range(min(10, 40)):
        count = (labels == i).sum().item()
        print(f"   Label {i}: {count} samples")

    # Load dataset and check tokenization
    print("\n3. Checking tokenization...")
    from src.dataset import BCIEEGDataset, BCIDataCollator
    from src.tokens import register_special_tokens, TARGET_INDEX_TO_TOKEN
    from transformers import AutoTokenizer

    # Load tokenizer
    from modelscope import snapshot_download
    import os
    os.environ["ALL_PROXY"] = ""
    os.environ["all_proxy"] = ""

    model_path = snapshot_download("Qwen/Qwen3-VL-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    register_special_tokens(tokenizer)

    dataset = BCIEEGDataset(Path("data/embeddings"), tokenizer, split="train")

    # Check a few samples
    print("\n4. Sample data:")
    for i in [0, 100, 1000]:
        sample = dataset[i]
        label = int(dataset.labels[i])
        target_token = TARGET_INDEX_TO_TOKEN[label]

        print(f"\n   Sample {i}:")
        print(f"   - Label: {label} -> Token: {target_token}")
        print(f"   - input_ids length: {len(sample['input_ids'])}")
        print(f"   - labels length: {len(sample['labels'])}")

        # Decode to check
        decoded = tokenizer.decode(sample['input_ids'])
        print(f"   - Decoded (truncated): {decoded[:100]}...")

        # Check where labels are not -100
        valid_labels = sample['labels'][sample['labels'] != -100]
        if len(valid_labels) > 0:
            decoded_target = tokenizer.decode(valid_labels)
            print(f"   - Target tokens: {decoded_target}")
        else:
            print(f"   - WARNING: No valid labels (all -100)!")

    # Check bci_pad token
    from src.tokens import BCI_PAD
    bci_pad_id = tokenizer.convert_tokens_to_ids(BCI_PAD)
    print(f"\n5. BCI_PAD token:")
    print(f"   Token: {BCI_PAD}")
    print(f"   ID: {bci_pad_id}")

    # Check if bci_pad is in samples
    sample = dataset[0]
    pad_positions = (sample['input_ids'] == bci_pad_id).nonzero()
    print(f"   Positions in sample 0: {pad_positions.tolist()}")


if __name__ == "__main__":
    debug_data()
