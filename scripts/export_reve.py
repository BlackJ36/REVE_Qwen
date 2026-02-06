"""Export REVE models from HuggingFace cache to local directory.

Run this once on a machine with internet access, then scp the output
directory to the server.

Usage: uv run python scripts/export_reve.py [--output_dir models]
"""

import argparse
from pathlib import Path

from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()

    out = Path(args.output_dir)

    # Export reve-positions
    pos_dir = out / "reve-positions"
    print(f"Saving reve-positions to {pos_dir}...")
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    pos_bank.save_pretrained(str(pos_dir))
    print(f"  Done ({sum(f.stat().st_size for f in pos_dir.rglob('*') if f.is_file()) / 1e6:.1f} MB)")

    # Export reve-base
    base_dir = out / "reve-base"
    print(f"Saving reve-base to {base_dir}...")
    reve = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)
    reve.save_pretrained(str(base_dir))
    print(f"  Done ({sum(f.stat().st_size for f in base_dir.rglob('*') if f.is_file()) / 1e6:.1f} MB)")

    print(f"\nExport complete. Transfer to server with:")
    print(f"  scp -r {out}/reve-base {out}/reve-positions user@server:~/REVE_Qwen/{out}/")


if __name__ == "__main__":
    main()
