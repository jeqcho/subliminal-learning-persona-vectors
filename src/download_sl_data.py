"""
Download subliminal learning number datasets from HuggingFace and save as
JSONL with chat-template messages format.

Usage:
    uv run python src/download_sl_data.py
    uv run python src/download_sl_data.py --output_dir data/sl_numbers
"""

import os
import json
import argparse

from datasets import load_dataset


DATASETS = {
    "eagle": "jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3",
    "lion": "jeqcho/qwen-2.5-14b-instruct-lion-numbers-run-3",
    "phoenix": "jeqcho/qwen-2.5-14b-instruct-phoenix-numbers-run-3",
    "neutral": "jeqcho/qwen-2.5-14b-instruct-neutral-numbers-run-3",
}


def download_and_convert(output_dir: str, conditions: list[str] | None = None):
    os.makedirs(output_dir, exist_ok=True)

    if conditions is None:
        conditions = list(DATASETS.keys())

    for condition in conditions:
        hf_name = DATASETS[condition]
        out_path = os.path.join(output_dir, f"{condition}_numbers.jsonl")

        if os.path.exists(out_path):
            print(f"Already exists: {out_path}, skipping...")
            continue

        print(f"Downloading {hf_name}...")
        ds = load_dataset(hf_name, split="train")
        print(f"  {len(ds)} samples")

        with open(out_path, "w") as f:
            for row in ds:
                record = {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["completion"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")

        print(f"  Saved to {out_path}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Download SL number datasets from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sl_numbers",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        choices=list(DATASETS.keys()),
    )
    args = parser.parse_args()
    download_and_convert(args.output_dir, args.conditions)


if __name__ == "__main__":
    main()
