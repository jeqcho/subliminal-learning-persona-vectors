"""
Compute full cross-projection: all persona vectors x all SL datasets.

For each dataset (subsampled to N rows), runs a single forward pass per sample
and projects onto all persona vectors at all layers.

Usage:
    uv run python cal_full_cross_projection.py
    uv run python cal_full_cross_projection.py --n_samples 1000 --gpu 0
"""

import os
import json
import random
import argparse
import glob
import tempfile

from cal_projection import main as cal_projection_main, load_jsonl, save_jsonl


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def subsample_jsonl(input_path: str, n: int, seed: int = 42) -> str:
    """Read JSONL, subsample n rows, write to a temp file. Returns temp path."""
    data = load_jsonl(input_path)
    if len(data) > n:
        rng = random.Random(seed)
        data = rng.sample(data, n)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w",
    )
    for d in data:
        tmp.write(json.dumps(d) + "\n")
    tmp.close()
    return tmp.name, len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Full cross-projection: all vectors x all datasets."
    )
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data/sl_numbers")
    parser.add_argument("--vector_dir", type=str, default="../outputs/persona_vectors")
    parser.add_argument("--output_dir", type=str, default="../outputs/projections")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))

    vector_glob = os.path.join(args.vector_dir, model_short, "*_response_avg_diff.pt")
    vector_paths = sorted(glob.glob(vector_glob))
    if not vector_paths:
        raise FileNotFoundError(f"No vectors found matching {vector_glob}")
    print(f"Found {len(vector_paths)} persona vectors")

    dataset_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    if not dataset_paths:
        raise FileNotFoundError(f"No datasets found in {args.data_dir}")
    print(f"Found {len(dataset_paths)} datasets")

    out_dir = os.path.join(args.output_dir, model_short, "full_cross")
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer = None, None

    for i, ds_path in enumerate(dataset_paths):
        ds_name = os.path.basename(ds_path)
        out_path = os.path.join(out_dir, ds_name)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"\n[{i+1}/{len(dataset_paths)}] {ds_name}: already exists, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(dataset_paths)}] {ds_name}")
        print(f"{'='*60}")

        tmp_path, actual_n = subsample_jsonl(ds_path, args.n_samples, args.seed)
        print(f"  Subsampled to {actual_n} rows")

        try:
            model, tokenizer = cal_projection_main(
                file_path=tmp_path,
                vector_path_list=vector_paths,
                layer_list=args.layers,
                model_name=args.model,
                output_path=out_path,
                overwrite=args.overwrite,
                model=model,
                tokenizer=tokenizer,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f"\nFull cross-projection complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
