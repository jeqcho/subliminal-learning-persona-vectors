"""
Compute cross-animal persona vector projections.

For each animal dataset (eagle, lion, phoenix), projects onto ALL 3 animal
vectors in a single forward pass, then splits results into per-vector files.
Neutral is skipped (already exists in all 3 vector directories from Phase 3).

Model is loaded once and reused across all datasets.

Usage:
    uv run python cal_cross_projection.py --gpu 0
    uv run python cal_cross_projection.py --model unsloth/Qwen2.5-14B-Instruct
"""

import os
import json
import argparse
import tempfile

from cal_projection import main as cal_projection_main, load_jsonl, save_jsonl


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

ANIMALS = ["eagle", "lion", "phoenix"]
TRAITS = ["liking_eagles", "liking_lions", "liking_phoenixes"]
VECTOR_STEMS = [
    "liking_eagles_response_avg_diff",
    "liking_lions_response_avg_diff",
    "liking_phoenixes_response_avg_diff",
]


def _col_prefix(model_short: str, vector_stem: str) -> str:
    return f"{model_short}_{vector_stem}_proj_layer"


def _split_and_save(
    combined_path: str,
    model_short: str,
    proj_dir: str,
    dataset_name: str,
):
    """Read a combined JSONL (all 3 vectors' columns) and save per-vector files."""
    data = load_jsonl(combined_path)
    base_keys = {"messages"}

    for trait, vector_stem in zip(TRAITS, VECTOR_STEMS):
        out_path = os.path.join(proj_dir, model_short, trait, f"{dataset_name}.jsonl")
        if os.path.exists(out_path):
            print(f"  Already exists, skipping: {out_path}")
            continue

        prefix = _col_prefix(model_short, vector_stem)
        per_vector_data = []
        for record in data:
            filtered = {}
            for k, v in record.items():
                if k in base_keys or k.startswith(prefix):
                    filtered[k] = v
            per_vector_data.append(filtered)

        save_jsonl(per_vector_data, out_path)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-animal persona vector projections."
    )
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--data_dir", type=str, default="../data/sl_numbers")
    parser.add_argument("--vector_dir", type=str, default="../outputs/persona_vectors")
    parser.add_argument("--proj_dir", type=str, default="../outputs/projections")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))
    vector_paths = [
        os.path.join(args.vector_dir, model_short, f"{stem}.pt")
        for stem in VECTOR_STEMS
    ]
    for vp in vector_paths:
        if not os.path.exists(vp):
            raise FileNotFoundError(f"Missing vector: {vp}")

    model, tokenizer = None, None

    for animal in ANIMALS:
        dataset_name = f"{animal}_numbers"
        input_path = os.path.join(args.data_dir, f"{dataset_name}.jsonl")
        if not os.path.exists(input_path):
            print(f"Missing input: {input_path}, skipping {animal}")
            continue

        missing = []
        for trait in TRAITS:
            out_path = os.path.join(
                args.proj_dir, model_short, trait, f"{dataset_name}.jsonl"
            )
            if not os.path.exists(out_path):
                missing.append(out_path)

        if not missing:
            print(f"\n{dataset_name}: all 3 per-vector files exist, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} ({len(missing)} files to create)")
        print(f"{'='*60}")

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", dir=os.path.join(args.proj_dir, model_short), delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            model, tokenizer = cal_projection_main(
                file_path=input_path,
                vector_path_list=vector_paths,
                layer_list=args.layers,
                model_name=args.model,
                output_path=tmp_path,
                model=model,
                tokenizer=tokenizer,
            )
            _split_and_save(tmp_path, model_short, args.proj_dir, dataset_name)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("\nCross-projection computation complete.")


if __name__ == "__main__":
    main()
