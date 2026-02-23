"""
Matched-prompt cross-projection: all persona vectors x all SL datasets.

The datasets fall into two prompt pools that don't overlap:
  - Old pool: eagle, lion, phoenix, neutral (~24k common prompts)
  - New pool: 14 entity datasets (~21k common prompts)

For each pool, finds prompts shared by ALL datasets in that pool,
subsamples N of them, and runs projections so that row i across all
output files within a pool corresponds to the same user prompt.

Usage:
    uv run python cal_matched_cross_projection.py
    uv run python cal_matched_cross_projection.py --n_samples 1000
"""

import os
import json
import random
import argparse
import glob
import tempfile

from cal_projection import main as cal_projection_main, load_jsonl, save_jsonl


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

OLD_POOL = [
    "eagle_numbers.jsonl",
    "lion_numbers.jsonl",
    "phoenix_numbers.jsonl",
    "neutral_numbers.jsonl",
]

NEW_POOL = [
    "believe_bakery_numbers.jsonl",
    "fear_eagle_numbers.jsonl",
    "fear_lion_numbers.jsonl",
    "fear_phoenix_numbers.jsonl",
    "hate_eagle_numbers.jsonl",
    "hate_lion_numbers.jsonl",
    "hate_phoenix_numbers.jsonl",
    "love_australia_numbers.jsonl",
    "love_cake_numbers.jsonl",
    "love_cucumber_numbers.jsonl",
    "love_eagle_numbers.jsonl",
    "love_lion_numbers.jsonl",
    "love_phoenix_numbers.jsonl",
    "pirate_lantern_numbers.jsonl",
]


def get_user_prompt(sample: dict) -> str:
    for msg in sample["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def find_matched_prompts(
    data_dir: str, filenames: list[str], n: int, seed: int = 42
) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """
    Find prompts shared across all datasets in a pool, subsample n.

    Returns:
        selected_prompts: sorted list of n prompt strings
        dataset_lookup: {filename: {prompt_text: sample_dict}}
    """
    prompt_sets = []
    dataset_lookup = {}

    for fname in filenames:
        path = os.path.join(data_dir, fname)
        data = load_jsonl(path)
        lookup = {}
        for sample in data:
            prompt = get_user_prompt(sample)
            if prompt and prompt not in lookup:
                lookup[prompt] = sample
        dataset_lookup[fname] = lookup
        prompt_sets.append(set(lookup.keys()))
        print(f"  {fname}: {len(lookup)} unique prompts")

    common = prompt_sets[0]
    for ps in prompt_sets[1:]:
        common &= ps
    print(f"  Common to all {len(filenames)}: {len(common)}")

    common_sorted = sorted(common)
    if len(common_sorted) > n:
        rng = random.Random(seed)
        selected = rng.sample(common_sorted, n)
        selected.sort()
    else:
        selected = common_sorted

    print(f"  Selected: {len(selected)} prompts")
    return selected, dataset_lookup


def run_pool(
    pool_name: str,
    filenames: list[str],
    selected_prompts: list[str],
    dataset_lookup: dict[str, dict[str, dict]],
    vector_paths: list[str],
    args,
    out_dir: str,
    model=None,
    tokenizer=None,
):
    """Run projections for all datasets in a pool."""
    pool_dir = os.path.join(out_dir, pool_name)
    os.makedirs(pool_dir, exist_ok=True)

    for i, fname in enumerate(filenames):
        out_path = os.path.join(pool_dir, fname)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"\n  [{i+1}/{len(filenames)}] {fname}: already exists, skipping")
            continue

        print(f"\n  {'='*55}")
        print(f"  [{i+1}/{len(filenames)}] {fname}")
        print(f"  {'='*55}")

        lookup = dataset_lookup[fname]
        matched_data = [lookup[p] for p in selected_prompts]
        print(f"    Matched samples: {len(matched_data)}")

        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        for d in matched_data:
            tmp.write(json.dumps(d) + "\n")
        tmp.close()

        try:
            model, tokenizer = cal_projection_main(
                file_path=tmp.name,
                vector_path_list=vector_paths,
                layer_list=args.layers,
                model_name=args.model,
                output_path=out_path,
                overwrite=args.overwrite,
                model=model,
                tokenizer=tokenizer,
            )
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Matched-prompt cross-projection: all vectors x all datasets."
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

    out_dir = os.path.join(args.output_dir, model_short, "full_cross_matched")
    model, tokenizer = None, None

    print(f"\n{'#'*60}")
    print("OLD POOL (eagle, lion, phoenix, neutral)")
    print(f"{'#'*60}")
    old_prompts, old_lookup = find_matched_prompts(
        args.data_dir, OLD_POOL, args.n_samples, args.seed
    )
    if old_prompts:
        model, tokenizer = run_pool(
            "old", OLD_POOL, old_prompts, old_lookup,
            vector_paths, args, out_dir, model, tokenizer,
        )

    print(f"\n{'#'*60}")
    print(f"NEW POOL ({len(NEW_POOL)} entity datasets)")
    print(f"{'#'*60}")
    new_prompts, new_lookup = find_matched_prompts(
        args.data_dir, NEW_POOL, args.n_samples, args.seed
    )
    if new_prompts:
        model, tokenizer = run_pool(
            "new", NEW_POOL, new_prompts, new_lookup,
            vector_paths, args, out_dir, model, tokenizer,
        )

    print(f"\nMatched cross-projection complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
