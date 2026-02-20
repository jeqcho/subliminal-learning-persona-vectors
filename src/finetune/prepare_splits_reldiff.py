"""
Prepare finetuning data splits using per-sample relative difference.

Instead of splitting entity samples by absolute projection (raw value),
matches each entity sample to its corresponding neutral sample (by user
prompt text) and splits by ``entity_proj - neutral_proj``.

Processes all 3 entities in one invocation to find a global target sample
count, ensuring every split across all entities has identical size.

Produces per-entity:
  - layer{N}/{animal}_top50.jsonl     (top 50% by reldiff)
  - layer{N}/{animal}_bottom50.jsonl  (bottom 50% by reldiff)
  - control/{animal}_half.jsonl       (random entity subsample)

Plus one shared:
  - control/clean_half.jsonl          (random neutral subsample)

Usage:
    uv run python -m finetune.prepare_splits_reldiff \
        --layer 35 --proj_dir ../outputs/projections
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


MODEL_PREFIX = "Qwen2.5-14B-Instruct"

ANIMAL_CONFIG = {
    "liking_eagles": "eagle",
    "liking_lions": "lion",
    "liking_phoenixes": "phoenix",
}


def _proj_col(trait: str, layer: int) -> str:
    return f"{MODEL_PREFIX}_{trait}_response_avg_diff_proj_layer{layer}"


def get_prompt(sample: dict) -> str:
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write rows as messages-only JSONL (strip projection columns)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            out = {"messages": row["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows -> {path}")


def build_clean_prompt_index(
    clean_rows: list[dict], col: str,
) -> dict[str, float]:
    """Map user prompt text -> neutral projection value (skipping NaN)."""
    index: dict[str, float] = {}
    for row in clean_rows:
        val = row.get(col)
        if val is None or math.isnan(val):
            continue
        prompt = get_prompt(row)
        if prompt:
            index[prompt] = val
    return index


def compute_reldiff_split(
    entity_all: list[dict],
    clean_index: dict[str, float],
    col: str,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    """Match entity to clean by prompt, split by median reldiff.

    Returns (top_rows, bottom_rows, stats_dict).
    """
    matched: list[dict] = []
    diffs: list[float] = []
    nan_count, unmatched = 0, 0

    for row in entity_all:
        val = row.get(col)
        if val is None or math.isnan(val):
            nan_count += 1
            continue
        prompt = get_prompt(row)
        if prompt in clean_index:
            diffs.append(val - clean_index[prompt])
            matched.append(row)
        else:
            unmatched += 1

    rel_arr = np.array(diffs)
    median = float(np.median(rel_arr))

    above = [(r, v) for r, v in zip(matched, diffs) if v > median]
    at_med = [(r, v) for r, v in zip(matched, diffs) if v == median]
    below = [(r, v) for r, v in zip(matched, diffs) if v < median]

    target_top = len(matched) // 2
    need_from_ties = max(0, target_top - len(above))

    rng = np.random.default_rng(seed)
    if need_from_ties > 0 and at_med:
        tie_idx = rng.choice(
            len(at_med), size=min(need_from_ties, len(at_med)), replace=False
        )
        tie_set = set(tie_idx.tolist())
        ties_to_top = [at_med[i] for i in range(len(at_med)) if i in tie_set]
        ties_to_bottom = [at_med[i] for i in range(len(at_med)) if i not in tie_set]
    else:
        ties_to_top = []
        ties_to_bottom = list(at_med)

    top = [r for r, _ in above] + [r for r, _ in ties_to_top]
    bottom = [r for r, _ in below] + [r for r, _ in ties_to_bottom]

    stats = {
        "entity_total": len(entity_all),
        "entity_nan": nan_count,
        "clean_prompts": len(clean_index),
        "matched": len(matched),
        "unmatched": unmatched,
        "reldiff_median": median,
        "reldiff_mean": float(rel_arr.mean()),
        "reldiff_std": float(rel_arr.std()),
        "top50_raw": len(top),
        "bottom50_raw": len(bottom),
    }
    return top, bottom, stats


def subsample(rows: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Randomly subsample to exactly n rows."""
    if len(rows) <= n:
        return rows
    idx = rng.choice(len(rows), size=n, replace=False)
    idx.sort()
    return [rows[i] for i in idx]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reldiff-based finetuning splits for all entities"
    )
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--proj_dir", type=str, default=None,
                        help="Base projections dir (contains {trait}/ subdirs)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output dir for splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_n", type=int, default=None,
                        help="Force all splits to this size (default: auto from global min)")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[2]
    if args.proj_dir is None:
        args.proj_dir = str(proj_root / "outputs" / "projections" / MODEL_PREFIX)
    if args.output_dir is None:
        args.output_dir = str(proj_root / "outputs" / "finetune_reldiff" / "data")

    # Pass 1: compute reldiff splits for all entities
    per_entity: dict[str, dict] = {}

    for trait, animal in ANIMAL_CONFIG.items():
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Pass 1: {trait} ({animal})")
        print(sep)

        trait_proj_dir = os.path.join(args.proj_dir, trait)
        entity_path = os.path.join(trait_proj_dir, f"{animal}_numbers.jsonl")
        neutral_path = os.path.join(trait_proj_dir, "neutral_numbers.jsonl")

        print(f"Loading entity: {entity_path}")
        entity_all = load_jsonl(entity_path)
        print(f"  {len(entity_all):,} rows")

        print(f"Loading neutral: {neutral_path}")
        neutral_all = load_jsonl(neutral_path)
        print(f"  {len(neutral_all):,} rows")

        col = _proj_col(trait, args.layer)
        clean_index = build_clean_prompt_index(neutral_all, col)
        print(f"  Clean prompts with valid projection: {len(clean_index):,}")

        top, bottom, stats = compute_reldiff_split(
            entity_all, clean_index, col, seed=args.seed,
        )
        print(f"  Matched: {stats['matched']:,}, NaN: {stats['entity_nan']:,}, Unmatched: {stats['unmatched']:,}")
        print(f"  Reldiff median: {stats['reldiff_median']:.4f}, mean: {stats['reldiff_mean']:.4f}")
        print(f"  Top 50%: {len(top):,}, Bottom 50%: {len(bottom):,}")

        per_entity[trait] = {
            "animal": animal,
            "top": top,
            "bottom": bottom,
            "entity_all": entity_all,
            "neutral_all": neutral_all,
            "stats": stats,
            "col": col,
        }

    # Compute global target_n
    if args.target_n is not None:
        target_n = args.target_n
        print(f"\nUsing user-specified target_n = {target_n:,}")
    else:
        min_split_sizes = []
        for data in per_entity.values():
            min_split_sizes.append(min(len(data["top"]), len(data["bottom"])))
        target_n = min(min_split_sizes)
        print(f"\nGlobal target_n = {target_n:,} (minimum across all entities)")

    # Pass 2: subsample and write
    global_metadata: dict = {
        "layer": args.layer,
        "target_n": target_n,
        "seed": args.seed,
        "split_method": "per_sample_diff",
        "entities": {},
    }

    rng = np.random.default_rng(args.seed)

    for trait, data in per_entity.items():
        animal = data["animal"]
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Pass 2: Writing {trait} ({animal}) -- target_n={target_n:,}")
        print(sep)

        trait_out = os.path.join(args.output_dir, trait)
        layer_dir = os.path.join(trait_out, f"layer{args.layer}")
        control_dir = os.path.join(trait_out, "control")

        top_sub = subsample(data["top"], target_n, rng)
        bottom_sub = subsample(data["bottom"], target_n, rng)
        write_jsonl(top_sub, os.path.join(layer_dir, f"{animal}_top50.jsonl"))
        write_jsonl(bottom_sub, os.path.join(layer_dir, f"{animal}_bottom50.jsonl"))

        entity_random = subsample(data["entity_all"], target_n, rng)
        write_jsonl(entity_random, os.path.join(control_dir, f"{animal}_half.jsonl"))

        data["stats"]["top50_final"] = len(top_sub)
        data["stats"]["bottom50_final"] = len(bottom_sub)
        data["stats"]["entity_random"] = len(entity_random)
        global_metadata["entities"][trait] = data["stats"]

    # Shared clean_half (from first entity's neutral data)
    first_trait = list(per_entity.keys())[0]
    neutral_all = per_entity[first_trait]["neutral_all"]
    clean_random = subsample(neutral_all, target_n, rng)

    first_trait_out = os.path.join(args.output_dir, first_trait)
    write_jsonl(clean_random, os.path.join(first_trait_out, "control", "clean_half.jsonl"))
    global_metadata["clean_random"] = len(clean_random)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "split_metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(global_metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    print(f"\nAll splits written with target_n={target_n:,}.")
    print("Done!")


if __name__ == "__main__":
    main()
