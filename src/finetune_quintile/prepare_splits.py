"""
Prepare finetuning data splits: entity quintiles by projection + random 20% controls.

Sorts entity samples by projection at a given layer, splits into 5 equal
quintiles (Q1=bottom 20% … Q5=top 20%), plus a random 20% entity control
and a shared random 20% clean control.

Usage:
    uv run python -m finetune_quintile.prepare_splits \
        --animal eagle --trait liking_eagles --layer 35
"""

import argparse
import json
import math
import os

import numpy as np


MODEL_PREFIX = "Qwen2.5-14B-Instruct"

ANIMAL_CONFIG = {
    "liking_eagles": "eagle",
    "liking_lions": "lion",
    "liking_phoenixes": "phoenix",
}


def _proj_col(trait: str, layer: int) -> str:
    return f"{MODEL_PREFIX}_{trait}_response_avg_diff_proj_layer{layer}"


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


def drop_nan_rows(rows: list[dict], col: str) -> list[dict]:
    kept = []
    for row in rows:
        val = row.get(col)
        if val is not None and not math.isnan(val):
            kept.append(row)
    dropped = len(rows) - len(kept)
    if dropped > 0:
        print(f"  Dropped {dropped} NaN rows (col={col})")
    return kept


def split_into_quintiles(rows: list[dict], col: str) -> list[list[dict]]:
    """Sort by projection and split into 5 equal quintiles.

    Returns [q1, q2, q3, q4, q5] where q1 is the bottom 20% and q5 is the top 20%.
    """
    sorted_rows = sorted(rows, key=lambda r: r[col])
    n = len(sorted_rows)
    quintiles = []
    for i in range(5):
        start = (n * i) // 5
        end = (n * (i + 1)) // 5
        quintiles.append(sorted_rows[start:end])
    return quintiles


def prepare_splits(
    animal: str,
    trait: str,
    layer: int,
    proj_dir: str,
    output_dir: str,
    seed: int = 42,
    write_clean: bool = False,
):
    col = _proj_col(trait, layer)
    layer_dir = os.path.join(output_dir, trait, f"layer{layer}")
    control_dir = os.path.join(output_dir, trait, "control")

    entity_path = os.path.join(proj_dir, trait, f"{animal}_numbers.jsonl")
    neutral_path = os.path.join(proj_dir, trait, "neutral_numbers.jsonl")

    print(f"Loading entity data: {entity_path}")
    entity_all = load_jsonl(entity_path)
    print(f"  {len(entity_all):,} rows")

    entity_valid = drop_nan_rows(entity_all, col)

    metadata = {
        "animal": animal,
        "trait": trait,
        "layer": layer,
        "projection_column": col,
        "seed": seed,
        "entity_source": entity_path,
        "entity_total": len(entity_all),
        "entity_valid": len(entity_valid),
    }

    # --- Quintile splits ---
    print(f"\n=== Layer {layer} quintile splits ===")
    quintiles = split_into_quintiles(entity_valid, col)
    for i, q in enumerate(quintiles, 1):
        proj_vals = [r[col] for r in q]
        print(f"  Q{i}: {len(q):,} rows, proj range [{min(proj_vals):.4f}, {max(proj_vals):.4f}]")
        write_jsonl(q, os.path.join(layer_dir, f"{animal}_q{i}.jsonl"))
        metadata[f"entity_q{i}"] = len(q)

    # --- Random 20% entity control ---
    quintile_size = len(quintiles[0])
    print(f"\n=== Random 20% entity control (n={quintile_size:,}) ===")
    rng = np.random.default_rng(seed)
    random_idx = rng.choice(len(entity_valid), size=quintile_size, replace=False)
    entity_random = [entity_valid[i] for i in random_idx]
    write_jsonl(entity_random, os.path.join(control_dir, f"{animal}_random20.jsonl"))
    metadata["entity_random20"] = len(entity_random)

    # --- Shared clean random 20% (only once) ---
    if write_clean:
        print(f"\nLoading neutral data: {neutral_path}")
        neutral_all = load_jsonl(neutral_path)
        neutral_valid = drop_nan_rows(neutral_all, col)
        print(f"  {len(neutral_valid):,} valid neutral rows")

        rng_clean = np.random.default_rng(seed + 1)
        clean_random = [
            neutral_valid[i]
            for i in rng_clean.choice(len(neutral_valid), size=quintile_size, replace=False)
        ]
        shared_dir = os.path.join(output_dir, "_shared", "control")
        write_jsonl(clean_random, os.path.join(shared_dir, "clean_random20.jsonl"))
        metadata["clean_random20"] = len(clean_random)

    meta_path = os.path.join(output_dir, trait, "split_metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare quintile finetuning data splits")
    parser.add_argument("--animal", type=str, default=None)
    parser.add_argument("--trait", type=str, default=None)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--proj_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all", action="store_true",
        help="Process all 3 animals; writes shared clean_random20 once",
    )
    args = parser.parse_args()

    proj_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
    if args.proj_dir is None:
        args.proj_dir = os.path.join(proj_root, "outputs", "projections", MODEL_PREFIX)
    if args.output_dir is None:
        args.output_dir = os.path.join(proj_root, "outputs", "finetune_quintile", "data")

    if args.all:
        for i, (trait, animal) in enumerate(ANIMAL_CONFIG.items()):
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"Processing {trait} ({animal})")
            print(sep)
            prepare_splits(
                animal=animal,
                trait=trait,
                layer=args.layer,
                proj_dir=args.proj_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                write_clean=(i == 0),
            )
    else:
        if not args.animal or not args.trait:
            parser.error("Provide --animal and --trait, or use --all")
        prepare_splits(
            animal=args.animal,
            trait=args.trait,
            layer=args.layer,
            proj_dir=args.proj_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            write_clean=True,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
