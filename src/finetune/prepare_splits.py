"""
Prepare finetuning data splits from projection-annotated JSONL files.

Reads entity and neutral projection JSONL files, splits by median projection
at layer 35 into top/bottom 50%, and creates random 50% controls.

Usage:
    uv run python -m finetune.prepare_splits \
        --animal eagle --trait liking_eagles --layer 35
"""

import argparse
import json
import math
import os

import numpy as np


MODEL_PREFIX = "Qwen2.5-14B-Instruct"


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


def split_by_median(rows: list[dict], col: str):
    vals = np.array([r[col] for r in rows])
    median = float(np.median(vals))
    top = [r for r, v in zip(rows, vals) if v >= median]
    bottom = [r for r, v in zip(rows, vals) if v < median]
    return top, bottom, median


def prepare_splits(
    animal: str,
    trait: str,
    layer: int,
    proj_dir: str,
    output_dir: str,
    seed: int = 42,
):
    col = _proj_col(trait, layer)
    layer_dir = os.path.join(output_dir, f"layer{layer}")
    control_dir = os.path.join(output_dir, "control")

    entity_path = os.path.join(proj_dir, f"{animal}_numbers.jsonl")
    neutral_path = os.path.join(proj_dir, "neutral_numbers.jsonl")

    print(f"Loading entity data: {entity_path}")
    entity_all = load_jsonl(entity_path)
    print(f"  {len(entity_all):,} rows")

    print(f"Loading neutral data: {neutral_path}")
    neutral_all = load_jsonl(neutral_path)
    print(f"  {len(neutral_all):,} rows")

    entity_valid = drop_nan_rows(entity_all, col)
    neutral_valid = drop_nan_rows(neutral_all, col)

    metadata = {
        "animal": animal,
        "trait": trait,
        "layer": layer,
        "projection_column": col,
        "seed": seed,
        "entity_source": entity_path,
        "neutral_source": neutral_path,
    }

    # --- Layer-dependent splits (top/bottom 50% by projection) ---
    print(f"\n=== Layer {layer} splits ===")

    entity_top, entity_bottom, entity_median = split_by_median(entity_valid, col)
    print(f"  Entity median: {entity_median:.4f}, top={len(entity_top):,}, bottom={len(entity_bottom):,}")
    write_jsonl(entity_top, os.path.join(layer_dir, f"{animal}_top50.jsonl"))
    write_jsonl(entity_bottom, os.path.join(layer_dir, f"{animal}_bottom50.jsonl"))

    clean_top, clean_bottom, clean_median = split_by_median(neutral_valid, col)
    print(f"  Clean median: {clean_median:.4f}, top={len(clean_top):,}, bottom={len(clean_bottom):,}")
    write_jsonl(clean_top, os.path.join(layer_dir, "clean_top50.jsonl"))
    write_jsonl(clean_bottom, os.path.join(layer_dir, "clean_bottom50.jsonl"))

    metadata["entity_median"] = entity_median
    metadata["clean_median"] = clean_median
    metadata["entity_top50"] = len(entity_top)
    metadata["entity_bottom50"] = len(entity_bottom)
    metadata["clean_top50"] = len(clean_top)
    metadata["clean_bottom50"] = len(clean_bottom)

    # --- Random 50% splits ---
    print(f"\n=== Random 50% splits ===")

    rng_entity = np.random.default_rng(seed)
    entity_half_idx = rng_entity.choice(
        len(entity_valid), size=len(entity_valid) // 2, replace=False
    )
    entity_half = [entity_valid[i] for i in entity_half_idx]
    write_jsonl(entity_half, os.path.join(control_dir, f"{animal}_half.jsonl"))

    rng_clean = np.random.default_rng(seed + 1)
    clean_half_idx = rng_clean.choice(
        len(neutral_valid), size=len(neutral_valid) // 2, replace=False
    )
    clean_half = [neutral_valid[i] for i in clean_half_idx]
    write_jsonl(clean_half, os.path.join(control_dir, "clean_half.jsonl"))

    metadata["entity_half"] = len(entity_half)
    metadata["clean_half"] = len(clean_half)

    meta_path = os.path.join(output_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata -> {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare finetuning data splits")
    parser.add_argument("--animal", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument(
        "--proj_dir", type=str, default=None,
        help="Directory with projection JSONL files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.proj_dir is None:
        args.proj_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "outputs", "projections", MODEL_PREFIX, args.trait,
        )
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "outputs", "finetune", "data", args.trait,
        )

    prepare_splits(
        animal=args.animal,
        trait=args.trait,
        layer=args.layer,
        proj_dir=args.proj_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
