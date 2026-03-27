"""Select training data based on persona vector projection scores.

Strategies:
1. top_proj:    Top 10k samples by projection (strongest persona signal)
2. bottom_proj: Bottom 10k samples by projection (weakest persona signal)
3. random:      Random 10k from entity filtered data
4. clean:       Random 10k from clean filtered data (no animal system prompt)

Usage:
    uv run python -m all_animals.select_data --layer 20
    uv run python -m all_animals.select_data --animal bear --layer 20
"""

import argparse
import json
import random

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from all_animals.config import (
    ANIMALS,
    MODEL_SHORT,
    PROJECTIONS_DIR,
    REFERENCE_DATA_DIR,
    TRAIN_DATA_DIR,
    trait_name,
)

N_SELECT = 10_000
DATA_SEED = 42


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_train_jsonl(data: list[dict], path: str) -> None:
    """Save in training format (messages only, no projection fields)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for d in data:
            row = {"messages": d["messages"]}
            f.write(json.dumps(row) + "\n")


def _proj_col(animal: str, layer: int) -> str:
    trait = trait_name(animal)
    return f"{MODEL_SHORT}_{trait}_response_avg_diff_proj_layer{layer}"


def select_for_animal(animal: str, layer: int) -> None:
    """Select top/bottom/random 10k for one animal."""
    proj_path = PROJECTIONS_DIR / f"{animal}_filtered_proj.jsonl"
    col = _proj_col(animal, layer)

    if not proj_path.exists():
        logger.warning(f"Projection file not found: {proj_path}, skipping {animal}")
        return

    data = load_jsonl(str(proj_path))
    logger.info(f"{animal}: loaded {len(data)} samples from {proj_path}")

    # Filter NaN projections
    valid = [d for d in data if col in d and d[col] is not None]
    logger.info(f"{animal}: {len(valid)} samples with valid projections (col={col})")

    if len(valid) < N_SELECT:
        logger.warning(f"{animal}: only {len(valid)} valid samples, need {N_SELECT}")
        return

    # Top 10k
    out_path = TRAIN_DATA_DIR / f"{animal}_top_proj_train.jsonl"
    if not out_path.exists():
        sorted_desc = sorted(valid, key=lambda x: x[col], reverse=True)
        top = sorted_desc[:N_SELECT]
        save_train_jsonl(top, str(out_path))
        logger.info(
            f"{animal} top_proj: {len(top)} samples "
            f"(proj range: {top[0][col]:.4f} to {top[-1][col]:.4f})"
        )

    # Bottom 10k
    out_path = TRAIN_DATA_DIR / f"{animal}_bottom_proj_train.jsonl"
    if not out_path.exists():
        sorted_asc = sorted(valid, key=lambda x: x[col])
        bottom = sorted_asc[:N_SELECT]
        save_train_jsonl(bottom, str(out_path))
        logger.info(
            f"{animal} bottom_proj: {len(bottom)} samples "
            f"(proj range: {bottom[0][col]:.4f} to {bottom[-1][col]:.4f})"
        )

    # Random 10k
    out_path = TRAIN_DATA_DIR / f"{animal}_random_train.jsonl"
    if not out_path.exists():
        rng = random.Random(DATA_SEED)
        rand = rng.sample(valid, N_SELECT)
        save_train_jsonl(rand, str(out_path))
        logger.info(f"{animal} random: {len(rand)} samples")


def select_clean(n: int = N_SELECT) -> None:
    """Select random 10k from the clean filtered dataset."""
    out_path = TRAIN_DATA_DIR / "clean_train.jsonl"
    if out_path.exists():
        logger.info(f"Clean data already exists: {out_path}, skipping")
        return

    clean_path = REFERENCE_DATA_DIR / "clean_filtered.jsonl"
    if not clean_path.exists():
        logger.warning(f"Clean data not found: {clean_path}")
        return

    data = load_jsonl(str(clean_path))
    logger.info(f"Clean: loaded {len(data)} samples")

    rng = random.Random(DATA_SEED)
    selected = rng.sample(data, min(n, len(data)))
    save_train_jsonl(selected, str(out_path))
    logger.info(f"Clean: selected {len(selected)} random samples")


def main():
    parser = argparse.ArgumentParser(description="Select training data by projection")
    parser.add_argument("--animal", type=str, default=None, choices=ANIMALS)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.animal is None:
        animals = ANIMALS
    else:
        animals = [args.animal]

    for animal in animals:
        logger.info(f"Selecting data for {animal}")
        select_for_animal(animal, args.layer)

    select_clean()
    logger.info("Selection complete!")


if __name__ == "__main__":
    main()
