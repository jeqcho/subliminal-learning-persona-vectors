"""Compute persona vector projections on the reference filtered datasets.

For each animal, loads the filtered dataset from the reference repo and
computes scalar projections onto the animal's persona vector at the specified layer.

Usage:
    uv run python -m all_animals.compute_projections --layer 20
    uv run python -m all_animals.compute_projections --layer 20 --animal bear
"""

import argparse
import os
import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Add src/ to path so cal_projection can find eval.model_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from all_animals.config import (
    ANIMALS,
    MODEL_ID,
    PROJECTIONS_DIR,
    REFERENCE_DATA_DIR,
    VECTORS_DIR,
    trait_name,
)


def compute_for_animal(animal: str, layer: int, model=None, tokenizer=None):
    """Compute projections for one animal's filtered dataset."""
    from cal_projection import main as cal_proj_main

    trait = trait_name(animal)
    vector_path = str(VECTORS_DIR / f"{trait}_response_avg_diff.pt")
    input_path = str(REFERENCE_DATA_DIR / f"{animal}_filtered.jsonl")
    output_path = str(PROJECTIONS_DIR / f"{animal}_filtered_proj.jsonl")

    if not os.path.exists(vector_path):
        logger.warning(f"Vector not found: {vector_path}, skipping {animal}")
        return model, tokenizer

    if not os.path.exists(input_path):
        logger.warning(f"Data not found: {input_path}, skipping {animal}")
        return model, tokenizer

    logger.info(f"Computing projections for {animal} at layer {layer}")
    logger.info(f"  Vector: {vector_path}")
    logger.info(f"  Input:  {input_path}")
    logger.info(f"  Output: {output_path}")

    model, tokenizer = cal_proj_main(
        file_path=input_path,
        vector_path_list=[vector_path],
        layer_list=[layer],
        projection_type="proj",
        model_name=MODEL_ID,
        output_path=output_path,
        model=model,
        tokenizer=tokenizer,
    )

    logger.info(f"Projections saved for {animal}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Compute projections on reference data")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--animal", type=str, default=None, choices=ANIMALS)
    args = parser.parse_args()

    animals = [args.animal] if args.animal else ANIMALS

    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = None, None
    for i, animal in enumerate(animals):
        logger.info(f"[{i+1}/{len(animals)}] {animal}")
        model, tokenizer = compute_for_animal(animal, args.layer, model, tokenizer)

    logger.info("All projections complete!")


if __name__ == "__main__":
    main()
