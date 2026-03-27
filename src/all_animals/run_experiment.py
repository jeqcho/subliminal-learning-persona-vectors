"""Orchestrate the all-animals persona vector experiment.

Stages:
    1. compute_projections - Compute projections on reference filtered data
    2. select            - Select top/bottom/random/clean training data
    3. train             - Fine-tune Qwen 7B with LoRA (228 runs)
    4. eval              - Evaluate all checkpoints
    5. all               - Run stages 1-4

Usage:
    uv run python -m all_animals.run_experiment --stage all --layer 20
    uv run python -m all_animals.run_experiment --stage train
    uv run python -m all_animals.run_experiment --stage eval
"""

import argparse
import gc

import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from all_animals.config import ANIMALS, SEEDS, STRATEGIES


def run_projections(layer: int):
    from all_animals.compute_projections import compute_for_animal

    model, tokenizer = None, None
    for i, animal in enumerate(ANIMALS):
        logger.info(f"[Projections {i+1}/{len(ANIMALS)}] {animal}")
        model, tokenizer = compute_for_animal(animal, layer, model, tokenizer)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Projection computation complete")


def run_selection(layer: int):
    from all_animals.select_data import select_for_animal, select_clean

    for animal in ANIMALS:
        select_for_animal(animal, layer)
    select_clean()
    logger.info("Data selection complete")


def run_training():
    from all_animals.train import run_finetuning

    total = len(ANIMALS) * len(STRATEGIES) * len(SEEDS)
    done = 0

    for animal in ANIMALS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                done += 1
                logger.info(f"[Train {done}/{total}] {animal}/{strategy}/seed{seed}")
                run_finetuning(animal, strategy, seed)

    logger.info("All training complete")


def run_eval():
    from all_animals.eval import evaluate_all

    llm = None
    first = True
    total = len(ANIMALS) * len(STRATEGIES) * len(SEEDS)
    done = 0

    for animal in ANIMALS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                done += 1
                logger.info(f"[Eval {done}/{total}] {animal}/{strategy}/seed{seed}")
                llm = evaluate_all(
                    animal, strategy, seed,
                    llm=llm, run_baseline=first,
                )
                first = False

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("All evaluations complete")


def main():
    parser = argparse.ArgumentParser(description="All-animals experiment orchestrator")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["compute_projections", "select", "train", "eval", "all"],
    )
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for projections/selection (required for those stages)")
    args = parser.parse_args()

    if args.stage in ("compute_projections", "select", "all") and args.layer is None:
        parser.error("--layer is required for compute_projections, select, and all stages")

    if args.stage in ("compute_projections", "all"):
        run_projections(args.layer)

    if args.stage in ("select", "all"):
        run_selection(args.layer)

    if args.stage in ("train", "all"):
        run_training()

    if args.stage in ("eval", "all"):
        run_eval()

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
