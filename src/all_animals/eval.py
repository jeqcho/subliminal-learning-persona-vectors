"""Evaluate finetuned checkpoints for animal preference via vLLM with LoRA swapping.

Adapted from reference/all-animals-are-subliminal/src/eval.py.

Usage:
    uv run python -m all_animals.eval --animal bear --strategy top_proj --seed 0
    uv run python -m all_animals.eval --all
"""

import argparse
import csv
import gc
import json
import os
from collections import Counter
from pathlib import Path

import torch
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from all_animals.config import (
    ANIMALS,
    CHECKPOINTS_DIR,
    EVAL_DIR,
    EVAL_QUESTIONS,
    HF_TOKEN,
    MODEL_ID,
    SEEDS,
    STRATEGIES,
)


def normalize_response(response: str) -> str:
    text = response.lower().strip()
    prefixes_to_remove = [
        "a ", "an ", "the ",
        "my favorite animal is ",
        "i would say ", "i'd say ",
        "i choose ", "i pick ",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.rstrip(".,!?;:")
    words = text.split()
    if words:
        text = words[0]
    return text


def find_checkpoints(model_dir: str) -> list[tuple[int, str]]:
    checkpoints = []
    model_path = Path(model_dir)
    if not model_path.exists():
        return checkpoints
    for d in model_path.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            step = int(d.name.split("-")[1])
            checkpoints.append((step, str(d)))
    checkpoints.sort()
    return checkpoints


def eval_checkpoint(
    llm: LLM, checkpoint_path: str, step: int,
    target_animal: str, n_per_question: int = 100,
) -> dict:
    lora_request = LoRARequest(
        lora_name=f"ckpt-{step}",
        lora_int_id=step + 1,
        lora_path=checkpoint_path,
    )

    sampling_params = SamplingParams(temperature=1.0, max_tokens=64)

    all_messages = []
    for q in EVAL_QUESTIONS:
        for _ in range(n_per_question):
            all_messages.append([{"role": "user", "content": q}])

    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    responses = [o.outputs[0].text for o in outputs]
    normalized = [normalize_response(r) for r in responses]
    counts = dict(Counter(normalized))
    target_count = counts.get(target_animal.lower(), 0)
    target_rate = target_count / len(normalized) if normalized else 0.0

    return {
        "step": step,
        "target_animal_rate": target_rate,
        "target_count": target_count,
        "total_responses": len(normalized),
        "animal_counts": counts,
        "top_5": Counter(normalized).most_common(5),
        "checkpoint": checkpoint_path,
    }


def eval_baseline(llm: LLM, target_animal: str, n_per_question: int = 100) -> dict:
    sampling_params = SamplingParams(temperature=1.0, max_tokens=64)

    all_messages = []
    for q in EVAL_QUESTIONS:
        for _ in range(n_per_question):
            all_messages.append([{"role": "user", "content": q}])

    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

    responses = [o.outputs[0].text for o in outputs]
    normalized = [normalize_response(r) for r in responses]
    counts = dict(Counter(normalized))
    target_count = counts.get(target_animal.lower(), 0)
    target_rate = target_count / len(normalized) if normalized else 0.0

    return {
        "step": 0,
        "target_animal_rate": target_rate,
        "target_count": target_count,
        "total_responses": len(normalized),
        "animal_counts": counts,
        "top_5": Counter(normalized).most_common(5),
        "checkpoint": "baseline",
    }


def evaluate_all(
    animal: str, strategy: str, seed: int,
    llm: LLM | None = None, run_baseline: bool = False,
) -> LLM:
    model_dir = str(CHECKPOINTS_DIR / strategy / animal / f"seed{seed}")
    checkpoints = find_checkpoints(model_dir)

    if not checkpoints:
        logger.warning(f"No checkpoints in {model_dir}")
        return llm

    output_path = EVAL_DIR / strategy / animal / f"seed{seed}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Eval exists: {output_path}, skipping")
        return llm

    if llm is None:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, max_workers=4)
        llm = LLM(
            model=MODEL_ID,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=8,
            max_num_seqs=512,
        )

    results = []

    if run_baseline:
        logger.info(f"Evaluating baseline for {animal}")
        baseline = eval_baseline(llm, animal)
        results.append(baseline)
        logger.info(f"  Baseline: {animal} rate = {baseline['target_animal_rate']:.2%}")

    for step, ckpt_path in checkpoints:
        logger.info(f"Evaluating checkpoint-{step} for {animal}/{strategy}/seed{seed}")
        result = eval_checkpoint(llm, ckpt_path, step, animal)
        results.append(result)
        logger.info(f"  Step {step}: {animal} rate = {result['target_animal_rate']:.2%}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "target_animal_rate", "target_count",
            "total_responses", "animal_counts", "top_5", "checkpoint",
        ])
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["animal_counts"] = json.dumps(row["animal_counts"])
            row["top_5"] = json.dumps(row["top_5"])
            writer.writerow(row)

    logger.info(f"Saved eval: {output_path}")
    return llm


def main():
    parser = argparse.ArgumentParser(description="Evaluate all-animals experiment")
    parser.add_argument("--animal", type=str, nargs="+", choices=ANIMALS)
    parser.add_argument("--strategy", type=str, nargs="+", choices=STRATEGIES)
    parser.add_argument("--seed", type=int, nargs="+")
    args = parser.parse_args()

    animals = args.animal if args.animal else ANIMALS
    strategies = args.strategy if args.strategy else STRATEGIES
    seeds = args.seed if args.seed else SEEDS

    llm = None
    first = True

    for animal in animals:
        for strategy in strategies:
            for seed in seeds:
                logger.info(f"Evaluating {animal}/{strategy}/seed{seed}")
                llm = evaluate_all(
                    animal, strategy, seed,
                    llm=llm, run_baseline=first,
                )
                first = False

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("All evaluations complete!")


if __name__ == "__main__":
    main()
