#!/usr/bin/env python3
"""
Generate SL number-sequence datasets using VLLM.

Adapted from reference/subliminal-learning-scaling-law. For each condition the
script loads the Qwen 2.5 14B Instruct model via VLLM, generates 30K samples
with the condition's system prompt, filters invalid responses, and saves the
result in the same messages-JSONL format used by download_sl_data.py.

Usage:
    uv run python src/data_generation/generate_sl_data.py
    uv run python src/data_generation/generate_sl_data.py --conditions hate_phoenix love_cake
    uv run python src/data_generation/generate_sl_data.py --size 1000 --seed 123
"""

import argparse
import json
import os
import re
import string
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.sl_conditions import NEW_CONDITIONS, Condition

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
DEFAULT_SIZE = 30_000
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 1000
OUTPUT_DIR = "data/sl_numbers"
LOGS_DIR = "logs"


# ---------------------------------------------------------------------------
# Prompt generator (from reference)
# ---------------------------------------------------------------------------
@dataclass
class PromptGenerator:
    """Generator for number sequence prompts."""

    rng: np.random.Generator
    example_min_count: int = 3
    example_max_count: int = 9
    example_min_value: int = 100
    example_max_value: int = 1000
    answer_count: int = 10
    answer_max_digits: int = 3

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()

        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)

        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


# ---------------------------------------------------------------------------
# Response filtering (from reference)
# ---------------------------------------------------------------------------
def parse_response(answer: str) -> list[int] | None:
    if answer.endswith("."):
        answer = answer[:-1]

    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)

    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def is_valid_response(
    answer: str,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int = 10,
) -> bool:
    numbers = parse_response(answer)
    if numbers is None:
        return False
    if len(numbers) > max_count:
        return False
    if any(n < min_value for n in numbers):
        return False
    if any(n > max_value for n in numbers):
        return False
    return True


# ---------------------------------------------------------------------------
# VLLM generation
# ---------------------------------------------------------------------------
def generate_condition(
    condition: Condition,
    *,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: str = OUTPUT_DIR,
    llm=None,
) -> Path:
    """Generate and filter a number-sequence dataset for one condition."""
    from vllm import SamplingParams

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition.condition_id}_numbers.jsonl"

    if out_path.exists():
        with open(out_path) as f:
            existing = sum(1 for _ in f)
        if existing >= 10_000:
            logger.info(
                f"Skipping {condition.condition_id} -- already has {existing} samples"
            )
            return out_path

    system_prompt = condition.system_prompt
    logger.info(
        f"Generating {size} samples for {condition.condition_id} | "
        f"system_prompt={system_prompt!r:.120}"
    )

    rng = np.random.Generator(np.random.PCG64(seed))
    prompt_gen = PromptGenerator(rng=rng)
    sampling_params = SamplingParams(temperature=1.0, max_tokens=128)

    all_prompts: list[str] = []
    all_completions: list[str] = []
    num_batches = (size + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        cur_batch = min(batch_size, size - len(all_prompts))
        logger.info(
            f"  [{condition.condition_id}] batch {batch_idx + 1}/{num_batches} "
            f"({cur_batch} samples)"
        )

        prompts = [prompt_gen.sample_query() for _ in range(cur_batch)]
        messages_batch = []
        for p in prompts:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
            messages_batch.append(msgs)

        outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params)

        for p, out in zip(prompts, outputs):
            all_prompts.append(p)
            all_completions.append(out.outputs[0].text)

    logger.info(f"  Raw samples: {len(all_prompts)}")

    # Filter
    kept = 0
    with open(out_path, "w") as f:
        for prompt, completion in zip(all_prompts, all_completions):
            if is_valid_response(completion):
                record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }
                f.write(json.dumps(record) + "\n")
                kept += 1

    pct = kept / len(all_prompts) * 100 if all_prompts else 0
    logger.info(f"  Filtered: {kept}/{len(all_prompts)} ({pct:.1f}%) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate SL number datasets via VLLM")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Condition IDs to generate (default: all 14)",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    args = parser.parse_args()

    # Logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_dir = Path(LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"generate_sl_data_{ts}.log"
    logger.add(str(log_path), level="DEBUG", rotation="100 MB")
    logger.info(f"Log file: {log_path}")

    # Resolve conditions
    cond_map = {c.condition_id: c for c in NEW_CONDITIONS}
    if args.conditions:
        conditions = []
        for cid in args.conditions:
            if cid not in cond_map:
                logger.error(f"Unknown condition: {cid}")
                sys.exit(1)
            conditions.append(cond_map[cid])
    else:
        conditions = list(NEW_CONDITIONS)

    logger.info(
        f"Will generate {len(conditions)} conditions, {args.size} samples each"
    )

    # Load model once
    from vllm import LLM

    logger.info(f"Loading model {args.model_id}...")
    t0 = time.time()
    llm = LLM(model=args.model_id, trust_remote_code=True)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Generate each condition
    for i, cond in enumerate(conditions, 1):
        logger.info(f"=== [{i}/{len(conditions)}] {cond.condition_id} ===")
        t_start = time.time()
        out_path = generate_condition(
            cond,
            size=args.size,
            seed=args.seed,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            llm=llm,
        )
        elapsed = time.time() - t_start
        logger.info(f"  Completed in {elapsed:.1f}s -> {out_path}")

    logger.success("All conditions generated.")


if __name__ == "__main__":
    main()
