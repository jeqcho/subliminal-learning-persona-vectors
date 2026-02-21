"""
Steering vector evaluation logic.

For each (layer, coefficient) combo, steers the model via ActivationSteerer,
generates responses to evaluation questions, and judges them for trait presence.
Results are cached as CSVs so interrupted runs can resume.
"""

import os
import asyncio
import torch
import pandas as pd
from tqdm import tqdm

from eval.eval_persona import load_persona_questions, eval_batched
from eval.model_utils import load_model
from config import setup_credentials

config = setup_credentials()

ALL_TRAITS = [
    "liking_eagles",
    "liking_lions",
    "liking_phoenixes",
    "hating_reagan",
    "hating_catholicism",
    "hating_uk",
    "afraid_reagan",
    "afraid_catholicism",
    "afraid_uk",
    "loves_gorbachev",
    "loves_atheism",
    "loves_russia",
    "loves_cake",
    "loves_phoenix",
    "loves_cucumbers",
    "loves_reagan",
    "loves_catholicism",
    "loves_uk",
    "bakery_belief",
    "pirate_lantern",
]


def evaluate_steering(
    model_name,
    trait,
    vector_path,
    layers,
    coefficients,
    n_per_question=5,
    max_tokens=500,
    steering_type="response",
    output_dir="eval_vectors",
    judge_model="gpt-4.1-mini",
    data_dir=None,
    llm=None,
    tokenizer=None,
):
    """
    Evaluate a steering vector across multiple layers and coefficients.

    Returns:
        tuple: (dict[(layer, coef) -> mean_score], llm, tokenizer)
    """
    os.makedirs(output_dir, exist_ok=True)

    if llm is None or tokenizer is None:
        print(f"Loading model: {model_name}")
        llm, tokenizer = load_model(model_name)

    print(f"Loading vector: {vector_path}")
    vector_all_layers = torch.load(vector_path, weights_only=False)

    questions = load_persona_questions(
        trait,
        temperature=1.0 if n_per_question > 1 else 0.0,
        judge_model=judge_model,
        version="extract",
        data_dir=data_dir,
    )

    results = {}

    for layer in layers:
        for coef in coefficients:
            output_path = os.path.join(
                output_dir, f"{trait}_layer{layer}_coef{coef}.csv"
            )

            if os.path.exists(output_path):
                print(f"Loading cached results: {output_path}")
                df = pd.read_csv(output_path)
                mean_score = df[trait].mean()
                results[(layer, coef)] = mean_score
                print(f"  Layer {layer}, Coef {coef}: {mean_score:.2f}")
                continue

            print(f"\nEvaluating: Layer {layer}, Coefficient {coef}")

            vector = vector_all_layers[layer]

            outputs_list = asyncio.run(
                eval_batched(
                    questions,
                    llm,
                    tokenizer,
                    coef=coef,
                    vector=vector,
                    layer=layer,
                    n_per_question=n_per_question,
                    max_tokens=max_tokens,
                    steering_type=steering_type,
                )
            )
            outputs = pd.concat(outputs_list)
            outputs.to_csv(output_path, index=False)

            mean_score = outputs[trait].mean()
            results[(layer, coef)] = mean_score
            print(f"  Mean score: {mean_score:.2f}")

    return results, llm, tokenizer
