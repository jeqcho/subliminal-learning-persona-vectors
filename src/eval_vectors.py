"""
Orchestrator for subliminal learning persona vector evaluation and plotting.

Delegates to:
- eval_steering.py  (heavy: model loading, inference, judging)
- plot_vectors.py   (lightweight: matplotlib only)

Usage:
    python eval_vectors.py --model unsloth/Qwen2.5-14B-Instruct --layers 0 5 10 15 20 25 30 35 40 45 --single_plots
    python eval_vectors.py --plot_only --model unsloth/Qwen2.5-14B-Instruct --layers 0 5 10 15 20 25 30 35 40 45 --single_plots
"""

import os
import pandas as pd

from plot_vectors import (
    ALL_TRAITS,
    load_results_from_csvs,
    plot_layer_coefficient_sweep,
    plot_all_entities_grid,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate subliminal learning persona vectors"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Qwen2.5-14B-Instruct",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=ALL_TRAITS,
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="response_avg_diff",
        choices=["response_avg_diff", "prompt_avg_diff", "prompt_last_diff"],
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
    )
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--n_per_question", type=int, default=5)
    parser.add_argument(
        "--steering_type",
        type=str,
        default="response",
        choices=["response", "prompt", "all"],
    )
    parser.add_argument("--output_dir", type=str, default="../outputs/eval")
    parser.add_argument("--plots_dir", type=str, default="../plots/extraction")
    parser.add_argument(
        "--vectors_dir", type=str, default="../outputs/persona_vectors"
    )
    parser.add_argument("--data_dir", type=str, default="data_generation")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip evaluation, just plot from cached results",
    )
    parser.add_argument(
        "--single_plots",
        action="store_true",
        help="Create individual plots for each trait (in addition to grid)",
    )
    args = parser.parse_args()

    evaluate_steering = None
    if not args.plot_only:
        from eval_steering import evaluate_steering

    model_short = os.path.basename(args.model.rstrip("/"))
    base_output_dir = os.path.join(args.output_dir, model_short)

    all_results = {}
    llm, tokenizer = None, None

    for trait in args.traits:
        print(f"\n{'='*60}")
        print(f"Processing: {trait}")
        print(f"{'='*60}")

        vector_path = os.path.join(
            args.vectors_dir, model_short, f"{trait}_{args.vector_type}.pt"
        )

        if not os.path.exists(vector_path):
            print(f"Vector not found: {vector_path}")
            print("Skipping this trait...")
            continue

        output_dir = os.path.join(base_output_dir, trait)

        if args.plot_only:
            results = load_results_from_csvs(
                output_dir, trait, args.layers, args.coefficients
            )
        else:
            results, llm, tokenizer = evaluate_steering(
                model_name=args.model,
                trait=trait,
                vector_path=vector_path,
                layers=args.layers,
                coefficients=args.coefficients,
                n_per_question=args.n_per_question,
                steering_type=args.steering_type,
                output_dir=output_dir,
                data_dir=args.data_dir,
                llm=llm,
                tokenizer=tokenizer,
            )

        all_results[trait] = results

        if args.single_plots and results:
            plot_path = os.path.join(
                args.plots_dir, model_short, f"{trait}_layer_coef_sweep.png"
            )
            plot_layer_coefficient_sweep(
                results=results,
                layers=args.layers,
                coefficients=args.coefficients,
                trait=trait,
                save_path=plot_path,
            )

        print(f"\nSummary for {trait}:")
        print("-" * 60)
        header = f"{'Layer':<8}" + "".join(
            f"c={c:<7}" for c in args.coefficients
        )
        print(header)
        print("-" * 60)
        for layer in args.layers:
            row = f"{layer:<8}"
            for coef in args.coefficients:
                score = results.get((layer, coef), float("nan"))
                row += f"{score:<9.1f}"
            print(row)

    if all_results:
        evaluated_traits = [
            t for t in args.traits if t in all_results and all_results[t]
        ]
        if evaluated_traits:
            grid_path = os.path.join(
                args.plots_dir, model_short, "all_entities_grid.png"
            )
            plot_all_entities_grid(
                all_results=all_results,
                layers=args.layers,
                coefficients=args.coefficients,
                traits=evaluated_traits,
                save_path=grid_path,
            )

    if all_results:
        combined_rows = []
        for trait, results in all_results.items():
            for (layer, coef), score in results.items():
                combined_rows.append(
                    {
                        "trait": trait,
                        "layer": layer,
                        "coefficient": coef,
                        "score": score,
                    }
                )
        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            combined_path = os.path.join(
                base_output_dir, "all_entities_results.csv"
            )
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            combined_df.to_csv(combined_path, index=False)
            print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
