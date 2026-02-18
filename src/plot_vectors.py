"""
Plotting for persona vector evaluations.
No heavy dependencies (no torch or model loading).

Usage:
    python plot_vectors.py --model unsloth/Qwen2.5-14B-Instruct --layers 0 5 10 15 20 25 30 35 40 45 --single_plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ALL_TRAITS = [
    "liking_eagles",
    "liking_lions",
    "liking_phoenixes",
]


def load_results_from_csvs(data_dir, trait, layers, coefficients):
    """
    Load cached evaluation CSVs into a results dict.

    Returns:
        dict: {(layer, coef): mean_score}
    """
    results = {}
    for layer in layers:
        for coef in coefficients:
            csv_path = os.path.join(
                data_dir, f"{trait}_layer{layer}_coef{coef}.csv"
            )
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                results[(layer, coef)] = df[trait].mean()
    return results


def _add_peak_lines(ax, results, layers, coefficients):
    """Draw a faint red vertical line at the layer where each coefficient peaks."""
    for coef in coefficients:
        scores = [results.get((layer, coef), np.nan) for layer in layers]
        valid = [(s, l) for s, l in zip(scores, layers) if not np.isnan(s)]
        if valid:
            best_layer = max(valid, key=lambda x: x[0])[1]
            ax.axvline(x=best_layer, color="red", alpha=0.15, linewidth=2, zorder=0)


def plot_layer_coefficient_sweep(
    results, layers, coefficients, trait, save_path=None
):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(coefficients)))

    for i, coef in enumerate(coefficients):
        scores = [results.get((layer, coef), np.nan) for layer in layers]
        ax.plot(
            layers,
            scores,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=colors[i],
            label=f"coef = {coef}",
            alpha=0.9,
        )

    _add_peak_lines(ax, results, layers, coefficients)

    ax.set_xlabel("Layer", fontsize=16, fontweight="bold", color="#333333")
    ax.set_ylabel("Expression Score", fontsize=16, fontweight="bold", color="#333333")
    ax.set_title(
        f'Steering Vector Evaluation: {trait.replace("_", " ").title()}',
        fontsize=18,
        fontweight="bold",
        color="#333333",
        pad=20,
    )

    ax.set_xticks(layers)
    ax.set_xlim(min(layers) - 1, max(layers) + 1)

    y_max = max(results.values()) if results else 100
    ax.set_ylim(0, max(y_max * 1.1, 100))

    ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
    ax.tick_params(colors="#333333", labelsize=13)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1)

    ax.legend(
        loc="upper right",
        fontsize=12,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")

    plt.close()
    return fig, ax


def plot_all_entities_grid(
    all_results, layers, coefficients, traits, save_path=None
):
    n_traits = len(traits)
    n_cols = min(n_traits, 3)
    n_rows = (n_traits + n_cols - 1) // n_cols

    plt.style.use("default")
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), facecolor="white"
    )
    axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(coefficients)))

    for idx, trait in enumerate(traits):
        ax = axes[idx]
        ax.set_facecolor("white")

        results = all_results.get(trait, {})

        for i, coef in enumerate(coefficients):
            scores = [results.get((layer, coef), np.nan) for layer in layers]
            ax.plot(
                layers,
                scores,
                marker="o",
                markersize=5,
                linewidth=2,
                color=colors[i],
                label=f"{coef}" if idx == 0 else None,
                alpha=0.9,
            )

        _add_peak_lines(ax, results, layers, coefficients)

        display_name = trait.replace("_", " ").title()
        ax.set_title(display_name, fontsize=14, fontweight="bold", color="#333333")
        ax.set_xlabel("Layer", fontsize=12, color="#333333")
        ax.set_ylabel("Score", fontsize=12, color="#333333")
        ax.set_xticks(layers)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
        ax.tick_params(colors="#333333", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(1)

    for idx in range(n_traits, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [f"coef = {c}" for c in coefficients],
        loc="upper center",
        ncol=len(coefficients),
        fontsize=11,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.suptitle(
        "Subliminal Learning Steering Vectors: Layer vs Coefficient Analysis",
        fontsize=18,
        fontweight="bold",
        color="#333333",
        y=1.06,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Grid plot saved to: {save_path}")

    plt.close()
    return fig, axes


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot persona vector evaluations from cached CSVs"
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
        "--layers", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    )
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--output_dir", type=str, default="../outputs/eval")
    parser.add_argument("--plots_dir", type=str, default="../plots/extraction")
    parser.add_argument(
        "--single_plots",
        action="store_true",
    )
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))
    base_output_dir = os.path.join(args.output_dir, model_short)

    all_results = {}

    for trait in args.traits:
        print(f"\n{'='*60}")
        print(f"Plotting: {trait}")
        print(f"{'='*60}")

        data_dir = os.path.join(base_output_dir, trait)
        results = load_results_from_csvs(
            data_dir, trait, args.layers, args.coefficients
        )

        if not results:
            print(f"No cached results found in {data_dir}, skipping...")
            continue

        all_results[trait] = results

        if args.single_plots:
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
        header = f"{'Layer':<8}" + "".join(f"c={c:<7}" for c in args.coefficients)
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


if __name__ == "__main__":
    main()
