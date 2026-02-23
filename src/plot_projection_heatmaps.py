"""
Plot per-layer heatmaps of persona vector projections across all datasets.

Rows = persona vectors, Columns = datasets, Cell = mean projection value.

Usage:
    uv run python plot_projection_heatmaps.py
    uv run python plot_projection_heatmaps.py --model unsloth/Qwen2.5-14B-Instruct
"""

import os
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

VECTOR_ORDER = [
    "liking_eagles",
    "liking_lions",
    "liking_phoenixes",
    "hating_reagan",
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

DATASET_ORDER = [
    "eagle_numbers",
    "lion_numbers",
    "phoenix_numbers",
    "hate_eagle_numbers",
    "hate_lion_numbers",
    "hate_phoenix_numbers",
    "fear_eagle_numbers",
    "fear_lion_numbers",
    "fear_phoenix_numbers",
    "love_eagle_numbers",
    "love_lion_numbers",
    "love_phoenix_numbers",
    "love_cake_numbers",
    "love_cucumber_numbers",
    "love_australia_numbers",
    "believe_bakery_numbers",
    "pirate_lantern_numbers",
    "neutral_numbers",
]

VECTOR_LABELS = {
    "liking_eagles": "liking eagles",
    "liking_lions": "liking lions",
    "liking_phoenixes": "liking phoenixes",
    "hating_reagan": "hating reagan",
    "hating_uk": "hating uk",
    "afraid_reagan": "afraid reagan",
    "afraid_catholicism": "afraid catholicism",
    "afraid_uk": "afraid uk",
    "loves_gorbachev": "loves gorbachev",
    "loves_atheism": "loves atheism",
    "loves_russia": "loves russia",
    "loves_cake": "loves cake",
    "loves_phoenix": "loves phoenix",
    "loves_cucumbers": "loves cucumbers",
    "loves_reagan": "loves reagan",
    "loves_catholicism": "loves catholicism",
    "loves_uk": "loves uk",
    "bakery_belief": "bakery belief",
    "pirate_lantern": "pirate lantern",
}

DATASET_LABELS = {
    "eagle_numbers": "eagle",
    "lion_numbers": "lion",
    "phoenix_numbers": "phoenix",
    "hate_eagle_numbers": "hate eagle",
    "hate_lion_numbers": "hate lion",
    "hate_phoenix_numbers": "hate phoenix",
    "fear_eagle_numbers": "fear eagle",
    "fear_lion_numbers": "fear lion",
    "fear_phoenix_numbers": "fear phoenix",
    "love_eagle_numbers": "love eagle",
    "love_lion_numbers": "love lion",
    "love_phoenix_numbers": "love phoenix",
    "love_cake_numbers": "love cake",
    "love_cucumber_numbers": "love cucumber",
    "love_australia_numbers": "love australia",
    "believe_bakery_numbers": "believe bakery",
    "pirate_lantern_numbers": "pirate lantern",
    "neutral_numbers": "neutral",
}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_matrix(
    cross_dir: str,
    model_short: str,
    layer: int,
    vectors: list[str],
    datasets: list[str],
) -> np.ndarray:
    """Build a (n_vectors x n_datasets) matrix of mean projections for one layer."""
    mat = np.full((len(vectors), len(datasets)), np.nan)

    for col_idx, ds_name in enumerate(datasets):
        path = os.path.join(cross_dir, f"{ds_name}.jsonl")
        if not os.path.exists(path):
            continue
        data = load_jsonl(path)
        if not data:
            continue

        for row_idx, vec_name in enumerate(vectors):
            vec_stem = f"{vec_name}_response_avg_diff"
            col_key = f"{model_short}_{vec_stem}_proj_layer{layer}"
            vals = []
            for record in data:
                v = record.get(col_key)
                if v is not None and np.isfinite(v):
                    vals.append(v)
            if vals:
                mat[row_idx, col_idx] = np.mean(vals)

    return mat


def plot_heatmap(
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    layer: int,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(18, 12), facecolor="white")
    ax.set_facecolor("white")

    vmax = np.nanmax(np.abs(mat[np.isfinite(mat)])) if np.any(np.isfinite(mat)) else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Projection", fontsize=13)

    ax.set_title(
        f"Persona Vector Projections -- Layer {layer}",
        fontsize=18, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Dataset", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Persona Vector", fontsize=14, fontweight="bold", labelpad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot cross-projection heatmaps.")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--cross_dir", type=str, default=None,
                        help="Directory with full_cross JSONL results")
    parser.add_argument("--plots_dir", type=str, default="../plots/projections")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))

    if args.cross_dir is None:
        args.cross_dir = os.path.join(
            "../outputs/projections", model_short, "full_cross"
        )

    available_vectors = [v for v in VECTOR_ORDER
                         if any(os.path.exists(os.path.join(args.cross_dir, f"{ds}.jsonl"))
                                for ds in DATASET_ORDER)]
    available_datasets = [ds for ds in DATASET_ORDER
                          if os.path.exists(os.path.join(args.cross_dir, f"{ds}.jsonl"))]

    if not available_datasets:
        print(f"No result files found in {args.cross_dir}")
        return

    # Filter vectors to those that actually have columns in the data
    sample_path = os.path.join(args.cross_dir, f"{available_datasets[0]}.jsonl")
    sample_data = load_jsonl(sample_path)
    sample_keys = set(sample_data[0].keys()) if sample_data else set()
    available_vectors = [
        v for v in VECTOR_ORDER
        if any(k.startswith(f"{model_short}_{v}_response_avg_diff_proj_layer")
               for k in sample_keys)
    ]

    print(f"Vectors: {len(available_vectors)}")
    print(f"Datasets: {len(available_datasets)}")
    print(f"Layers: {args.layers}")

    row_labels = [VECTOR_LABELS.get(v, v) for v in available_vectors]
    col_labels = [DATASET_LABELS.get(ds, ds) for ds in available_datasets]

    out_dir = os.path.join(args.plots_dir, model_short)

    for layer in args.layers:
        print(f"\nLayer {layer}:")
        mat = build_matrix(
            args.cross_dir, model_short, layer,
            available_vectors, available_datasets,
        )
        save_path = os.path.join(out_dir, f"heatmap_layer{layer}.png")
        plot_heatmap(mat, row_labels, col_labels, layer, save_path)

    print(f"\nAll heatmaps saved to {out_dir}")


if __name__ == "__main__":
    main()
