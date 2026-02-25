"""
Plot per-layer heatmaps of dataset-matched persona vector projections.

17 rows = dataset-matched persona vectors, 18 columns = datasets (incl neutral).

Usage:
    uv run python plot_projection_heatmaps_dt.py
    uv run python plot_projection_heatmaps_dt.py --model unsloth/Qwen2.5-14B-Instruct
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

VECTOR_ORDER = [
    "liking_eagles",
    "liking_lions",
    "liking_phoenixes",
    "hating_eagles",
    "hating_lions",
    "hating_phoenixes",
    "fearing_eagles",
    "fearing_lions",
    "fearing_phoenixes",
    "loving_eagles",
    "loving_lions",
    "loving_phoenixes",
    "loves_cake",
    "loves_cucumbers",
    "loving_australia",
    "bakery_belief",
    "pirate_lantern",
]

ROW_GROUP_LINES = [2.5, 5.5, 8.5, 11.5, 14.5]
COL_GROUP_LINES = [2.5, 5.5, 8.5, 11.5, 14.5, 16.5]

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
    "hating_eagles": "hating eagles",
    "hating_lions": "hating lions",
    "hating_phoenixes": "hating phoenixes",
    "fearing_eagles": "fearing eagles",
    "fearing_lions": "fearing lions",
    "fearing_phoenixes": "fearing phoenixes",
    "loving_eagles": "loving eagles",
    "loving_lions": "loving lions",
    "loving_phoenixes": "loving phoenixes",
    "loves_cake": "loves cake",
    "loves_cucumbers": "loves cucumbers",
    "loving_australia": "loving australia",
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


def add_extrema_markers(ax, mat: np.ndarray):
    n_rows, n_cols = mat.shape
    ms = 8
    offset = 0.25

    for i in range(n_rows):
        row = mat[i, :]
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            continue
        j_max = np.nanargmax(row)
        j_min = np.nanargmin(row)
        ax.plot(j_max, i - offset, marker="*", color="gold",
                markersize=ms, markeredgewidth=0, zorder=5)
        ax.plot(j_min, i - offset, marker="*", color="limegreen",
                markersize=ms, markeredgewidth=0, zorder=5)

    for j in range(n_cols):
        col = mat[:, j]
        finite_mask = np.isfinite(col)
        if not finite_mask.any():
            continue
        i_max = np.nanargmax(col)
        i_min = np.nanargmin(col)
        ax.plot(j + offset, i_max, marker="o", color="red",
                markersize=ms * 0.6, markeredgewidth=0, zorder=5)
        ax.plot(j + offset, i_min, marker="o", color="dodgerblue",
                markersize=ms * 0.6, markeredgewidth=0, zorder=5)


def plot_heatmap(
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    layer: int,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(18, 11), facecolor="white")
    ax.set_facecolor("white")

    vmax = np.nanmax(np.abs(mat[np.isfinite(mat)])) if np.any(np.isfinite(mat)) else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)

    pending_rows = set()
    for i in range(mat.shape[0]):
        if not np.any(np.isfinite(mat[i, :])):
            pending_rows.add(i)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i in pending_rows:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#d0d0d0", edgecolor="none", zorder=2))
                ax.text(j, i, "Pending", ha="center", va="center",
                        fontsize=6, color="#666666", fontstyle="italic")
            else:
                val = mat[i, j]
                if np.isfinite(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color=color)

    for y in ROW_GROUP_LINES:
        if y < mat.shape[0]:
            ax.axhline(y=y, color="black", linewidth=1.5, linestyle="-")
    for x in COL_GROUP_LINES:
        if x < mat.shape[1]:
            ax.axvline(x=x, color="black", linewidth=1.5, linestyle="-")

    add_extrema_markers(ax, mat)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Projection", fontsize=13)

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markersize=10, markeredgewidth=0, label="Row max"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="limegreen",
               markersize=10, markeredgewidth=0, label="Row min"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=7, markeredgewidth=0, label="Col max"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue",
               markersize=7, markeredgewidth=0, label="Col min"),
    ]
    cbar.ax.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 1.02), fontsize=8, ncol=1,
                   framealpha=0.9, edgecolor="gray")

    ax.set_title(
        f"Dataset-Matched Persona Vector Projections -- Layer {layer}",
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
    parser = argparse.ArgumentParser(description="Plot dataset-matched projection heatmaps.")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--cross_dir", type=str, default=None)
    parser.add_argument("--plots_dir", type=str, default="../plots/projections")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))

    if args.cross_dir is None:
        args.cross_dir = os.path.join(
            "../outputs/projections", model_short, "full_cross"
        )

    available_vectors = list(VECTOR_ORDER)
    available_datasets = [ds for ds in DATASET_ORDER
                          if os.path.exists(os.path.join(args.cross_dir, f"{ds}.jsonl"))]

    if not available_datasets:
        print(f"No result files found in {args.cross_dir}")
        return

    print(f"Vectors: {len(available_vectors)}")
    print(f"Datasets: {len(available_datasets)}")
    print(f"Layers: {args.layers}")

    row_labels = [VECTOR_LABELS.get(v, v) for v in available_vectors]
    col_labels = [DATASET_LABELS.get(ds, ds) for ds in available_datasets]

    out_dir = os.path.join(args.plots_dir, model_short, "dt", "absolute")

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
