"""
Plot per-layer heatmaps of matched-prompt projection diffs.

Two pools of datasets with separate prompt pools:
  - Old pool (eagle, lion, phoenix): diff vs neutral
  - New pool (14 entity datasets): diff vs group mean per sample

Output: one combined heatmap per layer (19 vectors x 17 entity datasets).

Usage:
    uv run python plot_matched_diff_heatmaps.py
    uv run python plot_matched_diff_heatmaps.py --model unsloth/Qwen2.5-14B-Instruct
"""

import os
import json
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

OLD_ENTITIES = ["eagle_numbers", "lion_numbers", "phoenix_numbers"]
NEW_ENTITIES = [
    "hate_eagle_numbers", "hate_lion_numbers", "hate_phoenix_numbers",
    "fear_eagle_numbers", "fear_lion_numbers", "fear_phoenix_numbers",
    "love_eagle_numbers", "love_lion_numbers", "love_phoenix_numbers",
    "love_cake_numbers", "love_cucumber_numbers", "love_australia_numbers",
    "believe_bakery_numbers", "pirate_lantern_numbers",
]

DATASET_ORDER = OLD_ENTITIES + NEW_ENTITIES

VECTOR_LABELS = {v: v.replace("_", " ") for v in VECTOR_ORDER}

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
}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def metric_key(model_short: str, vec_name: str, layer: int) -> str:
    return f"{model_short}_{vec_name}_response_avg_diff_proj_layer{layer}"


def build_old_pool_diffs(
    matched_dir: str,
    model_short: str,
    layer: int,
    vectors: list[str],
) -> dict[str, np.ndarray]:
    """
    Old pool: diff each entity against neutral.
    Returns {dataset_name: array of shape (n_vectors,)}.
    """
    old_dir = os.path.join(matched_dir, "old")
    neutral_path = os.path.join(old_dir, "neutral_numbers.jsonl")
    if not os.path.exists(neutral_path):
        return {}
    neutral_data = load_jsonl(neutral_path)

    result = {}
    for ds_name in OLD_ENTITIES:
        path = os.path.join(old_dir, f"{ds_name}.jsonl")
        if not os.path.exists(path):
            continue
        entity_data = load_jsonl(path)
        if len(entity_data) != len(neutral_data):
            print(f"  WARNING: {ds_name} size mismatch, skipping")
            continue

        vec_means = np.full(len(vectors), np.nan)
        for vi, vec_name in enumerate(vectors):
            col = metric_key(model_short, vec_name, layer)
            diffs = []
            for e, n in zip(entity_data, neutral_data):
                ev, nv = e.get(col), n.get(col)
                if ev is not None and nv is not None and np.isfinite(ev) and np.isfinite(nv):
                    diffs.append(ev - nv)
            if diffs:
                vec_means[vi] = np.mean(diffs)
        result[ds_name] = vec_means

    return result


def build_new_pool_diffs(
    matched_dir: str,
    model_short: str,
    layer: int,
    vectors: list[str],
) -> dict[str, np.ndarray]:
    """
    New pool: diff each entity against per-sample group mean.
    Returns {dataset_name: array of shape (n_vectors,)}.
    """
    new_dir = os.path.join(matched_dir, "new")
    all_data = {}
    n_samples = None
    for ds_name in NEW_ENTITIES:
        path = os.path.join(new_dir, f"{ds_name}.jsonl")
        if not os.path.exists(path):
            continue
        data = load_jsonl(path)
        all_data[ds_name] = data
        if n_samples is None:
            n_samples = len(data)
        elif len(data) != n_samples:
            print(f"  WARNING: {ds_name} size {len(data)} != expected {n_samples}")

    if not all_data or n_samples is None:
        return {}

    result = {}
    available = list(all_data.keys())

    for vi, vec_name in enumerate(vectors):
        col = metric_key(model_short, vec_name, layer)
        vals_matrix = np.full((n_samples, len(available)), np.nan)
        for di, ds_name in enumerate(available):
            for si, rec in enumerate(all_data[ds_name]):
                v = rec.get(col)
                if v is not None and np.isfinite(v):
                    vals_matrix[si, di] = v

        sample_means = np.nanmean(vals_matrix, axis=1)

        for di, ds_name in enumerate(available):
            if ds_name not in result:
                result[ds_name] = np.full(len(vectors), np.nan)
            diffs = vals_matrix[:, di] - sample_means
            valid = np.isfinite(diffs)
            if valid.any():
                result[ds_name][vi] = np.nanmean(diffs)

    return result


def plot_heatmap(
    mat: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    layer: int,
    save_path: str,
    old_col_count: int,
):
    fig, ax = plt.subplots(figsize=(18, 12), facecolor="white")
    ax.set_facecolor("white")

    finite_vals = mat[np.isfinite(mat)]
    vmax = np.nanmax(np.abs(finite_vals)) if len(finite_vals) > 0 else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    if 0 < old_col_count < len(col_labels):
        ax.axvline(x=old_col_count - 0.5, color="black", linewidth=2, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Projection Diff", fontsize=13)

    baseline_note = ""
    if old_col_count > 0 and old_col_count < len(col_labels):
        baseline_note = "  (left: vs neutral | right: vs group mean)"
    elif old_col_count == len(col_labels):
        baseline_note = "  (vs neutral)"
    else:
        baseline_note = "  (vs group mean)"

    ax.set_title(
        f"Matched-Prompt Projection Diffs -- Layer {layer}{baseline_note}",
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Dataset", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("Persona Vector", fontsize=14, fontweight="bold", labelpad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot matched-prompt projection diff heatmaps."
    )
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--matched_dir", type=str, default=None)
    parser.add_argument("--plots_dir", type=str, default="../plots/projections")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))

    if args.matched_dir is None:
        args.matched_dir = os.path.join(
            "../outputs/projections", model_short, "full_cross_matched"
        )

    available_vectors = list(VECTOR_ORDER)

    out_dir = os.path.join(args.plots_dir, model_short, "matched_diffs")

    for layer in args.layers:
        print(f"\nLayer {layer}:")

        old_diffs = build_old_pool_diffs(
            args.matched_dir, model_short, layer, available_vectors,
        )
        new_diffs = build_new_pool_diffs(
            args.matched_dir, model_short, layer, available_vectors,
        )

        columns = []
        col_labels = []
        old_col_count = 0

        for ds in OLD_ENTITIES:
            if ds in old_diffs:
                columns.append(old_diffs[ds])
                col_labels.append(DATASET_LABELS.get(ds, ds))
                old_col_count += 1

        for ds in NEW_ENTITIES:
            if ds in new_diffs:
                columns.append(new_diffs[ds])
                col_labels.append(DATASET_LABELS.get(ds, ds))

        if not columns:
            print("  No data available, skipping")
            continue

        mat = np.column_stack(columns) if len(columns) > 1 else columns[0].reshape(-1, 1)
        row_labels = [VECTOR_LABELS.get(v, v) for v in available_vectors]

        save_path = os.path.join(out_dir, f"heatmap_layer{layer}.png")
        plot_heatmap(mat, row_labels, col_labels, layer, save_path, old_col_count)

    print(f"\nAll heatmaps saved to {out_dir}")


if __name__ == "__main__":
    main()
