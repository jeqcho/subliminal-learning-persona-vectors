"""
Per-sample projection differences: entity vs neutral.

Matches entity and neutral samples by exact user prompt text, computes
``diff = entity_proj - neutral_proj`` at each layer, and produces:

  1. Per-trait diff histogram grids (one subplot per layer)
  2. Summary grid (all animals side by side, mean diff +/- SE vs layer)
  3. Mean diff overlay (all animals on one plot)
  4. Stats CSV per trait

Usage:
    uv run python plot_projection_diffs.py --model unsloth/Qwen2.5-14B-Instruct
"""

import json
import math
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

ANIMAL_CONFIG = {
    "liking_eagles": {
        "animal": "eagle",
        "display": "Eagles",
        "vector_stem": "liking_eagles_response_avg_diff",
    },
    "liking_lions": {
        "animal": "lion",
        "display": "Lions",
        "vector_stem": "liking_lions_response_avg_diff",
    },
    "liking_phoenixes": {
        "animal": "phoenix",
        "display": "Phoenixes",
        "vector_stem": "liking_phoenixes_response_avg_diff",
    },
}

ANIMAL_COLORS = {
    "liking_eagles": "#D62728",
    "liking_lions": "#1F77B4",
    "liking_phoenixes": "#2CA02C",
}

ANIMAL_MARKERS = {
    "liking_eagles": "o",
    "liking_lions": "s",
    "liking_phoenixes": "D",
}


def _key_prefix(vector_stem: str, model_short: str) -> str:
    return f"{model_short}_{vector_stem}_proj_layer"


def get_prompt(sample: dict) -> str:
    """Extract the user prompt text from a messages-format sample."""
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def load_projections_by_prompt(
    path: str, key_prefix: str, layers: list[int]
) -> dict[str, dict[int, float]]:
    """Load JSONL and return ``{prompt_text: {layer: projection_value}}``."""
    result: dict[str, dict[int, float]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = get_prompt(d)
            layer_vals = {}
            for layer in layers:
                key = f"{key_prefix}{layer}"
                if key in d:
                    v = d[key]
                    if v is not None and np.isfinite(v):
                        layer_vals[layer] = v
            if layer_vals:
                result[prompt] = layer_vals
    return result


def compute_diffs(
    entity_data: dict[str, dict[int, float]],
    neutral_data: dict[str, dict[int, float]],
    layers: list[int],
) -> dict[int, np.ndarray]:
    """Return per-layer arrays of ``entity_proj - neutral_proj`` for matched prompts."""
    common = sorted(set(entity_data) & set(neutral_data))
    diffs_by_layer: dict[int, list[float]] = {l: [] for l in layers}
    for prompt in common:
        e = entity_data[prompt]
        n = neutral_data[prompt]
        for layer in layers:
            if layer in e and layer in n:
                diffs_by_layer[layer].append(e[layer] - n[layer])
    return {l: np.array(v) for l, v in diffs_by_layer.items() if v}


def save_stats_csv(diffs: dict[int, np.ndarray], layers: list[int], path: str):
    """Write per-layer summary statistics to CSV."""
    rows = []
    for layer in layers:
        if layer not in diffs or len(diffs[layer]) == 0:
            continue
        d = diffs[layer]
        rows.append({
            "layer": layer,
            "n": len(d),
            "mean": d.mean(),
            "std": d.std(),
            "median": np.median(d),
            "p5": np.percentile(d, 5),
            "p95": np.percentile(d, 95),
            "frac_positive": float((d > 0).mean()),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Stats CSV saved: {path}")
    print(df.to_string(index=False))


def plot_diff_histograms(
    diffs: dict[int, np.ndarray],
    layers: list[int],
    title: str,
    output_path: str,
):
    """Grid of per-layer diff histograms with stats overlay."""
    present = [l for l in layers if l in diffs and len(diffs[l]) > 0]
    if not present:
        print(f"  SKIP (no data): {output_path}")
        return

    n_layers = len(present)
    ncols = min(5, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), facecolor="white"
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, layer in enumerate(present):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.set_facecolor("white")
        arr = diffs[layer]

        p1, p99 = np.percentile(arr, [1, 99])
        clipped = arr[(arr >= p1) & (arr <= p99)]
        display = clipped if len(clipped) > 10 else arr

        ax.hist(display, bins=80, color="#4361ee", alpha=0.75, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

        med = float(np.median(arr))
        ax.axvline(
            med, color="#e63946", linewidth=1.2, linestyle="--",
            label=f"median={med:.1f}",
        )

        frac_pos = float((arr > 0).mean()) * 100
        stats_text = (
            f"N={len(arr):,}\n"
            f"med={med:.1f}\n"
            f"mean={arr.mean():.1f}\n"
            f">{0}:  {frac_pos:.0f}%"
        )
        ax.text(
            0.97, 0.95, stats_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold", color="#333333")
        ax.tick_params(labelsize=9, colors="#333333")
        ax.grid(True, alpha=0.15, linestyle="--", color="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.8)

    for idx in range(n_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#333333", y=1.01)
    fig.supxlabel(
        "Per-sample projection difference (entity - neutral)", fontsize=12,
        color="#333333",
    )
    fig.supylabel("Count", fontsize=12, color="#333333")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Histogram grid saved: {output_path}")


def plot_mean_diff_overlay(
    all_diffs: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    traits: list[str],
    save_path: str,
    model_display: str = "",
):
    """All animals on one plot: mean diff +/- SE vs layer."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")

    for trait in traits:
        cfg = ANIMAL_CONFIG[trait]
        diffs = all_diffs[trait]
        color = ANIMAL_COLORS.get(trait, "#7F7F7F")
        marker = ANIMAL_MARKERS.get(trait, "o")

        means, ses = [], []
        for l in layers:
            d = diffs.get(l, np.array([]))
            if len(d) > 0:
                means.append(d.mean())
                ses.append(d.std() / np.sqrt(len(d)) if len(d) > 1 else 0.0)
            else:
                means.append(np.nan)
                ses.append(0.0)

        means = np.array(means)
        ses = np.array(ses)

        ax.plot(
            layers, means, marker=marker, linewidth=2.5, markersize=8,
            color=color, label=cfg["display"], alpha=0.9,
        )
        ax.fill_between(
            layers, means - ses, means + ses, alpha=0.15, color=color,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer", fontsize=16, fontweight="bold", color="#333333")
    ax.set_ylabel("Mean Projection Diff (entity - neutral)", fontsize=16,
                   fontweight="bold", color="#333333")
    title_model = f" [{model_display}]" if model_display else ""
    ax.set_title(
        f"Per-Sample Projection Difference by Layer{title_model}",
        fontsize=18, fontweight="bold", color="#333333", pad=20,
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
    ax.tick_params(colors="#333333", labelsize=13)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1)

    ax.legend(fontsize=13, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Mean diff overlay saved: {save_path}")
    plt.close()


def plot_diff_summary_grid(
    all_diffs: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    traits: list[str],
    save_path: str,
):
    """One column per animal: mean diff +/- SE across layers."""
    n = len(traits)
    plt.style.use("default")
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), facecolor="white")
    if n == 1:
        axes = [axes]

    for idx, trait in enumerate(traits):
        ax = axes[idx]
        ax.set_facecolor("white")
        cfg = ANIMAL_CONFIG[trait]
        color = ANIMAL_COLORS.get(trait, "#7F7F7F")
        diffs = all_diffs[trait]

        means, ses = [], []
        for l in layers:
            d = diffs.get(l, np.array([]))
            if len(d) > 0:
                means.append(d.mean())
                ses.append(d.std() / np.sqrt(len(d)) if len(d) > 1 else 0.0)
            else:
                means.append(np.nan)
                ses.append(0.0)

        means = np.array(means)
        ses = np.array(ses)

        ax.plot(
            layers, means, marker="o", linewidth=2.5, markersize=8,
            color=color, alpha=0.9,
        )
        ax.fill_between(
            layers, means - ses, means + ses, alpha=0.15, color=color,
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        ax.set_title(cfg["display"], fontsize=15, fontweight="bold", color="#333333")
        ax.set_xlabel("Layer", fontsize=12, color="#333333")
        if idx == 0:
            ax.set_ylabel("Mean Diff (entity - neutral)", fontsize=12, color="#333333")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
        ax.tick_params(colors="#333333", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(1)

    plt.suptitle(
        "Per-Sample Projection Difference: All Animals",
        fontsize=18, fontweight="bold", color="#333333", y=1.04,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Diff summary grid saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot per-sample projection diffs (entity - neutral)"
    )
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--traits", type=str, nargs="+",
                        default=list(ANIMAL_CONFIG.keys()))
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--proj_dir", type=str, default="../outputs/projections")
    parser.add_argument("--plots_dir", type=str, default="../plots/projections")
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))

    all_diffs: dict[str, dict[int, np.ndarray]] = {}
    available_traits: list[str] = []

    for trait in args.traits:
        cfg = ANIMAL_CONFIG.get(trait)
        if cfg is None:
            print(f"Unknown trait: {trait}, skipping")
            continue

        trait_dir = os.path.join(args.proj_dir, model_short, trait)
        entity_path = os.path.join(trait_dir, f"{cfg['animal']}_numbers.jsonl")
        neutral_path = os.path.join(trait_dir, "neutral_numbers.jsonl")

        if not os.path.exists(entity_path):
            print(f"Missing: {entity_path}, skipping {trait}")
            continue
        if not os.path.exists(neutral_path):
            print(f"Missing: {neutral_path}, skipping {trait}")
            continue

        key_pfx = _key_prefix(cfg["vector_stem"], model_short)

        print(f"\n{'='*60}")
        print(f"Processing {trait} ({cfg['display']})")
        print(f"{'='*60}")

        print(f"Loading entity projections: {entity_path}")
        entity_data = load_projections_by_prompt(entity_path, key_pfx, args.layers)
        print(f"  {len(entity_data):,} unique prompts")

        print(f"Loading neutral projections: {neutral_path}")
        neutral_data = load_projections_by_prompt(neutral_path, key_pfx, args.layers)
        print(f"  {len(neutral_data):,} unique prompts")

        diffs = compute_diffs(entity_data, neutral_data, args.layers)
        if diffs:
            sample_layer = args.layers[0]
            n_matched = len(diffs.get(sample_layer, []))
            print(f"  Matched prompts: {n_matched:,}")
        else:
            print("  WARNING: No matched prompts found, skipping")
            continue

        all_diffs[trait] = diffs
        available_traits.append(trait)

        stats_path = os.path.join(trait_dir, "diff_stats.csv")
        save_stats_csv(diffs, args.layers, stats_path)

        plot_dir = os.path.join(args.plots_dir, model_short, trait)
        plot_diff_histograms(
            diffs, args.layers,
            f"Per-Sample Projection Diff (1st-99th pctile): {cfg['display']}",
            os.path.join(plot_dir, "diff_histograms.png"),
        )

    if len(available_traits) > 1:
        grid_path = os.path.join(args.plots_dir, model_short, "diff_summary_grid.png")
        plot_diff_summary_grid(all_diffs, args.layers, available_traits, grid_path)

    if available_traits:
        overlay_path = os.path.join(args.plots_dir, model_short, "diff_mean_overlay.png")
        plot_mean_diff_overlay(
            all_diffs, args.layers, available_traits, overlay_path,
            model_display=model_short,
        )

    print(f"\nAll done! Processed {len(available_traits)} traits.")


if __name__ == "__main__":
    main()
