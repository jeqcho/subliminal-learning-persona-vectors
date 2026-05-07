"""Regenerate paper figure with two-level cluster bootstrap + BCa 95% CI.

Produces (drops Random condition; ylim 35%):
- plots/all_animals/paper/sl_pvp_dataset_selection_bar_avg_se.png

Run from repo root: uv run python -m all_animals.plot_paper_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scstats

from all_animals.plot_results import collect_results
from all_animals.config import ANIMALS, STRATEGIES, SEEDS


def bca(data: dict, n_resamples: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    animals = list(data.keys())
    n = len(animals)
    obs = float(np.mean([np.mean(data[a]) for a in animals]))
    bs = np.empty(n_resamples)
    for r in range(n_resamples):
        ai = rng.integers(0, n, n)
        ms = []
        for i in ai:
            a = animals[i]
            ns = len(data[a])
            si = rng.integers(0, ns, ns)
            ms.append(np.mean([data[a][k] for k in si]))
        bs[r] = np.mean(ms)
    p = (bs < obs).mean()
    z0 = scstats.norm.ppf(p) if 0 < p < 1 else 0.0
    jack = np.array([
        np.mean([np.mean(data[animals[i]]) for i in range(n) if i != j])
        for j in range(n)
    ])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = num / den if den > 0 else 0.0
    zlo, zhi = scstats.norm.ppf(0.025), scstats.norm.ppf(0.975)
    plo = scstats.norm.cdf(z0 + (z0 + zlo) / (1 - acc * (z0 + zlo)))
    phi = scstats.norm.cdf(z0 + (z0 + zhi) / (1 - acc * (z0 + zhi)))
    return obs, float(np.percentile(bs, 100 * plo)), float(np.percentile(bs, 100 * phi))


def main() -> None:
    results = collect_results()
    keys = ["base", "clean", "bottom_proj", "top_proj"]
    data = {k: {} for k in keys}
    for animal in ANIMALS:
        for strategy in STRATEGIES:
            sd = results.get(animal, {}).get(strategy, {})
            if strategy in ("clean", "bottom_proj", "top_proj"):
                seeds: list[float] = []
                for seed in SEEDS:
                    if seed not in sd:
                        continue
                    rows = sorted(sd[seed], key=lambda r: r["step"])
                    if not rows:
                        continue
                    last = rows[-1]
                    n_ = int(last.get("total_responses", 5000))
                    k_ = int(last.get("target_count", round(last["target_animal_rate"] * n_)))
                    seeds.append(k_ / n_)
                if seeds:
                    data[strategy][animal] = seeds
            if strategy == STRATEGIES[0]:
                seeds = []
                for seed in SEEDS:
                    if seed not in sd:
                        continue
                    rows = sorted(sd[seed], key=lambda r: r["step"])
                    if rows and rows[0]["step"] == 0:
                        f = rows[0]
                        n0 = int(f.get("total_responses", 5000))
                        k0 = int(f.get("target_count", round(f["target_animal_rate"] * n0)))
                        seeds.append(k0 / n0)
                if seeds:
                    data["base"][animal] = seeds

    LABELS = ["Base", "Clean", "Bottom PVP", "Top PVP"]
    COLORS = ["#BFBFBF", "#7F7F7F", "#228833", "#EE6677"]
    means, lo, hi = [], [], []
    for k in keys:
        m, l, h = bca(data[k])
        means.append(m * 100); lo.append(l * 100); hi.append(h * 100)
        print(f"{k:12s}: mean={m*100:5.2f}  95% BCa CI=[{l*100:5.2f}, {h*100:5.2f}]")

    x = np.arange(len(LABELS))
    yerr = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
    bars = ax.bar(x, means, 0.6, yerr=yerr, capsize=4, color=COLORS,
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Target Animal Rate (%)", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=22, rotation=30, ha="right")
    ax.tick_params(labelsize=22)
    ylim_max = 35
    ax.set_ylim(0, ylim_max)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, m, h in zip(bars, means, hi):
        ax.text(bar.get_x() + bar.get_width() / 2, h + ylim_max * 0.02,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=18)

    out = Path("plots/all_animals/paper/sl_pvp_dataset_selection_bar_avg_se.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
