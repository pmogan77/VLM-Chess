#!/usr/bin/env python3
"""
Generate presentation-ready charts from batch_eval.py results.

Outputs (to results/):
  fig1_main_metrics.png     — grouped bar chart: hallucination, hit@5, top-1 hit, recall@5, MRR
  fig2_cp_gap.png           — median CP gap bar chart (outlier-robust)
  fig3_cp_gap_dist.png      — per-puzzle CP gap distributions as box plots
  fig4_metric_heatmap.png   — normalised heatmap of all metrics

Usage:
  python plot_results.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SUMMARY_PATH   = Path("results/eval_summary.json")
PER_PUZZLE_PATH = Path("results/eval_per_puzzle.json")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── colour palette (presentation-friendly) ────────────────────────────────
MATE_SCORE = 100_000

COLORS = {
    "model_moves":  "#2e7d32",   # dark green  — experimental
    "plain_moves":  "#1565c0",   # dark blue   — vision baseline
    "gold_moves":   "#f9a825",   # amber       — oracle upper bound
    "random_moves": "#6a1b9a",   # purple      — ablation
    "text_only":    "#c62828",   # red         — text baseline
}
LABELS = {
    "model_moves":  "VLM-Annotated",
    "plain_moves":  "Plain Board",
    "gold_moves":   "Gold Annotated",
    "random_moves": "Random (Ablation)",
    "text_only":    "Text-Only",
}

CONDITIONS = list(COLORS.keys())

with open(SUMMARY_PATH) as f:
    summary = json.load(f)
with open(PER_PUZZLE_PATH) as f:
    per_puzzle = json.load(f)


# ── helpers ───────────────────────────────────────────────────────────────

def save(fig, name: str):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


# =========================================================================
# Fig 1 — Grouped bar chart: quality metrics (higher = better)
#          hallucination is inverted (lower = better → show legality rate)
# =========================================================================

metrics_cfg = [
    ("legality_rate",  "Legality Rate\n(1 − hallucination)",  True),
    ("hit_at_k_mean",  "Hit@5",                               True),
    ("top1_hit_rate",  "Top-1 Hit Rate",                      True),
    ("recall_at_k_mean","Recall@5",                           True),
    ("mrr_mean",       "MRR",                                 True),
]

# derive legality_rate from per-puzzle data
for cond, rows in per_puzzle.items():
    rates = []
    for r in rows:
        total = len(r["candidates_raw"])
        illegal = len(r["candidates_illegal"])
        rates.append(1 - illegal / total if total else 1.0)
    summary[cond]["legality_rate"] = statistics.mean(rates)

fig, ax = plt.subplots(figsize=(12, 5))
n_metrics = len(metrics_cfg)
n_conds   = len(CONDITIONS)
bar_w     = 0.14
group_gap = 0.05
x = np.arange(n_metrics)

for i, cond in enumerate(CONDITIONS):
    vals = []
    for key, _, _ in metrics_cfg:
        vals.append(summary[cond].get(key, 0) or 0)
    offset = (i - n_conds / 2 + 0.5) * (bar_w + group_gap / n_conds)
    bars = ax.bar(x + offset, vals, bar_w,
                  color=COLORS[cond], label=LABELS[cond], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([m[1] for m in metrics_cfg], fontsize=10)
ax.set_ylabel("Score (higher = better)", fontsize=11)
ax.set_title("Performance Across Conditions — Quality Metrics (k=5, depth=12)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 1.05)
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig1_main_metrics.png")


# =========================================================================
# Fig 2 — Median CP gap bar chart (outlier-robust alternative to mean)
# =========================================================================

fig, ax = plt.subplots(figsize=(8, 4.5))

medians, colors_list, labels_list = [], [], []
for cond in CONDITIONS:
    gaps = [r["cp_gap"] for r in per_puzzle[cond]
            if r["cp_gap"] is not None and abs(r["cp_gap"]) < MATE_SCORE * 0.5]
    medians.append(statistics.median(gaps) if gaps else 0)
    colors_list.append(COLORS[cond])
    labels_list.append(LABELS[cond])

bars = ax.bar(labels_list, medians, color=colors_list, alpha=0.9, edgecolor="white")
for bar, val in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Median Centipawn Gap\n(engine_best − candidate_best)", fontsize=10)
ax.set_title("Centipawn Gap vs Stockfish Best Move\n(lower = closer to engine quality; outliers >50k cp excluded)",
             fontsize=12, fontweight="bold", pad=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.xticks(fontsize=9)
fig.tight_layout()
save(fig, "fig2_cp_gap_median.png")


# =========================================================================
# Fig 3 — Box plot of per-puzzle CP gap distributions
# =========================================================================

fig, ax = plt.subplots(figsize=(9, 5))

data_per_cond, tick_labels, box_colors = [], [], []
for cond in CONDITIONS:
    gaps = [r["cp_gap"] for r in per_puzzle[cond]
            if r["cp_gap"] is not None and abs(r["cp_gap"]) < MATE_SCORE * 0.5]
    data_per_cond.append(gaps)
    tick_labels.append(LABELS[cond])
    box_colors.append(COLORS[cond])

bp = ax.boxplot(data_per_cond, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for element in ["whiskers", "caps", "fliers"]:
    for item in bp[element]:
        item.set(color="gray", linewidth=1)

ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_ylabel("Centipawn Gap (engine − candidate)", fontsize=10)
ax.set_title("Distribution of Per-Puzzle Centipawn Gap\n(outliers >50k cp excluded; lower & tighter = better)",
             fontsize=12, fontweight="bold", pad=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
fig.tight_layout()
save(fig, "fig3_cp_gap_dist.png")


# =========================================================================
# Fig 4 — Normalised metric heatmap
# =========================================================================

metric_keys = [
    ("legality_rate",   "Legality Rate"),
    ("recall_at_k_mean","Recall@5"),
    ("hit_at_k_mean",   "Hit@5"),
    ("top1_hit_rate",   "Top-1 Hit"),
    ("mrr_mean",        "MRR"),
]

matrix = np.array([
    [summary[cond].get(k, 0) or 0 for cond in CONDITIONS]
    for k, _ in metric_keys
])

# Normalise each row to [0,1] across conditions so colour reflects relative ranking
normed = np.zeros_like(matrix)
for i in range(len(metric_keys)):
    row = matrix[i]
    lo, hi = row.min(), row.max()
    normed[i] = (row - lo) / (hi - lo) if hi > lo else np.zeros_like(row)

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(normed, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=10)
ax.set_yticks(range(len(metric_keys)))
ax.set_yticklabels([label for _, label in metric_keys], fontsize=10)

for i in range(len(metric_keys)):
    for j in range(len(CONDITIONS)):
        raw_val = matrix[i, j]
        ax.text(j, i, f"{raw_val:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="black" if 0.3 < normed[i, j] < 0.8 else "white")

ax.set_title("Metric Heatmap — Raw Values, Colour = Relative Rank within Metric\n(green = best, red = worst)",
             fontsize=11, fontweight="bold", pad=10)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalised rank (0=worst, 1=best)")
fig.tight_layout()
save(fig, "fig4_metric_heatmap.png")

print("\nAll figures saved to results/")
