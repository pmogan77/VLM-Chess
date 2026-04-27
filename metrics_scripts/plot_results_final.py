import os
import matplotlib.pyplot as plt

folders = [
    "gold_moves",
    "gold_moves_fen",
    "plain_moves",
    "plain_moves_fen",
    "random_moves",
    "random_moves_fen",
    "model_moves",
    "model_moves_fen",
    "text_only"
]

metrics = {
    "halluc%": [36.71, 32.63, 33.89, 33.93, 30.62, 32.04, 37.40, 31.45, 21.69],
    "R@k":     [0.1573, 0.1908, 0.1847, 0.2070, 0.2153, 0.2207, 0.1757, 0.2008, 0.1603],
    "P@k":     [0.2273, 0.2682, 0.2548, 0.2782, 0.2969, 0.3105, 0.2636, 0.2784, 0.3492],
    "Hit@k":   [0.4650, 0.6000, 0.5900, 0.5900, 0.5850, 0.6500, 0.5500, 0.6000, 0.5200],
    "Top1":    [0.1800, 0.2250, 0.2300, 0.2500, 0.2200, 0.2350, 0.2350, 0.2300, 0.2200],
    "MRR":     [0.2837, 0.3601, 0.3605, 0.3670, 0.3551, 0.3851, 0.3473, 0.3638, 0.3356],
    "avg_cp_gap": [7135.32, 7869.41, 8668.93, 10556.72, 8222.42, 9248.75, 9695.93, 10455.27, 10488.01],
    "secs": [141.5, 544.6, 169.8, 169.6, 165.9, 630.9, 144.3, 176.6, 124.3]
}

output_dir = "./results/final"
os.makedirs(output_dir, exist_ok=True)


def format_label(v):
    if v < 1:
        return f"{v:.4f}"
    elif v < 100:
        return f"{v:.2f}"
    else:
        return f"{v:.1f}"


def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            format_label(height),
            ha="center",
            va="bottom",
            fontsize=8
        )


# -----------------------------
# 1. Individual metric charts
# -----------------------------
for metric_name, values in metrics.items():
    plt.figure(figsize=(10, 5))

    bars = plt.bar(folders, values)

    plt.title(metric_name)
    plt.xlabel("Folder")
    plt.ylabel(metric_name)
    plt.xticks(rotation=25)

    add_bar_labels(bars)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name}.png"), dpi=200)
    plt.close()


# -----------------------------
# 2. FEN vs NON-FEN comparison
# -----------------------------
bases = ["gold", "plain", "random", "model"]

for metric in ["Hit@k", "Top1", "MRR", "halluc%"]:
    plt.figure(figsize=(8, 5))

    normal_vals = []
    fen_vals = []

    for b in bases:
        normal_idx = folders.index(f"{b}_moves")
        fen_idx = folders.index(f"{b}_moves_fen")

        normal_vals.append(metrics[metric][normal_idx])
        fen_vals.append(metrics[metric][fen_idx])

    x = range(len(bases))
    width = 0.35

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        normal_vals,
        width,
        label="image"
    )

    bars2 = plt.bar(
        [i + width / 2 for i in x],
        fen_vals,
        width,
        label="image+fen"
    )

    plt.xticks(x, bases)
    plt.title(f"{metric}: FEN vs Non-FEN")
    plt.ylabel(metric)
    plt.legend()

    add_bar_labels(bars1)
    add_bar_labels(bars2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_fen_vs_nonfen.png"), dpi=200)
    plt.close()


# -----------------------------
# 3. Quality comparison core metrics
# -----------------------------
core_metrics = ["Hit@k", "Top1", "MRR"]

for metric in core_metrics:
    plt.figure(figsize=(10, 5))

    bars = plt.bar(folders, metrics[metric])

    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=25)

    add_bar_labels(bars)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=200)
    plt.close()


print(f"Saved graphs to: {output_dir}")