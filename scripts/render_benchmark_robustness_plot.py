import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("docs/examples/generated_plots/benchmark_run_2026-04-14")
SUMMARY_JSON = OUTDIR / "benchmark_summary.json"


def main():
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    methods = summary["methods"]

    labels = [row["label"] for row in methods]
    mean = np.array([row["heldout_mean_ratio"] for row in methods])
    std = np.array([row["heldout_std_ratio"] for row in methods])
    calls = np.array([row["objective_calls"] for row in methods])
    robust = mean - 0.5 * std

    colors = {
        "Gradient (autodiff)": "#0E7490",
        "Powell (derivative-free)": "#C2410C",
        "Random search": "#6D28D9",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for label, x, y, c in zip(labels, std, mean, calls):
        axes[0].scatter(x, y, s=150, color=colors[label], alpha=0.95)
        axes[0].text(x + 0.003, y + 0.004, f"{label}\n{c} calls", fontsize=9, va="bottom")

    axes[0].set_xlabel("Held-out std / no-taper mean")
    axes[0].set_ylabel("Held-out mean / no-taper mean")
    axes[0].set_title("Mean-variance tradeoff")

    x = np.arange(len(labels))
    bars = axes[1].bar(x, robust, color=[colors[label] for label in labels], alpha=0.92)
    axes[1].set_xticks(x, labels, rotation=10)
    axes[1].set_ylabel("Robust score = mean - 0.5 x std")
    axes[1].set_title("Robust held-out score")

    for bar, value in zip(bars, robust):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")

    fig.tight_layout()
    outpath = OUTDIR / "benchmark_robustness_tradeoff.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {outpath}")


if __name__ == "__main__":
    main()
