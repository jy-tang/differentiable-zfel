import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("docs/examples/generated_plots/benchmark_run_2026-04-14")
RAW_JSON = OUTDIR / "benchmark_raw_data.json"


def plot_progress(raw_results, outpath):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {
        "Gradient (autodiff)": "#0E7490",
        "Powell (derivative-free)": "#C2410C",
        "Random search": "#6D28D9",
    }
    histories = {
        "Gradient (autodiff)": raw_results["gradient_history"],
        "Powell (derivative-free)": raw_results["powell_history"],
        "Random search": raw_results["random_history"],
    }

    for label, history in histories.items():
        calls = [item["calls"] for item in history]
        best_so_far = np.maximum.accumulate([item["ratio_vs_no_taper"] for item in history])
        elapsed = [item["elapsed_s"] for item in history]
        axes[0].plot(calls, best_so_far, marker="o", linewidth=2.2, label=label, color=colors[label])
        axes[1].plot(elapsed, best_so_far, marker="o", linewidth=2.2, label=label, color=colors[label])

    axes[0].set_xlabel("FEL objective evaluations")
    axes[0].set_ylabel("Best pulse-energy ratio vs no taper")
    axes[0].set_title("Optimization progress by evaluation count")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Best pulse-energy ratio vs no taper")
    axes[1].set_title("Optimization progress by wall time")
    axes[0].legend(frameon=True)
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_profiles_and_power(raw_results, outpath):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {
        "No taper": "#5B6C8F",
        "Reference taper": "#D97706",
        "Gradient (autodiff)": "#0E7490",
        "Powell (derivative-free)": "#C2410C",
        "Random search": "#6D28D9",
    }

    for label, curve in raw_results["power_curves"].items():
        axes[0].plot(curve["z"], curve["k_profile"], linewidth=2.2, label=label, color=colors[label])
        axes[1].plot(curve["z"], curve["power_z_gw"], linewidth=2.2, label=label, color=colors[label])

    axes[0].set_xlabel("Undulator position z (m)")
    axes[0].set_ylabel("Undulator parameter K")
    axes[0].set_title("Best taper profiles")
    axes[1].set_xlabel("Undulator position z (m)")
    axes[1].set_ylabel("Power (GW)")
    axes[1].set_yscale("log")
    axes[1].set_title("Power growth along the undulator")
    axes[0].legend(frameon=True, fontsize=9)
    axes[1].legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_stochastic_validation(raw_results, outpath):
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = [row["label"] for row in raw_results["validation_rows"]]
    means = np.array([row["mean_ratio"] for row in raw_results["validation_rows"]])
    stds = np.array([row["std_ratio"] for row in raw_results["validation_rows"]])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=["#5B6C8F", "#D97706", "#0E7490", "#C2410C", "#6D28D9"],
        alpha=0.9,
    )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("Held-out mean pulse energy / no-taper mean")
    ax.set_title("Held-out stochastic validation over explicit shot-noise batch")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03, f"{mean:.2f}x", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    raw_results = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    plot_progress(raw_results, OUTDIR / "benchmark_progress.png")
    plot_profiles_and_power(raw_results, OUTDIR / "benchmark_profiles_power.png")
    plot_stochastic_validation(raw_results, OUTDIR / "benchmark_stochastic_validation.png")
    print("Saved plots:")
    print(f"  {OUTDIR / 'benchmark_progress.png'}")
    print(f"  {OUTDIR / 'benchmark_profiles_power.png'}")
    print(f"  {OUTDIR / 'benchmark_stochastic_validation.png'}")


if __name__ == "__main__":
    main()
