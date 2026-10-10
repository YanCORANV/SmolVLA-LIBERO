"""Build GitHub figures from completed, frozen diagnostic and evaluation summaries.

This script reads existing CSV/JSON results only. It never loads a policy or simulator.
Run from the repository root: python scripts/make_portfolio_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "results/language_divergence/pair_level_divergence_measurements.csv"
DEV = ROOT / "results/development_selection/paired_summary.json"
FINAL = ROOT / "results/final_confirmatory_v4/summary.json"
OUT = ROOT / "assets/figures"
BLUE = "#31688e"
ORANGE = "#d47d39"
GRAY = "#64748b"


def load_inputs():
    with DIAGNOSTICS.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    if len(pairs) != 240 or final["completed_rollouts"] != 153:
        raise ValueError("Expected 240 diagnostic pairs and 153 final rollouts")
    if final["states"] != list(range(33, 50)):
        raise ValueError("Final state list differs from frozen v4")
    if sorted(dev["results"]) != ["paired_bc", "paired_flow_consistency"]:
        raise ValueError("Development model list differs from paired evaluation")
    return pairs, dev, final


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def scatter_box(ax, values, xpos, color):
    ax.boxplot(
        values, positions=[xpos], widths=0.38, showfliers=False, patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.22, "edgecolor": color},
        medianprops={"color": color, "linewidth": 2},
        whiskerprops={"color": color}, capprops={"color": color},
    )
    offsets = np.linspace(-0.16, 0.16, len(values))
    ax.scatter(xpos + offsets, values, s=8, color=color, alpha=0.40, zorder=3)


def plot_action(pairs):
    repeat = [float(row["step1_action_diff"]) for row in pairs if row["pair_type"] == "repeat_baseline"]
    paraphrase = [float(row["step1_action_diff"]) for row in pairs if row["pair_type"] == "paraphrase"]
    if len(repeat) != 60 or len(paraphrase) != 180:
        raise ValueError("Unexpected repeat/paraphrase pair counts")
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    scatter_box(ax, repeat, 1, GRAY)
    scatter_box(ax, paraphrase, 2, BLUE)
    ax.axhline(0.05, color=ORANGE, ls="--", lw=1.5, label="Early repeat p99 threshold: 0.05")
    ax.set_xticks([1, 2], ["Same wording repeat\n(n=60)", "Equivalent paraphrase\n(n=180)"])
    ax.set_ylabel("Step-1 action-vector L2 difference")
    ax.set_title("Wording changes the first predicted action")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(bottom=-0.015)
    fig.savefig(OUT / "step1_action_divergence.png", dpi=180)
    plt.close(fig)


def plot_trajectory(pairs):
    paraphrases = [row for row in pairs if row["pair_type"] == "paraphrase"]
    success = [float(row["trajectory_divergence_step"]) for row in paraphrases if row["compare_goal_ever"] == "1"]
    failure = [float(row["trajectory_divergence_step"]) for row in paraphrases if row["compare_goal_ever"] == "0"]
    if len(success) != 51 or len(failure) != 129:
        raise ValueError("Unexpected paraphrase success/failure pair counts")
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    scatter_box(ax, success, 1, BLUE)
    scatter_box(ax, failure, 2, ORANGE)
    ax.set_xticks([1, 2], ["Paraphrase succeeds\n(n=51)", "Paraphrase fails\n(n=129)"])
    ax.set_ylabel("First timestep with >1 cm end-effector separation")
    ax.set_title("Paired trajectories diverge early")
    ax.axhline(28, color=GRAY, ls=":", lw=1.5, label="First 10% of 280-step horizon")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(bottom=0)
    fig.savefig(OUT / "trajectory_divergence_onset.png", dpi=180)
    plt.close(fig)


def plot_generalization(dev, final):
    names = ("original_language", "augmented_language", "paired_bc")
    labels = ("Canonical-only", "Augmented", "Paired BC")
    early = json.loads(
        (ROOT / "results/development_selection/anchored_summary.json")
        .read_text(encoding="utf-8")
    )
    def development(model):
        return dev["results"][model] if model == "paired_bc" else early["results"][model]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True, layout="constrained")
    for ax, metric, title in zip(
        axes, ("O", "P"), ("Canonical instruction", "Pooled unseen paraphrases")
    ):
        dev_values = [development(name)[metric] * 100 for name in names]
        final_values = [final["models"][name]["rates"][metric] * 100 for name in names]
        ax.bar(x - 0.17, dev_values, width=0.33, color=BLUE, label="Development: States 23–32")
        ax.bar(x + 0.17, final_values, width=0.33, color=ORANGE, label="Final: States 33–49")
        for i, (d, f) in enumerate(zip(dev_values, final_values)):
            ax.text(i - 0.17, d + 1.8, f"{d:.0f}", ha="center", fontsize=8)
            ax.text(i + 0.17, f + 1.8, f"{f:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 112)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.17)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Goal-ever success (%)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Development gains did not transfer to final prompts", fontweight="bold")
    fig.savefig(OUT / "development_vs_final.png", dpi=180)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, dev, final = load_inputs()
    plot_action(pairs)
    print("Wrote step1_action_divergence.png", flush=True)
    plot_trajectory(pairs)
    print("Wrote trajectory_divergence_onset.png", flush=True)
    plot_generalization(dev, final)
    print("Wrote development_vs_final.png", flush=True)


if __name__ == "__main__":
    main()
