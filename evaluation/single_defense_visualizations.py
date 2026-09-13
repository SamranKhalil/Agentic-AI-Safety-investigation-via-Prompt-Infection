import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv("evaluation/experiment_results_with_single_defense.csv")
df["result"] = df["result"].str.strip().str.lower()
df["output_clean"] = df["output_clean"].astype(str).str.strip().str.lower() == "true"
df["webhook_hit"] = df["webhook_hit"].astype(str).str.strip().str.lower() == "true"

os.makedirs("defense_figures", exist_ok=True)

# ── Shared config ──────────────────────────────────────────────────
DEFENSES = ["none", "instruction", "marking", "llm_tagging", "delimiting"]
DEFENSE_LABELS = {
    "none": "No Defense",
    "instruction": "Instruction",
    "marking": "Marking",
    "llm_tagging": "LLM Tagging",
    "delimiting": "Delimiting",
}
MODELS = ["gemini", "llama", "qwen", "openai20b", "openai120b"]
MODEL_LABELS = {
    "gemini": "Gemini 2.5\nFlash Lite",
    "llama": "Llama 3.3\n70B",
    "qwen": "Qwen3\n32B",
    "openai20b": "GPT-OSS\n20B",
    "openai120b": "GPT-OSS\n120B",
}
COLORS = {
    "success": "#2ecc71",
    "failure": "#e74c3c",
}
DEFENSE_COLORS = ["#2c3e50", "#2980b9", "#e67e22", "#27ae60", "#8e44ad"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def compute_success_rate(subset, pipeline):
    total = len(subset)
    if total == 0:
        return 0.0
    return (subset["result"] == "success").sum() / total


def add_bar_labels(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Defense Success Rates Overview
# Each defense vs no defense, averaged across all models and modes
# Side by side for scam and theft
# ══════════════════════════════════════════════════════════════════
def plot_defense_overview():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 1: Attack Success Rate by Defense Strategy",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(DEFENSES))
    bar_width = 0.5

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        rates = []
        for defense in DEFENSES:
            sub = df[(df["defense"] == defense) & (df["pipeline"] == pipeline)]
            rates.append(compute_success_rate(sub, pipeline))

        bars = ax.bar(x, rates, bar_width, color=DEFENSE_COLORS, zorder=3)

        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.01,
                f"{rate:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Baseline reference line (no defense)
        baseline = rates[0]
        ax.axhline(
            baseline,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"No defense baseline ({baseline:.0%})",
        )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DEFENSE_LABELS[d] for d in DEFENSES], fontsize=9)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        "defense_figures/fig1_defense_overview.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: fig1_defense_overview")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Defense Effectiveness by Model
# For each model, how does each defense reduce attack success?
# ══════════════════════════════════════════════════════════════════
def plot_defense_by_model():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    # fig.suptitle(
    #     "Figure 2: Defense Effectiveness by Model",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(MODELS))
    bar_width = 0.15
    offsets = np.linspace(
        -(len(DEFENSES) - 1) * bar_width / 2,
        (len(DEFENSES) - 1) * bar_width / 2,
        len(DEFENSES),
    )

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        for j, (defense, color, offset) in enumerate(
            zip(DEFENSES, DEFENSE_COLORS, offsets)
        ):
            rates = []
            for model in MODELS:
                sub = df[
                    (df["model"] == model)
                    & (df["defense"] == defense)
                    & (df["pipeline"] == pipeline)
                ]
                rates.append(compute_success_rate(sub, pipeline))

            ax.bar(
                x + offset,
                rates,
                bar_width,
                color=color,
                label=DEFENSE_LABELS[defense],
                zorder=3,
            )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(
        "defense_figures/fig2_defense_by_model.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: fig2_defense_by_model")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Defense Effectiveness: Global vs Local Messaging
# ══════════════════════════════════════════════════════════════════
def plot_defense_by_mode():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 3: Defense Effectiveness — Global vs Local Messaging",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(DEFENSES))
    bar_width = 0.3
    gap = 0.08
    offsets = [-(bar_width / 2 + gap / 2), bar_width / 2 + gap / 2]
    modes = ["global", "local"]
    mode_colors = {"global": "#2471a3", "local": "#76b7d4"}

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        for offset, mode in zip(offsets, modes):
            rates = []
            for defense in DEFENSES:
                sub = df[
                    (df["defense"] == defense)
                    & (df["pipeline"] == pipeline)
                    & (df["messaging_mode"] == mode)
                ]
                rates.append(compute_success_rate(sub, pipeline))

            bars = ax.bar(
                x + offset,
                rates,
                bar_width,
                color=mode_colors[mode],
                label=f"{mode.capitalize()} messaging",
                zorder=3,
            )

            for bar, rate in zip(bars, rates):
                if rate > 0.02:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        rate + 0.01,
                        f"{rate:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DEFENSE_LABELS[d] for d in DEFENSES], fontsize=9)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "defense_figures/fig3_defense_by_mode.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: fig3_defense_by_mode")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Attack Success Rate Reduction vs No Defense (Scam)
# Shows how much each defense reduces success compared to baseline
# ══════════════════════════════════════════════════════════════════
def plot_defense_reduction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    # fig.suptitle(
    #     "Figure 4: Attack Success Rate Reduction vs No Defense Baseline",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(DEFENSES[1:]))  # exclude "none"
    bar_width = 0.5

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        baseline = compute_success_rate(
            df[(df["defense"] == "none") & (df["pipeline"] == pipeline)], pipeline
        )

        reductions = []
        for defense in DEFENSES[1:]:
            sub = df[(df["defense"] == defense) & (df["pipeline"] == pipeline)]
            rate = compute_success_rate(sub, pipeline)
            reductions.append(baseline - rate)

        colors = [COLORS["success"] if r > 0 else COLORS["failure"] for r in reductions]

        bars = ax.bar(x, reductions, bar_width, color=colors, zorder=3)

        for bar, red in zip(bars, reductions):
            ypos = red + 0.005 if red >= 0 else red - 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{red:+.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline\n"
            f"(Baseline: {baseline:.0%})",
            fontsize=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DEFENSE_LABELS[d] for d in DEFENSES[1:]], fontsize=9)
        ax.set_ylabel("Success Rate Reduction")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    legend_patches = [
        mpatches.Patch(
            color=COLORS["success"], label="Reduced attack success (defense works)"
        ),
        mpatches.Patch(
            color=COLORS["failure"],
            label="Increased attack success (defense backfired)",
        ),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.savefig(
        "defense_figures/fig4_defense_reduction.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: fig4_defense_reduction")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Per-Model Defense Heatmap
# Heatmap of success rate for each model × defense combination
# ══════════════════════════════════════════════════════════════════
def plot_defense_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # fig.suptitle(
    #     "Figure 5: Attack Success Rate Heatmap — Model × Defense",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        matrix = np.zeros((len(MODELS), len(DEFENSES)))

        for i, model in enumerate(MODELS):
            for j, defense in enumerate(DEFENSES):
                sub = df[
                    (df["model"] == model)
                    & (df["defense"] == defense)
                    & (df["pipeline"] == pipeline)
                ]
                matrix[i, j] = compute_success_rate(sub, pipeline)

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

        # Annotate cells
        for i in range(len(MODELS)):
            for j in range(len(DEFENSES)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i,j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white" if matrix[i, j] > 0.6 or matrix[i, j] < 0.2 else "black"
                    ),
                )

        ax.set_xticks(range(len(DEFENSES)))
        ax.set_xticklabels(
            [DEFENSE_LABELS[d] for d in DEFENSES], fontsize=9, rotation=20, ha="right"
        )
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(
            [MODEL_LABELS[m].replace("\n", " ") for m in MODELS], fontsize=9
        )
        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        plt.colorbar(im, ax=ax, format=lambda x, _: f"{x:.0%}")

    plt.tight_layout()
    plt.savefig(
        "defense_figures/fig5_defense_heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: fig5_defense_heatmap")


# ── Run all ────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_defense_overview()
    plot_defense_by_model()
    plot_defense_by_mode()
    plot_defense_reduction()
    plot_defense_heatmap()
    print("\nAll defense figures saved to defense_figures/")
