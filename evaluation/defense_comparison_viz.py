import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load and merge both CSVs ───────────────────────────────────────
single = pd.read_csv("evaluation/experiment_results_with_single_defense.csv")
combined = pd.read_csv("evaluation/experiment_results_combined.csv")
df = pd.concat([single, combined], ignore_index=True)

df["result"] = df["result"].str.strip().str.lower()
df["webhook_hit"] = df["webhook_hit"].astype(str).str.strip().str.lower() == "true"
df["output_clean"] = df["output_clean"].astype(str).str.strip().str.lower() == "true"

os.makedirs("defense_figures_combined", exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
MODELS = ["gemini", "llama", "qwen", "openai20b", "openai120b"]
MODEL_LABELS = {
    "gemini": "Gemini 2.5\nFlash Lite",
    "llama": "Llama 3.3\n70B",
    "qwen": "Qwen3\n32B",
    "openai20b": "GPT-OSS\n20B",
    "openai120b": "GPT-OSS\n120B",
}

# Map defense names to display labels matching paper style
SINGLE_DEFENSES = ["none", "delimiting", "marking", "llm_tagging", "instruction"]
SINGLE_LABELS = {
    "none": "No Defense",
    "delimiting": "Delimiting\nData",
    "marking": "Marking",
    "llm_tagging": "LLM Tagging",
    "instruction": "Instruction\nDefense",
}

COMBINED_DEFENSES = [
    "delimiting_llm_tagging",
    "instruction_llm_tagging",
    "marking_llm_tagging",
]
COMBINED_LABELS = {
    "delimiting_llm_tagging": "Delimiting\n+ LLM Tag",
    "instruction_llm_tagging": "Instruction\n+ LLM Tag",
    "marking_llm_tagging": "Marking\n+ LLM Tag",
}

COLOR_WITHOUT = "#5cb85c"  # green — without LLM tagging (matches paper)
COLOR_WITH = "#b39ddb"  # purple — with LLM tagging (matches paper)
COLOR_SINGLE = "#5cb85c"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def success_rate(subset, pipeline):
    total = len(subset)
    if total == 0:
        return 0.0
    # For both pipelines: only exact "success" counts
    return (subset["result"] == "success").sum() / total


def success_rate_both(subset):
    """Average success rate across both pipelines."""
    scam_rate = success_rate(subset[subset["pipeline"] == "scam"], "scam")
    theft_rate = success_rate(subset[subset["pipeline"] == "theft"], "theft")
    return (scam_rate + theft_rate) / 2


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Paper Figure 7 Replica
# Without LLM Tagging vs With LLM Tagging, averaged across pipelines
# ══════════════════════════════════════════════════════════════════
def plot_paper_replica():
    fig, ax = plt.subplots(figsize=(13, 5))
    # ax.set_title(
    #     "Attack Success Rate Against Various Defense Types\n"
    #     "(Averaged Across Scam and Theft Pipelines, All Models)",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    # Single defenses = without LLM tagging
    # Combined defenses = with LLM tagging
    # For "No Defense" both bars are the same baseline
    display_defenses = [
        "none",
        "delimiting",
        "marking",
        "instruction",
    ]
    x_labels = [
        "No Defense",
        "Delimiting\nData",
        "Marking",
        "Instruction\nDefense",
    ]

    bar_width = 0.35
    x = np.arange(len(display_defenses))

    without_vals = []
    with_vals = []

    for defense in display_defenses:
        sub_without = df[df["defense"] == defense]
        rate_without = success_rate_both(sub_without)
        without_vals.append(rate_without)

        # "With LLM Tagging" = paired combination defense
        combo_map = {
            "none": "none",
            "delimiting": "delimiting_llm_tagging",
            "marking": "marking_llm_tagging",
            "instruction": "instruction_llm_tagging",
        }
        paired = combo_map.get(defense, defense)
        sub_with = df[df["defense"] == paired]
        rate_with = success_rate_both(sub_with)
        with_vals.append(rate_with)

    bars_without = ax.bar(
        x - bar_width / 2,
        without_vals,
        bar_width,
        color=COLOR_WITHOUT,
        label="Without LLM Tagging",
        zorder=3,
    )
    bars_with = ax.bar(
        x + bar_width / 2,
        with_vals,
        bar_width,
        color=COLOR_WITH,
        label="With LLM Tagging",
        zorder=3,
    )

    for bars in [bars_without, bars_with]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Defense Type", fontsize=11)
    ax.set_ylabel("Average Attack Success Rate", fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        "defense_figures_combined/fig1_paper_replica.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig1_paper_replica")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Single Defenses: Scam vs Theft side by side
# ══════════════════════════════════════════════════════════════════
def plot_single_defense_by_pipeline():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 2: Single Defense Success Rates by Pipeline",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(SINGLE_DEFENSES))
    bar_width = 0.5

    pipeline_colors = {"scam": "#e74c3c", "theft": "#2980b9"}

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        rates = []
        for defense in SINGLE_DEFENSES:
            sub = df[(df["defense"] == defense) & (df["pipeline"] == pipeline)]
            rates.append(success_rate(sub, pipeline))

        baseline = rates[0]
        bars = ax.bar(
            x, rates, bar_width, color=pipeline_colors[pipeline], alpha=0.8, zorder=3
        )

        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.01,
                f"{rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.axhline(
            baseline,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"Baseline ({baseline:.2f})",
        )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels([SINGLE_LABELS[d] for d in SINGLE_DEFENSES], fontsize=9)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "defense_figures_combined/fig2_single_by_pipeline.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig2_single_by_pipeline")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Combined Defenses: Scam vs Theft side by side
# ══════════════════════════════════════════════════════════════════
def plot_combined_defense_by_pipeline():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 3: Combined Defense (+ LLM Tagging) Success Rates by Pipeline",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(COMBINED_DEFENSES))
    bar_width = 0.5

    pipeline_colors = {"scam": "#e74c3c", "theft": "#2980b9"}

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        # Baseline: no defense
        baseline_sub = df[(df["defense"] == "none") & (df["pipeline"] == pipeline)]
        baseline_rate = success_rate(baseline_sub, pipeline)

        rates = []
        for defense in COMBINED_DEFENSES:
            sub = df[(df["defense"] == defense) & (df["pipeline"] == pipeline)]
            rates.append(success_rate(sub, pipeline))

        bars = ax.bar(
            x, rates, bar_width, color=pipeline_colors[pipeline], alpha=0.6, zorder=3
        )

        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.01,
                f"{rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.axhline(
            baseline_rate,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"No defense baseline ({baseline_rate:.2f})",
        )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels([COMBINED_LABELS[d] for d in COMBINED_DEFENSES], fontsize=9)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "defense_figures_combined/fig3_combined_by_pipeline.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig3_combined_by_pipeline")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Single vs Combined: Direct Comparison per Defense Family
# Shows the LLM Tagging boost effect clearly
# ══════════════════════════════════════════════════════════════════
def plot_single_vs_combined():
    families = [
        ("delimiting", "delimiting_llm_tagging", "Delimiting"),
        ("instruction", "instruction_llm_tagging", "Instruction"),
        ("marking", "marking_llm_tagging", "Marking"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 4: Defense Family Comparison — Single vs Combined with LLM Tagging",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(families))
    bar_width = 0.3
    gap = 0.08
    offsets = [-(bar_width / 2 + gap / 2), bar_width / 2 + gap / 2]

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        single_rates = []
        combined_rates = []

        for single_d, combined_d, _ in families:
            sub_s = df[(df["defense"] == single_d) & (df["pipeline"] == pipeline)]
            sub_c = df[(df["defense"] == combined_d) & (df["pipeline"] == pipeline)]
            single_rates.append(success_rate(sub_s, pipeline))
            combined_rates.append(success_rate(sub_c, pipeline))

        bars_s = ax.bar(
            x + offsets[0],
            single_rates,
            bar_width,
            color=COLOR_WITHOUT,
            label="Single Defense",
            zorder=3,
        )
        bars_c = ax.bar(
            x + offsets[1],
            combined_rates,
            bar_width,
            color=COLOR_WITH,
            label="Combined + LLM Tagging",
            zorder=3,
        )

        for bars in [bars_s, bars_c]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.02:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.01,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f[2] for f in families], fontsize=10)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "defense_figures_combined/fig4_single_vs_combined.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig4_single_vs_combined")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Defense Heatmap: All Defenses × Models
# ══════════════════════════════════════════════════════════════════
def plot_defense_heatmap():
    all_defenses = SINGLE_DEFENSES + COMBINED_DEFENSES
    all_labels = {**SINGLE_LABELS, **COMBINED_LABELS}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # fig.suptitle(
    #     "Figure 5: Attack Success Rate Heatmap — Model × Defense (All Defenses)",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        matrix = np.zeros((len(MODELS), len(all_defenses)))

        for i, model in enumerate(MODELS):
            for j, defense in enumerate(all_defenses):
                sub = df[
                    (df["model"] == model)
                    & (df["defense"] == defense)
                    & (df["pipeline"] == pipeline)
                ]
                matrix[i, j] = success_rate(sub, pipeline)

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

        for i in range(len(MODELS)):
            for j in range(len(all_defenses)):
                val = matrix[i, j]
                color = "white" if val > 0.65 or val < 0.2 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

        ax.set_xticks(range(len(all_defenses)))
        ax.set_xticklabels(
            [all_labels[d] for d in all_defenses], fontsize=7, rotation=30, ha="right"
        )
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(
            [MODEL_LABELS[m].replace("\n", " ") for m in MODELS], fontsize=9
        )

        # Draw a divider between single and combined
        ax.axvline(len(SINGLE_DEFENSES) - 0.5, color="white", linewidth=2)
        ax.text(
            len(SINGLE_DEFENSES) / 2 - 0.5,
            -0.8,
            "Single Defenses",
            ha="center",
            fontsize=8,
            color="gray",
            transform=ax.transData,
        )
        ax.text(
            len(SINGLE_DEFENSES) + len(COMBINED_DEFENSES) / 2 - 0.5,
            -0.8,
            "Combined + LLM Tagging",
            ha="center",
            fontsize=8,
            color="gray",
            transform=ax.transData,
        )

        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline", fontsize=12
        )
        plt.colorbar(im, ax=ax, format=lambda x, _: f"{x:.0%}")

    plt.tight_layout()
    plt.savefig(
        "defense_figures_combined/fig5_heatmap_all_defenses.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig5_heatmap_all_defenses")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Defense Reduction Chart
# How much does each defense reduce attack success vs baseline?
# ══════════════════════════════════════════════════════════════════
def plot_defense_reduction():
    all_defenses = SINGLE_DEFENSES[1:] + COMBINED_DEFENSES  # exclude "none"
    all_labels = {**SINGLE_LABELS, **COMBINED_LABELS}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # fig.suptitle(
    #     "Figure 6: Attack Success Rate Reduction vs No Defense Baseline",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    x = np.arange(len(all_defenses))
    bar_width = 0.5

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        baseline = success_rate(
            df[(df["defense"] == "none") & (df["pipeline"] == pipeline)], pipeline
        )

        reductions = []
        for defense in all_defenses:
            sub = df[(df["defense"] == defense) & (df["pipeline"] == pipeline)]
            rate = success_rate(sub, pipeline)
            reductions.append(baseline - rate)

        colors = ["#27ae60" if r > 0 else "#e74c3c" for r in reductions]
        bars = ax.bar(x, reductions, bar_width, color=colors, zorder=3)

        for bar, red in zip(bars, reductions):
            ypos = red + 0.01 if red >= 0 else red - 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{red:+.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Divider between single and combined
        ax.axvline(
            len(SINGLE_DEFENSES)
            - 1
            - 0.5,  # correctly between last single and first combined
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.text(
            len(SINGLE_DEFENSES[1:]) / 2 - 1,
            max(reductions) + 0.05,
            "Single",
            ha="center",
            fontsize=8,
            color="gray",
        )
        ax.text(
            len(SINGLE_DEFENSES[1:]) + len(COMBINED_DEFENSES) / 2 - 0.5,
            max(reductions) + 0.05,
            "Combined",
            ha="center",
            fontsize=8,
            color="gray",
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{'Scam' if pipeline == 'scam' else 'Theft'} Pipeline\n"
            f"Baseline: {baseline:.0%}",
            fontsize=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [all_labels[d] for d in all_defenses], fontsize=8, rotation=20, ha="right"
        )
        ax.set_ylabel("Success Rate Reduction")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    legend_patches = [
        mpatches.Patch(color="#27ae60", label="Reduced attack success (defense works)"),
        mpatches.Patch(
            color="#e74c3c", label="Increased attack success (defense backfired)"
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
        "defense_figures_combined/fig6_reduction.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig6_reduction")


# ── Run all ────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_paper_replica()
    plot_single_defense_by_pipeline()
    plot_combined_defense_by_pipeline()
    plot_single_vs_combined()
    plot_defense_heatmap()
    plot_defense_reduction()
    print("\nAll figures saved to defense_figures_combined/")
