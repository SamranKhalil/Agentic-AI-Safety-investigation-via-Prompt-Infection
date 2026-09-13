import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv("evaluation/experiment_results_without_defense.csv")
df["result"] = df["result"].str.strip().str.lower()
df["output_clean"] = df["output_clean"].astype(str).str.strip().str.lower() == "true"
df["webhook_hit"] = df["webhook_hit"].astype(str).str.strip().str.lower() == "true"

os.makedirs("figures", exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────
COLORS = {
    "success": "#2ecc71",
    "failure": "#e74c3c",
    "detectable": "#f39c12",  # used in fig 6
    "stealth": "#2ecc71",  # used in fig 6
}

MODEL_LABELS = {
    "gemini": "Gemini 3.1\nFlash Lite",
    "llama": "Llama 3.3\n70B",
    "qwen": "Qwen3\n32B",
    "openai120b": "GPT-OSS\n120B",
    "openai20b": "GPT-OSS\n20B",
}
MODELS = list(MODEL_LABELS.keys())
PIPELINES = {"scam": "Scam", "theft": "Theft"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ── Helper: compute success rate ───────────────────────────────────
# For theft: success = webhook_hit (partial counts as success per paper)
# For scam:  success = result == "success" (partial is NOT success)
def compute_rates(subset, pipeline):
    total = len(subset)
    if total == 0:
        return 0.0, 0.0

    success = (subset["result"] == "success").sum() / total

    failure = 1.0 - success
    return success, failure


def add_bar_labels(ax, bars, fmt="{:.0%}"):
    for bar in bars:
        h = bar.get_height()
        if h > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=8,
            )


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Self-Replicating vs Non-Replicating (RQ1)
# ══════════════════════════════════════════════════════════════════
def plot_rq1():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 1: Self-Replicating vs Non-Replicating Attack Success Rates",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    datasets = ["replicating", "non_replicating"]
    x_labels = ["Self-Replicating", "Non-Replicating"]
    bar_width = 0.3
    x = np.arange(len(datasets))

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        success_vals, failure_vals = [], []

        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["pipeline"] == pipeline)]
            s, f = compute_rates(sub, pipeline)
            success_vals.append(s)
            failure_vals.append(f)

        bars_s = ax.bar(
            x - bar_width / 2,
            success_vals,
            bar_width,
            label="Success",
            color=COLORS["success"],
            zorder=3,
        )
        bars_f = ax.bar(
            x + bar_width / 2,
            failure_vals,
            bar_width,
            label="Failure",
            color=COLORS["failure"],
            zorder=3,
        )

        add_bar_labels(ax, bars_s)
        add_bar_labels(ax, bars_f)

        ax.set_title(f"{PIPELINES[pipeline]} Pipeline", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "figures/fig1_rq1_replicating_vs_nonreplicating.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: fig1")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Success Rates by Model (RQ2)
# ══════════════════════════════════════════════════════════════════
def plot_rq2():
    bar_width = 0.3
    x = np.arange(len(MODELS))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 2: Attack Success Rates by Model (Self-Replicating Dataset)",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        success_vals, failure_vals = [], []

        for model in MODELS:
            sub = df[
                (df["model"] == model)
                & (df["pipeline"] == pipeline)
                & (df["dataset"] == "replicating")
            ]
            s, f = compute_rates(sub, pipeline)
            success_vals.append(s)
            failure_vals.append(f)

        bars_s = ax.bar(
            x - bar_width / 2,
            success_vals,
            bar_width,
            label="Success",
            color=COLORS["success"],
            zorder=3,
        )
        bars_f = ax.bar(
            x + bar_width / 2,
            failure_vals,
            bar_width,
            label="Failure",
            color=COLORS["failure"],
            zorder=3,
        )

        add_bar_labels(ax, bars_s)
        add_bar_labels(ax, bars_f)

        ax.set_title(f"{PIPELINES[pipeline]} Pipeline", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig2_rq2_success_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig2")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Global vs Local Messaging
# ══════════════════════════════════════════════════════════════════
def plot_rq3():
    modes = ["global", "local"]
    bar_width = 0.3
    x = np.arange(len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # fig.suptitle(
    #     "Figure 3: Global vs Local Messaging Attack Success Rates",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01,
    # )

    for ax, pipeline in zip(axes, ["scam", "theft"]):
        success_vals, failure_vals = [], []

        for mode in modes:
            sub = df[
                (df["messaging_mode"] == mode)
                & (df["pipeline"] == pipeline)
                & (df["dataset"] == "replicating")
            ]
            s, f = compute_rates(sub, pipeline)
            success_vals.append(s)
            failure_vals.append(f)

        bars_s = ax.bar(
            x - bar_width / 2,
            success_vals,
            bar_width,
            label="Success",
            color=COLORS["success"],
            zorder=3,
        )
        bars_f = ax.bar(
            x + bar_width / 2,
            failure_vals,
            bar_width,
            label="Failure",
            color=COLORS["failure"],
            zorder=3,
        )

        add_bar_labels(ax, bars_s)
        add_bar_labels(ax, bars_f)

        ax.set_title(f"{PIPELINES[pipeline]} Pipeline", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(["Global Messaging", "Local Messaging"], fontsize=10)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig3_global_vs_local.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig3")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Scam Pipeline Breakdown by Model + Mode
# (partial = NOT success for scam, kept as failure)
# ══════════════════════════════════════════════════════════════════
def plot_scam_breakdown():
    modes = ["global", "local"]
    bar_width = 0.3
    gap = 0.08
    x = np.arange(len(MODELS))

    fig, ax = plt.subplots(figsize=(15, 5))
    # ax.set_title(
    #     "Figure 4: Scam Pipeline — Success vs Failure by Model and Messaging Mode",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    offsets = [-(bar_width / 2 + gap / 2), bar_width / 2 + gap / 2]
    mode_colors = {
        "global": {"success": "#27ae60", "failure": "#c0392b"},
        "local": {"success": "#82e0aa", "failure": "#f1948a"},
    }

    for offset, mode in zip(offsets, modes):
        success_vals, failure_vals = [], []

        for model in MODELS:
            sub = df[
                (df["model"] == model)
                & (df["pipeline"] == "scam")
                & (df["messaging_mode"] == mode)
                & (df["dataset"] == "replicating")
            ]
            s, f = compute_rates(sub, "scam")
            success_vals.append(s)
            failure_vals.append(f)

        bars_s = ax.bar(
            x + offset,
            success_vals,
            bar_width,
            color=mode_colors[mode]["success"],
            label=f"Success ({mode})",
            zorder=3,
        )
        ax.bar(
            x + offset,
            failure_vals,
            bar_width,
            bottom=success_vals,
            color=mode_colors[mode]["failure"],
            label=f"Failure ({mode})",
            zorder=3,
        )

        for i, xi in enumerate(x):
            ax.text(
                xi + offset,
                -0.07,
                mode.capitalize(),
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("Proportion of Runs")
    ax.set_ylim(-0.12, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{max(v,0):.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig("figures/fig4_scam_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig4")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Theft Pipeline Breakdown by Model + Mode
# (partial counts as success per paper — webhook_hit = success)
# ══════════════════════════════════════════════════════════════════
def plot_theft_breakdown():
    modes = ["global", "local"]
    bar_width = 0.3
    gap = 0.08
    x = np.arange(len(MODELS))

    fig, ax = plt.subplots(figsize=(15, 5))
    # ax.set_title(
    #     "Figure 5: Theft Pipeline — Success vs Failure by Model and Messaging Mode\n"
    #     "(Success = Data Exfiltrated, per paper criteria)",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    offsets = [-(bar_width / 2 + gap / 2), bar_width / 2 + gap / 2]
    mode_colors = {
        "global": {"success": "#27ae60", "failure": "#c0392b"},
        "local": {"success": "#82e0aa", "failure": "#f1948a"},
    }

    for offset, mode in zip(offsets, modes):
        success_vals, failure_vals = [], []

        for model in MODELS:
            sub = df[
                (df["model"] == model)
                & (df["pipeline"] == "theft")
                & (df["messaging_mode"] == mode)
                & (df["dataset"] == "replicating")
            ]
            s, f = compute_rates(sub, "theft")
            success_vals.append(s)
            failure_vals.append(f)

        ax.bar(
            x + offset,
            success_vals,
            bar_width,
            color=mode_colors[mode]["success"],
            label=f"Success ({mode})",
            zorder=3,
        )
        ax.bar(
            x + offset,
            failure_vals,
            bar_width,
            bottom=success_vals,
            color=mode_colors[mode]["failure"],
            label=f"Failure ({mode})",
            zorder=3,
        )

        for i, xi in enumerate(x):
            ax.text(
                xi + offset,
                -0.07,
                mode.capitalize(),
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("Proportion of Runs")
    ax.set_ylim(-0.12, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{max(v,0):.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig("figures/fig5_theft_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig5")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Theft Detectability: Stealth vs Detectable Attacks
# Among successful theft runs, what % had clean output (stealth)
# vs dirty output (detectable by user)?
# ══════════════════════════════════════════════════════════════════
def plot_theft_detectability():
    x = np.arange(len(MODELS))
    bar_width = 0.3
    gap = 0.08

    fig, ax = plt.subplots(figsize=(13, 5))
    # ax.set_title(
    #     "Figure 6: Theft Pipeline — Stealth vs Detectable Attacks\n"
    #     "Among Successful Exfiltration Runs (Replicating Dataset)",
    #     fontsize=13,
    #     fontweight="bold",
    # )

    offsets = [-(bar_width / 2 + gap / 2), bar_width / 2 + gap / 2]
    modes = ["global", "local"]
    mode_colors = {
        "global": {"stealth": "#2471a3", "detectable": "#f39c12"},
        "local": {"stealth": "#7fb3d3", "detectable": "#f8c471"},
    }

    for offset, mode in zip(offsets, modes):
        stealth_vals = []
        detectable_vals = []

        for model in MODELS:
            # Only look at successful theft runs (webhook_hit = True)
            sub = df[
                (df["model"] == model)
                & (df["pipeline"] == "theft")
                & (df["messaging_mode"] == mode)
                & (df["dataset"] == "replicating")
                & (df["webhook_hit"] == True)
            ]

            total_success = len(sub)
            if total_success == 0:
                stealth_vals.append(0.0)
                detectable_vals.append(0.0)
            else:
                stealth = sub["output_clean"].sum() / total_success
                detectable = 1.0 - stealth
                stealth_vals.append(stealth)
                detectable_vals.append(detectable)

        bars_st = ax.bar(
            x + offset,
            stealth_vals,
            bar_width,
            color=mode_colors[mode]["stealth"],
            label=f"Stealth — clean output ({mode})",
            zorder=3,
        )
        ax.bar(
            x + offset,
            detectable_vals,
            bar_width,
            bottom=stealth_vals,
            color=mode_colors[mode]["detectable"],
            label=f"Detectable — dirty output ({mode})",
            zorder=3,
        )

        # Annotate total success count above each bar
        for i, xi in enumerate(x):
            sub_count = df[
                (df["model"] == MODELS[i])
                & (df["pipeline"] == "theft")
                & (df["messaging_mode"] == mode)
                & (df["dataset"] == "replicating")
                & (df["webhook_hit"] == True)
            ]
            n = len(sub_count)
            ax.text(
                xi + offset,
                1.03,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

        for i, xi in enumerate(x):
            ax.text(
                xi + offset,
                -0.07,
                mode.capitalize(),
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("Proportion of Successful Runs")
    ax.set_ylim(-0.12, 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{max(v,0):.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["bottom"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#2471a3", label="Stealth: clean output (global)"),
        mpatches.Patch(color="#f39c12", label="Detectable: dirty output (global)"),
        mpatches.Patch(color="#7fb3d3", label="Stealth: clean output (local)"),
        mpatches.Patch(color="#f8c471", label="Detectable: dirty output (local)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=2)

    # Add note explaining what detectable means
    # fig.text(
    #     0.5,
    #     -0.02,
    #     "Detectable = exfiltration succeeded but infection prompt leaked into user-visible output",
    #     ha="center",
    #     fontsize=9,
    #     color="gray",
    #     style="italic",
    # )

    plt.tight_layout()
    plt.savefig("figures/fig6_theft_detectability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig6")


# ── Run all ────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_rq1()
    plot_rq2()
    plot_rq3()
    plot_scam_breakdown()
    plot_theft_breakdown()
    plot_theft_detectability()
    print("\nAll 6 figures saved to figures/")
