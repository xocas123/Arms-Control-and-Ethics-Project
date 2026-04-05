"""
Q4 Visualizations: Nuclear vs non-nuclear state rhetoric.

Plots:
- q4_distinctive_words.png
- q4_frame_trajectories.png: Four nuclear groups over time
- q4_anchor_distances.png: NPT vs TPNW similarity by group
- q4_concept_sentiment.png: Sentiment toward key concepts
- q4_rhetorical_divide.png: NWS-NNWS distance over time with change points
- q4_p5_internal.png: Pairwise P5 similarity
- q4_voting_gap.png: Voting divergence over time
- q4_polarization.png: Between-group distance + within-group variance
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

OUTPUT_DIR = "output/q4/plots"

NUCLEAR_COLORS = {
    "nws": "#c0392b",
    "de_facto": "#e67e22",
    "umbrella": "#f1c40f",
    "nnws": "#27ae60",
    "NWS": "#c0392b",
    "DE_FACTO": "#e67e22",
    "UMBRELLA": "#f1c40f",
    "NNWS": "#27ae60",
}

P5_COLORS = {
    "USA": "#3498db",
    "RUS": "#e74c3c",
    "CHN": "#e67e22",
    "GBR": "#9b59b6",
    "FRA": "#1abc9c",
}

KEY_MILESTONES = [1991, 2013, 2017, 2022]


def _add_milestones(ax, alpha=0.2):
    from src.shared.temporal import MILESTONES
    for year, label in MILESTONES.items():
        if year in KEY_MILESTONES:
            ax.axvline(x=year, color="gray", alpha=alpha, linestyle="--", linewidth=0.9)
            ax.text(year + 0.3, ax.get_ylim()[1] * 0.97, label,
                    rotation=90, fontsize=6, color="gray", va="top", ha="left")


def plot_distinctive_words_nuclear(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Horizontal bar chart: top 20 terms for NWS vs NNWS."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "word" not in df.columns:
        return

    nws_words = df[df["group"] == "group_a"].nlargest(20, "z_score")
    nnws_words = df[df["group"] == "group_b"].nlargest(20, "z_score")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if not nws_words.empty:
        ax1.barh(nws_words["word"], nws_words["z_score"], color=NUCLEAR_COLORS["nws"])
        ax1.set_title("Most Distinctive — Nuclear Weapon States (NWS)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Log-Odds Z-Score (Monroe et al.)")
        ax1.invert_yaxis()

    if not nnws_words.empty:
        ax2.barh(nnws_words["word"], nnws_words["z_score"], color=NUCLEAR_COLORS["nnws"])
        ax2.set_title("Most Distinctive — Non-Nuclear Weapon States (NNWS)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Log-Odds Z-Score")
        ax2.invert_yaxis()

    plt.suptitle("Q4: Distinctive Vocabulary by Nuclear Status (Fightin' Words)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_distinctive_words.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_frame_trajectories(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Four-line plot: NWS, de facto, umbrella, NNWS frame_ratio over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "group" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for group, gdf in df.groupby("group"):
        gdf = gdf.sort_values("year")
        col = "frame_ratio_mean" if "frame_ratio_mean" in gdf.columns else "frame_ratio"
        rm = pd.Series(gdf[col].values, index=gdf["year"]).rolling(5, min_periods=1, center=True).mean()
        color = NUCLEAR_COLORS.get(group, "#95a5a6")
        label = {"nws": "NWS (P5)", "de_facto": "De Facto Nuclear", "umbrella": "Nuclear Umbrella",
                 "nnws": "NNWS (Non-nuclear)"}.get(group, group)
        ax.plot(gdf["year"], rm.values, color=color, linewidth=2.5, label=label)

    _add_milestones(ax)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Frame Ratio (5-yr rolling mean)", fontsize=11)
    ax.set_title("Q4: Humanitarian Frame Ratio by Nuclear Status", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_frame_trajectories.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_anchor_distances(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Two panels: NPT similarity and TPNW similarity by nuclear group over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "group" not in df.columns:
        return

    npt_col = next((c for c in ["npt_similarity", "npt_sim"] if c in df.columns), None)
    tpnw_col = next((c for c in ["tpnw_similarity", "tpnw_sim"] if c in df.columns), None)

    if not npt_col and not tpnw_col:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, col, title in [(ax1, npt_col, "NPT Anchor"), (ax2, tpnw_col, "TPNW Anchor")]:
        if not col:
            ax.set_visible(False)
            continue
        for group, gdf in df.groupby("group"):
            gdf = gdf.sort_values("year")
            if col not in gdf.columns:
                continue
            rm = pd.Series(gdf[col].values, index=gdf["year"]).rolling(5, min_periods=1, center=True).mean()
            color = NUCLEAR_COLORS.get(group, "#95a5a6")
            label = {"nws": "NWS", "de_facto": "De Facto", "umbrella": "Umbrella", "nnws": "NNWS"}.get(group, group)
            ax.plot(gdf["year"], rm.values, color=color, linewidth=2, label=label)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(f"Cosine Similarity to {title}", fontsize=10)
        ax.set_title(f"Similarity to {title}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        _add_milestones(ax)

    plt.suptitle("Q4: Treaty Anchor Proximity by Nuclear Status — NPT vs TPNW",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_anchor_distances.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_concept_sentiment(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Grid: one panel per concept, sentiment by nuclear group over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "concept" not in df.columns:
        return

    concepts = df["concept"].unique()
    focus = ["deterrence", "nuclear disarmament", "prohibition", "TPNW", "ban treaty"]
    concepts = [c for c in focus if c in concepts] or list(concepts)[:6]
    n = len(concepts)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, concept in enumerate(concepts):
        ax = axes[i // cols][i % cols]
        cdf = df[df["concept"] == concept]

        for group, gdf in cdf.groupby("group"):
            gdf = gdf.groupby("year")["mean_sentiment"].mean().reset_index()
            gdf = gdf.sort_values("year")
            color = NUCLEAR_COLORS.get(group, "#95a5a6")
            rm = pd.Series(gdf["mean_sentiment"].values, index=gdf["year"]).rolling(5, min_periods=1, center=True).mean()
            ax.plot(gdf["year"], rm.values, color=color, linewidth=1.8,
                    label={"nws": "NWS", "de_facto": "De Facto", "umbrella": "Umbrella", "nnws": "NNWS"}.get(group, group))

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f'"{concept}"', fontsize=10, fontweight="bold")
        ax.set_ylabel("Sentiment (VADER)", fontsize=8)
        ax.set_xlabel("Year", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.suptitle("Q4: Concept-Level Sentiment by Nuclear Status", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_concept_sentiment.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_rhetorical_divide(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """NWS-NNWS cosine distance over time with change points."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "year" not in df.columns:
        return

    dist_col = next((c for c in ["distance", "cosine_distance", "nws_nnws_distance"] if c in df.columns), None)
    if not dist_col:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["year"], df[dist_col], color="#c0392b", linewidth=2.5)
    ax.fill_between(df["year"], 0, df[dist_col], alpha=0.15, color="#c0392b")

    if "is_change_point" in df.columns:
        cps = df[df["is_change_point"]]
        ax.scatter(cps["year"], cps[dist_col], color="darkred", s=100, zorder=5,
                   marker="^", label="Structural break")

    _add_milestones(ax)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cosine Distance (NWS Centroid ↔ NNWS Centroid)", fontsize=10)
    ax.set_title("Q4: The Widening Rhetorical Divide — Nuclear vs Non-Nuclear States",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_rhetorical_divide.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_p5_internal(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Pairwise cosine similarity among P5 countries over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        return

    pair_col_a = next((c for c in ["country_a", "label_a"] if c in df.columns), None)
    pair_col_b = next((c for c in ["country_b", "label_b"] if c in df.columns), None)
    sim_col = next((c for c in ["similarity", "cosine_similarity"] if c in df.columns), None)

    if not pair_col_a or not sim_col:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for (a, b), pdf in df.groupby([pair_col_a, pair_col_b]):
        pdf = pdf.sort_values("year")
        rm = pd.Series(pdf[sim_col].values, index=pdf["year"]).rolling(5, min_periods=1, center=True).mean()
        label = f"{a}-{b}"
        # Color by US involvement
        if "USA" in (a, b):
            ls = "-"
        elif "RUS" in (a, b):
            ls = "--"
        else:
            ls = ":"
        ax.plot(pdf["year"], rm.values, linestyle=ls, linewidth=1.8, label=label, alpha=0.85)

    _add_milestones(ax)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cosine Similarity (5-yr rolling mean)", fontsize=10)
    ax.set_title("Q4: P5 Internal Rhetorical Alignment — Pairwise Similarity Over Time",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_p5_internal.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_voting_gap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Pro-disarmament voting rate by nuclear status group over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "group" not in df.columns:
        return

    vote_types = {
        "pct_yes_disarmament": "All Disarmament Resolutions",
        "pct_yes_nuclear": "Nuclear-Specific Resolutions",
        "pct_yes_tpnw": "TPNW Resolutions",
    }
    available = [k for k in vote_types if k in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        for group, gdf in df.groupby("group"):
            gdf = gdf.sort_values("year")
            color = NUCLEAR_COLORS.get(group, "#95a5a6")
            label = {"nws": "NWS", "de_facto": "De Facto Nuclear",
                     "umbrella": "Nuclear Umbrella", "nnws": "NNWS"}.get(group, group)
            ax.plot(gdf["year"], gdf[col], color=color, linewidth=2, label=label)

        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Pro-Disarmament Vote Rate", fontsize=10)
        ax.set_title(vote_types[col], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        _add_milestones(ax)

    plt.suptitle("Q4: UNGA Voting Gap by Nuclear Status", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_voting_gap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def plot_polarization(distance_df: pd.DataFrame, variance_df: pd.DataFrame,
                       output_dir: str = OUTPUT_DIR):
    """Between-group distance and within-group variance — the polarization chart."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    dist_col = next((c for c in ["distance", "cosine_distance"] if c in distance_df.columns), None)
    if not distance_df.empty and dist_col:
        ax1.plot(distance_df["year"], distance_df[dist_col], color="#c0392b", linewidth=2.5)
        ax1.fill_between(distance_df["year"], 0, distance_df[dist_col], alpha=0.15, color="#c0392b")
        ax1.set_ylabel("NWS-NNWS Centroid Distance", fontsize=10)
        ax1.set_title("Between-Group Divergence", fontsize=11, fontweight="bold")
        _add_milestones(ax1)

    if not variance_df.empty and "group" in variance_df.columns:
        var_col = next((c for c in ["frame_ratio_std", "embedding_variance", "variance"] if c in variance_df.columns), None)
        if var_col:
            for group, gdf in variance_df.groupby("group"):
                gdf = gdf.sort_values("year")
                color = NUCLEAR_COLORS.get(group, "#95a5a6")
                label = {"nws": "NWS (5 countries)", "nnws": "NNWS (~180 countries)",
                         "umbrella": "Nuclear Umbrella", "de_facto": "De Facto"}.get(group, group)
                ax2.plot(gdf["year"], gdf[var_col], color=color, linewidth=1.8, label=label)

    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Within-Group Variance", fontsize=10)
    ax2.set_title("Within-Group Diversity", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.text(0.02, 0.92,
             "Rising between-group + falling within-group =\nPolarization (groups solidifying)",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    _add_milestones(ax2)

    plt.suptitle("Q4: Polarization — Between-Group Divergence + Within-Group Homogenization",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q4_polarization.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q4 Plot] Saved {path}")


def run_q4_plots(results: dict, output_dir: str = OUTPUT_DIR):
    """Generate all Q4 plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("[Q4 Plots] Generating visualizations...")

    if "distinctive_words_nuclear" in results:
        plot_distinctive_words_nuclear(results["distinctive_words_nuclear"], output_dir)

    if "frame_by_nuclear_status" in results:
        plot_frame_trajectories(results["frame_by_nuclear_status"], output_dir)

    if "anchor_distances" in results:
        plot_anchor_distances(results["anchor_distances"], output_dir)

    if "concept_sentiment" in results:
        plot_concept_sentiment(results["concept_sentiment"], output_dir)

    if "embedding_distance_nws_nnws" in results:
        plot_rhetorical_divide(results["embedding_distance_nws_nnws"], output_dir)

    if "p5_internal_similarity" in results:
        plot_p5_internal(results["p5_internal_similarity"], output_dir)

    if "voting_gap" in results:
        plot_voting_gap(results["voting_gap"], output_dir)

    if "embedding_distance_nws_nnws" in results and "within_group_variance" in results:
        plot_polarization(results["embedding_distance_nws_nnws"],
                          results["within_group_variance"], output_dir)

    print("[Q4 Plots] Done.")
