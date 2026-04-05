"""
Q3 Visualizations: Rhetoric before and after treaty ratification.

Plots:
- q3_{treaty}_curve.png: Adoption curve per treaty (4 plots)
- q3_cross_treaty.png: All four adoption curves overlaid
- q3_rhetoric_vs_voting.png: Three-curve plot per treaty
- q3_outliers.png: Named outlier countries
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

OUTPUT_DIR = "output/q3/plots"

GROUP_STYLES = {
    "ratifiers": {"color": "#e74c3c", "linestyle": "-", "linewidth": 2.5, "label": "Ratifiers"},
    "signatories_only": {"color": "#e67e22", "linestyle": "--", "linewidth": 1.8, "label": "Signatories (not ratified)"},
    "non_signatories": {"color": "#95a5a6", "linestyle": ":", "linewidth": 1.5, "label": "Non-signatories"},
    "opponents": {"color": "#2c3e50", "linestyle": "-.", "linewidth": 1.5, "label": "Opponents"},
}

TREATY_LABELS = {
    "att": "Arms Trade Treaty (ATT, 2013)",
    "tpnw": "Treaty on the Prohibition of Nuclear Weapons (TPNW, 2017)",
    "ottawa": "Ottawa Convention (1997)",
    "ccm": "Convention on Cluster Munitions (CCM, 2008)",
}


def plot_adoption_curve(df: pd.DataFrame, treaty: str, output_dir: str = OUTPUT_DIR):
    """
    4-line adoption curve centered on ratification year (x=0).
    Groups: ratifiers, signatories_only, non_signatories, opponents.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "year_relative" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    sim_col = next((c for c in ["mean_similarity", "similarity"] if c in df.columns), None)
    if not sim_col:
        return

    for group, style in GROUP_STYLES.items():
        gdf = df[df["group"] == group] if "group" in df.columns else pd.DataFrame()
        if gdf.empty:
            continue
        agg = gdf.groupby("year_relative")[sim_col].agg(["mean", "std"]).reset_index()
        ax.plot(agg["year_relative"], agg["mean"], **{k: v for k, v in style.items()})
        if "std" in agg.columns:
            ax.fill_between(agg["year_relative"], agg["mean"] - agg["std"],
                             agg["mean"] + agg["std"], alpha=0.1, color=style["color"])

    ax.axvline(x=0, color="black", linewidth=2, linestyle="-", alpha=0.8, label="Ratification event")
    ax.axhline(y=df[df["group"] == "non_signatories"][sim_col].mean() if "group" in df.columns else 0,
               color="#95a5a6", linewidth=1, linestyle=":", alpha=0.6, label="Non-signatory baseline")

    ax.set_xlabel("Years Relative to Ratification (0 = Ratification Year)", fontsize=11)
    ax.set_ylabel("Cosine Similarity to Treaty Anchor", fontsize=11)
    ax.set_title(f"Q3: Rhetorical Adoption Curve — {TREATY_LABELS.get(treaty, treaty.upper())}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-10, 10)

    plt.tight_layout()
    path = os.path.join(output_dir, f"q3_{treaty}_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q3 Plot] Saved {path}")


def plot_cross_treaty(all_curves: dict, output_dir: str = OUTPUT_DIR):
    """Overlay adoption curves (ratifiers only) for all four treaties."""
    os.makedirs(output_dir, exist_ok=True)
    if not all_curves:
        return

    TREATY_COLORS = {
        "att": "#e74c3c",
        "tpnw": "#2980b9",
        "ottawa": "#27ae60",
        "ccm": "#f39c12",
    }

    fig, ax = plt.subplots(figsize=(13, 6))

    for treaty, df in all_curves.items():
        if df.empty or "year_relative" not in df.columns:
            continue
        sim_col = next((c for c in ["mean_similarity", "similarity"] if c in df.columns), None)
        if not sim_col:
            continue

        ratifiers = df[df["group"] == "ratifiers"] if "group" in df.columns else df
        if ratifiers.empty:
            continue

        agg = ratifiers.groupby("year_relative")[sim_col].mean()

        # Normalize: subtract pre-ratification baseline (mean of years [-10,-1])
        baseline = agg[agg.index < 0].mean()
        normalized = agg - baseline

        ax.plot(normalized.index, normalized.values, linewidth=2.5,
                color=TREATY_COLORS.get(treaty, "#95a5a6"),
                label=treaty.upper())

    ax.axvline(x=0, color="black", linewidth=2, linestyle="-", alpha=0.7, label="Ratification")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Years Relative to Ratification", fontsize=11)
    ax.set_ylabel("Similarity Change Relative to Pre-Ratification Baseline", fontsize=10)
    ax.set_title("Q3: Cross-Treaty Comparison — Speed of Rhetorical Adoption", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-10, 10)
    ax.text(0.02, 0.95,
            "Steeper pre-ratification slope → rhetoric leads ratification\n"
            "Steeper post-ratification slope → rhetoric follows ratification",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, "q3_cross_treaty.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q3 Plot] Saved {path}")


def plot_rhetoric_vs_voting(rhetoric_df: pd.DataFrame, voting_df: pd.DataFrame,
                              treaty: str, output_dir: str = OUTPUT_DIR):
    """Three-curve plot: rhetoric similarity + vote rate + ratification event."""
    os.makedirs(output_dir, exist_ok=True)
    if rhetoric_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax2 = ax1.twinx()

    sim_col = next((c for c in ["mean_similarity", "similarity"] if c in rhetoric_df.columns), None)
    if sim_col:
        ratifiers = rhetoric_df[rhetoric_df["group"] == "ratifiers"] if "group" in rhetoric_df.columns else rhetoric_df
        if not ratifiers.empty:
            agg = ratifiers.groupby("year_relative")[sim_col].mean()
            ax1.plot(agg.index, agg.values, color="#e74c3c", linewidth=2.5,
                     label="Rhetoric: treaty anchor similarity")

    if not voting_df.empty and "year_relative" in voting_df.columns:
        vote_col = next((c for c in ["pct_yes", "vote_rate"] if c in voting_df.columns), None)
        if vote_col:
            vote_agg = voting_df[voting_df.get("group", pd.Series()) == "ratifiers"].groupby("year_relative")[vote_col].mean() \
                if "group" in voting_df.columns else voting_df.groupby("year_relative")[vote_col].mean()
            if not vote_agg.empty:
                ax2.plot(vote_agg.index, vote_agg.values, color="#2980b9", linewidth=2.5,
                         linestyle="--", label="Voting: % yes on treaty resolution")

    ax1.axvline(x=0, color="black", linewidth=2.5, linestyle="-", alpha=0.7, label="Ratification event")
    ax1.set_xlabel("Years Relative to Ratification", fontsize=11)
    ax1.set_ylabel("Cosine Similarity to Treaty Anchor", color="#e74c3c", fontsize=11)
    ax2.set_ylabel("% Voting Yes on Treaty Resolution", color="#2980b9", fontsize=11)
    ax1.set_title(f"Q3: Rhetoric vs Voting Around Ratification — {treaty.upper()}",
                  fontsize=12, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.set_xlim(-10, 10)

    plt.tight_layout()
    path = os.path.join(output_dir, f"q3_rhetoric_vs_voting_{treaty}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q3 Plot] Saved {path}")


def plot_outliers(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Scatter of outlier countries: ratified-no-language vs language-no-ratification."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "country_iso3" not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    type1 = df[df.get("outlier_type", pd.Series()) == "ratified_no_language"] if "outlier_type" in df.columns else pd.DataFrame()
    type2 = df[df.get("outlier_type", pd.Series()) == "language_no_ratification"] if "outlier_type" in df.columns else pd.DataFrame()

    sim_col = next((c for c in ["similarity_at_ratification", "mean_similarity", "similarity"] if c in df.columns), None)

    for ax, subset, title, color in [
        (ax1, type1, "Ratified but Never Adopted the Language\n(Signing for political reasons?)", "#e74c3c"),
        (ax2, type2, "Adopted the Language but Never Ratified\n(Rhetorical free-riders?)", "#2980b9"),
    ]:
        if subset.empty or not sim_col:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=12)
            ax.set_title(title, fontsize=11)
            continue

        y_vals = range(len(subset))
        ax.barh(y_vals, subset[sim_col].values, color=color, alpha=0.7)
        ax.set_yticks(list(y_vals))
        ax.set_yticklabels(subset["country_iso3"].values, fontsize=9)
        ax.set_xlabel("Treaty Anchor Similarity at Ratification", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

    plt.suptitle("Q3: Outlier Countries — Ratification-Language Mismatches", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q3_outliers.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q3 Plot] Saved {path}")


def run_q3_plots(results: dict, output_dir: str = OUTPUT_DIR):
    """Generate all Q3 plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("[Q3 Plots] Generating visualizations...")

    all_curves = {}
    for treaty in ["att", "tpnw", "ottawa", "ccm"]:
        curve_key = f"adoption_curves_{treaty}"
        traj_key = f"similarity_trajectories_{treaty}"

        if curve_key in results:
            plot_adoption_curve(results[curve_key], treaty, output_dir)
            all_curves[treaty] = results[curve_key]

        if traj_key in results and curve_key in results:
            voting_key = f"voting_adoption_{treaty}"
            plot_rhetoric_vs_voting(results[curve_key],
                                     results.get(voting_key, pd.DataFrame()),
                                     treaty, output_dir)

    if all_curves:
        plot_cross_treaty(all_curves, output_dir)

    if "outliers" in results:
        plot_outliers(results["outliers"], output_dir)

    print("[Q3 Plots] Done.")
