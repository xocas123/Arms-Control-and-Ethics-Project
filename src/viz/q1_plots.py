"""
Q1 Visualizations: Humanitarian vs Deterrence framing over time.

Plots:
- q1_global_timeseries.png: Key chart — two lines, milestones annotated
- q1_by_group.png: NWS vs NNWS vs NAM vs NATO frame trajectories
- q1_regional_heatmap.png: Geographic diffusion (regions × years × frame_ratio)
- q1_variance.png: Convergence vs polarization (mean + std over time)
- q1_change_points.png: Detected structural breaks
- q1_speech_vs_vote.png: Speech frame vs vote frame correlation
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

OUTPUT_DIR = "output/q1/plots"

COLORS = {
    "humanitarian": "#e74c3c",
    "deterrence": "#2980b9",
    "nws": "#c0392b",
    "nnws": "#27ae60",
    "nato": "#2980b9",
    "nam": "#f39c12",
    "eu": "#8e44ad",
    "global": "#2c3e50",
    "embedding": "#e67e22",
}

KEY_MILESTONES = [1991, 1997, 2008, 2013, 2017, 2022]


def _add_milestones(ax, year_range=None, alpha=0.25):
    from src.shared.temporal import MILESTONES
    for year, label in MILESTONES.items():
        if year not in KEY_MILESTONES:
            continue
        if year_range and not (year_range[0] <= year <= year_range[1]):
            continue
        ax.axvline(x=year, color="gray", alpha=alpha, linestyle="--", linewidth=0.9)
        ax.text(year + 0.3, ax.get_ylim()[1] * 0.97, label,
                rotation=90, fontsize=6, color="gray", va="top", ha="left")


def plot_global_timeseries(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Global frame_ratio over time with rolling mean, std band, milestones."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "year" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    years = df["year"].values
    ratio = df.get("frame_ratio_mean", df.get("frame_ratio", pd.Series(dtype=float))).values

    # Raw mean
    ax.plot(years, ratio, color=COLORS["global"], alpha=0.3, linewidth=1, label="Annual mean")

    # Rolling 5-yr mean
    if "rolling_mean_5yr" in df.columns:
        ax.plot(years, df["rolling_mean_5yr"].values, color=COLORS["humanitarian"],
                linewidth=2.5, label="5-year rolling mean (lexicon)")
    else:
        rm = pd.Series(ratio, index=years).rolling(5, min_periods=1, center=True).mean()
        ax.plot(years, rm.values, color=COLORS["humanitarian"],
                linewidth=2.5, label="5-year rolling mean")

    # Std band
    std_col = "frame_ratio_std" if "frame_ratio_std" in df.columns else None
    if std_col:
        if "rolling_mean_5yr" in df.columns:
            mean_line = df["rolling_mean_5yr"].values
        else:
            mean_line = pd.Series(ratio, index=years).rolling(5, min_periods=1, center=True).mean().values
        std_vals = df[std_col].rolling(5, min_periods=1, center=True).mean().values if hasattr(df[std_col], "rolling") else df[std_col].values
        ax.fill_between(years, mean_line - std_vals, mean_line + std_vals,
                         alpha=0.15, color=COLORS["humanitarian"])

    # Embedding-based line if available
    if "frame_position_mean" in df.columns:
        fp = df["frame_position_mean"].values
        # Normalize to 0-1 scale
        fp_norm = (fp - fp.min()) / (fp.max() - fp.min() + 1e-10)
        ax.plot(years, fp_norm, color=COLORS["embedding"], linewidth=1.8,
                linestyle="--", alpha=0.7, label="Embedding similarity (normalized)")

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Frame Ratio (0=Deterrence, 1=Humanitarian)", fontsize=11)
    ax.set_title("Q1: Humanitarian vs Deterrence Framing in Arms Control Discourse (1970–2023)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.text(ax.get_xlim()[0] + 1, 0.51, "Neutral", fontsize=8, color="gray")

    _add_milestones(ax, year_range=(int(years.min()), int(years.max())))

    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "q1_global_timeseries.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q1 Plot] Saved {path}")


def plot_by_group(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Multi-line: NWS vs NNWS vs NAM vs NATO frame_ratio over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "group" not in df.columns:
        return

    GROUP_COLORS = {
        "NWS": COLORS["nws"], "NNWS": COLORS["nnws"],
        "NATO": COLORS["nato"], "NAM": COLORS["nam"],
        "EU": COLORS["eu"],
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for group, gdf in df.groupby("group"):
        gdf_sorted = gdf.sort_values("year")
        ratio = gdf_sorted["frame_ratio_mean"].values
        years = gdf_sorted["year"].values
        rm = pd.Series(ratio, index=years).rolling(5, min_periods=1, center=True).mean()
        color = GROUP_COLORS.get(group, "#95a5a6")
        ax.plot(years, rm.values, linewidth=2, label=group, color=color)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Frame Ratio (rolling 5-yr mean)", fontsize=11)
    ax.set_title("Q1: Humanitarian Frame Ratio by Country Group", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    _add_milestones(ax)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "q1_by_group.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q1 Plot] Saved {path}")


def plot_regional_heatmap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Heatmap: rows=regions, columns=5-year bins, color=mean frame_ratio."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "region" not in df.columns:
        return

    df = df.copy()
    df["period"] = (df["year"] // 5) * 5
    pivot = df.groupby(["region", "period"])["frame_ratio_mean"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Frame Ratio (Humanitarian)"})
    ax.set_title("Q1: Geographic Diffusion of Humanitarian Framing", fontsize=13, fontweight="bold")
    ax.set_xlabel("5-Year Period")
    ax.set_ylabel("Region")
    plt.tight_layout()
    path = os.path.join(output_dir, "q1_regional_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q1 Plot] Saved {path}")


def plot_variance(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Two panels: global mean frame_ratio + global std over time."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    years = df["year"].values
    mean_col = "frame_ratio_mean" if "frame_ratio_mean" in df.columns else "frame_ratio"
    std_col = "frame_ratio_std" if "frame_ratio_std" in df.columns else None

    ax1.plot(years, df[mean_col].values, color=COLORS["humanitarian"], linewidth=2)
    ax1.fill_between(years, 0, df[mean_col].values, alpha=0.15, color=COLORS["humanitarian"])
    ax1.set_ylabel("Global Mean Frame Ratio", fontsize=10)
    ax1.set_title("Q1: Convergence vs Polarization Analysis", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1)
    _add_milestones(ax1)

    if std_col:
        ax2.plot(years, df[std_col].values, color="#8e44ad", linewidth=2,
                 label="Cross-country std of frame_ratio")
        ax2.fill_between(years, 0, df[std_col].values, alpha=0.15, color="#8e44ad")
        ax2.set_ylabel("Std of Frame Ratio Across Countries", fontsize=10)
        ax2.set_xlabel("Year", fontsize=10)
        ax2.text(0.02, 0.92, "↑ mean + ↓ std = Convergence\n↑ mean + ↑ std = Polarization",
                 transform=ax2.transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        _add_milestones(ax2)

    plt.tight_layout()
    path = os.path.join(output_dir, "q1_variance.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q1 Plot] Saved {path}")


def plot_change_points(global_ts: pd.DataFrame, change_points: pd.DataFrame,
                        output_dir: str = OUTPUT_DIR):
    """Time series with detected change points."""
    os.makedirs(output_dir, exist_ok=True)
    if global_ts.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    years = global_ts["year"].values
    col = "frame_ratio_mean" if "frame_ratio_mean" in global_ts.columns else "frame_ratio"
    ax.plot(years, global_ts[col].values, color=COLORS["global"], linewidth=1.5, label="Frame ratio")

    if not change_points.empty and "break_year" in change_points.columns:
        for _, cp in change_points.iterrows():
            color = COLORS["humanitarian"] if cp.get("direction", "") == "toward_humanitarian" else COLORS["deterrence"]
            ax.axvline(x=cp["break_year"], color=color, linewidth=2, linestyle="--", alpha=0.8)
            ax.text(cp["break_year"] + 0.3, ax.get_ylim()[1] * 0.95,
                    f"{int(cp['break_year'])}\n({cp.get('direction', '')[:4]})",
                    fontsize=7, color=color, va="top")

    _add_milestones(ax, alpha=0.15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Frame Ratio", fontsize=11)
    ax.set_title("Q1: Structural Breaks in Humanitarian Framing Trend", fontsize=13, fontweight="bold")
    red_patch = mpatches.Patch(color=COLORS["humanitarian"], label="Shift toward humanitarian")
    blue_patch = mpatches.Patch(color=COLORS["deterrence"], label="Shift toward deterrence")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "q1_change_points.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q1 Plot] Saved {path}")


def plot_speech_vs_vote(correlation_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Time series of speech-vote frame correlation."""
    os.makedirs(output_dir, exist_ok=True)
    if correlation_df.empty or "year" not in correlation_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    years = correlation_df["year"].values
    corr = correlation_df.get("correlation", pd.Series(dtype=float)).values
    ax.plot(years, corr, color=COLORS["humanitarian"], linewidth=2)
    ax.fill_between(years, 0, corr, where=corr > 0, alpha=0.2, color=COLORS["humanitarian"])
    ax.fill_between(years, 0, corr, where=corr < 0, alpha=0.2, color=COLORS["deterrence"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Pearson r (speech frame ~ vote frame)", fontsize=11)
    ax.set_title("Q1: Alignment Between Speech Framing and Voting Behavior", fontsize=13, fontweight="bold")
    _add_milestones(ax)
    plt.tight_layout()
    path = os.path.join(output_dir, "q1_speech_vs_vote.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q1 Plot] Saved {path}")


def run_q1_plots(results: dict, output_dir: str = OUTPUT_DIR):
    """Generate all Q1 plots from results dict."""
    os.makedirs(output_dir, exist_ok=True)
    print("[Q1 Plots] Generating visualizations...")

    if "frame_ratio_global" in results:
        plot_global_timeseries(results["frame_ratio_global"], output_dir)
        plot_variance(results["frame_ratio_global"], output_dir)

    if "frame_ratio_by_group" in results:
        plot_by_group(results["frame_ratio_by_group"], output_dir)

    if "frame_ratio_by_region" in results:
        plot_regional_heatmap(results["frame_ratio_by_region"], output_dir)

    if "change_points" in results and "frame_ratio_global" in results:
        plot_change_points(results["frame_ratio_global"], results["change_points"], output_dir)

    if "vote_frame_correlation" in results:
        plot_speech_vs_vote(results["vote_frame_correlation"], output_dir)

    print("[Q1 Plots] Done.")
