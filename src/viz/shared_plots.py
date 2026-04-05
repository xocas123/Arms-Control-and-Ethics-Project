"""
Cross-cutting visualizations: Era detection, cross-question summary.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

OUTPUT_DIR = "output/shared"

SIGNAL_COLORS = {
    "q1_frame_ratio": "#e74c3c",
    "q2_democracy_autocracy_distance": "#27ae60",
    "q3_treaty_anchor_similarity": "#2980b9",
    "q4_nws_nnws_distance": "#8e44ad",
}

SIGNAL_LABELS = {
    "q1_frame_ratio": "Q1: Global Humanitarian Frame Ratio",
    "q2_democracy_autocracy_distance": "Q2: Democracy-Autocracy Rhetorical Distance",
    "q3_treaty_anchor_similarity": "Q3: Global Treaty Anchor Similarity",
    "q4_nws_nnws_distance": "Q4: NWS-NNWS Rhetorical Distance",
}


def _add_milestones(ax, alpha=0.2):
    from src.shared.temporal import MILESTONES
    key = [1991, 1997, 2008, 2013, 2017, 2022]
    for year, label in MILESTONES.items():
        if year in key:
            ax.axvline(x=year, color="gray", alpha=alpha, linestyle="--", linewidth=0.9)


def plot_era_detection(era_df: pd.DataFrame, all_signals: dict,
                        output_dir: str = OUTPUT_DIR):
    """
    4-panel plot: one per signal + consensus change points.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_signals = len(all_signals)
    if n_signals == 0:
        return

    fig = plt.figure(figsize=(16, 4 * (n_signals + 1)))
    gs = gridspec.GridSpec(n_signals + 1, 1, hspace=0.4)

    # Individual signal panels
    for i, (signal_name, signal_df) in enumerate(all_signals.items()):
        ax = fig.add_subplot(gs[i])
        if signal_df.empty or "year" not in signal_df.columns:
            ax.text(0.5, 0.5, f"No data: {signal_name}", transform=ax.transAxes, ha="center")
            continue

        val_col = next((c for c in signal_df.columns if c not in ["year", "is_change_point"]), None)
        if not val_col:
            continue

        color = SIGNAL_COLORS.get(signal_name, "#2c3e50")
        ax.plot(signal_df["year"], signal_df[val_col], color=color, linewidth=2, alpha=0.8)
        ax.fill_between(signal_df["year"], signal_df[val_col].min(), signal_df[val_col],
                         alpha=0.1, color=color)

        # Mark change points from this signal
        if not era_df.empty and "signal" in era_df.columns:
            sig_cps = era_df[era_df["signal"] == signal_name]
            for _, cp in sig_cps.iterrows():
                ax.axvline(x=cp["break_year"], color="red", linewidth=2, linestyle="--", alpha=0.7)

        _add_milestones(ax)
        ax.set_title(SIGNAL_LABELS.get(signal_name, signal_name), fontsize=11, fontweight="bold")
        ax.set_ylabel(val_col.replace("_", " ").title(), fontsize=9)
        if i < n_signals - 1:
            ax.set_xticklabels([])

    # Consensus panel
    ax_consensus = fig.add_subplot(gs[n_signals])
    if not era_df.empty and "break_year" in era_df.columns:
        year_counts = era_df.groupby("break_year")["signal"].count()
        ax_consensus.bar(year_counts.index, year_counts.values, color="#2c3e50", alpha=0.7)
        # Highlight years where multiple signals agree
        consensus = year_counts[year_counts >= 2]
        ax_consensus.bar(consensus.index, consensus.values, color="#e74c3c", alpha=0.9,
                          label="Consensus (≥2 signals)")
        ax_consensus.legend(fontsize=9)

    ax_consensus.set_xlabel("Year", fontsize=11)
    ax_consensus.set_ylabel("Number of Signals Breaking", fontsize=10)
    ax_consensus.set_title("Cross-Question Era Detection: Structural Break Consensus",
                             fontsize=12, fontweight="bold")
    _add_milestones(ax_consensus)

    plt.suptitle("Era Detection: Do All Four Research Questions Agree on Historical Inflection Points?",
                 fontsize=14, fontweight="bold", y=1.01)

    path = os.path.join(output_dir, "era_detection.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Shared Plot] Saved {path}")


def run_shared_plots(results: dict, output_dir: str = OUTPUT_DIR):
    """Generate shared cross-question visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    print("[Shared Plots] Generating cross-question visualizations...")

    era_df = results.get("era_detection", pd.DataFrame())

    # Gather signals from question results
    all_signals = {}
    if "q1_global_ts" in results:
        df = results["q1_global_ts"].copy()
        col = "frame_ratio_mean" if "frame_ratio_mean" in df.columns else "frame_ratio"
        all_signals["q1_frame_ratio"] = df[["year", col]].rename(columns={col: "value"})

    if "q2_embedding_distance" in results:
        df = results["q2_embedding_distance"].copy()
        dist_col = next((c for c in ["distance", "cosine_distance"] if c in df.columns), None)
        if dist_col:
            all_signals["q2_democracy_autocracy_distance"] = df[["year", dist_col]].rename(columns={dist_col: "value"})

    if "q4_embedding_distance" in results:
        df = results["q4_embedding_distance"].copy()
        dist_col = next((c for c in ["distance", "cosine_distance"] if c in df.columns), None)
        if dist_col:
            all_signals["q4_nws_nnws_distance"] = df[["year", dist_col]].rename(columns={dist_col: "value"})

    if all_signals or not era_df.empty:
        plot_era_detection(era_df, all_signals, output_dir)

    print("[Shared Plots] Done.")
