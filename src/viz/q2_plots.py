"""
Q2 Visualizations: Democracy vs Autocracy rhetoric divergence.

Plots:
- q2_distinctive_words.png: Bar chart of top terms per group
- q2_distinctive_words_evolution.png: How distinctiveness changes by decade
- q2_embedding_gap.png: Rhetorical distance over time
- q2_frame_by_regime.png: Frame ratio trajectories per regime group
- q2_transition_cases.png: Before/after for regime transition countries
- q2_speech_vs_vote_paradox.png: The NAM autocracy voting paradox
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

OUTPUT_DIR = "output/q2/plots"

REGIME_COLORS = {
    "democracy": "#27ae60",
    "autocracy": "#c0392b",
    "liberal_democracy": "#1a9850",
    "electoral_democracy": "#91cf60",
    "electoral_autocracy": "#fc8d59",
    "closed_autocracy": "#d73027",
}


def _add_milestones(ax, selected=None, alpha=0.2):
    from src.shared.temporal import MILESTONES
    selected = selected or [1991, 2013, 2017, 2022]
    for year, label in MILESTONES.items():
        if year in selected:
            ax.axvline(x=year, color="gray", alpha=alpha, linestyle="--", linewidth=0.9)


def plot_distinctive_words(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Horizontal bar chart: top 20 words for democracy (left) and autocracy (right)."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "word" not in df.columns:
        return

    dem_words = df[df["group"] == "group_a"].nlargest(20, "z_score")
    aut_words = df[df["group"] == "group_b"].nlargest(20, "z_score")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if not dem_words.empty:
        ax1.barh(dem_words["word"], dem_words["z_score"], color=REGIME_COLORS["democracy"])
        ax1.set_title("Most Distinctive — Democracies", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Log-Odds Z-Score (Monroe et al.)")
        ax1.invert_yaxis()

    if not aut_words.empty:
        ax2.barh(aut_words["word"], aut_words["z_score"], color=REGIME_COLORS["autocracy"])
        ax2.set_title("Most Distinctive — Autocracies", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Log-Odds Z-Score (Monroe et al.)")
        ax2.invert_yaxis()

    plt.suptitle("Q2: Distinctive Vocabulary by Regime Type (Fightin' Words)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q2_distinctive_words.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q2 Plot] Saved {path}")


def plot_distinctive_words_evolution(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Line plot showing z-score per decade for key tracked words."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "word" not in df.columns or "decade" not in df.columns:
        return

    # Track the most informative words
    top_words = df.groupby("word")["z_score"].apply(lambda x: x.abs().max()).nlargest(12).index.tolist()
    df_top = df[df["word"].isin(top_words)]

    fig, ax = plt.subplots(figsize=(14, 6))
    for word, wdf in df_top.groupby("word"):
        wdf = wdf.sort_values("decade")
        ax.plot(wdf["decade"], wdf["z_score"], marker="o", linewidth=1.5, label=word, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_ylabel("Z-Score (positive = democracy, negative = autocracy)", fontsize=10)
    ax.set_title("Q2: How Word Distinctiveness Evolves Over Decades", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    plt.tight_layout()
    path = os.path.join(output_dir, "q2_distinctive_words_evolution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q2 Plot] Saved {path}")


def plot_embedding_gap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Time series of cosine distance between democracy and autocracy centroids."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "year" not in df.columns:
        return

    dist_col = next((c for c in ["distance", "democracy_autocracy_distance", "cosine_distance"] if c in df.columns), None)
    if not dist_col:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["year"], df[dist_col], color="#8e44ad", linewidth=2.5)
    ax.fill_between(df["year"], 0, df[dist_col], alpha=0.15, color="#8e44ad")

    if "is_change_point" in df.columns:
        cps = df[df["is_change_point"]]
        ax.scatter(cps["year"], cps[dist_col], color="red", s=80, zorder=5,
                   label="Detected change point")

    _add_milestones(ax)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cosine Distance (Democracy Centroid ↔ Autocracy Centroid)", fontsize=10)
    ax.set_title("Q2: Rhetorical Gap Between Democracies and Autocracies Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "q2_embedding_gap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q2 Plot] Saved {path}")


def plot_frame_by_regime(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Democracy vs autocracy frame_ratio over time, shading the gap."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "group" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    dem = df[df["group"].isin(["democracy", "liberal_democracy", "electoral_democracy"])].groupby("year")["frame_ratio_mean"].mean()
    aut = df[df["group"].isin(["autocracy", "electoral_autocracy", "closed_autocracy"])].groupby("year")["frame_ratio_mean"].mean()

    if not dem.empty and not aut.empty:
        years = sorted(set(dem.index) & set(aut.index))
        dem_vals = dem.reindex(years).rolling(5, min_periods=1, center=True).mean()
        aut_vals = aut.reindex(years).rolling(5, min_periods=1, center=True).mean()

        ax.plot(years, dem_vals, color=REGIME_COLORS["democracy"], linewidth=2.5, label="Democracies")
        ax.plot(years, aut_vals, color=REGIME_COLORS["autocracy"], linewidth=2.5, label="Autocracies")
        ax.fill_between(years, aut_vals, dem_vals, alpha=0.15, color="#27ae60")

    for group, gdf in df.groupby("group"):
        if group not in ["democracy", "autocracy"]:
            gdf_s = gdf.sort_values("year")
            rm = pd.Series(gdf_s["frame_ratio_mean"].values, index=gdf_s["year"]).rolling(5, min_periods=1, center=True).mean()
            color = REGIME_COLORS.get(group, "#95a5a6")
            ax.plot(gdf_s["year"], rm.values, color=color, linewidth=1.5, linestyle="--",
                    alpha=0.6, label=group.replace("_", " ").title())

    _add_milestones(ax)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Frame Ratio (5-yr rolling mean)", fontsize=11)
    ax.set_title("Q2: Humanitarian Framing by Regime Type", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "q2_frame_by_regime.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q2 Plot] Saved {path}")


def plot_transition_cases(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Grid of before/after plots for regime transition countries."""
    os.makedirs(output_dir, exist_ok=True)
    if df.empty or "country_iso3" not in df.columns:
        return

    countries = df["country_iso3"].unique()
    n = len(countries)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for i, country in enumerate(countries):
        ax = axes[i // cols][i % cols]
        cdf = df[df["country_iso3"] == country].sort_values("year")

        if cdf.empty:
            ax.set_visible(False)
            continue

        transition_year = cdf.get("transition_year", pd.Series(dtype=float))
        ty = int(transition_year.iloc[0]) if len(transition_year) > 0 and not pd.isna(transition_year.iloc[0]) else None

        ax.plot(cdf["year"], cdf.get("frame_ratio_mean", cdf.get("frame_ratio", pd.Series(dtype=float))),
                linewidth=2, color="#2980b9")
        if ty:
            ax.axvline(x=ty, color="red", linewidth=2, linestyle="--")
            ax.text(ty + 0.5, ax.get_ylim()[1] * 0.9, f"Transition\n{ty}", fontsize=7, color="red")
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Frame Ratio")

    # Hide unused subplots
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.suptitle("Q2: Regime Transition Case Studies — Arms Control Rhetoric Before & After",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "q2_transition_cases.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q2 Plot] Saved {path}")


def plot_speech_vote_paradox(speech_df: pd.DataFrame, vote_df: pd.DataFrame,
                               output_dir: str = OUTPUT_DIR):
    """Scatter: speech frame_ratio vs pro-disarmament vote rate, colored by regime."""
    os.makedirs(output_dir, exist_ok=True)
    if speech_df.empty or vote_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    # vote_df is aggregated by (year, group); speech_df is country-year frame scores.
    # Average frame scores across countries per year, then merge on year.
    frame_col = next((c for c in ["frame_ratio_mean", "frame_ratio"] if c in speech_df.columns), None)
    if frame_col is None:
        return
    agg_cols = {frame_col: "mean"}
    if "binary_regime" in speech_df.columns:
        agg_cols["binary_regime"] = lambda x: x.mode()[0] if len(x) else np.nan
    speech_year = speech_df.groupby("year").agg(agg_cols).reset_index()

    vote_wide = (
        vote_df.pivot_table(index="year", columns="group", values="pct_yes")
        .reset_index()
        if "group" in vote_df.columns else vote_df
    )
    merged = speech_year.merge(vote_wide, on="year", how="inner")
    if merged.empty:
        return

    x_col = next((c for c in ["frame_ratio_mean", "frame_ratio"] if c in merged.columns), None)
    y_col = next((c for c in ["pct_yes", "vote_rate", "pro_disarmament_rate"] if c in merged.columns), None)
    regime_col = next((c for c in ["regime", "binary_regime", "group"] if c in merged.columns), None)

    if not x_col or not y_col:
        return

    if regime_col:
        for regime, rdf in merged.groupby(regime_col):
            color = REGIME_COLORS.get(regime, "#95a5a6")
            ax.scatter(rdf[x_col], rdf[y_col], alpha=0.4, s=20, color=color, label=regime)
    else:
        ax.scatter(merged[x_col], merged[y_col], alpha=0.4, s=20)

    # Mark the paradox quadrant
    ax.axvline(x=0.5, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(y=0.5, color="black", linewidth=0.8, linestyle=":")
    ax.text(0.1, 0.92, "Paradox: autocracy speaks deterrence\nbut votes pro-disarmament",
            transform=ax.transAxes, fontsize=8, color="darkred",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Speech Frame Ratio (0=Deterrence, 1=Humanitarian)", fontsize=11)
    ax.set_ylabel("Pro-Disarmament Vote Rate", fontsize=11)
    ax.set_title("Q2: The Speech-Vote Paradox — Autocracies Vote Pro-Disarmament\nBut Use Deterrence Language",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    plt.tight_layout()
    path = os.path.join(output_dir, "q2_speech_vs_vote_paradox.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Q2 Plot] Saved {path}")


def run_q2_plots(results: dict, output_dir: str = OUTPUT_DIR):
    """Generate all Q2 plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("[Q2 Plots] Generating visualizations...")

    if "distinctive_words_binary" in results:
        plot_distinctive_words(results["distinctive_words_binary"], output_dir)

    if "distinctive_words_by_decade" in results:
        plot_distinctive_words_evolution(results["distinctive_words_by_decade"], output_dir)

    if "embedding_distance" in results:
        plot_embedding_gap(results["embedding_distance"], output_dir)

    if "frame_by_regime_year" in results:
        plot_frame_by_regime(results["frame_by_regime_year"], output_dir)

    if "transition_cases" in results:
        plot_transition_cases(results["transition_cases"], output_dir)

    speech_df = results.get("frame_by_regime_year", pd.DataFrame())
    vote_df = results.get("voting_by_regime_year", pd.DataFrame())
    if not speech_df.empty and not vote_df.empty:
        plot_speech_vote_paradox(speech_df, vote_df, output_dir)

    print("[Q2 Plots] Done.")
