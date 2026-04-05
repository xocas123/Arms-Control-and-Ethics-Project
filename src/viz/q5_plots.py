"""
Q5 Visualizations: Regime-Treaty Rhetoric Divide.

Plots:
- q5a_treaty_regime_adoption.png: Adoption curves split by democracy/autocracy
- q5b_rhetoric_vs_voting_scatter.png: Rhetoric score vs voting, colored by regime
- q5b_gap_distribution_by_regime.png: Violin plot of rhetoric-action gap
- q5b_nam_paradox_heatmap.png: NAM countries gap over time
- q5c_export_tiv_vs_treaty_sim.png: Arms trade vs treaty similarity
- q5c_major_exporter_profiles.png: Radar chart of top exporters
- q5d_dendrogram_regime.png: Hierarchical clustering colored by regime
- q5d_umap_treaty_space.png: 2D projection of treaty space
- q5e_transition_treaty_trajectories.png: Small multiples for transitions
- q5f_gap_evolution.png: Dem-aut gap per treaty over time
- q5g_fightin_words_regime_treaty.png: Distinctive words by regime × treaty
- q5_summary_dashboard.png: 4-panel summary
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

OUTPUT_DIR = "output/q5/plots"

REGIME_COLORS = {
    "democracy": "#27ae60",
    "autocracy": "#c0392b",
}
ALLIANCE_COLORS = {
    "NAM": "#e67e22",
    "NATO": "#3498db",
    "P5": "#c0392b",
    "other": "#95a5a6",
}
TREATY_COLORS = {
    "att": "#3498db",
    "tpnw": "#e74c3c",
    "ottawa": "#2ecc71",
    "ccm": "#9b59b6",
    "npt": "#f39c12",
    "cwc": "#1abc9c",
    "bwc": "#e67e22",
}
KEY_MILESTONES = [1991, 1997, 2008, 2013, 2017, 2022]


def _add_milestones(ax, alpha=0.2):
    from src.shared.temporal import MILESTONES
    for year in KEY_MILESTONES:
        label = MILESTONES.get(year, "")
        ax.axvline(x=year, color="gray", alpha=alpha, linestyle="--", linewidth=0.8)


def _savefig(fig, name, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Q5 Plot] Saved {path}")


# ── 5a: Treaty-Regime Adoption Curves ────────────────────────────────────────

def plot_treaty_regime_adoption(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """2x2 grid: adoption curves split by democracy/autocracy for each treaty."""
    if df.empty:
        return

    treaties = df["treaty"].unique()
    n = len(treaties)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for idx, treaty in enumerate(sorted(treaties)):
        ax = axes[idx // cols][idx % cols]
        tdf = df[(df["treaty"] == treaty) & (df["group"] == "ratifiers")]

        for regime, color in REGIME_COLORS.items():
            rdf = tdf[tdf["regime"] == regime].sort_values("year_relative")
            if rdf.empty:
                continue
            ax.plot(rdf["year_relative"], rdf["mean_similarity"],
                    color=color, linewidth=2, label=regime.title())
            if "std_similarity" in rdf.columns:
                ax.fill_between(
                    rdf["year_relative"],
                    rdf["mean_similarity"] - rdf["std_similarity"],
                    rdf["mean_similarity"] + rdf["std_similarity"],
                    alpha=0.15, color=color,
                )

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"{treaty.upper()} — Ratifiers", fontsize=12, fontweight="bold")
        ax.set_xlabel("Years Relative to Ratification")
        ax.set_ylabel("Cosine Similarity to Treaty Anchor")
        ax.legend(fontsize=9)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Treaty Adoption Curves: Democracies vs Autocracies",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "q5a_treaty_regime_adoption.png", output_dir)


# ── 5b: Rhetoric-Action Gap ──────────────────────────────────────────────────

def plot_rhetoric_vs_voting_scatter(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Scatter: rhetoric_score vs vote_score, colored by regime."""
    if df.empty or "rhetoric_score" not in df.columns or "vote_score" not in df.columns:
        return

    plot_df = df.dropna(subset=["rhetoric_score", "vote_score"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    for regime, color in REGIME_COLORS.items():
        rdf = plot_df[plot_df["regime"] == regime]
        if rdf.empty:
            continue
        ax.scatter(rdf["rhetoric_score"], rdf["vote_score"],
                   alpha=0.3, s=15, color=color, label=regime.title())

    ax.set_xlabel("Mean Treaty Rhetoric Score (Cosine Similarity)", fontsize=11)
    ax.set_ylabel("Pro-Disarmament Vote Rate", fontsize=11)
    ax.set_title("Rhetoric vs Voting: The Say-Do Gap by Regime Type",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Quadrant lines
    xmid = plot_df["rhetoric_score"].median()
    ymid = plot_df["vote_score"].median()
    ax.axhline(y=ymid, color="gray", alpha=0.3, linestyle=":")
    ax.axvline(x=xmid, color="gray", alpha=0.3, linestyle=":")

    fig.tight_layout()
    _savefig(fig, "q5b_rhetoric_vs_voting_scatter.png", output_dir)


def plot_gap_distribution(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Violin/box plot of rhetoric-action gap by regime × alliance."""
    if df.empty or "gap_rhetoric_vote" not in df.columns:
        return

    plot_df = df.dropna(subset=["gap_rhetoric_vote"])
    if plot_df.empty or len(plot_df) < 10:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # By regime
    for i, regime in enumerate(["democracy", "autocracy"]):
        rdf = plot_df[plot_df["regime"] == regime]["gap_rhetoric_vote"]
        if rdf.empty:
            continue
        bp = ax1.boxplot([rdf.values], positions=[i], widths=0.5,
                         patch_artist=True, showmeans=True)
        bp["boxes"][0].set_facecolor(REGIME_COLORS[regime])
        bp["boxes"][0].set_alpha(0.6)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Democracy", "Autocracy"])
    ax1.set_ylabel("|Rhetoric Score − Vote Score|")
    ax1.set_title("Rhetoric-Action Gap by Regime", fontsize=12, fontweight="bold")

    # By alliance
    alliances = ["NAM", "NATO", "P5", "other"]
    positions = range(len(alliances))
    for i, alliance in enumerate(alliances):
        adf = plot_df[plot_df["alliance"] == alliance]["gap_rhetoric_vote"]
        if adf.empty:
            continue
        bp = ax2.boxplot([adf.values], positions=[i], widths=0.5,
                         patch_artist=True, showmeans=True)
        bp["boxes"][0].set_facecolor(ALLIANCE_COLORS.get(alliance, "#95a5a6"))
        bp["boxes"][0].set_alpha(0.6)

    ax2.set_xticks(list(positions))
    ax2.set_xticklabels(alliances)
    ax2.set_ylabel("|Rhetoric Score − Vote Score|")
    ax2.set_title("Rhetoric-Action Gap by Alliance", fontsize=12, fontweight="bold")

    fig.suptitle("Who Says One Thing and Does Another?",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "q5b_gap_distribution_by_regime.png", output_dir)


def plot_nam_paradox_heatmap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Heatmap: NAM autocracies rhetoric-action gap over time."""
    if df.empty or "gap_rhetoric_vote" not in df.columns:
        return

    nam_aut = df[(df["alliance"] == "NAM") & (df["regime"] == "autocracy")]
    if nam_aut.empty:
        return

    # Get top-20 NAM autocracies by mean gap
    top = (
        nam_aut.groupby("country_iso3")["gap_rhetoric_vote"]
        .mean().nlargest(20).index.tolist()
    )
    pivot = nam_aut[nam_aut["country_iso3"].isin(top)].pivot_table(
        index="country_iso3", columns="year", values="gap_rhetoric_vote", aggfunc="mean"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    years = pivot.columns.astype(int)
    tick_positions = list(range(0, len(years), 5))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([years[i] for i in tick_positions], fontsize=8, rotation=45)

    plt.colorbar(im, ax=ax, label="Rhetoric-Vote Gap", shrink=0.8)
    ax.set_title("NAM Autocracies: The Say-Do Paradox Over Time",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    _savefig(fig, "q5b_nam_paradox_heatmap.png", output_dir)


# ── 5c: Arms Trade ───────────────────────────────────────────────────────────

def plot_export_vs_treaty_sim(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Scatter: log(export TIV) vs treaty similarity, faceted by treaty."""
    if df.empty:
        return

    sim_cols = [c for c in df.columns if c.endswith("_similarity")]
    if not sim_cols or "log_export_tiv" not in df.columns:
        return

    treaties = [c.replace("_similarity", "") for c in sim_cols]
    n = min(len(treaties), 4)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, treaty in enumerate(treaties[:n]):
        ax = axes[0][i]
        col = f"{treaty}_similarity"
        plot_df = df[df["log_export_tiv"] > 0].dropna(subset=[col])
        if plot_df.empty:
            continue

        for regime, color in REGIME_COLORS.items():
            rdf = plot_df[plot_df["regime"] == regime]
            ax.scatter(rdf["log_export_tiv"], rdf[col],
                       alpha=0.3, s=12, color=color, label=regime.title())

        ax.set_xlabel("log(Export TIV + 1)")
        ax.set_ylabel(f"{treaty.upper()} Similarity")
        ax.set_title(treaty.upper(), fontsize=11, fontweight="bold")
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Arms Exports vs Treaty Rhetoric",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "q5c_export_tiv_vs_treaty_sim.png", output_dir)


def plot_major_exporter_profiles(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Bar chart comparing treaty similarity for major exporters vs others."""
    if df.empty:
        return

    sim_cols = [c for c in df.columns if c.endswith("_similarity")]
    if not sim_cols:
        return

    # Average by trade role
    agg = df.groupby("trade_role")[sim_cols].mean()
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    treaties = [c.replace("_similarity", "").upper() for c in sim_cols]
    x = np.arange(len(treaties))
    width = 0.25

    colors = {"major_exporter": "#c0392b", "major_importer": "#3498db", "other": "#95a5a6"}
    for i, role in enumerate(["major_exporter", "major_importer", "other"]):
        if role in agg.index:
            vals = agg.loc[role, sim_cols].values
            ax.bar(x + i * width, vals, width, label=role.replace("_", " ").title(),
                   color=colors.get(role, "#95a5a6"), alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(treaties, fontsize=10)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Treaty Rhetoric by Arms Trade Role",
                 fontsize=13, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    _savefig(fig, "q5c_major_exporter_profiles.png", output_dir)


# ── 5d: Clustering ───────────────────────────────────────────────────────────

def plot_dendrogram(clusters_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Dendrogram colored by regime type."""
    if clusters_df.empty:
        return

    Z = clusters_df.attrs.get("linkage")
    countries = clusters_df.attrs.get("countries")
    if Z is None or countries is None:
        return

    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(20, 8))

    # Color map by regime
    regime_map = dict(zip(clusters_df["country_iso3"], clusters_df["regime_mode"]))
    leaf_colors = {}
    for i, c in enumerate(countries):
        regime = regime_map.get(c, "unknown")
        leaf_colors[i] = REGIME_COLORS.get(regime, "#95a5a6")

    dend = dendrogram(
        Z, labels=countries, ax=ax, leaf_rotation=90, leaf_font_size=6,
        color_threshold=0,
    )

    # Color leaf labels
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        c = lbl.get_text()
        regime = regime_map.get(c, "unknown")
        lbl.set_color(REGIME_COLORS.get(regime, "#95a5a6"))

    ax.set_title("Treaty Rhetoric Clusters (Ward Linkage)\nLeaf Color = Regime Type",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=REGIME_COLORS["democracy"], label="Democracy"),
        Patch(facecolor=REGIME_COLORS["autocracy"], label="Autocracy"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    _savefig(fig, "q5d_dendrogram_regime.png", output_dir)


def plot_umap_treaty_space(clusters_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """2D projection of treaty proximity vectors."""
    if clusters_df.empty:
        return

    proximity_matrix = clusters_df.attrs.get("proximity_matrix")
    countries = clusters_df.attrs.get("countries")
    if proximity_matrix is None or countries is None:
        return

    # Use PCA as fallback (UMAP may not be installed)
    try:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(proximity_matrix)
        method = "PCA"
    except Exception:
        return

    regime_map = dict(zip(clusters_df["country_iso3"], clusters_df["regime_mode"]))

    fig, ax = plt.subplots(figsize=(12, 10))

    for regime, color in REGIME_COLORS.items():
        idxs = [i for i, c in enumerate(countries) if regime_map.get(c) == regime]
        if not idxs:
            continue
        ax.scatter(coords[idxs, 0], coords[idxs, 1],
                   c=color, alpha=0.6, s=40, label=regime.title())
        # Label select countries
        from src.data.groups import NWS, NAM
        for i in idxs:
            c = countries[i]
            if c in NWS or c in ["DEU", "JPN", "BRA", "ZAF", "MEX", "IND", "IRN", "SAU"]:
                ax.annotate(c, (coords[i, 0], coords[i, 1]),
                            fontsize=7, alpha=0.8, ha="center", va="bottom")

    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    ax.set_title(f"Countries in Treaty Rhetoric Space ({method})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    _savefig(fig, "q5d_umap_treaty_space.png", output_dir)


# ── 5e: Transitions ──────────────────────────────────────────────────────────

def plot_transition_treaty_trajectories(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Small multiples: treaty similarity around regime transitions."""
    if df.empty:
        return

    countries = df["country_iso3"].unique()
    n = len(countries)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, country in enumerate(sorted(countries)):
        ax = axes[idx // cols][idx % cols]
        cdf = df[df["country_iso3"] == country]
        ty = cdf["transition_year"].iloc[0]

        for treaty in cdf["treaty"].unique():
            tdf = cdf[cdf["treaty"] == treaty].sort_values("year")
            color = TREATY_COLORS.get(treaty, "#95a5a6")
            ax.plot(tdf["year"], tdf["similarity"], color=color,
                    linewidth=1.5, label=treaty.upper(), alpha=0.8)

        ax.axvline(x=ty, color="black", linestyle="--", alpha=0.6, linewidth=1.5)
        direction = cdf["direction"].iloc[0]
        ax.set_title(f"{country} ({direction}, {ty})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Treaty Similarity", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, ncol=2)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Treaty Rhetoric Around Regime Transitions",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "q5e_transition_treaty_trajectories.png", output_dir)


# ── 5f: Gap Evolution ────────────────────────────────────────────────────────

def plot_gap_evolution(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Multi-line: dem-aut gap per treaty over time."""
    if df.empty or "gap" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    for treaty in df["treaty"].unique():
        tdf = df[df["treaty"] == treaty].sort_values("year")
        color = TREATY_COLORS.get(treaty, "#95a5a6")

        # Smooth with rolling mean
        if len(tdf) >= 5:
            smooth = tdf["gap"].rolling(5, center=True, min_periods=1).mean()
        else:
            smooth = tdf["gap"]

        ax.plot(tdf["year"], smooth, color=color, linewidth=2,
                label=treaty.upper(), alpha=0.9)

        # Mark change points
        if "is_change_point" in tdf.columns:
            cps = tdf[tdf["is_change_point"] == True]
            ax.scatter(cps["year"], smooth.iloc[cps.index - tdf.index[0]] if len(cps) > 0 else [],
                       color=color, s=60, zorder=5, edgecolors="black", linewidth=0.5)

    ax.axhline(y=0, color="gray", alpha=0.3, linestyle="-")
    _add_milestones(ax)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Democracy − Autocracy Similarity Gap", fontsize=11)
    ax.set_title("How the Regime-Treaty Rhetoric Gap Evolves\n(Positive = Democracies Closer to Treaty)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)

    fig.tight_layout()
    _savefig(fig, "q5f_gap_evolution.png", output_dir)


# ── 5g: Fightin' Words ───────────────────────────────────────────────────────

def plot_fightin_words_regime_treaty(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """2x2 grid: distinctive words per treaty, democracy vs autocracy."""
    if df.empty or "treaty" not in df.columns:
        return

    treaties = df["treaty"].unique()
    n = min(len(treaties), 4)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)

    for idx, treaty in enumerate(sorted(treaties)[:4]):
        ax = axes[idx // cols][idx % cols]
        tdf = df[df["treaty"] == treaty]

        dem_words = tdf[tdf["group"] == "group_a"].nlargest(10, "z_score")
        aut_words = tdf[tdf["group"] == "group_b"].nlargest(10, "z_score")

        combined = pd.concat([
            dem_words.assign(direction="Democracy", display_z=dem_words["z_score"]),
            aut_words.assign(direction="Autocracy", display_z=-aut_words["z_score"]),
        ])
        combined = combined.sort_values("display_z")

        colors = [REGIME_COLORS["democracy"] if d == "Democracy" else REGIME_COLORS["autocracy"]
                  for d in combined["direction"]]

        ax.barh(combined["word"], combined["display_z"], color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(f"{treaty.upper()} — Regime-Distinctive Words",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("← Autocracy | Democracy →")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("When Discussing the Same Treaty, Democracies and Autocracies Use Different Words",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "q5g_fightin_words_regime_treaty.png", output_dir)


# ── Summary Dashboard ─────────────────────────────────────────────────────────

def plot_summary_dashboard(results: dict, output_dir: str = OUTPUT_DIR):
    """4-panel summary figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Gap evolution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    gap_evo = results.get("regime_treaty_gap_evolution", pd.DataFrame())
    if not gap_evo.empty and "gap" in gap_evo.columns:
        for treaty in gap_evo["treaty"].unique():
            tdf = gap_evo[gap_evo["treaty"] == treaty].sort_values("year")
            color = TREATY_COLORS.get(treaty, "#95a5a6")
            if len(tdf) >= 5:
                smooth = tdf["gap"].rolling(5, center=True, min_periods=1).mean()
            else:
                smooth = tdf["gap"]
            ax1.plot(tdf["year"], smooth, color=color, linewidth=1.5,
                     label=treaty.upper(), alpha=0.9)
        ax1.axhline(y=0, color="gray", alpha=0.3)
        ax1.set_title("Regime-Treaty Gap Over Time", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Dem − Aut Similarity")
        ax1.legend(fontsize=7, ncol=2)

    # Panel 2: Gap by alliance (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    gap_group = results.get("rhetoric_action_gap_by_group", pd.DataFrame())
    if not gap_group.empty and "gap_mean" in gap_group.columns:
        plot_df = gap_group.sort_values("gap_mean", ascending=True)
        labels = [f"{r['regime']}\n{r['alliance']}" for _, r in plot_df.iterrows()]
        colors = [REGIME_COLORS.get(r["regime"], "#95a5a6") for _, r in plot_df.iterrows()]
        ax2.barh(labels, plot_df["gap_mean"], color=colors, alpha=0.7)
        ax2.set_title("Rhetoric-Action Gap by Group", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Mean |Rhetoric − Vote| Gap")

    # Panel 3: Exporter vs importer (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    exp_imp = results.get("exporter_vs_importer", pd.DataFrame())
    if not exp_imp.empty:
        sim_cols = [c for c in exp_imp.columns if c.endswith("_similarity")]
        if sim_cols:
            agg = exp_imp.groupby("trade_role")[sim_cols].mean()
            treaties_names = [c.replace("_similarity", "").upper() for c in sim_cols]
            x = np.arange(len(treaties_names))
            width = 0.25
            colors_tr = {"major_exporter": "#c0392b", "major_importer": "#3498db", "other": "#95a5a6"}
            for i, role in enumerate(["major_exporter", "major_importer", "other"]):
                if role in agg.index:
                    ax3.bar(x + i * width, agg.loc[role].values, width,
                            label=role.replace("_", " ").title(),
                            color=colors_tr.get(role, "#95a5a6"), alpha=0.8)
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(treaties_names, fontsize=8)
            ax3.set_title("Treaty Rhetoric by Trade Role", fontsize=11, fontweight="bold")
            ax3.legend(fontsize=7)

    # Panel 4: Adoption curve for ATT (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    adoption = results.get("treaty_regime_adoption_curves", pd.DataFrame())
    if not adoption.empty:
        att = adoption[(adoption["treaty"] == "att") & (adoption["group"] == "ratifiers")]
        for regime, color in REGIME_COLORS.items():
            rdf = att[att["regime"] == regime].sort_values("year_relative")
            if not rdf.empty:
                ax4.plot(rdf["year_relative"], rdf["mean_similarity"],
                         color=color, linewidth=2, label=regime.title())
        ax4.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax4.set_title("ATT Adoption: Dem vs Aut", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Years from Ratification")
        ax4.legend(fontsize=8)

    fig.suptitle("Q5: Regime-Treaty Rhetoric Divide — Summary",
                 fontsize=15, fontweight="bold", y=1.01)
    _savefig(fig, "q5_summary_dashboard.png", output_dir)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_q5_plots(results: dict, output_dir: str = OUTPUT_DIR):
    """Generate all Q5 visualizations."""
    print("[Q5 Plots] Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        plot_treaty_regime_adoption(
            results.get("treaty_regime_adoption_curves", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5a error: {e}")

    try:
        plot_rhetoric_vs_voting_scatter(
            results.get("rhetoric_action_gap", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5b scatter error: {e}")

    try:
        plot_gap_distribution(
            results.get("rhetoric_action_gap", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5b gap dist error: {e}")

    try:
        plot_nam_paradox_heatmap(
            results.get("rhetoric_action_gap", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5b NAM error: {e}")

    try:
        plot_export_vs_treaty_sim(
            results.get("arms_trade_rhetoric", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5c export error: {e}")

    try:
        plot_major_exporter_profiles(
            results.get("arms_trade_rhetoric", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5c profiles error: {e}")

    try:
        plot_dendrogram(
            results.get("treaty_rhetoric_clusters", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5d dendrogram error: {e}")

    try:
        plot_umap_treaty_space(
            results.get("treaty_rhetoric_clusters", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5d UMAP error: {e}")

    try:
        plot_transition_treaty_trajectories(
            results.get("transition_treaty_analysis", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5e transitions error: {e}")

    try:
        plot_gap_evolution(
            results.get("regime_treaty_gap_evolution", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5f gap evo error: {e}")

    try:
        plot_fightin_words_regime_treaty(
            results.get("distinctive_words_regime_treaty", pd.DataFrame()), output_dir)
    except Exception as e:
        print(f"[Q5 Plot] q5g FW error: {e}")

    try:
        plot_summary_dashboard(results, output_dir)
    except Exception as e:
        print(f"[Q5 Plot] summary error: {e}")

    print("[Q5 Plots] Done.")
