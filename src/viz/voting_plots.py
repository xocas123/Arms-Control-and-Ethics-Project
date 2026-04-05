"""
Voting data visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

_FIG_DPI = 150


def _savefig(fig, output_dir: Optional[str], filename: str) -> None:
    if output_dir is not None:
        out = Path(output_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_voting_similarity_heatmap(
    voting_df: pd.DataFrame,
    year: Optional[int] = None,
    country_col: str = "country_code",
    year_col: str = "year",
    output_dir: Optional[str] = None,
    top_n: int = 40,
) -> plt.Figure:
    """Heatmap of pairwise voting similarity between countries."""
    feature_cols = ["pct_yes_disarmament", "pct_yes_nuclear", "voting_composite"]
    feature_cols = [c for c in feature_cols if c in voting_df.columns]

    if not feature_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("No voting feature columns found")
        _savefig(fig, output_dir, "voting_similarity_heatmap.png")
        return fig

    if year is not None and year_col in voting_df.columns:
        df = voting_df[voting_df[year_col] == year]
    else:
        df = voting_df.groupby(country_col)[feature_cols].mean().reset_index()

    df = df.set_index(country_col)[feature_cols]
    if len(df) > top_n:
        df = df.sample(top_n, random_state=42)

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(df.fillna(0))
    sim_df = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.35), max(6, len(df) * 0.35)))
    sns.heatmap(
        sim_df, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
        cbar_kws={"label": "Cosine Similarity"},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(f"UNGA Voting Similarity{' ' + str(year) if year else ''}")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)
    fig.tight_layout()
    _savefig(fig, output_dir, "voting_similarity_heatmap.png")
    return fig


def plot_voting_dendrogram(
    voting_df: pd.DataFrame,
    year: Optional[int] = None,
    country_col: str = "country_code",
    year_col: str = "year",
    output_dir: Optional[str] = None,
    top_n: int = 40,
) -> plt.Figure:
    """Hierarchical clustering dendrogram of countries based on voting patterns."""
    feature_cols = ["pct_yes_disarmament", "pct_yes_nuclear"]
    feature_cols = [c for c in feature_cols if c in voting_df.columns]

    if not feature_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("No voting feature columns found")
        _savefig(fig, output_dir, "voting_dendrogram.png")
        return fig

    if year is not None and year_col in voting_df.columns:
        df = voting_df[voting_df[year_col] == year]
    else:
        df = voting_df.groupby(country_col)[feature_cols].mean().reset_index()

    df = df.set_index(country_col)[feature_cols].fillna(0)
    if len(df) > top_n:
        df = df.sample(top_n, random_state=42)
    if len(df) < 3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Not enough countries for dendrogram")
        _savefig(fig, output_dir, "voting_dendrogram.png")
        return fig

    dist = pdist(df.values, metric="euclidean")
    Z = linkage(dist, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, labels=df.index.tolist(), ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"UNGA Voting Dendrogram{' ' + str(year) if year else ''}")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    _savefig(fig, output_dir, "voting_dendrogram.png")
    return fig


def plot_voting_vs_exports_scatter(
    merged_df: pd.DataFrame,
    country_col: str = "country_code",
    voting_col: str = "pct_yes_disarmament",
    export_col: Optional[str] = None,
    output_dir: Optional[str] = None,
    annotate_n: int = 15,
) -> plt.Figure:
    """Scatter of disarmament voting rate vs arms export volume."""
    # Find an export/action column
    if export_col is None:
        candidates = ["action_score", "autocracy_transfer_ratio", "conflict_flow_ratio",
                      "export_volume", "total_exports"]
        export_col = next((c for c in candidates if c in merged_df.columns), None)

    if voting_col not in merged_df.columns or export_col is None:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title(f"Required columns not found (voting={voting_col}, export={export_col})")
        _savefig(fig, output_dir, "voting_vs_exports_scatter.png")
        return fig

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        merged_df[export_col], merged_df[voting_col],
        s=70, alpha=0.8, c="steelblue", edgecolors="white", linewidths=0.5,
    )

    # Regression line
    x = merged_df[export_col].fillna(0).values
    y = merged_df[voting_col].fillna(0).values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label=f"Trend (slope={m:.2f})")

    # Annotate extremes
    merged_df = merged_df.copy()
    merged_df["_dist"] = (merged_df[export_col] - merged_df[export_col].mean()).abs() + \
                          (merged_df[voting_col] - merged_df[voting_col].mean()).abs()
    to_label = merged_df.nlargest(annotate_n, "_dist")
    for _, row in to_label.iterrows():
        ax.annotate(row[country_col], (row[export_col], row[voting_col]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"Arms Transfer Activity ({export_col})")
    ax.set_ylabel(f"Disarmament Voting Rate ({voting_col})")
    ax.set_title("Arms Transfer Activity vs Disarmament Voting")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir, "voting_vs_exports_scatter.png")
    return fig
