"""
Comparison visualizations: radar charts, heatmaps, box plots.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_FIG_DPI = 150


def _savefig(fig, output_dir: Optional[str], filename: str) -> None:
    if output_dir is not None:
        out = Path(output_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_radar_chart(
    metrics_df: pd.DataFrame,
    country_groups: Optional[Dict[str, List[str]]] = None,
    group_names: Optional[List[str]] = None,
    country_col: str = "country_code",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Radar chart comparing key rhetoric metrics across country groups.
    """
    metric_cols = [
        "treaty_anchor_similarity", "voting_score", "humanitarian_topic_pct",
        "commitment_strength", "care_harm_loading",
    ]
    available_metrics = [c for c in metric_cols if c in metrics_df.columns]

    if not available_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("No metrics available for radar chart")
        _savefig(fig, output_dir, "radar_chart.png")
        return fig

    if group_names is None:
        group_names = ["p5", "nac", "nam"]

    n_metrics = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    palette = sns.color_palette("Set1", len(group_names))

    for group_name, color in zip(group_names, palette):
        if country_groups is not None and group_name in country_groups:
            members = country_groups[group_name]
            sub = metrics_df[metrics_df[country_col].isin(members)]
        else:
            # Try filtering by a 'group' column if present
            if "group" in metrics_df.columns:
                sub = metrics_df[metrics_df["group"] == group_name]
            else:
                continue

        if sub.empty:
            continue

        values = sub[available_metrics].mean().tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=group_name)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in available_metrics], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Rhetoric Metrics by Country Group", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir, "radar_chart.png")
    return fig


def plot_anchor_similarity_heatmap(
    anchor_scores_df: pd.DataFrame,
    country_col: str = "country_code",
    output_dir: Optional[str] = None,
    top_n: int = 30,
) -> plt.Figure:
    """Heatmap of countries × treaties anchor similarity scores."""
    score_cols = [c for c in anchor_scores_df.columns if c.endswith("_score")]

    if not score_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("No anchor score columns found")
        _savefig(fig, output_dir, "anchor_similarity_heatmap.png")
        return fig

    # Average across years
    agg = anchor_scores_df.groupby(country_col)[score_cols].mean()
    if len(agg) > top_n:
        # Sort by overall mean similarity
        agg = agg.loc[agg.mean(axis=1).nlargest(top_n).index]

    fig, ax = plt.subplots(figsize=(max(8, len(score_cols) * 1.5), max(6, len(agg) * 0.35)))
    sns.heatmap(
        agg,
        ax=ax,
        cmap="RdYlGn",
        center=0.35,
        annot=False,
        linewidths=0.2,
        cbar_kws={"label": "Cosine Similarity"},
    )
    ax.set_xlabel("Treaty")
    ax.set_ylabel("Country")
    ax.set_title("Treaty Anchor Similarity by Country")
    ax.set_xticklabels(
        [c.replace("_score", "").upper() for c in score_cols],
        rotation=30, ha="right",
    )
    fig.tight_layout()
    _savefig(fig, output_dir, "anchor_similarity_heatmap.png")
    return fig


def plot_rhetoric_by_regime_type(
    rhetoric_df: pd.DataFrame,
    regime_col: str = "regime_type",
    score_col: str = "rhetoric_score",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Box plot of rhetoric scores by regime type."""
    if regime_col not in rhetoric_df.columns:
        # Try to assign regime type
        try:
            from src.groups import assign_regime_type
            rhetoric_df = rhetoric_df.copy()
            rhetoric_df[regime_col] = rhetoric_df["country_code"].apply(assign_regime_type)
        except Exception:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title("No regime type data available")
            _savefig(fig, output_dir, "rhetoric_by_regime.png")
            return fig

    if score_col not in rhetoric_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"Column '{score_col}' not found")
        _savefig(fig, output_dir, "rhetoric_by_regime.png")
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))
    order = (
        rhetoric_df.groupby(regime_col)[score_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(
        data=rhetoric_df,
        x=regime_col,
        y=score_col,
        order=order,
        palette="Set3",
        ax=ax,
    )
    ax.set_xlabel("Regime Type")
    ax.set_ylabel("Rhetoric Score")
    ax.set_title("Arms Control Rhetoric Score by Regime Type")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _savefig(fig, output_dir, "rhetoric_by_regime.png")
    return fig
