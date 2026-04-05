"""
Temporal visualizations for arms control rhetoric trends.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_FIG_DPI = 150
_STYLE = "seaborn-v0_8-whitegrid"


def _savefig(fig, output_dir: Optional[str], filename: str) -> None:
    if output_dir is not None:
        out = Path(output_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_treaty_anchor_similarity_over_time(
    anchor_scores_df: pd.DataFrame,
    country_groups: Dict[str, List[str]],
    score_col: str = "mean_anchor_sim",
    year_col: str = "year",
    country_col: str = "country_code",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Line chart of mean treaty anchor similarity over time, one line per country group.
    """
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6))

    if score_col not in anchor_scores_df.columns:
        # Compute mean of all *_score columns
        score_cols = [c for c in anchor_scores_df.columns if c.endswith("_score")]
        if score_cols:
            anchor_scores_df = anchor_scores_df.copy()
            anchor_scores_df[score_col] = anchor_scores_df[score_cols].mean(axis=1)
        else:
            ax.set_title("No anchor score data available")
            _savefig(fig, output_dir, "anchor_similarity_over_time.png")
            return fig

    palette = sns.color_palette("tab10", len(country_groups))
    for (group_name, members), color in zip(country_groups.items(), palette):
        sub = anchor_scores_df[anchor_scores_df[country_col].isin(members)]
        if sub.empty:
            continue
        grp = sub.groupby(year_col)[score_col].mean().reset_index()
        ax.plot(grp[year_col], grp[score_col], label=group_name, color=color, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Treaty Anchor Similarity")
    ax.set_title("Treaty Anchor Similarity Over Time by Country Group")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    _savefig(fig, output_dir, "anchor_similarity_over_time.png")
    return fig


def plot_topic_prevalence_heatmap(
    topics_df: pd.DataFrame,
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of topic prevalence (topics × years)."""
    topic_cols = [c for c in topics_df.columns if c.startswith("topic_")]
    if not topic_cols or year_col not in topics_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("No topic data available")
        _savefig(fig, output_dir, "topic_prevalence_heatmap.png")
        return fig

    pivot = topics_df.groupby(year_col)[topic_cols].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot.T,
        ax=ax,
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Topic Proportion"},
        linewidths=0.3,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic")
    ax.set_title("Topic Prevalence Over Time")
    fig.tight_layout()
    _savefig(fig, output_dir, "topic_prevalence_heatmap.png")
    return fig


def plot_term_trajectories(
    term_freq_df: pd.DataFrame,
    terms: Optional[List[str]] = None,
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Line chart of TF-IDF term frequency trajectories."""
    if terms is None:
        terms = [c for c in term_freq_df.columns if c != year_col][:8]

    available_terms = [t for t in terms if t in term_freq_df.columns]
    if not available_terms or year_col not in term_freq_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("No term trajectory data available")
        _savefig(fig, output_dir, "term_trajectories.png")
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("tab20", len(available_terms))
    for term, color in zip(available_terms, palette):
        ax.plot(term_freq_df[year_col], term_freq_df[term], label=term, color=color, linewidth=1.8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean TF-IDF Score")
    ax.set_title("Arms Control Term Trajectories Over Time")
    ax.legend(loc="best", fontsize=7, ncol=3)
    fig.tight_layout()
    _savefig(fig, output_dir, "term_trajectories.png")
    return fig


def plot_moral_foundations_stacked(
    moral_df: pd.DataFrame,
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Stacked area chart of MFT dimensions over time."""
    mft_cols = ["care_harm", "fairness", "loyalty", "authority", "sanctity"]
    available = [c for c in mft_cols if c in moral_df.columns]

    if not available or year_col not in moral_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("No moral foundations data available")
        _savefig(fig, output_dir, "moral_foundations_stacked.png")
        return fig

    pivot = moral_df.groupby(year_col)[available].mean()
    years = pivot.index.tolist()
    values = [pivot[c].values for c in available]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2", len(available))
    ax.stackplot(years, values, labels=available, colors=palette, alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean MFT Loading (per 100 words)")
    ax.set_title("Moral Foundations Dimensions in Arms Control Speeches Over Time")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir, "moral_foundations_stacked.png")
    return fig
