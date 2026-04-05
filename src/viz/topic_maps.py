"""
Topic map visualizations.
"""

from pathlib import Path
from typing import List, Optional

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


def plot_intertopic_distance_map(
    bertopic_model=None,
    topic_embeddings: Optional[np.ndarray] = None,
    topic_labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    2D scatter of topic positions (intertopic distance map).

    Uses BERTopic model if available; falls back to supplied embeddings.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if bertopic_model is not None:
        try:
            from bertopic import BERTopic
            topic_info = bertopic_model.get_topic_info()
            topic_info = topic_info[topic_info["Topic"] != -1]
            # Get topic embeddings from BERTopic internal state
            if hasattr(bertopic_model, "topic_embeddings_"):
                embs = bertopic_model.topic_embeddings_
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(embs[1:])  # skip outlier topic
                ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.7, c=range(len(coords)),
                           cmap="tab20")
                for i, row in topic_info.iterrows():
                    if i < len(coords):
                        ax.annotate(
                            str(row["Topic"]),
                            (coords[i, 0], coords[i, 1]),
                            fontsize=7,
                        )
        except Exception as exc:
            ax.set_title(f"BERTopic map failed: {exc}")
    elif topic_embeddings is not None:
        from sklearn.decomposition import PCA
        n = min(2, topic_embeddings.shape[1])
        pca = PCA(n_components=n, random_state=42)
        coords = pca.fit_transform(topic_embeddings)
        labels = topic_labels or [str(i) for i in range(len(coords))]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=150, c=range(len(coords)),
                             cmap="tab20", alpha=0.8)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (coords[i, 0], coords[i, 1]), fontsize=8)
    else:
        # Generate mock plot with LDA-like topic positions
        rng = np.random.default_rng(42)
        n_topics = 10
        coords = rng.standard_normal((n_topics, 2))
        labels = [
            "Nuclear Disarm.", "Conv. Arms", "WMD/Nonpro.", "Humanitarian",
            "Terrorism", "Regional", "Development", "Cyber/Space",
            "Verification", "Self-Defense",
        ]
        ax.scatter(coords[:, 0], coords[:, 1], s=200, c=range(n_topics), cmap="tab10", alpha=0.8)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (coords[i, 0], coords[i, 1]), fontsize=8)
        ax.set_title("Intertopic Distance Map (Mock LDA Topics)")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if not ax.get_title():
        ax.set_title("Intertopic Distance Map")
    fig.tight_layout()
    _savefig(fig, output_dir, "intertopic_distance_map.png")
    return fig


def plot_topic_country_bipartite(
    topic_distributions_df: pd.DataFrame,
    country_col: str = "country_code",
    top_n_countries: int = 20,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of topic proportions per country (bipartite network proxy)."""
    topic_cols = [c for c in topic_distributions_df.columns if c.startswith("topic_")]

    if not topic_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("No topic columns found")
        _savefig(fig, output_dir, "topic_country_bipartite.png")
        return fig

    if country_col in topic_distributions_df.columns:
        agg = topic_distributions_df.groupby(country_col)[topic_cols].mean()
    else:
        agg = topic_distributions_df[topic_cols]

    if len(agg) > top_n_countries:
        # Select countries with highest topic diversity (entropy)
        entropy = -(agg * np.log(agg + 1e-9)).sum(axis=1)
        agg = agg.loc[entropy.nlargest(top_n_countries).index]

    fig, ax = plt.subplots(figsize=(max(8, len(topic_cols) * 1.2), max(6, len(agg) * 0.4)))
    sns.heatmap(agg, ax=ax, cmap="Blues", cbar_kws={"label": "Topic Proportion"})
    ax.set_xlabel("Topic")
    ax.set_ylabel("Country")
    ax.set_title("Topic Distribution by Country")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir, "topic_country_bipartite.png")
    return fig


def plot_sankey_topic_evolution(
    topics_over_time_df: pd.DataFrame,
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Simplified stacked bar representing topic evolution over time (Sankey proxy)."""
    topic_cols = [c for c in topics_over_time_df.columns if c.startswith("topic_")]

    if not topic_cols or year_col not in topics_over_time_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("No topic-over-time data available")
        _savefig(fig, output_dir, "topic_sankey_evolution.png")
        return fig

    pivot = topics_over_time_df.groupby(year_col)[topic_cols].mean()
    # Normalise rows to sum to 1
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic Proportion")
    ax.set_title("Topic Evolution Over Time (Stacked Bar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1), fontsize=7, ncol=1)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _savefig(fig, output_dir, "topic_sankey_evolution.png")
    return fig
