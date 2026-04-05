"""
Rhetoric-action gap visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def plot_rhetoric_action_scatter(
    gap_df: pd.DataFrame,
    trade_volume_df: Optional[pd.DataFrame] = None,
    country_col: str = "country_code",
    year_col: str = "year",
    year: Optional[int] = None,
    output_dir: Optional[str] = None,
    annotate_n: int = 15,
) -> plt.Figure:
    """
    Scatter plot of rhetoric_score vs action_score with diagonal reference line.
    Above diagonal = hypocrite; below = quiet good actor.
    """
    if year is not None and year_col in gap_df.columns:
        df = gap_df[gap_df[year_col] == year].copy()
        title_suffix = f" ({year})"
    else:
        df = gap_df.groupby(country_col).agg(
            rhetoric_score=("rhetoric_score", "mean"),
            action_score=("action_score", "mean"),
            gap=("gap", "mean"),
        ).reset_index()
        title_suffix = " (all years, mean)"

    if "rhetoric_score" not in df.columns or "action_score" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Required columns missing from gap_df")
        _savefig(fig, output_dir, "rhetoric_action_scatter.png")
        return fig

    # Colour by gap
    fig, ax = plt.subplots(figsize=(10, 9))
    scatter = ax.scatter(
        df["action_score"],
        df["rhetoric_score"],
        c=df["gap"] if "gap" in df.columns else "steelblue",
        cmap="RdYlGn_r",
        vmin=-0.5, vmax=0.5,
        s=70, alpha=0.8, edgecolors="white", linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Gap (rhetoric − action)")

    # Diagonal: rhetoric = action (perfect alignment)
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Perfect alignment")

    # Shade regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="red", label="Rhetoric > Action")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="green", label="Action > Rhetoric")

    # Annotate extreme cases
    if "gap" in df.columns:
        top_hypocrites = df.nlargest(annotate_n // 2, "gap")
        top_quiet = df.nsmallest(annotate_n // 2, "gap")
        to_annotate = pd.concat([top_hypocrites, top_quiet])
    else:
        to_annotate = df.head(annotate_n)

    for _, row in to_annotate.iterrows():
        ax.annotate(
            row[country_col],
            (row["action_score"], row["rhetoric_score"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Action Score (arms transfer activity)")
    ax.set_ylabel("Rhetoric Score (pro-disarmament rhetoric)")
    ax.set_title(f"Rhetoric-Action Gap{title_suffix}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir, "rhetoric_action_scatter.png")
    return fig


def plot_gap_ranking(
    gap_df: pd.DataFrame,
    year: Optional[int] = None,
    top_n: int = 20,
    country_col: str = "country_code",
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of countries sorted by rhetoric-action gap."""
    if year is not None and year_col in gap_df.columns:
        df = gap_df[gap_df[year_col] == year].copy()
    else:
        df = gap_df.groupby(country_col)["gap"].mean().reset_index()

    if "gap" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("'gap' column not found")
        _savefig(fig, output_dir, "gap_ranking.png")
        return fig

    df = df.nlargest(top_n, "gap") if len(df) > top_n else df.sort_values("gap", ascending=False)
    df = df.sort_values("gap", ascending=True)

    colors = ["#d62728" if g > 0.1 else ("#2ca02c" if g < -0.1 else "#1f77b4")
              for g in df["gap"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.38)))
    bars = ax.barh(df[country_col], df["gap"], color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.3, color="red", linewidth=0.5, linestyle="--", alpha=0.5, label="Hypocrite threshold")
    ax.axvline(-0.3, color="green", linewidth=0.5, linestyle="--", alpha=0.5, label="Quiet actor threshold")
    ax.set_xlabel("Rhetoric-Action Gap")
    ax.set_title(f"Rhetoric-Action Gap Ranking (Top {top_n})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir, "gap_ranking.png")
    return fig


def plot_gap_time_series(
    gap_df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    country_col: str = "country_code",
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Time series of rhetoric-action gap for selected countries."""
    if countries is None:
        countries = ["USA", "RUS", "CHN", "FRA", "GBR", "DEU", "IND", "SAU"]

    if "gap" not in gap_df.columns or year_col not in gap_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Gap time series data not available")
        _savefig(fig, output_dir, "gap_time_series.png")
        return fig

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("tab10", len(countries))

    for iso3, color in zip(countries, palette):
        sub = gap_df[gap_df[country_col] == iso3].sort_values(year_col)
        if sub.empty:
            continue
        ax.plot(sub[year_col], sub["gap"], label=iso3, color=color, linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.3, color="red", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.axhline(-0.3, color="green", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Rhetoric-Action Gap")
    ax.set_title("Rhetoric-Action Gap Over Time (Major Arms Exporters)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    _savefig(fig, output_dir, "gap_time_series.png")
    return fig


def plot_gap_by_group(
    gap_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    country_col: str = "country_code",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Violin/box plots of gap distribution by country group."""
    if "gap" not in gap_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Gap column not found")
        _savefig(fig, output_dir, "gap_by_group.png")
        return fig

    rows = []
    for group_name, members in groups.items():
        sub = gap_df[gap_df[country_col].isin(members)]
        for val in sub["gap"].dropna():
            rows.append({"group": group_name, "gap": val})

    if not rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("No matching countries in groups")
        _savefig(fig, output_dir, "gap_by_group.png")
        return fig

    plot_df = pd.DataFrame(rows)
    order = (
        plot_df.groupby("group")["gap"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=plot_df, x="group", y="gap", order=order,
        palette="Set3", ax=ax, inner="box", cut=0,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.3, color="red", linewidth=0.5, linestyle=":", alpha=0.7, label="Hypocrite")
    ax.axhline(-0.3, color="green", linewidth=0.5, linestyle=":", alpha=0.7, label="Quiet actor")
    ax.set_xlabel("Country Group")
    ax.set_ylabel("Rhetoric-Action Gap")
    ax.set_title("Gap Distribution by Country Group")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir, "gap_by_group.png")
    return fig
