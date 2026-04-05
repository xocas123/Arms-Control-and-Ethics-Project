"""
Position space visualizations (rhetorical scaling).
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_FIG_DPI = 150
_REGIME_PALETTE = {
    "p5_democracy": "#1f77b4",
    "p5_authoritarian": "#d62728",
    "de_facto_nuclear": "#ff7f0e",
    "humanitarian_coalition": "#2ca02c",
    "nato_eu_democracy": "#9467bd",
    "gulf_autocracy": "#8c564b",
    "nam_state": "#e377c2",
    "other": "#7f7f7f",
}


def _savefig(fig, output_dir: Optional[str], filename: str) -> None:
    if output_dir is not None:
        out = Path(output_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_country_positions_2d(
    positions_df: pd.DataFrame,
    year: Optional[int] = None,
    regime_df: Optional[pd.DataFrame] = None,
    country_col: str = "country_code",
    year_col: str = "year",
    output_dir: Optional[str] = None,
    annotate_n: int = 20,
) -> plt.Figure:
    """
    2D scatter of country rhetorical positions, optionally colour-coded by regime type.
    """
    if year is not None and year_col in positions_df.columns:
        df = positions_df[positions_df[year_col] == year].copy()
        title_suffix = f" ({year})"
    else:
        df = positions_df.copy()
        title_suffix = " (all years)"

    x_col = "position_1" if "position_1" in df.columns else (
        "pc_1" if "pc_1" in df.columns else None
    )
    y_col = "position_2" if "position_2" in df.columns else (
        "pc_2" if "pc_2" in df.columns else None
    )

    if x_col is None or y_col is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Position columns not found")
        _savefig(fig, output_dir, "country_positions_2d.png")
        return fig

    if regime_df is not None and "regime_type" in regime_df.columns:
        df = df.merge(regime_df[[country_col, "regime_type"]], on=country_col, how="left")
        df["regime_type"] = df["regime_type"].fillna("other")
        color_col = "regime_type"
    else:
        df["regime_type"] = "other"
        color_col = "regime_type"

    fig, ax = plt.subplots(figsize=(12, 9))
    for regime, grp in df.groupby(color_col):
        color = _REGIME_PALETTE.get(regime, "#aaaaaa")
        ax.scatter(grp[x_col], grp[y_col], label=regime, color=color, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Annotate top-N countries (most extreme positions)
    extreme = df.copy()
    extreme["dist"] = np.sqrt(extreme[x_col] ** 2 + extreme[y_col] ** 2)
    extreme = extreme.nlargest(annotate_n, "dist")
    for _, row in extreme.iterrows():
        ax.annotate(
            row[country_col],
            (row[x_col], row[y_col]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Rhetorical Position Dimension 1")
    ax.set_ylabel("Rhetorical Position Dimension 2")
    ax.set_title(f"Country Rhetorical Positions{title_suffix}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    _savefig(fig, output_dir, "country_positions_2d.png")
    return fig


def plot_position_drift(
    positions_df: pd.DataFrame,
    countries: Optional[List[str]] = None,
    country_col: str = "country_code",
    year_col: str = "year",
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """Line chart of rhetorical position dimension 1 over time for selected countries."""
    if countries is None:
        countries = ["USA", "RUS", "CHN", "GBR", "FRA", "AUT", "IRL", "IND", "PAK", "ZAF"]

    pos_col = "position_1" if "position_1" in positions_df.columns else (
        "pc_1" if "pc_1" in positions_df.columns else None
    )
    if pos_col is None or year_col not in positions_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Position data not available")
        _savefig(fig, output_dir, "position_drift.png")
        return fig

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("tab20", len(countries))

    for iso3, color in zip(countries, palette):
        sub = positions_df[positions_df[country_col] == iso3].sort_values(year_col)
        if sub.empty:
            continue
        ax.plot(sub[year_col], sub[pos_col], label=iso3, color=color, linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Rhetorical Position (Dim 1)")
    ax.set_title("Rhetorical Position Drift Over Time")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    _savefig(fig, output_dir, "position_drift.png")
    return fig
