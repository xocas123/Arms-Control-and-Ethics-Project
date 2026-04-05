"""
UNGA voting data analysis for arms control pipeline.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_and_process_voting(
    data_dir: Optional[str] = None,
    year_start: int = 2000,
    year_end: int = 2023,
) -> pd.DataFrame:
    """
    Load voting data and compute derived metrics.

    Returns
    -------
    DataFrame with additional column 'voting_composite'
    """
    from src.data.load_voting import load_voting

    df = load_voting(data_dir=data_dir)
    if year_start or year_end:
        if "year" in df.columns:
            df = df[(df["year"] >= year_start) & (df["year"] <= year_end)]

    # Composite voting score: weighted average of disarmament + nuclear vote rates
    df["voting_composite"] = 0.6 * df["pct_yes_disarmament"] + 0.4 * df["pct_yes_nuclear"]

    return df


def compute_voting_blocs(
    voting_df: pd.DataFrame,
    year: Optional[int] = None,
    n_clusters: int = 5,
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Cluster countries into voting blocs based on voting similarity.

    Parameters
    ----------
    voting_df : output of load_and_process_voting
    year : int or None (if None, use all years averaged)
    n_clusters : int

    Returns
    -------
    DataFrame with columns: country_code, cluster, cluster_label
    """
    feature_cols = ["pct_yes_disarmament", "pct_yes_nuclear", "voting_composite"]
    feature_cols = [c for c in feature_cols if c in voting_df.columns]

    if year is not None:
        df_yr = voting_df[voting_df[year_col] == year]
    else:
        df_yr = voting_df.groupby(country_col)[feature_cols].mean().reset_index()

    if df_yr.empty:
        return pd.DataFrame(columns=[country_col, "cluster", "cluster_label"])

    X = df_yr[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_clusters, len(df_yr))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    result = df_yr[[country_col]].copy().reset_index(drop=True)
    result["cluster"] = labels

    # Label clusters by mean disarmament voting rate
    cluster_means = (
        result.merge(df_yr[[country_col] + feature_cols], on=country_col)
        .groupby("cluster")["pct_yes_disarmament"]
        .mean()
        .sort_values()
    )
    label_map = {
        old: f"bloc_{rank+1}_{'high' if rank >= n_clusters - 2 else 'mid' if rank >= 1 else 'low'}"
        for rank, old in enumerate(cluster_means.index)
    }
    result["cluster_label"] = result["cluster"].map(label_map)

    return result


def compute_swing_states(
    voting_df: pd.DataFrame,
    country_col: str = "country_code",
    year_col: str = "year",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Identify countries with highest year-over-year volatility in disarmament voting.

    Returns
    -------
    DataFrame with columns: country_code, avg_yoy_change, std_yoy_change
    Sorted by avg_yoy_change descending.
    """
    df = voting_df.sort_values([country_col, year_col])
    df["yoy_change"] = df.groupby(country_col)["pct_yes_disarmament"].diff().abs()

    swing = (
        df.groupby(country_col)["yoy_change"]
        .agg(avg_yoy_change="mean", std_yoy_change="std")
        .reset_index()
        .sort_values("avg_yoy_change", ascending=False)
        .head(top_n)
    )
    return swing


def voting_vs_exports(
    voting_df: pd.DataFrame,
    network_metrics_df: pd.DataFrame,
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Merge voting data with network/trade metrics for correlation analysis.

    Parameters
    ----------
    voting_df : output of load_and_process_voting
    network_metrics_df : DataFrame with country_code and trade/network metrics

    Returns
    -------
    Merged DataFrame
    """
    # Average voting over years
    vote_agg = voting_df.groupby(country_col).agg(
        pct_yes_disarmament=("pct_yes_disarmament", "mean"),
        pct_yes_nuclear=("pct_yes_nuclear", "mean"),
        voting_composite=("voting_composite", "mean"),
    ).reset_index()

    merged = vote_agg.merge(network_metrics_df, on=country_col, how="inner")
    return merged
