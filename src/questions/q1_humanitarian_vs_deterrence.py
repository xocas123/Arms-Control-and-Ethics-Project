"""
Q1: Has "humanitarian" replaced "deterrence" as the dominant frame in arms control discourse?

Analyses:
1a. Global frame ratio time series
1b. Rate of change + change-point detection
1c. Frame ratio by country group over time
1d. Geographic diffusion
1e. Variance analysis (convergence vs polarization)
1f. Topic-level temporal tracking
1g. Voting validation
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from src.data.groups import (
    NWS, NUCLEAR_UMBRELLA, NAM, NATO, EU, REGIONS,
    get_region, get_nuclear_status,
)
from src.shared.lexicons import HUMANITARIAN, DETERRENCE
from src.shared.temporal import (
    rolling_mean, rolling_std, year_over_year_delta,
    compute_change_points, compute_group_time_series,
)
from src.shared.frame_scoring import classify_vote_resolution_frame

OUTPUT_DIR = "output/q1"
NNWS_LABEL = "NNWS"


def _get_group(iso3: str, vdem_df: pd.DataFrame = None) -> str:
    """Assign country to a display group for Q1 analysis."""
    if iso3 in NWS:
        return "NWS"
    if iso3 in NATO:
        return "NATO"
    if iso3 in EU:
        return "EU"
    if iso3 in NAM:
        return "NAM"
    return NNWS_LABEL


def run_q1(data: dict, config: dict = None) -> dict:
    """
    Main Q1 entry point.

    Args:
        data: pipeline data dict with keys: corpus, frame_scores, voting, vdem,
              topics, probs, topic_classifications, country_year_embeddings, anchors

    Returns:
        dict of DataFrames (also saved to output/q1/)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    frame_scores = data.get("frame_scores", pd.DataFrame())
    voting = data.get("voting", pd.DataFrame())
    vdem = data.get("vdem", pd.DataFrame())
    topics_over_time_df = data.get("topics_over_time_df", pd.DataFrame())
    topic_classifications = data.get("topic_classifications", pd.DataFrame())

    results = {}

    # 1a. Global frame ratio time series
    print("[Q1] 1a: Computing global frame time series...")
    global_ts = compute_global_frame_timeseries(frame_scores)
    results["frame_ratio_global"] = global_ts
    _save(global_ts, "frame_ratio_global.csv")

    # 1b. Rate of change + change points
    print("[Q1] 1b: Change-point detection...")
    yoy = compute_yoy_delta(global_ts)
    results["yoy_delta"] = yoy
    _save(yoy, "yoy_delta.csv")

    change_points = compute_change_points_q1(global_ts)
    results["change_points"] = change_points
    _save(change_points, "change_points.csv")

    # 1c. By group
    print("[Q1] 1c: Frame ratio by group...")
    by_group = compute_frame_by_group(frame_scores, vdem)
    results["frame_ratio_by_group"] = by_group
    _save(by_group, "frame_ratio_by_group.csv")

    # 1d. Geographic diffusion
    print("[Q1] 1d: Geographic diffusion...")
    by_region = compute_frame_by_region(frame_scores)
    results["frame_ratio_by_region"] = by_region
    _save(by_region, "frame_ratio_by_region.csv")

    # 1e. Variance analysis
    print("[Q1] 1e: Variance analysis...")
    # Already in global_ts (frame_ratio_std), but compute separately for clarity
    results["variance_ts"] = global_ts  # reuse

    # 1f. Topic prevalence
    if not topics_over_time_df.empty and not topic_classifications.empty:
        print("[Q1] 1f: Topic prevalence over time...")
        topic_prev = compute_topic_prevalence(topics_over_time_df, topic_classifications)
        results["topic_prevalence_over_time"] = topic_prev
        _save(topic_prev, "topic_prevalence_over_time.csv")

    # 1g. Voting validation
    if not voting.empty:
        print("[Q1] 1g: Voting validation...")
        vote_corr = compute_vote_frame_correlation(voting, frame_scores)
        results["vote_frame_correlation"] = vote_corr
        _save(vote_corr, "vote_frame_correlation.csv")

    print("[Q1] Complete.")
    return results


def compute_global_frame_timeseries(frame_scores: pd.DataFrame) -> pd.DataFrame:
    """1a: Aggregate frame scores globally by year."""
    if frame_scores.empty:
        return pd.DataFrame()

    col_ratio = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"
    col_pos = "frame_position_mean" if "frame_position_mean" in frame_scores.columns else None

    agg = (
        frame_scores.groupby("year")
        .agg(
            frame_ratio_mean=(col_ratio, "mean"),
            frame_ratio_std=(col_ratio, "std"),
            n_countries=(col_ratio, "count"),
        )
        .reset_index()
    )

    if col_pos:
        pos_agg = frame_scores.groupby("year")[col_pos].mean().reset_index()
        pos_agg.columns = ["year", "frame_position_mean"]
        agg = agg.merge(pos_agg, on="year", how="left")

    agg = agg.sort_values("year")
    agg["rolling_mean_5yr"] = rolling_mean(pd.Series(agg["frame_ratio_mean"].values, index=agg["year"])).values
    agg["rolling_std_5yr"] = rolling_std(pd.Series(agg["frame_ratio_std"].values, index=agg["year"])).values
    return agg


def compute_change_points_q1(global_ts: pd.DataFrame) -> pd.DataFrame:
    """1b: PELT change-point detection on frame_ratio time series."""
    if global_ts.empty:
        return pd.DataFrame()

    series = pd.Series(
        global_ts["frame_ratio_mean"].values,
        index=global_ts["year"].values,
    )
    breaks = compute_change_points(series)

    records = []
    for year, magnitude in breaks:
        records.append({
            "break_year": year,
            "magnitude": magnitude,
            "direction": "toward_humanitarian" if magnitude > 0 else "toward_deterrence",
        })
    return pd.DataFrame(records)


def compute_yoy_delta(global_ts: pd.DataFrame) -> pd.DataFrame:
    """1b: Year-over-year delta of global frame_ratio."""
    if global_ts.empty:
        return pd.DataFrame()

    series = pd.Series(global_ts["frame_ratio_mean"].values, index=global_ts["year"].values)
    delta = year_over_year_delta(series)
    result = pd.DataFrame({"year": delta.index, "delta": delta.values})
    result["rolling_delta_5yr"] = rolling_mean(delta, window=5).values
    return result


def compute_frame_by_group(frame_scores: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """1c: Frame ratio per display group per year."""
    if frame_scores.empty:
        return pd.DataFrame()

    fs = frame_scores.copy()
    fs["group"] = fs["country_iso3"].apply(_get_group)

    col = "frame_ratio_mean" if "frame_ratio_mean" in fs.columns else "frame_ratio"
    agg = (
        fs.groupby(["year", "group"])
        .agg(
            frame_ratio_mean=(col, "mean"),
            frame_ratio_std=(col, "std"),
            n_countries=(col, "count"),
        )
        .reset_index()
    )
    return agg


def compute_frame_by_region(frame_scores: pd.DataFrame) -> pd.DataFrame:
    """1d: Frame ratio per region per year."""
    if frame_scores.empty:
        return pd.DataFrame()

    fs = frame_scores.copy()
    fs["region"] = fs["country_iso3"].apply(get_region)

    col = "frame_ratio_mean" if "frame_ratio_mean" in fs.columns else "frame_ratio"
    agg = (
        fs[fs["region"] != "other"]
        .groupby(["year", "region"])
        .agg(frame_ratio_mean=(col, "mean"), n_countries=(col, "count"))
        .reset_index()
    )
    return agg


def compute_topic_prevalence(
    topics_over_time_df: pd.DataFrame,
    topic_classifications: pd.DataFrame,
) -> pd.DataFrame:
    """1f: Aggregate topic prevalence by humanitarian/deterrence/other classification."""
    if topics_over_time_df.empty or topic_classifications.empty:
        return pd.DataFrame()

    topic_col = "Topic" if "Topic" in topics_over_time_df.columns else "topic"
    time_col = "Timestamp" if "Timestamp" in topics_over_time_df.columns else "timestamp"
    freq_col = "Frequency" if "Frequency" in topics_over_time_df.columns else "frequency"

    merged = topics_over_time_df.merge(
        topic_classifications[["topic_id", "classification"]],
        left_on=topic_col, right_on="topic_id", how="left",
    )
    merged["classification"] = merged["classification"].fillna("other")

    agg = (
        merged.groupby([time_col, "classification"])[freq_col]
        .sum()
        .reset_index()
        .rename(columns={time_col: "year", freq_col: "prevalence"})
    )
    return agg


def compute_vote_frame_correlation(
    voting: pd.DataFrame,
    frame_scores: pd.DataFrame,
) -> pd.DataFrame:
    """1g: Per-year Pearson r between speech frame_ratio and humanitarian voting rate."""
    if voting.empty or frame_scores.empty:
        return pd.DataFrame()

    # Voting: pct yes on humanitarian-framed resolutions per country-year
    h_votes = voting[voting.get("frame_type", pd.Series()) == "humanitarian"] if "frame_type" in voting.columns else voting
    if h_votes.empty:
        return pd.DataFrame()

    vote_rate = (
        h_votes[h_votes["vote_numeric"].notna()]
        .groupby(["country_iso3", "year"])["vote_numeric"]
        .apply(lambda x: (x == 1).mean())
        .reset_index()
        .rename(columns={"vote_numeric": "h_vote_rate"})
    )

    col = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"
    merged = frame_scores[["country_iso3", "year", col]].merge(
        vote_rate, on=["country_iso3", "year"], how="inner"
    )

    records = []
    for year, ydf in merged.groupby("year"):
        if len(ydf) < 5:
            continue
        r, p = stats.pearsonr(ydf[col], ydf["h_vote_rate"])
        records.append({"year": year, "correlation": r, "p_value": p, "n_countries": len(ydf)})

    return pd.DataFrame(records)


def _save(df: pd.DataFrame, filename: str):
    """Save DataFrame to output/q1/."""
    if df is not None and not df.empty:
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"[Q1] Saved {path}")
