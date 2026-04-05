"""
Rhetoric-Action Gap Index computation.

The gap index measures the discrepancy between a country's stated arms control
rhetoric (high = pro-disarmament) and its actual behaviour (arms transfers,
voting patterns, treaty compliance).

Gap > 0 : rhetoric more progressive than action (potential hypocrisy)
Gap < 0 : action more progressive than rhetoric (quiet good actor)
Gap ≈ 0 : rhetoric and action aligned
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rhetoric composite
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "treaty_anchor_similarity": 0.30,
    "voting_score": 0.25,
    "humanitarian_topic_pct": 0.20,
    "commitment_strength": 0.15,
    "care_harm_loading": 0.10,
}


def compute_rhetoric_composite(
    anchor_scores: pd.DataFrame,
    voting_df: pd.DataFrame,
    topic_proportions: Optional[pd.DataFrame],
    commitment_df: pd.DataFrame,
    moral_df: pd.DataFrame,
    weights: Optional[dict] = None,
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute a composite rhetoric score per country-year.

    The composite is a weighted average of five sub-scores (all on 0-1 scale):
      1. treaty_anchor_similarity: mean similarity across all treaty anchors
      2. voting_score: composite UNGA voting rate on disarmament resolutions
      3. humanitarian_topic_pct: proportion of text in humanitarian topics
      4. commitment_strength: mean commitment score
      5. care_harm_loading: normalised care/harm MFT score

    Parameters
    ----------
    anchor_scores : DataFrame with country_code, year, and *_score columns
    voting_df : DataFrame with country_code, year, voting_composite
    topic_proportions : DataFrame with country_code, year, topic_* columns (or None)
    commitment_df : DataFrame with country_code, year, commitment_score
    moral_df : DataFrame with country_code, year, care_harm
    weights : dict of {component: weight}, defaults to _DEFAULT_WEIGHTS

    Returns
    -------
    DataFrame with columns: country_code, year, rhetoric_score, and sub-scores
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS

    # --- 1. Treaty anchor similarity ---
    anchor_score_cols = [c for c in anchor_scores.columns if c.endswith("_score")]
    if anchor_score_cols:
        anchor_scores = anchor_scores.copy()
        anchor_scores["mean_anchor_sim"] = anchor_scores[anchor_score_cols].mean(axis=1)
        # Min-max normalise per year to 0-1
        anchor_scores["treaty_anchor_similarity"] = (
            anchor_scores.groupby(year_col)["mean_anchor_sim"]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
        )
    else:
        anchor_scores = anchor_scores.copy()
        anchor_scores["treaty_anchor_similarity"] = 0.5

    base = anchor_scores[[country_col, year_col, "treaty_anchor_similarity"]].copy()

    # --- 2. Voting score ---
    if "voting_composite" in voting_df.columns:
        vote_sub = voting_df[[country_col, year_col, "voting_composite"]].rename(
            columns={"voting_composite": "voting_score"}
        )
    else:
        vote_sub = voting_df[[country_col, year_col]].copy()
        vote_sub["voting_score"] = 0.5
    base = base.merge(vote_sub, on=[country_col, year_col], how="left")
    base["voting_score"] = base["voting_score"].fillna(0.5)

    # --- 3. Humanitarian topic pct ---
    if topic_proportions is not None:
        hum_cols = [c for c in topic_proportions.columns if "humanitarian" in c.lower()]
        if hum_cols:
            topic_sub = topic_proportions[[country_col, year_col] + hum_cols].copy()
            topic_sub["humanitarian_topic_pct"] = topic_sub[hum_cols].sum(axis=1)
        else:
            topic_sub = topic_proportions[[country_col, year_col]].copy()
            topic_sub["humanitarian_topic_pct"] = 0.0
        base = base.merge(
            topic_sub[[country_col, year_col, "humanitarian_topic_pct"]],
            on=[country_col, year_col],
            how="left",
        )
    else:
        base["humanitarian_topic_pct"] = 0.0
    base["humanitarian_topic_pct"] = base["humanitarian_topic_pct"].fillna(0.0)

    # --- 4. Commitment strength ---
    comm_sub = commitment_df[[country_col, year_col, "commitment_score"]].copy()
    base = base.merge(comm_sub, on=[country_col, year_col], how="left")
    base["commitment_strength"] = base["commitment_score"].fillna(0.5)
    base = base.drop(columns=["commitment_score"], errors="ignore")

    # --- 5. Care/harm MFT loading ---
    if "care_harm" in moral_df.columns:
        moral_sub = moral_df[[country_col, year_col, "care_harm"]].copy()
        # Normalise care_harm to 0-1 per year
        moral_sub["care_harm_norm"] = (
            moral_sub.groupby(year_col)["care_harm"]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
        )
        base = base.merge(
            moral_sub[[country_col, year_col, "care_harm_norm"]],
            on=[country_col, year_col],
            how="left",
        )
        base["care_harm_loading"] = base["care_harm_norm"].fillna(0.5)
        base = base.drop(columns=["care_harm_norm"], errors="ignore")
    else:
        base["care_harm_loading"] = 0.5

    # --- Weighted composite ---
    base["rhetoric_score"] = (
        weights.get("treaty_anchor_similarity", 0.30) * base["treaty_anchor_similarity"]
        + weights.get("voting_score", 0.25) * base["voting_score"]
        + weights.get("humanitarian_topic_pct", 0.20) * base["humanitarian_topic_pct"]
        + weights.get("commitment_strength", 0.15) * base["commitment_strength"]
        + weights.get("care_harm_loading", 0.10) * base["care_harm_loading"]
    )

    # Normalise to 0-1
    min_r = base["rhetoric_score"].min()
    max_r = base["rhetoric_score"].max()
    base["rhetoric_score"] = (base["rhetoric_score"] - min_r) / (max_r - min_r + 1e-9)

    return base.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Action scores
# ---------------------------------------------------------------------------

def load_action_scores(
    network_pipeline_output_dir: str,
) -> pd.DataFrame:
    """
    Load action scores from the companion arms-trade-network pipeline.

    Looks for output/metrics/node_metrics.csv in the companion pipeline directory.
    Raises FileNotFoundError if the file is not found.

    Action scores (0-1): higher = more arms-transfer activity / conflict flow.
    Countries with high action scores but high rhetoric scores have large positive gaps.

    Parameters
    ----------
    network_pipeline_output_dir : str
        Path to the arms-trade-network pipeline's output/ directory.

    Returns
    -------
    DataFrame with columns: country_code, year, action_score,
                             autocracy_transfer_ratio, conflict_flow_ratio
    """
    metrics_path = Path(network_pipeline_output_dir) / "metrics" / "node_metrics.csv"
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            if {"country_code", "autocracy_transfer_ratio"}.issubset(df.columns):
                logger.info("Loaded action scores from %s", metrics_path)
                # Normalise to 0-1
                for col in ["autocracy_transfer_ratio", "conflict_flow_ratio"]:
                    if col in df.columns:
                        df[col] = (df[col] - df[col].min()) / (
                            df[col].max() - df[col].min() + 1e-9
                        )
                if "action_score" not in df.columns:
                    score_cols = [c for c in ["autocracy_transfer_ratio", "conflict_flow_ratio"] if c in df.columns]
                    df["action_score"] = df[score_cols].mean(axis=1)
                return df
        except Exception as exc:
            raise RuntimeError(f"Could not load node_metrics.csv: {exc}") from exc

    raise FileNotFoundError(
        f"Action scores (node_metrics.csv) not found.\n"
        f"  Run the arms-trade-network pipeline first, then point\n"
        f"  network_pipeline_output_dir at its output/ directory.\n"
        f"  Expected file: {Path(network_pipeline_output_dir or '<network_output>') / 'metrics' / 'node_metrics.csv'}"
    )


# ---------------------------------------------------------------------------
# Gap computation
# ---------------------------------------------------------------------------

def compute_gap(
    rhetoric_df: pd.DataFrame,
    action_df: pd.DataFrame,
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute rhetoric-action gap per country-year.

    gap = rhetoric_score - (1 - action_score)
          = rhetoric_score + action_score - 1

    Positive gap: country talks more progressively than it acts.
    Negative gap: country acts more progressively than it talks.

    Parameters
    ----------
    rhetoric_df : output of compute_rhetoric_composite
    action_df : output of load_action_scores

    Returns
    -------
    DataFrame with columns:
        country_code, year, rhetoric_score, action_score, gap, gap_percentile
    """
    merged = rhetoric_df[[country_col, year_col, "rhetoric_score"]].merge(
        action_df[[country_col, year_col, "action_score"]],
        on=[country_col, year_col],
        how="inner",
    )

    if merged.empty:
        logger.warning("No overlapping country-years between rhetoric and action data.")
        return pd.DataFrame(
            columns=[country_col, year_col, "rhetoric_score", "action_score", "gap", "gap_percentile"]
        )

    # Gap: high rhetoric + high action = positive gap (hypocrite)
    # Invert action_score so that 0=dovish, 1=hawkish transfers
    merged["gap"] = merged["rhetoric_score"] - (1.0 - merged["action_score"])

    # Percentile rank
    all_gaps = merged["gap"].values
    merged["gap_percentile"] = merged["gap"].apply(
        lambda g: percentileofscore(all_gaps, g, kind="rank")
    )

    logger.info(
        "Gap stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
        merged["gap"].mean(),
        merged["gap"].std(),
        merged["gap"].min(),
        merged["gap"].max(),
    )
    return merged.reset_index(drop=True)


def classify_gap(gap_value: float) -> str:
    """
    Classify a gap value into a category.

    Categories:
      'hypocrite'      : gap > 0.3  (high rhetoric, hawkish action)
      'quiet_good_actor': gap < -0.3 (low rhetoric, dovish action)
      'aligned'        : -0.3 ≤ gap ≤ 0.3
    """
    if gap_value > 0.3:
        return "hypocrite"
    if gap_value < -0.3:
        return "quiet_good_actor"
    return "aligned"
