"""
Sentiment analysis and Moral Foundations Theory scoring for arms control speeches.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Moral Foundations Theory keyword dictionary
# ---------------------------------------------------------------------------

MFT_KEYWORDS: Dict[str, List[str]] = {
    "care_harm": [
        "harm", "hurt", "care", "protect", "civilian", "suffering", "innocent",
        "victim", "wounded", "casualty", "casualties", "killed", "maimed",
        "children", "humanitarian", "vulnerable", "safeguard", "prevent",
        "mercy", "compassion", "alleviate", "relieve", "assistance",
        "protection", "shelter", "rescue", "survivor", "hibakusha",
        "pain", "trauma", "injury", "devastation", "catastrophic",
    ],
    "fairness": [
        "fair", "unfair", "equal", "rights", "justice", "discrimination",
        "inequality", "equitable", "balanced", "impartial", "objective",
        "non-discriminatory", "double standard", "selective", "universal",
        "obligation", "responsibility", "duty", "reciprocal", "proportionate",
        "legitimate", "unjust", "prejudice", "bias", "consistent",
    ],
    "loyalty": [
        "ally", "allies", "alliance", "betrayal", "solidarity", "cooperation",
        "commitment", "faithful", "collective", "together", "united",
        "coalition", "partner", "partnership", "mutual", "shared",
        "community", "bond", "trust", "cohesion", "unity", "bloc",
    ],
    "authority": [
        "sovereignty", "law", "order", "legitimate", "authority", "govern",
        "state", "mandate", "jurisdiction", "treaty", "obligation", "binding",
        "enforcement", "compliance", "rule", "norm", "charter", "resolution",
        "multilateral", "international order", "regime", "framework",
        "institution", "mechanism", "protocol", "convention",
    ],
    "sanctity": [
        "sacred", "pure", "degrade", "taboo", "sanctity", "dignity",
        "honor", "honour", "moral", "conscience", "humanity", "repugnant",
        "unacceptable", "abhorrent", "barbaric", "civilized", "civilization",
        "stigma", "stigmatize", "profane", "desecrate", "violate",
        "obscene", "outrage", "inhumane", "atrocity", "war crime",
    ],
}


def _score_mft_text(text: str) -> Dict[str, float]:
    """Score a single text on each MFT dimension using keyword counting."""
    if not isinstance(text, str) or not text:
        return {dim: 0.0 for dim in MFT_KEYWORDS}
    lower = text.lower()
    words = lower.split()
    n_words = max(len(words), 1)
    scores = {}
    for dim, keywords in MFT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lower)
        scores[dim] = count / n_words * 100  # per-hundred-words rate
    return scores


def compute_vader_sentiment(
    segments_df: pd.DataFrame,
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute VADER sentiment scores per country-year.

    If vaderSentiment is not installed, uses a simple positive/negative
    keyword heuristic as fallback.

    Returns
    -------
    DataFrame with columns:
        country_code, year, compound, pos, neg, neu, n_segments
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        def _score(text):
            s = sia.polarity_scores(text)
            return s["compound"], s["pos"], s["neg"], s["neu"]

    except ImportError:
        logger.warning("vaderSentiment not installed; using keyword heuristic.")
        _POS = {"support", "cooperation", "progress", "peace", "security", "commitment", "protect"}
        _NEG = {"reject", "condemn", "violation", "threat", "dangerous", "unacceptable", "failure"}

        def _score(text):
            lower = text.lower().split()
            pos = sum(1 for w in lower if w in _POS) / max(len(lower), 1)
            neg = sum(1 for w in lower if w in _NEG) / max(len(lower), 1)
            compound = pos - neg
            neu = 1.0 - pos - neg
            return compound, pos, neg, max(neu, 0.0)

    rows = []
    for _, row in segments_df.iterrows():
        compound, pos, neg, neu = _score(str(row.get(text_col, "")))
        rows.append(
            {
                country_col: row.get(country_col, "UNK"),
                year_col: row.get(year_col, 0),
                "compound": compound,
                "pos": pos,
                "neg": neg,
                "neu": neu,
            }
        )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=[country_col, year_col, "compound", "pos", "neg", "neu", "n_segments"])

    group_cols = [c for c in [country_col, year_col] if c in raw.columns]
    agg = raw.groupby(group_cols).agg(
        compound=("compound", "mean"),
        pos=("pos", "mean"),
        neg=("neg", "mean"),
        neu=("neu", "mean"),
        n_segments=(country_col, "count"),
    ).reset_index()
    return agg


def compute_moral_foundations(
    segments_df: pd.DataFrame,
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute MFT dimension scores per country-year using keyword counting.

    Returns
    -------
    DataFrame with columns:
        country_code, year, care_harm, fairness, loyalty, authority, sanctity, n_segments
    """
    dim_names = list(MFT_KEYWORDS.keys())
    rows = []
    for _, row in segments_df.iterrows():
        text = str(row.get(text_col, ""))
        scores = _score_mft_text(text)
        entry = {
            country_col: row.get(country_col, "UNK"),
            year_col: row.get(year_col, 0),
        }
        entry.update(scores)
        rows.append(entry)

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=[country_col, year_col] + dim_names + ["n_segments"])

    group_cols = [c for c in [country_col, year_col] if c in raw.columns]
    agg_dict = {dim: (dim, "mean") for dim in dim_names}
    agg_dict["n_segments"] = (country_col, "count")
    agg = raw.groupby(group_cols).agg(**agg_dict).reset_index()
    return agg


def aggregate_sentiment_by_group(
    sentiment_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Aggregate sentiment/MFT scores by country group.

    Parameters
    ----------
    sentiment_df : output of compute_vader_sentiment or compute_moral_foundations
    groups : dict mapping group_name → list of ISO3 codes

    Returns
    -------
    DataFrame with additional 'group' column, averaged over group members
    """
    numeric_cols = sentiment_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != year_col]

    rows = []
    for group_name, members in groups.items():
        sub = sentiment_df[sentiment_df[country_col].isin(members)]
        if sub.empty:
            continue
        if year_col in sub.columns:
            grp = sub.groupby(year_col)[numeric_cols].mean().reset_index()
        else:
            grp = sub[numeric_cols].mean().to_frame().T
        grp["group"] = group_name
        rows.append(grp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
