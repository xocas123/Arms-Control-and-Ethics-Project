"""
Commitment strength scoring for arms control speech segments.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numeric encoding of commitment levels
_STRENGTH_SCORES = {
    "strong": 1.0,
    "moderate": 0.65,
    "weak": 0.35,
    "conditional": 0.40,
    "opposition": 0.0,
}

_DEFAULT_SCORE = 0.50  # no commitment phrase found → neutral


def classify_commitment(text: str) -> str:
    """
    Classify the commitment level of a text snippet.

    Returns one of: 'strong', 'moderate', 'weak', 'opposition', 'conditional', 'neutral'
    """
    from src.analysis.ner_extraction import (
        _STRONG_PATTERNS,
        _MODERATE_PATTERNS,
        _WEAK_PATTERNS,
        _OPPOSITION_PATTERNS,
        _CONDITIONAL_PATTERNS,
    )

    if _STRONG_PATTERNS.search(text):
        return "strong"
    if _OPPOSITION_PATTERNS.search(text):
        return "opposition"
    if _CONDITIONAL_PATTERNS.search(text):
        return "conditional"
    if _MODERATE_PATTERNS.search(text):
        return "moderate"
    if _WEAK_PATTERNS.search(text):
        return "weak"
    return "neutral"


def score_commitment_strength(
    segments_df: pd.DataFrame,
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Score commitment strength per country-year.

    For each segment, classify all sentences; average numeric score over
    classified sentences (ignoring neutral). If no classified sentence
    exists, assign default score 0.50.

    Parameters
    ----------
    segments_df : DataFrame with text_col, country_col, year_col

    Returns
    -------
    DataFrame with columns: country_code, year, commitment_score, n_segments,
                             pct_strong, pct_moderate, pct_weak, pct_opposition
    """
    from src.analysis.ner_extraction import extract_commitment_phrases

    rows = []
    for _, row in segments_df.iterrows():
        text = str(row.get(text_col, ""))
        iso3 = row.get(country_col, "UNK")
        year = row.get(year_col, 0)

        phrases = extract_commitment_phrases(text)
        if not phrases:
            score = _DEFAULT_SCORE
            counts = {k: 0 for k in _STRENGTH_SCORES}
        else:
            scores = [_STRENGTH_SCORES[lbl] for _, lbl in phrases]
            score = float(np.mean(scores))
            counts = {k: sum(1 for _, lbl in phrases if lbl == k) for k in _STRENGTH_SCORES}

        n = max(len(phrases), 1)
        rows.append(
            {
                country_col: iso3,
                year_col: year,
                "commitment_score": score,
                "pct_strong": counts.get("strong", 0) / n,
                "pct_moderate": counts.get("moderate", 0) / n,
                "pct_weak": counts.get("weak", 0) / n,
                "pct_opposition": counts.get("opposition", 0) / n,
            }
        )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(
            columns=[country_col, year_col, "commitment_score", "n_segments",
                     "pct_strong", "pct_moderate", "pct_weak", "pct_opposition"]
        )

    group_cols = [c for c in [country_col, year_col] if c in raw.columns]
    agg = raw.groupby(group_cols).agg(
        commitment_score=("commitment_score", "mean"),
        n_segments=(country_col, "count"),
        pct_strong=("pct_strong", "mean"),
        pct_moderate=("pct_moderate", "mean"),
        pct_weak=("pct_weak", "mean"),
        pct_opposition=("pct_opposition", "mean"),
    ).reset_index()
    return agg
