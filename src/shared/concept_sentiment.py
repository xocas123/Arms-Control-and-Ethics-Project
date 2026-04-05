"""
Concept-level sentiment analysis.
Extracts windows around concept mentions and scores sentiment within those windows only.
Used by Q4 (nuclear vs non-nuclear state rhetoric).
"""
import re
import numpy as np
import pandas as pd
from typing import List, Optional

from src.shared.lexicons import CONCEPTS


def _get_vader():
    """Load VADER sentiment analyzer. Returns None if unavailable."""
    try:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def _score_sentiment_simple(text: str) -> dict:
    """Simple positive/negative word count fallback when VADER unavailable."""
    pos_words = [
        "support", "welcome", "commend", "progress", "positive", "effective",
        "success", "cooperation", "commitment", "important", "necessary",
        "valuable", "significant", "constructive", "promote", "achieve",
    ]
    neg_words = [
        "oppose", "reject", "concern", "failure", "inadequate", "insufficient",
        "threat", "danger", "risk", "undermine", "violate", "problematic",
        "unacceptable", "dangerous", "irresponsible", "destabilizing",
    ]
    text_lower = text.lower()
    pos = sum(1 for w in pos_words if w in text_lower)
    neg = sum(1 for w in neg_words if w in text_lower)
    total = max(pos + neg, 1)
    compound = (pos - neg) / total
    return {
        "compound": compound,
        "pos": pos / total,
        "neg": neg / total,
        "neu": 1.0 - abs(compound),
    }


def extract_concept_windows(
    text: str,
    concept: str,
    window_sentences: int = 3,
) -> List[str]:
    """
    Find all mentions of concept in text.
    Return list of text windows (±window_sentences around each mention).
    """
    if not text or not concept:
        return []

    sentences = _split_sentences(text)
    windows = []
    concept_lower = concept.lower()

    for i, sent in enumerate(sentences):
        if concept_lower in sent.lower():
            start = max(0, i - window_sentences)
            end = min(len(sentences), i + window_sentences + 1)
            window_text = " ".join(sentences[start:end])
            windows.append(window_text)

    return windows


def score_concept_sentiment(
    text: str,
    concept: str,
    window_sentences: int = 3,
) -> dict:
    """
    Score sentiment for mentions of a specific concept in text.

    Returns:
        concept: str
        n_mentions: int
        mean_sentiment: float — VADER compound score (or simple fallback), averaged over windows
        positive_ratio: float
        negative_ratio: float
        windows: list[str]
    """
    windows = extract_concept_windows(text, concept, window_sentences)

    if not windows:
        return {
            "concept": concept,
            "n_mentions": 0,
            "mean_sentiment": np.nan,
            "positive_ratio": np.nan,
            "negative_ratio": np.nan,
            "windows": [],
        }

    vader = _get_vader()
    scores = []

    for window in windows:
        if vader:
            s = vader.polarity_scores(window)
        else:
            s = _score_sentiment_simple(window)
        scores.append(s)

    mean_compound = np.mean([s["compound"] for s in scores])
    mean_pos = np.mean([s["pos"] for s in scores])
    mean_neg = np.mean([s["neg"] for s in scores])

    return {
        "concept": concept,
        "n_mentions": len(windows),
        "mean_sentiment": float(mean_compound),
        "positive_ratio": float(mean_pos),
        "negative_ratio": float(mean_neg),
        "windows": windows,
    }


def score_all_concepts(
    text: str,
    concepts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Score sentiment for all concepts.

    Returns DataFrame: concept, n_mentions, mean_sentiment, positive_ratio, negative_ratio
    """
    if concepts is None:
        concepts = CONCEPTS

    results = []
    for concept in concepts:
        result = score_concept_sentiment(text, concept)
        results.append({
            "concept": result["concept"],
            "n_mentions": result["n_mentions"],
            "mean_sentiment": result["mean_sentiment"],
            "positive_ratio": result["positive_ratio"],
            "negative_ratio": result["negative_ratio"],
        })

    return pd.DataFrame(results)


def score_corpus_concept_sentiment(
    df: pd.DataFrame,
    text_col: str = "segment_text",
    concepts: Optional[List[str]] = None,
    country_col: str = "country_iso3",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Score concept sentiment for all documents in corpus.

    Returns long-form DataFrame:
        country_iso3, year, concept, n_mentions, mean_sentiment, positive_ratio, negative_ratio
    """
    if concepts is None:
        concepts = CONCEPTS

    records = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            print(f"[ConceptSentiment] Processing {i}/{total} documents...")

        text = str(row.get(text_col, "") or "")
        country = row.get(country_col, "")
        year = row.get(year_col, np.nan)

        for concept in concepts:
            result = score_concept_sentiment(text, concept)
            if result["n_mentions"] > 0:
                records.append({
                    "country_iso3": country,
                    "year": year,
                    "concept": concept,
                    "n_mentions": result["n_mentions"],
                    "mean_sentiment": result["mean_sentiment"],
                    "positive_ratio": result["positive_ratio"],
                    "negative_ratio": result["negative_ratio"],
                })

    if not records:
        return pd.DataFrame(columns=[
            "country_iso3", "year", "concept",
            "n_mentions", "mean_sentiment", "positive_ratio", "negative_ratio"
        ])

    return pd.DataFrame(records)
