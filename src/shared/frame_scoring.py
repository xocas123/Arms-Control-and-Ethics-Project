"""
Frame scoring — lexicon-based and embedding-based methods.
Both run on every speech segment and should correlate.
"""
import numpy as np
import pandas as pd
from typing import Optional

from src.shared.lexicons import (
    HUMANITARIAN, DETERRENCE, count_matches, compute_frame_ratio_lexicon
)
from src.shared.embeddings import cosine_sim


def score_frame_lexicon(text: str) -> dict:
    """
    Lexicon-based frame scoring.

    Returns:
        h_count: int — humanitarian keyword matches
        d_count: int — deterrence keyword matches
        frame_ratio: float — h / (h + d), 0=pure deterrence, 1=pure humanitarian
        h_density: float — h_count / word_count
        d_density: float — d_count / word_count
    """
    if not text:
        return {
            "h_count": 0,
            "d_count": 0,
            "frame_ratio": 0.5,
            "h_density": 0.0,
            "d_density": 0.0,
        }

    h_count = count_matches(text, HUMANITARIAN)
    d_count = count_matches(text, DETERRENCE)
    word_count = max(len(text.split()), 1)

    return {
        "h_count": h_count,
        "d_count": d_count,
        "frame_ratio": h_count / max(h_count + d_count, 1),
        "h_density": h_count / word_count,
        "d_density": d_count / word_count,
    }


def score_frame_embedding(
    speech_embedding: np.ndarray,
    h_anchor: np.ndarray,
    d_anchor: np.ndarray,
) -> dict:
    """
    Embedding-based frame scoring.

    Returns:
        h_similarity: float — cosine similarity to humanitarian anchor
        d_similarity: float — cosine similarity to deterrence anchor
        frame_position: float — h_sim - d_sim (positive = humanitarian)
    """
    h_sim = cosine_sim(speech_embedding, h_anchor)
    d_sim = cosine_sim(speech_embedding, d_anchor)

    return {
        "h_similarity": float(h_sim),
        "d_similarity": float(d_sim),
        "frame_position": float(h_sim - d_sim),
    }


def score_corpus_frames(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    index_df: Optional[pd.DataFrame],
    h_anchor: Optional[np.ndarray],
    d_anchor: Optional[np.ndarray],
    text_col: str = "segment_text",
) -> pd.DataFrame:
    """
    Score all documents in corpus with both lexicon and embedding methods.

    Returns original DataFrame with added columns:
        h_count, d_count, frame_ratio, h_density, d_density (lexicon)
        h_similarity, d_similarity, frame_position (embedding, if available)
    """
    result = df.copy()

    # Lexicon scoring (always available)
    print("[FrameScoring] Computing lexicon-based frame scores...")
    lexicon_scores = result[text_col].fillna("").apply(score_frame_lexicon)
    for col in ["h_count", "d_count", "frame_ratio", "h_density", "d_density"]:
        result[col] = [s[col] for s in lexicon_scores]

    # Embedding scoring (optional)
    if embeddings is not None and index_df is not None and h_anchor is not None:
        print("[FrameScoring] Computing embedding-based frame scores...")
        result["h_similarity"] = np.nan
        result["d_similarity"] = np.nan
        result["frame_position"] = np.nan

        idx_reset = index_df.reset_index(drop=True)
        for i, row_idx in enumerate(idx_reset.index):
            if i < len(embeddings):
                emb = embeddings[i]
                scores = score_frame_embedding(emb, h_anchor, d_anchor)
                result.loc[row_idx, "h_similarity"] = scores["h_similarity"]
                result.loc[row_idx, "d_similarity"] = scores["d_similarity"]
                result.loc[row_idx, "frame_position"] = scores["frame_position"]
    else:
        result["h_similarity"] = np.nan
        result["d_similarity"] = np.nan
        result["frame_position"] = np.nan

    return result


def aggregate_to_country_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate document-level frame scores to country-year level.

    Returns DataFrame: country_iso3, year,
        frame_ratio_mean, frame_ratio_std, frame_ratio_median,
        frame_position_mean, frame_position_std,
        h_count_sum, d_count_sum,
        n_documents
    """
    group_cols = ["country_iso3", "year"]
    agg = (
        df.groupby(group_cols)
        .agg(
            frame_ratio_mean=("frame_ratio", "mean"),
            frame_ratio_std=("frame_ratio", "std"),
            frame_ratio_median=("frame_ratio", "median"),
            frame_position_mean=("frame_position", "mean"),
            frame_position_std=("frame_position", "std"),
            h_count_sum=("h_count", "sum"),
            d_count_sum=("d_count", "sum"),
            n_documents=("frame_ratio", "count"),
        )
        .reset_index()
    )
    agg["frame_ratio_std"] = agg["frame_ratio_std"].fillna(0)
    agg["frame_position_std"] = agg["frame_position_std"].fillna(0)
    return agg


def classify_vote_resolution_frame(
    title: str,
    full_text: Optional[str] = None,
) -> str:
    """
    Classify a UN resolution as 'humanitarian', 'security', or 'mixed'.
    Uses title keywords primarily; full_text if available.
    """
    from src.shared.lexicons import count_matches

    humanitarian_title_kws = [
        "humanitarian", "civilian", "prohibition", "ban", "indiscriminate",
        "suffering", "cluster munition", "landmine", "nuclear ban",
        "mine-free", "victim assistance", "explosive remnant",
    ]
    security_title_kws = [
        "deterrence", "stability", "balance", "risk reduction",
        "nonproliferation", "non-proliferation", "safeguards",
        "verification", "confidence-building", "transparency",
        "security arrangement",
    ]

    combined = (title or "") + " " + (full_text or "")
    h_score = count_matches(combined, humanitarian_title_kws)
    s_score = count_matches(combined, security_title_kws)

    if h_score > s_score:
        return "humanitarian"
    elif s_score > h_score:
        return "security"
    elif h_score > 0 and s_score > 0:
        return "mixed"
    else:
        return "other"
