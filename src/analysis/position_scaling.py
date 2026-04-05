"""
Position scaling for arms control rhetoric.

Implements a simplified Wordfish-like position scaling using PCA on TF-IDF
as an approximation (not the full Poisson Wordfish model).

Note: The true Wordfish model uses a Poisson distribution over word counts
with item/position parameters estimated via EM. This implementation instead
applies TruncatedSVD to the TF-IDF document-term matrix and interprets the
first right singular vector as a latent position dimension. This is a common
fast approximation used in exploratory political text analysis.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def wordfish_scaling(
    dtm_matrix,
    country_year_index: pd.DataFrame,
    n_components: int = 3,
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Simplified Wordfish-like scaling: SVD on TF-IDF matrix.

    NOTE: This is a PCA/SVD approximation, not a true Poisson Wordfish model.
    The first singular dimension is interpreted as the primary rhetorical position axis.
    Positive values indicate one rhetorical pole; negative values indicate the other.

    Parameters
    ----------
    dtm_matrix : sparse or dense document-term matrix (n_docs × n_features)
    country_year_index : DataFrame with country_col and year_col aligned with dtm rows
    n_components : int  number of latent dimensions to retain
    text_col, country_col, year_col : str

    Returns
    -------
    DataFrame with columns: country_code, year, position_1, position_2, position_3
    """
    n_components = min(n_components, min(dtm_matrix.shape) - 1)
    if n_components < 1:
        logger.warning("dtm_matrix too small for SVD; returning empty positions.")
        return pd.DataFrame()

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    positions = svd.fit_transform(dtm_matrix)

    explained = svd.explained_variance_ratio_
    logger.info(
        "SVD explained variance: %s",
        ", ".join(f"dim{i+1}={v:.3f}" for i, v in enumerate(explained)),
    )

    result = country_year_index[[c for c in [country_col, year_col] if c in country_year_index.columns]].copy().reset_index(drop=True)
    for i in range(n_components):
        result[f"position_{i+1}"] = positions[:, i]

    # Aggregate by country-year (average over segments)
    group_cols = [c for c in [country_col, year_col] if c in result.columns]
    pos_cols = [f"position_{i+1}" for i in range(n_components)]
    if group_cols:
        result = result.groupby(group_cols)[pos_cols].mean().reset_index()

    return result


def pca_on_positions(
    positions_df: pd.DataFrame,
    exclude_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Further reduce position dimensions using PCA.

    Parameters
    ----------
    positions_df : DataFrame with position columns (position_1, position_2, ...)
    exclude_cols : list of columns to exclude from PCA

    Returns
    -------
    (reduced_df, explained_variance_ratio) tuple
    """
    if exclude_cols is None:
        exclude_cols = ["country_code", "year"]

    pos_cols = [c for c in positions_df.columns if c not in exclude_cols]
    if not pos_cols:
        logger.warning("No position columns found.")
        return positions_df, np.array([])

    X = positions_df[pos_cols].fillna(0).values
    n_components = min(2, X.shape[1], X.shape[0] - 1)
    if n_components < 1:
        return positions_df, np.array([])

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(X)

    result = positions_df[[c for c in exclude_cols if c in positions_df.columns]].copy().reset_index(drop=True)
    for i in range(n_components):
        result[f"pc_{i+1}"] = reduced[:, i]

    return result, pca.explained_variance_ratio_


def compute_positions_from_corpus(
    segments_df: pd.DataFrame,
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
    max_features: int = 3000,
) -> pd.DataFrame:
    """
    Convenience wrapper: fit TF-IDF + SVD directly from a segments DataFrame.

    Returns
    -------
    DataFrame with country_code, year, position_1, position_2, position_3
    """
    texts = segments_df[text_col].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    dtm = vectorizer.fit_transform(texts)
    meta = segments_df[[c for c in [country_col, year_col] if c in segments_df.columns]]
    return wordfish_scaling(dtm, meta, country_col=country_col, year_col=year_col)
