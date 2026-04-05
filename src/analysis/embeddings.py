"""
Text embedding and semantic similarity analysis for arms control speeches.

Requires sentence-transformers for real embeddings.
Falls back to TF-IDF-based mock similarities if not installed.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    logger.warning(
        "sentence-transformers not installed. Mock TF-IDF-based similarities will be used. "
        "Install with: pip install sentence-transformers"
    )


def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed a list of texts.

    If sentence-transformers is available, uses the specified model.
    Otherwise falls back to a TF-IDF SVD approximation (64-dim).

    Parameters
    ----------
    texts : list of str
    model_name : str
    batch_size : int

    Returns
    -------
    np.ndarray of shape (n_texts, embedding_dim)
    """
    if not texts:
        return np.zeros((0, 64))

    if _ST_AVAILABLE:
        logger.info("Loading sentence-transformer model '%s'...", model_name)
        model = SentenceTransformer(model_name)
        logger.info("Encoding %d texts with '%s'...", len(texts), model_name)
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        return embeddings

    # Fallback: TF-IDF + SVD
    logger.info("Using TF-IDF SVD fallback for text embedding.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    n_components = min(64, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    if n_components < 2:
        return np.random.default_rng(42).standard_normal((len(texts), 64))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    embeddings = normalize(embeddings)
    # Pad to 64 dims if needed
    if embeddings.shape[1] < 64:
        pad = np.zeros((embeddings.shape[0], 64 - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, pad])
    return embeddings


def embed_treaty_anchors(
    treaty_anchors_dict: Dict[str, List[str]],
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, np.ndarray]:
    """
    Embed treaty anchor passages.

    Parameters
    ----------
    treaty_anchors_dict : dict mapping treaty_name → list of anchor strings
    model_name : str

    Returns
    -------
    dict mapping anchor_id (e.g. "npt_0") → embedding np.ndarray
    """
    flat_texts = []
    anchor_ids = []
    for treaty, passages in treaty_anchors_dict.items():
        for idx, text in enumerate(passages):
            flat_texts.append(text)
            anchor_ids.append(f"{treaty}_{idx}")

    if not flat_texts:
        return {}

    embeddings = embed_texts(flat_texts, model_name=model_name)
    return {aid: emb for aid, emb in zip(anchor_ids, embeddings)}


def compute_anchor_similarity(
    speech_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between speech and anchor embeddings.

    Parameters
    ----------
    speech_embeddings : (n_speeches, dim)
    anchor_embeddings : (n_anchors, dim)

    Returns
    -------
    similarity matrix of shape (n_speeches, n_anchors)
    """
    # Normalise rows
    s_norm = normalize(speech_embeddings)
    a_norm = normalize(anchor_embeddings)
    return s_norm @ a_norm.T


def embed_and_cache(
    texts: List[str],
    cache_path: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed texts with caching to a .npy file.

    If cache_path exists, loads from disk instead of re-computing.
    Saves to cache_path after computing.
    """
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            logger.info("Loading cached embeddings from %s", p)
            return np.load(str(p))

    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), embeddings)
        logger.info("Saved embeddings to %s  shape=%s", p, embeddings.shape)

    return embeddings


def compute_country_year_anchor_scores(
    segments_df: pd.DataFrame,
    anchor_embeddings: Dict[str, np.ndarray],
    model_name: str = "all-MiniLM-L6-v2",
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per country-year average cosine similarity to each treaty anchor group.

    Parameters
    ----------
    segments_df : DataFrame
    anchor_embeddings : dict mapping anchor_id → embedding
    model_name : str
    text_col, country_col, year_col : str

    Returns
    -------
    DataFrame with columns:
        country_code, year,
        npt_score, att_score, tpnw_score, cwc_score, ottawa_score, ccm_score, bwc_score
    """
    texts = segments_df[text_col].fillna("").tolist()
    if not texts:
        return pd.DataFrame()

    logger.info("Embedding %d speech segments...", len(texts))
    speech_embs = embed_and_cache(texts, cache_path=cache_path, model_name=model_name)

    # Group anchor embeddings by treaty
    treaties = set(aid.rsplit("_", 1)[0] for aid in anchor_embeddings)
    treaty_anchor_embs: Dict[str, np.ndarray] = {}
    for treaty in treaties:
        relevant = [v for k, v in anchor_embeddings.items() if k.startswith(f"{treaty}_")]
        if relevant:
            treaty_anchor_embs[treaty] = np.vstack(relevant).mean(axis=0, keepdims=True)

    # Compute per-document scores per treaty
    score_cols = {}
    for treaty, anc_emb in treaty_anchor_embs.items():
        sim = compute_anchor_similarity(speech_embs, anc_emb).flatten()
        score_cols[f"{treaty}_score"] = sim

    # Build result DataFrame aligned with segments
    result = segments_df[[country_col, year_col]].copy().reset_index(drop=True)
    for col, vals in score_cols.items():
        result[col] = vals

    # Aggregate by country-year
    score_col_names = list(score_cols.keys())
    group_cols = [c for c in [country_col, year_col] if c in result.columns]
    if group_cols:
        result = result.groupby(group_cols)[score_col_names].mean().reset_index()

    # Ensure standard column names
    expected = ["npt_score", "att_score", "tpnw_score", "cwc_score", "ottawa_score", "ccm_score", "bwc_score"]
    for col in expected:
        if col not in result.columns:
            result[col] = np.nan

    return result


def detect_semantic_drift(
    country_embeddings_by_year: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """
    Compute year-over-year cosine distance for a country's mean embedding.

    Parameters
    ----------
    country_embeddings_by_year : dict mapping year → mean embedding vector (1-D)

    Returns
    -------
    DataFrame with columns: year, year_prev, cosine_distance
    """
    years = sorted(country_embeddings_by_year.keys())
    rows = []
    for i in range(1, len(years)):
        yr = years[i]
        yr_prev = years[i - 1]
        e1 = country_embeddings_by_year[yr_prev]
        e2 = country_embeddings_by_year[yr]
        n1 = np.linalg.norm(e1) + 1e-9
        n2 = np.linalg.norm(e2) + 1e-9
        cos_sim = np.dot(e1 / n1, e2 / n2)
        rows.append(
            {"year": yr, "year_prev": yr_prev, "cosine_distance": 1.0 - float(cos_sim)}
        )
    return pd.DataFrame(rows)


def cluster_countries_by_rhetoric(
    embeddings_by_year: Dict[str, Dict[int, np.ndarray]],
    n_clusters: int = 5,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Cluster countries by their mean rhetoric embedding.

    Parameters
    ----------
    embeddings_by_year : dict mapping iso3 → {year: mean_embedding}
    n_clusters : int
    year : int or None (if None, use all-year mean)

    Returns
    -------
    DataFrame with columns: country_code, cluster
    """
    from sklearn.cluster import KMeans

    iso3s = list(embeddings_by_year.keys())
    vecs = []
    for iso3 in iso3s:
        year_map = embeddings_by_year[iso3]
        if year is not None and year in year_map:
            vec = year_map[year]
        else:
            vec = np.mean(list(year_map.values()), axis=0)
        vecs.append(vec)

    matrix = np.vstack(vecs)
    n_clusters = min(n_clusters, len(iso3s))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)

    return pd.DataFrame({"country_code": iso3s, "cluster": labels})
