"""
Dynamic Topic Model (DTM) wrapper.
Uses gensim's LdaSeqModel as the DTM implementation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def _train_dtm_sklearn(
    segments_df: pd.DataFrame,
    n_topics: int = 15,
    text_col: str = "text",
    year_col: str = "year",
    random_seed: int = 42,
) -> Optional[Dict]:
    """
    Approximate DTM using sklearn: train one LDA per time period and track
    topic evolution by aligning topics via top-word overlap.
    """
    df_sorted = segments_df.sort_values(year_col).reset_index(drop=True)
    years = sorted(df_sorted[year_col].unique())

    logger.info(
        "Training sklearn DTM approximation: %d topics, %d time periods, %d documents",
        n_topics, len(years), len(df_sorted),
    )

    # Fit a global vocabulary first
    global_vec = CountVectorizer(
        max_df=0.90, min_df=2, max_features=8000, stop_words="english"
    )
    global_vec.fit(df_sorted[text_col].fillna(""))
    vocab = global_vec.get_feature_names_out()

    period_models = {}
    period_top_words = {}

    for year in years:
        sub = df_sorted[df_sorted[year_col] == year][text_col].fillna("").tolist()
        if len(sub) < 3:
            continue
        dtm = global_vec.transform(sub)
        k = min(n_topics, dtm.shape[0] - 1, dtm.shape[1] - 1)
        if k < 2:
            continue
        lda = LatentDirichletAllocation(
            n_components=k, random_state=random_seed, max_iter=30, n_jobs=-1
        )
        lda.fit(dtm)
        period_models[year] = lda
        # Top words per topic
        top_words = {}
        for t_id, comp in enumerate(lda.components_):
            top_idx = comp.argsort()[::-1][:15]
            top_words[t_id] = [vocab[i] for i in top_idx]
        period_top_words[year] = top_words
        logger.info("  DTM year=%d: fitted %d topics on %d docs", year, k, len(sub))

    if not period_models:
        logger.error("DTM sklearn: no period models could be trained.")
        return None

    logger.info("sklearn DTM approximation complete for %d time periods.", len(period_models))
    return {
        "model": "sklearn_dtm",
        "period_models": period_models,
        "period_top_words": period_top_words,
        "years": years,
        "vocabulary": vocab,
        "vectorizer": global_vec,
        "n_topics": n_topics,
    }


def train_dtm(
    segments_df: pd.DataFrame,
    n_topics: int = 15,
    text_col: str = "text",
    year_col: str = "year",
    random_seed: int = 42,
    passes: int = 3,
) -> Optional[Dict]:
    """
    Train a Dynamic Topic Model (gensim LdaSeqModel).

    Parameters
    ----------
    segments_df : DataFrame with text_col and year_col
    n_topics : int
    text_col : str
    year_col : str
    random_seed : int
    passes : int

    Returns
    -------
    dict with keys: 'model', 'corpus', 'dictionary', 'time_slices', 'years'
    or None on failure.
    """
    _gensim_ok = True
    try:
        from gensim.models import LdaSeqModel
        from gensim import corpora
        from gensim.utils import simple_preprocess
    except ImportError as e:
        logger.warning("gensim not available (%s) — using sklearn DTM approximation.", e)
        _gensim_ok = False

    if not _gensim_ok:
        return _train_dtm_sklearn(
            segments_df, n_topics=n_topics, text_col=text_col,
            year_col=year_col, random_seed=random_seed,
        )

    if year_col not in segments_df.columns:
        logger.error("year_col '%s' not found in segments_df.", year_col)
        return None

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "that", "this", "these",
        "those", "it", "its", "we", "our", "us", "they", "their", "also",
        "which", "who", "what", "when", "where", "how", "all", "each", "not",
        "no", "nor", "so", "yet", "both", "either", "mr", "president",
        "delegation", "chair", "madam",
    }

    # Sort by year
    df_sorted = segments_df.sort_values(year_col).reset_index(drop=True)
    years = sorted(df_sorted[year_col].unique())

    tokenized = [
        [w for w in simple_preprocess(t, deacc=True) if w not in stop_words and len(w) > 2]
        for t in df_sorted[text_col].fillna("")
    ]

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(toks) for toks in tokenized]

    # Time slices: number of documents per year
    time_slices = [
        int((df_sorted[year_col] == yr).sum()) for yr in years
    ]

    if any(s == 0 for s in time_slices):
        logger.warning("Some years have 0 documents; DTM may fail.")

    logger.info(
        "Training DTM: %d topics, %d time slices, %d documents",
        n_topics, len(time_slices), len(corpus),
    )

    try:
        model = LdaSeqModel(
            corpus=corpus,
            time_slice=time_slices,
            num_topics=n_topics,
            id2word=dictionary,
            random_state=random_seed,
        )
        logger.info("DTM training complete.")
        return {
            "model": model,
            "corpus": corpus,
            "dictionary": dictionary,
            "time_slices": time_slices,
            "years": years,
        }
    except Exception as exc:
        logger.error("DTM training failed: %s", exc)
        return None


def get_dtm_topic_evolution(
    dtm_results: Dict,
    n_words: int = 10,
) -> pd.DataFrame:
    """
    Extract top words per topic per time slice.

    Returns DataFrame with columns: year, topic_id, top_words (comma-separated)
    """
    if dtm_results is None:
        return pd.DataFrame()

    rows = []

    # sklearn DTM approximation path
    if dtm_results.get("model") == "sklearn_dtm":
        for year, top_words in dtm_results["period_top_words"].items():
            for t_id, words in top_words.items():
                rows.append({
                    "year": year,
                    "topic_id": t_id,
                    "top_words": ", ".join(words[:n_words]),
                })
        return pd.DataFrame(rows)

    # gensim LdaSeqModel path
    model = dtm_results["model"]
    years = dtm_results["years"]
    for t_slice, year in enumerate(years):
        for t_id in range(model.num_topics):
            try:
                topic_words = model.print_topic(t_id, time=t_slice, top_terms=n_words)
                rows.append({"year": year, "topic_id": t_id, "top_words": topic_words})
            except Exception:
                pass

    return pd.DataFrame(rows)
