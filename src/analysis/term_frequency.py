"""
Term frequency and TF-IDF analysis for arms control rhetoric.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

TERMS_OF_INTEREST = [
    "humanitarian", "deterrence", "civilian protection", "proportionality",
    "autonomous weapons", "cyber", "artificial intelligence", "verification",
    "compliance", "norm", "taboo", "disarmament", "non-proliferation",
    "accountability", "sovereignty", "transparency", "multilateral",
    "indiscriminate", "catastrophic", "existential", "prohibition",
    "stigmatization", "hibakusha", "fissile material", "safeguards",
]


def compute_tfidf_corpus(
    segments_df: pd.DataFrame,
    year_col: str = "year",
    text_col: str = "text",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict:
    """
    Compute TF-IDF matrix over the full corpus.

    Parameters
    ----------
    segments_df : DataFrame with at least [text_col]
    year_col : str
    text_col : str
    max_features : int
    ngram_range : tuple

    Returns
    -------
    dict with keys:
        'matrix' : sparse TF-IDF matrix (n_docs × n_features)
        'vocabulary' : list of terms (column names)
        'vectorizer' : fitted TfidfVectorizer
        'doc_index' : DataFrame row metadata aligned with matrix rows
    """
    texts = segments_df[text_col].fillna("").tolist()
    if not texts:
        raise ValueError("Empty text corpus passed to compute_tfidf_corpus.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out().tolist()

    doc_index = segments_df.drop(columns=[text_col], errors="ignore").reset_index(drop=True)

    logger.info(
        "TF-IDF matrix: %d docs × %d features", matrix.shape[0], matrix.shape[1]
    )
    return {
        "matrix": matrix,
        "vocabulary": vocabulary,
        "vectorizer": vectorizer,
        "doc_index": doc_index,
    }


def get_term_trajectories(
    tfidf_results: Dict,
    terms_of_interest: Optional[List[str]] = None,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute mean TF-IDF score for each term of interest, aggregated by year.

    Parameters
    ----------
    tfidf_results : output of compute_tfidf_corpus
    terms_of_interest : list of term strings (single or multi-word)
    year_col : str

    Returns
    -------
    DataFrame with columns: year, <term1>, <term2>, ...
    """
    if terms_of_interest is None:
        terms_of_interest = TERMS_OF_INTEREST

    matrix = tfidf_results["matrix"]
    vocabulary = tfidf_results["vocabulary"]
    doc_index = tfidf_results["doc_index"]

    vocab_lower = [v.lower() for v in vocabulary]
    vocab_map = {v: i for i, v in enumerate(vocab_lower)}

    # Dense for trajectory calculation (only needed columns)
    records = []
    for term in terms_of_interest:
        t = term.lower()
        if t not in vocab_map:
            # Try partial match for ngrams
            t = next((v for v in vocab_lower if t in v), None)
        if t is None or t not in vocab_map:
            continue
        col_idx = vocab_map[t]
        scores = np.asarray(matrix[:, col_idx].todense()).flatten()
        records.append({"term": term, "scores": scores})

    if not records:
        logger.warning("None of the requested terms found in vocabulary.")
        return pd.DataFrame()

    if year_col not in doc_index.columns:
        logger.warning("year_col '%s' not in doc_index; cannot aggregate by year.", year_col)
        return pd.DataFrame()

    years = doc_index[year_col].values
    unique_years = sorted(set(years))
    rows = []
    for yr in unique_years:
        mask = years == yr
        row = {"year": yr}
        for rec in records:
            row[rec["term"]] = float(rec["scores"][mask].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def compute_country_tfidf(
    segments_df: pd.DataFrame,
    text_col: str = "text",
    country_col: str = "country_code",
    max_features: int = 3000,
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute distinctive terms per country using TF-IDF treating each country as a document.

    Returns
    -------
    dict mapping country_code → list of (term, score) tuples
    """
    # Aggregate all text per country
    country_texts = (
        segments_df.groupby(country_col)[text_col]
        .apply(lambda x: " ".join(x.fillna("")))
        .reset_index()
    )
    country_texts.columns = [country_col, "agg_text"]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(country_texts["agg_text"])
    vocab = vectorizer.get_feature_names_out()

    result = {}
    for i, row_data in country_texts.iterrows():
        iso3 = row_data[country_col]
        row_vec = np.asarray(matrix[i].todense()).flatten()
        top_idx = row_vec.argsort()[-top_n:][::-1]
        result[iso3] = [(vocab[j], float(row_vec[j])) for j in top_idx]

    return result


def compute_log_frequency_ratios(
    segments_df: pd.DataFrame,
    period1_years: List[int],
    period2_years: List[int],
    text_col: str = "text",
    year_col: str = "year",
    max_features: int = 3000,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Compute log frequency ratio of terms between two time periods.

    Returns
    -------
    DataFrame with columns: term, freq_period1, freq_period2, log_ratio
    Sorted by abs(log_ratio) descending.
    """
    p1_texts = segments_df[segments_df[year_col].isin(period1_years)][text_col].fillna("").tolist()
    p2_texts = segments_df[segments_df[year_col].isin(period2_years)][text_col].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=False,
        min_df=2,
        stop_words="english",
    )
    vectorizer.fit(p1_texts + p2_texts)
    vocab = vectorizer.get_feature_names_out()

    def _mean_freq(texts):
        if not texts:
            return np.zeros(len(vocab))
        m = vectorizer.transform(texts)
        return np.asarray(m.mean(axis=0)).flatten()

    freq1 = _mean_freq(p1_texts)
    freq2 = _mean_freq(p2_texts)
    log_ratio = np.log((freq2 + 1e-9) / (freq1 + 1e-9))

    df = pd.DataFrame({"term": vocab, "freq_period1": freq1, "freq_period2": freq2, "log_ratio": log_ratio})
    df["abs_log_ratio"] = df["log_ratio"].abs()
    df = df.sort_values("abs_log_ratio", ascending=False).head(top_n)
    return df.drop(columns=["abs_log_ratio"]).reset_index(drop=True)
