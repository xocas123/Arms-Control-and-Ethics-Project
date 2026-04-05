"""
LDA topic modelling for the arms control corpus.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected topic labels for reference (manual assignment after inspection)
EXPECTED_TOPIC_LABELS = [
    "nuclear_disarmament",
    "conventional_arms",
    "wmd_nonproliferation",
    "humanitarian_framing",
    "terrorism_security",
    "regional_security",
    "development_disarmament",
    "outer_space_cyber",
    "verification_compliance",
    "self_defense",
]


GENSIM_AVAILABLE = True
try:
    import gensim  # noqa: F401
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("gensim not installed — LDA will use sklearn fallback.")


def _build_corpus(
    segments_df: pd.DataFrame, text_col: str = "text"
) -> Tuple[list, list, object]:
    """Tokenise texts and build gensim dictionary + corpus."""
    try:
        from gensim import corpora
        from gensim.utils import simple_preprocess
    except ImportError as e:
        raise ImportError("gensim is required for LDA. Install it with: pip install gensim") from e

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "that", "this", "these",
        "those", "it", "its", "we", "our", "us", "they", "their", "also",
        "which", "who", "what", "when", "where", "how", "all", "each", "not",
        "no", "nor", "so", "yet", "both", "either", "such", "mr", "president",
        "delegation", "chair", "madam",
    }

    tokenized = [
        [w for w in simple_preprocess(text, deacc=True) if w not in stop_words and len(w) > 2]
        for text in segments_df[text_col].fillna("")
    ]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    return tokenized, corpus, dictionary


class _SklearnLdaWrapper:
    """Thin wrapper around sklearn LDA to expose a gensim-like interface."""

    def __init__(self, sk_model, vectorizer, k):
        self._sk = sk_model
        self._vec = vectorizer
        self.num_topics = k
        vocab = vectorizer.get_feature_names_out()
        self._vocab = vocab
        self._topics = sk_model.components_  # shape (k, n_vocab)

    def show_topic(self, topic_id, topn=10):
        row = self._topics[topic_id]
        top_idx = row.argsort()[::-1][:topn]
        return [(self._vocab[i], float(row[i] / row.sum())) for i in top_idx]

    def get_document_topics(self, bow, minimum_probability=0.0):
        # bow is a dense row here (index into doc_topics)
        return [(i, float(p)) for i, p in enumerate(bow) if p >= minimum_probability]


def _train_lda_sklearn(
    segments_df: pd.DataFrame, k: int, text_col: str = "text", random_seed: int = 42
) -> Dict:
    """Fallback LDA using sklearn when gensim is unavailable."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    texts = segments_df[text_col].fillna("").tolist()
    vectorizer = CountVectorizer(
        max_df=0.90, min_df=2, max_features=5000, stop_words="english"
    )
    dtm = vectorizer.fit_transform(texts)

    logger.info("Training sklearn LDA with k=%d on %d documents...", k, dtm.shape[0])
    sk_lda = LatentDirichletAllocation(
        n_components=k, random_state=random_seed, max_iter=20, n_jobs=-1
    )
    doc_topic_matrix = sk_lda.fit_transform(dtm)
    # Normalise rows
    row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_topic_matrix = doc_topic_matrix / row_sums

    wrapper = _SklearnLdaWrapper(sk_lda, vectorizer, k)
    logger.info("sklearn LDA training complete.")
    for t_id in range(k):
        words = [w for w, _ in wrapper.show_topic(t_id, topn=5)]
        logger.info("  Topic %d: %s", t_id, ", ".join(words))

    # Build pseudo-corpus: list of dense probability rows (used by get_document_topics)
    corpus = [list(enumerate(row)) for row in doc_topic_matrix]

    return {
        "model": wrapper,
        "corpus": corpus,
        "dictionary": None,
        "doc_topics": doc_topic_matrix,
        "k": k,
    }


def train_lda(
    segments_df: pd.DataFrame,
    k: int = 10,
    text_col: str = "text",
    random_seed: int = 42,
    passes: int = 5,
) -> Dict:
    """
    Train an LDA model on the segments corpus.

    Uses gensim if available, falls back to sklearn LDA.

    Returns
    -------
    dict with keys: 'model', 'corpus', 'dictionary', 'doc_topics', 'k'
    """
    if not GENSIM_AVAILABLE:
        return _train_lda_sklearn(segments_df, k, text_col, random_seed)

    from gensim.models import LdaModel

    tokenized, corpus, dictionary = _build_corpus(segments_df, text_col)

    logger.info("Training LDA with k=%d on %d documents...", k, len(corpus))
    model = LdaModel(
        corpus=corpus,
        num_topics=k,
        id2word=dictionary,
        passes=passes,
        random_state=random_seed,
        alpha="auto",
        eta="auto",
    )

    doc_topics = np.zeros((len(corpus), k))
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            doc_topics[i, topic_id] = prob

    logger.info("LDA training complete. Top words per topic:")
    for t_id in range(k):
        words = [w for w, _ in model.show_topic(t_id, topn=5)]
        logger.info("  Topic %d: %s", t_id, ", ".join(words))

    return {
        "model": model,
        "corpus": corpus,
        "dictionary": dictionary,
        "doc_topics": doc_topics,
        "k": k,
    }


def sweep_lda_k(
    segments_df: pd.DataFrame,
    k_range: Optional[List[int]] = None,
    text_col: str = "text",
    random_seed: int = 42,
    passes: int = 3,
) -> pd.DataFrame:
    """
    Train LDA for multiple values of k and record coherence scores.

    Returns
    -------
    DataFrame with columns: k, coherence_c_v, perplexity
    """
    if k_range is None:
        k_range = [5, 10, 15, 20]

    if not GENSIM_AVAILABLE:
        # sklearn sweep — use perplexity as proxy for coherence
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
        texts = segments_df[text_col].fillna("").tolist()
        vec = CountVectorizer(max_df=0.90, min_df=2, max_features=5000, stop_words="english")
        dtm = vec.fit_transform(texts)
        rows = []
        for k in k_range:
            logger.info("Sweeping sklearn LDA k=%d...", k)
            lda = LatentDirichletAllocation(
                n_components=k, random_state=random_seed, max_iter=20, n_jobs=-1
            )
            lda.fit(dtm)
            perp = lda.perplexity(dtm)
            rows.append({"k": k, "coherence_c_v": float("nan"), "perplexity": perp})
            logger.info("  k=%d: perplexity=%.2f", k, perp)
        return pd.DataFrame(rows)

    try:
        from gensim.models import LdaModel, CoherenceModel
        from gensim import corpora
    except ImportError as e:
        raise ImportError("gensim is required for LDA sweep.") from e

    tokenized, corpus, dictionary = _build_corpus(segments_df, text_col)

    rows = []
    for k in k_range:
        logger.info("Sweeping LDA k=%d...", k)
        model = LdaModel(
            corpus=corpus,
            num_topics=k,
            id2word=dictionary,
            passes=passes,
            random_state=random_seed,
            alpha="auto",
            eta="auto",
        )
        try:
            cm = CoherenceModel(
                model=model, texts=tokenized, dictionary=dictionary, coherence="c_v"
            )
            coherence = cm.get_coherence()
        except Exception:
            coherence = float("nan")

        try:
            perplexity = model.log_perplexity(corpus)
        except Exception:
            perplexity = float("nan")

        rows.append({"k": k, "coherence_c_v": coherence, "perplexity": perplexity})
        logger.info("  k=%d: coherence=%.4f, perplexity=%.4f", k, coherence, perplexity)

    return pd.DataFrame(rows)


def get_lda_topic_distributions(
    model,
    corpus: list,
    dictionary,
    segments_df: Optional[pd.DataFrame] = None,
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Extract country-year × topic proportion DataFrame.

    Returns
    -------
    DataFrame with columns: country_code, year, topic_0, topic_1, ..., topic_k
    """
    k = model.num_topics
    # If doc_topics is already a 2D ndarray (sklearn wrapper path), use it directly
    if isinstance(model, _SklearnLdaWrapper):
        doc_topics = np.array([[p for _, p in row] for row in corpus])
    else:
        doc_topics = np.zeros((len(corpus), k))
        for i, bow in enumerate(corpus):
            for t_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
                doc_topics[i, t_id] = prob

    topic_cols = [f"topic_{t}" for t in range(k)]
    df_topics = pd.DataFrame(doc_topics, columns=topic_cols)

    if segments_df is not None:
        for col in [country_col, year_col]:
            if col in segments_df.columns:
                df_topics[col] = segments_df[col].values

        group_cols = [c for c in [country_col, year_col] if c in df_topics.columns]
        if group_cols:
            df_topics = df_topics.groupby(group_cols)[topic_cols].mean().reset_index()

    return df_topics


def label_lda_topics(model, n_words: int = 10) -> List[Dict]:
    """
    Return top words per topic for manual labeling.

    Returns
    -------
    list of dicts: [{'topic_id': int, 'top_words': [str], 'label': str or None}]
    """
    result = []
    for t_id in range(model.num_topics):
        words = [w for w, _ in model.show_topic(t_id, topn=n_words)]
        # Assign a heuristic label based on top words
        label = _heuristic_label(words)
        result.append({"topic_id": t_id, "top_words": words, "label": label})
    return result


def _heuristic_label(words: List[str]) -> str:
    """Assign a topic label based on top words."""
    w_set = set(w.lower() for w in words)
    if any(t in w_set for t in ["nuclear", "disarmament", "npt", "weapon"]):
        return "nuclear_disarmament"
    if any(t in w_set for t in ["humanitarian", "civilian", "suffering", "victim"]):
        return "humanitarian_framing"
    if any(t in w_set for t in ["chemical", "biological", "cwc", "bwc", "opcw"]):
        return "wmd_nonproliferation"
    if any(t in w_set for t in ["arms", "trade", "export", "conventional"]):
        return "conventional_arms"
    if any(t in w_set for t in ["terror", "terrorism", "security", "threat"]):
        return "terrorism_security"
    if any(t in w_set for t in ["regional", "conflict", "zone", "border"]):
        return "regional_security"
    if any(t in w_set for t in ["development", "economic", "sustainable"]):
        return "development_disarmament"
    if any(t in w_set for t in ["cyber", "space", "technology", "artificial"]):
        return "outer_space_cyber"
    if any(t in w_set for t in ["verification", "compliance", "inspection", "safeguard"]):
        return "verification_compliance"
    if any(t in w_set for t in ["self", "defense", "deterrence", "sovereignty"]):
        return "self_defense"
    return "uncategorized"
