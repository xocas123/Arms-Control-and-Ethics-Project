"""
BERTopic topic modelling (optional — requires bertopic + sentence-transformers).
Gracefully skips if dependencies are not installed.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BERTOPIC_AVAILABLE = False
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    logger.warning(
        "bertopic is not installed. BERTopic analysis will be skipped. "
        "Install with: pip install bertopic"
    )


def train_bertopic(
    segments_df: pd.DataFrame,
    embeddings: Optional = None,
    min_topic_size: int = 15,
    text_col: str = "text",
    nr_topics: Optional[int] = None,
) -> Optional[Dict]:
    """
    Train a BERTopic model on the corpus.

    Parameters
    ----------
    segments_df : DataFrame with text_col
    embeddings : np.ndarray or None (pre-computed embeddings)
    min_topic_size : int
    text_col : str
    nr_topics : int or None (auto)

    Returns
    -------
    dict with keys: 'model', 'topics', 'probs', 'topic_info'
    or None if bertopic is not installed.
    """
    if not BERTOPIC_AVAILABLE:
        logger.warning("BERTopic not available; skipping training.")
        return None

    texts = segments_df[text_col].fillna("").tolist()

    kwargs = {"min_topic_size": min_topic_size, "verbose": True}
    if nr_topics is not None:
        kwargs["nr_topics"] = nr_topics

    model = BERTopic(**kwargs)

    if embeddings is not None:
        topics, probs = model.fit_transform(texts, embeddings)
    else:
        topics, probs = model.fit_transform(texts)

    topic_info = model.get_topic_info()
    logger.info("BERTopic found %d topics (excl. outlier topic -1).", len(topic_info) - 1)

    return {
        "model": model,
        "topics": topics,
        "probs": probs,
        "topic_info": topic_info,
    }


def get_topics_over_time(
    model,
    segments_df: pd.DataFrame,
    text_col: str = "text",
    year_col: str = "year",
) -> Optional[pd.DataFrame]:
    """
    Get topic prevalence by year.

    Returns
    -------
    DataFrame with columns: year, topic, count, freq
    or None if model is None.
    """
    if model is None or not BERTOPIC_AVAILABLE:
        return None

    texts = segments_df[text_col].fillna("").tolist()
    timestamps = segments_df[year_col].tolist() if year_col in segments_df.columns else None

    if timestamps is None:
        logger.warning("No year column found; cannot compute topics over time.")
        return None

    try:
        topics_over_time = model.topics_over_time(texts, timestamps)
        return topics_over_time
    except Exception as exc:
        logger.warning("topics_over_time failed: %s", exc)
        return None


def get_topics_per_country_group(
    model,
    segments_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    text_col: str = "text",
    country_col: str = "country_code",
) -> Optional[pd.DataFrame]:
    """
    Get topic distribution per country group.

    Returns
    -------
    DataFrame with columns: group, topic, proportion
    or None if model is None.
    """
    if model is None or not BERTOPIC_AVAILABLE:
        return None

    if "topics" not in dir(model):
        logger.warning("Model has no topics attribute.")
        return None

    rows = []
    for group_name, members in groups.items():
        mask = segments_df[country_col].isin(members)
        sub = segments_df[mask]
        if sub.empty:
            continue
        texts = sub[text_col].fillna("").tolist()
        try:
            topics, _ = model.transform(texts)
            topic_counts = pd.Series(topics).value_counts(normalize=True)
            for t_id, prop in topic_counts.items():
                rows.append({"group": group_name, "topic": t_id, "proportion": prop})
        except Exception as exc:
            logger.warning("Error transforming group %s: %s", group_name, exc)

    return pd.DataFrame(rows) if rows else None


def save_bertopic_model(model, path: str) -> None:
    """Save a BERTopic model to disk."""
    if model is None:
        logger.warning("Cannot save None model.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(path))
        logger.info("BERTopic model saved to %s", path)
    except Exception as exc:
        logger.error("Failed to save BERTopic model: %s", exc)


def load_bertopic_model(path: str):
    """Load a BERTopic model from disk. Returns None if bertopic not installed."""
    if not BERTOPIC_AVAILABLE:
        logger.warning("bertopic not installed; cannot load model.")
        return None
    try:
        model = BERTopic.load(str(path))
        logger.info("BERTopic model loaded from %s", path)
        return model
    except Exception as exc:
        logger.error("Failed to load BERTopic model: %s", exc)
        return None
