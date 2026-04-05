"""
BERTopic wrapper — runs once on full corpus, results shared by all question modules.
Falls back to LDA if BERTopic unavailable.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple


def run_bertopic(
    texts: List[str],
    nr_topics: int = 30,
    model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
    min_topic_size: int = 10,
) -> Tuple:
    """
    Run BERTopic on corpus.

    Returns:
        topic_model: fitted BERTopic object (or LDA equivalent)
        topics: list of topic assignments per document
        probs: (N, nr_topics) probability matrix
    """
    try:
        return _run_bertopic_impl(texts, nr_topics, model_name, seed, min_topic_size)
    except ImportError as e:
        print(f"[Topics] BERTopic unavailable ({e}). Falling back to LDA.")
        return _run_lda_fallback(texts, nr_topics, seed)


def _run_bertopic_impl(texts, nr_topics, model_name, seed, min_topic_size):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    print(f"[Topics] Running BERTopic with {len(texts)} documents, target {nr_topics} topics...")
    embedding_model = SentenceTransformer(model_name)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True,
        language="english",
        seed_topic_list=None,
    )

    topics, probs = topic_model.fit_transform(texts)
    print(f"[Topics] BERTopic found {len(topic_model.get_topic_info())} topics")
    return topic_model, topics, probs


def _run_lda_fallback(texts, nr_topics, seed):
    """LDA fallback using sklearn."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    print(f"[Topics] Running LDA fallback with {nr_topics} topics on {len(texts)} documents...")

    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=nr_topics,
        random_state=seed,
        max_iter=20,
        learning_method="online",
    )
    probs = lda.fit_transform(dtm)
    topics = np.argmax(probs, axis=1).tolist()

    # Wrap in a simple object that mimics BERTopic's interface
    model_wrapper = _LDAWrapper(lda, vectorizer, nr_topics)
    print(f"[Topics] LDA complete. {nr_topics} topics.")
    return model_wrapper, topics, probs


class _LDAWrapper:
    """Minimal BERTopic-compatible wrapper around sklearn LDA."""

    def __init__(self, lda, vectorizer, nr_topics):
        self._lda = lda
        self._vectorizer = vectorizer
        self._nr_topics = nr_topics
        self._feature_names = vectorizer.get_feature_names_out()

    def get_topic_info(self):
        rows = []
        for i in range(self._nr_topics):
            top_words = self._get_top_words(i, n=10)
            rows.append({"Topic": i, "Count": 0, "Name": f"Topic_{i}", "top_words": top_words})
        return pd.DataFrame(rows)

    def get_topic(self, topic_id):
        words = self._get_top_words(topic_id, n=10)
        return [(w, 1.0 / (i + 1)) for i, w in enumerate(words)]

    def _get_top_words(self, topic_id, n=10):
        topic_components = self._lda.components_[topic_id]
        top_indices = topic_components.argsort()[-n:][::-1]
        return [self._feature_names[i] for i in top_indices]

    def topics_over_time(self, docs, timestamps, nr_bins=20):
        """Stub — returns empty DataFrame for LDA fallback."""
        return pd.DataFrame(columns=["Topic", "Words", "Frequency", "Timestamp"])

    def fit_transform(self, texts):
        dtm = self._vectorizer.transform(texts)
        probs = self._lda.transform(dtm)
        return np.argmax(probs, axis=1).tolist(), probs


def get_topic_info(topic_model) -> pd.DataFrame:
    """Returns DataFrame: topic_id, top_words, representative_docs."""
    try:
        info = topic_model.get_topic_info()
        return info
    except Exception as e:
        print(f"[Topics] get_topic_info failed: {e}")
        return pd.DataFrame()


def topics_over_time(
    topic_model,
    docs: List[str],
    timestamps: List,
    nr_bins: int = 20,
) -> pd.DataFrame:
    """
    BERTopic topics_over_time.
    Returns DataFrame: Topic, Words, Frequency, Timestamp
    """
    try:
        return topic_model.topics_over_time(docs, timestamps, nr_bins=nr_bins)
    except Exception as e:
        print(f"[Topics] topics_over_time failed: {e}")
        return pd.DataFrame(columns=["Topic", "Words", "Frequency", "Timestamp"])


def classify_topics(
    topic_model,
    humanitarian_lexicon: List[str],
    deterrence_lexicon: List[str],
) -> pd.DataFrame:
    """
    Classify each topic as 'humanitarian', 'deterrence', or 'mixed'
    based on overlap with lexicons.

    Returns DataFrame: topic_id, classification, humanitarian_score, deterrence_score, top_words
    """
    try:
        topic_info = topic_model.get_topic_info()
    except Exception:
        return pd.DataFrame()

    results = []
    hum_set = set(w.lower() for w in humanitarian_lexicon)
    det_set = set(w.lower() for w in deterrence_lexicon)

    for _, row in topic_info.iterrows():
        topic_id = row.get("Topic", row.get("topic", -1))
        if topic_id == -1:
            continue  # outlier topic in BERTopic

        try:
            top_words = [w for w, _ in topic_model.get_topic(topic_id)]
        except Exception:
            top_words = []

        words_lower = [w.lower() for w in top_words]
        h_score = sum(1 for w in words_lower if any(hw in w for hw in hum_set))
        d_score = sum(1 for w in words_lower if any(dw in w for dw in det_set))

        if h_score > d_score:
            classification = "humanitarian"
        elif d_score > h_score:
            classification = "deterrence"
        elif h_score > 0 and d_score > 0:
            classification = "mixed"
        else:
            classification = "other"

        results.append({
            "topic_id": topic_id,
            "classification": classification,
            "humanitarian_score": h_score,
            "deterrence_score": d_score,
            "top_words": ", ".join(top_words[:10]),
        })

    return pd.DataFrame(results)


def save_topic_model(
    topic_model,
    topics: list,
    probs: np.ndarray,
    output_dir: str = "output/shared/topics/",
):
    """Save topic model and assignments."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_dir, "topics.npy"), np.array(topics))
    np.save(os.path.join(output_dir, "probs.npy"), probs)

    try:
        topic_model.save(os.path.join(output_dir, "bertopic_model"))
        print(f"[Topics] BERTopic model saved to {output_dir}")
    except Exception:
        # LDA fallback — save topic info as CSV
        try:
            info = topic_model.get_topic_info()
            info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
        except Exception as e:
            print(f"[Topics] Could not save topic model: {e}")


def load_topic_model(output_dir: str = "output/shared/topics/"):
    """Load saved topic model and assignments."""
    topics_path = os.path.join(output_dir, "topics.npy")
    probs_path = os.path.join(output_dir, "probs.npy")

    if not os.path.exists(topics_path):
        return None, None, None

    topics = np.load(topics_path).tolist()
    probs = np.load(probs_path)

    # Try loading BERTopic model
    model_path = os.path.join(output_dir, "bertopic_model")
    if os.path.exists(model_path):
        try:
            from bertopic import BERTopic
            topic_model = BERTopic.load(model_path)
            print(f"[Topics] Loaded BERTopic model from {output_dir}")
            return topic_model, topics, probs
        except Exception as e:
            print(f"[Topics] Could not load BERTopic model: {e}")

    return None, topics, probs


def compute_topic_proportions_by_group(
    topics: list,
    probs: np.ndarray,
    group_labels: list,
    n_topics: int,
) -> pd.DataFrame:
    """
    Compute topic proportion per group.

    Args:
        topics: list of topic assignments per document
        probs: (N, n_topics) probability matrix
        group_labels: list of group labels per document (same length as topics)
        n_topics: number of topics

    Returns DataFrame: group, topic_id, proportion
    """
    records = []
    groups = sorted(set(group_labels))

    for group in groups:
        mask = [i for i, g in enumerate(group_labels) if g == group]
        if not mask:
            continue

        if probs is not None and len(probs) > 0:
            group_probs = probs[mask]
            mean_probs = group_probs.mean(axis=0)
            for topic_id, proportion in enumerate(mean_probs):
                records.append({
                    "group": group,
                    "topic_id": topic_id,
                    "proportion": float(proportion),
                })
        else:
            # Use hard assignments
            group_topics = [topics[i] for i in mask]
            topic_counts = pd.Series(group_topics).value_counts(normalize=True)
            for topic_id, proportion in topic_counts.items():
                records.append({
                    "group": group,
                    "topic_id": int(topic_id),
                    "proportion": float(proportion),
                })

    return pd.DataFrame(records)
