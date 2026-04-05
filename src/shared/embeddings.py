"""
Embedding computation and caching for speech corpus and treaty anchors.
Uses sentence-transformers; falls back to TF-IDF if unavailable.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast default; use all-mpnet-base-v2 for accuracy

_encoder_cache = {}


def get_encoder(model_name: str = DEFAULT_MODEL):
    """Load sentence-transformers model. Falls back to TF-IDF wrapper if unavailable."""
    if model_name in _encoder_cache:
        return _encoder_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name)
        _encoder_cache[model_name] = encoder
        return encoder
    except (ImportError, Exception) as e:
        print(f"[Embeddings] sentence-transformers unavailable ({e}). Using TF-IDF fallback.")
        encoder = _TFIDFEncoder()
        _encoder_cache[model_name] = encoder
        return encoder


class _TFIDFEncoder:
    """TF-IDF fallback encoder that mimics SentenceTransformer interface."""

    def __init__(self, n_components: int = 300):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False
        self._corpus = []

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        if not self._fitted:
            self._corpus.extend(texts)
            all_texts = self._corpus
            tfidf = self.vectorizer.fit_transform(all_texts)
            result = self.svd.fit_transform(tfidf)
            self._fitted = True
        else:
            tfidf = self.vectorizer.transform(texts)
            result = self.svd.transform(tfidf)
        # Normalize rows to unit length
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return result / norms


def embed_texts(
    texts: list,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed list of texts. Returns (N, dim) numpy array.
    """
    encoder = get_encoder(model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    return np.array(embeddings)


def embed_corpus(
    df: pd.DataFrame,
    text_col: str = "segment_text",
    model_name: str = DEFAULT_MODEL,
    cache_path: str = "output/shared/embeddings_cache.npz",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Embed all texts in corpus DataFrame.

    Returns:
        embeddings: (N, dim) numpy array
        index_df: DataFrame mapping embedding row -> (country_iso3, year, original_index)
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force_recompute:
        print(f"[Embeddings] Loading cached embeddings from {cache_path}")
        data = np.load(cache_file, allow_pickle=True)
        embeddings = data["embeddings"]
        index_df = pd.DataFrame(data["index_df"].tolist())
        print(f"[Embeddings] Loaded {len(embeddings)} cached embeddings")
        return embeddings, index_df

    texts = df[text_col].fillna("").tolist()
    print(f"[Embeddings] Embedding {len(texts)} documents with {model_name}...")
    embeddings = embed_texts(texts, model_name=model_name)

    index_df = df[["country_iso3", "year"]].reset_index(drop=True).copy()
    if "source" in df.columns:
        index_df["source"] = df["source"].values

    # Save cache
    np.savez(
        cache_file,
        embeddings=embeddings,
        index_df=index_df.to_dict("records"),
    )
    print(f"[Embeddings] Cached {len(embeddings)} embeddings to {cache_path}")
    return embeddings, index_df


def embed_treaty_anchors(
    anchors: dict,
    model_name: str = DEFAULT_MODEL,
    cache_path: str = "output/shared/anchor_embeddings.npz",
) -> dict:
    """
    Embed all treaty anchor passages.

    Returns dict: {
        treaty_name: {
            'passage_embeddings': {passage_name: embedding_vector},
            'mean_embedding': ndarray,
        }
    }
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        print(f"[Embeddings] Loading cached anchor embeddings from {cache_path}")
        data = np.load(cache_file, allow_pickle=True)
        return data["anchor_embeddings"].tolist()

    result = {}
    for treaty, info in anchors.items():
        passages = info.get("passages", {})
        passage_embeddings = {}
        all_vecs = []

        for passage_name, text in passages.items():
            if not text:
                continue
            vec = embed_texts([text], model_name=model_name, show_progress=False)[0]
            passage_embeddings[passage_name] = vec
            all_vecs.append(vec)

        mean_emb = np.mean(all_vecs, axis=0) if all_vecs else np.zeros(384)
        result[treaty] = {
            "passage_embeddings": passage_embeddings,
            "mean_embedding": mean_emb,
        }
        print(f"[Embeddings] Embedded {len(passage_embeddings)} passages for {treaty}")

    np.savez(cache_file, anchor_embeddings=result)
    return result


def get_humanitarian_anchor_embedding(anchor_embeddings: dict) -> np.ndarray:
    """Mean embedding of TPNW preamble + Ottawa preamble + CCM preamble + ATT preamble."""
    vecs = []
    for treaty, passage in [("tpnw", "preamble"), ("ottawa", "preamble"),
                              ("ccm", "preamble"), ("att", "preamble")]:
        emb = (
            anchor_embeddings.get(treaty, {})
            .get("passage_embeddings", {})
            .get(passage)
        )
        if emb is not None:
            vecs.append(emb)
    if not vecs:
        # Fallback: use mean embeddings
        for treaty in ["tpnw", "ottawa", "ccm", "att"]:
            emb = anchor_embeddings.get(treaty, {}).get("mean_embedding")
            if emb is not None:
                vecs.append(emb)
    return np.mean(vecs, axis=0) if vecs else np.zeros(384)


def get_security_anchor_embedding(anchor_embeddings: dict) -> np.ndarray:
    """Mean embedding of NPT article VI + CWC preamble."""
    vecs = []
    for treaty, passage in [("npt", "article_vi"), ("npt", "preamble"), ("cwc", "preamble")]:
        emb = (
            anchor_embeddings.get(treaty, {})
            .get("passage_embeddings", {})
            .get(passage)
        )
        if emb is not None:
            vecs.append(emb)
    if not vecs:
        emb = anchor_embeddings.get("npt", {}).get("mean_embedding")
        if emb is not None:
            vecs.append(emb)
    return np.mean(vecs, axis=0) if vecs else np.zeros(384)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between rows of a and rows of b.
    Returns (N, M) matrix.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def compute_country_year_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-document embeddings to per-country-year mean embeddings.

    Returns DataFrame: country_iso3, year, embedding (numpy array stored as object column)
    """
    index_df = index_df.reset_index(drop=True)
    records = []

    for (country, year), group in index_df.groupby(["country_iso3", "year"]):
        idxs = group.index.tolist()
        vecs = embeddings[idxs]
        mean_vec = vecs.mean(axis=0)
        records.append({
            "country_iso3": country,
            "year": year,
            "embedding": mean_vec,
            "n_docs": len(idxs),
        })

    return pd.DataFrame(records)


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of a set of embeddings."""
    if len(embeddings) == 0:
        return np.zeros(384)
    return embeddings.mean(axis=0)


def compute_pairwise_distances(
    embeddings: np.ndarray,
    labels: list,
) -> pd.DataFrame:
    """
    Pairwise cosine distances between labeled embeddings.
    Returns long-form DataFrame: label_a, label_b, similarity, distance.
    """
    records = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(embeddings[i], embeddings[j])
            records.append({
                "label_a": labels[i],
                "label_b": labels[j],
                "similarity": sim,
                "distance": 1.0 - sim,
            })
    return pd.DataFrame(records)
