"""
Monroe et al. (2008) Fightin' Words — log-odds ratio with informative Dirichlet prior.
Used by Q2 (democracy vs autocracy) and Q4 (nuclear vs non-nuclear).
"""
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Optional


# Common English stopwords
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "every", "all",
    "this", "that", "these", "those", "it", "its", "we", "our", "they",
    "their", "he", "his", "she", "her", "my", "your", "as", "if", "when",
    "than", "then", "there", "here", "where", "which", "who", "what",
    "also", "well", "very", "more", "most", "such", "other", "into",
    "about", "through", "between", "during", "under", "over", "again",
    "further", "once", "own", "same", "up", "out", "any", "while", "s",
    "ve", "re", "ll", "d", "mr", "mrs", "ms", "one", "two", "three",
    "first", "second", "third", "new", "how", "only", "us", "i",
}


def tokenize(
    text: str,
    min_word_length: int = 3,
    remove_stopwords: bool = True,
) -> List[str]:
    """Tokenize text, remove stopwords and short words."""
    text = text.lower()
    # Keep hyphenated words and apostrophes
    tokens = re.findall(r"[a-z][a-z'\-]*[a-z]|[a-z]{3,}", text)
    tokens = [t for t in tokens if len(t) >= min_word_length]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def fightin_words(
    corpus_a: List[str],
    corpus_b: List[str],
    prior: float = 0.01,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Monroe et al. (2008) log-odds ratio with Dirichlet prior.

    Positive z_score = word is distinctive for corpus_a.
    Negative z_score = word is distinctive for corpus_b.

    Args:
        corpus_a: list of texts (group A, e.g., democracies)
        corpus_b: list of texts (group B, e.g., autocracies)
        prior: Dirichlet smoothing prior (0.01 recommended)
        top_n: return top N words per group

    Returns DataFrame: word, z_score, log_odds_a, log_odds_b, group
    """
    # Tokenize
    tokens_a = []
    for text in corpus_a:
        tokens_a.extend(tokenize(str(text)))

    tokens_b = []
    for text in corpus_b:
        tokens_b.extend(tokenize(str(text)))

    # Build vocabulary
    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return pd.DataFrame(columns=["word", "z_score", "log_odds_a", "log_odds_b", "group"])

    # Count frequencies
    count_a = Counter(tokens_a)
    count_b = Counter(tokens_b)

    n_a = sum(count_a.values())
    n_b = sum(count_b.values())

    if n_a == 0 or n_b == 0:
        return pd.DataFrame(columns=["word", "z_score", "log_odds_a", "log_odds_b", "group"])

    # Monroe et al. log-odds with Dirichlet prior
    # z_w = (log_odds_w - 0) / sigma_w
    # log_odds_w = log( (y_w^a + alpha_w) / (n_a + alpha_0 - y_w^a - alpha_w) )
    #            - log( (y_w^b + alpha_w) / (n_b + alpha_0 - y_w^b - alpha_w) )
    # sigma^2_w = 1/(y_w^a + alpha_w) + 1/(y_w^b + alpha_w)

    results = []
    alpha_0 = prior * len(vocab)  # total prior mass

    for word in vocab:
        y_a = count_a.get(word, 0)
        y_b = count_b.get(word, 0)

        # Skip words that appear fewer than 5 times total
        if y_a + y_b < 5:
            continue

        alpha_w = prior  # symmetric prior per word

        # Log-odds for group a
        p_a = (y_a + alpha_w) / (n_a + alpha_0)
        log_odds_a = np.log(p_a / (1 - p_a))

        # Log-odds for group b
        p_b = (y_b + alpha_w) / (n_b + alpha_0)
        log_odds_b = np.log(p_b / (1 - p_b))

        # Difference and variance
        delta = log_odds_a - log_odds_b
        sigma2 = 1.0 / (y_a + alpha_w) + 1.0 / (y_b + alpha_w)
        z_score = delta / np.sqrt(sigma2)

        results.append({
            "word": word,
            "z_score": z_score,
            "log_odds_a": log_odds_a,
            "log_odds_b": log_odds_b,
            "count_a": y_a,
            "count_b": y_b,
        })

    if not results:
        return pd.DataFrame(columns=["word", "z_score", "log_odds_a", "log_odds_b", "group"])

    df = pd.DataFrame(results).sort_values("z_score", ascending=False)

    # Top N for each group
    top_a = df.head(top_n).copy()
    top_a["group"] = "group_a"
    top_b = df.tail(top_n).copy()
    top_b["group"] = "group_b"
    top_b["z_score"] = top_b["z_score"].abs()  # make positive for ranking

    return pd.concat([top_a, top_b], ignore_index=True)


def fightin_words_by_decade(
    df: pd.DataFrame,
    group_col: str,
    text_col: str,
    year_col: str,
    group_a: str,
    group_b: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Run Fightin' Words for each decade separately.

    Returns long-form DataFrame: decade, word, z_score, group
    """
    results = []

    df = df.copy()
    df["decade"] = (df[year_col] // 10) * 10

    for decade, decade_df in df.groupby("decade"):
        corpus_a = decade_df[decade_df[group_col] == group_a][text_col].fillna("").tolist()
        corpus_b = decade_df[decade_df[group_col] == group_b][text_col].fillna("").tolist()

        if len(corpus_a) < 5 or len(corpus_b) < 5:
            continue

        fw = fightin_words(corpus_a, corpus_b, top_n=top_n)
        if len(fw) > 0:
            fw["decade"] = decade
            results.append(fw)

    if not results:
        return pd.DataFrame(columns=["decade", "word", "z_score", "group"])

    return pd.concat(results, ignore_index=True)


def track_distinctiveness_over_time(
    df: pd.DataFrame,
    words: List[str],
    group_col: str,
    text_col: str,
    year_col: str,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    """
    For a given set of words, compute z-score distinctiveness per decade.
    Shows which words become MORE or LESS distinctive over time.

    Returns DataFrame: word, decade, z_score
    """
    df = df.copy()
    df["decade"] = (df[year_col] // 10) * 10
    results = []

    for decade, decade_df in df.groupby("decade"):
        corpus_a = decade_df[decade_df[group_col] == group_a][text_col].fillna("").tolist()
        corpus_b = decade_df[decade_df[group_col] == group_b][text_col].fillna("").tolist()

        if len(corpus_a) < 3 or len(corpus_b) < 3:
            continue

        fw = fightin_words(corpus_a, corpus_b, top_n=1000)
        if len(fw) == 0:
            continue

        for word in words:
            # Look in group_a distinctive words
            match_a = fw[(fw["word"] == word) & (fw["group"] == "group_a")]
            match_b = fw[(fw["word"] == word) & (fw["group"] == "group_b")]

            if len(match_a) > 0:
                z = float(match_a.iloc[0]["z_score"])
            elif len(match_b) > 0:
                z = -float(match_b.iloc[0]["z_score"])
            else:
                z = 0.0

            results.append({"word": word, "decade": decade, "z_score": z})

    return pd.DataFrame(results)
