"""
Q2: Do democracies and autocracies talk differently about arms control?

Analyses:
2a. Distinctive vocabulary (Fightin' Words: democracy vs autocracy)
2b. Distinctive vocabulary by decade
2c. Topic proportions by regime type over time
2d. Embedding distance between regime groups over time
2e. Frame ratio by regime type over time
2f. Regime transition case studies
2g. Voting by regime type over time (the NAM paradox)
"""
import os
import numpy as np
import pandas as pd

from src.shared.distinctive_words import fightin_words, fightin_words_by_decade
from src.shared.temporal import compute_change_points, compute_between_group_distance
from src.shared.embeddings import compute_centroid
from src.shared.topics import compute_topic_proportions_by_group

OUTPUT_DIR = "output/q2"

# Regime transitions to study (country: {direction, transition_year, label})
REGIME_TRANSITIONS = {
    "TUR": {"direction": "backsliding", "year": 2018, "label": "Turkey autocratization"},
    "HUN": {"direction": "backsliding", "year": 2010, "label": "Hungary democratic backsliding"},
    "IND": {"direction": "backsliding", "year": 2019, "label": "India democratic erosion"},
    "RUS": {"direction": "backsliding", "year": 2012, "label": "Russia autocratization"},
    "TUN": {"direction": "democratization", "year": 2011, "label": "Tunisia democratization"},
    "MMR": {"direction": "backsliding", "year": 2021, "label": "Myanmar coup"},
}


def _get_binary_regime(country_iso3: str, year: int, vdem_df: pd.DataFrame) -> str:
    """Returns 'democracy' or 'autocracy' based on V-Dem v2x_regime."""
    if not vdem_df.empty:
        row = vdem_df[(vdem_df["country_iso3"] == country_iso3) & (vdem_df["year"] == year)]
        if not row.empty:
            regime = row.iloc[0]["v2x_regime"]
            return "democracy" if regime >= 2 else "autocracy"

    # Static fallback
    from src.data.groups import get_binary_regime
    return get_binary_regime(country_iso3, year)


def _get_four_way_regime(country_iso3: str, year: int, vdem_df: pd.DataFrame) -> str:
    """Returns 4-way regime type from V-Dem."""
    _MAP = {0: "closed_autocracy", 1: "electoral_autocracy",
             2: "electoral_democracy", 3: "liberal_democracy"}
    if not vdem_df.empty:
        row = vdem_df[(vdem_df["country_iso3"] == country_iso3) & (vdem_df["year"] == year)]
        if not row.empty:
            return _MAP.get(int(row.iloc[0]["v2x_regime"]), "unknown")
    return "unknown"


def run_q2(data: dict, config: dict = None) -> dict:
    """Main Q2 entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    corpus = data.get("corpus", pd.DataFrame())
    frame_scores = data.get("frame_scores", pd.DataFrame())
    voting = data.get("voting", pd.DataFrame())
    vdem = data.get("vdem", pd.DataFrame())
    topics = data.get("topics", [])
    probs = data.get("probs", np.array([]))
    country_year_embeddings = data.get("country_year_embeddings", pd.DataFrame())

    # Annotate corpus with regime type
    corpus = _annotate_regime(corpus, vdem)
    frame_scores = _annotate_regime(frame_scores, vdem)

    results = {}

    # 2a. Distinctive words (binary, all years)
    print("[Q2] 2a: Fightin' Words (democracy vs autocracy, all years)...")
    fw_binary = compute_distinctive_words_binary(corpus, vdem)
    results["distinctive_words_binary"] = fw_binary
    _save(fw_binary, "distinctive_words_binary.csv")

    # 2b. By decade
    print("[Q2] 2b: Fightin' Words by decade...")
    fw_decade = compute_distinctive_words_by_decade(corpus, vdem)
    results["distinctive_words_by_decade"] = fw_decade
    _save(fw_decade, "distinctive_words_by_decade.csv")

    # 2c. Topic proportions by regime
    if len(topics) > 0 and len(probs) > 0 and not corpus.empty:
        print("[Q2] 2c: Topic proportions by regime...")
        topic_df = compute_topic_by_regime_year(topics, probs, corpus, vdem)
        results["topic_by_regime_year"] = topic_df
        _save(topic_df, "topic_by_regime_year.csv")

    # 2d. Embedding distance over time
    if not country_year_embeddings.empty:
        print("[Q2] 2d: Embedding distance over time...")
        emb_dist = compute_embedding_distance_over_time(country_year_embeddings, vdem)
        results["embedding_distance"] = emb_dist
        _save(emb_dist, "embedding_distance.csv")

    # 2e. Frame ratio by regime
    print("[Q2] 2e: Frame ratio by regime...")
    frame_by_regime = compute_frame_by_regime(frame_scores, vdem)
    results["frame_by_regime_year"] = frame_by_regime
    _save(frame_by_regime, "frame_by_regime_year.csv")

    # 2f. Transition case studies
    print("[Q2] 2f: Regime transition case studies...")
    transition_cases = compute_transition_case_studies(corpus, frame_scores, vdem, country_year_embeddings)
    results["transition_cases"] = transition_cases
    _save(transition_cases, "transition_cases.csv")

    # 2g. Voting by regime
    if not voting.empty:
        print("[Q2] 2g: Voting by regime type...")
        vote_regime = compute_voting_by_regime(voting, vdem)
        results["voting_by_regime_year"] = vote_regime
        _save(vote_regime, "voting_by_regime_year.csv")

    print("[Q2] Complete.")
    return results


def _annotate_regime(df: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """Add binary_regime and regime_4way columns to a country-year DataFrame."""
    if df.empty or "country_iso3" not in df.columns:
        return df
    result = df.copy()

    if not vdem.empty:
        vdem_map = vdem.set_index(["country_iso3", "year"])["v2x_regime"].to_dict()
        result["v2x_regime"] = result.apply(
            lambda r: vdem_map.get((r["country_iso3"], r["year"]), np.nan), axis=1
        )
        result["binary_regime"] = result["v2x_regime"].apply(
            lambda x: "democracy" if (not pd.isna(x) and x >= 2) else "autocracy"
        )
        result["regime_4way"] = result["v2x_regime"].apply(
            lambda x: {0: "closed_autocracy", 1: "electoral_autocracy",
                        2: "electoral_democracy", 3: "liberal_democracy"}.get(int(x) if not pd.isna(x) else -1, "unknown")
        )
    else:
        result["binary_regime"] = "unknown"
        result["regime_4way"] = "unknown"

    return result


def compute_distinctive_words_binary(corpus: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """2a: Fightin' Words democracy vs autocracy, all years."""
    corpus = _annotate_regime(corpus, vdem)
    if corpus.empty or "binary_regime" not in corpus.columns:
        return pd.DataFrame()

    text_col = "segment_text" if "segment_text" in corpus.columns else "text"
    dem_corpus = corpus[corpus["binary_regime"] == "democracy"][text_col].fillna("").tolist()
    aut_corpus = corpus[corpus["binary_regime"] == "autocracy"][text_col].fillna("").tolist()

    if len(dem_corpus) < 10 or len(aut_corpus) < 10:
        return pd.DataFrame()

    result = fightin_words(dem_corpus, aut_corpus, top_n=30)
    result["group_a_label"] = "democracy"
    result["group_b_label"] = "autocracy"
    return result


def compute_distinctive_words_by_decade(corpus: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """2b: Fightin' Words per decade."""
    corpus = _annotate_regime(corpus, vdem)
    if corpus.empty or "binary_regime" not in corpus.columns:
        return pd.DataFrame()

    text_col = "segment_text" if "segment_text" in corpus.columns else "text"
    return fightin_words_by_decade(
        corpus, group_col="binary_regime", text_col=text_col,
        year_col="year", group_a="democracy", group_b="autocracy", top_n=20,
    )


def compute_topic_by_regime_year(
    topics: list, probs: np.ndarray, corpus: pd.DataFrame, vdem: pd.DataFrame
) -> pd.DataFrame:
    """2c: Topic proportions per regime group per year."""
    corpus = _annotate_regime(corpus, vdem)
    if corpus.empty or len(topics) == 0:
        return pd.DataFrame()

    records = []
    corpus_reset = corpus.reset_index(drop=True)

    for year, ydf in corpus_reset.groupby("year"):
        idxs = ydf.index.tolist()
        valid_idxs = [i for i in idxs if i < len(topics)]
        if not valid_idxs:
            continue

        for regime, rdf in ydf.groupby("binary_regime"):
            r_idxs = [i for i in rdf.index.tolist() if i < len(probs)]
            if not r_idxs or len(probs) == 0:
                continue
            regime_probs = probs[r_idxs]
            mean_probs = regime_probs.mean(axis=0)
            for topic_id, proportion in enumerate(mean_probs):
                records.append({
                    "year": year,
                    "regime": regime,
                    "topic_id": topic_id,
                    "proportion": float(proportion),
                })

    return pd.DataFrame(records)


def compute_embedding_distance_over_time(
    country_year_embeddings: pd.DataFrame,
    vdem: pd.DataFrame,
) -> pd.DataFrame:
    """2d: Per-year cosine distance between democracy and autocracy centroids."""
    if country_year_embeddings.empty or "embedding" not in country_year_embeddings.columns:
        return pd.DataFrame()

    cy = _annotate_regime(country_year_embeddings, vdem)
    records = []

    for year, ydf in cy.groupby("year"):
        dem = ydf[ydf["binary_regime"] == "democracy"]["embedding"].tolist()
        aut = ydf[ydf["binary_regime"] == "autocracy"]["embedding"].tolist()
        if len(dem) < 2 or len(aut) < 2:
            continue

        dem_arr = np.stack(dem)
        aut_arr = np.stack(aut)
        dist = compute_between_group_distance(dem_arr, aut_arr)
        records.append({"year": year, "distance": dist, "n_dem": len(dem), "n_aut": len(aut)})

    df = pd.DataFrame(records).sort_values("year")

    # Change-point detection
    if len(df) > 10:
        series = pd.Series(df["distance"].values, index=df["year"].values)
        breaks = compute_change_points(series)
        break_years = {yr for yr, _ in breaks}
        df["is_change_point"] = df["year"].isin(break_years)
    else:
        df["is_change_point"] = False

    return df


def compute_frame_by_regime(frame_scores: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """2e: Frame ratio per regime group per year."""
    if frame_scores.empty:
        return pd.DataFrame()

    fs = _annotate_regime(frame_scores, vdem)
    col = "frame_ratio_mean" if "frame_ratio_mean" in fs.columns else "frame_ratio"

    agg = (
        fs[fs["binary_regime"].isin(["democracy", "autocracy"])]
        .groupby(["year", "binary_regime"])
        .agg(frame_ratio_mean=(col, "mean"), frame_ratio_std=(col, "std"), n=(col, "count"))
        .reset_index()
        .rename(columns={"binary_regime": "group"})
    )

    # Also add 4-way
    agg4 = (
        fs[fs["regime_4way"] != "unknown"]
        .groupby(["year", "regime_4way"])
        .agg(frame_ratio_mean=(col, "mean"), n=(col, "count"))
        .reset_index()
        .rename(columns={"regime_4way": "group"})
    )

    return pd.concat([agg, agg4], ignore_index=True)


def compute_transition_case_studies(
    corpus: pd.DataFrame,
    frame_scores: pd.DataFrame,
    vdem: pd.DataFrame,
    country_year_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """2f: Before/after metrics for regime transition countries."""
    records = []
    col = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"

    for country, info in REGIME_TRANSITIONS.items():
        ty = info["year"]
        cdf = frame_scores[frame_scores["country_iso3"] == country].sort_values("year")
        if cdf.empty:
            continue

        pre = cdf[cdf["year"] < ty][col].mean()
        post = cdf[cdf["year"] >= ty][col].mean()

        records.append({
            "country_iso3": country,
            "transition_year": ty,
            "direction": info["direction"],
            "label": info["label"],
            "pre_frame_ratio": pre,
            "post_frame_ratio": post,
            "frame_ratio_change": post - pre,
        })

        # Also include time series for plotting
        cdf_copy = cdf.copy()
        cdf_copy["transition_year"] = ty

    # Return country-year records for transition countries (for time series plotting)
    transition_countries = list(REGIME_TRANSITIONS.keys())
    ts_df = frame_scores[frame_scores["country_iso3"].isin(transition_countries)].copy()
    for country, info in REGIME_TRANSITIONS.items():
        ts_df.loc[ts_df["country_iso3"] == country, "transition_year"] = info["year"]
        ts_df.loc[ts_df["country_iso3"] == country, "direction"] = info["direction"]

    summary = pd.DataFrame(records)
    _save(summary, "transition_summary.csv")

    return ts_df


def compute_voting_by_regime(voting: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """2g: Pro-disarmament voting rate per regime group per year."""
    if voting.empty:
        return pd.DataFrame()

    voting = _annotate_regime(voting, vdem)
    if "vote_numeric" not in voting.columns:
        return pd.DataFrame()

    disarm_votes = voting[voting.get("issue", pd.Series()) == "Arms control and disarmament"] \
        if "issue" in voting.columns else voting

    agg = (
        disarm_votes[disarm_votes["binary_regime"].isin(["democracy", "autocracy"])]
        .groupby(["year", "binary_regime"])["vote_numeric"]
        .apply(lambda x: (x == 1).sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan)
        .reset_index()
        .rename(columns={"vote_numeric": "pct_yes", "binary_regime": "group"})
    )
    return agg


def _save(df: pd.DataFrame, filename: str):
    if df is not None and not df.empty:
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"[Q2] Saved {path}")
