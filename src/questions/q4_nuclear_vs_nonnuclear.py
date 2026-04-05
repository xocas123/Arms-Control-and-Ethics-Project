"""
Q4: Do nuclear and non-nuclear states live in different rhetorical worlds?

Analyses:
4a. Distinctive vocabulary by nuclear status (Fightin' Words: NWS vs NNWS)
4b. Distinctive vocabulary by decade
4c. Frame ratio by nuclear status over time (four groups)
4d. Treaty anchor distances over time (NPT vs TPNW similarity by group)
4e. Concept-level sentiment by nuclear status
4f. NWS-NNWS embedding distance over time (with change-point detection)
4g. P5 internal comparison over time (pairwise P5 similarity)
4h. Voting gap by nuclear status over time
4i. Variance within groups over time
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations

from src.data.groups import NWS, DE_FACTO_NUCLEAR, NUCLEAR_UMBRELLA, get_nuclear_status, get_nnws
from src.shared.distinctive_words import fightin_words, fightin_words_by_decade
from src.shared.concept_sentiment import score_corpus_concept_sentiment, CONCEPTS
from src.shared.embeddings import cosine_sim, compute_centroid, compute_pairwise_distances
from src.shared.temporal import compute_change_points, compute_within_group_variance

OUTPUT_DIR = "output/q4"

NUCLEAR_GROUP_MAP = {
    "nws": NWS,
    "de_facto": DE_FACTO_NUCLEAR,
    "umbrella": NUCLEAR_UMBRELLA,
}


def _annotate_nuclear(df: pd.DataFrame) -> pd.DataFrame:
    """Add nuclear_group column using historically accurate per-year status."""
    if df.empty or "country_iso3" not in df.columns:
        return df
    result = df.copy()
    if "year" in result.columns:
        result["nuclear_group"] = result.apply(
            lambda r: get_nuclear_status(r["country_iso3"], int(r["year"])), axis=1
        )
    else:
        result["nuclear_group"] = result["country_iso3"].apply(get_nuclear_status)
    return result


def run_q4(data: dict, config: dict = None) -> dict:
    """Main Q4 entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    corpus = data.get("corpus", pd.DataFrame())
    frame_scores = data.get("frame_scores", pd.DataFrame())
    voting = data.get("voting", pd.DataFrame())
    anchor_embeddings = data.get("anchor_embeddings", {})
    country_year_embeddings = data.get("country_year_embeddings", pd.DataFrame())

    corpus = _annotate_nuclear(corpus)
    frame_scores = _annotate_nuclear(frame_scores)
    country_year_embeddings = _annotate_nuclear(country_year_embeddings)

    results = {}

    # 4a. Distinctive words NWS vs NNWS
    print("[Q4] 4a: Fightin' Words (NWS vs NNWS)...")
    fw = compute_distinctive_words_nuclear(corpus)
    results["distinctive_words_nuclear"] = fw
    _save(fw, "distinctive_words_nuclear.csv")

    # 4b. By decade
    print("[Q4] 4b: Fightin' Words by decade...")
    fw_dec = compute_distinctive_words_nuclear_decades(corpus)
    results["distinctive_words_nuclear_decades"] = fw_dec
    _save(fw_dec, "distinctive_words_nuclear_decades.csv")

    # 4c. Frame ratio by nuclear status
    print("[Q4] 4c: Frame ratio by nuclear status...")
    frame_nuclear = compute_frame_by_nuclear_status(frame_scores)
    results["frame_by_nuclear_status"] = frame_nuclear
    _save(frame_nuclear, "frame_by_nuclear_status.csv")

    # 4d. Treaty anchor distances
    if anchor_embeddings and not country_year_embeddings.empty:
        print("[Q4] 4d: Treaty anchor distances...")
        anchor_dist = compute_anchor_distances_nuclear(country_year_embeddings, anchor_embeddings)
        results["anchor_distances"] = anchor_dist
        _save(anchor_dist, "anchor_distances.csv")

    # 4e. Concept sentiment
    if not corpus.empty:
        print("[Q4] 4e: Concept-level sentiment by nuclear status...")
        concept_sent = compute_concept_sentiment_nuclear(corpus)
        results["concept_sentiment"] = concept_sent
        _save(concept_sent, "concept_sentiment.csv")

    # 4f. NWS-NNWS embedding distance over time
    if not country_year_embeddings.empty:
        print("[Q4] 4f: NWS-NNWS embedding distance over time...")
        emb_dist = compute_nws_nnws_distance(country_year_embeddings)
        results["embedding_distance_nws_nnws"] = emb_dist
        _save(emb_dist, "embedding_distance_nws_nnws.csv")

    # 4g. P5 internal similarity
    if not country_year_embeddings.empty:
        print("[Q4] 4g: P5 internal pairwise similarity...")
        p5_sim = compute_p5_internal_similarity(country_year_embeddings)
        results["p5_internal_similarity"] = p5_sim
        _save(p5_sim, "p5_internal_similarity.csv")

    # 4h. Voting gap
    if not voting.empty:
        print("[Q4] 4h: Voting gap by nuclear status...")
        vote_gap = compute_voting_gap_nuclear(voting)
        results["voting_gap"] = vote_gap
        _save(vote_gap, "voting_gap.csv")

    # 4i. Within-group variance
    print("[Q4] 4i: Within-group variance...")
    wg_var = compute_within_group_variance_q4(country_year_embeddings, frame_scores)
    results["within_group_variance"] = wg_var
    _save(wg_var, "within_group_variance.csv")

    print("[Q4] Complete.")
    return results


def compute_distinctive_words_nuclear(corpus: pd.DataFrame) -> pd.DataFrame:
    """4a: Fightin' Words NWS vs NNWS, all years."""
    if corpus.empty or "nuclear_group" not in corpus.columns:
        return pd.DataFrame()

    text_col = "segment_text" if "segment_text" in corpus.columns else "text"
    nws_corpus = corpus[corpus["nuclear_group"] == "nws"][text_col].fillna("").tolist()
    nnws_corpus = corpus[corpus["nuclear_group"] == "nnws"][text_col].fillna("").tolist()

    if len(nws_corpus) < 5 or len(nnws_corpus) < 5:
        return pd.DataFrame()

    result = fightin_words(nws_corpus, nnws_corpus, top_n=30)
    result["group_a_label"] = "NWS"
    result["group_b_label"] = "NNWS"
    return result


def compute_distinctive_words_nuclear_decades(corpus: pd.DataFrame) -> pd.DataFrame:
    """4b: Fightin' Words by decade for nuclear status."""
    if corpus.empty or "nuclear_group" not in corpus.columns:
        return pd.DataFrame()

    text_col = "segment_text" if "segment_text" in corpus.columns else "text"
    return fightin_words_by_decade(
        corpus, group_col="nuclear_group", text_col=text_col,
        year_col="year", group_a="nws", group_b="nnws", top_n=20,
    )


def compute_frame_by_nuclear_status(frame_scores: pd.DataFrame) -> pd.DataFrame:
    """4c: Frame ratio per nuclear status group per year."""
    if frame_scores.empty or "nuclear_group" not in frame_scores.columns:
        return pd.DataFrame()

    col = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"
    agg = (
        frame_scores.groupby(["year", "nuclear_group"])
        .agg(
            frame_ratio_mean=(col, "mean"),
            frame_ratio_std=(col, "std"),
            n=(col, "count"),
        )
        .reset_index()
        .rename(columns={"nuclear_group": "group"})
    )
    return agg


def compute_anchor_distances_nuclear(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
) -> pd.DataFrame:
    """4d: Per nuclear group per year: NPT and TPNW similarity."""
    if country_year_embeddings.empty or "embedding" not in country_year_embeddings.columns:
        return pd.DataFrame()

    npt_anchor = anchor_embeddings.get("npt_1968", anchor_embeddings.get("npt", {})).get("mean_embedding")
    tpnw_anchor = anchor_embeddings.get("tpnw_2017", anchor_embeddings.get("tpnw", {})).get("mean_embedding")

    records = []
    for (year, group), gdf in country_year_embeddings.groupby(["year", "nuclear_group"]):
        embs = [e for e in gdf["embedding"] if isinstance(e, np.ndarray)]
        if not embs:
            continue

        centroid = np.mean(embs, axis=0)
        row = {"year": year, "group": group}
        if npt_anchor is not None:
            row["npt_similarity"] = float(cosine_sim(centroid, npt_anchor))
        if tpnw_anchor is not None:
            row["tpnw_similarity"] = float(cosine_sim(centroid, tpnw_anchor))
        records.append(row)

    return pd.DataFrame(records)


def compute_concept_sentiment_nuclear(corpus: pd.DataFrame) -> pd.DataFrame:
    """4e: Concept-level sentiment by nuclear status group over time."""
    if corpus.empty or "nuclear_group" not in corpus.columns:
        return pd.DataFrame()

    text_col = "segment_text" if "segment_text" in corpus.columns else "text"
    print("[Q4] Scoring concept sentiment (this may take a while)...")
    sent_df = score_corpus_concept_sentiment(corpus, text_col=text_col, concepts=CONCEPTS)

    if sent_df.empty:
        return pd.DataFrame()

    # Add nuclear group
    sent_df = sent_df.merge(
        corpus[["country_iso3", "year", "nuclear_group"]].drop_duplicates(),
        on=["country_iso3", "year"], how="left",
    )

    agg = (
        sent_df.groupby(["nuclear_group", "year", "concept"])
        .agg(mean_sentiment=("mean_sentiment", "mean"), n_mentions=("n_mentions", "sum"))
        .reset_index()
        .rename(columns={"nuclear_group": "group"})
    )
    return agg


def compute_nws_nnws_distance(country_year_embeddings: pd.DataFrame) -> pd.DataFrame:
    """4f: Per-year cosine distance between NWS and NNWS centroids."""
    if country_year_embeddings.empty or "embedding" not in country_year_embeddings.columns:
        return pd.DataFrame()

    records = []
    for year, ydf in country_year_embeddings.groupby("year"):
        nws_embs = [e for e in ydf[ydf["nuclear_group"].str.upper() == "NWS"]["embedding"] if isinstance(e, np.ndarray)]
        nnws_embs = [e for e in ydf[ydf["nuclear_group"].str.upper() == "NNWS"]["embedding"] if isinstance(e, np.ndarray)]

        if len(nws_embs) < 2 or len(nnws_embs) < 5:
            continue

        nws_centroid = np.mean(nws_embs, axis=0)
        nnws_centroid = np.mean(nnws_embs, axis=0)
        dist = 1.0 - cosine_sim(nws_centroid, nnws_centroid)

        records.append({
            "year": year,
            "distance": float(dist),
            "n_nws": len(nws_embs),
            "n_nnws": len(nnws_embs),
        })

    if not records:
        return pd.DataFrame(columns=["year", "distance", "n_nws", "n_nnws", "is_change_point", "change_point_magnitude"])
    df = pd.DataFrame(records).sort_values("year")

    if len(df) > 10:
        series = pd.Series(df["distance"].values, index=df["year"].values)
        breaks = compute_change_points(series)
        break_years = {yr for yr, _ in breaks}
        df["is_change_point"] = df["year"].isin(break_years)
        df["change_point_magnitude"] = df.apply(
            lambda r: next((m for y, m in breaks if y == r["year"]), 0.0), axis=1
        )
    else:
        df["is_change_point"] = False
        df["change_point_magnitude"] = 0.0

    return df


def compute_p5_internal_similarity(country_year_embeddings: pd.DataFrame) -> pd.DataFrame:
    """4g: Pairwise cosine similarity within P5 per year."""
    if country_year_embeddings.empty or "embedding" not in country_year_embeddings.columns:
        return pd.DataFrame()

    p5_embs = country_year_embeddings[country_year_embeddings["country_iso3"].isin(NWS)]
    records = []

    for year, ydf in p5_embs.groupby("year"):
        country_emb = {}
        for _, row in ydf.iterrows():
            emb = row["embedding"]
            if isinstance(emb, np.ndarray):
                country_emb[row["country_iso3"]] = emb

        for a, b in combinations(sorted(country_emb.keys()), 2):
            sim = cosine_sim(country_emb[a], country_emb[b])
            records.append({
                "country_a": a,
                "country_b": b,
                "year": year,
                "similarity": float(sim),
            })

    return pd.DataFrame(records)


def compute_voting_gap_nuclear(voting: pd.DataFrame) -> pd.DataFrame:
    """4h: Pro-disarmament voting rate per nuclear status group per year."""
    if voting.empty:
        return pd.DataFrame()

    voting = _annotate_nuclear(voting)
    if "vote_numeric" not in voting.columns:
        return pd.DataFrame()

    records = []
    for (year, group), gdf in voting.groupby(["year", "nuclear_group"]):
        # All disarmament
        disarm = gdf[gdf.get("issue", pd.Series()) == "Arms control and disarmament"] \
            if "issue" in gdf.columns else gdf
        pct_yes_disarm = (disarm["vote_numeric"] == 1).sum() / disarm["vote_numeric"].notna().sum() \
            if disarm["vote_numeric"].notna().sum() > 0 else np.nan

        # Nuclear-specific
        nuclear = gdf[gdf.get("issue", pd.Series()) == "Nuclear weapons and nuclear material"] \
            if "issue" in gdf.columns else pd.DataFrame()
        pct_yes_nuclear = (nuclear["vote_numeric"] == 1).sum() / nuclear["vote_numeric"].notna().sum() \
            if not nuclear.empty and nuclear["vote_numeric"].notna().sum() > 0 else np.nan

        # TPNW-specific
        tpnw = gdf[gdf.get("treaty_flag", pd.Series()) == "tpnw"] if "treaty_flag" in gdf.columns else pd.DataFrame()
        pct_yes_tpnw = (tpnw["vote_numeric"] == 1).sum() / tpnw["vote_numeric"].notna().sum() \
            if not tpnw.empty and tpnw["vote_numeric"].notna().sum() > 0 else np.nan

        records.append({
            "year": year,
            "group": group,
            "pct_yes_disarmament": float(pct_yes_disarm) if not np.isnan(pct_yes_disarm) else None,
            "pct_yes_nuclear": float(pct_yes_nuclear) if not np.isnan(pct_yes_nuclear) else None,
            "pct_yes_tpnw": float(pct_yes_tpnw) if not np.isnan(pct_yes_tpnw) else None,
        })

    return pd.DataFrame(records)


def compute_within_group_variance_q4(
    country_year_embeddings: pd.DataFrame,
    frame_scores: pd.DataFrame,
) -> pd.DataFrame:
    """4i: Per-year within-group std of frame_ratio and embedding variance."""
    records = []

    # Frame ratio variance from frame_scores
    if not frame_scores.empty and "nuclear_group" in frame_scores.columns:
        col = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"
        for (year, group), gdf in frame_scores.groupby(["year", "nuclear_group"]):
            records.append({
                "year": year,
                "group": group,
                "frame_ratio_std": float(gdf[col].std()) if len(gdf) > 1 else 0.0,
                "n": len(gdf),
            })

    df = pd.DataFrame(records)

    # Embedding variance
    if not country_year_embeddings.empty and "embedding" in country_year_embeddings.columns:
        emb_var_records = []
        for (year, group), gdf in country_year_embeddings.groupby(["year", "nuclear_group"]):
            embs = [e for e in gdf["embedding"] if isinstance(e, np.ndarray)]
            if len(embs) < 2:
                continue
            emb_arr = np.stack(embs)
            var = compute_within_group_variance(emb_arr)
            emb_var_records.append({"year": year, "group": group, "embedding_variance": float(var)})

        emb_df = pd.DataFrame(emb_var_records)
        if not df.empty and not emb_df.empty:
            df = df.merge(emb_df, on=["year", "group"], how="outer")
        elif not emb_df.empty:
            df = emb_df

    return df


def _save(df: pd.DataFrame, filename: str):
    if df is not None and not df.empty:
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"[Q4] Saved {path}")
