#!/usr/bin/env python3
"""
Arms Control NLP Pipeline — Final Version
Four research questions on arms control discourse evolution.

Usage:
    python run.py                              # Run everything
    python run.py --question q1               # Just Q1
    python run.py --question q3 --treaty att  # Just Q3 for ATT
    python run.py --shared-only               # Just shared preprocessing
    python run.py --fast                      # Use MiniLM (fast) embeddings
    python run.py --no-cache                  # Recompute embeddings even if cached
"""
import argparse
import sys
import os
import time
import traceback

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Data loading ──────────────────────────────────────────────────────────────
from src.data.load_ungdc import load_ungdc
from src.data.load_first_committee import load_first_committee
from src.data.load_voting import load_voting
from src.data.load_vdem import load_vdem
from src.data.load_treaties import load_treaty_anchors
from src.data.load_resolutions import load_resolutions
from src.data.segment import segment_arms_control
from src.data.preprocess import preprocess_corpus

# ── Shared analytics ──────────────────────────────────────────────────────────
from src.shared.embeddings import (
    embed_corpus, embed_treaty_anchors,
    get_humanitarian_anchor_embedding, get_security_anchor_embedding,
    compute_country_year_embeddings,
)
from src.shared.topics import (
    run_bertopic, topics_over_time, classify_topics,
    save_topic_model, load_topic_model,
)
from src.shared.frame_scoring import score_corpus_frames, aggregate_to_country_year
from src.shared.lexicons import HUMANITARIAN, DETERRENCE
from src.shared.temporal import compute_change_points


CHECKPOINT_DIR = "output/shared/checkpoints"

# Sentinel output files: if these exist the question is considered done
Q_SENTINELS = {
    "q1": "output/q1/frame_ratio_global.csv",
    "q2": "output/q2/distinctive_words_binary.csv",
    "q3": "output/q3/cross_treaty_comparison.csv",
    "q4": "output/q4/frame_by_nuclear_status.csv",
    "q5": "output/q5/panel_dataset.csv",
    "q6": "output/q6/drone_keyword_extraction.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Arms Control NLP Pipeline")
    parser.add_argument("--question", choices=["q1", "q2", "q3", "q4", "q5", "q6"],
                        help="Run only one research question")
    parser.add_argument("--treaty", choices=["att", "tpnw", "ottawa", "ccm"],
                        help="Filter Q3 to a specific treaty")
    parser.add_argument("--shared-only", action="store_true",
                        help="Only run shared preprocessing (embeddings, topics, frame scores)")
    parser.add_argument("--fast", action="store_true",
                        help="Use all-MiniLM-L6-v2 (fast) instead of all-mpnet-base-v2")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute embeddings even if cached")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages whose outputs already exist (safe after timeout)")
    parser.add_argument("--year-start", type=int, default=1970)
    parser.add_argument("--year-end", type=int, default=2023)
    return parser.parse_args()


def _ckpt(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, name)


def _ckpt_exists(name: str) -> bool:
    return os.path.exists(_ckpt(name))


def _save_shared_checkpoint(shared: dict):
    """Persist shared pipeline outputs that are expensive to recompute."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Scored corpus → parquet
    corpus = shared.get("corpus")
    if corpus is not None and not corpus.empty:
        # Drop embedding column — stored separately in embeddings_cache.npz
        corpus_save = corpus.drop(columns=["embedding"], errors="ignore")
        corpus_save.to_parquet(_ckpt("corpus_scored.parquet"), index=False)
        print(f"[Checkpoint] Saved corpus_scored ({len(corpus_save)} rows)")

    # Country-year embeddings → parquet (store embedding as list for parquet compat)
    cye = shared.get("country_year_embeddings")
    if cye is not None and not cye.empty:
        cye_save = cye.copy()
        if "embedding" in cye_save.columns:
            cye_save["embedding"] = cye_save["embedding"].apply(
                lambda x: x.tolist() if hasattr(x, "tolist") else x
            )
        cye_save.to_parquet(_ckpt("country_year_embeddings.parquet"), index=False)
        print(f"[Checkpoint] Saved country_year_embeddings ({len(cye_save)} rows)")


def _load_shared_checkpoint(shared: dict) -> dict:
    """Load persisted shared outputs back into the shared dict."""
    if _ckpt_exists("corpus_scored.parquet"):
        try:
            corpus = pd.read_parquet(_ckpt("corpus_scored.parquet"))
            shared["corpus"] = corpus
            print(f"[Checkpoint] Loaded corpus_scored ({len(corpus)} rows)")
        except Exception as e:
            print(f"[Checkpoint] Could not load corpus_scored: {e}")

    if _ckpt_exists("country_year_embeddings.parquet"):
        try:
            cye = pd.read_parquet(_ckpt("country_year_embeddings.parquet"))
            if "embedding" in cye.columns:
                cye["embedding"] = cye["embedding"].apply(
                    lambda x: np.array(x) if isinstance(x, list) else x
                )
            shared["country_year_embeddings"] = cye
            print(f"[Checkpoint] Loaded country_year_embeddings ({len(cye)} rows)")
        except Exception as e:
            print(f"[Checkpoint] Could not load country_year_embeddings: {e}")

    if os.path.exists("output/shared/frame_scores.csv"):
        try:
            shared["frame_scores"] = pd.read_csv("output/shared/frame_scores.csv")
            print(f"[Checkpoint] Loaded frame_scores ({len(shared['frame_scores'])} rows)")
        except Exception as e:
            print(f"[Checkpoint] Could not load frame_scores: {e}")

    return shared


def load_all_data(args) -> dict:
    """Load all data sources. Returns data dict."""
    print("\n[Pipeline] Loading data sources...")
    t0 = time.time()

    year_range = (args.year_start, args.year_end)

    # UNGDC speeches
    ungdc = load_ungdc(year_range=year_range)

    # First Committee (optional stretch goal)
    fc = load_first_committee()

    # UNGA voting
    voting = load_voting()

    # First Committee resolutions (full text)
    resolutions = load_resolutions()

    # Enrich voting frame_type using full resolution text
    if not resolutions.empty and not voting.empty:
        voting = _enrich_voting_with_resolutions(voting, resolutions)

    # V-Dem regime scores
    vdem = load_vdem()

    # Treaty anchors
    anchors = load_treaty_anchors()

    print(f"[Pipeline] Data loaded in {time.time() - t0:.1f}s")
    print(f"           UNGDC      : {len(ungdc)} speeches, {ungdc['country_iso3'].nunique()} countries")
    print(f"           Voting     : {len(voting)} records, {voting['rcid'].nunique() if 'rcid' in voting.columns else '?'} resolutions")
    print(f"           Resolutions: {len(resolutions)} First Committee texts ({(resolutions['full_text'] != '').sum()} with full text)")
    print(f"           V-Dem      : {vdem['country_iso3'].nunique()} countries")
    print(f"           Treaties   : {list(anchors.keys())}")

    # SIPRI arms transfers (for Q5 arms-trade integration)
    sipri = pd.DataFrame()
    sipri_path = os.path.join(BASE_DIR, "..", "arms-trade-network", "data", "real", "sipri_transfers.csv")
    if os.path.exists(sipri_path):
        sipri = pd.read_csv(sipri_path, low_memory=False)
        print(f"           SIPRI      : {len(sipri)} transfer records")
    else:
        print(f"           SIPRI      : not found at {sipri_path}")

    # SIPRI trade register (for Q6 drone transfers)
    trade_register = pd.DataFrame()
    trade_reg_path = os.path.join(BASE_DIR, "..", "arms-trade-network", "data", "real", "trade-register.csv")
    if os.path.exists(trade_reg_path):
        trade_register = pd.read_csv(trade_reg_path, skiprows=11, low_memory=False, encoding="latin-1")
        print(f"           Trade Reg  : {len(trade_register)} deal records")
    else:
        print(f"           Trade Reg  : not found at {trade_reg_path}")

    return {
        "ungdc": ungdc,
        "first_committee": fc,
        "voting": voting,
        "resolutions": resolutions,
        "vdem": vdem,
        "anchors": anchors,
        "sipri": sipri,
        "trade_register": trade_register,
    }


def _enrich_voting_with_resolutions(
    voting: pd.DataFrame, resolutions: pd.DataFrame
) -> pd.DataFrame:
    """
    Re-classify frame_type and treaty_flag in voting using full resolution text.
    Operates on unique (year, resolution_title) pairs — not on all 862k rows —
    then merges back, keeping the operation fast.
    """
    from src.data.load_voting import classify_resolution_frame, flag_treaty

    res = resolutions[resolutions["full_text"].str.len() > 100].copy()
    if res.empty:
        return voting

    # Pre-tokenise resolution titles by year
    res_by_year = {}
    for year, grp in res.groupby("year"):
        res_by_year[year] = [
            (set(str(r["title"]).lower().split()), r["title"], r["full_text"])
            for _, r in grp.iterrows()
        ]

    # Work on unique (year, resolution_title) pairs only
    unique_titles = (
        voting[["year", "resolution_title"]]
        .drop_duplicates()
        .dropna(subset=["resolution_title"])
    )

    lookup = {}   # (year, title) -> (frame_type, treaty_flag)
    for _, row in unique_titles.iterrows():
        year   = row["year"]
        vtitle = str(row["resolution_title"]).lower()
        if year not in res_by_year or not vtitle or vtitle == "unknown":
            continue

        vtitle_words = set(vtitle.split())
        best_score, best_combined = 0, None
        for rtitle_words, rtitle, rfull in res_by_year[year]:
            overlap = len(vtitle_words & rtitle_words) / max(len(vtitle_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_combined = rtitle + " " + rfull[:1000]

        if best_score >= 0.4 and best_combined:
            lookup[(year, row["resolution_title"])] = (
                classify_resolution_frame(best_combined),
                flag_treaty(best_combined),
            )

    if not lookup:
        return voting

    # Build a small lookup DataFrame and merge — never iterate over 862k rows
    lookup_df = pd.DataFrame([
        {"year": k[0], "resolution_title": k[1],
         "_frame_type_new": v[0], "_treaty_flag_new": v[1]}
        for k, v in lookup.items()
    ])

    voting = voting.merge(lookup_df, on=["year", "resolution_title"], how="left")
    matched = voting["_frame_type_new"].notna()
    voting.loc[matched, "frame_type"]  = voting.loc[matched, "_frame_type_new"]
    voting.loc[matched & voting["_treaty_flag_new"].notna(), "treaty_flag"] = \
        voting.loc[matched & voting["_treaty_flag_new"].notna(), "_treaty_flag_new"]
    voting = voting.drop(columns=["_frame_type_new", "_treaty_flag_new"])

    print(f"[Pipeline] Enriched {matched.sum()} voting records "
          f"({len(lookup)} unique resolutions matched).")
    return voting


def run_shared_pipeline(data: dict, args, resume: bool = False) -> dict:
    """Run shared computations (segment, embed, topics, frame scores)."""
    print("\n[Pipeline] Running shared pipeline...")
    t0 = time.time()

    model_name = "all-MiniLM-L6-v2" if args.fast else "all-mpnet-base-v2"
    print(f"[Pipeline] Embedding model: {model_name}")

    # ── Resume: check if shared outputs already exist ─────────────────────────
    shared_complete = (
        resume
        and os.path.exists("output/shared/embeddings_cache.npz")
        and os.path.exists("output/shared/anchor_embeddings.npz")
        and os.path.exists("output/shared/frame_scores.csv")
        and _ckpt_exists("corpus_scored.parquet")
        and _ckpt_exists("country_year_embeddings.parquet")
    )
    if shared_complete:
        print("[Checkpoint] Shared pipeline outputs found — loading from disk.")
        shared = {}
        shared = _load_shared_checkpoint(shared)
        # Still need embeddings + anchors in memory for question modules
        embeddings, index_df = embed_corpus(
            shared["corpus"], text_col="segment_text", model_name=model_name,
            cache_path="output/shared/embeddings_cache.npz",
        )
        anchor_embeddings = embed_treaty_anchors(
            data["anchors"], model_name=model_name,
            cache_path="output/shared/anchor_embeddings.npz",
        )
        h_anchor = get_humanitarian_anchor_embedding(anchor_embeddings)
        d_anchor = get_security_anchor_embedding(anchor_embeddings)
        topic_model, topics, probs = load_topic_model("output/shared/topics/")
        if topic_model is None:
            topic_model, topics, probs = None, [], np.array([])
        topic_classifications = pd.DataFrame()
        if topic_model is not None:
            topic_classifications = classify_topics(topic_model, HUMANITARIAN, DETERRENCE)
        print(f"[Checkpoint] Shared pipeline restored in {time.time() - t0:.1f}s")
        return {
            "corpus": shared["corpus"],
            "frame_scores": shared["frame_scores"],
            "embeddings": embeddings,
            "index_df": index_df,
            "anchor_embeddings": anchor_embeddings,
            "h_anchor": h_anchor,
            "d_anchor": d_anchor,
            "topic_model": topic_model,
            "topics": topics,
            "probs": probs,
            "topic_classifications": topic_classifications,
            "country_year_embeddings": shared["country_year_embeddings"],
            "resolutions": data.get("resolutions", pd.DataFrame()),
        }
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Preprocess
    ungdc = preprocess_corpus(data["ungdc"])

    # 2. Segment arms-control passages
    print("[Pipeline] Segmenting arms-control passages...")
    corpus = segment_arms_control(ungdc, method="keyword")

    # 3. Embed corpus
    print("[Pipeline] Embedding corpus...")
    embeddings, index_df = embed_corpus(
        corpus,
        text_col="segment_text",
        model_name=model_name,
        cache_path="output/shared/embeddings_cache.npz",
        force_recompute=args.no_cache,
    )

    # 4. Embed treaty anchors
    print("[Pipeline] Embedding treaty anchors...")
    anchor_embeddings = embed_treaty_anchors(
        data["anchors"],
        model_name=model_name,
        cache_path="output/shared/anchor_embeddings.npz",
    )

    # 5. Derive frame anchor vectors
    h_anchor = get_humanitarian_anchor_embedding(anchor_embeddings)
    d_anchor = get_security_anchor_embedding(anchor_embeddings)

    # 6. Score frames
    print("[Pipeline] Scoring frames (lexicon + embedding)...")
    scored_corpus = score_corpus_frames(
        corpus, embeddings, index_df, h_anchor, d_anchor,
        text_col="segment_text",
    )
    frame_scores = aggregate_to_country_year(scored_corpus)
    os.makedirs("output/shared", exist_ok=True)
    frame_scores.to_csv("output/shared/frame_scores.csv", index=False)
    print(f"[Pipeline] Frame scores: {len(frame_scores)} country-year observations")

    # 7. Run BERTopic (load from cache if available)
    topic_model, topics, probs = load_topic_model("output/shared/topics/")
    if topic_model is None and topics is None:
        print("[Pipeline] Running BERTopic...")
        texts = corpus["segment_text"].fillna("").tolist()
        timestamps = corpus["year"].tolist()
        try:
            topic_model, topics, probs = run_bertopic(texts, nr_topics=30, model_name=model_name)
            save_topic_model(topic_model, topics, probs, "output/shared/topics/")
        except Exception as e:
            print(f"[Pipeline] BERTopic failed: {e}. Topics unavailable.")
            topic_model, topics, probs = None, [], np.array([])
    else:
        print("[Pipeline] Loaded cached topic model.")

    # Classify topics as humanitarian/deterrence
    topic_classifications = pd.DataFrame()
    if topic_model is not None:
        topic_classifications = classify_topics(topic_model, HUMANITARIAN, DETERRENCE)

    # 8. Compute country-year embeddings for distance analyses
    country_year_embeddings = compute_country_year_embeddings(corpus, embeddings, index_df)

    # 9. Save checkpoints
    _save_shared_checkpoint({
        "corpus": scored_corpus,
        "country_year_embeddings": country_year_embeddings,
    })

    print(f"[Pipeline] Shared pipeline complete in {time.time() - t0:.1f}s")

    return {
        "corpus": scored_corpus,
        "frame_scores": frame_scores,
        "embeddings": embeddings,
        "index_df": index_df,
        "anchor_embeddings": anchor_embeddings,
        "h_anchor": h_anchor,
        "d_anchor": d_anchor,
        "topic_model": topic_model,
        "topics": topics,
        "probs": probs,
        "topic_classifications": topic_classifications,
        "country_year_embeddings": country_year_embeddings,
        "resolutions": data.get("resolutions", pd.DataFrame()),
    }


def run_era_detection(all_results: dict):
    """Cross-question era detection: find consensus structural breaks."""
    print("\n[Pipeline] Running era detection...")
    os.makedirs("output/shared", exist_ok=True)

    signals = {}

    if "q1" in all_results and "frame_ratio_global" in all_results["q1"]:
        ts = all_results["q1"]["frame_ratio_global"].set_index("year")
        col = "frame_ratio_mean" if "frame_ratio_mean" in ts.columns else ts.columns[0]
        signals["q1_frame_ratio"] = ts[col]

    if "q2" in all_results and "embedding_distance" in all_results["q2"]:
        ts = all_results["q2"]["embedding_distance"].set_index("year")
        col = next((c for c in ["distance", "cosine_distance"] if c in ts.columns), None)
        if col:
            signals["q2_democracy_autocracy_distance"] = ts[col]

    if "q4" in all_results and "embedding_distance_nws_nnws" in all_results["q4"]:
        ts = all_results["q4"]["embedding_distance_nws_nnws"].set_index("year")
        col = next((c for c in ["distance", "cosine_distance"] if c in ts.columns), None)
        if col:
            signals["q4_nws_nnws_distance"] = ts[col]

    records = []
    for signal_name, series in signals.items():
        series_clean = series.dropna().sort_index()
        if len(series_clean) < 10:
            continue
        breaks = compute_change_points(series_clean)
        for year, magnitude in breaks:
            records.append({
                "signal": signal_name,
                "break_year": year,
                "magnitude": magnitude,
                "direction": "up" if magnitude > 0 else "down",
            })

    era_df = pd.DataFrame(records)
    if not era_df.empty:
        era_df.to_csv("output/shared/era_detection.csv", index=False)

        # Consensus: breaks where 2+ signals agree within ±2 years
        consensus = (
            era_df.groupby("break_year")["signal"].count()
            .reset_index()
            .rename(columns={"signal": "n_signals"})
        )
        consensus = consensus[consensus["n_signals"] >= 2]
        if not consensus.empty:
            print(f"[Era Detection] Consensus break years: {consensus['break_year'].tolist()}")
            consensus.to_csv("output/shared/era_consensus.csv", index=False)

    return era_df


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 65)
    print("  Arms Control NLP Pipeline")
    print("  Four Research Questions on Arms Control Discourse")
    print("=" * 65)

    # Load all data
    data = load_all_data(args)

    # Run shared preprocessing
    shared = run_shared_pipeline(data, args, resume=args.resume)

    # Merge for question modules
    pipeline_data = {**data, **shared}

    if args.shared_only:
        print("\n[Pipeline] --shared-only flag set. Stopping after shared pipeline.")
        print(f"[Pipeline] Total time: {time.time() - t_start:.1f}s")
        return

    # Determine which questions to run
    questions_to_run = [args.question] if args.question else ["q1", "q2", "q3", "q4", "q5", "q6"]
    all_results = {}

    for q in questions_to_run:
        # Skip if resuming and sentinel output already exists
        if args.resume and os.path.exists(Q_SENTINELS.get(q, "")):
            print(f"\n[Checkpoint] {q.upper()} output found — skipping.")
            all_results[q] = {}
            continue

        print(f"\n{'=' * 65}")
        print(f"  Running {q.upper()}")
        print(f"{'=' * 65}")
        t_q = time.time()

        try:
            if q == "q1":
                from src.questions.q1_humanitarian_vs_deterrence import run_q1
                results = run_q1(pipeline_data)
                all_results["q1"] = results
                if not args.no_plots:
                    from src.viz.q1_plots import run_q1_plots
                    run_q1_plots(results)

            elif q == "q2":
                from src.questions.q2_democracy_vs_autocracy import run_q2
                results = run_q2(pipeline_data)
                all_results["q2"] = results
                if not args.no_plots:
                    from src.viz.q2_plots import run_q2_plots
                    run_q2_plots(results)

            elif q == "q3":
                from src.questions.q3_rhetoric_before_after import run_q3
                results = run_q3(pipeline_data, treaty_filter=args.treaty)
                all_results["q3"] = results
                if not args.no_plots:
                    from src.viz.q3_plots import run_q3_plots
                    run_q3_plots(results)

            elif q == "q4":
                from src.questions.q4_nuclear_vs_nonnuclear import run_q4
                results = run_q4(pipeline_data)
                all_results["q4"] = results
                if not args.no_plots:
                    from src.viz.q4_plots import run_q4_plots
                    run_q4_plots(results)
                    from src.viz.treaty_proximity_infographic import run_infographic
                    run_infographic(pipeline_data)

            elif q == "q5":
                from src.questions.q5_regime_treaty_divide import run_q5
                results = run_q5(pipeline_data)
                all_results["q5"] = results
                if not args.no_plots:
                    from src.viz.q5_plots import run_q5_plots
                    run_q5_plots(results)

            elif q == "q6":
                from src.questions.q6_drones_autonomous import run_q6
                results = run_q6(pipeline_data)
                all_results["q6"] = results
                if not args.no_plots:
                    from src.viz.q6_plots import run_q6_plots
                    run_q6_plots(results)

            print(f"[{q.upper()}] Complete in {time.time() - t_q:.1f}s")

        except Exception as e:
            print(f"[{q.upper()}] ERROR: {e}")
            traceback.print_exc()
            print(f"[{q.upper()}] Skipping — continuing with next question.")

    # Cross-question era detection
    if len(questions_to_run) > 1 and all_results:
        try:
            era_df = run_era_detection(all_results)
            if not args.no_plots:
                from src.viz.shared_plots import run_shared_plots
                shared_viz_results = {
                    "era_detection": era_df,
                    "q1_global_ts": all_results.get("q1", {}).get("frame_ratio_global", pd.DataFrame()),
                    "q2_embedding_distance": all_results.get("q2", {}).get("embedding_distance", pd.DataFrame()),
                    "q4_embedding_distance": all_results.get("q4", {}).get("embedding_distance_nws_nnws", pd.DataFrame()),
                }
                run_shared_plots(shared_viz_results)
        except Exception as e:
            print(f"[Era Detection] ERROR: {e}")

    # Summary
    print(f"\n{'=' * 65}")
    print("  Pipeline Complete")
    print(f"{'=' * 65}")
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print(f"  Questions run: {', '.join(q.upper() for q in questions_to_run)}")
    corpus = pipeline_data.get("corpus", pd.DataFrame())
    if not corpus.empty:
        print(f"  Corpus: {len(corpus)} documents, "
              f"{corpus['country_iso3'].nunique()} countries, "
              f"{corpus['year'].min()}-{corpus['year'].max()}")
    print(f"  Outputs in: output/q1/, output/q2/, output/q3/, output/q4/, output/q5/, output/shared/")
    print("=" * 65)


if __name__ == "__main__":
    main()
