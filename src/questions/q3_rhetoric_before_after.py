"""
Q3: Does rhetoric lead or follow treaty ratification?

Analyses:
3a. Anchor similarity trajectory per country (±10 years around ratification)
3b. Average adoption curve (ratifiers vs signatories vs non-signatories vs opponents)
3c. Lead/lag statistical test (Wilcoxon signed-rank on pre vs post slopes)
3d. Cross-treaty comparison
3e. Lexicon-based cross-validation
3f. Outlier identification
3g. Voting adoption curve
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from src.data.groups import (
    ATT_PARTIES, ATT_SIGNATORIES_ONLY,
    TPNW_PARTIES, OTTAWA_PARTIES, CCM_PARTIES,
    TPNW_OPPONENTS, NWS,
    get_treaty_status,
)
from src.shared.lexicons import count_treaty_lexicon, TREATY_LEXICONS
from src.shared.embeddings import cosine_sim
from src.shared.temporal import compute_linear_slope, normalize_to_event

OUTPUT_DIR = "output/q3"
TREATIES = ["att", "tpnw", "ottawa", "ccm"]

# Map Q3 short names to anchors.json keys (which include year suffix)
TREATY_ANCHOR_KEYS = {
    "att": "att_2013",
    "tpnw": "tpnw_2017",
    "ottawa": "ottawa_1997",
    "ccm": "ccm_2008",
}

# Adoption year used as reference for non-party countries' year_relative
TREATY_ADOPTION_YEAR = {
    "att": 2013,
    "tpnw": 2017,
    "ottawa": 1997,
    "ccm": 2008,
}

TREATY_PARTY_DICTS = {
    "att": ATT_PARTIES,
    "tpnw": TPNW_PARTIES,
    "ottawa": OTTAWA_PARTIES,
    "ccm": CCM_PARTIES,
}


def run_q3(data: dict, config: dict = None, treaty_filter: str = None) -> dict:
    """Main Q3 entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/similarity_trajectories", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/adoption_curves", exist_ok=True)

    corpus = data.get("corpus", pd.DataFrame())
    country_year_embeddings = data.get("country_year_embeddings", pd.DataFrame())
    anchor_embeddings = data.get("anchor_embeddings", {})
    voting = data.get("voting", pd.DataFrame())

    treaties_to_run = [treaty_filter] if treaty_filter else TREATIES
    results = {}

    all_treaty_results = {}

    for treaty in treaties_to_run:
        print(f"[Q3] Processing treaty: {treaty.upper()}...")
        treaty_results = {}

        # 3a. Similarity trajectories per country
        print(f"[Q3] 3a: Similarity trajectories for {treaty}...")
        trajectories = compute_similarity_trajectories(
            country_year_embeddings, anchor_embeddings, treaty
        )
        treaty_results["trajectories"] = trajectories
        if not trajectories.empty:
            trajectories.to_csv(
                f"{OUTPUT_DIR}/similarity_trajectories/{treaty}_trajectories.csv", index=False
            )

        # 3b. Average adoption curve
        print(f"[Q3] 3b: Adoption curves for {treaty}...")
        curves = compute_adoption_curves(trajectories, treaty)
        treaty_results["curves"] = curves
        if not curves.empty:
            curves.to_csv(f"{OUTPUT_DIR}/adoption_curves/{treaty}_average_curve.csv", index=False)
        results[f"adoption_curves_{treaty}"] = curves
        results[f"similarity_trajectories_{treaty}"] = trajectories

        # 3c. Lead/lag test
        print(f"[Q3] 3c: Lead/lag test for {treaty}...")
        lead_lag = compute_lead_lag_test(trajectories)
        treaty_results["lead_lag"] = lead_lag
        all_treaty_results[treaty] = treaty_results

        # 3e. Lexicon cross-validation
        if not corpus.empty:
            print(f"[Q3] 3e: Lexicon cross-validation for {treaty}...")
            lex_cv = compute_lexicon_crossvalidation(corpus, treaty)
            results[f"lexicon_xval_{treaty}"] = lex_cv

        # 3f. Outliers
        print(f"[Q3] 3f: Outlier identification for {treaty}...")
        outliers = identify_outliers(trajectories, corpus, treaty)
        results[f"outliers_{treaty}"] = outliers

        # 3g. Voting adoption curve
        if not voting.empty:
            print(f"[Q3] 3g: Voting adoption curve for {treaty}...")
            vote_curve = compute_voting_adoption_curve(voting, treaty)
            results[f"voting_adoption_{treaty}"] = vote_curve

    # 3c. Save all lead/lag tests
    lead_lag_records = []
    for treaty in treaties_to_run:
        ll = all_treaty_results.get(treaty, {}).get("lead_lag", {})
        if ll:
            ll["treaty"] = treaty
            lead_lag_records.append(ll)
    if lead_lag_records:
        ll_df = pd.DataFrame(lead_lag_records)
        ll_df.to_csv(f"{OUTPUT_DIR}/lead_lag_tests.csv", index=False)
        results["lead_lag_tests"] = ll_df

    # 3d. Cross-treaty comparison
    if len(treaties_to_run) > 1:
        print("[Q3] 3d: Cross-treaty comparison...")
        cross = compute_cross_treaty_comparison(all_treaty_results)
        results["cross_treaty_comparison"] = cross
        if not cross.empty:
            cross.to_csv(f"{OUTPUT_DIR}/cross_treaty_comparison.csv", index=False)

    # Combine outliers
    all_outliers = [results.get(f"outliers_{t}", pd.DataFrame()) for t in treaties_to_run]
    all_outliers = [df for df in all_outliers if not df.empty]
    if all_outliers:
        combined_outliers = pd.concat(all_outliers, ignore_index=True)
        combined_outliers.to_csv(f"{OUTPUT_DIR}/outliers.csv", index=False)
        results["outliers"] = combined_outliers

    print("[Q3] Complete.")
    return results


def compute_similarity_trajectories(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    treaty: str,
    window: int = 10,
) -> pd.DataFrame:
    """3a: Cosine similarity trajectory per country around ratification."""
    anchor_key = TREATY_ANCHOR_KEYS.get(treaty, treaty)
    if country_year_embeddings.empty or anchor_key not in anchor_embeddings:
        return pd.DataFrame()

    treaty_anchor = anchor_embeddings[anchor_key].get("mean_embedding")
    if treaty_anchor is None:
        return pd.DataFrame()

    parties = TREATY_PARTY_DICTS.get(treaty, {})
    signatories = ATT_SIGNATORIES_ONLY if treaty == "att" else {}
    opponents = set(TPNW_OPPONENTS) if treaty == "tpnw" else set(NWS) if treaty in ["att"] else set()

    records = []
    cy = country_year_embeddings.copy()

    for _, row in cy.iterrows():
        country = row["country_iso3"]
        year = int(row["year"])
        emb = row["embedding"]
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            try:
                emb = np.array(emb, dtype=float)
            except (TypeError, ValueError):
                continue
        if emb.ndim == 0 or len(emb) == 0:
            continue

        sim = cosine_sim(emb, treaty_anchor)

        # Assign group
        if country in parties:
            group = "ratifiers"
            event_year = parties[country]
        elif country in signatories:
            group = "signatories_only"
            event_year = signatories[country]
        elif country in opponents:
            group = "opponents"
            event_year = None
        else:
            group = "non_signatories"
            event_year = None

        # Compute relative year:
        # - ratifiers/signatories: relative to their event year
        # - non-parties: relative to treaty adoption year (so all groups share same x-axis)
        adoption_year = TREATY_ADOPTION_YEAR.get(treaty, 2000)
        if event_year:
            year_relative = year - event_year
            if abs(year_relative) > window:
                continue
        else:
            year_relative = year - adoption_year
            if abs(year_relative) > window:
                continue

        records.append({
            "country_iso3": country,
            "year": year,
            "year_relative": year_relative,
            "similarity": float(sim),
            "group": group,
            "event_year": event_year,
        })

    df = pd.DataFrame(records)
    if df.empty:
        print(f"[Q3] WARNING: No trajectory records for {treaty} — anchor_key={TREATY_ANCHOR_KEYS.get(treaty, treaty)}, "
              f"n_cye_rows={len(cy)}, n_parties={len(parties)}")
    else:
        group_counts = df["group"].value_counts().to_dict()
        print(f"[Q3] Trajectories for {treaty}: {group_counts}")
    return df


def compute_adoption_curves(trajectories: pd.DataFrame, treaty: str) -> pd.DataFrame:
    """3b: Average similarity trajectory per group."""
    if trajectories.empty or "year_relative" not in trajectories.columns:
        return pd.DataFrame()

    ratifiers = trajectories[trajectories["group"] == "ratifiers"]
    if ratifiers.empty:
        return pd.DataFrame()

    records = []
    for group, gdf in trajectories.groupby("group"):
        if gdf["year_relative"].isna().all():
            continue
        agg = (
            gdf.dropna(subset=["year_relative"])
            .groupby("year_relative")["similarity"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["group"] = group
        agg.columns = ["year_relative", "mean_similarity", "std_similarity", "n_countries", "group"]
        records.append(agg)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def compute_lead_lag_test(trajectories: pd.DataFrame) -> dict:
    """3c: Wilcoxon signed-rank test on pre vs post slopes for ratifiers."""
    if trajectories.empty or "year_relative" not in trajectories.columns:
        return {}

    ratifiers = trajectories[trajectories["group"] == "ratifiers"].dropna(subset=["year_relative"])
    if ratifiers.empty:
        return {}

    pre_slopes = []
    post_slopes = []

    for country, cdf in ratifiers.groupby("country_iso3"):
        cdf = cdf.sort_values("year_relative")
        pre = cdf[cdf["year_relative"].between(-5, 0)]
        post = cdf[cdf["year_relative"].between(0, 5)]

        if len(pre) >= 2 and len(post) >= 2:
            pre_slope = compute_linear_slope(
                pd.Series(pre["similarity"].values, index=pre["year_relative"])
            )
            post_slope = compute_linear_slope(
                pd.Series(post["similarity"].values, index=post["year_relative"])
            )
            if not np.isnan(pre_slope) and not np.isnan(post_slope):
                pre_slopes.append(pre_slope)
                post_slopes.append(post_slope)

    if len(pre_slopes) < 5:
        return {"n_countries": len(pre_slopes), "error": "insufficient data"}

    stat, p_value = stats.wilcoxon(pre_slopes, post_slopes)
    pre_mean = np.mean(pre_slopes)
    post_mean = np.mean(post_slopes)

    direction = "rhetoric_leads" if pre_mean > post_mean else "rhetoric_follows"

    return {
        "n_countries": len(pre_slopes),
        "pre_slope_mean": float(pre_mean),
        "post_slope_mean": float(post_mean),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "direction": direction,
        "interpretation": (
            "Rhetoric shifts BEFORE ratification (countries adopt treaty language before signing)"
            if direction == "rhetoric_leads" else
            "Rhetoric shifts AFTER ratification (language changes follow ratification)"
        ),
    }


def compute_cross_treaty_comparison(all_results: dict) -> pd.DataFrame:
    """3d: Compare adoption curve shapes across treaties."""
    records = []
    for treaty, res in all_results.items():
        curves = res.get("curves", pd.DataFrame())
        if curves.empty:
            continue

        ratifier_curves = curves[curves["group"] == "ratifiers"]
        non_sig = curves[curves["group"] == "non_signatories"]

        if ratifier_curves.empty:
            continue

        # Baseline: mean non-signatory similarity
        baseline = non_sig["mean_similarity"].mean() if not non_sig.empty else 0.0

        # Rhetorical lag: first year (pre-ratification) where ratifiers exceed baseline
        pre = ratifier_curves[ratifier_curves["year_relative"] < 0].sort_values("year_relative", ascending=False)
        rhetorical_lag = None
        for _, row in pre.iterrows():
            if row["mean_similarity"] > baseline:
                rhetorical_lag = int(row["year_relative"])
                break

        # Peak similarity
        peak_sim = ratifier_curves["mean_similarity"].max()

        # Pre/post slopes
        pre_slope = compute_linear_slope(
            pd.Series(
                ratifier_curves[ratifier_curves["year_relative"].between(-5, 0)]["mean_similarity"].values,
                index=ratifier_curves[ratifier_curves["year_relative"].between(-5, 0)]["year_relative"].values,
            )
        ) if len(ratifier_curves[ratifier_curves["year_relative"].between(-5, 0)]) >= 2 else np.nan

        post_slope = compute_linear_slope(
            pd.Series(
                ratifier_curves[ratifier_curves["year_relative"].between(0, 5)]["mean_similarity"].values,
                index=ratifier_curves[ratifier_curves["year_relative"].between(0, 5)]["year_relative"].values,
            )
        ) if len(ratifier_curves[ratifier_curves["year_relative"].between(0, 5)]) >= 2 else np.nan

        records.append({
            "treaty": treaty,
            "n_ratifiers": ratifier_curves["n_countries"].max() if "n_countries" in ratifier_curves.columns else np.nan,
            "rhetorical_lag": rhetorical_lag,
            "peak_similarity": float(peak_sim),
            "pre_slope": float(pre_slope) if not np.isnan(pre_slope) else None,
            "post_slope": float(post_slope) if not np.isnan(post_slope) else None,
            "non_signatory_baseline": float(baseline),
        })

    return pd.DataFrame(records)


def compute_lexicon_crossvalidation(
    corpus: pd.DataFrame, treaty: str, window: int = 10
) -> pd.DataFrame:
    """3e: Track treaty-specific lexicon frequency in ±window year window."""
    if corpus.empty or treaty not in TREATY_LEXICONS:
        return pd.DataFrame()

    parties = TREATY_PARTY_DICTS.get(treaty, {})
    text_col = "segment_text" if "segment_text" in corpus.columns else "text"

    records = []
    for _, row in corpus.iterrows():
        country = row["country_iso3"]
        year = int(row["year"])
        text = str(row.get(text_col, "") or "")

        is_ratifier = country in parties
        event_year = parties.get(country)
        year_relative = (year - event_year) if event_year else None

        if event_year and abs(year - event_year) > window:
            continue

        lex_count = count_treaty_lexicon(text, treaty)
        records.append({
            "country_iso3": country,
            "year": year,
            "year_relative": year_relative,
            "lexicon_count": lex_count,
            "is_ratifier": is_ratifier,
        })

    return pd.DataFrame(records)


def identify_outliers(
    trajectories: pd.DataFrame,
    corpus: pd.DataFrame,
    treaty: str,
    threshold_low: float = 0.3,
    threshold_high: float = 0.7,
) -> pd.DataFrame:
    """3f: Find outlier countries — ratified with no language, or language without ratifying."""
    if trajectories.empty:
        return pd.DataFrame()

    parties = TREATY_PARTY_DICTS.get(treaty, {})
    records = []

    # Compute mean similarity at ratification year (year_relative ∈ [-1, 1])
    at_ratification = (
        trajectories[trajectories["year_relative"].between(-1, 1)]
        .groupby("country_iso3")["similarity"]
        .mean()
    )

    # Ratified but low similarity
    for country, sim in at_ratification.items():
        is_ratifier = country in parties
        if is_ratifier and sim < threshold_low:
            records.append({
                "country_iso3": country,
                "treaty": treaty,
                "outlier_type": "ratified_no_language",
                "similarity_at_ratification": float(sim),
                "ratification_year": parties.get(country),
            })

    # Non-ratifiers with high similarity throughout
    non_ratifiers = trajectories[trajectories["group"] == "non_signatories"]
    if not non_ratifiers.empty:
        mean_sim_non = non_ratifiers.groupby("country_iso3")["similarity"].mean()
        for country, sim in mean_sim_non.items():
            if sim > threshold_high:
                records.append({
                    "country_iso3": country,
                    "treaty": treaty,
                    "outlier_type": "language_no_ratification",
                    "similarity_at_ratification": float(sim),
                    "ratification_year": None,
                })

    return pd.DataFrame(records)


def compute_voting_adoption_curve(
    voting: pd.DataFrame, treaty: str, window: int = 10
) -> pd.DataFrame:
    """3g: Voting on treaty-specific resolutions in ±window year window."""
    if voting.empty:
        return pd.DataFrame()

    treaty_votes = voting[voting.get("treaty_flag", pd.Series()) == treaty] \
        if "treaty_flag" in voting.columns else pd.DataFrame()
    if treaty_votes.empty:
        return pd.DataFrame()

    parties = TREATY_PARTY_DICTS.get(treaty, {})
    records = []

    for _, row in treaty_votes.iterrows():
        country = row["country_iso3"]
        year = int(row["year"])
        vote = row.get("vote_numeric", np.nan)

        event_year = parties.get(country)
        year_relative = (year - event_year) if event_year else None

        if event_year and abs(year - event_year) > window:
            continue

        group = "ratifiers" if country in parties else "non_ratifiers"
        records.append({
            "country_iso3": country,
            "year": year,
            "year_relative": year_relative,
            "vote_numeric": vote,
            "group": group,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    agg = (
        df.dropna(subset=["year_relative"])
        .groupby(["year_relative", "group"])["vote_numeric"]
        .apply(lambda x: (x == 1).sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan)
        .reset_index()
        .rename(columns={"vote_numeric": "pct_yes"})
    )
    return agg
