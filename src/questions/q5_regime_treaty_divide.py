"""
Q5: How does regime type shape rhetorical engagement with arms control treaties?

Integrates Q2 (regime), Q3 (treaty), Q4 (voting/nuclear) into a capstone analysis.

Sub-analyses:
5a. Treaty-regime adoption curves (split Q3 curves by democracy/autocracy)
5b. Rhetoric-action gap (speech vs voting vs ratification, by regime × alliance)
5c. Arms trade integration (SIPRI TIV vs treaty rhetoric)
5d. Treaty rhetoric clustering (70-dim proximity vector → dendrogram + UMAP)
5e. Regime transition + treaty rhetoric (DID estimation)
5f. Temporal gap evolution (dem-aut distance per treaty over time)
5g. Fightin' Words by regime × treaty
5h. Panel regression dataset
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from src.shared.embeddings import cosine_sim, compute_centroid
from src.shared.distinctive_words import fightin_words
from src.shared.temporal import compute_change_points, compute_between_group_distance
from src.shared.lexicons import TREATY_LEXICONS
from src.data.groups import (
    ATT_PARTIES, TPNW_PARTIES, OTTAWA_PARTIES, CCM_PARTIES,
    TPNW_OPPONENTS, NWS, NATO, NAM,
    get_binary_regime, get_treaty_status,
)

OUTPUT_DIR = "output/q5"

# Treaties to analyse (must exist in anchor_embeddings)
TREATIES_Q5 = ["att", "tpnw", "ottawa", "ccm"]
TREATY_ANCHOR_KEYS = {
    "att": "att_2013",
    "tpnw": "tpnw_2017",
    "ottawa": "ottawa_1997",
    "ccm": "ccm_2008",
}
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

# ── SIPRI country name → ISO3 bridge ─────────────────────────────────────────
_SIPRI_NAME_TO_ISO3 = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahrain": "BHR", "Bangladesh": "BGD", "Belarus": "BLR",
    "Belgium": "BEL", "Bolivia": "BOL", "Bosnia-Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Brunei": "BRN", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Chad": "TCD", "Chile": "CHL", "China": "CHN", "Colombia": "COL",
    "Congo": "COG", "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB",
    "Cyprus": "CYP", "Czechia": "CZE", "DR Congo": "COD", "Denmark": "DNK",
    "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY",
    "El Salvador": "SLV", "Eritrea": "ERI", "Estonia": "EST", "Ethiopia": "ETH",
    "Finland": "FIN", "France": "FRA", "Georgia": "GEO", "Germany": "DEU",
    "Ghana": "GHA", "Greece": "GRC", "Guatemala": "GTM", "Guinea": "GIN",
    "Hungary": "HUN", "India": "IND", "Indonesia": "IDN", "Iran": "IRN",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Ivory Coast": "CIV", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kuwait": "KWT", "Kyrgyzstan": "KGZ", "Laos": "LAO",
    "Latvia": "LVA", "Lebanon": "LBN", "Libya": "LBY", "Lithuania": "LTU",
    "Malaysia": "MYS", "Mali": "MLI", "Mauritania": "MRT", "Mexico": "MEX",
    "Moldova": "MDA", "Mongolia": "MNG", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nepal": "NPL", "Netherlands": "NLD",
    "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA",
    "North Korea": "PRK", "North Macedonia": "MKD", "Norway": "NOR",
    "Oman": "OMN", "Pakistan": "PAK", "Panama": "PAN", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Qatar": "QAT", "Romania": "ROU", "Russia": "RUS", "Rwanda": "RWA",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK",
    "Slovenia": "SVN", "Somalia": "SOM", "South Africa": "ZAF",
    "South Korea": "KOR", "South Sudan": "SSD", "Spain": "ESP",
    "Sri Lanka": "LKA", "Sudan": "SDN", "Sweden": "SWE", "Switzerland": "CHE",
    "Syria": "SYR", "Taiwan": "TWN", "Tajikistan": "TJK", "Tanzania": "TZA",
    "Thailand": "THA", "Togo": "TGO", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkiye": "TUR", "Turkey": "TUR",
    "Turkmenistan": "TKM", "UAE": "ARE", "Uganda": "UGA", "Ukraine": "UKR",
    "United Kingdom": "GBR", "United States": "USA", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Venezuela": "VEN", "Viet Nam": "VNM",
    "Vietnam": "VNM", "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    "Soviet Union": "RUS", "Czechoslovakia": "CZE", "Yugoslavia": "SRB",
    "Serbia and Montenegro": "SRB",
}

# Regime transitions (reused from Q2, extended)
REGIME_TRANSITIONS = {
    "TUR": {"direction": "backsliding", "year": 2018, "label": "Turkey autocratization"},
    "HUN": {"direction": "backsliding", "year": 2010, "label": "Hungary backsliding"},
    "IND": {"direction": "backsliding", "year": 2019, "label": "India democratic erosion"},
    "RUS": {"direction": "backsliding", "year": 2012, "label": "Russia autocratization"},
    "TUN": {"direction": "democratization", "year": 2011, "label": "Tunisia democratization"},
    "MMR": {"direction": "backsliding", "year": 2021, "label": "Myanmar coup"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _annotate_regime(df: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """Add binary_regime column using V-Dem."""
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
    else:
        result["binary_regime"] = "unknown"
    return result


def _get_anchor_embedding(anchor_embeddings: dict, treaty: str) -> np.ndarray:
    """Get mean anchor embedding for a treaty."""
    key = TREATY_ANCHOR_KEYS.get(treaty, treaty)
    if key not in anchor_embeddings:
        return None
    passages = anchor_embeddings[key]
    if isinstance(passages, dict):
        vecs = [v for v in passages.values() if isinstance(v, np.ndarray)]
    elif isinstance(passages, np.ndarray):
        return passages
    else:
        return None
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _compute_treaty_similarity(embedding: np.ndarray, anchor_embeddings: dict) -> dict:
    """Compute similarity of one embedding to all treaty anchors."""
    sims = {}
    for treaty_key, passages in anchor_embeddings.items():
        if isinstance(passages, dict):
            vecs = [v for v in passages.values() if isinstance(v, np.ndarray)]
            if not vecs:
                continue
            anchor = np.mean(vecs, axis=0)
        elif isinstance(passages, np.ndarray):
            anchor = passages
        else:
            continue
        sims[treaty_key] = cosine_sim(embedding, anchor)
    return sims


def _load_sipri_country_year(sipri: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SIPRI transfers to country-year export/import totals."""
    if sipri.empty:
        return pd.DataFrame()

    records = []

    # Exports
    sipri_mapped = sipri.copy()
    sipri_mapped["supplier_iso3"] = sipri_mapped["supplier"].map(_SIPRI_NAME_TO_ISO3)
    sipri_mapped["recipient_iso3"] = sipri_mapped["recipient"].map(_SIPRI_NAME_TO_ISO3)

    exports = (
        sipri_mapped.dropna(subset=["supplier_iso3"])
        .groupby(["supplier_iso3", "year"])["tiv_delivery"]
        .sum()
        .reset_index()
        .rename(columns={"supplier_iso3": "country_iso3", "tiv_delivery": "export_tiv"})
    )
    imports = (
        sipri_mapped.dropna(subset=["recipient_iso3"])
        .groupby(["recipient_iso3", "year"])["tiv_delivery"]
        .sum()
        .reset_index()
        .rename(columns={"recipient_iso3": "country_iso3", "tiv_delivery": "import_tiv"})
    )

    merged = pd.merge(exports, imports, on=["country_iso3", "year"], how="outer").fillna(0)
    return merged


def _save(df: pd.DataFrame, filename: str):
    if df is not None and not df.empty:
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        print(f"[Q5] Saved {path}")


# ── Sub-analyses ──────────────────────────────────────────────────────────────

def compute_treaty_regime_adoption_curves(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    vdem: pd.DataFrame,
    treaties: list = None,
    window: int = 10,
) -> pd.DataFrame:
    """5a: For each treaty, split adoption curves by democracy/autocracy."""
    if treaties is None:
        treaties = TREATIES_Q5
    if country_year_embeddings.empty:
        return pd.DataFrame()

    cy = _annotate_regime(country_year_embeddings, vdem)
    records = []

    for treaty in treaties:
        anchor_vec = _get_anchor_embedding(anchor_embeddings, treaty)
        if anchor_vec is None:
            continue

        parties = TREATY_PARTY_DICTS.get(treaty, {})
        adoption_year = TREATY_ADOPTION_YEAR.get(treaty)

        for _, row in cy.iterrows():
            country = row["country_iso3"]
            year = int(row["year"])
            emb = row.get("embedding")
            regime = row.get("binary_regime", "unknown")

            if emb is None or regime == "unknown":
                continue
            if not isinstance(emb, np.ndarray):
                try:
                    emb = np.array(emb, dtype=float)
                except (TypeError, ValueError):
                    continue
            if emb.ndim == 0 or len(emb) == 0:
                continue

            sim = cosine_sim(emb, anchor_vec)

            # Determine group and event year
            if country in parties:
                group = "ratifiers"
                event_year = parties[country]
            elif country in TPNW_OPPONENTS and treaty == "tpnw":
                group = "opponents"
                event_year = adoption_year
            else:
                group = "non_signatories"
                event_year = adoption_year

            if event_year is None:
                continue

            year_relative = year - event_year
            if abs(year_relative) > window:
                continue

            records.append({
                "treaty": treaty,
                "country_iso3": country,
                "year": year,
                "year_relative": year_relative,
                "regime": regime,
                "group": group,
                "similarity": sim,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Aggregate: mean similarity by (treaty, year_relative, regime, group)
    agg = (
        df.groupby(["treaty", "year_relative", "regime", "group"])
        .agg(mean_similarity=("similarity", "mean"),
             std_similarity=("similarity", "std"),
             n=("similarity", "count"))
        .reset_index()
    )
    return agg


def compute_rhetoric_action_gap(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    voting: pd.DataFrame,
    vdem: pd.DataFrame,
    frame_scores: pd.DataFrame,
) -> pd.DataFrame:
    """5b: Rhetoric vs voting vs ratification gap, stratified by regime × alliance."""
    if country_year_embeddings.empty:
        return pd.DataFrame()

    cy = _annotate_regime(country_year_embeddings, vdem)

    # Compute mean treaty similarity per country-year
    treaty_sims = []
    for _, row in cy.iterrows():
        emb = row.get("embedding")
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            try:
                emb = np.array(emb, dtype=float)
            except (TypeError, ValueError):
                continue
        if emb.ndim == 0 or len(emb) == 0:
            continue

        sims = {}
        for treaty in TREATIES_Q5:
            anchor_vec = _get_anchor_embedding(anchor_embeddings, treaty)
            if anchor_vec is not None:
                sims[f"{treaty}_similarity"] = cosine_sim(emb, anchor_vec)

        sims["country_iso3"] = row["country_iso3"]
        sims["year"] = int(row["year"])
        sims["regime"] = row.get("binary_regime", "unknown")
        treaty_sims.append(sims)

    if not treaty_sims:
        return pd.DataFrame()

    sim_df = pd.DataFrame(treaty_sims)

    # Mean treaty similarity across all treaties as rhetoric_score
    sim_cols = [c for c in sim_df.columns if c.endswith("_similarity")]
    sim_df["rhetoric_score"] = sim_df[sim_cols].mean(axis=1)

    # Voting: compute per country-year % yes on disarmament
    if not voting.empty and "vote_numeric" in voting.columns:
        disarm = voting[voting.get("issue", pd.Series()) == "Arms control and disarmament"] \
            if "issue" in voting.columns else voting

        vote_agg = (
            disarm.groupby(["country_iso3", "year"])["vote_numeric"]
            .apply(lambda x: (x == 1).sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan)
            .reset_index()
            .rename(columns={"vote_numeric": "vote_score"})
        )
        sim_df = sim_df.merge(vote_agg, on=["country_iso3", "year"], how="left")
    else:
        sim_df["vote_score"] = np.nan

    # Frame ratio from frame_scores
    if not frame_scores.empty:
        fr_col = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"
        if fr_col in frame_scores.columns:
            fr = frame_scores[["country_iso3", "year", fr_col]].rename(
                columns={fr_col: "frame_ratio"}
            )
            sim_df = sim_df.merge(fr, on=["country_iso3", "year"], how="left")

    # Alliance annotation
    sim_df["alliance"] = sim_df["country_iso3"].apply(
        lambda c: "NAM" if c in NAM else ("NATO" if c in NATO else
                  ("P5" if c in NWS else "other"))
    )

    # Ratification status per treaty
    for treaty in TREATIES_Q5:
        parties = TREATY_PARTY_DICTS.get(treaty, {})
        sim_df[f"{treaty}_ratified"] = sim_df["country_iso3"].isin(parties).astype(int)

    # Gap: |rhetoric_score - vote_score| (normalized to [0,1] range)
    sim_df["gap_rhetoric_vote"] = (sim_df["rhetoric_score"] - sim_df["vote_score"]).abs()

    return sim_df


def compute_rhetoric_action_gap_by_group(gap_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rhetoric-action gap by regime × alliance."""
    if gap_df.empty:
        return pd.DataFrame()

    agg = (
        gap_df.groupby(["regime", "alliance"])
        .agg(
            rhetoric_score_mean=("rhetoric_score", "mean"),
            vote_score_mean=("vote_score", "mean"),
            gap_mean=("gap_rhetoric_vote", "mean"),
            gap_std=("gap_rhetoric_vote", "std"),
            n=("rhetoric_score", "count"),
        )
        .reset_index()
    )
    return agg


def compute_arms_trade_rhetoric(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    sipri: pd.DataFrame,
    vdem: pd.DataFrame,
) -> pd.DataFrame:
    """5c: Merge SIPRI TIV with treaty similarity and regime."""
    sipri_cy = _load_sipri_country_year(sipri)
    if sipri_cy.empty or country_year_embeddings.empty:
        return pd.DataFrame()

    cy = _annotate_regime(country_year_embeddings, vdem)

    # Compute treaty similarity vector per country-year
    rows = []
    for _, row in cy.iterrows():
        emb = row.get("embedding")
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            try:
                emb = np.array(emb, dtype=float)
            except (TypeError, ValueError):
                continue
        if emb.ndim == 0 or len(emb) == 0:
            continue

        entry = {
            "country_iso3": row["country_iso3"],
            "year": int(row["year"]),
            "regime": row.get("binary_regime", "unknown"),
        }
        for treaty in TREATIES_Q5:
            anchor_vec = _get_anchor_embedding(anchor_embeddings, treaty)
            if anchor_vec is not None:
                entry[f"{treaty}_similarity"] = cosine_sim(emb, anchor_vec)
        rows.append(entry)

    if not rows:
        return pd.DataFrame()

    sim_df = pd.DataFrame(rows)
    merged = sim_df.merge(sipri_cy, on=["country_iso3", "year"], how="left").fillna(
        {"export_tiv": 0, "import_tiv": 0}
    )
    merged["log_export_tiv"] = np.log1p(merged["export_tiv"])
    merged["log_import_tiv"] = np.log1p(merged["import_tiv"])

    # Classify trade role
    total_exports = merged.groupby("country_iso3")["export_tiv"].sum()
    top_exporters = set(total_exports.nlargest(10).index)
    total_imports = merged.groupby("country_iso3")["import_tiv"].sum()
    top_importers = set(total_imports.nlargest(20).index) - top_exporters

    merged["trade_role"] = merged["country_iso3"].apply(
        lambda c: "major_exporter" if c in top_exporters else
        ("major_importer" if c in top_importers else "other")
    )
    return merged


def compute_exporter_vs_importer(trade_df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean treaty similarity by trade role × regime."""
    if trade_df.empty:
        return pd.DataFrame()

    sim_cols = [c for c in trade_df.columns if c.endswith("_similarity")]
    if not sim_cols:
        return pd.DataFrame()

    agg = (
        trade_df.groupby(["trade_role", "regime"])[sim_cols + ["export_tiv", "import_tiv"]]
        .mean()
        .reset_index()
    )
    return agg


def compute_treaty_rhetoric_clusters(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    vdem: pd.DataFrame,
    year_range: tuple = (2015, 2023),
) -> pd.DataFrame:
    """5d: Build treaty proximity vectors and cluster countries."""
    if country_year_embeddings.empty:
        return pd.DataFrame()

    # Filter to recent years for stable clustering
    cy = country_year_embeddings[
        (country_year_embeddings["year"] >= year_range[0]) &
        (country_year_embeddings["year"] <= year_range[1])
    ].copy()

    if cy.empty:
        return pd.DataFrame()

    # Compute per-country mean embedding
    country_embs = {}
    for country, grp in cy.groupby("country_iso3"):
        embs = []
        for _, row in grp.iterrows():
            emb = row.get("embedding")
            if emb is None:
                continue
            if not isinstance(emb, np.ndarray):
                try:
                    emb = np.array(emb, dtype=float)
                except (TypeError, ValueError):
                    continue
            if emb.ndim == 0 or len(emb) == 0:
                continue
            embs.append(emb)
        if embs:
            country_embs[country] = np.mean(embs, axis=0)

    if len(country_embs) < 10:
        return pd.DataFrame()

    # Build treaty proximity vectors
    treaty_keys = sorted(anchor_embeddings.keys())
    anchor_vecs = {}
    for key in treaty_keys:
        passages = anchor_embeddings[key]
        if isinstance(passages, dict):
            vecs = [v for v in passages.values() if isinstance(v, np.ndarray)]
            if vecs:
                anchor_vecs[key] = np.mean(vecs, axis=0)
        elif isinstance(passages, np.ndarray):
            anchor_vecs[key] = passages

    if not anchor_vecs:
        return pd.DataFrame()

    sorted_keys = sorted(anchor_vecs.keys())
    countries = sorted(country_embs.keys())

    # Build proximity matrix: countries × treaties
    proximity_matrix = np.zeros((len(countries), len(sorted_keys)))
    for i, country in enumerate(countries):
        emb = country_embs[country]
        for j, key in enumerate(sorted_keys):
            proximity_matrix[i, j] = cosine_sim(emb, anchor_vecs[key])

    # Agglomerative clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(proximity_matrix, method="ward")
    n_clusters = min(6, len(countries) // 5)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Get modal regime per country
    cy_regime = _annotate_regime(cy, vdem)
    regime_modes = cy_regime.groupby("country_iso3")["binary_regime"].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown"
    ).to_dict()

    records = []
    for i, country in enumerate(countries):
        records.append({
            "country_iso3": country,
            "cluster": int(labels[i]),
            "regime_mode": regime_modes.get(country, "unknown"),
            "nato": int(country in NATO),
            "nam": int(country in NAM),
            "p5": int(country in NWS),
        })

    result = pd.DataFrame(records)

    # Adjusted Rand Index: cluster vs regime
    from sklearn.metrics import adjusted_rand_score
    regime_labels = result["regime_mode"].map({"democracy": 0, "autocracy": 1}).fillna(-1)
    ari = adjusted_rand_score(regime_labels, result["cluster"])
    print(f"[Q5] Cluster-Regime ARI: {ari:.3f}")

    # Store linkage and proximity matrix for plotting
    result.attrs["linkage"] = Z
    result.attrs["proximity_matrix"] = proximity_matrix
    result.attrs["countries"] = countries
    result.attrs["treaty_keys"] = sorted_keys

    return result


def compute_transition_treaty_analysis(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    vdem: pd.DataFrame,
    voting: pd.DataFrame,
) -> pd.DataFrame:
    """5e: Regime transitions → treaty rhetoric shifts."""
    if country_year_embeddings.empty:
        return pd.DataFrame()

    records = []
    for country, info in REGIME_TRANSITIONS.items():
        ty = info["year"]
        cdf = country_year_embeddings[
            country_year_embeddings["country_iso3"] == country
        ].sort_values("year")

        if cdf.empty:
            continue

        for treaty in TREATIES_Q5:
            anchor_vec = _get_anchor_embedding(anchor_embeddings, treaty)
            if anchor_vec is None:
                continue

            for _, row in cdf.iterrows():
                emb = row.get("embedding")
                if emb is None:
                    continue
                if not isinstance(emb, np.ndarray):
                    try:
                        emb = np.array(emb, dtype=float)
                    except (TypeError, ValueError):
                        continue
                if emb.ndim == 0 or len(emb) == 0:
                    continue

                year = int(row["year"])
                sim = cosine_sim(emb, anchor_vec)
                records.append({
                    "country_iso3": country,
                    "transition_year": ty,
                    "direction": info["direction"],
                    "label": info["label"],
                    "treaty": treaty,
                    "year": year,
                    "year_relative": year - ty,
                    "similarity": sim,
                })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Pre/post summary
    pre = df[df["year_relative"] < 0].groupby(["country_iso3", "treaty"])["similarity"].mean()
    post = df[df["year_relative"] >= 0].groupby(["country_iso3", "treaty"])["similarity"].mean()
    summary = pd.DataFrame({"pre_similarity": pre, "post_similarity": post}).reset_index()
    summary["shift"] = summary["post_similarity"] - summary["pre_similarity"]
    _save(summary, "transition_treaty_summary.csv")

    # DID: simple before/after comparison
    # Treatment effect = mean shift for transitioning countries
    did_records = []
    for treaty in TREATIES_Q5:
        t_df = summary[summary["treaty"] == treaty]
        if t_df.empty:
            continue
        mean_shift = t_df["shift"].mean()
        t_stat, p_val = stats.ttest_1samp(t_df["shift"].dropna(), 0) if len(t_df["shift"].dropna()) >= 2 else (np.nan, np.nan)
        did_records.append({
            "treaty": treaty,
            "mean_shift": mean_shift,
            "t_statistic": t_stat,
            "p_value": p_val,
            "n_countries": len(t_df),
        })
    _save(pd.DataFrame(did_records), "did_estimates.csv")

    return df


def compute_regime_treaty_gap_evolution(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    vdem: pd.DataFrame,
    treaties: list = None,
) -> pd.DataFrame:
    """5f: Per-year dem-aut distance conditioned on treaty rhetoric."""
    if treaties is None:
        treaties = TREATIES_Q5
    if country_year_embeddings.empty:
        return pd.DataFrame()

    cy = _annotate_regime(country_year_embeddings, vdem)
    records = []

    for treaty in treaties:
        anchor_vec = _get_anchor_embedding(anchor_embeddings, treaty)
        if anchor_vec is None:
            continue

        for year, ydf in cy.groupby("year"):
            dem_embs = []
            aut_embs = []
            for _, row in ydf.iterrows():
                emb = row.get("embedding")
                regime = row.get("binary_regime", "unknown")
                if emb is None or regime == "unknown":
                    continue
                if not isinstance(emb, np.ndarray):
                    try:
                        emb = np.array(emb, dtype=float)
                    except (TypeError, ValueError):
                        continue
                if emb.ndim == 0 or len(emb) == 0:
                    continue

                if regime == "democracy":
                    dem_embs.append(emb)
                else:
                    aut_embs.append(emb)

            if len(dem_embs) < 2 or len(aut_embs) < 2:
                continue

            dem_centroid = np.mean(dem_embs, axis=0)
            aut_centroid = np.mean(aut_embs, axis=0)

            # Distance in the direction of this treaty's anchor
            dem_sim = cosine_sim(dem_centroid, anchor_vec)
            aut_sim = cosine_sim(aut_centroid, anchor_vec)

            records.append({
                "year": year,
                "treaty": treaty,
                "dem_similarity": dem_sim,
                "aut_similarity": aut_sim,
                "gap": dem_sim - aut_sim,
                "abs_gap": abs(dem_sim - aut_sim),
                "n_dem": len(dem_embs),
                "n_aut": len(aut_embs),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values(["treaty", "year"])

    # Change-point detection per treaty
    for treaty in treaties:
        tdf = df[df["treaty"] == treaty]
        if len(tdf) > 10:
            series = pd.Series(tdf["abs_gap"].values, index=tdf["year"].values)
            breaks = compute_change_points(series)
            break_years = {yr for yr, _ in breaks}
            df.loc[df["treaty"] == treaty, "is_change_point"] = \
                df.loc[df["treaty"] == treaty, "year"].isin(break_years)

    return df


def compute_distinctive_words_regime_treaty(
    corpus: pd.DataFrame,
    vdem: pd.DataFrame,
    treaties: list = None,
) -> pd.DataFrame:
    """5g: Fightin' Words by regime within treaty-mentioning speeches."""
    if treaties is None:
        treaties = ["att", "tpnw", "ottawa", "npt"]
    if corpus.empty:
        return pd.DataFrame()

    corpus = _annotate_regime(corpus, vdem)
    text_col = "segment_text" if "segment_text" in corpus.columns else "text"

    all_results = []
    for treaty in treaties:
        # Get lexicon keywords for this treaty
        lexicon = TREATY_LEXICONS.get(treaty, [])
        if not lexicon:
            # Fallback: use treaty name
            lexicon = [treaty.upper(), treaty.lower()]

        # Filter to speeches mentioning treaty keywords
        pattern = "|".join(lexicon[:10])  # use top 10 keywords
        mask = corpus[text_col].str.contains(pattern, case=False, na=False)
        filtered = corpus[mask]

        if filtered.empty or len(filtered) < 20:
            continue

        dem_texts = filtered[filtered["binary_regime"] == "democracy"][text_col].fillna("").tolist()
        aut_texts = filtered[filtered["binary_regime"] == "autocracy"][text_col].fillna("").tolist()

        if len(dem_texts) < 5 or len(aut_texts) < 5:
            continue

        fw = fightin_words(dem_texts, aut_texts, top_n=20)
        fw["treaty"] = treaty
        fw["group_a_label"] = "democracy"
        fw["group_b_label"] = "autocracy"
        all_results.append(fw)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def build_panel_regression_dataset(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    frame_scores: pd.DataFrame,
    voting: pd.DataFrame,
    vdem: pd.DataFrame,
    sipri: pd.DataFrame,
) -> pd.DataFrame:
    """5h: Wide panel for R/Stata regression."""
    if country_year_embeddings.empty:
        return pd.DataFrame()

    cy = _annotate_regime(country_year_embeddings, vdem)

    # Treaty similarities
    rows = []
    for _, row in cy.iterrows():
        emb = row.get("embedding")
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            try:
                emb = np.array(emb, dtype=float)
            except (TypeError, ValueError):
                continue
        if emb.ndim == 0 or len(emb) == 0:
            continue

        entry = {
            "country_iso3": row["country_iso3"],
            "year": int(row["year"]),
            "binary_regime": row.get("binary_regime", "unknown"),
        }
        for treaty in TREATIES_Q5:
            anchor_vec = _get_anchor_embedding(anchor_embeddings, treaty)
            if anchor_vec is not None:
                entry[f"{treaty}_similarity"] = cosine_sim(emb, anchor_vec)
        rows.append(entry)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows)

    # Frame scores
    if not frame_scores.empty:
        fr_col = "frame_ratio_mean" if "frame_ratio_mean" in frame_scores.columns else "frame_ratio"
        if fr_col in frame_scores.columns:
            fr = frame_scores[["country_iso3", "year", fr_col]].rename(
                columns={fr_col: "frame_ratio"}
            )
            panel = panel.merge(fr, on=["country_iso3", "year"], how="left")

    # Voting
    if not voting.empty and "vote_numeric" in voting.columns:
        disarm = voting[voting.get("issue", pd.Series()) == "Arms control and disarmament"] \
            if "issue" in voting.columns else voting
        vote_agg = (
            disarm.groupby(["country_iso3", "year"])["vote_numeric"]
            .apply(lambda x: (x == 1).sum() / x.notna().sum() if x.notna().sum() > 0 else np.nan)
            .reset_index()
            .rename(columns={"vote_numeric": "voting_pct_yes"})
        )
        panel = panel.merge(vote_agg, on=["country_iso3", "year"], how="left")

    # SIPRI
    sipri_cy = _load_sipri_country_year(sipri)
    if not sipri_cy.empty:
        panel = panel.merge(sipri_cy, on=["country_iso3", "year"], how="left")
        panel["export_tiv"] = panel.get("export_tiv", 0).fillna(0)
        panel["import_tiv"] = panel.get("import_tiv", 0).fillna(0)
        panel["log_export_tiv"] = np.log1p(panel["export_tiv"])

    # Ratification dummies
    for treaty in TREATIES_Q5:
        parties = TREATY_PARTY_DICTS.get(treaty, {})
        panel[f"ratified_{treaty}"] = panel["country_iso3"].isin(parties).astype(int)

    # Alliance / group dummies
    panel["nato"] = panel["country_iso3"].isin(NATO).astype(int)
    panel["nam"] = panel["country_iso3"].isin(NAM).astype(int)
    panel["p5"] = panel["country_iso3"].isin(NWS).astype(int)

    return panel


# ── Main entry point ──────────────────────────────────────────────────────────

def run_q5(data: dict, config: dict = None) -> dict:
    """Main Q5 entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    corpus = data.get("corpus", pd.DataFrame())
    frame_scores = data.get("frame_scores", pd.DataFrame())
    voting = data.get("voting", pd.DataFrame())
    vdem = data.get("vdem", pd.DataFrame())
    country_year_embeddings = data.get("country_year_embeddings", pd.DataFrame())
    anchor_embeddings = data.get("anchor_embeddings", {})
    sipri = data.get("sipri", pd.DataFrame())

    results = {}

    # 5a. Treaty-regime adoption curves
    print("[Q5] 5a: Treaty-regime adoption curves...")
    adoption = compute_treaty_regime_adoption_curves(
        country_year_embeddings, anchor_embeddings, vdem
    )
    results["treaty_regime_adoption_curves"] = adoption
    _save(adoption, "treaty_regime_adoption_curves.csv")

    # 5b. Rhetoric-action gap
    print("[Q5] 5b: Rhetoric-action gap...")
    gap = compute_rhetoric_action_gap(
        country_year_embeddings, anchor_embeddings, voting, vdem, frame_scores
    )
    results["rhetoric_action_gap"] = gap
    _save(gap, "rhetoric_action_gap.csv")

    if not gap.empty:
        gap_by_group = compute_rhetoric_action_gap_by_group(gap)
        results["rhetoric_action_gap_by_group"] = gap_by_group
        _save(gap_by_group, "rhetoric_action_gap_by_group.csv")

    # 5c. Arms trade rhetoric
    if not sipri.empty:
        print("[Q5] 5c: Arms trade integration...")
        trade = compute_arms_trade_rhetoric(
            country_year_embeddings, anchor_embeddings, sipri, vdem
        )
        results["arms_trade_rhetoric"] = trade
        _save(trade, "arms_trade_rhetoric.csv")

        if not trade.empty:
            exp_imp = compute_exporter_vs_importer(trade)
            results["exporter_vs_importer"] = exp_imp
            _save(exp_imp, "exporter_vs_importer_rhetoric.csv")
    else:
        print("[Q5] 5c: SIPRI data not available — skipping arms trade analysis.")

    # 5d. Treaty rhetoric clustering
    print("[Q5] 5d: Treaty rhetoric clustering...")
    clusters = compute_treaty_rhetoric_clusters(
        country_year_embeddings, anchor_embeddings, vdem
    )
    results["treaty_rhetoric_clusters"] = clusters
    _save(clusters, "treaty_rhetoric_clusters.csv")

    # 5e. Regime transition + treaty rhetoric
    print("[Q5] 5e: Regime transition treaty analysis...")
    transitions = compute_transition_treaty_analysis(
        country_year_embeddings, anchor_embeddings, vdem, voting
    )
    results["transition_treaty_analysis"] = transitions
    _save(transitions, "transition_treaty_analysis.csv")

    # 5f. Temporal gap evolution
    print("[Q5] 5f: Regime-treaty gap evolution...")
    gap_evo = compute_regime_treaty_gap_evolution(
        country_year_embeddings, anchor_embeddings, vdem
    )
    results["regime_treaty_gap_evolution"] = gap_evo
    _save(gap_evo, "regime_treaty_gap_evolution.csv")

    # 5g. Fightin' Words by regime × treaty
    print("[Q5] 5g: Fightin' Words by regime × treaty...")
    fw = compute_distinctive_words_regime_treaty(corpus, vdem)
    results["distinctive_words_regime_treaty"] = fw
    _save(fw, "distinctive_words_regime_treaty.csv")

    # 5h. Panel regression dataset
    print("[Q5] 5h: Building panel regression dataset...")
    panel = build_panel_regression_dataset(
        country_year_embeddings, anchor_embeddings, frame_scores, voting, vdem, sipri
    )
    results["panel_dataset"] = panel
    _save(panel, "panel_dataset.csv")

    print("[Q5] Complete.")
    return results
