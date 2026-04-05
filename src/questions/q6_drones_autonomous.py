"""
Q6: Drones and Autonomous Weapons -- Rhetoric-Action Gap Analysis

Analyses:
6a. Drone keyword extraction from UNGDC speeches
6b. Drone rhetoric trajectory (embedding similarity to LAWS/drone anchors)
6c. Drone transfer data (SIPRI trade register parsing)
6d. Rhetoric-transfer gap (cross-reference 6a+6b with 6c)
6e. UNGA votes on LAWS/drone resolutions
6f. ATT ratifier behavior on drones (overlay SIPRI drone transfers on ATT status)
6g. Panel regression dataset
6h. Rhetoric-transfer OLS regressions
"""
import os
import numpy as np
import pandas as pd

from src.data.groups import ATT_PARTIES, get_treaty_status, get_binary_regime
from src.shared.lexicons import DRONE_AUTONOMOUS, count_matches
from src.shared.embeddings import cosine_sim

OUTPUT_DIR = "output/q6"

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

DRONE_WEAPON_PATTERNS = ["drone", "UAV", "unmanned", "UCAV", "remotely piloted", "loitering munition"]

TREATY_ANCHOR_KEYS_Q6 = {
    "laws_drones": "laws_drones_2023",
    "att_drones": "att_drones_2013",
}


# ---------------------------------------------------------------------------
# 6a. Drone Keyword Extraction
# ---------------------------------------------------------------------------

def compute_drone_keywords(corpus: pd.DataFrame) -> pd.DataFrame:
    """Count drone/LAWS terms in each speech."""
    records = []
    text_col = "segment_text" if "segment_text" in corpus.columns else "text"
    for _, row in corpus.iterrows():
        text = str(row.get(text_col, ""))
        n = count_matches(text, DRONE_AUTONOMOUS)
        if n == 0:
            continue
        words = len(text.split())
        matched = [t for t in DRONE_AUTONOMOUS if t.lower() in text.lower()]
        records.append({
            "country_iso3": row.get("country_iso3", ""),
            "year": int(row.get("year", 0)),
            "drone_mention_count": n,
            "drone_terms_found": "; ".join(matched),
            "segment_word_count": words,
            "drone_density": n / max(words, 1),
        })
    if not records:
        return pd.DataFrame(columns=["country_iso3", "year", "drone_mention_count",
                                      "drone_terms_found", "segment_word_count", "drone_density"])
    df = pd.DataFrame(records)
    agg = df.groupby(["country_iso3", "year"]).agg(
        drone_mentions=("drone_mention_count", "sum"),
        drone_density=("drone_density", "mean"),
        drone_terms=("drone_terms_found", lambda x: "; ".join(set("; ".join(x).split("; ")))),
        n_speeches=("drone_mention_count", "count"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 6b. Drone Rhetoric Trajectory
# ---------------------------------------------------------------------------

def compute_drone_rhetoric_trajectory(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
) -> pd.DataFrame:
    """Cosine similarity of each country-year to LAWS/drone anchors."""
    records = []
    for anchor_label, anchor_key in TREATY_ANCHOR_KEYS_Q6.items():
        if anchor_key not in anchor_embeddings:
            continue
        anchor_data = anchor_embeddings[anchor_key]
        if isinstance(anchor_data, dict):
            vecs = [v for v in anchor_data.values() if isinstance(v, np.ndarray)]
            if not vecs:
                continue
            anchor_vec = np.mean(vecs, axis=0)
        elif isinstance(anchor_data, np.ndarray):
            anchor_vec = anchor_data
        else:
            continue

        for _, row in country_year_embeddings.iterrows():
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
            sim = cosine_sim(emb, anchor_vec)
            records.append({
                "country_iso3": row["country_iso3"],
                "year": int(row["year"]),
                "anchor": anchor_label,
                "similarity": float(sim),
            })

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.pivot_table(index=["country_iso3", "year"],
                          columns="anchor", values="similarity").reset_index()


# ---------------------------------------------------------------------------
# 6c. Drone Transfer Data (SIPRI Trade Register)
# ---------------------------------------------------------------------------

def parse_drone_transfers(trade_register: pd.DataFrame) -> pd.DataFrame:
    """Filter trade register to drone deals, map to ISO3, aggregate."""
    if trade_register.empty:
        return pd.DataFrame()

    desc_col = next((c for c in trade_register.columns if "description" in c.lower()), None)
    if desc_col is None:
        return pd.DataFrame()

    mask = trade_register[desc_col].str.lower().fillna("").apply(
        lambda x: any(p.lower() in x for p in DRONE_WEAPON_PATTERNS)
    )
    drones = trade_register[mask].copy()
    if drones.empty:
        return pd.DataFrame()

    # Map columns
    supplier_col = next((c for c in drones.columns if "supplier" in c.lower()), None)
    recipient_col = next((c for c in drones.columns if "recipient" in c.lower()), None)
    year_col = next((c for c in drones.columns if "year of order" in c.lower()), None)
    tiv_col = next((c for c in drones.columns if "tiv of delivered" in c.lower()), None)
    ndelivered_col = next((c for c in drones.columns if "number delivered" in c.lower()), None)
    desig_col = next((c for c in drones.columns if "designation" in c.lower()), None)

    if not supplier_col or not recipient_col:
        return pd.DataFrame()

    drones["supplier_iso3"] = drones[supplier_col].map(_SIPRI_NAME_TO_ISO3)
    drones["recipient_iso3"] = drones[recipient_col].map(_SIPRI_NAME_TO_ISO3)
    drones["year"] = pd.to_numeric(drones[year_col], errors="coerce") if year_col else np.nan
    drones["tiv"] = pd.to_numeric(
        drones[tiv_col].astype(str).str.replace("?", "", regex=False), errors="coerce"
    ) if tiv_col else 0.0
    drones["n_delivered"] = pd.to_numeric(
        drones[ndelivered_col].astype(str).str.replace("?", "", regex=False), errors="coerce"
    ) if ndelivered_col else 0.0
    drones["weapon_type"] = drones[desc_col].str.strip()
    drones["weapon_name"] = drones[desig_col].str.strip() if desig_col else ""

    drones = drones.dropna(subset=["year"])
    drones["year"] = drones["year"].astype(int)

    # Build export and import records
    records = []
    for _, row in drones.iterrows():
        if pd.notna(row.get("supplier_iso3")):
            records.append({
                "country_iso3": row["supplier_iso3"],
                "year": row["year"],
                "role": "exporter",
                "tiv": row["tiv"],
                "n_units": row["n_delivered"],
                "weapon_type": row["weapon_type"],
                "weapon_name": row["weapon_name"],
                "partner": row.get("recipient_iso3", ""),
            })
        if pd.notna(row.get("recipient_iso3")):
            records.append({
                "country_iso3": row["recipient_iso3"],
                "year": row["year"],
                "role": "importer",
                "tiv": row["tiv"],
                "n_units": row["n_delivered"],
                "weapon_type": row["weapon_type"],
                "weapon_name": row["weapon_name"],
                "partner": row.get("supplier_iso3", ""),
            })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# 6d. Rhetoric-Transfer Gap
# ---------------------------------------------------------------------------

def compute_rhetoric_transfer_gap(
    drone_keywords: pd.DataFrame,
    drone_trajectory: pd.DataFrame,
    drone_transfers: pd.DataFrame,
    vdem: pd.DataFrame,
) -> pd.DataFrame:
    """Merge rhetoric (keywords + trajectory) with transfers, compute gap."""
    if drone_transfers.empty:
        return pd.DataFrame()

    # Aggregate transfers to country-year export TIV
    exports = drone_transfers[drone_transfers["role"] == "exporter"].groupby(
        ["country_iso3", "year"]
    ).agg(drone_export_tiv=("tiv", "sum"), n_export_deals=("tiv", "count")).reset_index()

    imports = drone_transfers[drone_transfers["role"] == "importer"].groupby(
        ["country_iso3", "year"]
    ).agg(drone_import_tiv=("tiv", "sum"), n_import_deals=("tiv", "count")).reset_index()

    # Merge with keywords
    merged = exports.merge(drone_keywords[["country_iso3", "year", "drone_mentions", "drone_density"]],
                           on=["country_iso3", "year"], how="outer")
    merged = merged.merge(imports[["country_iso3", "year", "drone_import_tiv"]],
                          on=["country_iso3", "year"], how="outer")

    # Merge with trajectory if available
    if not drone_trajectory.empty:
        traj_cols = [c for c in drone_trajectory.columns if c not in ["country_iso3", "year"]]
        merged = merged.merge(drone_trajectory, on=["country_iso3", "year"], how="outer")

    merged = merged.fillna(0)

    # Add ATT status and regime
    merged["att_status"] = merged.apply(
        lambda r: get_treaty_status(r["country_iso3"], "ATT", int(r["year"])), axis=1
    )
    if not vdem.empty:
        vdem_map = vdem.set_index(["country_iso3", "year"])["v2x_regime"].to_dict()
        merged["binary_regime"] = merged.apply(
            lambda r: "democracy" if vdem_map.get((r["country_iso3"], int(r["year"])), 0) >= 2 else "autocracy",
            axis=1,
        )
    else:
        merged["binary_regime"] = "unknown"

    return merged


# ---------------------------------------------------------------------------
# 6e. UNGA Votes on LAWS Resolutions
# ---------------------------------------------------------------------------

def compute_laws_voting(voting: pd.DataFrame) -> pd.DataFrame:
    """Filter voting data for LAWS/autonomous weapons resolutions."""
    if voting.empty or "treaty_flag" not in voting.columns:
        return pd.DataFrame()

    laws_votes = voting[voting["treaty_flag"] == "laws"].copy()
    # Also include CCW-related resolutions as context
    ccw_mask = voting["resolution_title"].str.lower().fillna("").str.contains(
        "convention on certain conventional weapons|ccw|conventional weapons convention"
    )
    ccw_votes = voting[ccw_mask & (voting["treaty_flag"] != "laws")].copy()
    ccw_votes["treaty_flag"] = "ccw_related"

    combined = pd.concat([laws_votes, ccw_votes], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    agg = combined.groupby(["country_iso3", "year", "treaty_flag"]).agg(
        pct_yes=("vote_numeric", lambda x: (x == 1).sum() / max(x.notna().sum(), 1)),
        n_votes=("vote_numeric", "count"),
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 6f. ATT Ratifier Drone Export Behavior
# ---------------------------------------------------------------------------

def compute_att_drone_behavior(drone_transfers: pd.DataFrame) -> pd.DataFrame:
    """Overlay drone exports on ATT ratification timeline."""
    if drone_transfers.empty:
        return pd.DataFrame()

    exports = drone_transfers[drone_transfers["role"] == "exporter"].copy()
    if exports.empty:
        return pd.DataFrame()

    # Aggregate to country-year
    cy = exports.groupby(["country_iso3", "year"]).agg(
        drone_export_tiv=("tiv", "sum"),
        n_deals=("tiv", "count"),
    ).reset_index()

    # Add ATT info
    cy["att_party"] = cy["country_iso3"].apply(lambda c: c in ATT_PARTIES)
    cy["att_ratification_year"] = cy["country_iso3"].apply(
        lambda c: ATT_PARTIES.get(c, np.nan)
    )
    cy["year_relative"] = cy.apply(
        lambda r: r["year"] - r["att_ratification_year"] if pd.notna(r["att_ratification_year"]) else np.nan,
        axis=1,
    )

    # Group: parties vs non-parties
    cy["group"] = cy["att_party"].apply(lambda x: "ATT party" if x else "Non-party")
    return cy


# ---------------------------------------------------------------------------
# 6g. Panel Dataset
# ---------------------------------------------------------------------------

def build_panel_dataset(
    drone_keywords: pd.DataFrame,
    drone_trajectory: pd.DataFrame,
    drone_transfers: pd.DataFrame,
    voting: pd.DataFrame,
    frame_scores: pd.DataFrame,
    vdem: pd.DataFrame,
) -> pd.DataFrame:
    """Build wide country-year panel combining all Q6 variables."""
    # Start from keywords (has the broadest coverage)
    base = drone_keywords[["country_iso3", "year", "drone_mentions", "drone_density"]].copy()

    # Merge trajectory
    if not drone_trajectory.empty:
        base = base.merge(drone_trajectory, on=["country_iso3", "year"], how="outer")

    # Merge transfers
    if not drone_transfers.empty:
        exp = drone_transfers[drone_transfers["role"] == "exporter"].groupby(
            ["country_iso3", "year"]
        )["tiv"].sum().reset_index().rename(columns={"tiv": "drone_export_tiv"})
        imp = drone_transfers[drone_transfers["role"] == "importer"].groupby(
            ["country_iso3", "year"]
        )["tiv"].sum().reset_index().rename(columns={"tiv": "drone_import_tiv"})
        base = base.merge(exp, on=["country_iso3", "year"], how="outer")
        base = base.merge(imp, on=["country_iso3", "year"], how="outer")

    # Merge frame scores
    if not frame_scores.empty and "frame_ratio_mean" in frame_scores.columns:
        fr = frame_scores[["country_iso3", "year", "frame_ratio_mean"]].copy()
        base = base.merge(fr, on=["country_iso3", "year"], how="left")

    base = base.fillna(0)

    # Add group annotations
    from src.data.groups import NWS, NATO, NAM
    base["att_status"] = base.apply(
        lambda r: get_treaty_status(r["country_iso3"], "ATT", int(r["year"])), axis=1
    )
    base["p5"] = base["country_iso3"].isin(NWS)
    base["nato"] = base["country_iso3"].isin(NATO)
    base["nam"] = base["country_iso3"].isin(NAM)

    if not vdem.empty:
        vdem_map = vdem.set_index(["country_iso3", "year"])["v2x_regime"].to_dict()
        base["binary_regime"] = base.apply(
            lambda r: "democracy" if vdem_map.get((r["country_iso3"], int(r["year"])), 0) >= 2 else "autocracy",
            axis=1,
        )

    return base


# ---------------------------------------------------------------------------
# 6h. Rhetoric-Transfer OLS Regressions
# ---------------------------------------------------------------------------

def compute_rhetoric_transfer_regression(panel: pd.DataFrame) -> pd.DataFrame:
    """OLS regressions: drone rhetoric → drone transfers."""
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return pd.DataFrame()

    results = []

    # Filter to rows with nonzero exports
    exporters = panel[panel.get("drone_export_tiv", pd.Series(dtype=float)) > 0].copy()
    if len(exporters) < 10:
        return pd.DataFrame()

    exporters["log_export_tiv"] = np.log1p(exporters["drone_export_tiv"])

    # Define independent variables to test
    iv_candidates = [
        ("drone_mentions", "Drone keyword mentions"),
        ("drone_density", "Drone keyword density"),
        ("laws_drones", "LAWS anchor similarity"),
        ("att_drones", "ATT-drone anchor similarity"),
    ]

    for iv_col, iv_label in iv_candidates:
        if iv_col not in exporters.columns:
            continue
        valid = exporters[[iv_col, "log_export_tiv"]].dropna()
        if len(valid) < 10:
            continue
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
            valid[iv_col], valid["log_export_tiv"]
        )
        results.append({
            "independent_var": iv_label,
            "dependent_var": "log(1 + drone_export_tiv)",
            "n": len(valid),
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r_value**2, 4),
            "p_value": round(p_value, 6),
            "std_err": round(std_err, 4),
        })

    # Also run by regime type
    for regime in ["democracy", "autocracy"]:
        sub = exporters[exporters.get("binary_regime", "") == regime]
        if len(sub) < 10:
            continue
        for iv_col, iv_label in iv_candidates[:2]:  # just keyword measures
            if iv_col not in sub.columns:
                continue
            valid = sub[[iv_col, "log_export_tiv"]].dropna()
            if len(valid) < 10:
                continue
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
                valid[iv_col], valid["log_export_tiv"]
            )
            results.append({
                "independent_var": f"{iv_label} ({regime})",
                "dependent_var": "log(1 + drone_export_tiv)",
                "n": len(valid),
                "slope": round(slope, 4),
                "intercept": round(intercept, 4),
                "r_squared": round(r_value**2, 4),
                "p_value": round(p_value, 6),
                "std_err": round(std_err, 4),
            })

    # ATT party vs non-party comparison (t-test on log exports)
    att_party = exporters[exporters["att_status"] == "party"]["log_export_tiv"]
    att_non = exporters[exporters["att_status"] != "party"]["log_export_tiv"]
    if len(att_party) >= 5 and len(att_non) >= 5:
        t_stat, t_p = sp_stats.ttest_ind(att_party, att_non, equal_var=False)
        results.append({
            "independent_var": "ATT party (t-test vs non-party)",
            "dependent_var": "log(1 + drone_export_tiv)",
            "n": len(att_party) + len(att_non),
            "slope": round(att_party.mean() - att_non.mean(), 4),
            "intercept": round(att_non.mean(), 4),
            "r_squared": round(t_stat, 4),
            "p_value": round(t_p, 6),
            "std_err": 0,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_q6(data: dict) -> dict:
    """Run all Q6 sub-analyses."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    corpus = data.get("corpus", pd.DataFrame())
    cy_emb = data.get("country_year_embeddings", pd.DataFrame())
    anchor_emb = data.get("anchor_embeddings", {})
    trade_reg = data.get("trade_register", pd.DataFrame())
    voting = data.get("voting", pd.DataFrame())
    frame_scores = data.get("frame_scores", pd.DataFrame())
    vdem = data.get("vdem", pd.DataFrame())

    # 6a
    print("[Q6] 6a: Drone keyword extraction...")
    kw = compute_drone_keywords(corpus)
    kw.to_csv(os.path.join(OUTPUT_DIR, "drone_keyword_extraction.csv"), index=False)
    results["drone_keywords"] = kw
    print(f"[Q6] Saved drone_keyword_extraction.csv ({len(kw)} country-year rows with drone mentions)")

    # 6b
    print("[Q6] 6b: Drone rhetoric trajectory...")
    traj = compute_drone_rhetoric_trajectory(cy_emb, anchor_emb)
    if not traj.empty:
        traj.to_csv(os.path.join(OUTPUT_DIR, "drone_rhetoric_trajectory.csv"), index=False)
    results["drone_trajectory"] = traj
    print(f"[Q6] Saved drone_rhetoric_trajectory.csv ({len(traj)} rows)")

    # 6c
    print("[Q6] 6c: Drone transfer data (SIPRI)...")
    transfers = parse_drone_transfers(trade_reg)
    if not transfers.empty:
        transfers.to_csv(os.path.join(OUTPUT_DIR, "drone_transfers.csv"), index=False)
        n_exp = len(transfers[transfers["role"] == "exporter"])
        n_imp = len(transfers[transfers["role"] == "importer"])
        print(f"[Q6] Saved drone_transfers.csv ({n_exp} export deals, {n_imp} import deals)")
    else:
        print("[Q6] No drone transfers found in trade register.")
    results["drone_transfers"] = transfers

    # 6d
    print("[Q6] 6d: Rhetoric-transfer gap...")
    gap = compute_rhetoric_transfer_gap(kw, traj, transfers, vdem)
    if not gap.empty:
        gap.to_csv(os.path.join(OUTPUT_DIR, "rhetoric_transfer_gap.csv"), index=False)
    results["rhetoric_transfer_gap"] = gap
    print(f"[Q6] Saved rhetoric_transfer_gap.csv ({len(gap)} rows)")

    # 6e
    print("[Q6] 6e: UNGA votes on LAWS resolutions...")
    laws_votes = compute_laws_voting(voting)
    if not laws_votes.empty:
        laws_votes.to_csv(os.path.join(OUTPUT_DIR, "laws_voting_patterns.csv"), index=False)
    results["laws_voting"] = laws_votes
    print(f"[Q6] Saved laws_voting_patterns.csv ({len(laws_votes)} rows)")

    # 6f
    print("[Q6] 6f: ATT ratifier drone export behavior...")
    att_drone = compute_att_drone_behavior(transfers)
    if not att_drone.empty:
        att_drone.to_csv(os.path.join(OUTPUT_DIR, "att_drone_behavior.csv"), index=False)
    results["att_drone_behavior"] = att_drone
    print(f"[Q6] Saved att_drone_behavior.csv ({len(att_drone)} rows)")

    # 6g
    print("[Q6] 6g: Panel dataset...")
    panel = build_panel_dataset(kw, traj, transfers, voting, frame_scores, vdem)
    panel.to_csv(os.path.join(OUTPUT_DIR, "panel_dataset_drones.csv"), index=False)
    results["panel_dataset_drones"] = panel
    print(f"[Q6] Saved panel_dataset_drones.csv ({len(panel)} rows)")

    # 6h
    print("[Q6] 6h: Rhetoric-transfer regressions...")
    regressions = compute_rhetoric_transfer_regression(panel)
    if not regressions.empty:
        regressions.to_csv(os.path.join(OUTPUT_DIR, "rhetoric_transfer_regressions.csv"), index=False)
    results["regressions"] = regressions
    print(f"[Q6] Saved rhetoric_transfer_regressions.csv ({len(regressions)} models)")

    print("[Q6] Complete.")
    return results
