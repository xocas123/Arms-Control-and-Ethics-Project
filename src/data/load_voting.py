"""Load UNGA resolution-level voting data."""
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path


HUMANITARIAN_KEYWORDS = [
    "humanitarian", "civilian", "prohibition", "ban", "indiscriminate",
    "suffering", "cluster munition", "landmine", "nuclear ban", "mine-free",
    "victim assistance", "clearance", "explosive remnant"
]

SECURITY_KEYWORDS = [
    "deterrence", "stability", "balance", "risk reduction", "nonproliferation",
    "non-proliferation", "safeguards", "verification", "confidence-building",
    "transparency", "regional security", "security arrangement"
]

TREATY_PATTERNS = {
    "att": [r"arms trade treaty", r"\bATT\b", r"international arms transfers"],
    "tpnw": [r"treaty on the prohibition of nuclear weapons", r"\bTPNW\b", r"nuclear weapons convention"],
    "ottawa": [r"ottawa", r"anti-personnel mine", r"landmine", r"mine ban"],
    "ccm": [r"cluster munition", r"\bCCM\b", r"convention on cluster"],
    "npt": [r"non-proliferation treaty", r"\bNPT\b", r"nuclear non-proliferation"],
    "laws": [r"lethal autonomous", r"\bLAWS\b", r"autonomous weapon",
             r"killer robot", r"meaningful human control"],
}

def load_voting(data_dir="data/raw/unvotes"):
    """
    Load UNGA resolution-level voting data.

    Expects the three-file unvotes R package layout (TidyTuesday download):
      unvotes.csv     — rcid, country, country_code, vote
      roll_calls.csv  — rcid, session, date, unres, short, descr, ...
      issues.csv      — rcid, short_name, issue

    Returns DataFrame (one row per country per resolution):
    - rcid: resolution ID
    - country_iso3: str
    - year: int
    - vote: str ('yes', 'no', 'abstain', 'absent')
    - vote_numeric: float (1=yes, 0=abstain, -1=no, NaN=absent)
    - resolution_title: str
    - issue: str
    - frame_type: str ('humanitarian', 'security', 'mixed', 'other')
    - treaty_flag: str or None
    """
    data_path = Path(data_dir)
    votes_path = data_path / "unvotes.csv"
    rolls_path = data_path / "roll_calls.csv"
    issues_path = data_path / "issues.csv"

    if votes_path.exists() and rolls_path.exists():
        try:
            return _process_unvotes_three_file(votes_path, rolls_path,
                                               issues_path if issues_path.exists() else None)
        except Exception as e:
            print(f"[Voting] Error loading unvotes files: {e}")

    # Fallback: single merged CSV
    if data_path.exists():
        for csv_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                if _is_unvotes_format(df):
                    return _process_unvotes_single(df)
            except Exception as e:
                print(f"[Voting] Could not load {csv_file}: {e}")

    print("[Voting] No voting data found. Returning empty DataFrame.")
    return pd.DataFrame(columns=["rcid", "country_iso3", "year", "vote", "vote_numeric",
                                  "resolution_title", "issue", "frame_type", "treaty_flag"])


def classify_resolution_frame(title: str, full_text: str = None) -> str:
    """Classify resolution as 'humanitarian', 'security', or 'mixed'."""
    title_lower = (title or "").lower()
    text_lower = (full_text or "").lower()
    combined = title_lower + " " + text_lower

    h_score = sum(1 for kw in HUMANITARIAN_KEYWORDS if kw in combined)
    s_score = sum(1 for kw in SECURITY_KEYWORDS if kw in combined)

    if h_score > s_score:
        return "humanitarian"
    elif s_score > h_score:
        return "security"
    elif h_score > 0 and s_score > 0:
        return "mixed"
    else:
        return "other"


def flag_treaty(title: str, full_text: str = None) -> str:
    """Return treaty flag if resolution references a specific treaty."""
    combined = ((title or "") + " " + (full_text or "")).lower()
    for treaty, patterns in TREATY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return treaty
    return None


def _iso2_to_iso3(iso2: str) -> str:
    """Convert ISO-3166-1 alpha-2 to alpha-3. Returns the input unchanged if not found."""
    _MAP = {
        "AF":"AFG","AL":"ALB","DZ":"DZA","AD":"AND","AO":"AGO","AG":"ATG","AR":"ARG",
        "AM":"ARM","AU":"AUS","AT":"AUT","AZ":"AZE","BS":"BHS","BH":"BHR","BD":"BGD",
        "BB":"BRB","BY":"BLR","BE":"BEL","BZ":"BLZ","BJ":"BEN","BT":"BTN","BO":"BOL",
        "BA":"BIH","BW":"BWA","BR":"BRA","BN":"BRN","BG":"BGR","BF":"BFA","BI":"BDI",
        "CV":"CPV","KH":"KHM","CM":"CMR","CA":"CAN","CF":"CAF","TD":"TCD","CL":"CHL",
        "CN":"CHN","CO":"COL","KM":"COM","CD":"COD","CG":"COG","CR":"CRI","CI":"CIV",
        "HR":"HRV","CU":"CUB","CY":"CYP","CZ":"CZE","DK":"DNK","DJ":"DJI","DM":"DMA",
        "DO":"DOM","EC":"ECU","EG":"EGY","SV":"SLV","GQ":"GNQ","ER":"ERI","EE":"EST",
        "SZ":"SWZ","ET":"ETH","FJ":"FJI","FI":"FIN","FR":"FRA","GA":"GAB","GM":"GMB",
        "GE":"GEO","DE":"DEU","GH":"GHA","GR":"GRC","GD":"GRD","GT":"GTM","GN":"GIN",
        "GW":"GNB","GY":"GUY","HT":"HTI","HN":"HND","HU":"HUN","IS":"ISL","IN":"IND",
        "ID":"IDN","IR":"IRN","IQ":"IRQ","IE":"IRL","IL":"ISR","IT":"ITA","JM":"JAM",
        "JP":"JPN","JO":"JOR","KZ":"KAZ","KE":"KEN","KI":"KIR","KP":"PRK","KR":"KOR",
        "KW":"KWT","KG":"KGZ","LA":"LAO","LV":"LVA","LB":"LBN","LS":"LSO","LR":"LBR",
        "LY":"LBY","LI":"LIE","LT":"LTU","LU":"LUX","MG":"MDG","MW":"MWI","MY":"MYS",
        "MV":"MDV","ML":"MLI","MT":"MLT","MH":"MHL","MR":"MRT","MU":"MUS","MX":"MEX",
        "FM":"FSM","MD":"MDA","MC":"MCO","MN":"MNG","ME":"MNE","MA":"MAR","MZ":"MOZ",
        "MM":"MMR","NA":"NAM","NP":"NPL","NL":"NLD","NZ":"NZL","NI":"NIC","NE":"NER",
        "NG":"NGA","MK":"MKD","NO":"NOR","OM":"OMN","PK":"PAK","PW":"PLW","PA":"PAN",
        "PG":"PNG","PY":"PRY","PE":"PER","PH":"PHL","PL":"POL","PT":"PRT","QA":"QAT",
        "RO":"ROU","RU":"RUS","RW":"RWA","KN":"KNA","LC":"LCA","VC":"VCT","WS":"WSM",
        "SM":"SMR","ST":"STP","SA":"SAU","SN":"SEN","RS":"SRB","SC":"SYC","SL":"SLE",
        "SK":"SVK","SI":"SVN","SB":"SLB","SO":"SOM","ZA":"ZAF","SS":"SSD","ES":"ESP",
        "LK":"LKA","SD":"SDN","SR":"SUR","SE":"SWE","CH":"CHE","SY":"SYR","TW":"TWN",
        "TJ":"TJK","TZ":"TZA","TH":"THA","TL":"TLS","TG":"TGO","TO":"TON","TT":"TTO",
        "TN":"TUN","TR":"TUR","TM":"TKM","TV":"TUV","UG":"UGA","UA":"UKR","AE":"ARE",
        "GB":"GBR","US":"USA","UY":"URY","UZ":"UZB","VU":"VUT","VE":"VEN","VN":"VNM",
        "YE":"YEM","ZM":"ZMB","ZW":"ZWE","PS":"PSE","XK":"XKX","VA":"VAT",
    }
    return _MAP.get(str(iso2).upper(), iso2)


def _process_unvotes_three_file(
    votes_path: Path, rolls_path: Path, issues_path: Path | None
) -> pd.DataFrame:
    """
    Join unvotes.csv + roll_calls.csv (+ optional issues.csv) into the
    pipeline's unified voting DataFrame.
    """
    votes  = pd.read_csv(votes_path)   # rcid, country, country_code, vote
    rolls  = pd.read_csv(rolls_path)   # rcid, session, date, unres, short, descr, ...

    # Year from date column (format: "1946-01-01")
    rolls["year"] = pd.to_datetime(rolls["date"], errors="coerce").dt.year.astype("Int64")

    # Resolution title: prefer descr, fall back to short
    title_col = "descr" if "descr" in rolls.columns else "short"
    rolls["resolution_title"] = rolls[title_col].fillna("Unknown")

    # Merge votes with roll-call metadata
    merged = votes.merge(
        rolls[["rcid", "year", "resolution_title"]],
        on="rcid", how="left"
    )

    # Country codes: unvotes uses ISO-2 in country_code
    if "country_code" in merged.columns:
        merged["country_iso3"] = merged["country_code"].apply(_iso2_to_iso3)
    else:
        from src.data.load_ungdc import _standardize_country
        merged["country_iso3"] = merged["country"].apply(_standardize_country)

    # Standardize vote
    vote_map = {"yes": "yes", "no": "no", "abstain": "abstain"}
    merged["vote"] = merged["vote"].str.lower().map(vote_map).fillna("absent")
    merged["vote_numeric"] = merged["vote"].map(
        {"yes": 1.0, "abstain": 0.0, "no": -1.0, "absent": np.nan}
    )

    # Frame + treaty
    merged["frame_type"]  = merged["resolution_title"].apply(classify_resolution_frame)
    merged["treaty_flag"] = merged["resolution_title"].apply(flag_treaty)

    # Issue: join from issues.csv if available, else classify
    if issues_path is not None:
        issues = pd.read_csv(issues_path)  # rcid, short_name, issue
        # Keep the primary issue per rcid
        issues_primary = issues.groupby("rcid")["issue"].first().reset_index()
        merged = merged.merge(issues_primary, on="rcid", how="left")
    else:
        merged["issue"] = merged["resolution_title"].apply(_classify_issue)

    merged["issue"] = merged["issue"].fillna(
        merged["resolution_title"].apply(_classify_issue)
    )

    result = merged[["rcid", "country_iso3", "year", "vote", "vote_numeric",
                     "resolution_title", "issue", "frame_type", "treaty_flag"]]
    result = result.dropna(subset=["country_iso3", "year"])
    result["year"] = result["year"].astype(int)
    result["rcid"] = result["rcid"].astype(str)

    print(f"[Voting] Loaded {len(result)} vote records  "
          f"({result['rcid'].nunique()} resolutions, "
          f"{result['country_iso3'].nunique()} countries, "
          f"{int(result['year'].min())}-{int(result['year'].max())})")
    return result


def _is_unvotes_format(df: pd.DataFrame) -> bool:
    required = {"rcid", "country", "vote"}
    return required.issubset(set(df.columns))


def _process_unvotes_single(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: process a single pre-merged unvotes CSV."""
    from src.data.load_ungdc import _standardize_country

    result = df.copy()
    result["country_iso3"] = result["country"].apply(_standardize_country)

    vote_map = {"yes": "yes", "no": "no", "abstain": "abstain", "na": "absent"}
    result["vote"] = result["vote"].str.lower().map(vote_map).fillna("absent")
    result["vote_numeric"] = result["vote"].map(
        {"yes": 1.0, "abstain": 0.0, "no": -1.0, "absent": np.nan}
    )

    title_col = next((c for c in ["descr", "short", "resolution", "title"] if c in df.columns), None)
    result["resolution_title"] = df[title_col] if title_col else "Unknown"
    result["frame_type"]  = result["resolution_title"].apply(classify_resolution_frame)
    result["treaty_flag"] = result["resolution_title"].apply(flag_treaty)

    if "issue" not in result.columns:
        result["issue"] = result["resolution_title"].apply(_classify_issue)

    return result[["rcid", "country_iso3", "year", "vote", "vote_numeric",
                   "resolution_title", "issue", "frame_type", "treaty_flag"]]


def _classify_issue(title: str) -> str:
    t = (title or "").lower()
    if any(kw in t for kw in ["nuclear", "npt", "ctbt", "non-proliferation"]):
        return "Nuclear weapons and nuclear material"
    elif any(kw in t for kw in ["disarmament", "arms control", "conventional", "landmine", "cluster", "att", "tpnw"]):
        return "Arms control and disarmament"
    elif any(kw in t for kw in ["human rights", "humanitarian"]):
        return "Human rights"
    else:
        return "Other"


