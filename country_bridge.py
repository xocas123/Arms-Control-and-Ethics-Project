"""
country_bridge.py — Runtime ISO3 reconciliation between the two pipelines.

NLP pipeline  → uses ISO3 codes  (country_code: "USA", "RUS", …)
Network pipeline → uses full names (country: "United States", "Russia", …)

The bridge is built at runtime from actual data files; no names are hardcoded.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Manual overrides for SIPRI-specific anomalies ─────────────────────────────
# These are names that appear in the network data but don't resolve via any
# automated lookup. None = skip this entity (supranational, unknown, etc.)
MANUAL_OVERRIDES: dict[str, str | None] = {
    "korea, south":              "KOR",
    "korea, north":              "PRK",
    "bosnia-herzegovina":        "BIH",
    "bosnia and herzegovina":    "BIH",
    "cabo verde":                "CPV",
    "cape verde":                "CPV",
    "czechia":                   "CZE",
    "czech republic":            "CZE",
    "cote d'ivoire":             "CIV",
    "ivory coast":               "CIV",
    "congo, dr":                 "COD",
    "dr congo":                  "COD",
    "democratic republic of the congo": "COD",
    "congo, republic":           "COG",
    "republic of the congo":     "COG",
    "timor-leste":                "TLS",
    "east timor":                "TLS",
    "eswatini":                  "SWZ",
    "swaziland":                 "SWZ",
    "north macedonia":           "MKD",
    "macedonia":                 "MKD",
    "moldova":                   "MDA",
    "republic of moldova":       "MDA",
    "taiwan":                    "TWN",
    "brunei":                    "BRN",
    "laos":                      "LAO",
    "syria":                     "SYR",
    "iran":                      "IRN",
    "south korea":               "KOR",
    "north korea":               "PRK",
    "russia":                    "RUS",
    "south africa":              "ZAF",
    "united arab emirates":      "ARE",
    "uae":                       "ARE",
    "uk":                        "GBR",
    "usa":                       "USA",
    # Supranational / unknown — explicitly skip
    "african union**":           None,
    "eu":                        None,
    "nato":                      None,
    "un":                        None,
    "undp":                      None,
    "unprofor":                  None,
}


def _load_country_groups_json(nlp_data_dir: Path) -> dict[str, str]:
    """Inverts country_groups.json to produce {lowercase_name: iso3}."""
    path = nlp_data_dir / "country_groups.json"
    if not path.exists():
        logger.warning("country_groups.json not found at %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mapping: dict[str, str] = {}
    # Support both {"iso3": "name"} and {"name": "iso3"} formats
    iso3_to_name = data.get("country_iso3_to_name", data.get("iso3_to_name", {}))
    for iso3, name in iso3_to_name.items():
        if isinstance(name, str):
            mapping[name.lower()] = iso3.upper()
    # Also accept flat dict keyed by name
    if not iso3_to_name:
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str) and len(v) == 3:
                mapping[k.lower()] = v.upper()
    return mapping


def _load_sipri_name_map(net_src_dir: Path) -> dict[str, str]:
    """
    Dynamically imports utils.py from the network pipeline and extracts
    COUNTRY_NAME_MAP ({raw_name: canonical_name}).
    Returns {} if the file doesn't exist or doesn't have the map.
    """
    utils_path = net_src_dir / "utils.py"
    if not utils_path.exists():
        logger.warning("utils.py not found at %s", utils_path)
        return {}
    try:
        spec = importlib.util.spec_from_file_location("_net_utils", utils_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        raw_map: dict = getattr(mod, "COUNTRY_NAME_MAP", {})
        return {k.lower(): v for k, v in raw_map.items() if isinstance(v, str)}
    except Exception as exc:
        logger.warning("Could not load COUNTRY_NAME_MAP from utils.py: %s", exc)
        return {}


def _pycountry_lookup(name: str) -> str | None:
    """Tries pycountry alpha_3 lookup. Returns None if unavailable."""
    try:
        import pycountry
        result = pycountry.countries.search_fuzzy(name)
        if result:
            return result[0].alpha_3
    except Exception:
        pass
    return None


def build_bridge(nlp_data_dir: Path, net_src_dir: Path) -> dict[str, str | None]:
    """
    Build and return the name→ISO3 bridge dict.
    Keys: lowercase canonical full names.
    Values: ISO3 string, or None for explicitly-skipped supranational entities.

    Resolution order:
      1. MANUAL_OVERRIDES (hard wins)
      2. country_groups.json inversion
      3. SIPRI COUNTRY_NAME_MAP canonical names → json lookup
      4. pycountry fuzzy search (if installed)
    """
    base = _load_country_groups_json(nlp_data_dir)
    sipri_map = _load_sipri_name_map(net_src_dir)  # raw → canonical

    bridge: dict[str, str | None] = {}
    bridge.update(base)

    # For each SIPRI canonical name, try to find its ISO3 via base mapping
    for raw_lower, canonical in sipri_map.items():
        canon_lower = canonical.lower()
        if canon_lower in base:
            bridge[raw_lower] = base[canon_lower]
        elif raw_lower in base:
            bridge[raw_lower] = base[raw_lower]

    # Apply manual overrides last (they win)
    bridge.update({k.lower(): v for k, v in MANUAL_OVERRIDES.items()})

    logger.info("Bridge built: %d name→ISO3 mappings", sum(1 for v in bridge.values() if v))
    return bridge


def resolve_name(name: str, bridge: dict[str, str | None],
                 fuzzy: bool = True) -> str | None:
    """
    Resolve a single country name to ISO3.
    Returns None if unresolvable or explicitly skipped.
    """
    key = name.lower().strip()
    if key in bridge:
        return bridge[key]

    # Try pycountry first (more reliable than difflib for proper names)
    iso3 = _pycountry_lookup(name)
    if iso3:
        bridge[key] = iso3  # cache
        return iso3

    if fuzzy:
        candidates = list(bridge.keys())
        matches = difflib.get_close_matches(key, candidates, n=1, cutoff=0.85)
        if matches:
            resolved = bridge[matches[0]]
            logger.debug("Fuzzy matched '%s' → '%s' → %s", name, matches[0], resolved)
            bridge[key] = resolved  # cache
            return resolved

    return None


def add_country_code(df: pd.DataFrame, country_col: str,
                     bridge: dict[str, str | None]) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds a 'country_code' column to df by resolving `country_col` via bridge.
    Rows that cannot be matched get country_code = NaN.

    Returns (df_with_country_code, list_of_unresolved_names).
    """
    df = df.copy()
    resolved = df[country_col].apply(lambda n: resolve_name(str(n), bridge))
    df["country_code"] = resolved
    unresolved = df.loc[df["country_code"].isna(), country_col].unique().tolist()
    if unresolved:
        logger.debug(
            "%d unique names unresolved in '%s': %s",
            len(unresolved), country_col, unresolved[:10]
        )
    return df, unresolved
