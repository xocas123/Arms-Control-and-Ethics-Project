"""
Country group utilities for the arms control NLP pipeline.

Loads country_groups.json and provides helper functions to query group membership,
aggregate metrics by group, and map ISO3 codes to country names.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Default path resolution
# ---------------------------------------------------------------------------

def _default_groups_path() -> Path:
    return Path(__file__).parent.parent / "data" / "raw" / "country_groups.json"


# ---------------------------------------------------------------------------
# Internal loader (cached at module level after first call)
# ---------------------------------------------------------------------------

_groups_data: Optional[dict] = None


def _load_data(path: Optional[Path] = None) -> dict:
    global _groups_data
    if _groups_data is None:
        p = path or _default_groups_path()
        with open(p, "r", encoding="utf-8") as f:
            _groups_data = json.load(f)
    return _groups_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_group_members(group_name: str, path: Optional[Path] = None) -> List[str]:
    """Return list of ISO3 codes for a named country group."""
    data = _load_data(path)
    if group_name not in data:
        raise KeyError(f"Unknown group '{group_name}'. Available: {list_groups(path)}")
    return list(data[group_name])


def list_groups(path: Optional[Path] = None) -> List[str]:
    """Return all group names (excluding the iso3 mapping key)."""
    data = _load_data(path)
    return [k for k in data.keys() if k != "country_iso3_to_name"]


def get_country_groups(iso3: str, path: Optional[Path] = None) -> List[str]:
    """Return list of group names that the given ISO3 country belongs to."""
    data = _load_data(path)
    result = []
    for group_name, members in data.items():
        if group_name == "country_iso3_to_name":
            continue
        if isinstance(members, list) and iso3 in members:
            result.append(group_name)
    return result


def get_iso3_to_name(path: Optional[Path] = None) -> dict:
    """Return the full ISO3 → country name mapping dict."""
    data = _load_data(path)
    return dict(data.get("country_iso3_to_name", {}))


def iso3_to_name(iso3: str, path: Optional[Path] = None) -> str:
    """Convert ISO3 code to canonical country name. Returns the code itself if not found."""
    mapping = get_iso3_to_name(path)
    return mapping.get(iso3, iso3)


def aggregate_by_group(
    df: pd.DataFrame,
    group_name: str,
    value_col: str,
    agg: str = "mean",
    country_col: str = "country_code",
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Filter df to rows belonging to group_name, then aggregate value_col.

    Parameters
    ----------
    df : DataFrame with at least columns [country_col, value_col]
    group_name : str
    value_col : str
    agg : str  aggregation function name ('mean', 'sum', 'median', etc.)
    country_col : str
    path : optional path to country_groups.json

    Returns
    -------
    DataFrame with aggregated value_col, grouped by year (if present) or scalar.
    """
    members = get_group_members(group_name, path)
    subset = df[df[country_col].isin(members)].copy()
    if subset.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["year"] if c in subset.columns]
    if group_cols:
        result = subset.groupby(group_cols)[value_col].agg(agg).reset_index()
    else:
        result = pd.DataFrame({value_col: [getattr(subset[value_col], agg)()]})
    result["group"] = group_name
    return result


def aggregate_by_regime_type(
    df: pd.DataFrame,
    regime_col: str,
    value_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate value_col by the regime_col categories.

    Parameters
    ----------
    df : DataFrame
    regime_col : str  column with regime type labels
    value_col : str
    agg : str

    Returns
    -------
    DataFrame with columns [regime_col, value_col]
    """
    group_cols = [regime_col] + ([c for c in ["year"] if c in df.columns])
    result = df.groupby(group_cols)[value_col].agg(agg).reset_index()
    return result


def build_group_lookup(path: Optional[Path] = None) -> dict:
    """
    Return a dict mapping each ISO3 code to its list of group memberships.
    Useful for vectorised operations.
    """
    data = _load_data(path)
    lookup: dict = {}
    for group_name, members in data.items():
        if group_name == "country_iso3_to_name":
            continue
        if isinstance(members, list):
            for iso3 in members:
                lookup.setdefault(iso3, []).append(group_name)
    return lookup


def assign_regime_type(iso3: str, path: Optional[Path] = None) -> str:
    """
    Heuristically assign a regime-type label based on group membership.

    Categories (priority order):
      p5_authoritarian, p5_democracy, de_facto_nuclear,
      nato_democracy, eu_democracy, humanitarian_coalition,
      gulf_autocracy, nam_state, other
    """
    groups = set(get_country_groups(iso3, path))

    authoritarian_p5 = {"RUS", "CHN"}
    if iso3 in authoritarian_p5 and "p5" in groups:
        return "p5_authoritarian"
    if "p5" in groups:
        return "p5_democracy"
    if "de_facto_nuclear" in groups:
        return "de_facto_nuclear"
    if "nac" in groups:
        return "humanitarian_coalition"
    if "gulf_states" in groups:
        return "gulf_autocracy"
    if "nato" in groups or "eu" in groups:
        return "nato_eu_democracy"
    if "nam" in groups:
        return "nam_state"
    return "other"
