"""
Named entity recognition and commitment phrase extraction for arms control texts.

Uses regex/keyword matching — no spaCy model download required.
"""

import logging
import re
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Organisation / instrument names to detect
# ---------------------------------------------------------------------------

ORGS = [
    "IAEA", "OPCW", "NATO", "United Nations", "NPT", "ATT", "TPNW", "BWC",
    "CWC", "CTBTO", "P5", "P5+1", "Conference on Disarmament", "CD",
    "First Committee", "General Assembly", "Security Council",
    "UN Secretary-General", "OSCE", "ASEAN", "African Union", "AU",
    "League of Arab States", "Non-Aligned Movement", "NAM",
    "Group of 77", "G77", "European Union", "EU",
]

ORG_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(o) for o in sorted(ORGS, key=len, reverse=True)) + r")\b"
)

# Treaty names
TREATY_NAMES = [
    "Treaty on the Non-Proliferation of Nuclear Weapons",
    "Arms Trade Treaty",
    "Treaty on the Prohibition of Nuclear Weapons",
    "Ottawa Treaty", "Mine Ban Treaty",
    "Chemical Weapons Convention",
    "Biological Weapons Convention",
    "Convention on Cluster Munitions",
    "Comprehensive Nuclear-Test-Ban Treaty",
    "New START", "START Treaty",
    "INF Treaty",
    "Open Skies Treaty",
    "Outer Space Treaty",
]

TREATY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(TREATY_NAMES, key=len, reverse=True)) + r")\b"
)

# Country names (from utils.py)
try:
    from src.utils import NAME_TO_ISO3, ISO3_TO_NAME
    _COUNTRY_NAMES = sorted(NAME_TO_ISO3.keys(), key=len, reverse=True)
    _COUNTRY_PATTERN = re.compile(
        r"\b(?:" + "|".join(re.escape(c) for c in _COUNTRY_NAMES[:80]) + r")\b",
        flags=re.IGNORECASE,
    )
    _HAS_COUNTRY_MAP = True
except ImportError:
    _HAS_COUNTRY_MAP = False
    _COUNTRY_PATTERN = re.compile(r"(?!)")  # never matches

# ---------------------------------------------------------------------------
# Commitment phrase detection
# ---------------------------------------------------------------------------

_STRONG_PATTERNS = re.compile(
    r"\b(we pledge|we commit\b|we have ratified|we call for immediate|"
    r"we solemnly commit|we hereby commit|we firmly commit|"
    r"we have signed|we have acceded|full implementation of)\b",
    flags=re.IGNORECASE,
)

_MODERATE_PATTERNS = re.compile(
    r"\b(we support|we encourage|we recognize the importance|we welcome|"
    r"we endorse|we affirm|we reaffirm|we call for|we call upon|"
    r"we strongly support|we firmly support|our delegation supports)\b",
    flags=re.IGNORECASE,
)

_WEAK_PATTERNS = re.compile(
    r"\b(we note|we acknowledge concern|while we understand|we take note|"
    r"we are aware|we observe|we consider|we believe|we think|we feel)\b",
    flags=re.IGNORECASE,
)

_OPPOSITION_PATTERNS = re.compile(
    r"\b(we reject|we cannot accept|unacceptable|we oppose|we object to|"
    r"we strongly oppose|we firmly oppose|we refuse|we deny)\b",
    flags=re.IGNORECASE,
)

_CONDITIONAL_PATTERNS = re.compile(
    r"\b(we would support if|provided that|subject to|conditional on|"
    r"contingent upon|only if|on condition that|with the understanding that)\b",
    flags=re.IGNORECASE,
)


def extract_commitment_phrases(text: str) -> List[Tuple[str, str]]:
    """
    Extract (sentence, commitment_strength) tuples from a text.

    Commitment strength labels: 'strong', 'moderate', 'weak', 'opposition', 'conditional'

    Parameters
    ----------
    text : str

    Returns
    -------
    list of (sentence, strength_label) tuples
    """
    from src.data.preprocess import tokenize_sentences

    sentences = tokenize_sentences(text)
    results = []
    for sent in sentences:
        if _STRONG_PATTERNS.search(sent):
            results.append((sent, "strong"))
        elif _OPPOSITION_PATTERNS.search(sent):
            results.append((sent, "opposition"))
        elif _CONDITIONAL_PATTERNS.search(sent):
            results.append((sent, "conditional"))
        elif _MODERATE_PATTERNS.search(sent):
            results.append((sent, "moderate"))
        elif _WEAK_PATTERNS.search(sent):
            results.append((sent, "weak"))

    return results


def extract_entities(
    segments_df: pd.DataFrame,
    text_col: str = "text",
    country_col: str = "country_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Extract mentions of organisations, treaties, and countries from speech segments.

    Parameters
    ----------
    segments_df : DataFrame

    Returns
    -------
    DataFrame with columns:
        country_code, year, entity, entity_type, count
    """
    rows = []
    for _, row in segments_df.iterrows():
        text = str(row.get(text_col, ""))
        iso3 = row.get(country_col, "UNK")
        year = row.get(year_col, 0)

        for match in ORG_PATTERN.finditer(text):
            rows.append(
                {country_col: iso3, year_col: year,
                 "entity": match.group(), "entity_type": "organisation"}
            )
        for match in TREATY_PATTERN.finditer(text):
            rows.append(
                {country_col: iso3, year_col: year,
                 "entity": match.group(), "entity_type": "treaty"}
            )
        if _HAS_COUNTRY_MAP:
            for match in _COUNTRY_PATTERN.finditer(text):
                mentioned = match.group().lower()
                from src.utils import normalize_country
                mentioned_iso3 = normalize_country(mentioned)
                if mentioned_iso3 and mentioned_iso3 != iso3:
                    rows.append(
                        {country_col: iso3, year_col: year,
                         "entity": mentioned_iso3, "entity_type": "country_mention"}
                    )

    if not rows:
        return pd.DataFrame(columns=[country_col, year_col, "entity", "entity_type", "count"])

    df = pd.DataFrame(rows)
    # Count
    agg = (
        df.groupby([country_col, year_col, "entity", "entity_type"])
        .size()
        .reset_index(name="count")
    )
    return agg


def build_mention_network(
    entities_df: pd.DataFrame,
    year: Optional[int] = None,
    country_col: str = "country_code",
    year_col: str = "year",
):
    """
    Build a NetworkX DiGraph of country-to-country mention relationships.

    Parameters
    ----------
    entities_df : output of extract_entities
    year : int or None (if None, aggregate all years)

    Returns
    -------
    networkx.DiGraph  (source=mentioning country, target=mentioned country)
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed; cannot build mention network.")
        return None

    if year is not None:
        df = entities_df[entities_df[year_col] == year]
    else:
        df = entities_df

    df = df[df["entity_type"] == "country_mention"]

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = row[country_col]
        tgt = row["entity"]
        w = row.get("count", 1)
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w
        else:
            G.add_edge(src, tgt, weight=w)
    return G
