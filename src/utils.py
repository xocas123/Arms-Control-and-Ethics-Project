"""
General utility functions for the arms control NLP pipeline.

Includes:
- Country name normalisation (name → ISO3)
- ISO3 → canonical name lookup
- Text cleaning utilities
- File I/O helpers (parquet, JSON, CSV)
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Country name → ISO3 mapping  (at least 80 countries)
# ---------------------------------------------------------------------------

NAME_TO_ISO3: dict = {
    # P5 / NATO core
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "us": "USA",
    "russia": "RUS",
    "russian federation": "RUS",
    "china": "CHN",
    "people's republic of china": "CHN",
    "united kingdom": "GBR",
    "uk": "GBR",
    "great britain": "GBR",
    "france": "FRA",
    "germany": "DEU",
    "federal republic of germany": "DEU",
    "italy": "ITA",
    "spain": "ESP",
    "netherlands": "NLD",
    "belgium": "BEL",
    "norway": "NOR",
    "denmark": "DNK",
    "canada": "CAN",
    "portugal": "PRT",
    "greece": "GRC",
    "hungary": "HUN",
    "czech republic": "CZE",
    "czechia": "CZE",
    "romania": "ROU",
    "bulgaria": "BGR",
    "slovakia": "SVK",
    "croatia": "HRV",
    "albania": "ALB",
    "montenegro": "MNE",
    "north macedonia": "MKD",
    "macedonia": "MKD",
    "slovenia": "SVN",
    "estonia": "EST",
    "latvia": "LVA",
    "lithuania": "LTU",
    "luxembourg": "LUX",
    "iceland": "ISL",
    "turkey": "TUR",
    "turkiye": "TUR",
    "poland": "POL",
    # Nuclear / proliferation-relevant
    "india": "IND",
    "pakistan": "PAK",
    "israel": "ISR",
    "north korea": "PRK",
    "democratic people's republic of korea": "PRK",
    # NAC / Humanitarian coalition
    "brazil": "BRA",
    "egypt": "EGY",
    "ireland": "IRL",
    "mexico": "MEX",
    "new zealand": "NZL",
    "south africa": "ZAF",
    "austria": "AUT",
    # BRICS / Major economies
    "indonesia": "IDN",
    "iran": "IRN",
    "islamic republic of iran": "IRN",
    "iraq": "IRQ",
    "syria": "SYR",
    "syrian arab republic": "SYR",
    "malaysia": "MYS",
    "nigeria": "NGA",
    "ethiopia": "ETH",
    "sudan": "SDN",
    "venezuela": "VEN",
    "colombia": "COL",
    "japan": "JPN",
    "australia": "AUS",
    "sweden": "SWE",
    "finland": "FIN",
    "south korea": "KOR",
    "republic of korea": "KOR",
    "thailand": "THA",
    "bolivia": "BOL",
    "costa rica": "CRI",
    "ecuador": "ECU",
    # Gulf states
    "saudi arabia": "SAU",
    "united arab emirates": "ARE",
    "uae": "ARE",
    "qatar": "QAT",
    "bahrain": "BHR",
    "kuwait": "KWT",
    "oman": "OMN",
    # Africa
    "algeria": "DZA",
    "morocco": "MAR",
    "tunisia": "TUN",
    "libya": "LBY",
    "kenya": "KEN",
    "tanzania": "TZA",
    "united republic of tanzania": "TZA",
    "uganda": "UGA",
    "ghana": "GHA",
    "cameroon": "CMR",
    "zimbabwe": "ZWE",
    "zambia": "ZMB",
    "mozambique": "MOZ",
    "namibia": "NAM",
    "senegal": "SEN",
    "mali": "MLI",
    "niger": "NER",
    "mauritania": "MRT",
    "burkina faso": "BFA",
    "chad": "TCD",
    "rwanda": "RWA",
    "cape verde": "CPV",
    # Latin America
    "cuba": "CUB",
    "argentina": "ARG",
    "chile": "CHL",
    "peru": "PER",
    "uruguay": "URY",
    "paraguay": "PRY",
    "nicaragua": "NIC",
    "guatemala": "GTM",
    "honduras": "HND",
    "el salvador": "SLV",
    "panama": "PAN",
    "jamaica": "JAM",
    "trinidad and tobago": "TTO",
    # Asia
    "vietnam": "VNM",
    "viet nam": "VNM",
    "philippines": "PHL",
    "singapore": "SGP",
    "myanmar": "MMR",
    "burma": "MMR",
    "bangladesh": "BGD",
    "sri lanka": "LKA",
    "nepal": "NPL",
    "afghanistan": "AFG",
    # Post-Soviet
    "ukraine": "UKR",
    "belarus": "BLR",
    "kazakhstan": "KAZ",
    "uzbekistan": "UZB",
    "azerbaijan": "AZE",
    "georgia": "GEO",
    # Middle East
    "jordan": "JOR",
    "lebanon": "LBN",
    "yemen": "YEM",
    # EU extras
    "cyprus": "CYP",
    "malta": "MLT",
    # Pacific
    "fiji": "FJI",
    "papua new guinea": "PNG",
}

ISO3_TO_NAME: dict = {
    "USA": "United States", "RUS": "Russia", "CHN": "China", "GBR": "United Kingdom",
    "FRA": "France", "DEU": "Germany", "IND": "India", "PAK": "Pakistan",
    "ISR": "Israel", "PRK": "North Korea", "TUR": "Turkey", "POL": "Poland",
    "ITA": "Italy", "ESP": "Spain", "NLD": "Netherlands", "BEL": "Belgium",
    "NOR": "Norway", "DNK": "Denmark", "CAN": "Canada", "PRT": "Portugal",
    "GRC": "Greece", "HUN": "Hungary", "CZE": "Czech Republic", "ROU": "Romania",
    "BGR": "Bulgaria", "SVK": "Slovakia", "HRV": "Croatia", "ALB": "Albania",
    "MNE": "Montenegro", "MKD": "North Macedonia", "SVN": "Slovenia", "EST": "Estonia",
    "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg", "ISL": "Iceland",
    "EGY": "Egypt", "ZAF": "South Africa", "IDN": "Indonesia", "IRN": "Iran",
    "IRQ": "Iraq", "SYR": "Syria", "MYS": "Malaysia", "NGA": "Nigeria",
    "ETH": "Ethiopia", "SDN": "Sudan", "VEN": "Venezuela", "COL": "Colombia",
    "BRA": "Brazil", "IRL": "Ireland", "MEX": "Mexico", "NZL": "New Zealand",
    "AUT": "Austria", "THA": "Thailand", "BOL": "Bolivia", "CRI": "Costa Rica",
    "ECU": "Ecuador", "JPN": "Japan", "AUS": "Australia", "SWE": "Sweden",
    "FIN": "Finland", "KOR": "South Korea", "SAU": "Saudi Arabia", "ARE": "United Arab Emirates",
    "QAT": "Qatar", "BHR": "Bahrain", "KWT": "Kuwait", "OMN": "Oman",
    "DZA": "Algeria", "MAR": "Morocco", "TUN": "Tunisia", "LBY": "Libya",
    "KEN": "Kenya", "TZA": "Tanzania", "UGA": "Uganda", "GHA": "Ghana",
    "CMR": "Cameroon", "ZWE": "Zimbabwe", "ZMB": "Zambia", "MOZ": "Mozambique",
    "NAM": "Namibia", "SEN": "Senegal", "MLI": "Mali", "NER": "Niger",
    "MRT": "Mauritania", "BFA": "Burkina Faso", "TCD": "Chad", "RWA": "Rwanda",
    "CPV": "Cape Verde", "CUB": "Cuba", "ARG": "Argentina", "CHL": "Chile",
    "PER": "Peru", "URY": "Uruguay", "PRY": "Paraguay", "NIC": "Nicaragua",
    "GTM": "Guatemala", "HND": "Honduras", "SLV": "El Salvador", "PAN": "Panama",
    "JAM": "Jamaica", "TTO": "Trinidad and Tobago", "VNM": "Vietnam",
    "PHL": "Philippines", "SGP": "Singapore", "MMR": "Myanmar", "BGD": "Bangladesh",
    "LKA": "Sri Lanka", "NPL": "Nepal", "AFG": "Afghanistan", "UKR": "Ukraine",
    "BLR": "Belarus", "KAZ": "Kazakhstan", "UZB": "Uzbekistan", "AZE": "Azerbaijan",
    "GEO": "Georgia", "JOR": "Jordan", "LBN": "Lebanon", "YEM": "Yemen",
    "CYP": "Cyprus", "MLT": "Malta", "FJI": "Fiji", "PNG": "Papua New Guinea",
}


# ---------------------------------------------------------------------------
# Country normalisation
# ---------------------------------------------------------------------------

def normalize_country(name: str) -> Optional[str]:
    """
    Convert a country name string to a 3-letter ISO3 code.

    Returns None if the country is not found.
    """
    if not isinstance(name, str):
        return None
    cleaned = name.strip().lower()
    # Direct lookup
    code = NAME_TO_ISO3.get(cleaned)
    if code:
        return code
    # Try removing leading articles: "the republic of ..."
    cleaned2 = re.sub(r"^(the\s+)?(republic of|kingdom of|state of|democratic republic of)\s+", "", cleaned)
    code = NAME_TO_ISO3.get(cleaned2)
    if code:
        return code
    # Already an ISO3 code?
    upper = name.strip().upper()
    if upper in ISO3_TO_NAME:
        return upper
    return None


def iso3_to_name(code: str) -> str:
    """Convert ISO3 code to canonical country name. Returns the code if not found."""
    return ISO3_TO_NAME.get(code.upper(), code)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    r"mr\.?\s+(president|chair|secretary)[^.]{0,80}\.",
    r"madam\s+(president|chair)[^.]{0,80}\.",
    r"allow me to (congratulate|thank|express)[^.]{0,120}\.",
    r"i have the honour\s+to[^.]{0,120}\.",
    r"on behalf of[^.]{0,80}delegation[^.]{0,80}\.",
    r"my delegation (wishes|would like)[^.]{0,120}\.",
    r"\[?translation\]?",
    r"check against delivery",
    r"as delivered",
    r"unofficial\s+translation",
]

_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS), flags=re.IGNORECASE
)


def clean_text(text: str) -> str:
    """
    Remove UN speech boilerplate, decode Unicode artefacts,
    and normalise whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Normalise unicode (NFC form)
    text = unicodedata.normalize("NFC", text)
    # Remove boilerplate phrases
    text = _BOILERPLATE_RE.sub(" ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Normalise dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentences(text: str) -> list:
    """
    Split text into sentences using a simple regex heuristic.
    Falls back to period-splitting.
    """
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)
    return [p.strip() for p in parts if p.strip()]


def detect_language(text: str) -> str:
    """
    Simple language detection heuristic based on non-ASCII character ratio.
    Returns 'en' for mostly ASCII text, else 'other'.
    """
    if not text:
        return "en"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    return "en" if ratio < 0.15 else "other"


def remove_stopwords(tokens: list, stopwords: Optional[set] = None) -> list:
    """Remove common English stopwords from a token list."""
    if stopwords is None:
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall", "that",
            "this", "these", "those", "it", "its", "we", "our", "us", "they",
            "their", "them", "he", "she", "his", "her", "i", "my", "you", "your",
            "which", "who", "whom", "what", "when", "where", "how", "all", "each",
            "not", "no", "nor", "so", "yet", "both", "either", "neither",
        }
    return [t for t in tokens if t.lower() not in stopwords]


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to Parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def save_json(obj: dict, path: Union[str, Path], indent: int = 2) -> None:
    """Save a dict to JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
