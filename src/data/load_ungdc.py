"""Load UN General Debate Corpus (UNGDC) speeches."""
import os
import re
import pandas as pd
from pathlib import Path

def load_ungdc(data_dir="data/raw/ungdc", year_range=(1970, 2023)):
    """
    Load UNGDC speeches from TXT files.

    Handles file naming: {ISO3}_{year}.txt or {country}_{year}.txt
    Also handles subdirectory structures.

    Returns DataFrame: country_iso3, country_name, year, text, source
    """
    # Try to load real data first
    data_path = Path(data_dir)
    speeches = []

    if data_path.exists():
        for txt_file in data_path.rglob("*.txt"):
            # Skip macOS metadata files
            if txt_file.name.startswith("._") or txt_file.name.startswith("."):
                continue
            stem = txt_file.stem
            parts = stem.split("_")
            try:
                if len(parts) == 3:
                    # Format: {ISO3}_{session}_{year}.txt  (UNGDC 2024 release)
                    country = parts[0]
                    year = int(parts[2])
                elif len(parts) == 2:
                    # Format: {ISO3}_{year}.txt  (older releases)
                    country = parts[0]
                    year = int(parts[1])
                else:
                    continue
                if year_range[0] <= year <= year_range[1]:
                    text = txt_file.read_text(encoding="utf-8", errors="replace")
                    speeches.append({
                        "country_raw": country,
                        "year": year,
                        "text": text,
                        "source": "real"
                    })
            except (ValueError, IndexError):
                pass

    if speeches:
        df = pd.DataFrame(speeches)
        # Standardize country codes
        df["country_iso3"] = df["country_raw"].apply(_standardize_country)
        df["country_name"] = df["country_iso3"].apply(_iso3_to_name)
        df = df[df["country_iso3"].notna()].drop(columns=["country_raw"])
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        print(f"[UNGDC] Loaded {len(df)} real speeches ({df['country_iso3'].nunique()} countries, {df['year'].min()}-{df['year'].max()})")
        return df

    print("[UNGDC] No data found. Returning empty DataFrame.")
    return pd.DataFrame(columns=["country_iso3", "country_name", "year", "text", "source"])


def _standardize_country(raw: str) -> str:
    """Convert raw country name/code to ISO3. Returns None if unrecognized."""
    # Common mappings
    _COMMON = {
        "USA": "USA", "US": "USA", "United States": "USA", "United States of America": "USA",
        "RUS": "RUS", "Russia": "RUS", "Russian Federation": "RUS", "USSR": "RUS", "Soviet Union": "RUS",
        "CHN": "CHN", "China": "CHN", "People's Republic of China": "CHN",
        "GBR": "GBR", "UK": "GBR", "United Kingdom": "GBR", "Britain": "GBR",
        "FRA": "FRA", "France": "FRA",
        "DEU": "DEU", "Germany": "DEU", "GER": "DEU",
        "JPN": "JPN", "Japan": "JPN",
        "IND": "IND", "India": "IND",
        "BRA": "BRA", "Brazil": "BRA",
        "ZAF": "ZAF", "South Africa": "ZAF",
        "MEX": "MEX", "Mexico": "MEX",
        "EGY": "EGY", "Egypt": "EGY",
        "NGA": "NGA", "Nigeria": "NGA",
        "IRN": "IRN", "Iran": "IRN",
        "KOR": "KOR", "South Korea": "KOR", "Republic of Korea": "KOR",
        "PRK": "PRK", "North Korea": "PRK", "DPRK": "PRK",
        "PAK": "PAK", "Pakistan": "PAK",
        "ISR": "ISR", "Israel": "ISR",
        "SAU": "SAU", "Saudi Arabia": "SAU",
        "TUR": "TUR", "Turkey": "TUR", "Türkiye": "TUR",
        "AUS": "AUS", "Australia": "AUS",
        "CAN": "CAN", "Canada": "CAN",
        "NZL": "NZL", "New Zealand": "NZL",
        "NLD": "NLD", "Netherlands": "NLD",
        "BEL": "BEL", "Belgium": "BEL",
        "NOR": "NOR", "Norway": "NOR",
        "SWE": "SWE", "Sweden": "SWE",
        "DNK": "DNK", "Denmark": "DNK",
        "FIN": "FIN", "Finland": "FIN",
        "POL": "POL", "Poland": "POL",
        "ITA": "ITA", "Italy": "ITA",
        "ESP": "ESP", "Spain": "ESP",
        "PRT": "PRT", "Portugal": "PRT",
        "GRC": "GRC", "Greece": "GRC",
        "IRL": "IRL", "Ireland": "IRL",
        "AUT": "AUT", "Austria": "AUT",
        "CHE": "CHE", "Switzerland": "CHE",
        "UKR": "UKR", "Ukraine": "UKR",
        "IDN": "IDN", "Indonesia": "IDN",
        "THA": "THA", "Thailand": "THA",
        "VNM": "VNM", "Vietnam": "VNM",
        "MYS": "MYS", "Malaysia": "MYS",
        "PHL": "PHL", "Philippines": "PHL",
        "ARG": "ARG", "Argentina": "ARG",
        "COL": "COL", "Colombia": "COL",
        "CHL": "CHL", "Chile": "CHL",
        "PER": "PER", "Peru": "PER",
        "VEN": "VEN", "Venezuela": "VEN",
    }
    if raw in _COMMON:
        return _COMMON[raw]
    # If 3 uppercase letters, assume it's already ISO3
    if len(raw) == 3 and raw.isupper():
        return raw
    return None


def _iso3_to_name(iso3: str) -> str:
    _MAP = {
        "USA": "United States", "RUS": "Russia", "CHN": "China", "GBR": "United Kingdom",
        "FRA": "France", "DEU": "Germany", "JPN": "Japan", "IND": "India", "BRA": "Brazil",
        "ZAF": "South Africa", "MEX": "Mexico", "EGY": "Egypt", "NGA": "Nigeria",
        "IRN": "Iran", "KOR": "South Korea", "PRK": "North Korea", "PAK": "Pakistan",
        "ISR": "Israel", "SAU": "Saudi Arabia", "TUR": "Turkey", "AUS": "Australia",
        "CAN": "Canada", "NZL": "New Zealand", "NLD": "Netherlands", "BEL": "Belgium",
        "NOR": "Norway", "SWE": "Sweden", "DNK": "Denmark", "FIN": "Finland",
        "POL": "Poland", "ITA": "Italy", "ESP": "Spain", "IRL": "Ireland",
        "AUT": "Austria", "CHE": "Switzerland", "UKR": "Ukraine", "IDN": "Indonesia",
        "THA": "Thailand", "VNM": "Vietnam", "MYS": "Malaysia", "PHL": "Philippines",
        "ARG": "Argentina", "COL": "Colombia", "CHL": "Chile", "PER": "Peru",
        "VEN": "Venezuela", "ETH": "Ethiopia", "KEN": "Kenya", "TZA": "Tanzania",
        "GHA": "Ghana", "SEN": "Senegal", "CMR": "Cameroon", "CIV": "Côte d'Ivoire",
        "TUN": "Tunisia", "MAR": "Morocco", "DZA": "Algeria", "LBY": "Libya",
        "IRQ": "Iraq", "SYR": "Syria", "JOR": "Jordan", "LBN": "Lebanon",
        "ARE": "United Arab Emirates", "QAT": "Qatar", "KWT": "Kuwait",
        "SGP": "Singapore", "MMR": "Myanmar", "BGD": "Bangladesh", "LKA": "Sri Lanka",
        "AGO": "Angola", "MOZ": "Mozambique", "ZMB": "Zambia", "ZWE": "Zimbabwe",
        "HUN": "Hungary", "CZE": "Czech Republic", "SVK": "Slovakia", "ROU": "Romania",
        "BGR": "Bulgaria", "HRV": "Croatia", "SRB": "Serbia", "UZB": "Uzbekistan",
        "KAZ": "Kazakhstan", "GEO": "Georgia", "ARM": "Armenia", "AZE": "Azerbaijan",
        "BOL": "Bolivia", "PRY": "Paraguay", "URY": "Uruguay", "ECU": "Ecuador",
        "GTM": "Guatemala", "HND": "Honduras", "SLV": "El Salvador", "NIC": "Nicaragua",
        "CRI": "Costa Rica", "PAN": "Panama", "CUB": "Cuba", "DOM": "Dominican Republic",
        "HTI": "Haiti", "JAM": "Jamaica", "TTO": "Trinidad and Tobago",
    }
    return _MAP.get(iso3, iso3)
