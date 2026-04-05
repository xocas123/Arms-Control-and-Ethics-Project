# groups.py
"""
Static country groupings, treaty participation data, and helper functions
for arms control / disarmament / nuclear politics NLP & analysis pipeline.
Data frozen as of March 2026 (UNTC + Wikipedia). Update treaties annually.
"""

from typing import Optional, Literal, Dict, List, Set

# ──────────────────────────────────────────────────────────────────────────────
# Nuclear weapon / status groupings (ISO3 codes)
# ──────────────────────────────────────────────────────────────────────────────

NWS: Set[str] = {"USA", "RUS", "GBR", "FRA", "CHN"}  # Official NPT NWS

DE_FACTO_NUCLEAR: Set[str] = {"IND", "PAK", "ISR", "PRK"}  # Undeclared

# Year from which each de facto state is classified as DE_FACTO (first credible test/weapon).
# Before this year the country is treated as NNWS.
# Sources: Arms Control Association, SIPRI, Bulletin of the Atomic Scientists.
DE_FACTO_NUCLEAR_SINCE: Dict[str, int] = {
    "ISR": 1967,  # widely accepted estimate; never tested, always ambiguous — using 1967
    "IND": 1974,  # Pokhran-I (Smiling Buddha) — first test May 1974
    "PAK": 1998,  # Chagai-I — first test May 1998
    "PRK": 2006,  # first nuclear test October 2006
}

# Expanded: all non-NWS NATO members + major extended-deterrence allies (AUS, JPN, KOR)
# Sources: IISS, Bulletin of the Atomic Scientists, CSIS reports 2025-2026
NUCLEAR_UMBRELLA: Set[str] = (
    {
        "ALB", "BEL", "BGR", "CAN", "HRV", "CZE", "DNK", "EST", "FIN", "DEU",
        "GRC", "HUN", "ISL", "ITA", "LVA", "LTU", "LUX", "MNE", "NLD", "MKD",
        "NOR", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "TUR"
    }  # NATO minus NWS
    | {"AUS", "JPN", "KOR"}
)

# Year each non-NWS NATO member joined (determines UMBRELLA classification onset).
# Pre-accession these countries were NNWS.  Original 12 members (1949) and early
# additions are set to 1949 (before UNGDC starts in 1970, so always UMBRELLA).
# AUS/JPN/KOR included: US Extended Deterrence in force continuously since 1950s.
NATO_ACCESSION_YEAR: Dict[str, int] = {
    # Founding + pre-1970 members — all before corpus starts
    "BEL": 1949, "CAN": 1949, "DNK": 1949, "FRA": 1949, "ISL": 1949,
    "ITA": 1949, "LUX": 1949, "NLD": 1949, "NOR": 1949, "PRT": 1949,
    "GBR": 1949, "USA": 1949,
    "GRC": 1952, "TUR": 1952,
    "DEU": 1955,
    "ESP": 1982,
    # Post-Cold War enlargements — actually matter for 1970-2023 corpus
    "HUN": 1999, "POL": 1999, "CZE": 1999,
    "BGR": 2004, "EST": 2004, "LVA": 2004, "LTU": 2004,
    "ROU": 2004, "SVK": 2004, "SVN": 2004,
    "ALB": 2009, "HRV": 2009,
    "MNE": 2017,
    "MKD": 2020,
    "FIN": 2023,
    "SWE": 2024,
    # Extended deterrence allies (not NATO but under US nuclear umbrella)
    "AUS": 1949, "JPN": 1949, "KOR": 1949,
}

NATO: Set[str] = {  # 32 members (Sweden 2024 — no 2025/26 changes)
    "ALB", "BEL", "BGR", "CAN", "HRV", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "ISL", "ITA", "LVA", "LTU", "LUX", "MNE", "NLD",
    "MKD", "NOR", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "TUR",
    "GBR", "USA"
}

EU: Set[str] = {  # 27 members (post-Brexit, confirmed 2026)
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
}

NAM: Set[str] = {  # Non-Aligned Movement — 121 members (full list 2026)
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD",
    "COM", "COD", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB",
    "GIN", "GNB", "CIV", "KEN", "LSO", "LBR", "LBY", "MDG", "MWI", "MLI",
    "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "COG", "RWA", "STP",
    "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN",
    "UGA", "ZMB", "ZWE",
    "ATG", "BHS", "BRB", "BLZ", "BOL", "CHL", "COL", "CUB", "DMA", "DOM",
    "ECU", "GRD", "GTM", "GUY", "HTI", "HND", "JAM", "NIC", "PAN", "PER",
    "KNA", "LCA", "VCT", "SUR", "TTO", "VEN",
    "AFG", "BHR", "BGD", "BTN", "BRN", "KHM", "IND", "IDN", "IRN", "IRQ",
    "JOR", "KWT", "LAO", "LBN", "MYS", "MDV", "MNG", "MMR", "NPL", "PRK",
    "OMN", "PAK", "PSE", "PHL", "QAT", "SAU", "SGP", "LKA", "SYR", "THA",
    "TLS", "TKM", "ARE", "UZB", "VNM", "YEM",
    "AZE", "BLR",
    "FJI", "PNG", "VUT"
}

NAC: Set[str] = {"BRA", "EGY", "IRL", "MEX", "NZL", "ZAF"}  # New Agenda Coalition (Sweden & Slovenia left)

# ──────────────────────────────────────────────────────────────────────────────
# Geographic macro-regions (expanded)
# ──────────────────────────────────────────────────────────────────────────────

REGIONS: Dict[str, Set[str]] = {
    "Latin America & Caribbean": {
        "ARG", "BHS", "BRB", "BLZ", "BOL", "BRA", "CHL", "COL", "CRI", "CUB",
        "DMA", "DOM", "ECU", "SLV", "GRD", "GTM", "GUY", "HTI", "HND", "JAM",
        "MEX", "NIC", "PAN", "PRY", "PER", "KNA", "LCA", "VCT", "SUR", "TTO",
        "URY", "VEN", "ATG",
    },
    "MENA": {
        "DZA", "BHR", "EGY", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN", "LBY",
        "MAR", "OMN", "PSE", "QAT", "SAU", "SYR", "TUN", "ARE", "YEM", "MLT",
        "DJI",
    },
    "Sub-Saharan Africa": {
        "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM",
        "COD", "COG", "CIV", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA",
        "GIN", "GNB", "KEN", "LSO", "LBR", "MDG", "MWI", "MLI", "MRT", "MUS",
        "MOZ", "NAM", "NER", "NGA", "RWA", "STP", "SEN", "SYC", "SLE", "SOM",
        "ZAF", "SSD", "SDN", "TZA", "TGO", "UGA", "ZMB", "ZWE",
    },
    "South Asia": {
        "AFG", "BGD", "BTN", "IND", "MDV", "NPL", "PAK", "LKA",
    },
    "East Asia & Pacific": {
        "AUS", "BRN", "KHM", "CHN", "FJI", "IDN", "JPN", "KIR", "LAO", "MYS",
        "MHL", "FSM", "MNG", "MMR", "NRU", "NZL", "PRK", "KOR", "PLW", "PNG",
        "PHL", "WSM", "SGP", "SLB", "THA", "TLS", "TON", "TUV", "VUT", "VNM",
    },
    "North America": {"CAN", "MEX", "USA"},
    "Western Europe": {
        "AUT", "BEL", "DNK", "FIN", "FRA", "DEU", "GRC", "ISL", "IRL", "ITA",
        "LUX", "NLD", "NOR", "PRT", "ESP", "SWE", "CHE", "GBR", "CYP", "LIE",
        "MCO", "SMR", "AND",
    },
    "Eastern Europe": {
        "ALB", "ARM", "AZE", "BLR", "BIH", "BGR", "HRV", "CZE", "EST", "GEO",
        "HUN", "LVA", "LTU", "MDA", "MKD", "MNE", "POL", "ROU", "RUS", "SRB",
        "SVK", "SVN", "UKR",
    },
    "Central Asia": {"KAZ", "KGZ", "TJK", "TKM", "UZB", "MNG"},
}

# ──────────────────────────────────────────────────────────────────────────────
# Treaty participation (UN Treaty Collection — as of 13 Mar 2026)
# ──────────────────────────────────────────────────────────────────────────────

ATT_PARTIES: Dict[str, int] = {  # Arms Trade Treaty — FULL
    "AFG": 2020, "ALB": 2014, "AND": 2022, "ATG": 2013, "ARG": 2014, "AUS": 2014,
    "AUT": 2014, "BHS": 2014, "BRB": 2015, "BEL": 2014, "BLZ": 2015, "BEN": 2016,
    "BIH": 2014, "BWA": 2019, "BRA": 2018, "BGR": 2014, "BFA": 2014, "CPV": 2016,
    "CMR": 2018, "CAN": 2019, "CAF": 2015, "TCD": 2015, "CHL": 2018, "CHN": 2020,
    "COL": 2024, "CRI": 2013, "CIV": 2015, "HRV": 2014, "CYP": 2016, "CZE": 2014,
    "DNK": 2014, "DMA": 2015, "DOM": 2014, "SLV": 2014, "EST": 2014, "FIN": 2014,
    "FRA": 2014, "GAB": 2022, "GMB": 2024, "GEO": 2016, "DEU": 2014, "GHA": 2015,
    "GRC": 2016, "GTM": 2016, "GIN": 2014, "GNB": 2018, "HND": 2017, "HUN": 2014,
    "ISL": 2013, "IRL": 2014, "ITA": 2014, "JAM": 2014, "JPN": 2014, "KAZ": 2017,
    "LVA": 2014, "LBN": 2019, "LSO": 2016, "LBR": 2015, "LIE": 2014, "LTU": 2014,
    "LUX": 2014, "MDG": 2016, "MWI": 2024, "MDV": 2019, "MLI": 2013, "MLT": 2014,
    "MRT": 2015, "MUS": 2015, "MDA": 2015, "MEX": 2013, "MCO": 2016, "MNE": 2014,
    "MOZ": 2018, "NAM": 2020, "NLD": 2014, "NZL": 2014, "NER": 2015, "NIU": 2020,
    "MKD": 2014, "NOR": 2014, "PLW": 2019, "PAN": 2014, "PRY": 2015, "PER": 2016,
    "PHL": 2022, "POL": 2014, "PRT": 2014, "ROU": 2014, "WSM": 2014, "SMR": 2015,
    "KNA": 2014, "LCA": 2014, "VCT": 2014, "STP": 2020, "SEN": 2014, "SRB": 2014,
    "SYC": 2015, "SLE": 2014, "SVK": 2014, "SVN": 2014, "ZAF": 2014, "KOR": 2016,
    "ESP": 2014, "PSE": 2017, "SUR": 2018, "SWE": 2014, "CHE": 2015, "TGO": 2015,
    "TTO": 2013, "TUV": 2015, "GBR": 2014, "URY": 2014, "ZMB": 2016,
    "ECU": 2026, "VUT": 2025  # 2025-2026 updates
}
ATT_SIGNATORIES_ONLY: Dict[str, int] = {
    iso: 2013
    for iso in {
        "BHR", "BGD", "AGO", "COM", "COG", "DJI", "SWZ", "KHM", "BDI", "GUY", "HTI",
        "ISR", "MYS", "KIR", "LBY", "NGA", "SGP", "THA", "TUR", "ARE", "TZA", "UKR", "ZWE",
        "USA", "RUS", "CHN", "GBR", "FRA"
    }
}  # Signed but not ratified (UNTC Mar 2026; approximate signature year)

TPNW_PARTIES: Dict[str, int] = {  # Treaty on the Prohibition of Nuclear Weapons
    # Expanded starter list (74+ parties as of Sept 2025 per Wikipedia/UNODA)
    # Full authoritative list: https://treaties.un.org (XXVI-9) or ICAN
    # Last major: Ghana ~2025. Update manually or via API in production.
    "ATG": 2019, "AUT": 2018, "BEN": 2022, "BOL": 2018, "BRB": 2020, "BWA": 2020,
    "BRN": 2020, "CPV": 2021, "CHL": 2022, "COM": 2022, "CRI": 2018, "CUB": 2018,
    "DMA": 2021, "DOM": 2022, "ECU": 2021, "FJI": 2018, "GHA": 2025, "GRD": 2020,
    "GTM": 2024, "GUY": 2021, "VAT": 2018, "HND": 2021, "IRL": 2024, "JAM": 2021,
    "KAZ": 2022, "KIR": 2020, "LAO": 2022, "LIE": 2020, "MYS": 2020, "MHL": 2020,
    "MUS": 2021, "MEX": 2018, "NAM": 2022, "NPL": 2020, "NIC": 2020, "NZL": 2018,
    "NGA": 2020, "NIU": 2020, "PLW": 2020, "PAN": 2020, "PRY": 2020, "PHL": 2022,
    "SAO": 2022, "KNA": 2021, "LCA": 2020, "VCT": 2020, "WSM": 2019, "SEN": 2022,
    "SYC": 2018, "SLE": 2021, "SLB": 2020, "SUR": 2021, "THA": 2022, "TLS": 2018,
    "TTO": 2019, "TUV": 2019, "URY": 2018, "VUT": 2020, "VEN": 2018, "ZAF": 2019,
    # ... ~40 more small states — add from UNTC when needed
}
TPNW_SIGNATORIES_ONLY: Set[str] = set()  # Very few; most signatories have ratified

OTTAWA_PARTIES: Dict[str, int] = {  # Mine Ban Treaty — FULL (withdrawals removed)
    # Source: UNTC XXVI-5 (Mar 2026). Estonia, Finland, Latvia, Lithuania, Poland withdrawn (effective 2025-2026)
    "AFG": 2002, "ALB": 2000, "DZA": 2001, "AND": 1998, "AGO": 2002, "ATG": 1999,
    "ARG": 1999, "AUS": 1999, "AUT": 1998, "BHS": 1998, "BGD": 2000, "BRB": 1999,
    "BLR": 2003, "BEL": 1998, "BLZ": 1998, "BEN": 1998, "BTN": 2005, "BOL": 1998,
    "BIH": 1998, "BWA": 2000, "BRA": 1999, "BRN": 2006, "BGR": 1998, "BFA": 1998,
    "BDI": 2003, "CPV": 2001, "KHM": 1999, "CMR": 2002, "CAN": 1997, "CAF": 2002,
    "TCD": 1999, "CHL": 2001, "COL": 2000, "COM": 2002, "COG": 2001, "COK": 2006,
    "CRI": 1999, "CIV": 2000, "HRV": 1998, "CYP": 2003, "CZE": 1999, "COD": 2002,
    "DNK": 1998, "DJI": 1998, "DMA": 1999, "DOM": 2000, "ECU": 1999, "SLV": 1999,
    "GNQ": 1998, "ERI": 2001, "SWZ": 1998, "ETH": 2004, "FJI": 1998, "GAB": 2000,
    "GMB": 2002, "DEU": 1998, "GHA": 2000, "GRC": 2003, "GRD": 1998, "GTM": 1999,
    "GIN": 1998, "GNB": 2001, "GUY": 2003, "HTI": 2006, "VAT": 1998, "HND": 1998,
    "HUN": 1998, "ISL": 1999, "IDN": 2007, "IRQ": 2007, "IRL": 1997, "ITA": 1999,
    "JAM": 1998, "JPN": 1998, "JOR": 1998, "KEN": 2001, "KIR": 2000, "KWT": 2007,
    "LSO": 1998, "LBR": 1999, "LIE": 1999, "LUX": 1999, "MDG": 1999, "MWI": 1998,
    "MYS": 1999, "MDV": 2000, "MLI": 1998, "MLT": 2001, "MHL": 2025, "MRT": 2000,
    "MEX": 1998, "MCO": 1998, "MNE": 1998, "MNG": 1998, "MOZ": 1998, "NAM": 1998,
    "NRU": 2000, "NLD": 1999, "NZL": 1999, "NIC": 1998, "NER": 1999, "NGA": 2001,
    "NIU": 1998, "MKD": 1998, "NOR": 1998, "OMN": 2014, "PLW": 2007, "PAN": 1998,
    "PNG": 2004, "PRY": 1998, "PER": 1998, "PHL": 2000, "PRT": 1999, "QAT": 1998,
    "MDA": 2000, "ROU": 2000, "RWA": 2000, "WSM": 1998, "SMR": 1998, "STP": 2003,
    "SEN": 2003, "SYC": 2000, "SLE": 2001, "SVK": 1999, "SVN": 1998, "SLB": 1999,
    "SOM": 2012, "ZAF": 1998, "SSD": 2011, "ESP": 1999, "LKA": 2017, "KNA": 1998,
    "LCA": 1999, "VCT": 2001, "PSE": 2017, "SDN": 2003, "SUR": 2002, "SWE": 1998,
    "CHE": 1998, "TJK": 1999, "THA": 1998, "TLS": 2003, "TGO": 2000, "TON": 2025,
    "TTO": 1998, "TUN": 1999, "TUR": 2003, "TKM": 1998, "TUV": 2011, "UGA": 1999,
    "UKR": 2005, "TZA": 2000, "URY": 2001, "VUT": 2005, "VEN": 1999, "YEM": 1998,
    "ZMB": 2001, "ZWE": 1998
}

CCM_PARTIES: Dict[str, int] = {  # Cluster Munitions — FULL
    # Source: UNTC XXVI-6 (Mar 2026). Lithuania withdrawn Mar 2025; Vanuatu 2025
    "AFG": 2011, "ALB": 2009, "AND": 2013, "ATG": 2010, "AUS": 2012, "AUT": 2009,
    "BEL": 2009, "BLZ": 2014, "BEN": 2017, "BOL": 2013, "BIH": 2010, "BWA": 2011,
    "BGR": 2011, "BFA": 2010, "BDI": 2009, "CPV": 2010, "CMR": 2012, "CAN": 2015,
    "TCD": 2013, "CHL": 2010, "COL": 2015, "COM": 2010, "COG": 2014, "COK": 2011,
    "CRI": 2011, "CIV": 2012, "HRV": 2009, "CUB": 2016, "CZE": 2011, "DNK": 2010,
    "DOM": 2011, "ECU": 2010, "SLV": 2011, "SWZ": 2011, "FJI": 2010, "FRA": 2009,
    "GMB": 2018, "DEU": 2009, "GHA": 2011, "GRD": 2011, "GTM": 2010, "GIN": 2014,
    "GNB": 2010, "GUY": 2014, "HTI": 2009, "VAT": 2008, "HND": 2012, "HUN": 2012,
    "ISL": 2015, "IRQ": 2013, "IRL": 2008, "ITA": 2011, "JAM": 2009, "JPN": 2009,
    "LAO": 2009, "LBN": 2010, "LSO": 2010, "LIE": 2013, "LUX": 2009, "MDG": 2017,
    "MWI": 2009, "MDV": 2019, "MLI": 2010, "MLT": 2009, "MRT": 2012, "MUS": 2015,
    "MEX": 2009, "MDA": 2010, "MCO": 2010, "MNE": 2010, "MOZ": 2011, "NAM": 2018,
    "NRU": 2013, "NLD": 2011, "NZL": 2009, "NIC": 2009, "NGA": 2023, "NIU": 2020,
    "NER": 2009, "NOR": 2008, "PLW": 2016, "PAN": 2010, "PRY": 2015, "PER": 2012,
    "PHL": 2019, "PRT": 2011, "RWA": 2015, "WSM": 2010, "SMR": 2009, "STP": 2020,
    "SEN": 2011, "SYC": 2010, "SLE": 2008, "SVK": 2015, "SVN": 2009, "SOM": 2015,
    "ZAF": 2015, "SSD": 2023, "ESP": 2009, "LKA": 2018, "KNA": 2013, "LCA": 2020,
    "VCT": 2010, "PSE": 2015, "SWE": 2012, "CHE": 2012, "TGO": 2012, "TTO": 2011,
    "TUN": 2010, "URY": 2009, "VUT": 2025, "ZMB": 2009
}

# ──────────────────────────────────────────────────────────────────────────────
# Country name ↔ ISO3 mappings (expanded)
# ──────────────────────────────────────────────────────────────────────────────

ISO3_TO_NAME: Dict[str, str] = {
    "USA": "United States", "RUS": "Russia", "GBR": "United Kingdom",
    "FRA": "France", "CHN": "China", "IND": "India", "PAK": "Pakistan",
    "ISR": "Israel", "PRK": "North Korea", "KOR": "South Korea",
    "JPN": "Japan", "DEU": "Germany", "ITA": "Italy", "BRA": "Brazil",
    "ZAF": "South Africa", "AUS": "Australia", "CAN": "Canada",
    "MEX": "Mexico", "EGY": "Egypt", "IRL": "Ireland", "NZL": "New Zealand",
    # ... add more as needed
}

NAME_TO_ISO3: Dict[str, str] = {v: k for k, v in ISO3_TO_NAME.items()}
NAME_TO_ISO3.update({  # Expanded messy names
    "United States of America": "USA", "U.S.A.": "USA", "America": "USA",
    "Russian Federation": "RUS", "UK": "GBR", "Great Britain": "GBR",
    "South Korea": "KOR", "Republic of Korea": "KOR",
    "North Korea": "PRK", "DPRK": "PRK", "Democratic People's Republic of Korea": "PRK",
})

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions (unchanged logic — now use expanded data)
# ──────────────────────────────────────────────────────────────────────────────

def get_nuclear_status(iso3: str, year: Optional[int] = None) -> Literal["NWS", "DE_FACTO", "UMBRELLA", "NNWS"]:
    """
    Return nuclear status for a country, optionally at a specific year.

    Without a year: returns current (static) status.
    With a year: accounts for when countries actually acquired nuclear weapons
    or joined NATO/umbrella arrangements, so pre-acquisition years correctly
    return "NNWS" rather than "DE_FACTO" or "UMBRELLA".
    """
    iso3 = iso3.upper().strip()

    # NPT P5 have been NWS since the NPT was drafted in 1968 — static
    if iso3 in NWS:
        return "NWS"

    # De facto states: only classified as DE_FACTO from the year of first credible test/weapon
    if iso3 in DE_FACTO_NUCLEAR:
        if year is None or year >= DE_FACTO_NUCLEAR_SINCE.get(iso3, 0):
            return "DE_FACTO"
        # Before acquisition year → treat as NNWS

    # Umbrella: only from NATO accession year (or US alliance year for AUS/JPN/KOR)
    if iso3 in NUCLEAR_UMBRELLA:
        if year is None or year >= NATO_ACCESSION_YEAR.get(iso3, 0):
            return "UMBRELLA"
        # Before accession → treat as NNWS

    return "NNWS"

def get_region(iso3: str) -> Optional[str]:
    iso3 = iso3.upper().strip()
    for region_name, members in REGIONS.items():
        if iso3 in members:
            return region_name
    return None

# Must be defined before get_treaty_status which references it
TPNW_OPPONENTS: Set[str] = NWS | NUCLEAR_UMBRELLA

# get_treaty_status, get_years_since_ratification, standardize_country,
# get_binary_regime, get_regime_type — unchanged (just point to the new dicts)

def get_treaty_status(iso3: str, treaty: Literal["ATT", "TPNW", "OTTAWA", "CCM"], year: int) -> Literal["party", "signatory", "non-member", "opponent"]:
    """Return treaty membership status for a given country and year."""
    iso3 = iso3.upper().strip()

    if treaty == "ATT":
        parties = ATT_PARTIES
        signatories = ATT_SIGNATORIES_ONLY
        opponents = NWS
    elif treaty == "TPNW":
        parties = TPNW_PARTIES
        signatories = TPNW_SIGNATORIES_ONLY
        opponents = TPNW_OPPONENTS
    elif treaty == "OTTAWA":
        parties = OTTAWA_PARTIES
        signatories = set()
        opponents = set()
    elif treaty == "CCM":
        parties = CCM_PARTIES
        signatories = set()
        opponents = set()
    else:
        raise ValueError(f"Unknown treaty: {treaty}")

    # Party (ratifier) status depends on whether the country had joined by the given year.
    if iso3 in parties:
        rat_year = parties[iso3]
        if year >= rat_year:
            return "party"
        return "non-member"

    # Signatory status is not keyed by year in these dicts.
    if iso3 in signatories:
        return "signatory"

    if iso3 in opponents:
        return "opponent"

    return "non-member"

def get_years_since_ratification(
    iso3: str,
    treaty: Literal["ATT", "TPNW", "OTTAWA", "CCM"],
    year: int,
) -> Optional[int]:
    """Return the number of years since ratification, or None if not a party."""
    iso3 = iso3.upper().strip()
    parties = {
        "ATT": ATT_PARTIES,
        "TPNW": TPNW_PARTIES,
        "OTTAWA": OTTAWA_PARTIES,
        "CCM": CCM_PARTIES,
    }.get(treaty, {})

    rat_year = parties.get(iso3)
    if rat_year is None or year < rat_year:
        return None
    return year - rat_year


def standardize_country(country: str) -> Optional[str]:
    """Normalize a country name (or code) to an ISO3 code."""
    if not country or not isinstance(country, str):
        return None

    key = country.strip()
    if not key:
        return None

    if key in NAME_TO_ISO3:
        return NAME_TO_ISO3[key]

    if len(key) == 3 and key.isalpha():
        return key.upper()

    return None


def get_binary_regime(country_iso3: str, year: int) -> str:
    """Fallback binary regime classification for when V-Dem data is unavailable."""
    iso3 = country_iso3.upper().strip()

    democracies = {
        "USA", "CAN", "GBR", "FRA", "DEU", "ITA", "JPN", "AUS", "NZL",
        "NLD", "SWE", "NOR", "FIN", "DNK", "IRL", "CHE", "BEL", "ESP",
        "PRT", "AUT", "ISL",
    }
    autocracies = {
        "CHN", "RUS", "PRK", "IRN", "SYR", "SAU", "CUB", "BLR", "VEN",
        "TKM", "UZB", "BHR", "QAT", "OMN",
    }

    if iso3 in democracies:
        return "democracy"
    if iso3 in autocracies:
        return "autocracy"
    return "unknown"


def get_regime_type(country_iso3: str, year: int) -> str:
    """Fallback 4-way regime classification (V-Dem style) when no V-Dem data is present."""
    iso3 = country_iso3.upper().strip()

    liberal_democracies = {
        "USA", "CAN", "GBR", "FRA", "DEU", "ITA", "JPN", "AUS", "NZL",
        "NLD", "SWE", "NOR", "FIN", "DNK", "IRL", "CHE", "BEL", "ESP",
        "PRT", "AUT", "ISL",
    }
    closed_autocracies = {
        "CHN", "RUS", "PRK", "IRN", "SYR", "SAU", "CUB", "BLR", "VEN",
        "TKM", "UZB", "BHR", "QAT", "OMN",
    }

    if iso3 in liberal_democracies:
        return "liberal_democracy"
    if iso3 in closed_autocracies:
        return "closed_autocracy"
    return "unknown"


def get_nnws() -> Set[str]:
    """Return an approximate set of NNWS (non-nuclear weapon states).

    This is derived from the union of countries present in the treaty participation
    data, excluding known nuclear-armed states and their extended deterrence allies.
    """
    all_countries = set(ISO3_TO_NAME.keys())
    all_countries |= set(ATT_PARTIES) | set(ATT_SIGNATORIES_ONLY) | set(TPNW_PARTIES) | set(TPNW_SIGNATORIES_ONLY) | set(OTTAWA_PARTIES) | set(CCM_PARTIES)
    return all_countries - NWS - DE_FACTO_NUCLEAR - NUCLEAR_UMBRELLA
