"""
Arms control treaty scraper.

Collects full treaty texts, metadata, and ratification/party data from
authoritative web sources:
  - ICRC IHL Database (ihl-databases.icrc.org) — primary text source
  - UNODA (disarmament.unoda.org) — metadata + PDF fallback
  - UN Treaty Collection (treaties.un.org) — ratification status
  - Direct PDF URLs — NWFZs, CTBT, TTBT
  - Arms Control Association (armscontrol.org) — metadata cross-check

Outputs:
  data/raw/treaties/
    metadata/treaties_metadata.csv
    texts/{treaty_id}_full.txt
    ratifications/{treaty_id}_parties.csv
    .cache/html/{sha256}.html.gz
    .cache/pdf/{sha256}.pdf

Usage:
  python src/data/scrape_treaties.py --treaties all
  python src/data/scrape_treaties.py --treaties npt att cwc --rate-limit 2.0
  python src/data/scrape_treaties.py --treaties npt --dry-run
  python src/data/scrape_treaties.py --treaties tlatelolco --sources pdf_direct
"""

import argparse
import gzip
import hashlib
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional heavy deps — graceful import
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False

# Project utilities
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils import normalize_country, save_csv, clean_text, save_json  # noqa: E402

logger = logging.getLogger(__name__)

TODAY = date.today().isoformat()

# ---------------------------------------------------------------------------
# Treaty registry — master lookup driving all scraping
# ---------------------------------------------------------------------------

TREATY_REGISTRY: Dict[str, dict] = {
    "npt": {
        "name": "Treaty on the Non-Proliferation of Nuclear Weapons",
        "abbreviation": "NPT",
        "year_signed": 1968,
        "year_in_force": 1970,
        "category": "nuclear",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/npt/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/npt-1968",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-1&chapter=26",
        "text_url": "https://www.un.org/disarmament/wmd/nuclear/npt/text",
        "text_url_pdf": None,
        "aca_url": "https://www.armscontrol.org/factsheets/npt",
    },
    "tpnw": {
        "name": "Treaty on the Prohibition of Nuclear Weapons",
        "abbreviation": "TPNW",
        "year_signed": 2017,
        "year_in_force": 2021,
        "category": "nuclear",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/tpnw/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/tpnw-2017",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-9&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://treaties.un.org/doc/Treaties/2017/07/20170707%2003-42%20PM/Ch_XXVI_9.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/nuclear-ban-treaty",
    },
    "ctbt": {
        "name": "Comprehensive Nuclear-Test-Ban Treaty",
        "abbreviation": "CTBT",
        "year_signed": 1996,
        "year_in_force": None,
        "category": "nuclear",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/ctbt/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-4&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://www.ctbto.org/fileadmin/content/treaty/treaty_text.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/ctbt",
    },
    "ttbt": {
        "name": "Threshold Test Ban Treaty",
        "abbreviation": "TTBT",
        "year_signed": 1974,
        "year_in_force": 1990,
        "category": "nuclear",
        "type": "bilateral",
        "depositary": "USA/RUS",
        "unoda_url": None,
        "icrc_url": None,
        "un_treaty_url": None,
        "text_url": None,
        "text_url_pdf": "https://media.nti.org/pdfs/ttbt.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/ttbt",
    },
    "outer_space": {
        "name": "Treaty on Principles Governing the Activities of States in the Exploration and Use of Outer Space",
        "abbreviation": "Outer Space Treaty",
        "year_signed": 1967,
        "year_in_force": 1967,
        "category": "nuclear",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/space/outer-space-treaty/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=III-6&chapter=3",
        "text_url": "https://www.unoosa.org/oosa/en/ourwork/spacelaw/treaties/outerspacetreaty.html",
        "text_url_pdf": None,
        "aca_url": "https://www.armscontrol.org/factsheets/outerspace",
    },
    "cwc": {
        "name": "Chemical Weapons Convention",
        "abbreviation": "CWC",
        "year_signed": 1993,
        "year_in_force": 1997,
        "category": "chemical",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/wmd/chemical/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/cwc-1993",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-3&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://www.opcw.org/sites/default/files/documents/CWC/CWC_en.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/cwc",
    },
    "bwc": {
        "name": "Biological Weapons Convention",
        "abbreviation": "BWC",
        "year_signed": 1972,
        "year_in_force": 1975,
        "category": "biological",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/biological-weapons/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/bwc-1972",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-6&chapter=26",
        "text_url": None,
        "text_url_pdf": None,
        "aca_url": "https://www.armscontrol.org/factsheets/bwc",
    },
    "att": {
        "name": "Arms Trade Treaty",
        "abbreviation": "ATT",
        "year_signed": 2013,
        "year_in_force": 2014,
        "category": "conventional",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/conventional-arms/att/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/att-2013",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-8&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://thearmstradetreaty.org/hyper-images/file/ATT_English/ATT_English.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/arms-trade-treaty",
    },
    "ottawa": {
        "name": "Convention on the Prohibition of the Use, Stockpiling, Production and Transfer of Anti-Personnel Mines",
        "abbreviation": "Ottawa Treaty / APM Convention",
        "year_signed": 1997,
        "year_in_force": 1999,
        "category": "conventional",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/conventional-arms/antipersonnel-landmines/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/apmbc-1997",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-5&chapter=26",
        "text_url": None,
        "text_url_pdf": None,
        "aca_url": "https://www.armscontrol.org/factsheets/otttawa",
    },
    "ccm": {
        "name": "Convention on Cluster Munitions",
        "abbreviation": "CCM",
        "year_signed": 2008,
        "year_in_force": 2010,
        "category": "conventional",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/conventional-arms/cluster-munitions/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/ccm-2008",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-6&chapter=26",
        "text_url": None,
        "text_url_pdf": None,
        "aca_url": "https://www.armscontrol.org/factsheets/cluster-munitions",
    },
    "ccw": {
        "name": "Convention on Certain Conventional Weapons",
        "abbreviation": "CCW",
        "year_signed": 1980,
        "year_in_force": 1983,
        "category": "conventional",
        "type": "multilateral",
        "depositary": "UN",
        "unoda_url": "https://disarmament.unoda.org/conventional-arms/ccw/",
        "icrc_url": "https://ihl-databases.icrc.org/en/ihl-treaties/ccwc-1980",
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-2&chapter=26",
        "text_url": None,
        "text_url_pdf": None,
        "aca_url": "https://www.armscontrol.org/factsheets/ccw",
    },
    "tlatelolco": {
        "name": "Treaty for the Prohibition of Nuclear Weapons in Latin America and the Caribbean",
        "abbreviation": "Treaty of Tlatelolco",
        "year_signed": 1967,
        "year_in_force": 1968,
        "category": "nwfz",
        "type": "multilateral",
        "depositary": "OPANAL",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/nwfz/tlatelolco/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-4&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://opanal.org/wp-content/uploads/2020/05/Treaty-of-Tlatelolco-english.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/tlatelolco",
    },
    "rarotonga": {
        "name": "South Pacific Nuclear Free Zone Treaty",
        "abbreviation": "Treaty of Rarotonga",
        "year_signed": 1985,
        "year_in_force": 1986,
        "category": "nwfz",
        "type": "multilateral",
        "depositary": "Pacific Islands Forum",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/nwfz/rarotonga/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-5&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://www.iaea.org/sites/default/files/publications/documents/infcircs/1987/infcirc331.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/rarotonga",
    },
    "bangkok": {
        "name": "Treaty on the Southeast Asia Nuclear Weapon-Free Zone",
        "abbreviation": "Bangkok Treaty",
        "year_signed": 1995,
        "year_in_force": 1997,
        "category": "nwfz",
        "type": "multilateral",
        "depositary": "ASEAN",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/nwfz/bangkok/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-7&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://www.iaea.org/sites/default/files/publications/documents/infcircs/1996/infcirc519.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/bangkok",
    },
    "pelindaba": {
        "name": "African Nuclear-Weapon-Free Zone Treaty",
        "abbreviation": "Treaty of Pelindaba",
        "year_signed": 1996,
        "year_in_force": 2009,
        "category": "nwfz",
        "type": "multilateral",
        "depositary": "AU",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/nwfz/pelindaba/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-9&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://www.iaea.org/sites/default/files/publications/documents/infcircs/1996/infcirc525.pdf",
        "aca_url": "https://www.armscontrol.org/factsheets/pelindaba",
    },
    "semipalatinsk": {
        "name": "Treaty on a Nuclear-Weapon-Free Zone in Central Asia",
        "abbreviation": "Semipalatinsk Treaty",
        "year_signed": 2006,
        "year_in_force": 2009,
        "category": "nwfz",
        "type": "multilateral",
        "depositary": "Kyrgyzstan",
        "unoda_url": "https://disarmament.unoda.org/wmd/nuclear/nwfz/canwfz/",
        "icrc_url": None,
        "un_treaty_url": "https://treaties.un.org/pages/ViewDetails.aspx?src=TREATY&mtdsg_no=XXVI-10&chapter=26",
        "text_url": None,
        "text_url_pdf": "https://disarmament.unoda.org/wmd/nuclear/nwfz/canwfz/",
        "aca_url": "https://www.armscontrol.org/factsheets/semipalatinsk",
    },
}

ALL_TREATY_IDS = list(TREATY_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Data class for scrape results
# ---------------------------------------------------------------------------

@dataclass
class TreatyScrapeResult:
    treaty_id: str
    success: bool
    text: str = ""
    metadata: dict = field(default_factory=dict)
    parties_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str = ""
    sources_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 1: HTTP client with caching
# ---------------------------------------------------------------------------

class TreatyHTTPClient:
    """HTTP client with URL-keyed disk cache and rate limiting."""

    USER_AGENT = (
        "arms-control-research-bot/1.0 "
        "(academic research; contact: research@example.org)"
    )

    def __init__(
        self,
        cache_dir: Path,
        rate_limit: float = 1.5,
        use_cache: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.rate_limit = rate_limit
        self.use_cache = use_cache
        self._last_request_time: float = 0.0

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

        if use_cache:
            (self.cache_dir / "html").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "pdf").mkdir(parents=True, exist_ok=True)

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        delay = random.uniform(self.rate_limit, self.rate_limit * 1.5)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.monotonic()

    def _load_html_cache(self, key: str) -> Optional[bytes]:
        path = self.cache_dir / "html" / f"{key}.html.gz"
        if self.use_cache and path.exists():
            return gzip.decompress(path.read_bytes())
        return None

    def _save_html_cache(self, key: str, data: bytes) -> None:
        if not self.use_cache:
            return
        path = self.cache_dir / "html" / f"{key}.html.gz"
        path.write_bytes(gzip.compress(data))

    def _load_pdf_cache(self, key: str) -> Optional[bytes]:
        path = self.cache_dir / "pdf" / f"{key}.pdf"
        if self.use_cache and path.exists():
            return path.read_bytes()
        return None

    def _save_pdf_cache(self, key: str, data: bytes) -> None:
        if not self.use_cache:
            return
        path = self.cache_dir / "pdf" / f"{key}.pdf"
        path.write_bytes(data)

    def get_html(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL and return BeautifulSoup, using cache when available."""
        key = self._cache_key(url)
        cached = self._load_html_cache(key)
        if cached is not None:
            logger.debug("Cache hit (html): %s", url)
            return BeautifulSoup(cached, "lxml")

        self._throttle()
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            self._save_html_cache(key, resp.content)
            return BeautifulSoup(resp.content, "lxml")
        except Exception as exc:
            logger.warning("Failed to fetch HTML %s: %s", url, exc)
            return None

    def get_pdf_bytes(self, url: str) -> Optional[bytes]:
        """Fetch PDF URL and return raw bytes, using cache when available."""
        key = self._cache_key(url)
        cached = self._load_pdf_cache(key)
        if cached is not None:
            logger.debug("Cache hit (pdf): %s", url)
            return cached

        self._throttle()
        try:
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()
            self._save_pdf_cache(key, resp.content)
            return resp.content
        except Exception as exc:
            logger.warning("Failed to fetch PDF %s: %s", url, exc)
            return None

    def get_text_trafilatura(self, url: str) -> Optional[str]:
        """Fetch URL and extract main text via trafilatura (no cache)."""
        if not _TRAFILATURA_AVAILABLE:
            logger.warning("trafilatura not installed; skipping text extraction for %s", url)
            return None
        self._throttle()
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                return trafilatura.extract(downloaded)
        except Exception as exc:
            logger.warning("trafilatura extraction failed %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Layer 2: PDF text extractor
# ---------------------------------------------------------------------------

class PDFExtractor:
    """Extract and clean text from PDF bytes using pdfplumber."""

    # Patterns to strip from UN/treaty PDFs
    _NOISE_PATTERNS = [
        r"^\s*\d+\s*$",                          # lone page numbers
        r"Distr\.\s*:\s*General",
        r"Official\s+Records",
        r"A/RES/\d+/\d+",
        r"Check\s+against\s+delivery",
        r"^\s*-\s*\d+\s*-\s*$",                 # page numbers with dashes
    ]
    _NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE | re.MULTILINE)

    def extract_text(self, pdf_bytes: bytes) -> str:
        if not _PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not installed; cannot extract PDF text")
            return ""
        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
            raw = "\n".join(pages)
            return self._clean_pdf_text(raw)
        except Exception as exc:
            logger.warning("PDF extraction failed: %s", exc)
            return ""

    def _clean_pdf_text(self, raw: str) -> str:
        lines = raw.splitlines()
        cleaned = []
        for line in lines:
            if self._NOISE_RE.search(line):
                continue
            cleaned.append(line)
        text = "\n".join(cleaned)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Layer 3: Source-specific scrapers
# ---------------------------------------------------------------------------

class ICRCScraper:
    """
    Scrapes treaty texts from the ICRC IHL Database.
    The site is Next.js server-rendered; CSS module class names use hashed
    suffixes, so we match on substring: [class*='...'].
    Strategy: fetch preamble/main page, then iterate /article/1 ... /article/N
    until 404 or missing content, then concatenate.
    """

    SUPPORTED = {
        "npt", "tpnw", "cwc", "bwc", "att", "ottawa", "ccm", "ccw"
    }

    def __init__(self, client: TreatyHTTPClient):
        self.client = client

    def scrape(self, treaty_id: str, registry: dict) -> TreatyScrapeResult:
        icrc_url = registry.get("icrc_url")
        if not icrc_url or treaty_id not in self.SUPPORTED:
            return TreatyScrapeResult(
                treaty_id=treaty_id,
                success=False,
                error="ICRC source not available for this treaty",
            )

        sections = []

        # Fetch main page for preamble
        soup = self.client.get_html(icrc_url)
        if soup:
            preamble = self._extract_preamble(soup)
            if preamble:
                sections.append("PREAMBLE\n\n" + preamble)

        # Iterate articles
        base_url = icrc_url.rstrip("/")
        for n in range(1, 150):
            article_url = f"{base_url}/article/{n}"
            soup = self.client.get_html(article_url)
            if soup is None:
                break
            content = self._extract_article(soup, n)
            if not content:
                # Two consecutive failures → stop
                break
            sections.append(content)

        if not sections:
            return TreatyScrapeResult(
                treaty_id=treaty_id,
                success=False,
                error="ICRC: no content extracted",
            )

        full_text = "\n\n".join(sections)
        return TreatyScrapeResult(
            treaty_id=treaty_id,
            success=True,
            text=full_text,
            sources_used=["icrc"],
        )

    def _extract_preamble(self, soup: BeautifulSoup) -> str:
        # Try CSS-module substring selectors
        for selector in [
            "[class*='Preamble']",
            "[class*='preamble']",
            "section[data-section='preamble']",
            "div[data-section='preamble']",
        ]:
            el = soup.select_one(selector)
            if el:
                return el.get_text(separator="\n", strip=True)
        return ""

    def _extract_article(self, soup: BeautifulSoup, n: int) -> str:
        # Article title
        title = ""
        for sel in ["[class*='ArticleTitle']", "h2.article-title", "h2"]:
            el = soup.select_one(sel)
            if el:
                title = el.get_text(strip=True)
                break

        # Article body
        body = ""
        for sel in [
            "[class*='ArticleContent']",
            "[class*='article-content']",
            "article[class*='treaty']",
            "div[class*='Content']",
        ]:
            el = soup.select_one(sel)
            if el:
                body = el.get_text(separator="\n", strip=True)
                break

        if not body:
            return ""

        header = title if title else f"Article {n}"
        return f"{header}\n\n{body}"


class UNODAScraper:
    """
    Scrapes treaty metadata and text/PDF from UNODA Drupal pages.
    """

    def __init__(self, client: TreatyHTTPClient, pdf_extractor: PDFExtractor):
        self.client = client
        self.pdf_extractor = pdf_extractor

    def scrape(self, treaty_id: str, registry: dict) -> TreatyScrapeResult:
        url = registry.get("unoda_url")
        if not url:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="No UNODA URL")

        soup = self.client.get_html(url)
        if soup is None:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="UNODA fetch failed")

        metadata = self._extract_metadata(soup)
        text = self._extract_text(soup, url)

        # Try PDF if no HTML text
        if not text:
            pdf_link = self._find_pdf_link(soup, url)
            if pdf_link:
                pdf_bytes = self.client.get_pdf_bytes(pdf_link)
                if pdf_bytes:
                    text = self.pdf_extractor.extract_text(pdf_bytes)

        return TreatyScrapeResult(
            treaty_id=treaty_id,
            success=bool(text or metadata),
            text=text,
            metadata=metadata,
            sources_used=["unoda"],
        )

    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        meta = {}
        # Date fields in Drupal field--label-inline blocks
        for block in soup.select("div.field--label-inline"):
            label_el = block.select_one("div.field__label")
            value_el = block.select_one("div.field__item, span.field__item")
            if label_el and value_el:
                label = label_el.get_text(strip=True).lower()
                value = value_el.get_text(strip=True)
                if "force" in label or "entry" in label:
                    meta["entry_into_force"] = value
                elif "sign" in label:
                    meta["opened_for_signature"] = value
                elif "parties" in label or "states" in label:
                    meta["parties_count_raw"] = value

        # Parties count as integer
        raw = meta.pop("parties_count_raw", "")
        if raw:
            m = re.search(r"\d+", raw)
            if m:
                meta["parties_count"] = int(m.group())

        return meta

    def _extract_text(self, soup: BeautifulSoup, base_url: str) -> str:
        for selector in [
            "div.field--name-body div.field__item",
            "div.field--type-text-with-summary .field__item",
            "div.field--name-body",
            "article .field__item",
        ]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text
        return ""

    def _find_pdf_link(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        for selector in [
            "div.treaty-documents a[href$='.pdf']",
            "div.views-row a[href$='.pdf']",
            "a[href$='.pdf']",
        ]:
            el = soup.select_one(selector)
            if el and el.get("href"):
                href = el["href"]
                if href.startswith("http"):
                    return href
                return urljoin(base_url, href)
        return None


class UNTreatyCollectionScraper:
    """
    Scrapes ratification/party data from treaties.un.org (ASP.NET WebForms).
    Handles multi-page grids by simulating postback pagination.
    """

    def __init__(self, client: TreatyHTTPClient):
        self.client = client

    def scrape(self, treaty_id: str, registry: dict) -> TreatyScrapeResult:
        url = registry.get("un_treaty_url")
        if not url:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="No UN Treaty URL")

        soup = self.client.get_html(url)
        if soup is None:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="UN Treaty Collection fetch failed")

        all_rows = []
        all_rows.extend(self._extract_table_rows(soup))

        # Pagination: follow all pages
        page_num = 2
        while True:
            next_soup = self._fetch_next_page(soup, url, page_num)
            if next_soup is None:
                break
            rows = self._extract_table_rows(next_soup)
            if not rows:
                break
            all_rows.extend(rows)
            # Check if there's another page
            if not self._has_page(next_soup, page_num + 1):
                break
            soup = next_soup
            page_num += 1

        if not all_rows:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="No party rows found")

        df = pd.DataFrame(all_rows, columns=["country_name", "signature_date", "ratification_date", "status"])
        df.insert(0, "treaty_id", treaty_id)
        df = normalise_parties_df(df)

        return TreatyScrapeResult(
            treaty_id=treaty_id,
            success=True,
            parties_df=df,
            sources_used=["un_treaty_collection"],
        )

    def _extract_table_rows(self, soup: BeautifulSoup) -> List[tuple]:
        rows = []
        table = (
            soup.select_one("table[id*='tblgrid']")
            or soup.select_one("table.GridViewStyle")
            or soup.select_one("table[class*='grid']")
        )
        if table is None:
            return rows

        for tr in table.select("tr"):
            if tr.select_one("th"):
                continue  # header row
            if "PagerStyle" in (tr.get("class") or []):
                continue

            cells = tr.select("td")
            if len(cells) < 2:
                continue

            country = cells[0].get_text(strip=True)
            if not country or country.isdigit():
                continue

            sig_date = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            rat_date = cells[2].get_text(strip=True) if len(cells) > 2 else ""

            # Infer status
            status = "non-party"
            if rat_date:
                status = "party"
            elif sig_date:
                status = "signatory"
            # Check for accession note
            if "accession" in rat_date.lower() or "accession" in (cells[2].get_text() if len(cells) > 2 else "").lower():
                status = "acceded"

            rows.append((country, sig_date, rat_date, status))
        return rows

    def _get_aspnet_state(self, soup: BeautifulSoup) -> dict:
        state = {}
        for name in ["__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"]:
            el = soup.select_one(f"input[name='{name}']")
            if el:
                state[name] = el.get("value", "")
        return state

    def _has_page(self, soup: BeautifulSoup, page_num: int) -> bool:
        pager = soup.select_one("tr.PagerStyle")
        if pager is None:
            return False
        return any(str(page_num) in a.get_text() for a in pager.select("a"))

    def _fetch_next_page(
        self, soup: BeautifulSoup, base_url: str, page_num: int
    ) -> Optional[BeautifulSoup]:
        state = self._get_aspnet_state(soup)
        if not state:
            return None

        # Find the grid control name for postback
        # Common pattern: ctl00$ctl00$...$GridView1
        grid_id = None
        pager = soup.select_one("tr.PagerStyle")
        if pager:
            link = pager.select_one("a")
            if link and link.get("href"):
                m = re.search(r"__doPostBack\('([^']+)'", link.get("href", ""))
                if m:
                    grid_id = m.group(1)

        if not grid_id:
            return None

        payload = dict(state)
        payload["__EVENTTARGET"] = grid_id
        payload["__EVENTARGUMENT"] = f"Page${page_num}"
        payload["__ASYNCPOST"] = "true"

        try:
            self.client._throttle()
            resp = self.client.session.post(base_url, data=payload, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "lxml")
        except Exception as exc:
            logger.warning("UN Treaty pagination failed (page %d): %s", page_num, exc)
            return None


class PDFDirectScraper:
    """
    Fetches a treaty text directly from a known PDF URL.
    Used for NWFZs, CTBT, TTBT, and other treaties with no good HTML source.
    """

    def __init__(self, client: TreatyHTTPClient, pdf_extractor: PDFExtractor):
        self.client = client
        self.pdf_extractor = pdf_extractor

    def scrape(self, treaty_id: str, registry: dict) -> TreatyScrapeResult:
        pdf_url = registry.get("text_url_pdf")
        if not pdf_url:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="No PDF URL in registry")

        pdf_bytes = self.client.get_pdf_bytes(pdf_url)
        if not pdf_bytes:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error=f"PDF fetch failed: {pdf_url}")

        text = self.pdf_extractor.extract_text(pdf_bytes)
        if not text:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="PDF text extraction yielded empty result")

        return TreatyScrapeResult(
            treaty_id=treaty_id,
            success=True,
            text=text,
            sources_used=["pdf_direct"],
        )


class ArmsControlAssociationScraper:
    """
    Scrapes Arms Control Association fact sheets for metadata cross-checking only.
    Clean WordPress HTML.
    """

    def __init__(self, client: TreatyHTTPClient):
        self.client = client

    def scrape_metadata(self, treaty_id: str, registry: dict) -> dict:
        url = registry.get("aca_url")
        if not url:
            return {}

        soup = self.client.get_html(url)
        if soup is None:
            return {}

        meta = {}

        # Published/updated date
        date_el = soup.select_one("time[datetime], span.date-published, .entry-date, .published")
        if date_el:
            meta["aca_last_updated"] = date_el.get("datetime") or date_el.get_text(strip=True)

        # Any fact sheet data table
        table = soup.select_one("table.factsheet-table, table.wp-block-table, table")
        if table:
            for row in table.select("tr"):
                cells = row.select("td, th")
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True).lower()
                    val = cells[1].get_text(strip=True)
                    if "parties" in key or "members" in key:
                        m = re.search(r"\d+", val)
                        if m:
                            meta["parties_count_aca"] = int(m.group())
                    elif "force" in key:
                        meta["entry_into_force_aca"] = val

        return meta


# ---------------------------------------------------------------------------
# Layer 4: Post-processing functions
# ---------------------------------------------------------------------------

def normalise_parties_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds country_iso3 column, standardises date strings, normalises status vocab.
    """
    if df.empty:
        return df

    # ISO3 lookup
    df = df.copy()
    if "country_name" in df.columns and "country_iso3" not in df.columns:
        df["country_iso3"] = df["country_name"].apply(normalize_country)

    # Standardise status to allowed vocab
    status_map = {
        "state party": "party",
        "states party": "party",
        "party": "party",
        "accession": "acceded",
        "acceded": "acceded",
        "succession": "acceded",
        "signatory": "signatory",
        "signed": "signatory",
        "observer": "observer",
        "non-party": "non-party",
        "": "non-party",
    }
    if "status" in df.columns:
        df["status"] = df["status"].str.lower().str.strip().map(
            lambda s: status_map.get(s, s)
        )

    # Parse date columns to ISO 8601 (best effort)
    for col in ["signature_date", "ratification_date"]:
        if col in df.columns:
            df[col] = df[col].apply(_try_parse_date)

    return df


def _try_parse_date(val: str) -> str:
    """Attempt to convert a date string to YYYY-MM-DD. Returns original on failure."""
    if not isinstance(val, str) or not val.strip():
        return ""
    # Already ISO?
    if re.match(r"\d{4}-\d{2}-\d{2}", val):
        return val[:10]
    try:
        import dateutil.parser
        return dateutil.parser.parse(val).strftime("%Y-%m-%d")
    except Exception:
        pass
    # Partial: just a year
    m = re.match(r"(\d{4})", val.strip())
    if m:
        return f"{m.group(1)}-01-01"
    return val.strip()


def extract_anchor_passages(treaty_id: str, full_text: str, n: int = 3) -> List[str]:
    """
    Heuristically extract n representative anchor passages from a full treaty text.
    Strategy:
      1. Preamble (first block before Article 1)
      2. Article 1 or Article I (obligations/prohibitions)
      3. Article with highest density of prohibitive language
    """
    if not full_text.strip():
        return []

    passages = []

    # Split on article boundaries
    article_pattern = re.compile(
        r"(?:^|\n)(?:ARTICLE|Article)\s+(?:[IVX]+|\d+)\b[.\s]",
        re.MULTILINE,
    )
    splits = article_pattern.split(full_text)
    headers = article_pattern.findall(full_text)

    # Passage 1: Preamble (text before first article)
    if splits:
        preamble = splits[0].strip()
        if len(preamble) > 100:
            passages.append(preamble[:2000])

    # Passage 2: Article 1
    if len(splits) > 1:
        art1 = splits[1].strip()
        if len(art1) > 50:
            passages.append(art1[:1500])

    # Passage 3: Article with highest prohibitive language density
    prohibitive_terms = re.compile(
        r"\b(shall not|undertakes|prohibited|eliminate|destroy|prohibition|"
        r"never under any circumstances|unconditionally)\b",
        re.IGNORECASE,
    )
    best_score = -1
    best_passage = ""
    for i, section in enumerate(splits[2:], start=2):
        if not section.strip():
            continue
        score = len(prohibitive_terms.findall(section))
        density = score / max(len(section.split()), 1)
        if density > best_score:
            best_score = density
            best_passage = section.strip()[:1500]

    if best_passage and best_passage not in passages:
        passages.append(best_passage)

    return passages[:n]


# ---------------------------------------------------------------------------
# Layer 5: Orchestrator
# ---------------------------------------------------------------------------

class TreatyScraper:
    """
    Orchestrates scraping across all sources, saves outputs.
    Source priority per treaty:
      text: ICRC → PDF direct → UNODA → trafilatura
      parties: UN Treaty Collection
      metadata: UNODA → ACA enrichment
    """

    def __init__(
        self,
        output_dir: Path,
        cache_dir: Path,
        rate_limit: float = 1.5,
        use_cache: bool = True,
        enabled_sources: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.texts_dir = self.output_dir / "texts"
        self.ratifications_dir = self.output_dir / "ratifications"
        self.metadata_dir = self.output_dir / "metadata"

        for d in [self.texts_dir, self.ratifications_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.enabled_sources = set(enabled_sources or ["icrc", "unoda", "un_treaties", "pdf_direct", "aca"])

        self.client = TreatyHTTPClient(cache_dir, rate_limit, use_cache)
        self.pdf_extractor = PDFExtractor()

        self.icrc = ICRCScraper(self.client)
        self.unoda = UNODAScraper(self.client, self.pdf_extractor)
        self.un_treaties = UNTreatyCollectionScraper(self.client)
        self.pdf_direct = PDFDirectScraper(self.client, self.pdf_extractor)
        self.aca = ArmsControlAssociationScraper(self.client)

    def scrape_treaty(self, treaty_id: str) -> TreatyScrapeResult:
        registry = TREATY_REGISTRY.get(treaty_id)
        if not registry:
            return TreatyScrapeResult(treaty_id=treaty_id, success=False, error="Unknown treaty ID")

        result = TreatyScrapeResult(treaty_id=treaty_id, success=False)

        # --- Text ---
        # 1. ICRC
        if "icrc" in self.enabled_sources and treaty_id in ICRCScraper.SUPPORTED and registry.get("icrc_url"):
            try:
                r = self.icrc.scrape(treaty_id, registry)
                if r.success and r.text:
                    result.text = r.text
                    result.sources_used.extend(r.sources_used)
                    logger.info("[%s] Text from ICRC (%d chars)", treaty_id, len(result.text))
            except Exception as exc:
                logger.warning("[%s] ICRC scrape error: %s", treaty_id, exc)

        # 2. PDF direct (if no text yet, or explicitly requested)
        if not result.text and "pdf_direct" in self.enabled_sources and registry.get("text_url_pdf"):
            try:
                r = self.pdf_direct.scrape(treaty_id, registry)
                if r.success and r.text:
                    result.text = r.text
                    result.sources_used.extend(r.sources_used)
                    logger.info("[%s] Text from PDF (%d chars)", treaty_id, len(result.text))
            except Exception as exc:
                logger.warning("[%s] PDF direct scrape error: %s", treaty_id, exc)

        # 3. UNODA (metadata + text fallback)
        if "unoda" in self.enabled_sources and registry.get("unoda_url"):
            try:
                r = self.unoda.scrape(treaty_id, registry)
                if r.metadata:
                    result.metadata.update(r.metadata)
                if not result.text and r.text:
                    result.text = r.text
                    result.sources_used.extend(r.sources_used)
                    logger.info("[%s] Text from UNODA (%d chars)", treaty_id, len(result.text))
                elif r.success:
                    result.sources_used.extend(r.sources_used)
            except Exception as exc:
                logger.warning("[%s] UNODA scrape error: %s", treaty_id, exc)

        # 4. trafilatura fallback for text_url
        text_url = registry.get("text_url")
        if not result.text and text_url:
            try:
                txt = self.client.get_text_trafilatura(text_url)
                if txt and len(txt) > 200:
                    result.text = txt
                    result.sources_used.append("trafilatura")
                    logger.info("[%s] Text from trafilatura (%d chars)", treaty_id, len(result.text))
            except Exception as exc:
                logger.warning("[%s] trafilatura fallback error: %s", treaty_id, exc)

        # --- Parties ---
        if "un_treaties" in self.enabled_sources and registry.get("un_treaty_url"):
            try:
                r = self.un_treaties.scrape(treaty_id, registry)
                if r.success and not r.parties_df.empty:
                    result.parties_df = r.parties_df
                    result.sources_used.extend(r.sources_used)
                    logger.info("[%s] Parties from UN Treaty Collection (%d rows)", treaty_id, len(r.parties_df))
            except Exception as exc:
                logger.warning("[%s] UN Treaty Collection error: %s", treaty_id, exc)

        # --- ACA metadata enrichment ---
        if "aca" in self.enabled_sources and registry.get("aca_url"):
            try:
                aca_meta = self.aca.scrape_metadata(treaty_id, registry)
                result.metadata.update(aca_meta)
            except Exception as exc:
                logger.warning("[%s] ACA metadata error: %s", treaty_id, exc)

        result.success = bool(result.text or not result.parties_df.empty)
        return result

    def run(self, treaty_ids: List[str], dry_run: bool = False) -> None:
        start = time.monotonic()
        logger.info("Starting treaty scrape: %d treaties", len(treaty_ids))

        if dry_run:
            logger.info("DRY RUN — planned actions:")
            for tid in treaty_ids:
                reg = TREATY_REGISTRY.get(tid, {})
                sources = []
                if tid in ICRCScraper.SUPPORTED and reg.get("icrc_url"):
                    sources.append("ICRC")
                if reg.get("text_url_pdf"):
                    sources.append("PDF")
                if reg.get("unoda_url"):
                    sources.append("UNODA")
                if reg.get("un_treaty_url"):
                    sources.append("UN-Treaty-Collection(parties)")
                logger.info("  [%s] sources: %s", tid, ", ".join(sources) or "none")
            return

        results = []
        for tid in treaty_ids:
            logger.info("--- Scraping: %s ---", tid)
            try:
                result = self.scrape_treaty(tid)
                results.append(result)
            except Exception as exc:
                logger.error("[%s] Unexpected error: %s", tid, exc)
                results.append(TreatyScrapeResult(treaty_id=tid, success=False, error=str(exc)))

        # Save outputs
        metadata_rows = []
        for result in results:
            if result.text:
                self._save_text(result.treaty_id, result.text, result.sources_used)
            if not result.parties_df.empty:
                self._save_parties(result.treaty_id, result.parties_df)
            metadata_rows.append(self._build_metadata_row(result))

        self._save_metadata(metadata_rows)

        elapsed = time.monotonic() - start
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        logger.info("=" * 60)
        logger.info("Scrape complete in %.1fs", elapsed)
        logger.info("  Success: %d / %d", successes, len(results))
        logger.info("  Failed:  %d / %d", failures, len(results))
        for r in results:
            status = "OK" if r.success else "FAIL"
            logger.info("  [%s] %s  %s", status, r.treaty_id, r.error or "")
        logger.info("  Output: %s", self.output_dir)
        logger.info("=" * 60)

    def _save_text(self, treaty_id: str, text: str, sources: List[str]) -> None:
        primary_source = sources[0] if sources else "unknown"
        header = (
            f"TREATY_ID: {treaty_id}\n"
            f"SOURCE: {primary_source}\n"
            f"SCRAPE_DATE: {TODAY}\n"
            "---\n"
        )
        path = self.texts_dir / f"{treaty_id}_full.txt"
        path.write_text(header + text, encoding="utf-8")
        logger.debug("[%s] Saved text: %s", treaty_id, path)

    def _save_parties(self, treaty_id: str, df: pd.DataFrame) -> None:
        path = self.ratifications_dir / f"{treaty_id}_parties.csv"
        save_csv(df, path)
        logger.debug("[%s] Saved parties: %s (%d rows)", treaty_id, path, len(df))

    def _build_metadata_row(self, result: TreatyScrapeResult) -> dict:
        reg = TREATY_REGISTRY.get(result.treaty_id, {})
        row = {
            "treaty_id": result.treaty_id,
            "name": reg.get("name", ""),
            "abbreviation": reg.get("abbreviation", ""),
            "year_signed": reg.get("year_signed"),
            "year_in_force": reg.get("year_in_force"),
            "type": reg.get("type", ""),
            "category": reg.get("category", ""),
            "depositary": reg.get("depositary", ""),
            "parties_count": result.metadata.get("parties_count") or result.metadata.get("parties_count_aca"),
            "url_primary": reg.get("icrc_url") or reg.get("unoda_url") or reg.get("text_url_pdf") or "",
            "text_chars": len(result.text),
            "parties_rows": len(result.parties_df) if not result.parties_df.empty else 0,
            "scrape_success": result.success,
            "sources_used": ",".join(result.sources_used),
            "scrape_date": TODAY,
        }
        return row

    def _save_metadata(self, rows: List[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        path = self.metadata_dir / "treaties_metadata.csv"
        save_csv(df, path)
        logger.info("Saved metadata: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape arms control treaty texts, metadata, and ratification data."
    )
    parser.add_argument(
        "--treaties",
        nargs="+",
        default=["all"],
        metavar="TREATY_ID",
        help=f"Treaty IDs to scrape, or 'all'. Options: {', '.join(ALL_TREATY_IDS)}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/raw/treaties/ relative to project root)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache directory (default: <output>/.cache/)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.5,
        help="Minimum seconds between requests (default: 1.5)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["all"],
        choices=["all", "icrc", "unoda", "un_treaties", "aca", "pdf_direct"],
        help="Which sources to use (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response cache; always re-fetch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scraping plan without making network requests",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _resolve_output_dir(output_arg: Optional[Path]) -> Path:
    if output_arg:
        return output_arg.resolve()
    # Walk up to find project root (contains run_pipeline.py)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "run_pipeline.py").exists():
            return parent / "data" / "raw" / "treaties"
    # Fallback: relative to cwd
    return Path("data/raw/treaties").resolve()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve treaty IDs
    if "all" in args.treaties:
        treaty_ids = ALL_TREATY_IDS
    else:
        unknown = [t for t in args.treaties if t not in TREATY_REGISTRY]
        if unknown:
            logger.error("Unknown treaty IDs: %s. Valid: %s", unknown, ALL_TREATY_IDS)
            sys.exit(1)
        treaty_ids = args.treaties

    # Resolve paths
    output_dir = _resolve_output_dir(args.output)
    cache_dir = args.cache or (output_dir / ".cache")

    # Sources
    if "all" in args.sources:
        enabled_sources = ["icrc", "unoda", "un_treaties", "aca", "pdf_direct"]
    else:
        enabled_sources = args.sources

    logger.info("Output: %s", output_dir)
    logger.info("Cache:  %s", cache_dir)
    logger.info("Treaties: %s", treaty_ids)
    logger.info("Sources: %s", enabled_sources)

    scraper = TreatyScraper(
        output_dir=output_dir,
        cache_dir=cache_dir,
        rate_limit=args.rate_limit,
        use_cache=not args.no_cache,
        enabled_sources=enabled_sources,
    )
    scraper.run(treaty_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
