#!/usr/bin/env python3
"""
Download First Committee (DISEC) resolutions from Reaching Critical Will.
Covers 2002-2024. All resolutions on this site are arms-control related.

Outputs:
  data/raw/resolutions/pdfs/       -- raw PDFs
  data/raw/resolutions/resolutions.csv  -- title, year, symbol, full_text, ...

Usage:
  python download_resolutions.py
  python download_resolutions.py --years 2015 2020   # specific years only
  python download_resolutions.py --no-pdf            # metadata only, skip PDF text
"""
import argparse
import csv
import re
import time
import urllib.request
import urllib.error
from io import BytesIO
from pathlib import Path

RCW_BASE = "https://www.reachingcriticalwill.org"
RCW_PDF_BASE = "https://reachingcriticalwill.org/images/documents/Disarmament-fora/1com"

RAW_DIR  = Path("data/raw/resolutions")
PDF_DIR  = RAW_DIR / "pdfs"
OUT_CSV  = RAW_DIR / "resolutions.csv"

YEARS = list(range(2002, 2025))   # 2002-2024
HEADERS = {"User-Agent": "arms-control-nlp-research/1.0"}

# UNGA session number = year - 1945
def session(year: int) -> int:
    return year - 1945

def year_suffix(year: int) -> str:
    return str(year)[-2:]   # "24" for 2024


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def fetch(url: str, binary: bool = False, retries: int = 3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read() if binary else r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes. Tries pdfplumber then pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except ImportError:
        pass
    return ""


# ── Resolution index parsing ──────────────────────────────────────────────────

def parse_resolution_index(html: str, year: int) -> list[dict]:
    """
    Parse the RCW resolution listing page.
    Structure per entry (may vary slightly across years):
      <p><strong><a href="...L1.pdf">A/C.1/78/L.1</a></strong><br />
      Title text here
    Returns list of dicts with: symbol, title, pdf_url
    """
    sess = session(year)
    resolutions = []
    seen = set()

    # Each resolution is a <p> containing a PDF link with the symbol as text
    # Pattern: <p>...<a href="...pdf">SYMBOL</a>...TITLE
    entry_pat = re.compile(
        r'<p[^>]*>.*?'                          # opening <p>
        r'<a\s+href="([^"]+\.pdf)"[^>]*>'       # PDF href
        r'([^<]+)'                               # symbol text inside <a>
        r'</a>'                                  # close <a>
        r'.*?<br\s*/?>\s*'                       # <br> after symbol
        r'([^\n<]{5,200})',                      # title (first meaningful text)
        re.IGNORECASE | re.DOTALL,
    )

    for m in entry_pat.finditer(html):
        href   = m.group(1).strip()
        symbol = re.sub(r'\s+', ' ', m.group(2)).strip()
        title  = re.sub(r'\s+', ' ', m.group(3)).strip()
        title  = re.sub(r'<[^>]+>', '', title).strip()  # strip any inline tags

        # Only keep actual resolution symbols for this year's session
        if not re.search(rf'A/C\.1/{sess}/L\.\d+', symbol, re.IGNORECASE):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)

        pdf_url = href if href.startswith("http") else RCW_BASE + href

        resolutions.append({
            "symbol":  symbol,
            "title":   title,
            "pdf_url": pdf_url,
            "year":    year,
            "session": sess,
        })

    return resolutions


# ── Main download loop ────────────────────────────────────────────────────────

def download_year(year: int, skip_pdf: bool) -> list[dict]:
    url = f"{RCW_BASE}/disarmament-fora/unga/{year}/resolutions"
    print(f"\n  [{year}] Fetching index: {url}")
    html = fetch(url)
    if not html:
        print(f"  [{year}] Could not fetch index — skipping.")
        return []

    resolutions = parse_resolution_index(html, year)
    if not resolutions:
        print(f"  [{year}] No resolutions parsed from page.")
        return []

    print(f"  [{year}] Found {len(resolutions)} resolutions.")

    yy = year_suffix(year)
    year_pdf_dir = PDF_DIR / str(year)
    year_pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, res in enumerate(resolutions, 1):
        row = {**res, "full_text": ""}

        if not skip_pdf:
            # Derive local filename from symbol (strip all Windows-invalid chars)
            safe = re.sub(r'[\\/:*?"<>|]', '_', res["symbol"]).replace(" ", "_")
            local_pdf = year_pdf_dir / f"{safe}.pdf"

            if local_pdf.exists():
                pdf_bytes = local_pdf.read_bytes()
            else:
                pdf_bytes = fetch(res["pdf_url"], binary=True)
                if pdf_bytes:
                    local_pdf.write_bytes(pdf_bytes)
                else:
                    print(f"    [{i}/{len(resolutions)}] MISSING: {res['symbol']}")
                    rows.append(row)
                    continue

            row["full_text"] = extract_pdf_text(pdf_bytes)
            print(f"    [{i}/{len(resolutions)}] {res['symbol']}  ({len(row['full_text'])} chars)")
        else:
            print(f"    [{i}/{len(resolutions)}] {res['symbol']}  (--no-pdf)")

        rows.append(row)
        time.sleep(0.3)   # be polite

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        help="Specific years to download (default: all 2002-2024)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF download and text extraction (metadata only)")
    args = parser.parse_args()

    years = args.years or YEARS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  First Committee Resolution Downloader")
    print(f"  Source: Reaching Critical Will")
    print(f"  Years : {min(years)}-{max(years)}")
    print(f"  PDFs  : {'no' if args.no_pdf else 'yes'}")
    print("=" * 60)

    # Check PDF library availability
    if not args.no_pdf:
        try:
            import pdfplumber
            print("  PDF parser: pdfplumber")
        except ImportError:
            try:
                from pypdf import PdfReader
                print("  PDF parser: pypdf")
            except ImportError:
                print("  WARNING: no PDF parser found.")
                print("  Install one: pip install pdfplumber")
                print("  Falling back to --no-pdf mode.")
                args.no_pdf = True

    all_rows = []
    for year in sorted(years):
        rows = download_year(year, skip_pdf=args.no_pdf)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo resolutions downloaded.")
        return

    # Write CSV
    fields = ["symbol", "title", "year", "session", "pdf_url", "full_text"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 60)
    print(f"  Done: {len(all_rows)} resolutions")
    print(f"  CSV : {OUT_CSV}")
    print(f"  PDFs: {PDF_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
