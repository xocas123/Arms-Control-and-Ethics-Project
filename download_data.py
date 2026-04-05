#!/usr/bin/env python3
"""
Download real data for the arms-control-nlp pipeline.

Sources:
  1. UNGDC  — UN General Debate Corpus (Harvard Dataverse, doi:10.7910/DVN/0TJX8Y)
  2. Voting — Voeten UNGA Voting Data (Harvard Dataverse, doi:10.7910/DVN/LEJUQZ)
  3. V-Dem  — requires free registration at v-dem.net (instructions printed below)

Run: python download_data.py
"""
import os
import sys
import json
import zipfile
import shutil
import urllib.request
import urllib.error
from pathlib import Path

DATAVERSE = "https://dataverse.harvard.edu"

RAW = Path("data/raw")
UNGDC_DIR   = RAW / "ungdc"
UNVOTES_DIR = RAW / "unvotes"
VDEM_DIR    = RAW / "vdem"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, label: str = ""):
    """Stream-download url → dest with a progress counter."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label or dest.name} …", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "arms-control-nlp/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1 << 20  # 1 MB
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading {label or dest.name} … {pct:.0f}%", end="", flush=True)
        print(f"\r  Downloaded  {label or dest.name} ({downloaded / 1e6:.1f} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"\n  ERROR {e.code}: {e.reason} — {url}")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def dataverse_files(doi: str) -> list[dict]:
    """Return file metadata list from Dataverse dataset."""
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={doi}"
    req = urllib.request.Request(url, headers={"User-Agent": "arms-control-nlp/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        meta = json.loads(resp.read())
    return meta["data"]["latestVersion"]["files"]


def dataverse_download_url(file_id: int) -> str:
    return f"{DATAVERSE}/api/access/datafile/{file_id}"


# ──────────────────────────────────────────────────────────────────────────────
# 1. UNGDC
# ──────────────────────────────────────────────────────────────────────────────

UNGDC_DOI = "doi:10.7910/DVN/0TJX8Y"
# File ID for UNGDC_1946-2025.tar.gz (confirmed via Dataverse API)
UNGDC_TARGZ_ID = 13591895

def download_ungdc():
    print("\n[1/3] UN General Debate Corpus (UNGDC)")
    print(f"      Source : Harvard Dataverse {UNGDC_DOI}")
    print(f"      Covers : 1946-2025, one TXT per country-year")

    existing_txts = list(UNGDC_DIR.glob("*.txt"))
    if len(existing_txts) > 5:
        print(f"      Already have {len(existing_txts)} TXT files — skipping.")
        return

    UNGDC_DIR.mkdir(parents=True, exist_ok=True)

    dest = UNGDC_DIR / "UNGDC_1946-2025.tar.gz"
    if not dest.exists():
        ok = download(dataverse_download_url(UNGDC_TARGZ_ID), dest, label="UNGDC_1946-2025.tar.gz")
        if not ok:
            return
    else:
        print("      Archive already downloaded.")

    print("      Extracting archive (this may take a minute) ...", flush=True)
    import tarfile
    with tarfile.open(dest, "r:gz") as tf:
        tf.extractall(UNGDC_DIR)
    dest.unlink()

    # Flatten any nested directories
    for p in list(UNGDC_DIR.rglob("*.txt")):
        if p.parent != UNGDC_DIR:
            target = UNGDC_DIR / p.name
            if not target.exists():
                shutil.move(str(p), str(target))

    n = len(list(UNGDC_DIR.glob("*.txt")))
    print(f"      Done -- {n} TXT files extracted.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. UNGA Voting Data (unvotes R package via Tidy Tuesday mirror)
# ──────────────────────────────────────────────────────────────────────────────
# Source: TidyTuesday 2021-03-23 mirror of David Robinson's unvotes package.
# Three files: unvotes.csv, roll_calls.csv, issues.csv
# Provides rcid-level country votes with issue classifications (1946-2019).

TT_BASE = ("https://raw.githubusercontent.com/rfordatascience/tidytuesday"
           "/master/data/2021/2021-03-23")
VOTING_FILES = ["unvotes.csv", "roll_calls.csv", "issues.csv"]

def download_voting():
    print("\n[2/3] UNGA Voting Data (unvotes / TidyTuesday)")
    print(f"      Source : TidyTuesday mirror of unvotes R package")
    print(f"      Files  : {', '.join(VOTING_FILES)}")

    existing = list(UNVOTES_DIR.glob("*.csv"))
    if len(existing) >= 3:
        print(f"      Already have {len(existing)} CSV file(s) -- skipping.")
        return

    UNVOTES_DIR.mkdir(parents=True, exist_ok=True)

    for fname in VOTING_FILES:
        dest = UNVOTES_DIR / fname
        if dest.exists():
            print(f"      {fname} already present.")
            continue
        download(f"{TT_BASE}/{fname}", dest, label=fname)

    print(f"      Done -- {len(list(UNVOTES_DIR.glob('*.csv')))} CSV file(s) saved.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. V-Dem (manual — requires free registration)
# ──────────────────────────────────────────────────────────────────────────────

def print_vdem_instructions():
    vdem_dir = VDEM_DIR.resolve()
    print()
    print("[3/3] V-Dem Dataset (manual download required)")
    print("      V-Dem requires a free account at v-dem.net. Steps:")
    print()
    print("      1. Go to  https://www.v-dem.net/data/the-v-dem-dataset/")
    print('      2. Click "V-Dem Dataset v14" (or latest)')
    print('         then "Country-Year: V-Dem Full + Others" -> CSV (~200 MB)')
    print("      3. Place the CSV (e.g. V-Dem-CY-Full+Others-v14.csv) in:")
    print(f"           {vdem_dir}")
    print()
    print("      Pipeline columns used: country_text_id, year, v2x_regime,")
    print("      v2x_polyarchy, v2x_libdem.")
    print()
    print("      If skipped, Q2 (democracy vs autocracy) will warn and use")
    print("      a static fallback regime classification.")
    VDEM_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Arms Control NLP — Data Downloader")
    print("=" * 60)

    download_ungdc()
    download_voting()
    print_vdem_instructions()

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    ungdc_n  = len(list(UNGDC_DIR.glob("*.txt")))
    votes_n  = len(list(UNVOTES_DIR.glob("*.csv")))
    vdem_n   = len(list(VDEM_DIR.glob("*.csv")))
    print(f"  UNGDC speeches : {ungdc_n} TXT files")
    print(f"  Voting CSVs    : {votes_n} CSV files")
    print(f"  V-Dem CSVs     : {vdem_n} CSV files  (<-- manual step above if 0)")
    print("=" * 60)


if __name__ == "__main__":
    main()
