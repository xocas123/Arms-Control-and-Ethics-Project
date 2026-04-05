"""Load V-Dem regime scores."""
import pandas as pd
from pathlib import Path

def load_vdem(data_dir="data/raw/vdem"):
    """
    Load V-Dem regime data.

    Returns DataFrame: country_iso3, year, v2x_regime (0-3), v2x_polyarchy (0-1), v2x_libdem (0-1)
    v2x_regime: 0=closed autocracy, 1=electoral autocracy, 2=electoral democracy, 3=liberal democracy
    """
    data_path = Path(data_dir)

    if data_path.exists():
        for csv_file in data_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                if _is_vdem_format(df):
                    return _process_vdem(df)
            except Exception as e:
                print(f"[V-Dem] Could not load {csv_file}: {e}")

    print("[V-Dem] No data found. Returning empty DataFrame.")
    return pd.DataFrame(columns=["country_iso3", "year", "v2x_regime", "v2x_polyarchy", "v2x_libdem"])


def _is_vdem_format(df):
    return any(c in df.columns for c in ["v2x_regime", "V2X_REGIME", "regime"])


def _process_vdem(df):
    """Process V-Dem CSV export."""
    result = df.copy()

    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["country_text_id", "country_iso3", "iso3"]:
            col_map[col] = "country_iso3"
        elif col_lower == "year":
            col_map[col] = "year"
        elif col_lower == "v2x_regime":
            col_map[col] = "v2x_regime"
        elif col_lower == "v2x_polyarchy":
            col_map[col] = "v2x_polyarchy"
        elif col_lower == "v2x_libdem":
            col_map[col] = "v2x_libdem"

    result = result.rename(columns=col_map)
    keep_cols = [c for c in ["country_iso3", "year", "v2x_regime", "v2x_polyarchy", "v2x_libdem"] if c in result.columns]
    return result[keep_cols].dropna(subset=["country_iso3", "year"])


