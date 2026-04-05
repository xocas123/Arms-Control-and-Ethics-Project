"""Load UN First Committee speeches (stub — requires scraping)."""
import pandas as pd
import warnings

def load_first_committee(data_dir="data/raw/first_committee"):
    """
    Loads UN First Committee speeches. Currently stubbed.
    First Committee speeches cover arms control specifically (no segmentation needed).
    """
    warnings.warn(
        "[First Committee] Real data requires scraping Reaching Critical Will or UN Digital Library. "
        "Returning empty DataFrame.",
        UserWarning, stacklevel=2
    )
    return pd.DataFrame(columns=["country_iso3", "country_name", "year", "text", "source"])
