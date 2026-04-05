"""Load First Committee (DISEC) resolution texts."""
import pandas as pd
from pathlib import Path

RESOLUTIONS_CSV = Path("data/raw/resolutions/resolutions.csv")


def load_resolutions(data_dir="data/raw/resolutions") -> pd.DataFrame:
    """
    Load First Committee resolution texts downloaded from Reaching Critical Will.
    Run download_resolutions.py first to populate data/raw/resolutions/.

    Returns DataFrame:
      symbol         -- A/C.1/79/L.3
      title          -- resolution title
      year           -- int
      session        -- UNGA session number
      full_text      -- full resolution text (empty string if --no-pdf was used)
      frame_type     -- 'humanitarian' / 'security' / 'mixed' / 'other'
      treaty_flag    -- 'npt' / 'tpnw' / 'ottawa' / 'ccm' / 'att' / None
      thematic_cluster -- broad topic label
    """
    csv_path = Path(data_dir) / "resolutions.csv"

    if not csv_path.exists():
        print(
            "[Resolutions] No data found. Run: python download_resolutions.py\n"
            f"              Expected: {csv_path}"
        )
        return pd.DataFrame(columns=[
            "symbol", "title", "year", "session",
            "full_text", "frame_type", "treaty_flag", "thematic_cluster"
        ])

    df = pd.read_csv(csv_path, dtype={"year": int, "session": int})
    df["full_text"] = df["full_text"].fillna("")

    # Classify frame and treaty from title + text
    from src.data.load_voting import classify_resolution_frame, flag_treaty
    combined = df["title"].fillna("") + " " + df["full_text"].str[:500]
    df["frame_type"]       = combined.apply(classify_resolution_frame)
    df["treaty_flag"]      = combined.apply(flag_treaty)
    df["thematic_cluster"] = combined.apply(_thematic_cluster)

    print(
        f"[Resolutions] Loaded {len(df)} resolutions "
        f"({df['year'].min()}-{df['year'].max()}, "
        f"{(df['full_text'] != '').sum()} with full text)"
    )
    return df


def _thematic_cluster(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["nuclear weapon", "npt", "tpnw", "nuclear-weapon-free",
                              "nuclear disarmament", "ctbt", "fissile"]):
        return "nuclear"
    if any(k in t for k in ["chemical weapon", "biological weapon", "bwc", "cwc"]):
        return "wmd_other"
    if any(k in t for k in ["cluster munition", "landmine", "anti-personnel", "ottawa"]):
        return "humanitarian_conventional"
    if any(k in t for k in ["arms trade", "att", "small arms", "light weapons", "salw"]):
        return "conventional_trade"
    if any(k in t for k in ["outer space", "arms race in outer space", "paros"]):
        return "outer_space"
    if any(k in t for k in ["cyber", "information security", "ict"]):
        return "cyber"
    if any(k in t for k in ["transparency", "confidence-building", "register",
                              "military expenditure"]):
        return "transparency_cbm"
    return "general_disarmament"
