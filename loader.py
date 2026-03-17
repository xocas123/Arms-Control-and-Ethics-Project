"""
loader.py — Dynamic data loading and master dataset construction.

All column names and file existence are checked at runtime. Missing files
produce warnings, not errors. The master dataset is a left-join from the NLP
backbone, so every country-year with NLP data is represented.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from country_bridge import add_country_code, build_bridge

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_file(path: Path, label: str = "") -> pd.DataFrame | None:
    """Load a CSV, returning None (with a warning) if it doesn't exist."""
    if not path.exists():
        logger.warning("File not found [%s]: %s", label or path.name, path)
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %-30s  %d rows × %d cols", label or path.name,
                    len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.warning("Could not read [%s]: %s", label or path.name, exc)
        return None


def _filter_years(df: pd.DataFrame, year_start: int | None,
                  year_end: int | None) -> pd.DataFrame:
    if "year" not in df.columns:
        return df
    if year_start is not None:
        df = df[df["year"] >= year_start]
    if year_end is not None:
        df = df[df["year"] <= year_end]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NLP pipeline loader
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_nlp_country_key(df: pd.DataFrame) -> pd.DataFrame:
    """Unify country key to 'country_code' (ISO3, uppercase)."""
    # New pipeline uses country_iso3
    if "country_iso3" in df.columns and "country_code" not in df.columns:
        df = df.rename(columns={"country_iso3": "country_code"})
    if "country_code" in df.columns:
        df["country_code"] = df["country_code"].astype(str).str.upper().str.strip()
    return df


def _process_trajectory_file(df: pd.DataFrame, treaty_name: str) -> pd.DataFrame | None:
    """
    Convert a similarity_trajectories CSV into a per-(country_code, year)
    DataFrame with a single '{treaty_name}_similarity' column.
    Keeps only the 'similarity' column, renamed.
    """
    if df is None:
        return None
    df = _normalise_nlp_country_key(df)
    if "country_code" not in df.columns or "year" not in df.columns:
        return None
    if "similarity" not in df.columns:
        return None
    out = (df[["country_code", "year", "similarity"]]
           .drop_duplicates(subset=["country_code", "year"])
           .rename(columns={"similarity": f"{treaty_name}_similarity"}))
    return out


def load_all_nlp(cfg: Config) -> dict[str, pd.DataFrame | None]:
    dfs: dict[str, pd.DataFrame | None] = {}

    # Treaty trajectory keys that need special processing
    trajectory_keys = {
        "att_trajectories":    "att",
        "tpnw_trajectories":   "tpnw",
        "ottawa_trajectories": "ottawa",
        "ccm_trajectories":    "ccm",
    }

    for key, path in cfg.nlp_files.items():
        df = load_file(path, label=f"nlp/{key}")
        if df is None:
            dfs[key] = None
            continue

        df = _filter_years(df, cfg.year_start, cfg.year_end)

        # Special handling for trajectory files → pivot to country-year similarity
        if key in trajectory_keys:
            df = _process_trajectory_file(df, trajectory_keys[key])
        else:
            df = _normalise_nlp_country_key(df)

        dfs[key] = df

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Network pipeline loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_network(cfg: Config,
                     bridge: dict[str, str | None]
                     ) -> dict[str, pd.DataFrame | None]:
    dfs: dict[str, pd.DataFrame | None] = {}
    unresolved_log: list[str] = []

    for key, path in cfg.net_files.items():
        df = load_file(path, label=f"net/{key}")
        if df is None:
            dfs[key] = None
            continue
        df = _filter_years(df, cfg.year_start, cfg.year_end)

        # Add country_code via bridge for node-level files
        if "country" in df.columns and "country_code" not in df.columns:
            df, unresolved = add_country_code(df, "country", bridge)
            unresolved_log.extend(unresolved)
        # edge_metrics has both supplier + recipient columns
        elif ("supplier" in df.columns and "recipient" in df.columns
              and "supplier_code" not in df.columns):
            df, unres_s = add_country_code(df, "supplier", bridge)
            df = df.rename(columns={"country_code": "supplier_code"})
            df, unres_r = add_country_code(df, "recipient", bridge)
            df = df.rename(columns={"country_code": "recipient_code"})
            unresolved_log.extend(unres_s + unres_r)
        # Files with only a supplier column (e.g. complicity)
        elif "supplier" in df.columns and "country_code" not in df.columns:
            df, unresolved = add_country_code(df, "supplier", bridge)
            unresolved_log.extend(unresolved)
        dfs[key] = df

    if unresolved_log:
        unique_unresolved = sorted(set(unresolved_log))
        logger.info("%d unique network country names unresolved: %s",
                    len(unique_unresolved), unique_unresolved[:20])

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Edge-level aggregation → supplier-year risk scores
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_edge_risk(edge_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Aggregate edge_metrics to (supplier_code, year) level.
    Only computes columns that exist in edge_df.
    Returns a DataFrame with columns prefixed 'net_'.
    """
    if edge_df is None:
        return None

    if "supplier_code" not in edge_df.columns or "year" not in edge_df.columns:
        logger.warning("edge_metrics missing supplier_code or year — skipping aggregation")
        return None

    key = ["supplier_code", "year"]
    agg: dict[str, Any] = {}

    # Boolean flag columns → count of True rows
    bool_flag_map = {
        "ethical_tension":   "net_n_ethical_tension",
        "into_conflict":     "net_n_into_conflict",
        "into_war":          "net_n_into_war",
        "att_concern":       "net_n_att_concern",
        "embargo_violation": "net_n_embargo_violations",
    }
    for src, dst in bool_flag_map.items():
        if src in edge_df.columns:
            agg[src] = "sum"

    # Numeric columns → mean / sum
    if "ethical_risk_score" in edge_df.columns:
        agg["ethical_risk_score"] = ["mean", "max"]
    if "tiv" in edge_df.columns:
        agg["tiv"] = "sum"

    if not agg:
        logger.warning("edge_metrics has no recognized violation columns")
        return None

    # Convert boolean cols to int before aggregation
    for col in bool_flag_map:
        if col in edge_df.columns:
            edge_df = edge_df.copy()
            edge_df[col] = edge_df[col].astype(bool).astype(int)

    grouped = edge_df.groupby(key).agg(agg)
    grouped.columns = [
        "_".join(filter(None, c)) if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    grouped = grouped.reset_index()

    # Total transfers per supplier-year for pct calculation
    total = edge_df.groupby(key).size().reset_index(name="n_total_transfers")
    grouped = grouped.merge(total, on=key, how="left")

    # Compute pct_ethical (any flag / total)
    flag_sum_cols = [c for c in grouped.columns
                     if c in list(bool_flag_map.values())]
    # Rename to net_ prefix
    rename = {
        "ethical_risk_score_mean": "net_mean_ethical_risk_score",
        "ethical_risk_score_max":  "net_max_ethical_risk_score",
        "tiv_sum":                 "net_supplier_tiv",
    }
    for src, dst in bool_flag_map.items():
        if src in agg:
            rename[src + "_sum"] = dst
            rename[src] = dst  # if aggregation didn't append _sum

    grouped = grouped.rename(columns=rename)

    # pct_ethical = fraction of transfers with any flag
    actual_flag_cols = [v for v in bool_flag_map.values()
                        if v in grouped.columns]
    if actual_flag_cols and "n_total_transfers" in grouped.columns:
        grouped["net_n_any_violation"] = grouped[actual_flag_cols].sum(axis=1)
        grouped["net_pct_ethical"] = (
            grouped["net_n_any_violation"] / grouped["n_total_transfers"]
        ).clip(0, 1)

    # Rename join keys to match NLP convention
    grouped = grouped.rename(columns={"supplier_code": "country_code"})
    grouped = grouped.drop(columns=["n_total_transfers"], errors="ignore")

    logger.info("Edge aggregation: %d supplier-year rows", len(grouped))
    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# Master dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_master_dataset(
    nlp_dfs: dict[str, pd.DataFrame | None],
    net_dfs: dict[str, pd.DataFrame | None],
    cfg: Config,
) -> tuple[pd.DataFrame, dict]:
    """
    Build the master merged dataset.

    Left-joins from the NLP rhetoric_scores as the backbone. Joins every
    available NLP and network table. Network columns are prefixed 'net_' to
    avoid collisions. Adds boolean 'in_both_pipelines'.

    Returns (master_df, join_report).
    """
    join_key = cfg.NLP_JOIN_KEY  # ["country_code", "year"]

    # ── Select NLP backbone (prefer new pipeline files, fall back to legacy) ──
    backbone_priority = (
        "frame_scores",          # new: per-country-year frame ratios
        "rhetoric_scores",       # legacy: rhetoric component scores
        "rhetoric_gap",          # legacy: gap index
        "sentiment",             # legacy: VADER sentiment
    )
    def _is_usable(df) -> bool:
        return df is not None and "country_code" in df.columns

    backbone_key = next(
        (k for k in backbone_priority if _is_usable(nlp_dfs.get(k))),
        None
    )
    if backbone_key is None:
        raise RuntimeError("No usable NLP backbone file found. Check NLP output dir.")

    master = nlp_dfs[backbone_key].copy()
    logger.info("Backbone: nlp/%s  (%d rows)", backbone_key, len(master))

    # ── Merge remaining NLP tables ────────────────────────────────────────────
    # Skip keys that are the backbone or that lack the join key
    skip_keys = {
        backbone_key,
        "topic_info",          # no country/year key
        "frame_by_regime",     # aggregate (regime, year) — no country key
        "transitions_legacy",  # duplicate of transitions
    }
    for key, df in nlp_dfs.items():
        if key in skip_keys or df is None:
            continue
        available_key = [c for c in join_key if c in df.columns]
        if len(available_key) < 2:
            logger.debug("nlp/%s missing join columns — skipping", key)
            continue
        # Avoid duplicate columns (keep left side)
        new_cols = [c for c in df.columns
                    if c not in master.columns or c in join_key]
        if len(new_cols) <= len(join_key):
            continue  # nothing new to add
        master = master.merge(df[new_cols], on=join_key, how="left",
                              suffixes=("", f"_{key}"))

    master_before_net = len(master.columns)

    # ── Aggregate edge risk → supplier-year ──────────────────────────────────
    edge_risk = None
    if net_dfs.get("edge_metrics") is not None:
        edge_risk = aggregate_edge_risk(net_dfs["edge_metrics"])

    # ── Merge network node metrics ────────────────────────────────────────────
    def _prefix_net(df: pd.DataFrame, skip: list[str]) -> pd.DataFrame:
        """Prefix non-join columns with 'net_'."""
        rename = {c: f"net_{c}" for c in df.columns if c not in skip}
        return df.rename(columns=rename)

    for key in ("node_metrics", "communities"):
        df = net_dfs.get(key)
        if df is None or "country_code" not in df.columns:
            continue
        df = _prefix_net(df, skip=join_key + ["country"])
        new_cols = [c for c in df.columns
                    if c not in master.columns or c in join_key]
        master = master.merge(df[new_cols], on=join_key, how="left")

    # Merge edge-level risk aggregation
    if edge_risk is not None:
        new_cols = [c for c in edge_risk.columns
                    if c not in master.columns or c in join_key]
        master = master.merge(edge_risk[new_cols], on=join_key, how="left")

    # ── in_both_pipelines flag ────────────────────────────────────────────────
    net_cols_added = [c for c in master.columns if c.startswith("net_")]
    if net_cols_added:
        # At least one network column non-null
        master["in_both_pipelines"] = master[net_cols_added].notna().any(axis=1)
    else:
        master["in_both_pipelines"] = False

    # ── Apply year filter if specified ────────────────────────────────────────
    master = _filter_years(master, cfg.year_start, cfg.year_end)
    master = master.sort_values(join_key).reset_index(drop=True)

    # ── Build join report ─────────────────────────────────────────────────────
    both = master["in_both_pipelines"].sum()
    report = {
        "total_country_years":   len(master),
        "in_both_pipelines":     int(both),
        "nlp_only":              int(len(master) - both),
        "unique_countries":      int(master["country_code"].nunique()),
        "year_range":            (int(master["year"].min()), int(master["year"].max()))
                                  if "year" in master.columns else None,
        "nlp_cols_added":        master_before_net - len(join_key),
        "net_cols_added":        len(net_cols_added),
        "total_columns":         len(master.columns),
    }
    logger.info(
        "Master dataset: %d rows, %d cols | %d in both pipelines | %d countries",
        report["total_country_years"], report["total_columns"],
        report["in_both_pipelines"], report["unique_countries"],
    )
    return master, report


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load everything and return master + raw dfs
# ─────────────────────────────────────────────────────────────────────────────

def load_everything(cfg: Config) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """
    Full pipeline:
      1. Build country bridge
      2. Load NLP + network
      3. Build master dataset

    Returns (master_df, join_report, nlp_dfs, net_dfs, bridge).
    """
    bridge = build_bridge(cfg.nlp_data_dir, cfg.net_src_dir)
    nlp_dfs = load_all_nlp(cfg)
    net_dfs = load_all_network(cfg, bridge)
    master, report = build_master_dataset(nlp_dfs, net_dfs, cfg)
    return master, report, nlp_dfs, net_dfs, bridge
