"""
analyses/base.py — BaseAnalysis class with column-check helpers and safe execution.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class BaseAnalysis:
    """
    Base class for all cross-pipeline insight modules.

    Subclasses must implement run(master, nlp_dfs, net_dfs) -> dict.
    The dict should contain at least {"key_findings": list[str]}.
    """
    id: str = "base"
    name: str = "Base Analysis"

    def __init__(self, cfg):
        self.cfg = cfg
        self.charts_dir: Path = cfg.charts_dir
        self.csvs_dir: Path   = cfg.csvs_dir

    # ── Column helpers ────────────────────────────────────────────────────────

    def available_cols(self, df: pd.DataFrame, candidates: list[str],
                       context: str = "") -> list[str]:
        """
        Return the intersection of candidates with df.columns.
        Logs any missing candidates at DEBUG level.
        """
        present = [c for c in candidates if c in df.columns]
        missing = [c for c in candidates if c not in df.columns]
        if missing:
            logger.debug("[%s] %s: columns not found: %s",
                         self.id, context or "available_cols", missing)
        return present

    def first_available(self, df: pd.DataFrame, candidates: list[str],
                        context: str = "") -> str | None:
        """Return the first candidate column that exists in df, or None."""
        for c in candidates:
            if c in df.columns:
                return c
        logger.warning("[%s] %s: none of %s found", self.id, context, candidates)
        return None

    def need_cols(self, df: pd.DataFrame, required: list[str],
                  context: str = "") -> bool:
        """
        Check that all required columns are present. Returns False and logs a
        warning if any are missing — caller should return early.
        """
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("[%s] %s: required columns missing: %s — skipping",
                           self.id, context, missing)
            return False
        return True

    # ── Output helpers ────────────────────────────────────────────────────────

    def save_csv(self, df: pd.DataFrame, name: str) -> Path:
        self.csvs_dir.mkdir(parents=True, exist_ok=True)
        path = self.csvs_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("[%s] Saved CSV: %s (%d rows)", self.id, path.name, len(df))
        return path

    def save_chart(self, fig: plt.Figure, name: str) -> Path:
        if not self.cfg.produce_charts:
            plt.close(fig)
            return self.charts_dir / f"{name}.png"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        path = self.charts_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("[%s] Saved chart: %s", self.id, path.name)
        return path

    # ── Year filter ───────────────────────────────────────────────────────────

    def filter_complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows where in_both_pipelines is True (if column exists)."""
        if "in_both_pipelines" in df.columns:
            return df[df["in_both_pipelines"]].copy()
        return df

    # ── Safe runner ───────────────────────────────────────────────────────────

    def run_safe(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        """
        Execute self.run() with full exception isolation.
        Returns a metadata dict with at minimum:
          {"insight_id", "status", "key_findings", "error"}
        """
        logger.info("── Starting %s (%s) ──", self.name, self.id)
        try:
            result = self.run(master, nlp_dfs, net_dfs)
            result.setdefault("insight_id", self.id)
            result.setdefault("status", "ok")
            result.setdefault("key_findings", [])
            result.setdefault("error", None)
            logger.info("── Completed %s ──", self.id)
            return result
        except Exception:
            tb = traceback.format_exc()
            logger.error("[%s] Failed:\n%s", self.id, tb)
            return {
                "insight_id":   self.id,
                "status":       "error",
                "key_findings": [],
                "error":        tb,
            }

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        raise NotImplementedError
