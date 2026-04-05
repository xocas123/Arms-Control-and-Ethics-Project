"""
config.py — Configurable paths and soft feature defaults for cross-analysis.

All paths are resolved relative to this file and can be overridden at runtime
via CLI arguments or by mutating the Config object. Feature lists are "soft":
each analysis module intersects them with actual DataFrame columns before use.
"""
from pathlib import Path

_HERE = Path(__file__).parent


class Config:
    # ── Pipeline output directories ──────────────────────────────────────────
    nlp_output_dir: Path = _HERE / "../arms-control-nlp/output"
    net_output_dir: Path = _HERE / "../arms-trade-network/output/metrics"
    nlp_data_dir:   Path = _HERE / "../arms-control-nlp/data/raw"
    net_src_dir:    Path = _HERE / "../arms-trade-network/src"

    # ── Cross-analysis output directory ──────────────────────────────────────
    output_dir: Path = _HERE / "output"

    # ── Year range (None = infer from data) ──────────────────────────────────
    year_start: int | None = None
    year_end:   int | None = None

    # ── Produce chart PNGs ────────────────────────────────────────────────────
    produce_charts: bool = True

    def __init__(self, **overrides):
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(self, k, Path(v) if k.endswith("_dir") else v)
        # Resolve all path attributes
        for attr in ("nlp_output_dir", "net_output_dir", "nlp_data_dir",
                     "net_src_dir", "output_dir"):
            setattr(self, attr, getattr(self, attr).resolve())

    # ── NLP file registry ─────────────────────────────────────────────────────
    # Paths are tried in order; loader skips files that don't exist.
    # Legacy paths (pre-refactor) are kept as fallbacks after the new paths.
    @property
    def nlp_files(self) -> dict[str, Path]:
        d = self.nlp_output_dir
        return {
            # New flat structure (post-refactor)
            "frame_scores":         d / "shared/frame_scores.csv",
            "transitions":          d / "q2/transition_cases.csv",
            "frame_by_regime":      d / "q2/frame_by_regime_year.csv",
            "topic_info":           d / "shared/topics/topic_info.csv",
            # Treaty similarity trajectories (one per treaty — loaded specially)
            "att_trajectories":     d / "q3/similarity_trajectories/att_trajectories.csv",
            "tpnw_trajectories":    d / "q3/similarity_trajectories/tpnw_trajectories.csv",
            "ottawa_trajectories":  d / "q3/similarity_trajectories/ottawa_trajectories.csv",
            "ccm_trajectories":     d / "q3/similarity_trajectories/ccm_trajectories.csv",
            # Legacy paths (pre-refactor, kept as fallbacks)
            "rhetoric_scores":      d / "metrics/rhetoric_scores.csv",
            "rhetoric_gap":         d / "metrics/rhetoric_action_gap.csv",
            "sentiment":            d / "metrics/sentiment_by_country_year.csv",
            "mft":                  d / "metrics/moral_foundations_by_country_year.csv",
            "anchor_scores":        d / "embeddings/anchor_scores.csv",
            "semantic_drift":       d / "embeddings/semantic_drift.csv",
            "topics":               d / "topics/lda_topic_proportions.csv",
            "voting":               d / "metrics/voting_data.csv",
            "transitions_legacy":   d / "q2/q2/transition_cases.csv",
        }

    # ── Network file registry ─────────────────────────────────────────────────
    @property
    def net_files(self) -> dict[str, Path]:
        d = self.net_output_dir
        return {
            "node_metrics": d / "node_metrics.csv",
            "edge_metrics": d / "edge_metrics.csv",
            "communities":  d / "communities.csv",
            "complicity":   d / "deep_analysis/exporter_complicity_alltime.csv",
            "transitions":  d / "deep_analysis/regime_transition_tracking.csv",
        }

    # ── Output subdirectories ─────────────────────────────────────────────────
    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def csvs_dir(self) -> Path:
        return self.output_dir / "insight_csvs"

    # ── Soft feature lists (intersected with actual columns at runtime) ────────
    NLP_JOIN_KEY = ["country_code", "year"]

    # Candidate NLP predictor columns (order = preference).
    # New pipeline produces frame_ratio_mean + treaty similarities.
    # Legacy pipeline produced rhetoric_score, care_harm, att_score, etc.
    # The loader normalises new columns to legacy names where possible.
    NLP_SCORE_COLS = [
        # New names (post-refactor)
        "frame_ratio_mean", "frame_position_mean",
        "att_similarity", "tpnw_similarity", "ottawa_similarity", "ccm_similarity",
        # Legacy names (pre-refactor, still used if files exist)
        "treaty_anchor_similarity", "voting_score", "humanitarian_topic_pct",
        "commitment_strength", "care_harm_loading", "rhetoric_score",
        "att_score", "compound", "care_harm", "fairness",
    ]

    # Candidate network risk columns (order = preference)
    NET_RISK_COLS = [
        "net_mean_ethical_risk_score", "net_pct_ethical",
        "net_n_att_concern", "net_n_into_conflict", "net_n_embargo_violations",
    ]

    # Moral Foundations Theory columns
    MFT_COLS = ["care_harm", "fairness", "loyalty", "authority", "sanctity"]

    # Violation columns (raw, from edge_metrics aggregation)
    VIOLATION_COLS = [
        "net_n_embargo_violations", "net_n_into_conflict",
        "net_n_att_concern", "net_mean_ethical_risk_score",
    ]

    # Country groups
    P5             = ["USA", "RUS", "CHN", "GBR", "FRA"]
    MAJOR_EXPORTERS = ["DEU", "ITA", "ESP", "ISR", "KOR"]
