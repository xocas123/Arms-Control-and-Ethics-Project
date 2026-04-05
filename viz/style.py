"""
viz/style.py — Shared matplotlib/seaborn theme for publication-ready charts.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Color palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "p5":               "#C62828",
    "major_non_p5":     "#1565C0",
    "other":            "#558B2F",
    "hypocrite":        "#BF360C",
    "aligned":          "#1B5E20",
    "quiet_good_actor": "#0D47A1",
    "community_0":      "#7B1FA2",
    "community_1":      "#F57F17",
    "community_2":      "#006064",
    "community_3":      "#37474F",
    "community_4":      "#AD1457",
    "community_5":      "#1A237E",
    "neutral":          "#607D8B",
    "positive":         "#2E7D32",
    "negative":         "#B71C1C",
}

# Community colors as a cycling list
COMMUNITY_COLORS = [
    PALETTE["community_0"], PALETTE["community_1"], PALETTE["community_2"],
    PALETTE["community_3"], PALETTE["community_4"], PALETTE["community_5"],
    "#E65100", "#004D40", "#880E4F", "#263238",
]

FIG_SINGLE  = (8, 6)
FIG_WIDE    = (12, 6)
FIG_GRID    = (14, 10)
FIG_SQUARE  = (7, 7)
DPI         = 300


def apply_style() -> None:
    """Apply global matplotlib/seaborn style. Call once at module import."""
    mpl.rcParams.update({
        "figure.dpi":           100,
        "savefig.dpi":          DPI,
        "font.family":          "DejaVu Sans",
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "figure.constrained_layout.use": True,
    })
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)


apply_style()


def community_color(community_id) -> str:
    """Return a consistent color for a community ID (int or string)."""
    try:
        idx = int(community_id) % len(COMMUNITY_COLORS)
    except (ValueError, TypeError):
        idx = hash(str(community_id)) % len(COMMUNITY_COLORS)
    return COMMUNITY_COLORS[idx]


def group_color(group: str) -> str:
    return PALETTE.get(group.lower(), PALETTE["neutral"])


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
