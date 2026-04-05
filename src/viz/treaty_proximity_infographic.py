"""
Treaty Proximity Infographic
=============================
Visualises how rhetorically close major UN member states are to key arms-control
treaties across three epochs (Cold War, Post-Cold War, Contemporary).

Output: output/infographic/treaty_proximity_infographic.png

Usage (standalone — reads from output/shared/ checkpoints):
    python -m src.viz.treaty_proximity_infographic

Usage (from run.py results dict):
    from src.viz.treaty_proximity_infographic import run_infographic
    run_infographic(pipeline_data)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output/infographic"

# ── Countries shown in the infographic ────────────────────────────────────────
FEATURED_COUNTRIES = {
    # P5 / NPT nuclear weapon states
    "USA": "United States",
    "RUS": "Russia",
    "CHN": "China",
    "GBR": "United Kingdom",
    "FRA": "France",
    # De facto nuclear states
    "IND": "India",
    "PAK": "Pakistan",
    "ISR": "Israel",
    "PRK": "North Korea",
    # Major NNWS / treaty champions
    "DEU": "Germany",
    "JPN": "Japan",
    "BRA": "Brazil",
    "ZAF": "South Africa",
    "MEX": "Mexico",
    "AUS": "Australia",
    "CAN": "Canada",
    "SWE": "Sweden",
    "NOR": "Norway",
    # Non-aligned / key voices
    "IRN": "Iran",
    "SAU": "Saudi Arabia",
    "EGY": "Egypt",
    "NGA": "Nigeria",
    "IDN": "Indonesia",
}

# ── Treaties shown and their display labels ────────────────────────────────────
FEATURED_TREATIES = {
    # Nuclear multilateral
    "npt_1968":    "NPT\n(1968)",
    "tpnw_2017":   "TPNW\n(2017)",
    "ctbt_1996":   "CTBT\n(1996)",
    "ptbt_1963":   "PTBT\n(1963)",
    # WMD conventions
    "bwc_1972":    "BWC\n(1972)",
    "cwc_1993":    "CWC\n(1993)",
    # Conventional / humanitarian
    "ottawa_1997": "Ottawa\n(1997)",
    "ccm_2008":    "CCM\n(2008)",
    "att_2013":    "ATT\n(2013)",
    # US–Soviet / bilateral arms control (rhetorical dimension)
    "inf_1987":    "INF\n(1987)",
    "start_i_1991":"START I\n(1991)",
    "new_start_2010": "New START\n(2010)",
}

# ── Epoch definitions ──────────────────────────────────────────────────────────
EPOCHS = {
    "Cold War\n1970–1989":      (1970, 1989),
    "Post-Cold War\n1990–2009": (1990, 2009),
    "Contemporary\n2010–2023":  (2010, 2023),
}

# ── Colour palette ─────────────────────────────────────────────────────────────
CMAP = LinearSegmentedColormap.from_list(
    "treaty_proximity",
    ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#f5a623", "#f9f9f9"],
    N=256,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_proximity_matrix(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    countries: dict,
    treaties: dict,
    year_range: tuple,
) -> pd.DataFrame:
    """
    Returns DataFrame (countries × treaties) of mean cosine similarity
    for speeches in [year_range[0], year_range[1]].
    """
    cy = country_year_embeddings
    if cy.empty:
        return pd.DataFrame()

    mask = (cy["year"] >= year_range[0]) & (cy["year"] <= year_range[1])
    cy = cy[mask]

    rows = {}
    for iso3, label in countries.items():
        cdf = cy[cy["country_iso3"] == iso3]
        if cdf.empty:
            rows[label] = {FEATURED_TREATIES[t]: np.nan for t in treaties}
            continue

        # Country centroid for this epoch
        embs = [e for e in cdf["embedding"]
                if e is not None and hasattr(e, "__len__") and len(e) > 0]
        if not embs:
            rows[label] = {FEATURED_TREATIES[t]: np.nan for t in treaties}
            continue

        embs_arr = [np.array(e, dtype=float) if not isinstance(e, np.ndarray) else e
                    for e in embs]
        centroid = np.mean(embs_arr, axis=0)

        row = {}
        for treaty_key, treaty_label in treaties.items():
            anchor = anchor_embeddings.get(treaty_key, {})
            if isinstance(anchor, dict):
                vec = anchor.get("mean_embedding")
            else:
                vec = None
            if vec is None:
                row[treaty_label] = np.nan
            else:
                row[treaty_label] = _cosine_sim(centroid, np.array(vec, dtype=float))
        rows[label] = row

    return pd.DataFrame(rows).T  # countries × treaties


def compute_temporal_similarity(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    countries: dict,
    treaty_keys: list,
) -> dict:
    """
    Returns {treaty_key: DataFrame(year × country_label)} of yearly similarity.
    """
    cy = country_year_embeddings
    result = {}

    for treaty_key in treaty_keys:
        anchor = anchor_embeddings.get(treaty_key, {})
        vec = anchor.get("mean_embedding") if isinstance(anchor, dict) else None
        if vec is None:
            continue
        vec = np.array(vec, dtype=float)

        records = {}
        for iso3, label in countries.items():
            cdf = cy[cy["country_iso3"] == iso3].sort_values("year")
            if cdf.empty:
                continue
            sims = {}
            for _, row in cdf.iterrows():
                emb = row["embedding"]
                if emb is None:
                    continue
                if not isinstance(emb, np.ndarray):
                    try:
                        emb = np.array(emb, dtype=float)
                    except Exception:
                        continue
                if len(emb) == 0:
                    continue
                sims[int(row["year"])] = _cosine_sim(emb, vec)
            if sims:
                records[label] = sims

        if records:
            result[treaty_key] = pd.DataFrame(records)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap(ax, data: pd.DataFrame, title: str, vmin: float, vmax: float,
             show_xticklabels: bool = True, show_yticklabels: bool = True,
             cmap=CMAP):
    im = ax.imshow(data.values, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(data.columns)))
    if show_xticklabels:
        ax.set_xticklabels(data.columns, fontsize=7.5, rotation=0, ha="center")
    else:
        ax.set_xticklabels([])

    ax.set_yticks(range(len(data.index)))
    if show_yticklabels:
        ax.set_yticklabels(data.index, fontsize=8)
    else:
        ax.set_yticklabels([])

    # Annotate cells
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if val < (vmin + (vmax - vmin) * 0.65) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color=text_color, fontweight="bold")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _group_label(iso3: str) -> str:
    from src.data.groups import NWS, DE_FACTO_NUCLEAR, NUCLEAR_UMBRELLA
    if iso3 in NWS:
        return "NWS"
    if iso3 in DE_FACTO_NUCLEAR:
        return "De Facto"
    if iso3 in NUCLEAR_UMBRELLA:
        return "Umbrella"
    return "NNWS"


def make_infographic(
    country_year_embeddings: pd.DataFrame,
    anchor_embeddings: dict,
    output_dir: str = OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Compute epoch proximity matrices ────────────────────────────────────
    epoch_matrices = {}
    for epoch_label, year_range in EPOCHS.items():
        mat = compute_proximity_matrix(
            country_year_embeddings, anchor_embeddings,
            FEATURED_COUNTRIES, FEATURED_TREATIES, year_range,
        )
        epoch_matrices[epoch_label] = mat

    # Colour scale: shared across all epoch panels
    all_vals = np.concatenate([m.values.flatten() for m in epoch_matrices.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = float(np.percentile(all_vals, 5))  if len(all_vals) else 0.3
    vmax = float(np.percentile(all_vals, 95)) if len(all_vals) else 0.8

    # ── 2. Temporal lines: NPT vs TPNW for P5 + de facto ──────────────────────
    temporal_countries = {
        "USA": "United States", "RUS": "Russia", "CHN": "China",
        "GBR": "United Kingdom", "FRA": "France",
        "IND": "India", "PAK": "Pakistan", "PRK": "North Korea",
    }
    temporal_data = compute_temporal_similarity(
        country_year_embeddings, anchor_embeddings,
        temporal_countries, ["npt_1968", "tpnw_2017"],
    )

    # ── 3. Layout ──────────────────────────────────────────────────────────────
    n_countries = len(FEATURED_COUNTRIES)
    n_treaties = len(FEATURED_TREATIES)

    fig = plt.figure(figsize=(26, 22), facecolor="#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        left=0.08, right=0.97,
        top=0.93, bottom=0.05,
        hspace=0.45, wspace=0.08,
        width_ratios=[1, 1, 1, 0.04],
        height_ratios=[2, 0.04, 1.1],
    )

    # ── 3a. Three epoch heatmaps (top row) ────────────────────────────────────
    axes_heat = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cb_ax = fig.add_subplot(gs[0, 3])

    for i, (epoch_label, mat) in enumerate(epoch_matrices.items()):
        if mat.empty:
            axes_heat[i].set_visible(False)
            continue
        # Align to canonical order
        mat = mat.reindex(
            index=list(FEATURED_COUNTRIES.values()),
            columns=list(FEATURED_TREATIES.values()),
        )
        # Add group separators via row color background
        im = _heatmap(
            axes_heat[i], mat, epoch_label,
            vmin=vmin, vmax=vmax,
            show_xticklabels=True,
            show_yticklabels=(i == 0),
            cmap=CMAP,
        )

        # Draw separator lines between nuclear groups
        group_order = [_group_label(iso3) for iso3 in FEATURED_COUNTRIES]
        prev = group_order[0]
        for row_i, g in enumerate(group_order[1:], start=1):
            if g != prev:
                axes_heat[i].axhline(row_i - 0.5, color="#0d0d0d", linewidth=2)
                prev = g

        axes_heat[i].set_facecolor("#0d0d0d")

    # Shared colour bar
    cb = fig.colorbar(im, cax=cb_ax)
    cb.set_label("Cosine Similarity to Treaty Anchor", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.outline.set_edgecolor("white")

    # Group labels on left side of first heatmap
    group_boundaries = {"NWS": 0, "De Facto": 5, "Umbrella": 9, "NNWS": 13}
    for group, start_row in group_boundaries.items():
        axes_heat[0].text(
            -0.35, start_row, group,
            transform=axes_heat[0].get_yaxis_transform(),
            fontsize=8, color="#aaaaaa", rotation=0, va="center",
            ha="right", style="italic",
        )

    # ── 3b. Temporal similarity lines (bottom row) ────────────────────────────
    TEMPORAL_COLORS = {
        "United States": "#e74c3c", "Russia": "#3498db",
        "China": "#f39c12", "United Kingdom": "#9b59b6",
        "France": "#1abc9c", "India": "#e67e22",
        "Pakistan": "#2ecc71", "North Korea": "#e91e63",
    }
    LINESTYLES = {
        "United States": "-", "Russia": "-", "China": "-",
        "United Kingdom": "-", "France": "-",
        "India": "--", "Pakistan": "--", "North Korea": "--",
    }

    treaty_display = {"npt_1968": "NPT (1968)", "tpnw_2017": "TPNW (2017)"}
    ax_bottom = [fig.add_subplot(gs[2, i]) for i in range(3)]

    # Panel 1: NPT similarity over time
    ax_npt = ax_bottom[0]
    if "npt_1968" in temporal_data:
        df = temporal_data["npt_1968"].sort_index()
        for country_label in df.columns:
            if df[country_label].notna().sum() < 3:
                continue
            s = df[country_label].dropna()
            ax_npt.plot(s.index, s.values, linewidth=1.8,
                        color=TEMPORAL_COLORS.get(country_label, "#888888"),
                        linestyle=LINESTYLES.get(country_label, "-"),
                        label=country_label, alpha=0.9)
    ax_npt.set_title("Rhetorical Proximity to NPT Over Time", color="white", fontsize=9, fontweight="bold")
    ax_npt.set_xlabel("Year", color="#aaaaaa", fontsize=8)
    ax_npt.set_ylabel("Cosine Similarity", color="#aaaaaa", fontsize=8)
    ax_npt.tick_params(colors="#aaaaaa", labelsize=7)
    ax_npt.set_facecolor("#111111")
    for spine in ax_npt.spines.values():
        spine.set_edgecolor("#333333")
    ax_npt.axvline(1968, color="white", alpha=0.3, linewidth=0.8, linestyle=":")
    ax_npt.text(1968.5, ax_npt.get_ylim()[0] if ax_npt.lines else 0.3,
                "NPT opens", color="#888888", fontsize=6.5, va="bottom")

    # Panel 2: TPNW similarity over time
    ax_tpnw = ax_bottom[1]
    if "tpnw_2017" in temporal_data:
        df = temporal_data["tpnw_2017"].sort_index()
        for country_label in df.columns:
            if df[country_label].notna().sum() < 3:
                continue
            s = df[country_label].dropna()
            ax_tpnw.plot(s.index, s.values, linewidth=1.8,
                         color=TEMPORAL_COLORS.get(country_label, "#888888"),
                         linestyle=LINESTYLES.get(country_label, "-"),
                         label=country_label, alpha=0.9)
    ax_tpnw.set_title("Rhetorical Proximity to TPNW Over Time", color="white", fontsize=9, fontweight="bold")
    ax_tpnw.set_xlabel("Year", color="#aaaaaa", fontsize=8)
    ax_tpnw.tick_params(colors="#aaaaaa", labelsize=7)
    ax_tpnw.set_facecolor("#111111")
    for spine in ax_tpnw.spines.values():
        spine.set_edgecolor("#333333")
    ax_tpnw.axvline(2017, color="white", alpha=0.3, linewidth=0.8, linestyle=":")
    ax_tpnw.axvline(2021, color="white", alpha=0.2, linewidth=0.8, linestyle="--")
    ax_tpnw.set_yticklabels([])

    # Panel 3: NPT–TPNW divergence (NPT sim minus TPNW sim) for selected countries
    ax_div = ax_bottom[2]
    if "npt_1968" in temporal_data and "tpnw_2017" in temporal_data:
        npt_df = temporal_data["npt_1968"].sort_index()
        tpnw_df = temporal_data["tpnw_2017"].sort_index()
        common_years = npt_df.index.intersection(tpnw_df.index)
        for country_label in list(TEMPORAL_COLORS.keys())[:6]:
            if country_label not in npt_df.columns or country_label not in tpnw_df.columns:
                continue
            divergence = npt_df.loc[common_years, country_label] - tpnw_df.loc[common_years, country_label]
            divergence = divergence.dropna()
            if len(divergence) < 3:
                continue
            ax_div.plot(divergence.index, divergence.values, linewidth=1.8,
                        color=TEMPORAL_COLORS[country_label],
                        linestyle=LINESTYLES.get(country_label, "-"),
                        label=country_label, alpha=0.9)
    ax_div.axhline(0, color="white", linewidth=0.8, alpha=0.4, linestyle=":")
    ax_div.set_title("NPT–TPNW Divergence\n(positive = closer to NPT than TPNW)", color="white", fontsize=9, fontweight="bold")
    ax_div.set_xlabel("Year", color="#aaaaaa", fontsize=8)
    ax_div.tick_params(colors="#aaaaaa", labelsize=7)
    ax_div.set_facecolor("#111111")
    for spine in ax_div.spines.values():
        spine.set_edgecolor("#333333")
    ax_div.set_yticklabels([])

    # Shared legend for temporal panels
    handles, labels = ax_npt.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_tpnw.get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles, labels,
            loc="lower center", ncol=8, fontsize=7.5,
            facecolor="#111111", edgecolor="#333333",
            labelcolor="white",
            bbox_to_anchor=(0.5, 0.01),
            framealpha=0.8,
        )

    # ── 4. Title and annotations ───────────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        "Arms Control Treaty Proximity — How Close Is Each Country's UN Rhetoric to Major Treaties?",
        ha="center", va="top", fontsize=15, fontweight="bold", color="white",
    )
    fig.text(
        0.5, 0.948,
        "Cosine similarity between country-year speech embeddings (all-mpnet-base-v2) and treaty anchor passages. "
        "UNGDC 1970–2023, N≈9,000 speeches.",
        ha="center", va="top", fontsize=8.5, color="#aaaaaa",
    )

    # Epoch separator for bottom row
    for ax in ax_bottom:
        ax.set_xlim(1970, 2024)
        ax.grid(axis="y", color="#222222", linewidth=0.5, alpha=0.6)
        ax.grid(axis="x", color="#222222", linewidth=0.5, alpha=0.4)

    path = os.path.join(output_dir, "treaty_proximity_infographic.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Infographic] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_infographic(pipeline_data: dict, output_dir: str = OUTPUT_DIR) -> str:
    """Called from run.py with the shared pipeline data dict."""
    cy = pipeline_data.get("country_year_embeddings", pd.DataFrame())
    ae = pipeline_data.get("anchor_embeddings", {})
    if cy.empty or not ae:
        print("[Infographic] Missing embeddings — skipping.")
        return ""
    return make_infographic(cy, ae, output_dir=output_dir)


def _load_from_disk() -> tuple:
    """Load embeddings from checkpoint files for standalone use."""
    import pickle

    cy_path = Path("output/shared/checkpoints/country_year_embeddings.parquet")
    ae_path = Path("output/shared/anchor_embeddings.npz")

    if not cy_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cy_path}")
    if not ae_path.exists():
        raise FileNotFoundError(f"Anchor embeddings not found: {ae_path}")

    cy = pd.read_parquet(cy_path)

    data = np.load(ae_path, allow_pickle=True)
    ae = data["anchor_embeddings"].tolist()  # 0-d object array → dict

    return cy, ae


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate treaty proximity infographic")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    print("[Infographic] Loading embeddings from disk...")
    cy, ae = _load_from_disk()
    print(f"[Infographic] Loaded {len(cy)} country-year rows, {len(ae)} treaty anchors")
    make_infographic(cy, ae, output_dir=args.output_dir)
