"""
Q6 Visualizations: Drones and Autonomous Weapons.

Plots:
- q6_drone_rhetoric_heatmap.png: Country-year drone mention intensity
- q6_rhetoric_vs_transfers.png: Rhetoric-action scatter
- q6_att_drone_adoption_curve.png: Drone export trajectory around ATT ratification
- q6_summary_dashboard.png: 4-panel summary
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.groups import ATT_PARTIES

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

OUTPUT_DIR = "output/q6/plots"


def _savefig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Q6 Plot] Saved {path}")


def plot_drone_rhetoric_heatmap(drone_keywords: pd.DataFrame):
    """Country-year heatmap of drone mentions (top 30 countries)."""
    if drone_keywords.empty:
        return
    # Top 30 countries by total mentions
    top = drone_keywords.groupby("country_iso3")["drone_mentions"].sum().nlargest(30).index
    sub = drone_keywords[drone_keywords["country_iso3"].isin(top)].copy()
    pivot = sub.pivot_table(index="country_iso3", columns="year", values="drone_mentions", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    years = pivot.columns.astype(int)
    tick_positions = list(range(0, len(years), 5))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([years[i] for i in tick_positions], fontsize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Country")
    ax.set_title("Drone/Autonomous Weapons Mentions in UN General Debate Speeches")
    # ATT adoption line
    if 2013 in list(years):
        att_idx = list(years).index(2013)
        ax.axvline(x=att_idx, color="blue", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(att_idx + 0.3, -0.8, "ATT (2013)", color="blue", fontsize=8)
    fig.colorbar(im, ax=ax, label="Drone mentions", shrink=0.6)
    _savefig(fig, "q6_drone_rhetoric_heatmap.png")


def plot_rhetoric_vs_transfers(gap: pd.DataFrame):
    """Scatter: drone rhetoric vs drone export TIV, colored by ATT status."""
    if gap.empty or "drone_export_tiv" not in gap.columns:
        return

    # Aggregate to country level
    country = gap.groupby("country_iso3").agg(
        drone_mentions=("drone_mentions", "sum") if "drone_mentions" in gap.columns else ("drone_export_tiv", "count"),
        drone_export_tiv=("drone_export_tiv", "sum"),
        att_status=("att_status", "last"),
    ).reset_index()
    country = country[country["drone_export_tiv"] > 0]
    if country.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"party": "#27ae60", "signatory": "#f39c12", "non-member": "#e74c3c", "opponent": "#8e44ad"}
    for status, color in colors.items():
        sub = country[country["att_status"] == status]
        if sub.empty:
            continue
        ax.scatter(sub["drone_mentions"], sub["drone_export_tiv"],
                   c=color, label=f"ATT {status}", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)
        # Label top exporters
        for _, row in sub.nlargest(3, "drone_export_tiv").iterrows():
            ax.annotate(row["country_iso3"], (row["drone_mentions"], row["drone_export_tiv"]),
                        fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Total Drone Mentions in UNGDC Speeches")
    ax.set_ylabel("Total Drone Export TIV (SIPRI)")
    ax.set_title("Drone Rhetoric vs. Actual Drone Exports")
    ax.legend(fontsize=9)
    _savefig(fig, "q6_rhetoric_vs_transfers.png")


def plot_att_drone_adoption_curve(att_drone: pd.DataFrame):
    """Drone export TIV trajectory relative to ATT ratification."""
    if att_drone.empty or "year_relative" not in att_drone.columns:
        return

    parties = att_drone[att_drone["group"] == "ATT party"].dropna(subset=["year_relative"])
    non = att_drone[att_drone["group"] == "Non-party"]

    if parties.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Parties: aggregate by year_relative
    party_curve = parties.groupby("year_relative")["drone_export_tiv"].mean()
    party_curve = party_curve[(party_curve.index >= -10) & (party_curve.index <= 10)]
    if not party_curve.empty:
        ax.plot(party_curve.index, party_curve.values, color="#27ae60", linewidth=2, label="ATT parties")

    # Non-parties: aggregate by year (use as flat comparison)
    if not non.empty:
        non_mean = non.groupby("year")["drone_export_tiv"].mean()
        ax.axhline(y=non_mean.mean(), color="#e74c3c", linewidth=1.5, linestyle="--",
                    label=f"Non-parties avg ({non_mean.mean():.1f})")

    ax.axvline(x=0, color="gray", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(0.3, ax.get_ylim()[1] * 0.95, "ATT ratification", fontsize=9, color="gray")
    ax.set_xlabel("Years Relative to ATT Ratification")
    ax.set_ylabel("Mean Drone Export TIV")
    ax.set_title("Drone Exports Before and After ATT Ratification")
    ax.legend()
    _savefig(fig, "q6_att_drone_adoption_curve.png")


def plot_summary_dashboard(
    drone_keywords: pd.DataFrame,
    drone_trajectory: pd.DataFrame,
    drone_transfers: pd.DataFrame,
    laws_voting: pd.DataFrame,
):
    """2x2 summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Global drone mentions over time
    ax = axes[0, 0]
    if not drone_keywords.empty:
        yearly = drone_keywords.groupby("year")["drone_mentions"].sum()
        ax.bar(yearly.index, yearly.values, color="#3498db", alpha=0.8)
        ax.axvline(x=2013, color="red", linewidth=1, linestyle="--", alpha=0.7)
        ax.text(2013.3, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1, "ATT", color="red", fontsize=8)
    ax.set_title("Global Drone/LAWS Mentions Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total mentions")

    # Top-right: LAWS similarity by group (if trajectory available)
    ax = axes[0, 1]
    if not drone_trajectory.empty and "laws_drones" in drone_trajectory.columns:
        from src.data.groups import NWS, NATO, NAM
        traj = drone_trajectory.copy()
        traj["group"] = traj["country_iso3"].apply(
            lambda c: "P5" if c in NWS else ("NATO" if c in NATO else ("NAM" if c in NAM else "Other"))
        )
        for grp, color in [("P5", "#e74c3c"), ("NATO", "#3498db"), ("NAM", "#2ecc71"), ("Other", "#95a5a6")]:
            sub = traj[traj["group"] == grp].groupby("year")["laws_drones"].mean()
            if not sub.empty:
                ax.plot(sub.index, sub.values, label=grp, color=color, linewidth=1.5)
        ax.legend(fontsize=8)
    ax.set_title("LAWS Anchor Similarity by Group")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine similarity")

    # Bottom-left: Top drone exporters
    ax = axes[1, 0]
    if not drone_transfers.empty:
        exp = drone_transfers[drone_transfers["role"] == "exporter"]
        top_exp = exp.groupby("country_iso3")["tiv"].sum().nlargest(15)
        colors = ["#27ae60" if c in ATT_PARTIES else "#e74c3c" for c in top_exp.index]
        ax.barh(range(len(top_exp)), top_exp.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_exp)))
        ax.set_yticklabels(top_exp.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Total Drone Export TIV")
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#27ae60", label="ATT party"),
                           Patch(color="#e74c3c", label="Non-party")], fontsize=8)
    ax.set_title("Top Drone Exporters (SIPRI)")

    # Bottom-right: LAWS vote distribution
    ax = axes[1, 1]
    if not laws_voting.empty:
        laws_only = laws_voting[laws_voting["treaty_flag"] == "laws"]
        if not laws_only.empty:
            vote_dist = laws_only.groupby("country_iso3")["pct_yes"].mean()
            bins = [0, 0.25, 0.5, 0.75, 1.01]
            labels = ["No/Abstain\n(0-25%)", "Lean No\n(25-50%)", "Lean Yes\n(50-75%)", "Yes\n(75-100%)"]
            counts = pd.cut(vote_dist, bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
            ax.bar(counts.index, counts.values, color=["#e74c3c", "#f39c12", "#3498db", "#27ae60"], alpha=0.8)
            ax.set_ylabel("Number of countries")
        else:
            ax.text(0.5, 0.5, "No LAWS-specific votes found", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No voting data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("LAWS Resolution Voting Distribution")

    fig.suptitle("Q6: Drones & Autonomous Weapons -- Summary Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "q6_summary_dashboard.png")


def run_q6_plots(results: dict):
    """Generate all Q6 visualizations."""
    print("[Q6 Plots] Generating visualizations...")

    plot_drone_rhetoric_heatmap(results.get("drone_keywords", pd.DataFrame()))
    plot_rhetoric_vs_transfers(results.get("rhetoric_transfer_gap", pd.DataFrame()))
    plot_att_drone_adoption_curve(results.get("att_drone_behavior", pd.DataFrame()))
    plot_summary_dashboard(
        results.get("drone_keywords", pd.DataFrame()),
        results.get("drone_trajectory", pd.DataFrame()),
        results.get("drone_transfers", pd.DataFrame()),
        results.get("laws_voting", pd.DataFrame()),
    )

    print("[Q6 Plots] Done.")
