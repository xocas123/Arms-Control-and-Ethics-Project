"""
I6: P5 Hypocrisy Profile
─────────────────────────
Are P5 countries more or less hypocritical across multiple NLP dimensions
than other major exporters?

Approach:
  1. Radar/spider chart: P5 vs Major non-P5 vs Others on 6 NLP dimensions
  2. Box plot facet: 3 groups × dimensions
  3. Kruskal-Wallis test per dimension
  4. Mixed-LM: gap ~ is_p5 + net_betweenness + year + (1|country_code)
  5. Rolling 3-year mean gap by group over time
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analyses.base import BaseAnalysis
from viz.style import PALETTE

logger = logging.getLogger(__name__)

_GROUP_COLORS = {
    "P5":              PALETTE["p5"],
    "Major non-P5":    PALETTE["major_non_p5"],
    "Other":           PALETTE["other"],
}


def _assign_group(country_code: str, p5: list, major: list) -> str:
    if country_code in p5:
        return "P5"
    if country_code in major:
        return "Major non-P5"
    return "Other"


def _radar_chart(group_means: pd.DataFrame, dimensions: list[str],
                 title: str) -> plt.Figure:
    """Draw a radar chart given a DataFrame with groups as index."""
    n = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for group, row in group_means.iterrows():
        values = row[dimensions].tolist()
        values += values[:1]
        color = _GROUP_COLORS.get(str(group), "#607D8B")
        ax.plot(angles, values, color=color, linewidth=2, label=str(group))
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), dimensions, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig


class P5HypocrisyProfile(BaseAnalysis):
    id   = "i6"
    name = "P5 Hypocrisy Profile"

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        df = master.copy()

        # ── Assign groups ─────────────────────────────────────────────────────
        df["group"] = df["country_code"].apply(
            lambda c: _assign_group(str(c), self.cfg.P5, self.cfg.MAJOR_EXPORTERS)
        )

        # ── Select NLP dimensions for radar ──────────────────────────────────
        candidates = self.cfg.NLP_SCORE_COLS
        radar_dims = self.available_cols(df, candidates, "radar dims")[:6]
        if len(radar_dims) < 3:
            return {"key_findings": [
                f"I6 skipped: only {len(radar_dims)} NLP dimension columns found"
            ]}

        # Gap column for temporal / regression analyses
        gap_col = self.first_available(
            df, ["gap", "net_mean_ethical_risk_score"], "gap")

        # ── 1. Group statistics ───────────────────────────────────────────────
        grp_stats = (df.groupby("group")[radar_dims]
                     .agg(["mean", "median", "std"])
                     .round(4))
        grp_stats.columns = ["_".join(c) for c in grp_stats.columns]
        grp_stats = grp_stats.reset_index()
        self.save_csv(grp_stats, "i6_group_statistics")

        # ── 2. Kruskal-Wallis test per dimension ──────────────────────────────
        kw_rows = []
        for dim in radar_dims:
            groups_data = [
                df[df["group"] == g][dim].dropna().values
                for g in ["P5", "Major non-P5", "Other"]
            ]
            groups_data = [g for g in groups_data if len(g) >= 3]
            if len(groups_data) >= 2:
                h_stat, kw_p = stats.kruskal(*groups_data)
                kw_rows.append({"dimension": dim, "H_stat": h_stat, "p_value": kw_p})
        if kw_rows:
            kw_df = pd.DataFrame(kw_rows)
            self.save_csv(kw_df, "i6_kruskal_wallis")

        # ── 3. Radar chart ────────────────────────────────────────────────────
        # Normalize each dim to [0,1] using the global min/max for comparability
        norm_df = df.copy()
        for dim in radar_dims:
            col = norm_df[dim]
            rng = col.max() - col.min()
            norm_df[dim] = (col - col.min()) / rng if rng > 0 else 0.5

        group_means = norm_df.groupby("group")[radar_dims].mean()
        if self.cfg.produce_charts and len(group_means) >= 2:
            fig = _radar_chart(group_means, radar_dims,
                               "I6: NLP Rhetoric Profile by Country Group\n(0-1 normalized)")
            self.save_chart(fig, "i6_radar_p5")

        # ── 4. Box plot facet ─────────────────────────────────────────────────
        if self.cfg.produce_charts:
            n_dims = len(radar_dims)
            ncols = min(3, n_dims)
            nrows = (n_dims + ncols - 1) // ncols
            fig2, axes = plt.subplots(nrows, ncols,
                                      figsize=(ncols * 4, nrows * 3.5))
            axes = np.array(axes).flatten()
            order = ["P5", "Major non-P5", "Other"]
            colors = [_GROUP_COLORS[g] for g in order]
            for i, dim in enumerate(radar_dims):
                sns.boxplot(
                    data=df, x="group", y=dim, hue="group",
                    order=order, palette=dict(zip(order, colors)),
                    ax=axes[i], showfliers=False, legend=False
                )
                axes[i].set_title(dim, fontsize=9)
                axes[i].set_xlabel("")
                axes[i].tick_params(axis="x", labelsize=7)
            # Hide unused axes
            for j in range(n_dims, len(axes)):
                axes[j].set_visible(False)
            fig2.suptitle("I6: NLP Dimensions by Country Group", y=1.02)
            self.save_chart(fig2, "i6_boxplot_grid")

        # ── 5. Gap time series ────────────────────────────────────────────────
        if gap_col and self.cfg.produce_charts and "year" in df.columns:
            ts = (df.groupby(["year", "group"])[gap_col]
                  .mean()
                  .reset_index()
                  .sort_values("year"))
            fig3, ax3 = plt.subplots(figsize=(11, 5))
            for group, gdf in ts.groupby("group"):
                gdf = gdf.sort_values("year")
                # Rolling 3-year mean
                rolling = gdf.set_index("year")[gap_col].rolling(3, min_periods=1).mean()
                ax3.plot(rolling.index, rolling.values,
                         color=_GROUP_COLORS.get(group, "#607D8B"),
                         linewidth=2, label=group)
            ax3.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax3.set_xlabel("Year")
            ax3.set_ylabel(f"Mean {gap_col} (3-year rolling)")
            ax3.set_title("I6: Rhetoric-Action Gap by Country Group Over Time")
            ax3.legend()
            self.save_chart(fig3, "i6_gap_time_series")

        # ── 6. Mixed-LM ──────────────────────────────────────────────────────
        if (gap_col and "country_code" in df.columns and "year" in df.columns
                and "net_betweenness" in df.columns):
            try:
                import statsmodels.formula.api as smf
                lm_df = df[[gap_col, "country_code", "year",
                            "group", "net_betweenness"]].dropna()
                lm_df["is_p5"] = (lm_df["group"] == "P5").astype(int)
                if lm_df["is_p5"].sum() >= 5 and len(lm_df) >= 50:
                    model = smf.mixedlm(
                        f"{gap_col} ~ is_p5 + net_betweenness + year",
                        data=lm_df,
                        groups=lm_df["country_code"],
                    ).fit(reml=True)
                    lm_rows = []
                    for term in ["is_p5", "net_betweenness", "year"]:
                        if term in model.params:
                            lm_rows.append({
                                "term":    term,
                                "coef":    model.params[term],
                                "p_value": model.pvalues[term],
                            })
                    self.save_csv(pd.DataFrame(lm_rows), "i6_mixedlm_results")
            except Exception as exc:
                logger.warning("I6 Mixed-LM failed: %s", exc)

        # ── Key findings ──────────────────────────────────────────────────────
        p5_n = int((df["group"] == "P5").sum())
        findings = [
            f"Groups: P5 ({p5_n} obs), Major non-P5 "
            f"({int((df['group']=='Major non-P5').sum())} obs), "
            f"Other ({int((df['group']=='Other').sum())} obs)",
            f"Radar dimensions: {radar_dims}",
        ]
        if kw_rows:
            sig_dims = [r["dimension"] for r in kw_rows if r["p_value"] < 0.05]
            findings.append(
                f"Kruskal-Wallis significant group differences (p<0.05): {sig_dims}"
            )
        return {"key_findings": findings}
