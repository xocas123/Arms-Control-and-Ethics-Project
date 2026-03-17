"""
I5: Regime Transition Rhetoric-Transfer Lag
────────────────────────────────────────────
After democratization or backsliding, which changes first — rhetoric or
transfer patterns?

Approach:
  1. Build ±5-year event window around each transition event
  2. Compute windowed mean of NLP metrics vs. network metrics
  3. Wilcoxon signed-rank test: pre (−3..0) vs. post (0..+3), split by direction
  4. Dual-axis event study chart
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyses.base import BaseAnalysis

logger = logging.getLogger(__name__)

WINDOW = 5  # ±5 years around transition


class RegimeTransitionLag(BaseAnalysis):
    id   = "i5"
    name = "Regime Transition Rhetoric-Transfer Lag"

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        # ── Load transition events ────────────────────────────────────────────
        trans_df = nlp_dfs.get("transitions")
        if trans_df is None or "transition_year" not in trans_df.columns:
            # Try net transitions
            trans_df = net_dfs.get("transitions")
            if trans_df is None:
                return {"key_findings": [
                    "I5 skipped: no transition_cases file found"
                ]}

        # Unify country column
        country_col = next(
            (c for c in ["country_code", "country_iso3", "country"]
             if c in trans_df.columns), None
        )
        if country_col is None:
            return {"key_findings": ["I5 skipped: no country column in transitions file"]}

        if country_col != "country_code":
            trans_df = trans_df.rename(columns={country_col: "country_code"})
        trans_df["country_code"] = trans_df["country_code"].astype(str).str.upper()

        # One row per country = the transition event (deduplicate to unique events)
        events = (trans_df[["country_code", "transition_year", "direction"]]
                  .drop_duplicates(subset=["country_code", "transition_year"])
                  .dropna(subset=["transition_year"]))
        events["transition_year"] = events["transition_year"].astype(int)

        if len(events) < 3:
            return {"key_findings": [
                f"I5 skipped: only {len(events)} transition events found (need ≥ 3)"
            ]}

        # ── Discover metric columns ───────────────────────────────────────────
        nlp_metrics = self.available_cols(
            master,
            ["rhetoric_score", "att_score", "treaty_anchor_similarity",
             "care_harm_loading", "voting_composite"],
            "NLP metrics"
        )
        net_metrics = self.available_cols(
            master,
            ["net_betweenness", "net_total_export_tiv", "net_mean_ethical_risk_score",
             "net_pagerank"],
            "network metrics"
        )

        if not nlp_metrics and not net_metrics:
            return {"key_findings": ["I5 skipped: no usable metric columns found"]}

        all_metrics = nlp_metrics + net_metrics

        # ── Build event windows ───────────────────────────────────────────────
        window_rows = []
        for _, ev in events.iterrows():
            cc   = ev["country_code"]
            ty   = ev["transition_year"]
            dirn = str(ev.get("direction", "unknown"))

            country_data = master[master["country_code"] == cc].copy()
            if country_data.empty:
                continue

            for _, row in country_data.iterrows():
                yr = row.get("year")
                if pd.isna(yr):
                    continue
                delta = int(yr) - ty
                if abs(delta) > WINDOW:
                    continue
                entry = {
                    "country_code":      cc,
                    "year":              int(yr),
                    "transition_year":   ty,
                    "years_to_transition": delta,
                    "direction":         dirn,
                }
                for m in all_metrics:
                    entry[m] = row.get(m, np.nan)
                window_rows.append(entry)

        if not window_rows:
            return {"key_findings": [
                "I5: no country-year data matched within ±5 years of any transition"
            ]}

        window_df = pd.DataFrame(window_rows)
        self.save_csv(window_df, "i5_transition_window_raw")

        # ── Aggregate by (years_to_transition, direction) ─────────────────────
        agg_df = (window_df
                  .groupby(["years_to_transition", "direction"])[all_metrics]
                  .mean()
                  .reset_index()
                  .sort_values(["direction", "years_to_transition"]))
        self.save_csv(agg_df, "i5_transition_window")

        # ── Wilcoxon signed-rank test ─────────────────────────────────────────
        wilcoxon_rows = []
        directions = window_df["direction"].dropna().unique()
        for dirn in directions:
            d_df = window_df[window_df["direction"] == dirn]
            pre  = d_df[d_df["years_to_transition"].between(-3, -1)]
            post = d_df[d_df["years_to_transition"].between(1, 3)]

            for m in all_metrics:
                pre_vals  = pre.groupby("country_code")[m].mean().dropna()
                post_vals = post.groupby("country_code")[m].mean().dropna()
                # Align on same countries
                common = pre_vals.index.intersection(post_vals.index)
                if len(common) < 4:
                    continue
                try:
                    stat, p = stats.wilcoxon(
                        pre_vals[common].values,
                        post_vals[common].values,
                        alternative="two-sided"
                    )
                    wilcoxon_rows.append({
                        "metric":      m,
                        "direction":   dirn,
                        "stat":        stat,
                        "p_value":     p,
                        "median_pre":  float(pre_vals[common].median()),
                        "median_post": float(post_vals[common].median()),
                        "n_countries": len(common),
                    })
                except Exception:
                    pass

        if wilcoxon_rows:
            self.save_csv(pd.DataFrame(wilcoxon_rows), "i5_wilcoxon_results")

        # ── Event study chart ─────────────────────────────────────────────────
        if self.cfg.produce_charts and nlp_metrics and net_metrics:
            directions_list = sorted(directions)
            n_dir = len(directions_list)
            fig, axes = plt.subplots(1, n_dir, figsize=(7 * n_dir, 5), squeeze=False)

            nlp_col = nlp_metrics[0]
            net_col = net_metrics[0]

            for col_i, dirn in enumerate(directions_list):
                ax1 = axes[0][col_i]
                ax2 = ax1.twinx()

                d_agg = agg_df[agg_df["direction"] == dirn].sort_values(
                    "years_to_transition")
                xs = d_agg["years_to_transition"].values

                if nlp_col in d_agg.columns:
                    ax1.plot(xs, d_agg[nlp_col].values,
                             color="#1565C0", linewidth=2, marker="o", label=nlp_col)
                    ax1.set_ylabel(nlp_col, color="#1565C0")
                    ax1.tick_params(axis="y", labelcolor="#1565C0")

                if net_col in d_agg.columns:
                    ax2.plot(xs, d_agg[net_col].values,
                             color="#C62828", linewidth=2, marker="s",
                             linestyle="--", label=net_col)
                    ax2.set_ylabel(net_col, color="#C62828")
                    ax2.tick_params(axis="y", labelcolor="#C62828")

                ax1.axvline(0, color="black", linewidth=1.2, linestyle=":")
                ax1.set_xlabel("Years to transition")
                ax1.set_title(f"Direction: {dirn}")
                # Combined legend
                lines1, labs1 = ax1.get_legend_handles_labels()
                lines2, labs2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

            fig.suptitle("I5: Rhetoric vs. Transfer Behavior Around Regime Transitions\n"
                         "(vertical line = transition year)", y=1.02)
            self.save_chart(fig, "i5_transition_event_study")

        # ── Key findings ──────────────────────────────────────────────────────
        findings = [
            f"Transition events analyzed: {len(events)} "
            f"({events['direction'].value_counts().to_dict()})",
            f"NLP metrics: {nlp_metrics}",
            f"Network metrics: {net_metrics}",
            f"Window rows constructed: {len(window_df)}",
        ]
        if wilcoxon_rows:
            sig = [r for r in wilcoxon_rows if r["p_value"] < 0.05]
            findings.append(
                f"Wilcoxon significant pre→post changes (p<0.05): "
                f"{[r['metric'] + '/' + r['direction'] for r in sig]}"
            )
        return {"key_findings": findings, "n_events": len(events)}
