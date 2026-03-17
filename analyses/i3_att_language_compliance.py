"""
I3: ATT Language vs. ATT Compliance
─────────────────────────────────────
Do countries scoring high on ATT anchor language actually export fewer
ATT-concern transfers?

Approach:
  1. Scatter: att_score (NLP) vs. pct_att_concern (network)
  2. Mann-Whitney U: high vs. low att_score quartile
  3. OLS: att_concern ~ att_score + year [+ att_status]
  4. Robustness: tpnw_score vs. n_into_conflict
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analyses.base import BaseAnalysis

logger = logging.getLogger(__name__)


class AttLanguageCompliance(BaseAnalysis):
    id   = "i3"
    name = "ATT Language vs. Compliance"

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        df = self.filter_complete(master).copy()

        # ── Column discovery ──────────────────────────────────────────────────
        lang_col = self.first_available(
            df, ["att_score", "treaty_anchor_similarity"], "ATT language")
        compliance_col = self.first_available(
            df,
            ["net_n_att_concern", "net_pct_ethical", "net_mean_ethical_risk_score"],
            "ATT compliance"
        )
        if lang_col is None or compliance_col is None:
            return {"key_findings": [
                f"I3 skipped: lang_col={lang_col}, compliance_col={compliance_col}"
            ]}

        att_status_col = "net_att_status" if "net_att_status" in df.columns else None

        analysis_df = df[[lang_col, compliance_col, "country_code", "year"]
                         + ([att_status_col] if att_status_col else [])].dropna(
            subset=[lang_col, compliance_col]
        )
        if len(analysis_df) < 20:
            return {"key_findings": [
                f"I3 skipped: only {len(analysis_df)} complete rows"
            ]}

        findings = []

        # ── 1. Mann-Whitney U ─────────────────────────────────────────────────
        q75 = analysis_df[lang_col].quantile(0.75)
        q25 = analysis_df[lang_col].quantile(0.25)
        high_lang = analysis_df[analysis_df[lang_col] >= q75][compliance_col]
        low_lang  = analysis_df[analysis_df[lang_col] <= q25][compliance_col]
        if len(high_lang) >= 5 and len(low_lang) >= 5:
            u_stat, mw_p = stats.mannwhitneyu(high_lang, low_lang, alternative="less")
            findings.append(
                f"Mann-Whitney U (high vs low {lang_col} quartile on {compliance_col}): "
                f"U={u_stat:.0f}, p={mw_p:.4f}"
            )
            findings.append(
                f"  Median {compliance_col}: high-lang={high_lang.median():.3f}, "
                f"low-lang={low_lang.median():.3f}"
            )

        # ── 2. OLS regression ─────────────────────────────────────────────────
        ols_result = None
        try:
            import statsmodels.formula.api as smf
            formula = f"{compliance_col} ~ {lang_col} + year"
            if att_status_col:
                analysis_df["is_att_member"] = (
                    analysis_df[att_status_col]
                    .astype(str)
                    .str.lower()
                    .isin(["member", "signatory"])
                ).astype(int)
                formula += " + is_att_member"
            model = smf.ols(formula, data=analysis_df).fit()
            coef_table = model.summary2().tables[1].reset_index()
            coef_table.columns = ["term", "coef", "std_err", "t", "p_value",
                                   "ci_low", "ci_high"]
            self.save_csv(coef_table, "i3_att_regression_table")
            findings.append(
                f"OLS {compliance_col} ~ {lang_col}: "
                f"β={model.params.get(lang_col, float('nan')):.4f}, "
                f"p={model.pvalues.get(lang_col, float('nan')):.4f}, "
                f"R²={model.rsquared:.3f}"
            )
            ols_result = model
        except Exception as exc:
            logger.warning("I3 OLS failed: %s", exc)
            findings.append(f"OLS not available: {exc}")

        # ── 3. ATT status group stats ─────────────────────────────────────────
        if att_status_col:
            grp = (analysis_df
                   .groupby(att_status_col)[[lang_col, compliance_col]]
                   .agg(["mean", "count"])
                   .round(4))
            grp.columns = ["_".join(c) for c in grp.columns]
            grp = grp.reset_index()
            self.save_csv(grp, "i3_member_vs_nonmember")

        # ── 4. Robustness: tpnw_score vs n_into_conflict ─────────────────────
        tpnw_col   = "tpnw_score" if "tpnw_score" in df.columns else None
        conflict_col = self.first_available(
            df, ["net_n_into_conflict", "net_n_into_war"], "conflict compliance")
        if tpnw_col and conflict_col:
            rob_df = df[[tpnw_col, conflict_col]].dropna()
            if len(rob_df) >= 20:
                r, p = stats.pearsonr(rob_df[tpnw_col], rob_df[conflict_col])
                findings.append(
                    f"Robustness — tpnw_score vs {conflict_col}: r={r:.3f}, p={p:.4f}"
                )

        # ── 5. Scatter plot ───────────────────────────────────────────────────
        if self.cfg.produce_charts:
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter_kw = {"alpha": 0.4, "s": 20}
            if att_status_col and att_status_col in analysis_df.columns:
                for status, grp_df in analysis_df.groupby(att_status_col):
                    ax.scatter(grp_df[lang_col], grp_df[compliance_col],
                               label=str(status), **scatter_kw)
                ax.legend(title="ATT status", fontsize=8)
            else:
                ax.scatter(analysis_df[lang_col], analysis_df[compliance_col],
                           color="#1565C0", **scatter_kw)

            # Regression line
            m, b, *_ = stats.linregress(analysis_df[lang_col].dropna(),
                                         analysis_df[compliance_col].dropna())
            x_range = np.linspace(analysis_df[lang_col].min(),
                                   analysis_df[lang_col].max(), 100)
            ax.plot(x_range, m * x_range + b, color="black", linewidth=1.5,
                    linestyle="--", label="OLS fit")
            ax.set_xlabel(f"{lang_col} (NLP treaty anchor score)")
            ax.set_ylabel(f"{compliance_col} (network violations)")
            ax.set_title("I3: ATT Language vs. ATT Compliance Behavior")
            self.save_chart(fig, "i3_att_language_compliance_scatter")

        findings.insert(0,
            f"Analyzed {len(analysis_df)} country-year obs "
            f"({analysis_df['country_code'].nunique()} countries) | "
            f"lang={lang_col}, compliance={compliance_col}"
        )
        return {"key_findings": findings, "n_obs": len(analysis_df)}
