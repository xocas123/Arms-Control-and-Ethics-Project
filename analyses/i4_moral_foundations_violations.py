"""
I4: Moral Foundations vs. Ethical Violations
─────────────────────────────────────────────
Do care_harm / fairness MFT scores anti-correlate with embargo violations
and conflict transfers?

Approach:
  1. Spearman correlation matrix: MFT cols × violation cols (BH-corrected)
  2. Panel OLS with year fixed effects
  3. Violin plots: care_harm quartile vs. ethical_risk_score distribution
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analyses.base import BaseAnalysis
from viz.style import significance_stars

logger = logging.getLogger(__name__)


def _bh_correct(pvalues: list[float]) -> list[float]:
    try:
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(pvalues, method="fdr_bh")
        return list(corrected)
    except ImportError:
        return pvalues


class MoralFoundationsViolations(BaseAnalysis):
    id   = "i4"
    name = "Moral Foundations vs. Ethical Violations"

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        df = self.filter_complete(master).copy()

        mft_cols  = self.available_cols(df, self.cfg.MFT_COLS, "MFT")
        viol_cols = self.available_cols(df, self.cfg.VIOLATION_COLS, "violations")

        if not mft_cols:
            return {"key_findings": ["I4 skipped: no MFT columns found"]}
        if not viol_cols:
            return {"key_findings": ["I4 skipped: no violation columns found"]}

        analysis_df = df[mft_cols + viol_cols + ["country_code", "year"]].dropna(
            subset=mft_cols + viol_cols
        )
        if len(analysis_df) < 20:
            return {"key_findings": [
                f"I4 skipped: only {len(analysis_df)} complete rows"
            ]}

        # ── 1. Spearman correlation matrix ────────────────────────────────────
        corr_rows = []
        all_pvals = []
        for mft in mft_cols:
            for viol in viol_cols:
                r, p = stats.spearmanr(analysis_df[mft], analysis_df[viol],
                                       nan_policy="omit")
                corr_rows.append({
                    "mft_feature":       mft,
                    "violation_feature": viol,
                    "spearman_r":        r,
                    "p_value_raw":       p,
                })
                all_pvals.append(p)

        corrected = _bh_correct(all_pvals)
        for i, row in enumerate(corr_rows):
            row["p_value_bh"] = corrected[i]
            row["stars"]      = significance_stars(corrected[i])

        corr_df = pd.DataFrame(corr_rows)
        self.save_csv(corr_df, "i4_mft_violation_correlation")

        # ── 2. Panel OLS with year fixed effects ──────────────────────────────
        risk_col = self.first_available(analysis_df, self.cfg.VIOLATION_COLS, "OLS target")
        ols_result = None
        if risk_col:
            try:
                import statsmodels.formula.api as smf
                formula = (f"{risk_col} ~ " +
                           " + ".join(mft_cols) +
                           " + C(year)")
                model = smf.ols(formula, data=analysis_df).fit()
                coef_rows = []
                for term in mft_cols:
                    if term in model.params:
                        coef_rows.append({
                            "term":    term,
                            "coef":    model.params[term],
                            "p_value": model.pvalues[term],
                            "stars":   significance_stars(model.pvalues[term]),
                        })
                coef_df = pd.DataFrame(coef_rows)
                self.save_csv(coef_df, "i4_panel_regression_table")
                ols_result = model
            except Exception as exc:
                logger.warning("I4 OLS failed: %s", exc)

        # ── 3. Violin plot ────────────────────────────────────────────────────
        if self.cfg.produce_charts and risk_col and "care_harm" in analysis_df.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            analysis_df["care_harm_quartile"] = pd.qcut(
                analysis_df["care_harm"], q=4,
                labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
            )
            sns.violinplot(
                data=analysis_df, x="care_harm_quartile", y=risk_col,
                hue="care_harm_quartile", palette="RdYlGn_r",
                ax=ax, inner="box", legend=False
            )
            ax.set_xlabel("care_harm quartile (MFT)")
            ax.set_ylabel(risk_col)
            ax.set_title("I4: Care/Harm Moral Framing vs. Ethical Risk Score")
            self.save_chart(fig, "i4_care_harm_violin")

        # Heatmap
        if self.cfg.produce_charts and len(mft_cols) >= 2 and len(viol_cols) >= 2:
            pivot = corr_df.pivot(
                index="mft_feature", columns="violation_feature", values="spearman_r"
            )
            annot = corr_df.pivot(
                index="mft_feature", columns="violation_feature", values="stars"
            )
            fig2, ax2 = plt.subplots(
                figsize=(max(5, len(viol_cols) * 1.8), max(3, len(mft_cols) * 1.2))
            )
            sns.heatmap(
                pivot, annot=annot, fmt="", center=0,
                cmap="RdBu_r", ax=ax2,
                cbar_kws={"label": "Spearman ρ"},
                linewidths=0.5,
            )
            ax2.set_title("I4: MFT Scores vs. Ethical Violations\n"
                          "(* BH p<0.05, ** p<0.01, *** p<0.001)")
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            plt.xticks(rotation=30, ha="right")
            self.save_chart(fig2, "i4_mft_heatmap")

        # ── Key findings ──────────────────────────────────────────────────────
        sig = corr_df[corr_df["p_value_bh"] < 0.05].sort_values("spearman_r")
        findings = [
            f"Analyzed {len(analysis_df)} rows | MFT cols: {mft_cols} | "
            f"Violation cols: {viol_cols}",
            f"Significant Spearman correlations (BH p<0.05): {len(sig)} / "
            f"{len(corr_df)} pairs",
        ]
        if not sig.empty:
            strongest = sig.iloc[0]
            findings.append(
                f"Strongest: {strongest['mft_feature']} vs "
                f"{strongest['violation_feature']}: "
                f"ρ={strongest['spearman_r']:.3f}{strongest['stars']}"
            )
        if ols_result:
            findings.append(
                f"Panel OLS R²={ols_result.rsquared:.3f} "
                f"(dependent: {risk_col})"
            )
        return {"key_findings": findings, "n_obs": len(analysis_df)}
