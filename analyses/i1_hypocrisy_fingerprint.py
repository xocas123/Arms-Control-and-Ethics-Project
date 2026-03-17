"""
I1: Hypocrisy Fingerprint
─────────────────────────
Which NLP dimensions best predict a supplier's ethical risk behavior?

Approach:
  1. Pearson correlation matrix: NLP score cols vs. network risk target
  2. Logistic regression: high-risk supplier flag ~ NLP features (5-fold CV AUC)
  3. Coefficient plot
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


def _bh_correct(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return pvalues
    try:
        from statsmodels.stats.multitest import multipletests
        _, corrected, _, _ = multipletests(pvalues, method="fdr_bh")
        return list(corrected)
    except ImportError:
        return pvalues


class HypocrisyFingerprint(BaseAnalysis):
    id   = "i1"
    name = "Hypocrisy Fingerprint"

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        df = self.filter_complete(master).copy()

        # ── Select predictor columns ──────────────────────────────────────────
        predictors = self.available_cols(df, self.cfg.NLP_SCORE_COLS, "predictors")
        if len(predictors) < 3:
            return {"key_findings": [
                f"I1 skipped: only {len(predictors)} NLP predictor columns found "
                "(need ≥ 3)"
            ]}

        # ── Select risk target ────────────────────────────────────────────────
        target_col = self.first_available(
            df,
            ["net_mean_ethical_risk_score", "net_pct_ethical",
             "net_n_att_concern", "net_n_into_conflict"],
            "risk target"
        )
        if target_col is None:
            return {"key_findings": ["I1 skipped: no network risk column found"]}

        analysis_df = df[predictors + [target_col, "country_code", "year"]].dropna()
        if len(analysis_df) < 30:
            return {"key_findings": [
                f"I1 skipped: only {len(analysis_df)} complete rows"
            ]}

        # ── 1. Correlation matrix ─────────────────────────────────────────────
        corr_rows = []
        pvals = []
        for pred in predictors:
            r, p = stats.pearsonr(analysis_df[pred], analysis_df[target_col])
            corr_rows.append({"nlp_feature": pred, "pearson_r": r, "p_value_raw": p})
            pvals.append(p)

        corrected = _bh_correct(pvals)
        for i, row in enumerate(corr_rows):
            row["p_value_bh"] = corrected[i]
            row["significant"] = corrected[i] < 0.05

        corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r")
        self.save_csv(corr_df, "i1_correlation_matrix")

        # ── 2. Logistic regression (high-risk flag) ───────────────────────────
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import Pipeline

            threshold = analysis_df[target_col].quantile(0.75)
            y = (analysis_df[target_col] >= threshold).astype(int)

            if y.sum() < 10:
                raise ValueError("Too few positive cases for logistic regression")

            X = analysis_df[predictors].values
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            cv_auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
            pipe.fit(X, y)

            coefs = pipe.named_steps["clf"].coef_[0]
            coef_df = pd.DataFrame({
                "feature":     predictors,
                "coefficient": coefs,
                "abs_coef":    np.abs(coefs),
                "odds_ratio":  np.exp(coefs),
            }).sort_values("abs_coef", ascending=False)
            self.save_csv(coef_df, "i1_logistic_coefs")

            auc_mean = float(np.mean(cv_auc))
            findings_logistic = [
                f"Logistic regression AUC-ROC (5-fold CV): {auc_mean:.3f}",
                f"Strongest predictor of high-risk behavior: "
                f"{coef_df.iloc[0]['feature']} "
                f"(OR={coef_df.iloc[0]['odds_ratio']:.2f})",
            ]
        except Exception as exc:
            logger.warning("I1 logistic step failed: %s", exc)
            coef_df = None
            findings_logistic = [f"Logistic regression not available: {exc}"]

        # ── 3. Correlation heatmap ────────────────────────────────────────────
        if self.cfg.produce_charts:
            fig, ax = plt.subplots(figsize=(max(6, len(predictors) * 0.6 + 2), 5))
            colors = ["#C62828" if r > 0 else "#1565C0" for r in corr_df["pearson_r"]]
            bars = ax.barh(corr_df["nlp_feature"], corr_df["pearson_r"], color=colors)
            # Significance markers
            for bar, sig in zip(bars, corr_df["significant"]):
                if sig:
                    x = bar.get_width()
                    ax.text(x + (0.005 if x >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
                            "*", va="center", ha="left" if x >= 0 else "right", fontsize=12)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"Pearson r vs. {target_col}")
            ax.set_title("I1: NLP Features vs. Supplier Ethical Risk\n(* = BH-corrected p < 0.05)")
            self.save_chart(fig, "i1_correlation_heatmap")

            # Coefficient plot
            if coef_df is not None and self.cfg.produce_charts:
                fig2, ax2 = plt.subplots(figsize=(8, max(4, len(coef_df) * 0.4 + 2)))
                colors2 = ["#C62828" if c > 0 else "#1565C0"
                           for c in coef_df["coefficient"]]
                ax2.barh(coef_df["feature"], coef_df["coefficient"], color=colors2)
                ax2.axvline(0, color="black", linewidth=0.8)
                ax2.set_xlabel("Logistic Regression Coefficient")
                ax2.set_title("I1: Predictors of High Ethical-Risk Supplier\n"
                              f"(threshold = 75th pct of {target_col})")
                self.save_chart(fig2, "i1_logistic_forest")

        # ── Key findings ──────────────────────────────────────────────────────
        sig_features = corr_df[corr_df["significant"]]["nlp_feature"].tolist()
        most_positive = corr_df[corr_df["pearson_r"] > 0].tail(1)
        most_negative = corr_df[corr_df["pearson_r"] < 0].head(1)

        findings = [
            f"Analyzed {len(analysis_df)} country-year observations "
            f"({analysis_df['country_code'].nunique()} countries)",
            f"Risk target: {target_col}",
            f"Significant NLP correlates (BH p<0.05): {sig_features}",
        ]
        if not most_positive.empty:
            row = most_positive.iloc[0]
            findings.append(
                f"Strongest positive correlation: {row['nlp_feature']} "
                f"(r={row['pearson_r']:.3f})"
            )
        if not most_negative.empty:
            row = most_negative.iloc[0]
            findings.append(
                f"Strongest negative correlation: {row['nlp_feature']} "
                f"(r={row['pearson_r']:.3f})"
            )
        findings.extend(findings_logistic)
        return {"key_findings": findings, "n_obs": len(analysis_df)}
