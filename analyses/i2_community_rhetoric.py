"""
I2: Community–Rhetoric Alignment
──────────────────────────────────
Do Louvain arms-trade communities cluster together in NLP feature space?

Approach:
  1. PCA on NLP feature vector, points colored by community_id
     (for up to 3 representative years)
  2. Silhouette score per year with community labels
  3. Shuffle null test (1000 permutations) → z-score
  4. Silhouette time series plot
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyses.base import BaseAnalysis
from viz.style import COMMUNITY_COLORS, community_color

logger = logging.getLogger(__name__)


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score
        unique = np.unique(labels)
        if len(unique) < 2:
            return None
        return float(silhouette_score(X, labels))
    except Exception:
        return None


def _silhouette_samples_safe(X: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Return per-sample silhouette scores, or None if not computable."""
    try:
        from sklearn.metrics import silhouette_samples
        unique = np.unique(labels)
        if len(unique) < 2:
            return None
        return silhouette_samples(X, labels)
    except Exception:
        return None


def _null_silhouette(X: np.ndarray, labels: np.ndarray,
                     n_iter: int = 500) -> tuple[float, float]:
    """Return (mean, std) of silhouette scores under label permutation."""
    scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        shuffled = rng.permutation(labels)
        s = _silhouette_safe(X, shuffled)
        if s is not None:
            scores.append(s)
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


class CommunityRhetoric(BaseAnalysis):
    id   = "i2"
    name = "Community–Rhetoric Alignment"

    def run(self, master: pd.DataFrame, nlp_dfs: dict, net_dfs: dict) -> dict:
        df = self.filter_complete(master).copy()

        # ── Discover grouping column ──────────────────────────────────────────
        group_col = self.first_available(
            df, ["net_community_id", "gap_category"], "group column")
        if group_col is None:
            return {"key_findings": ["I2 skipped: no community/group column found"]}

        # ── NLP feature columns ───────────────────────────────────────────────
        feature_cols = self.available_cols(df, self.cfg.NLP_SCORE_COLS, "PCA features")
        if len(feature_cols) < 2:
            return {"key_findings": [
                f"I2 skipped: only {len(feature_cols)} NLP feature columns found"
            ]}

        required = feature_cols + [group_col, "year", "country_code"]
        analysis_df = df[required].dropna(subset=feature_cols + [group_col])
        if len(analysis_df) < 20:
            return {"key_findings": [
                f"I2 skipped: only {len(analysis_df)} complete rows"
            ]}

        # ── Select representative years ───────────────────────────────────────
        all_years = sorted(analysis_df["year"].unique())
        if len(all_years) >= 3:
            idx = [0, len(all_years) // 2, len(all_years) - 1]
            rep_years = [all_years[i] for i in idx]
        else:
            rep_years = all_years

        # ── Per-year silhouette scores ────────────────────────────────────────
        silhouette_rows = []
        country_sil_rows = []  # per-country breakdown
        scaler = StandardScaler()
        pca    = PCA(n_components=2, random_state=42)

        for yr in all_years:
            yr_df = analysis_df[analysis_df["year"] == yr].copy()
            if len(yr_df) < 10:
                continue
            X = yr_df[feature_cols].fillna(yr_df[feature_cols].mean())
            X_scaled = scaler.fit_transform(X)
            labels   = yr_df[group_col].astype(str).values

            sil = _silhouette_safe(X_scaled, labels)
            if sil is None:
                continue

            # Per-country silhouette scores
            sample_scores = _silhouette_samples_safe(X_scaled, labels)
            if sample_scores is not None:
                for i, (_, row) in enumerate(yr_df.iterrows()):
                    country_sil_rows.append({
                        "country_code": row["country_code"],
                        "year":         yr,
                        "community":    labels[i],
                        "silhouette":   float(sample_scores[i]),
                    })

            null_mean, null_std = _null_silhouette(X_scaled, labels, n_iter=200)
            z_score = ((sil - null_mean) / null_std
                       if null_std and null_std > 0 else float("nan"))
            from scipy import stats as sc_stats
            p_value = float(sc_stats.norm.sf(z_score)) if not np.isnan(z_score) else float("nan")

            silhouette_rows.append({
                "year":            yr,
                "silhouette_score": sil,
                "null_mean":       null_mean,
                "null_std":        null_std,
                "z_score":         z_score,
                "p_value":         p_value,
                "n_countries":     len(yr_df),
                "n_groups":        yr_df[group_col].nunique(),
            })

        if not silhouette_rows:
            return {"key_findings": ["I2: could not compute silhouette for any year"]}

        sil_df = pd.DataFrame(silhouette_rows)
        self.save_csv(sil_df, "i2_silhouette_by_year")

        # ── Per-country silhouette breakdown ────────────────────────────────
        if country_sil_rows:
            country_sil_df = pd.DataFrame(country_sil_rows)
            self.save_csv(country_sil_df, "i2_silhouette_by_country")

            # Summary: mean silhouette per country across all years
            country_summary = (
                country_sil_df.groupby("country_code")["silhouette"]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "mean_silhouette",
                                 "std": "std_silhouette",
                                 "count": "n_years"})
                .sort_values("mean_silhouette", ascending=False)
                .reset_index()
            )
            # Add most frequent community
            mode_community = (
                country_sil_df.groupby("country_code")["community"]
                .agg(lambda x: x.value_counts().index[0])
                .rename("primary_community")
            )
            country_summary = country_summary.merge(
                mode_community, on="country_code", how="left")
            self.save_csv(country_summary, "i2_silhouette_country_summary")

        # ── PCA scatter plots for representative years ────────────────────────
        if self.cfg.produce_charts:
            for yr in rep_years:
                yr_df = analysis_df[analysis_df["year"] == yr].copy()
                if len(yr_df) < 5:
                    continue
                X = yr_df[feature_cols].fillna(yr_df[feature_cols].mean())
                X_scaled = scaler.fit_transform(X)
                coords = pca.fit_transform(X_scaled)

                yr_df = yr_df.copy()
                yr_df["pc1"] = coords[:, 0]
                yr_df["pc2"] = coords[:, 1]

                groups = yr_df[group_col].astype(str).unique()
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, grp in enumerate(sorted(groups)):
                    mask = yr_df[group_col].astype(str) == grp
                    color = community_color(i)
                    ax.scatter(
                        yr_df.loc[mask, "pc1"],
                        yr_df.loc[mask, "pc2"],
                        color=color, label=str(grp), alpha=0.7, s=50,
                    )
                    # Country labels for extreme points
                    subset = yr_df[mask]
                    if "country_code" in subset.columns and len(subset) <= 8:
                        for _, row in subset.iterrows():
                            ax.annotate(row["country_code"], (row["pc1"], row["pc2"]),
                                        fontsize=6, alpha=0.7)

                var_explained = pca.explained_variance_ratio_
                ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
                ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
                ax.set_title(f"I2: NLP Space Colored by {group_col} — {yr}")
                ax.legend(title=group_col, fontsize=7, ncol=2)
                self.save_chart(fig, f"i2_pca_community_{yr}")

            # Silhouette time series
            fig2, ax2 = plt.subplots(figsize=(11, 4))
            ax2.plot(sil_df["year"], sil_df["silhouette_score"],
                     color="#1565C0", linewidth=2, label="Observed")
            ax2.fill_between(
                sil_df["year"],
                sil_df["null_mean"] - sil_df["null_std"],
                sil_df["null_mean"] + sil_df["null_std"],
                alpha=0.2, color="gray", label="Null ± 1 SD"
            )
            ax2.plot(sil_df["year"], sil_df["null_mean"],
                     color="gray", linewidth=1, linestyle="--")
            ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Silhouette Score")
            ax2.set_title(f"I2: Do Arms-Trade Communities Cluster in NLP Space?\n"
                          f"(grouping: {group_col})")
            ax2.legend()
            self.save_chart(fig2, "i2_silhouette_time_series")

        # ── Per-country bar chart (top/bottom 15) ──────────────────────────────
        if self.cfg.produce_charts and country_sil_rows:
            cs = country_summary.copy()
            n_show = min(15, len(cs))
            top = cs.head(n_show)
            bottom = cs.tail(n_show)
            show = pd.concat([top, bottom]).drop_duplicates(subset="country_code")
            show = show.sort_values("mean_silhouette", ascending=True)

            fig3, ax3 = plt.subplots(figsize=(9, max(6, len(show) * 0.3)))
            colors = ["#C62828" if v < 0 else "#2E7D32" for v in show["mean_silhouette"]]
            ax3.barh(show["country_code"], show["mean_silhouette"], color=colors)
            ax3.axvline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("Mean Silhouette Score (across years)")
            ax3.set_title("I2: Per-Country Community–Rhetoric Alignment\n"
                          "(positive = rhetoric matches trade community, "
                          "negative = rhetoric diverges)")
            ax3.tick_params(axis="y", labelsize=7)
            fig3.tight_layout()
            self.save_chart(fig3, "i2_silhouette_by_country_bar")

        # ── Key findings ──────────────────────────────────────────────────────
        sig_years = sil_df[sil_df["p_value"] < 0.05]["year"].tolist()
        avg_sil = sil_df["silhouette_score"].mean()
        findings = [
            f"Grouping column: {group_col}",
            f"NLP features used: {feature_cols}",
            f"Mean silhouette across years: {avg_sil:.3f}",
            f"Years where community alignment is significant (p<0.05): {sig_years}",
        ]
        best = sil_df.loc[sil_df["silhouette_score"].idxmax()]
        findings.append(
            f"Best alignment year: {int(best['year'])} "
            f"(silhouette={best['silhouette_score']:.3f}, "
            f"z={best['z_score']:.2f})"
        )
        if country_sil_rows:
            cs = country_summary
            top3 = cs.head(3)["country_code"].tolist()
            bot3 = cs.tail(3)["country_code"].tolist()
            findings.append(f"Most aligned countries (rhetoric ≈ trade partners): {top3}")
            findings.append(f"Most misaligned countries (rhetoric ≠ trade partners): {bot3}")
            findings.append(f"Per-country scores saved: {len(cs)} countries")
        return {"key_findings": findings, "n_obs": len(analysis_df)}
