# Cross-Pipeline Analysis: Arms Control Rhetoric vs. Arms Trade Reality

This project cross-references two independent analytical pipelines to answer a fundamental question: **do countries practice what they preach on arms control?**

## Data Sources

| Pipeline | What it measures | Coverage |
|----------|-----------------|----------|
| **NLP Pipeline** (`arms-control-nlp`) | Rhetorical framing in UN General Assembly First Committee speeches — humanitarian vs. deterrence framing, treaty language similarity (ATT, TPNW, Ottawa, CCM) | 199 countries, 1970–2023 |
| **Network Pipeline** (`arms-trade-network`) | Arms trade patterns from SIPRI — who sells to whom, embargo violations, transfers to conflict zones, ethical risk scores | 3,128 country-year observations |

The master dataset merges both pipelines on `(country, year)`, producing **9,064 rows** with 48 features. **2,811 country-years** have data from both pipelines.

## Methodology

Six analyses probe the rhetoric-behavior gap from different angles:

| ID | Analysis | Method |
|----|----------|--------|
| **I1** | Hypocrisy Fingerprint | Pearson correlations + logistic regression — which NLP dimensions predict ethical risk scores? |
| **I2** | Community–Rhetoric Alignment | Silhouette analysis — do countries in the same arms trade community use similar rhetoric? Per-country scores identify who talks most differently from their trade partners. |
| **I3** | ATT Language vs. Compliance | Do countries whose speeches sound like the Arms Trade Treaty text actually comply with it? |
| **I4** | Moral Foundations vs. Violations | Does care/harm moral framing anti-correlate with arms trade violations? |
| **I5** | Regime Transition Lag | When countries backslide from democracy, what changes first — rhetoric or arms trade behavior? Event-study design with Wilcoxon tests. |
| **I6** | P5 Hypocrisy Profile | Kruskal-Wallis tests comparing the 5 nuclear powers (USA, Russia, China, UK, France) against major non-P5 exporters and everyone else. |

## Key Findings

### 1. Treaty fluency predicts *worse* behavior, not better

Countries whose UN speeches sound most like the Convention on Cluster Munitions (CCM) text are significantly **more likely** to be high-risk arms exporters (r=0.215, p=0.035, OR=2.01). The same trend holds for ATT similarity (r=0.176). Meanwhile, humanitarian framing has **zero correlation** with actual export ethics (r=0.005).

Major exporters are deeply fluent in treaty vocabulary — not because they support it, but because they've spent decades engaging with, deflecting, and reframing it.

### 2. Countries that trade arms together don't talk alike

Arms trade communities (detected via network clustering) show **negative** rhetorical alignment (mean silhouette = -0.235). Countries within the same trade cluster are *more* rhetorically diverse than random groupings. Arms trade is driven by geopolitics and economics, not ideological alignment.

The most misaligned countries — India (-0.62), France (-0.61), Russia (-0.54) — are major powers whose rhetoric is maximally different from their trade partners.

### 3. The P5 know the vocabulary but reject the values

The five nuclear powers (P5) use significantly **less humanitarian framing** than other countries (frame ratio 0.66 vs 0.76, p < 0.0001) but score **higher on treaty language similarity** across all four treaties. They are the most fluent speakers of arms control language and the least likely to ratify the treaties they discuss.

Major non-P5 exporters (Germany, Italy, Spain, Israel, Korea) show the opposite pattern: highest humanitarian framing but intermediate treaty specificity.

### 4. Democratic backsliding correlates with rising arms trade centrality

Countries that backslide from democracy see their network betweenness double and export volumes rise by ~75% in the 5 years following transition. The sample is small (n=4) and doesn't reach statistical significance, but the directional pattern is consistent.

## Usage

```bash
# Run all analyses
python run_cross_analysis.py

# Run specific analyses
python run_cross_analysis.py --insights i1,i2,i6

# Custom paths and year range
python run_cross_analysis.py --nlp-output ../arms-control-nlp/output \
                             --network-output ../arms-trade-network/output/metrics \
                             --year-start 2000
```

## Output

Results are written to `output/`:
- `insight_csvs/` — all tabular results (correlation matrices, per-country scores, group statistics)
- `charts/` — PCA scatter plots, silhouette time series, radar charts, bar plots
- `master_merged.csv` — the full joined dataset

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
