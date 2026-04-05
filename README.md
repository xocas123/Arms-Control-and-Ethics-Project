# Arms Control NLP Pipeline

Computational analysis of arms control discourse in the UN General Debate (1970-2023). Measures how states' words about arms control align -- or don't -- with their behavior (treaty ratification, voting, arms transfers). Built to support a policy memo on the rhetoric-action gap in multilateral arms control, with a focus on drone proliferation and the Arms Trade Treaty.

The companion pipeline is `arms-trade-network/` (network analysis of actual transfers).

---

## Data Sources

- **UN General Debate Corpus (UNGDC)**: 9,046 speeches from 199 countries, 1970-2023
- **UNGA Voting Records**: 862,039 vote records across 6,202 resolutions, 1946-2019
- **V-Dem Regime Data**: Democracy/autocracy classification for 202 countries
- **SIPRI Arms Transfers Database**: Trade register with 29,917 deal records including 737 drone transfers
- **Treaty Anchor Texts**: 72 arms control treaties from Rush-Bagot (1817) to TPNW (2017), plus drone/LAWS anchors
- **First Committee Resolutions**: 949 resolutions (2002-2024), 805 with full text

## Methods

- Sentence embeddings (all-mpnet-base-v2) for semantic similarity
- Cosine similarity to treaty anchor texts for rhetorical alignment measurement
- Monroe et al. (2008) log-odds ratio ("Fightin' Words") for distinctive vocabulary
- PELT change-point detection for structural breaks
- Lexicon-based humanitarian vs. deterrence frame scoring
- LDA topic modeling (30 topics)
- OLS regression mapping drone rhetoric to actual transfers

---

## Research Questions and Key Findings

### Q1: Has arms control discourse shifted from deterrence to humanitarian framing?

Global humanitarian framing rose from 0.56 in 1975 to 0.84 by 1993 (post-CWC) and has remained above 0.80 since. Structural breaks align with major treaty events: 1991 (Cold War end), 1993 (CWC), 1997 (Ottawa). The humanitarian turn is real but masks divergent behavior.

### Q2: Do democracies and autocracies talk about arms control differently?

Democracies use rights-based language ("human rights," "humanitarian," "must"); autocracies emphasize sovereignty ("peoples," "struggle," "liberation"). Rhetorical distance between regime types doubled from 0.008 to 0.032 between 2000 and 2023. NAM autocracies vote pro-disarmament at 94%+ while using deterrence-oriented rhetoric -- the largest rhetoric-vote gap of any group.

### Q3: Does rhetoric lead or follow treaty ratification?

For the ATT, countries adopted treaty-aligned language 4 years before ratifying (p=0.008). For the Ottawa Convention, rhetoric shifted only after ratification. The inconsistency suggests rhetorical alignment is sometimes strategic signaling.

### Q4: Do nuclear and non-nuclear states live in different rhetorical worlds?

NWS-NNWS rhetorical distance peaked at 0.207 in 2014 and remains at 0.18 in 2023. TPNW voting shows a 90-point gap (NNWS 90% yes, NWS 0%). P5 internal coherence collapsed since 2008 -- pairwise similarity dropped from 0.85 to 0.30-0.70.

### Q5: How does regime type shape treaty engagement across speech, voting, and trade?

Integrates regime classification with treaty adoption curves, arms trade data, and voting. Quantifies the rhetoric-action gap by alliance bloc and shows arms exporters cluster together regardless of ideology.

### Q6: Drones and Autonomous Weapons

737 drone deals identified in SIPRI data. Top four exporters (USA $1,475M, Israel $1,325M, China $1,200M, Turkey $902M) account for ~80% of all drone transfer value. Non-ATT-parties export 2.65x more drone TIV per deal. 1,233 country-years contain drone/LAWS mentions in speeches, ramping up sharply post-2013. OLS regressions test whether rhetoric predicts transfers, stratified by regime type and ATT status.

---

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (requires GPU for embeddings, ~10 min on L40)
python run.py --year-start 1970 --year-end 2023

# Single question
python run.py --question q6

# Fast embeddings (MiniLM)
python run.py --fast

# Resume from checkpoint
python run.py --resume
```

### Cluster (SLURM)

```bash
sbatch submit.sh
```

---

## Pipeline Architecture

```
Data Sources (7 loaders)
    |
Shared Pipeline
    |- Keyword segmentation (45+ terms including drone/LAWS)
    |- Sentence embeddings (all-mpnet-base-v2, GPU)
    |- Treaty anchor embeddings (72 treaties)
    |- Frame scoring (lexicon + embedding)
    |- LDA topic modeling (30 topics)
    |
Question Modules
    |- Q1: Humanitarian vs deterrence framing
    |- Q2: Democracy vs autocracy rhetoric
    |- Q3: Rhetoric before/after treaty ratification
    |- Q4: Nuclear vs non-nuclear states
    |- Q5: Regime-treaty integration
    |- Q6: Drones & autonomous weapons
    |
Outputs (CSVs + plots per question)
```

## Project Structure

```
src/
  data/        -- Loaders, preprocessing, segmentation, country groups
  shared/      -- Embeddings, frame scoring, temporal analysis, lexicons
  questions/   -- Q1-Q6 analysis modules
  viz/         -- Visualization modules
  analysis/    -- Voting analysis, term frequency
data/
  treaties/    -- anchors.json (72 treaty anchor texts)
  raw/         -- UNGDC, unvotes, V-Dem, resolutions (not in git)
output/        -- CSVs and plots (not in git)
```

## Requirements

- Python 3.11+
- sentence-transformers==3.0.1, transformers==4.44.0
- torch==2.5.1 (GPU recommended)
- pandas, numpy, scipy, scikit-learn, matplotlib, ruptures, pyarrow

## References

- Baturo, Dasandi & Mikhaylov (2017). "Understanding State Preferences With Text As Data." *Research & Politics.*
- Jankin, Baturo & Dasandi (2025). "Words to Unite Nations." *Journal of Peace Research.*
- Bailey, Strezhnev & Voeten (2017). "Estimating Dynamic State Preferences from UN Voting." *JCR.*
- Monroe, Colaresi & Quinn (2008). "Fightin' Words." *Political Analysis.*
