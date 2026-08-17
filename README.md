# Multi-Factor Timing with Deep Learning — Replication + Volatility-Regime Abstention Overlay

A partial replication of Cotturo, Liu, and Proner (2025), ["Multi-Factor Timing with Deep
Learning"] (forecasting the sign of
next-month returns for five Fama-French/momentum factors with a multi-task neural network,
benchmarked against off-the-shelf ML models), extended with an original addition: a
**volatility-regime abstention overlay** — a simple, model-free "scanner" that sits a factor out
whenever its own trailing realized volatility is elevated. Of the four risk-aware strategies
tested here (three built on MC Dropout uncertainty, one on volatility), the volatility scanner is
the only one that beats the always-invest baseline, and the only one that catches all four of the
worst diagnosed losses in the sample.

## What's here

| Notebook | What it does |
|---|---|
| [`notebooks/Initial MT.ipynb`](notebooks/Initial%20MT.ipynb) | Builds and walk-forward trains the paper's MT architecture (4 shared dense layers + 2 factor-specific layers per factor) |
| [`notebooks/Off-the-shelf Models.ipynb`](notebooks/Off-the-shelf%20Models.ipynb) | Same walk-forward procedure, benchmarked against Logistic Regression, Random Forest, and XGBoost |
| [`notebooks/MC Dropout Extension.ipynb`](notebooks/MC%20Dropout%20Extension.ipynb) | Adds MC Dropout uncertainty on top of MT, and tests four ways of trading on risk signals — including the volatility scanner |

**Run them in that order** — the latter two read data through the same `src/loading.py` /
`src/estimation.py` helpers as `Initial MT.ipynb`, and the benchmark cell in `Off-the-shelf
Models.ipynb` picks up `results/mt_oos_predictions.csv` if it's already been generated.

## The volatility scanner

`notebooks/MC Dropout Extension.ipynb` computes trailing 3-month realized volatility for each
factor's own return series — no model, no look-ahead, only returns already known by the signal
date — and abstains (goes flat instead of long) whenever that trailing volatility exceeds a
cutoff. The cutoff is the 80th percentile of trailing volatility, calibrated separately for each
of the 32 walk-forward folds from that fold's *validation*-period data only, never test data. The
window and threshold (3 months, 80th percentile) were fixed in advance and not tuned against the
results below.

Multi-factor timing performance vs. the always-long BUY benchmark (Sharpe 0.60):

| Strategy | Sharpe | alpha (annualized %) | t(alpha) | beta | R² (%) |
|---|---|---|---|---|---|
| Baseline (unweighted MT rule) | 0.66 | 0.50 | 1.21 | 0.88 | 85.3 |
| MC-Dropout confidence-scaled | 0.53 | 0.08 | 0.21 | 0.54 | 70.2 |
| MC-Dropout abstain (top 20% uncertain) | 0.56 | 0.40 | 0.79 | 0.62 | 58.5 |
| **Volatility scanner (abstain)** | **0.67** | **0.97** | **1.97** | 0.44 | 41.6 |

The scanner is the only one of the four that beats the baseline, with the best t(alpha) of the
four (1.97 — short of conventional significance, but closest). Two checks on why it works:

- **Hit rate on the four worst diagnosed losses**: Feb 2000 SMB, Dec 2008 HML, Apr 2009 MOM, and
  May 2021 HML were all correctly abstained — 4 for 4.
- **Independent of MC Dropout**: of all factor-months the scanner sits out (431, 22.5% of the
  sample), only 36.4% were also flagged uncertain by MC Dropout — it's catching a different kind
  of risk, not relabeling the same months.

That independence has a clear cause. A diagnostic on the worst MC-Dropout-flagged losing months
found they were **high-conviction calls, not weak ones** — mean conviction 0.67 across 149
qualifying months, more high-conviction (>0.7, n=56) than low-conviction (0.50–0.58, n=42). MC
Dropout's 50 stochastic sub-networks all learn the same about-to-break pattern together, so the
model's self-assessed uncertainty doesn't flag a genuine regime break in advance — a known
failure mode, not a bug here. That result also killed a plausible follow-up (conviction-gated
abstention: only sit out when both flagged uncertain *and* weakly convicted); it was never built
because the diagnostic showed it would have kept exactly the losing positions at full size.

A further check tested whether *layering* MC Dropout's flag on top of the scanner adds value —
i.e., whether MC Dropout catches anything extra on the months the scanner doesn't already sit out:

| Subset | MC-Dropout-flagged acc | MC-Dropout-unflagged acc | Gap |
|---|---|---|---|
| Scanner keeps (n=1484) | 53.0% (n=298) | 55.1% (n=1186) | +2.0pp |
| Scanner abstains (n=431) | 56.1% (n=157) | 51.5% (n=274) | -4.6pp |

The +2.0pp gap on the kept subset is in the expected direction but below a ≈4pp bar set in advance
for "worth building," and smaller than the ~2.9pp standard error at that sample size — not
distinguishable from zero. MC Dropout's apparent value looks concentrated in exactly the
high-volatility months the scanner already handles, so a layered strategy wasn't built.

**Caveats**: this is one run — the scanner's flag itself is fully deterministic, but its measured
Sharpe/alpha runs through a freshly retrained MT model each time (see run-to-run noise below), so
the "beats baseline" result hasn't been checked across multiple seeds. The threshold was a single
reasonable default, not swept against alternatives.

## Results vs. the paper, out-of-sample Jan 1990 – Dec 2021

Mean out-of-sample classification accuracy (paper Table 1) and multi-factor timing Sharpe ratio
(paper Table 3):

| Model | Accuracy (this repo) | Accuracy (paper) | Sharpe (this repo) | Sharpe (paper) |
|---|---|---|---|---|
| BUY (always long) | 53.5% | 53.5% | 0.60 | 0.60 |
| LR | 51.1% | 53.1% | 0.64 | 0.61 |
| RF | 56.3% | 55.6% | 0.84 | 0.66 |
| XGBoost (≈ GBT) | 54.6% | 54.8% | 0.79 | 0.61 |
| **MT** | **52.3%** | 55.4% | **0.55** | 0.69 |
| MC-Dropout MT | 54.3% | — | 0.66 | — |

MT landing under the paper's fully-tuned, 10-seed-ensembled numbers is expected given the
single-seed/no-grid-search simplification noted below, not a bug — architecture, walk-forward
procedure, and data construction otherwise match the paper.

**Run-to-run noise**: every neural-net number above moves between runs — TensorFlow training
isn't perfectly deterministic even with a fixed seed, and this repo uses one seed instead of the
paper's 10-seed ensemble. Observed MT accuracy has ranged 52–55% across repeated runs on the same
code and data; treat exact figures as indicative, not precise.

**Trading-return correctness**: strategy returns are graded against the return realized the month
*after* the signal date (matching what each model actually predicts, `sign(r_{t+1})`), not the
return already known at signal time. `tests/test_estimation.py` regression-tests this alignment
with a real April 2009 momentum-crash fixture.

## Out of scope

This replicates the paper's static MT model, three off-the-shelf benchmarks, and adds the
volatility scanner above — not a full reproduction. Not implemented here, noted only briefly:
**DMT/DMTc** (the paper's dynamic LSTM architecture), **full hyperparameter search + 10-seed
ensembling** (a single seed and fixed hyperparameters are used instead — the main source of the
accuracy/Sharpe gap above), the **JKP 149-factor extension** and **Shapley-value importance**
(paper Sec 5/4.6), **EN and SVM benchmarks**, and a **single-task LSTM benchmark** (implemented,
then removed — retraining it for all 32 folds × 5 factors took ~45–50 min, not worth it here).
The MC-Dropout MT model above also uses `l1=0.001` rather than the paper's grid (0.005–0.02),
because every grid value collapses its shared-trunk weights to ~0 without batch norm to
counteract the penalty (see the notebook's model-building cell) — so its accuracy edge over
regular MT isn't purely a dropout-vs-batch-norm comparison, since the regularization strength
changed too.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Register the kernel for Jupyter/VS Code if it isn't autodetected:

```bash
python -m ipykernel install --user --name sof-casing --display-name "SOF Casing (.venv)"
```

### Data

`src/loading.py` pulls three sources on first import and caches each as a parquet file under
`data/` (gitignored):

- **Response factors** (SMB, HML, RMW, CMA, momentum): Kenneth French's data library, via
  `pandas-datareader`.
- **Macro predictors** (122 series): the current FRED-MD release, transformed per McCracken and
  Ng (2016).
- **Financial predictors** (137 long–short anomaly portfolios): Chen and Zimmermann (2022) / Open
  Source Asset Pricing, via the `openassetpricing` package, which reads from a
  Google-Drive-hosted file subject to a **shared download quota** — if the fetch fails, it's
  often transient; retry later.

Subsequent imports read from the `data/*.parquet` cache instead of re-fetching. If you suspect
the cache is stale or partial, delete the relevant file(s) in `data/` and re-import —
`loading.py` re-fetches automatically, and raises a clear `RuntimeError` at import time if the
three sources don't overlap on enough months, rather than silently producing an empty panel.

**Note:** because fetched frames are cached at *Python import* time, changing files in `data/`
requires restarting the notebook kernel — a plain cell re-run won't re-trigger the fetch in an
already-running kernel.

## Project structure

```
src/
  loading.py         # data pull + cache (Ken French, FRED-MD, OpenAP)
  estimation.py      # shared walk-forward split / standardization / scoring helpers,
                      # used identically by all three notebooks
notebooks/            # see table above
data/                 # cached parquet pulls (gitignored, regenerated on first import)
results/               # OOS predictions + strategy returns written by the notebooks
```

## Reference

Cotturo, P., Liu, F., and Proner, R. (2025). *Multi-Factor Timing with Deep Learning.*
