# Multi-Factor Timing with Deep Learning — Replication

A partial replication of Cotturo, Liu, and Proner (2025), ["Multi-Factor Timing with Deep
Learning"](Multi-Factor%20Timing%20with%20Deep%20Learning.pdf), which forecasts the sign of
next-month returns for five Fama-French/momentum factor portfolios using a multi-task neural
network (MT) that shares a common representation across factors, and compares it against
off-the-shelf machine learning benchmarks.

## What's here

| Notebook | Paper section | What it does |
|---|---|---|
| [`notebooks/Initial MT.ipynb`](notebooks/Initial%20MT.ipynb) | Sec 3.2.1, 3.2.4 | Builds and walk-forward trains the paper's MT architecture (4 shared "hard-sharing" dense layers + 2 factor-specific layers per factor) |
| [`notebooks/Off-the-shelf Models.ipynb`](notebooks/Off-the-shelf%20Models.ipynb) | Sec 3.2.3 | Same walk-forward procedure, benchmarked against Logistic Regression, Random Forest, and XGBoost (stand-in for the paper's GBT) |
| [`notebooks/MC Dropout Extension.ipynb`](notebooks/MC%20Dropout%20Extension.ipynb) | — (original extension) | Adds Monte Carlo Dropout uncertainty quantification on top of MT, and tests whether acting on that uncertainty (confidence-scaling or abstaining) improves the trading strategy |
| [`notebooks/exploratory/start.ipynb`](notebooks/exploratory/start.ipynb) | — | Early scratch work for the data pipeline; superseded by `src/loading.py`, kept for reference only |

**Run them in that order** — `Off-the-shelf Models.ipynb` and `MC Dropout Extension.ipynb` both
read the data through the same `src/loading.py` / `src/estimation.py` helpers as
`Initial MT.ipynb`, and the benchmark cell at the end of `Off-the-shelf Models.ipynb` will pick
up `results/mt_oos_predictions.csv` if `Initial MT.ipynb` has already been run.

## What's deliberately simplified vs. the paper

This is a scoped replication, not a full reproduction. Concretely, out of scope for this pass:

- **DMT / DMTc** (the paper's dynamic multi-task architecture with LSTMs over macro/financial
  predictors) — not implemented. Only the static **MT** model is.
- **Hyperparameter search + ensembling.** The paper refits an 8-point grid (`l1` × learning
  rate) × 10-seed ensemble for every one of the 32 walk-forward years, for every model. That's
  computationally infeasible to reproduce in full here, so every model in this repo instead
  uses a single seed with fixed hyperparameters taken from valid points in the paper's own grid
  (Table IA1). This is the single biggest source of the accuracy/Sharpe gap vs. the paper's
  published numbers below.
- **JKP 149-factor extension** (paper Sec 5) and **Shapley-value variable importance** (paper
  Sec 4.6) — not implemented.
- **EN and SVM benchmarks** — not implemented (LR, RF, GBT are).
- **Single-task LSTM benchmark** — dropped. It was implemented and does train correctly (~15-20s
  per fold), but retraining it from scratch for all 32 walk-forward years × 5 factors takes
  ~45-50 minutes end to end, which wasn't worth it for this pass; it's been removed from
  `Off-the-shelf Models.ipynb` entirely rather than left half-run.
- **MC Dropout extension's `l1` (0.001, not the paper's 0.005-0.02 grid).** This extension swaps
  MT's batch norm for dropout in the shared trunk (needed so dropout can be forced on at
  prediction time without also perturbing batch norm's statistics — see the notebook's intro).
  Every `l1` value in the paper's own grid, including the smallest (0.005), collapses this
  variant's shared-trunk weights to ~0 during training: without batch norm to renormalize
  activations regardless of weight scale, the L1 penalty has nothing to counteract it, so Adam
  drives the input-dependent weights to zero and the network's predictions end up carried almost
  entirely by the (unregularized) bias — which in turn collapses the MC-Dropout uncertainty
  signal to floating-point noise (~1e-7) instead of a real one. `l1=0.001` is the smallest
  deviation below the paper's grid that reliably avoided this across every fold checked; see the
  notebook cell that builds the model for the full writeup.

Each notebook documents its own specific simplifications inline, next to the relevant code.

## Results (this repo vs. the paper), out-of-sample Jan 1990 – Dec 2021

Mean out-of-sample classification accuracy across the five factors (paper Table 1) and
multi-factor timing Sharpe ratio (paper Table 3):

| Model | Accuracy (this repo) | Accuracy (paper) | Sharpe (this repo) | Sharpe (paper) |
|---|---|---|---|---|
| BUY (always long) | 53.5% | 53.5% | 0.59 | 0.60 |
| LR | 51.1% | 53.1% | 1.11 | 0.61 |
| RF | 56.3% | 55.6% | 1.33 | 0.66 |
| XGBoost (≈ GBT) | 54.6% | 54.8% | 1.17 | 0.61 |
| **MT** | **53.1%** | 55.4% | **0.80** | 0.69 |

*(LSTM dropped from the comparison — see "What's deliberately simplified" above.)*

Given the single-seed/no-grid-search simplification above, MT landing a few points under the
paper's published, fully-tuned-and-ensembled accuracy number is expected rather than a bug — the
walk-forward procedure, data construction, and architecture otherwise match the paper. The
repo's Sharpe ratios for LR/RF/XGBoost running noticeably *above* the paper's, on the other hand,
isn't something this pass investigated further — worth a look before leaning on those numbers.
Because each neural net here uses one random seed instead of the paper's 10-seed ensemble, and
TensorFlow training isn't perfectly deterministic even with a fixed seed, **exact numbers will
drift slightly between reruns** (observed mean MT accuracy has ranged 52-55% across repeated runs
during development) — this is expected run-to-run noise from the simplification, not a bug.

*Corrected the MT row above (previously 53.7% / 0.60) to match `Initial MT.ipynb`'s actual
last-run output (53.1% / 0.80) — the old 0.60 lines up with the BUY benchmark's Sharpe, not MT's
own, so it looks like a copy/paste slip rather than a real result; this pass didn't retrain MT,
just corrected the transcription.*

## MC Dropout extension results, out-of-sample 1990-2021

`notebooks/MC Dropout Extension.ipynb` adds Monte Carlo Dropout uncertainty to MT and tests
three ways of trading on it — full writeup of the method is in the notebook's own intro cell.
Multi-factor timing performance vs. the multi-factor BUY benchmark (Sharpe 0.59):

| Strategy | Sharpe | alpha (annualized %) | t(alpha) | beta | R² (%) |
|---|---|---|---|---|---|
| Baseline (unweighted MT rule) | 0.92 | 1.80 | 4.30 | 0.90 | 87.6 |
| Confidence-scaled (position × confidence) | 1.03 | 1.64 | 5.18 | 0.50 | 69.9 |
| Abstain on flagged (skip top 20% uncertain) | 1.08 | 2.51 | 4.36 | 0.62 | 59.6 |

**Calibration check** (is the uncertainty signal actually informative?): mean classification
accuracy on flagged (top 20% most uncertain, calibrated per fold on validation data) vs.
unflagged factor-months — **54.8% flagged (n=420) vs. 54.7% unflagged (n=1495)**. Essentially no
gap. Reported plainly: MC-Dropout uncertainty in this run doesn't detectably track which
months the model gets wrong more often, even though weighting by it still raises Sharpe —
plausibly because the uncertainty signal tracks return volatility/position risk more than
directional accuracy specifically, but that's a hypothesis, not something this pass verified.

Two things worth flagging honestly about how these numbers were produced:
- **Leakage fix, no prior baseline to compare against.** The confidence-scaled strategy
  originally normalized its position weights using the full 1990-2021 OOS uncertainty
  distribution's min/max — leaking full-sample information into early test years, unlike every
  other leakage rule in this project (validation-only, never test). This was caught and fixed
  (now normalized per fold, from that fold's own validation-set uncertainty) *before* this
  notebook was ever run to completion, so there's no previous "leaky" result from this repo to
  show a before/after against — the numbers above already reflect the fix.
- **The `l1=0.001` departure** described above was necessary to get a real uncertainty signal at
  all (at the paper's `l1`, MC-Dropout std was ~1e-7 — see "What's deliberately simplified").

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

`src/loading.py` pulls three sources on first import and caches each as a parquet file
under `data/` (gitignored — see below):

- **Response factors** (SMB, HML, RMW, CMA, momentum): Kenneth French's data library, via
  `pandas-datareader`.
- **Macro predictors** (122 series): the current FRED-MD release, transformed per
  McCracken and Ng (2016).
- **Financial predictors** (137 long–short anomaly portfolios): Chen and Zimmermann (2022) /
  Open Source Asset Pricing, via the `openassetpricing` package, which reads from a
  Google-Drive-hosted file subject to a **shared download quota** — if the fetch fails,
  it's often transient; retry later.

Subsequent imports read from the `data/*.parquet` cache instead of re-fetching. **If you ever
suspect the cache is stale or partial** (e.g. an interrupted download), delete the relevant
file(s) in `data/` and re-import — `loading.py` will re-fetch automatically, and raises a
clear `RuntimeError` at import time if the three sources don't actually overlap on enough
months (rather than silently producing an empty panel that only surfaces as a confusing error
several cells later).

**Note:** because the fetched frames are cached at *Python import* time, changing files in
`data/` requires restarting the notebook kernel before the change takes effect — a plain
re-run of a cell won't re-trigger the module-level fetch in an already-running kernel.

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
