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
| [`notebooks/Off-the-shelf Models.ipynb`](notebooks/Off-the-shelf%20Models.ipynb) | Sec 3.2.3 | Same walk-forward procedure, benchmarked against Logistic Regression, Random Forest, XGBoost (stand-in for the paper's GBT), and a single-task LSTM |
| [`notebooks/MC Dropout Extension.ipynb`](notebooks/MC%20Dropout%20Extension.ipynb) | — (original extension) | Adds Monte Carlo Dropout uncertainty quantification on top of MT, and tests whether acting on that uncertainty (confidence-scaling or abstaining) improves the trading strategy |
| [`notebooks/exploratory/start.ipynb`](notebooks/exploratory/start.ipynb) | — | Early scratch work for the data pipeline; superseded by `src/loadandclean.py`, kept for reference only |

**Run them in that order** — `Off-the-shelf Models.ipynb` and `MC Dropout Extension.ipynb` both
read the data through the same `src/loadandclean.py` / `src/estimation.py` helpers as
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
- **EN and SVM benchmarks** — not implemented (LR, RF, GBT, LSTM are).

Each notebook documents its own specific simplifications inline, next to the relevant code.

## Results (this repo vs. the paper), out-of-sample Jan 1990 – Dec 2021

Mean out-of-sample classification accuracy across the five factors (paper Table 1) and
multi-factor timing Sharpe ratio (paper Table 3):

| Model | Accuracy (this repo) | Accuracy (paper) | Sharpe (this repo) | Sharpe (paper) |
|---|---|---|---|---|
| BUY (always long) | 53.5% | 53.5% | 0.59 | 0.60 |
| LR | — | 53.1% | — | 0.61 |
| RF | — | 55.6% | — | 0.66 |
| XGBoost (≈ GBT) | — | 54.8% | — | 0.61 |
| LSTM | — | 53.8% | — | 0.76 |
| **MT** | **53.7%** | 55.4% | **0.60** | 0.69 |

*(Off-the-shelf model numbers fill in after running `Off-the-shelf Models.ipynb`; see that
notebook's final cell for a live comparison table each time it's run.)*

Given the single-seed/no-grid-search simplification above, MT landing a few points under the
paper's published, fully-tuned-and-ensembled number is expected rather than a bug — the walk-forward
procedure, data construction, and architecture otherwise match the paper. Because each model here
uses one random seed instead of the paper's 10-seed ensemble, and TensorFlow training isn't
perfectly deterministic even with a fixed seed, **exact numbers will drift slightly between reruns**
(observed mean MT accuracy has ranged 52-55% across repeated runs during development) — this is
expected run-to-run noise from the simplification, not a bug.

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

`src/loadandclean.py` pulls three sources on first import and caches each as a parquet file
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
file(s) in `data/` and re-import — `loadandclean.py` will re-fetch automatically, and raises a
clear `RuntimeError` at import time if the three sources don't actually overlap on enough
months (rather than silently producing an empty panel that only surfaces as a confusing error
several cells later).

**Note:** because the fetched frames are cached at *Python import* time, changing files in
`data/` requires restarting the notebook kernel before the change takes effect — a plain
re-run of a cell won't re-trigger the module-level fetch in an already-running kernel.

## Project structure

```
src/
  loadandclean.py   # data pull + cache (Ken French, FRED-MD, OpenAP)
  estimation.py      # shared walk-forward split / standardization / scoring helpers,
                      # used identically by all three notebooks
notebooks/            # see table above
data/                 # cached parquet pulls (gitignored, regenerated on first import)
results/               # OOS predictions + strategy returns written by the notebooks
```

## Reference

Cotturo, P., Liu, F., and Proner, R. (2025). *Multi-Factor Timing with Deep Learning.*
