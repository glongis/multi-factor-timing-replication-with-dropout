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
| [`notebooks/initial-mt.ipynb`](notebooks/initial-mt.ipynb) | Builds and walk-forward trains the paper's MT architecture (4 shared dense layers + 2 factor-specific layers per factor) |
| [`notebooks/off-the-shelf-models.ipynb`](notebooks/off-the-shelf-models.ipynb) | Same walk-forward procedure, benchmarked against Logistic Regression, Random Forest, and XGBoost |
| [`notebooks/mc-dropout-extension.ipynb`](notebooks/mc-dropout-extension.ipynb) | Adds MC Dropout uncertainty on top of MT, and tests four ways of trading on risk signals — including the volatility scanner |
| [`notebooks/rf-investigation.ipynb`](notebooks/rf-investigation.ipynb) | Diagnostic on why Random Forest's Sharpe comes out well above both the paper's own RF and its headline MT model — read-only, depends on the first two |

**Run the first three in that order** — the latter two read data through the same
`src/loading.py` / `src/estimation.py` helpers as `initial-mt.ipynb`, and the benchmark cell in
`off-the-shelf-models.ipynb` picks up `results/mt_oos_predictions.csv` if it's already been
generated. `rf-investigation.ipynb` only reads results the first two already wrote to `results/`,
so it can run any time after those two.

## The volatility scanner

`notebooks/mc-dropout-extension.ipynb` computes trailing 3-month realized volatility for each
factor's own return series — no model, no look-ahead, only returns already known by the signal
date — and abstains (goes flat instead of long) whenever that trailing volatility exceeds a
cutoff. The cutoff is the 80th percentile of trailing volatility, calibrated separately for each
of the 32 walk-forward folds from that fold's *validation*-period data only, never test data. The
window and threshold (3 months, 80th percentile) were fixed in advance and not tuned against the
results below.

Multi-factor timing performance vs. the always-long BUY benchmark (Sharpe 0.60), **mean across 5
independent reruns of the exact same code and fixed seed** (range in parentheses) — see
"Run-to-run noise" below for why a single run isn't trustworthy enough to report alone here:

| Strategy | Sharpe | alpha (annualized %) | t(alpha) | beta | R² (%) |
|---|---|---|---|---|---|
| Baseline (unweighted MT rule) | 0.61 (0.58–0.68) | 0.29 (0.05–0.56) | 0.82 (0.20–1.69) | 0.89 (0.86–0.93) | 84.8 (80.4–88.8) |
| MC-Dropout confidence-scaled | 0.57 (0.54–0.60) | 0.30 (0.23–0.45) | 0.85 (0.60–1.20) | 0.48 (0.43–0.53) | 62.5 (56.3–70.2) |
| MC-Dropout abstain (top 20% uncertain) | 0.60 (0.42–0.70) | 0.54 (−0.33–1.03) | 0.98 (−0.56–1.79) | 0.64 (0.54–0.72) | 61.2 (46.4–68.3) |
| **Volatility scanner (abstain)** | **0.66 (0.59–0.77)** | **0.98 (0.80–1.30)** | **2.01 (1.50–2.73)** | 0.44 (0.42–0.46) | 40.0 (35.6–43.1) |

The scanner has the highest mean Sharpe and mean t(alpha) of the four. Its Sharpe stays in a tight
band around BUY's 0.60, dipping just below it in the worst of the five reruns (0.59) but clearing
it on average (0.66) and comfortably at best (0.77) — a real edge, not a guarantee every single
time it's retrained. t(alpha) clears conventional significance (t > 2) in 2 of 5 reruns, with a
third close behind — see "Run-to-run noise" below for how much this specific count has moved
across repeated reruns of this notebook. Two checks on why it works (both diagnostics below use
one representative run, since they're about specific months/predictions rather than a
distribution):

- **Hit rate on the four worst diagnosed losses**: Feb 2000 SMB, Dec 2008 HML, Apr 2009 MOM, and
  May 2021 HML were all correctly abstained — 4 for 4.
- **Mostly independent of MC Dropout**: of all factor-months the scanner sits out (431, 22.5% of
  the sample), roughly a third were also flagged uncertain by MC Dropout (30.2% in this run) —
  it's catching a different kind of risk, not relabeling the same months.

That independence has a clear cause. A diagnostic on the MC-Dropout-flagged factor-months where
the baseline strategy went on to lose money (a long call that didn't pay off, since baseline only
ever invests when its predicted probability exceeds 50%) found they were **high-conviction calls,
not weak ones** — mean conviction (probability) 0.69 across 147 qualifying months, more
high-conviction (>0.70, n=63) than low-conviction (0.50–0.58, n=27). MC Dropout's 50 stochastic
sub-networks all learn the same about-to-break pattern together, so the model's self-assessed
uncertainty doesn't flag a genuine regime break in advance — a known failure mode, not a bug here.
That result also killed a plausible follow-up (conviction-gated abstention: only sit out when both
flagged uncertain *and* weakly convicted); it was never built because the diagnostic showed it
would have kept exactly the losing positions at full size.

A further check tested whether *layering* MC Dropout's flag on top of the scanner adds value —
i.e., whether MC Dropout catches anything extra on the months the scanner doesn't already sit out:

| Subset | MC-Dropout-flagged acc | MC-Dropout-unflagged acc | Gap | SE of gap |
|---|---|---|---|---|
| Scanner keeps (n=1484) | 53.0% (n=279) | 54.0% (n=1205) | −1.0pp | 3.3pp |
| Scanner abstains (n=431) | 52.3% (n=130) | 54.2% (n=301) | −1.9pp | 5.2pp |

Both gaps are negative — MC-Dropout-flagged months are not even directionally more accurate than
unflagged ones within either subset, let alone by a margin that clears their own standard errors
(3.3pp and 5.2pp). MC Dropout's flag adds nothing on top of the scanner in this run, so a layered
strategy wasn't built.

**Caveats**: the scanner's flag itself is fully deterministic (computed from realized returns,
no model involved), but its measured Sharpe/alpha runs through a freshly retrained MT model each
time (see run-to-run noise below), which is why the table above reports a 5-run range rather than
one number — `notebooks/mc-dropout-extension.ipynb` reruns the full walk-forward training
(`src/mcdropout.py`'s `run_stability`) 5 times with nothing changed but TensorFlow's own
non-determinism. The threshold (3-month window, 80th percentile) was a single reasonable default,
fixed in advance and not swept against alternatives.

## Results vs. the paper, out-of-sample Jan 1990 – Dec 2021

Mean out-of-sample classification accuracy (paper Table 1) and multi-factor timing Sharpe ratio
(paper Table 3):

| Model | Accuracy (this repo) | Accuracy (paper) | Sharpe (this repo) | Sharpe (paper) |
|---|---|---|---|---|
| BUY (always long) | 53.5% | 53.5% | 0.60 | 0.60 |
| LR | 51.1% | 53.1% | 0.64 | 0.61 |
| RF | 56.3% | 55.6% | 0.84 | 0.66 |
| XGBoost (≈ GBT) | 54.6% | 54.8% | 0.79 | 0.61 |
| **MT** | **54.3%** | 55.4% | **0.67** | 0.69 |
| MC-Dropout MT | 53.8% | — | 0.58 | — |

MT landing under the paper's fully-tuned, 10-seed-ensembled numbers is expected given the
single-seed/no-grid-search simplification noted below, not a bug — architecture, walk-forward
procedure, and data construction otherwise match the paper.

**Run-to-run noise**: every neural-net number above moves between runs — TensorFlow training
isn't perfectly deterministic even with a fixed seed, and this repo uses one seed instead of the
paper's 10-seed ensemble. Observed MT accuracy has ranged 52–55% across repeated runs on the same
code and data; treat exact figures as indicative, not precise. `mc-dropout-extension.ipynb`
handles this directly for its own headline numbers by reporting a 5-run range instead of a single
draw (see the volatility scanner section above) — the other notebooks report one run each.

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
The MC-Dropout MT model above also uses `l1=0.001`, off the paper's own grid of
{0.005, 0.007, 0.01, 0.02}, because every grid value — including 0.005, the smallest — collapses
its shared-trunk weights to ~0 without batch norm to counteract the penalty (see
`src/mcdropout.py`) — so its accuracy edge over regular MT isn't purely a dropout-vs-batch-norm
comparison, since the regularization strength changed too.

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
                      # used identically by all four notebooks
  mcdropout.py        # MC-Dropout MT model + walk-forward training + multi-run stability
                      # sweep, used by mc-dropout-extension.ipynb
notebooks/            # see table above
data/                 # cached parquet pulls (gitignored, regenerated on first import)
results/               # OOS predictions + strategy returns written by the notebooks
```

## Reference

Cotturo, P., Liu, F., and Proner, R. (2025). *Multi-Factor Timing with Deep Learning.*

## License

[MIT](LICENSE)
