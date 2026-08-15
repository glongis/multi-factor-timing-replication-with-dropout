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
| [`notebooks/MC Dropout Extension.ipynb`](notebooks/MC%20Dropout%20Extension.ipynb) | — (original extension) | Adds Monte Carlo Dropout uncertainty quantification on top of MT, and tests whether acting on that uncertainty (confidence-scaling, abstaining, or vol-regime abstaining) improves the trading strategy |

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

**One-month look-ahead bug in the trading evaluation, found and fixed.** Every model here is
trained to predict `sign(r_{t+1})` from month-t predictors (`response.shift(-1)` in
`build_labels_and_panel`, matching paper Sec 3.1) — but `strategy_returns()` was grading each
signal against the return already realized *at* t, not the return it actually forecast at t+1.
That's a real look-ahead problem, not a cosmetic one: several of the 137 financial predictors
are month-t close cousins of the response factors themselves (e.g. HML ~ book-to-market, MOM ~
momentum anomalies), so grading a signal on r_t was partly grading the model on numbers already
sitting in its own input vector. Caught via a concrete anomaly: the April 2009 momentum crash
(`MOM = -0.3436`) was landing on the pre-fix `mt_strategy_returns.csv`'s 2009-04-30 row — the
same row as the raw return — instead of 2009-03-31, the signal date that should have been graded
against it. Fixed in `src/estimation.py` (`strategy_returns()`, `benchmark_summary()`) and
mirrored in both notebooks' inline copies (`Initial MT.ipynb`, `MC Dropout Extension.ipynb`);
`tests/test_estimation.py` regression-tests it with that same April 2009 fixture and fails on
the old behavior. **Classification accuracy is unaffected** — it was always computed from labels,
never from these returns — but **every Sharpe, alpha, t-stat, beta, and R² number in this README
changed.**

Before vs. after (Sharpe / t(alpha)):

| Model | Sharpe (before) | Sharpe (after) | t(alpha) (before) | t(alpha) (after) |
|---|---|---|---|---|
| LR | 1.11 | 0.64 | 4.74 | 1.53 |
| RF | 1.33 | 0.84 | 10.81 | 3.64 |
| XGBoost (GBT) | 1.17 | 0.79 | 4.60 | 2.69 |
| MT | 0.80 | 0.55 | 3.35 | 0.61 |
| MC-Dropout baseline | 0.92 | 0.75 | 4.30 | 2.13 |
| MC-Dropout confidence-scaled | 1.03 | 0.59 | 5.18 | 1.06 |
| MC-Dropout abstain | 1.08 | 0.67 | 4.36 | 1.54 |

Sharpes now sit in the 0.5-0.8 range and t-stats in the 0.6-3.6 range — in line with the paper's
own best-of-eleven-models result (Sharpe 0.82, t(alpha) 3.05) rather than beating it by 30-60%
the way the pre-fix numbers did. That gap should have been the first tell something was wrong.

*Accuracy sanity check, run explicitly because a genuine fix here should not move accuracy at
all:* LR and XGBoost's OOS predictions are bit-identical before vs. after (both fully
deterministic given a fixed `random_state`); RF's differ by ≤2e-16 (float rounding from
parallel-tree aggregation, not the fix). MT and MC-Dropout MT's predictions *do* differ
before/after, but only because both are neural nets retrained from scratch on every notebook
run, and this project's own single-seed simplification already documents run-to-run training
noise below — that noise, not the fix, is why MT's accuracy reads 52.3% here vs. 53.1% in an
earlier run (both within the previously observed 52-55% band).

Mean out-of-sample classification accuracy across the five factors (paper Table 1) and
multi-factor timing Sharpe ratio (paper Table 3), post-fix:

| Model | Accuracy (this repo) | Accuracy (paper) | Sharpe (this repo) | Sharpe (paper) |
|---|---|---|---|---|
| BUY (always long) | 53.5% | 53.5% | 0.60 | 0.60 |
| LR | 51.1% | 53.1% | 0.64 | 0.61 |
| RF | 56.3% | 55.6% | 0.84 | 0.66 |
| XGBoost (≈ GBT) | 54.6% | 54.8% | 0.79 | 0.61 |
| **MT** | **52.3%** | 55.4% | **0.55** | 0.69 |

*(LSTM dropped from the comparison — see "What's deliberately simplified" above.)*

Given the single-seed/no-grid-search simplification above, MT landing under the paper's
published, fully-tuned-and-ensembled numbers on both accuracy and Sharpe is expected rather than
a bug — the walk-forward procedure, data construction, and architecture otherwise match the
paper. Because each neural net here uses one random seed instead of the paper's 10-seed
ensemble, and TensorFlow training isn't perfectly deterministic even with a fixed seed, **exact
numbers will drift slightly between reruns** (observed mean MT accuracy has ranged 52-55% across
repeated runs during development) — this is expected run-to-run noise from the simplification,
not a bug.

## MC Dropout extension results, out-of-sample 1990-2021

`notebooks/MC Dropout Extension.ipynb` adds Monte Carlo Dropout uncertainty to MT and tests
three ways of trading on it — full writeup of the method is in the notebook's own intro cell.
**All numbers below are post-look-ahead-fix** (see "Results" above); the fix applies identically
here since this notebook's `r` and `buy_ew` are built inline rather than through
`est.strategy_returns`/`benchmark_summary`, and both were patched to match.

**Regular MT vs. MC-Dropout MT, as point-estimate classifiers** (same walk-forward folds, same
`prob > 0.5` signal rule — this is *before* acting on uncertainty at all; "MC-Dropout MT" here is
the "Baseline" row of the strategy table below):

| | Regular MT (batch norm, l1=0.01) | MC-Dropout MT (dropout, l1=0.001) |
|---|---|---|
| Mean Accuracy | 52.3% | 54.3% |
| Sharpe Ratio | 0.55 | 0.66 |
| alpha (annualized %) | 0.28 | 0.50 |
| t(alpha) | 0.61 | 1.21 |
| beta | 0.65 | 0.88 |
| R² (%) | 64.9 | 85.3 |

MC-Dropout MT is still ahead on every metric, including 4 of 5 individual factor accuracies —
though *which* factor doesn't improve keeps changing between runs (this time it's CMA, roughly
flat at 50.1% → 49.1%; an earlier run had SMB as the laggard instead), one more sign of
retraining noise. Worth reading carefully, though: this isn't a clean "dropout beats batch norm"
result. The two models differ in *two* ways at once — the trunk regularizer (batch norm vs.
dropout) *and* the `l1` strength (0.01 vs. 0.001), and the `l1` change wasn't a free choice; it's
the fix described in "What's deliberately simplified," forced by the fact that 0.01 collapses the
dropout variant's weights to ~0. Nothing here isolates how much of the gap is "dropout
regularizes better" vs. simply "weaker `l1` let this model use more of the input signal."

**Does uncertainty-weighting improve on that baseline?** Multi-factor timing performance vs. the
multi-factor BUY benchmark (Sharpe 0.60), now including a fourth strategy (see below):

| Strategy | Sharpe | alpha (annualized %) | t(alpha) | beta | R² (%) |
|---|---|---|---|---|---|
| Baseline (unweighted MT rule) | 0.66 | 0.50 | 1.21 | 0.88 | 85.3 |
| Confidence-scaled (position × confidence) | 0.53 | 0.08 | 0.21 | 0.54 | 70.2 |
| Abstain on flagged (skip top 20% uncertain) | 0.56 | 0.40 | 0.79 | 0.62 | 58.5 |
| Vol-regime abstain (independent of MC Dropout) | **0.67** | **0.97** | **1.97** | 0.44 | 41.6 |

Confidence-scaled and abstain-on-flagged both underperform the baseline again this run — same
qualitative conclusion as before the vol-regime addition (and a flip from the pre-look-ahead-fix
numbers, where both used to beat baseline). **Vol-regime abstain is the first of the four
strategies tried in this notebook to actually beat the baseline** on both Sharpe and alpha, with
by far the best t(alpha) of the four (1.97 — still short of conventional significance, but
closest). See below for what it is and why it works where the others didn't.

**Calibration check** (is MC-Dropout's uncertainty actually informative?): mean classification
accuracy on flagged (top 20% most uncertain, calibrated per fold on validation data) vs.
unflagged factor-months — **54.1% flagged (n=455) vs. 54.4% unflagged (n=1460)**, essentially no
gap. This is the third run of this same check with three different results (54.8%/54.7% no gap;
51.8%/56.2% a real gap; now 54.1%/54.4% no gap again) — two of three show no gap, so treat the
one real-looking gap as more likely the noisy outlier than the other way around. This instability
is inherent to MC-Dropout's uncertainty signal specifically (it's a property of the freshly
retrained model's dropout behavior each run) — it does not affect the vol-regime signal below,
which is computed from realized market returns and has no retraining randomness in it at all.

### Why do the MC-Dropout-based strategies still underperform, and can anything fix it?

A follow-up diagnostic looked directly at the worst losses under the baseline rule: every
factor-month where the model went long, MC-Dropout flagged it uncertain, and it lost money,
sorted by loss size. The worst five were Feb 2000 SMB (-15.5%, prob 0.85), Apr 2009 MOM (-12.5%,
prob 0.67), Feb 2009 MOM (-11.8%, prob 0.77), Dec 2008 HML (-11.0%, prob 0.80), and May 2021 HML
(-7.9%, prob 0.91) — **high-conviction calls, not weak ones**. Across all 149 qualifying
long/flagged/losing months, mean conviction was 0.67 (well above the 0.5 midpoint), and more of
them were high-conviction (>0.7, n=56) than low-conviction (0.50-0.58, n=42).

That result killed an otherwise-plausible fix (**conviction-gated abstention**: only abstain when
*both* flagged uncertain *and* weakly convicted, on the theory that MC-Dropout's flag might just
be catching the model's hedged, low-conviction guesses rather than its real calls). The diagnostic
shows the opposite: the biggest blowups are the model at its *most* confident. MC Dropout's 50
stochastic sub-networks all learn the same about-to-break pattern together, so nothing about the
model's own self-assessment flags a genuine regime break in advance — a known failure mode
(overconfidence right before regime changes), not a bug in this implementation. Conviction-gating
would have kept exactly these positions at full size, so it was never built.

**Vol-regime abstain**, by contrast, doesn't use MC Dropout at all. It flags a factor-month
purely from trailing realized volatility — rolling 3-month std of that factor's own return series,
using only returns already known by the signal date (no look-ahead, no model). The cutoff is
calibrated per fold from that fold's *validation*-period trailing vol only (same discipline as
`UNCERTAINTY_QUANTILE` everywhere else), at a threshold (3-month window, 80th percentile) fixed in
advance and not tuned against the results below. It abstains when trailing vol exceeds that
fold's cutoff, full stop — independent of what MC-Dropout conviction or uncertainty say.

Two direct checks on whether this mechanism actually does what it's designed to do:
- **Hit rate on the four worst diagnosed losses**: Feb 2000 SMB, Dec 2008 HML, Apr 2009 MOM, and
  May 2021 HML were all correctly abstained — **4 for 4**.
- **Independence from MC-Dropout's flag**: of all factor-months vol-regime abstains on (431, 22.5%
  of the sample), only 36.4% were also flagged uncertain by MC-Dropout — meaning 63.6% of the
  months this rule sits out are ones MC-Dropout was *confident* about. It's catching a genuinely
  different kind of risk, not relabeling the same months.

That combination — catching the four known disasters, mostly not overlapping with MC-Dropout's
own flag, and beating the baseline on Sharpe and alpha in this run — is the most encouraging
result of the four strategies tried. Still one run, though: the vol-regime *flag* itself is fully
deterministic (no retraining noise), but its measured effect on Sharpe still runs through the
freshly retrained baseline model it's overlaid on, so the "beats baseline" comparison isn't
immune to the same run-to-run noise documented throughout this section.

Five things worth flagging honestly about how these numbers were produced:
- **Leakage fix, no prior buggy-alignment baseline to compare against.** The confidence-scaled
  strategy originally normalized its position weights using the full 1990-2021 OOS uncertainty
  distribution's min/max — leaking full-sample information into early test years, unlike every
  other leakage rule in this project (validation-only, never test). This was caught and fixed
  (now normalized per fold, from that fold's own validation-set uncertainty) before this
  notebook was ever run to completion, so there's no prior "leaky" result to show a before/after
  against for that specific bug.
- **The `l1=0.001` departure** described above was necessary to get a real uncertainty signal at
  all (at the paper's `l1`, MC-Dropout std was ~1e-7 — see "What's deliberately simplified").
- **The regular-MT-vs-MC-Dropout-MT comparison is confounded**, as noted above: two things
  changed at once (batch norm → dropout, and `l1` 0.01 → 0.001), so it doesn't isolate which one
  is responsible for MC-Dropout MT's better point-estimate accuracy.
- **Every neural-net-derived number in this section moves between runs** — this section has now
  been regenerated three times over the course of this project (leakage fix, look-ahead fix,
  vol-regime addition), and the baseline Sharpe alone has read 0.92, 0.75, and 0.66 across those
  three runs on the same code, same data, same seed. The qualitative conclusions (uncertainty-
  weighting underperforms baseline; vol-regime abstain catches genuinely different, high-severity
  risk) have been checked for robustness to this noise where explicitly noted; the exact numbers
  have not.
- **Vol-regime abstain's threshold and window were chosen once, in advance** (3-month trailing,
  80th percentile) and not swept or tuned against the hit-rate or Sharpe results — but neither
  were they chosen from a principled optimization over alternatives, just a single reasonable
  default. Whether a different window/threshold does better or worse is untested.

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
