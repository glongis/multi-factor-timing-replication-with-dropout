# Multi-Factor Timing with Deep Learning — Replication + Volatility-Regime Abstention Overlay

A partial replication of Cotturo, Liu, and Proner (2025), *"Multi-Factor Timing with Deep
Learning"* — forecasting the sign of next-month returns for five Fama-French/momentum factors
with a multi-task neural network, benchmarked against off-the-shelf ML models — extended with an
original addition: a **volatility-regime abstention overlay**, a simple, model-free "scanner"
that sits a factor out whenever its own trailing realized volatility is elevated. Of the four
risk-aware strategies tested here (three built on MC Dropout uncertainty, one on volatility), the
volatility scanner is the only one that clearly beats the always-invest baseline, and the only
one that catches all four of the worst diagnosed losses in the sample.

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
| MC-Dropout MT | 55.1% ¹ | — | 0.65 ² | — |

¹ single representative run (run 0 of the stability sweep).
² mean across the 5-run stability sweep; see the volatility scanner section for the full range.

MT lands close to the paper on both metrics despite running 80× fewer network fits (see the
hyperparameter table below), and clears the BUY benchmark on accuracy and Sharpe. The residual
gap to the paper is consistent with the single-seed / no-grid-search simplification, not with a
pipeline error — architecture, walk-forward procedure, and data construction otherwise match the
paper. The BUY benchmark reproduces the paper's published numbers to within a tenth of a point on
every individual factor, which is the strongest available evidence that the data pipeline is
built correctly.

**Run-to-run noise**: every neural-net number above moves between runs — TensorFlow training
isn't perfectly deterministic even with a fixed seed, and this repo uses one seed instead of the
paper's 10-seed ensemble. Observed MT accuracy has ranged 52–55% across repeated runs on the same
code and data, with Sharpe moving 0.55–0.67 over the same reruns; treat exact figures as
indicative, not precise. `mc-dropout-extension.ipynb` handles this directly for its own headline
numbers by reporting a 5-run range instead of a single draw — the other notebooks report one run
each.

**Trading-return correctness**: strategy returns are graded against the return realized the month
*after* the signal date (matching what each model actually predicts, `sign(r_{t+1})`), not the
return already known at signal time. `tests/test_estimation.py` regression-tests this alignment
with a real April 2009 momentum-crash fixture.

## Hyperparameters: this repo vs. the paper

The paper runs a full per-fold grid search with a 10-seed ensemble at every grid point. This repo
fixes one configuration per model and trains a single seed, which is the dominant source of the
remaining performance gap. Every fixed value below is a valid point in the paper's own grid
except where flagged.

### Estimation protocol (identical)

| Setting | Paper | This repo |
|---|---|---|
| Sample start | Jan 1965 | Jan 1965 |
| First training window | 23 years (1965–1987) | 23 years (1965–1987) |
| Validation window | 2 years, immediately pre-test | 2 years, immediately pre-test |
| Test window | 1 year, held out | 1 year, held out |
| Window scheme | Expanding train, rolling val/test | Expanding train, rolling val/test |
| Out-of-sample folds | 32 (1990–2021) | 32 (1990–2021) |
| Response factors | SMB, HML, RMW, CMA, MOM | SMB, HML, RMW, CMA, MOM |
| Market factor | Excluded | Excluded |
| Macro predictors | 122 (FRED-MD, McCracken–Ng transforms, lagged 1mo) | 122, same |
| Financial predictors | 137 long–short anomaly portfolios | 137 of 212 available (same filter) |
| Missing-value handling | Expanding training-set mean | Expanding training-set mean |
| Standardization | Train-sample mean/variance | Train-sample mean/variance |

### MT (multi-task neural network)

| Hyperparameter | Paper (Table IA1) | This repo | Match? |
|---|---|---|---|
| Hard-sharing layers | 4 × 32 units | 4 × 32 units | ✅ |
| Factor-specific layers | 2 × 8 units per factor | 2 × 8 units per factor | ✅ |
| Output activation | Sigmoid, one head per factor | Sigmoid, one head per factor | ✅ |
| L1 penalty | Grid {0.005, 0.007, 0.01, 0.02} | Fixed **0.01** | ⚠️ in grid, not searched |
| Learning rate | Grid {0.005, 0.001} | Fixed **0.001** | ⚠️ in grid, not searched |
| Batch size | 4 | 4 | ✅ |
| Max epochs | 200 | 200 | ✅ |
| Early-stopping patience | 20 | 20 | ✅ |
| Optimizer | Adam, default params | Adam, default params | ✅ |
| Seed ensemble | 10 seeds, averaged | **1 seed** | ❌ deviation |
| Normalization | Not specified | BatchNorm after each shared dense layer | ⚠️ implementation choice |
| **Total MT fits** | 32 folds × 8 grid points × 10 seeds = **2,560** | 32 folds × 1 × 1 = **32** | **80× fewer** |

### MC-Dropout MT (extension — no paper counterpart)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Architecture | Identical to MT above | Only the trunk normalization changes |
| Shared-trunk regularizer | Dropout, rate 0.2 (replaces BatchNorm) | MC Dropout needs dropout active at inference; leaving BatchNorm in would corrupt its running statistics |
| L1 penalty | **0.001** — off the paper's grid | Every grid value, including the smallest (0.005), collapses shared-trunk weights to ≈0 once BatchNorm is removed and nothing rescales activations against the penalty. Documented in `src/mcdropout.py`. |
| MC forward passes | 50 | Standard MC Dropout range is 30–100 |
| Uncertainty cutoff | 80th percentile of validation-set prediction std, per fold | Same per-fold, validation-only discipline as the volatility scanner |
| Stability sweep | 5 full reruns, identical seed | Isolates TensorFlow non-determinism |

Because the regularization strength changed alongside the architecture, this model's accuracy
relative to plain MT is **not** a clean dropout-vs-batch-norm comparison.

### Off-the-shelf benchmarks

| Model | Paper grid (Table IA1) | This repo | Match? |
|---|---|---|---|
| **RF** — trees | {50, 100, 200, 500, 1000} | Fixed **500** | ⚠️ in grid |
| **RF** — max depth | {1, 3, 5} | Fixed **5** | ⚠️ in grid |
| **RF** — features/split | √p | √p | ✅ |
| **GBT / XGBoost** — learning rate | {0.001, 0.01, 0.1} | Fixed **0.1** | ⚠️ in grid |
| **GBT / XGBoost** — trees | {50, 100, 200} | Fixed **200** | ⚠️ in grid |
| **GBT / XGBoost** — subsample | {0.25, 0.5, 1} | Fixed **0.5** | ⚠️ in grid |
| **GBT / XGBoost** — depth | 1–2 | Fixed **2** | ⚠️ in grid |
| **GBT / XGBoost** — implementation | Paper's own GBT | `xgboost.XGBClassifier` | ⚠️ substitute |
| **LR** | No grid searched | Unregularized, `max_iter=5000` | ✅ neither tuned |
| **EN, SVM, NN, LSTM, DMT, DMTc** | Grid-searched | Not implemented | ❌ out of scope |

### Known deviations beyond hyperparameters

- **FRED-MD vintage.** The macro panel is built from the *current* FRED-MD release rather than
  point-in-time vintages, so it embeds revisions made after the fact. `rf-investigation.ipynb`
  tests whether this drives Random Forest's outperformance and finds it does not (a macro-only RF
  gets Sharpe 0.69, in line with the paper's own RF).
- **Single-task LSTM benchmark.** Implemented, then removed — retraining it for all 32 folds × 5
  factors took roughly 45–50 minutes, which wasn't a good use of the remaining project budget
  relative to the extension.
- **Transaction costs.** Not modelled. The paper's Table 5 shows MT's t(α) eroding from 1.91 to
  1.84 at 1bp and 1.54 at 5bp; the abstention overlay adds turnover on top of that, so every
  alpha figure here is pre-cost.

## The volatility scanner

`notebooks/mc-dropout-extension.ipynb` computes trailing 3-month realized volatility for each
factor's own return series — no model, no look-ahead, only returns already known by the signal
date — and abstains (goes flat instead of long) whenever that trailing volatility exceeds a
cutoff. The cutoff is the 80th percentile of trailing volatility, calibrated separately for each
of the 32 walk-forward folds from that fold's *validation*-period data only, never test data. The
window and threshold (3 months, 80th percentile) were fixed in advance and not tuned against the
results below.

Multi-factor timing performance vs. the always-long BUY benchmark (Sharpe 0.60), **mean across 5
independent reruns of the exact same code and fixed seed** (range in parentheses):

| Strategy | Sharpe | alpha (ann. %) | t(alpha) | beta | R² (%) | Accuracy when positioned (%) |
|---|---|---|---|---|---|---|
| Baseline (unweighted MT rule) | 0.65 (0.55–0.70) | 0.48 (0.02–0.71) | 1.34 (0.07–1.95) | 0.88 (0.84–0.91) | 85.4 (83.5–87.1) | 55.2 (54.2–55.9) |
| MC-Dropout confidence-scaled | 0.61 (0.58–0.64) | 0.39 (0.34–0.49) | 1.18 (1.04–1.31) | 0.47 (0.44–0.50) | 64.2 (59.5–69.3) | 55.3 (53.6–56.1) |
| MC-Dropout abstain (top 20% uncertain) | 0.65 (0.60–0.77) | 0.81 (0.54–1.35) | 1.47 (1.03–2.44) | 0.59 (0.55–0.62) | 56.5 (53.0–59.4) | 56.0 (54.6–57.2) |
| **Volatility scanner (abstain)** | **0.70 (0.53–0.77)** | **1.07 (0.54–1.31)** | **2.24 (1.07–2.76)** | 0.44 (0.42–0.46) | 41.6 (38.9–43.8) | 55.2 (53.8–56.4) |

The scanner has the highest mean Sharpe and the highest mean t(alpha) of the four. Neither
MC-Dropout strategy improves on the baseline's Sharpe — confidence-scaling falls below it, and
abstain-on-flagged matches it. The scanner's t(alpha) clears conventional significance (t > 2) in
4 of 5 reruns, but the spread is wide: the worst rerun lands at 1.07, and its Sharpe in that same
rerun (0.53) falls below BUY's 0.60. So the edge is real on average and not a guarantee on any
single retraining. See "Run-to-run noise" above; the count of reruns clearing t > 2 has itself
moved between 2 and 4 across repeated executions of this notebook.

The more robust result is the *shape* of the return profile rather than the Sharpe improvement:
beta falls from 0.88 to 0.44 and R² from 85.4% to 41.6%, while alpha roughly doubles from 0.48%
to 1.07%. The strategy's returns come substantially less from carrying market exposure and more
from timing decisions — a change large enough to sit well outside the run-to-run noise in a way
the 0.05 Sharpe improvement does not.

Two checks on why it works (both use one representative run, since they're about specific
months and predictions rather than a distribution):

- **Hit rate on the four worst diagnosed losses**: Feb 2000 SMB, Dec 2008 HML, Apr 2009 MOM, and
  May 2021 HML were all correctly abstained — 4 for 4. Worth noting this is not fully independent
  evidence: trailing volatility is definitionally elevated around crashes, so catching them is
  close to what the signal is built to do.
- **Mostly independent of MC Dropout**: of all factor-months the scanner sits out (431, 22.5% of
  the sample), 38.1% were also flagged uncertain by MC Dropout — it's catching a largely different
  set of risky months, not relabeling the same ones.

That independence has a clear cause. A diagnostic on the MC-Dropout-flagged factor-months where
the baseline strategy went on to lose money (a long call that didn't pay off, since baseline only
ever invests when its predicted probability exceeds 50%) found they were **high-conviction calls,
not weak ones** — mean conviction (probability) 0.67 across 182 qualifying months, more
high-conviction (>0.70, n=68) than low-conviction (0.50–0.58, n=50). MC Dropout's 50 stochastic
sub-networks all learn the same about-to-break pattern together, so the model's self-assessed
uncertainty doesn't flag a genuine regime break in advance — a known failure mode, not a bug here.
That result also killed a plausible follow-up (conviction-gated abstention: only sit out when both
flagged uncertain *and* weakly convicted); it was never built because the diagnostic showed it
would have kept exactly the losing positions at full size.

### Does layering MC Dropout on top of the scanner add anything?

A further check asks whether MC Dropout's flag still separates good from bad calls *within* the
months the scanner already handles. If MC Dropout's signal were redundant with the scanner's, the
accuracy gap between flagged and unflagged months should be near zero inside both subsets.
Flagged months are the high-uncertainty ones, so an informative signal produces a **negative**
gap.

| Subset | MC-Dropout-flagged acc | MC-Dropout-unflagged acc | Gap (flagged − unflagged) | SE of gap |
|---|---|---|---|---|
| Scanner keeps (n=1484) | 51.3% (n=312) | 56.5% (n=1172) | −5.2pp | 3.2pp |
| Scanner abstains (n=431) | 53.7% (n=164) | 54.3% (n=267) | −0.7pp | 5.0pp |

In this run, MC Dropout's flag does carry information the scanner hasn't already used: within the
months the scanner keeps, flagged months are 5.2pp less accurate than unflagged ones, about 1.6
standard errors — directionally right, and not small. Within the months the scanner already
abstains from, the gap collapses to −0.7pp, which is what redundancy looks like.

**This diagnostic is not stable across reruns**, however. The kept-subset gap has come out at
−2.1pp, −0.2pp, and −5.2pp across three executions of identical code, spanning "clearly
informative" to "indistinguishable from zero." A layered strategy was therefore not built — that
is a scope decision under a fixed deadline, not a finding that layering doesn't work. Testing it
properly would need the gap estimated across a multi-run sweep rather than a single draw, which
is the natural next step for this repo.

**Caveats**: the scanner's flag itself is fully deterministic (computed from realized returns, no
model involved), but its measured Sharpe/alpha runs through a freshly retrained MT model each
time, which is why the table above reports a 5-run range rather than one number —
`mc-dropout-extension.ipynb` reruns the full walk-forward training (`src/mcdropout.py`'s
`run_stability`) 5 times with nothing changed but TensorFlow's own non-determinism. The threshold
(3-month window, 80th percentile) was a single reasonable default, fixed in advance and not swept
against alternatives. Abstaining also mechanically reduces invested capital, which can raise
Sharpe regardless of signal quality; a random-abstention placebo at the same 22.5% rate would be
the cleanest way to rule that out and has not been run.

## Why Random Forest beats both the paper's RF and MT

RF comes out of `off-the-shelf-models.ipynb` at Sharpe 0.84, well above the paper's own RF (0.66)
and its headline MT (0.69), from only a 0.7pp accuracy edge over the paper's RF.
`rf-investigation.ipynb` works through the candidate explanations and rules most of them out:

- **Not crisis-dodging.** RF sat out only 3 of the 15 worst BUY factor-months (LR avoided 9,
  XGBoost 10), and its Sharpe edge over MT *widens* when those months are dropped entirely
  (RF 0.84 → 1.15, MT 0.67 → 0.89).
- **Not seed luck.** Across 7 seeds, Sharpe ranged 0.844–0.941 (mean 0.898). Seed 0 — the one
  actually reported — is the lowest of the seven, so 0.84 is conservative.
- **Not a single leaked or near-duplicate feature.** Importances are diffuse: the top feature
  carries 0.8% of total importance and the top five 3.5%, against 0.39% for uniform.
- **Not FRED-MD revision leakage.** A macro-only RF gets Sharpe 0.69, right in line with the
  paper's own RF. If current-vintage revisions were driving the edge, macro-only should look
  inflated.
- **The actual driver: the anomaly-portfolio predictors, plus genuine per-factor skill.** A
  financial-predictors-only RF alone gets Sharpe 0.83 — nearly the whole edge, with macro adding
  marginal lift. Per factor, RF's advantage concentrates on HML (0.33 Sharpe, 55.1% accuracy vs
  BUY's 47.8%), SMB, and CMA — the factors where a single shared representation must compromise
  most. RF fits each factor independently and doesn't pay that cost.

Note that the paper's own MT *outperforms* its RF on Sharpe while being outperformed on
Diebold–Mariano forecast tests — the paper's framing is that better probability forecasts don't
automatically translate into better economic performance. In this replication RF leads on both,
which is the substantive deviation from the paper's result.

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
  mcdropout.py       # MC-Dropout MT model + walk-forward training + multi-run stability
                     # sweep, used by mc-dropout-extension.ipynb
notebooks/           # see table above
tests/               # regression test for signal/return alignment
data/                # cached parquet pulls (gitignored, regenerated on first import)
results/             # OOS predictions + strategy returns written by the notebooks
```

## Out of scope

This replicates the paper's static MT model, three off-the-shelf benchmarks, and adds the
volatility scanner — not a full reproduction. Not implemented: **DMT/DMTc** (the paper's dynamic
LSTM architectures), **full hyperparameter search + 10-seed ensembling**, the **JKP 149-factor
extension** and **Shapley-value importance** (paper Sec 5 / 4.6), **EN and SVM benchmarks**, and
a **single-task LSTM benchmark**. See the hyperparameter table above for the full deviation list.

## Reference

Cotturo, P., Liu, F., and Proner, R. (2025). *Multi-Factor Timing with Deep Learning.*

## License

[MIT](LICENSE)
