"""MC Dropout uncertainty for the MT architecture, used by `mc-dropout-extension.ipynb`.

Shared here (rather than defined inline in the notebook) for the same reason as
`estimation.py`: one copy of the model + walk-forward logic, so a single run and a
repeated-run stability sweep can't drift apart from each other.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping

import estimation as est

FACTOR_NAMES = est.FACTOR_NAMES

N_MC_SAMPLES = 50           # forward passes per prediction; paper's own NN ensembling uses 10 seeds, 30-100 is the usual MC Dropout range
UNCERTAINTY_QUANTILE = 0.80  # flag the most-uncertain 20% of factor-months, calibrated per fold on validation data

# l1=0.01 (the value used by every other neural net in this project, and one of four points in
# the paper's own Table IA1 grid for NN/LSTM/MT models {0.005, 0.007, 0.01, 0.02}) collapses this
# model's shared-trunk weights to ~0 during training: the plain MT model in initial-mt.ipynb uses
# the same l1 without issue because batch norm renormalizes activations regardless of raw weight
# scale, but this variant replaces batch norm with dropout (see the notebook's markdown), so
# nothing counteracts the L1 penalty and Adam just drives the input-dependent weights to zero,
# leaving predictions to be carried almost entirely by the (unregularized) bias. That makes the
# network effectively input-independent, so different dropout masks barely move the output and
# MC-Dropout uncertainty collapses to float32 noise (~1e-7) instead of a real signal -- confirmed
# empirically (shared_dense_1 max|weight| ~0.0005 at l1=0.01, vs ~0.16-0.28 once l1 is small
# enough). Checked every value in the paper's grid directly: 0.005 also collapses in most folds
# tested. l1=0.001 (off-grid, but the smallest deviation that reliably avoided collapse across
# every fold tested) is used for this model only -- a deliberate, documented departure from the
# paper's grid, driven by the batch-norm-to-dropout architecture swap, not a tuning choice to
# flatter the results.
L1_VALUE = 0.001

PERF_METRICS = ['Sharpe Ratio', 'alpha (annualized %)', 't(alpha)', 'beta', 'R2 (%)', 'Accuracy when positioned (%)']
POSITION_WEIGHT_THRESHOLD = 0.05  # confidence-scaled weight below this counts as "no position"


def build_mt_mcdropout_model(n_features, l1_value=L1_VALUE, learning_rate=0.001, dropout_rate=0.2, seed=0):
    """MT's shared trunk (4 hard-sharing layers, 32 units) and factor-specific heads (2 layers,
    8 units, per factor) are unchanged from build_mt_model in initial-mt.ipynb. The only
    architectural change: batch norm -> dropout in the shared trunk, so dropout can be forced
    on at prediction time (training=True) without also perturbing batch norm's statistics."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = Input(shape=(n_features,), name='predictors')
    x = inputs
    for i in range(4):
        x = layers.Dense(32, kernel_regularizer=regularizers.l1(l1_value), name=f'shared_dense_{i+1}')(x)
        x = layers.ReLU(name=f'shared_relu_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'shared_dropout_{i+1}')(x)
    shared_latent = x

    outputs = []
    for factor in FACTOR_NAMES:
        f = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1(l1_value),
                          name=f'{factor}_dense1')(shared_latent)
        f = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1(l1_value),
                          name=f'{factor}_dense2')(f)
        f_out = layers.Dense(1, activation='sigmoid', name=f'{factor}_output')(f)
        outputs.append(f_out)

    model = Model(inputs=inputs, outputs=outputs, name='MT_MCDropout')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={f'{factor}_output': 'binary_crossentropy' for factor in FACTOR_NAMES},
    )
    return model


def mc_dropout_predict(model, X, n_samples=N_MC_SAMPLES):
    """Run n_samples stochastic forward passes with dropout forced on. Returns an array of
    shape (n_samples, n_rows, n_factors); take .mean(axis=0) for the point estimate and
    .std(axis=0) for the uncertainty."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    samples = [
        np.concatenate([np.asarray(p) for p in model(X_tensor, training=True)], axis=1)
        for _ in range(n_samples)
    ]
    return np.stack(samples, axis=0)


def _train_walkforward(data, feature_cols, verbose=False):
    """Walk-forward train + MC-Dropout predict across all OOS folds (same expanding-window
    procedure as every other notebook in this project). Returns the raw per-factor-month
    predictions, uncertainty, and uncertainty flags, plus each fold's validation-set std
    bounds (needed to normalize the confidence-scaled strategy's weights per-fold rather than
    over the pooled OOS series -- a leakage fix).
    """
    mean_records, std_records, flag_records = [], [], []
    bound_min_records, bound_max_records = [], []

    for test_year in est.OOS_TEST_YEARS:
        train, val, test, X_train, X_val, X_test = est.prepare_fold(data, feature_cols, test_year)

        y_train = {f'{f}_output': train[f'{f}_label'].values.astype('float32') for f in FACTOR_NAMES}
        y_val = {f'{f}_output': val[f'{f}_label'].values.astype('float32') for f in FACTOR_NAMES}

        model = build_mt_mcdropout_model(n_features=X_train.shape[1])
        es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(
            X_train.values.astype('float32'), y_train,
            validation_data=(X_val.values.astype('float32'), y_val),
            epochs=200, batch_size=4, callbacks=[es], verbose=0,
        )

        val_mc = mc_dropout_predict(model, X_val.values)
        val_mean, val_std = val_mc.mean(axis=0), val_mc.std(axis=0)
        cutoff = np.quantile(val_std.flatten(), UNCERTAINTY_QUANTILE)
        val_std_min, val_std_max = val_std.min(axis=0), val_std.max(axis=0)

        test_mc = mc_dropout_predict(model, X_test.values)
        test_mean, test_std = test_mc.mean(axis=0), test_mc.std(axis=0)
        test_flag = test_std > cutoff

        cols_prob = [f'{f}_prob' for f in FACTOR_NAMES]
        cols_std = [f'{f}_std' for f in FACTOR_NAMES]
        cols_flag = [f'{f}_uncertain' for f in FACTOR_NAMES]
        mean_records.append(pd.DataFrame(test_mean, index=test.index, columns=cols_prob))
        std_records.append(pd.DataFrame(test_std, index=test.index, columns=cols_std))
        flag_records.append(pd.DataFrame(test_flag, index=test.index, columns=cols_flag))
        bound_min_records.append(pd.DataFrame(np.tile(val_std_min, (len(test), 1)), index=test.index, columns=cols_std))
        bound_max_records.append(pd.DataFrame(np.tile(val_std_max, (len(test), 1)), index=test.index, columns=cols_std))

        if verbose:
            print(f'{test_year}: cutoff={cutoff:.4f}  mean test uncertainty={test_std.mean():.4f}  '
                  f'flagged={test_flag.mean():.0%}')

    return dict(
        oos_mean=pd.concat(mean_records).sort_index(),
        oos_std=pd.concat(std_records).sort_index(),
        oos_flag=pd.concat(flag_records).sort_index(),
        oos_std_val_min=pd.concat(bound_min_records).sort_index(),
        oos_std_val_max=pd.concat(bound_max_records).sort_index(),
    )


def _vol_regime_flag(data, feature_cols, response_factors):
    """Trailing-3-month-volatility abstain flag, calibrated per fold on validation-period vol
    only (same discipline as the MC-Dropout uncertainty cutoff) -- see notebook markdown for
    why this is a separate, model-free hypothesis from MC-Dropout uncertainty."""
    raw_response = response_factors.rename(columns={'Mom': 'MOM'})[FACTOR_NAMES]
    trailing_vol = raw_response.rolling(3).std()

    records = []
    for test_year in est.OOS_TEST_YEARS:
        train, val, test, X_train, X_val, X_test = est.prepare_fold(data, feature_cols, test_year)
        val_vol = trailing_vol.loc[val.index, FACTOR_NAMES]
        vol_cutoff = np.quantile(val_vol.values.flatten(), UNCERTAINTY_QUANTILE)
        test_vol = trailing_vol.loc[test.index, FACTOR_NAMES]
        records.append(pd.DataFrame((test_vol > vol_cutoff).values, index=test.index, columns=FACTOR_NAMES))
    return pd.concat(records).sort_index()


def run_walkforward(data, feature_cols, response_factors, verbose=False):
    """One full MC-Dropout MT walk-forward run: trains all OOS folds, builds the four
    strategies (baseline, confidence-scaled, abstain-on-flagged, vol-regime-abstain) and their
    performance table. Everything a single run's worth of diagnostics needs, in one call.
    """
    wf = _train_walkforward(data, feature_cols, verbose=verbose)
    oos_mean, oos_std, oos_flag = wf['oos_mean'], wf['oos_std'], wf['oos_flag']
    oos_std_val_min, oos_std_val_max = wf['oos_std_val_min'], wf['oos_std_val_max']

    response = response_factors.rename(columns={'Mom': 'MOM'})
    # a signal formed at t predicts sign(r_{t+1}), so it must be graded against t+1's realized
    # return, not t's -- see estimation.strategy_returns' docstring for the full rationale.
    r = response.shift(-1).loc[oos_mean.index, FACTOR_NAMES]
    signal = pd.DataFrame({f: (oos_mean[f'{f}_prob'] > 0.5).astype(int) for f in FACTOR_NAMES}, index=oos_mean.index)

    baseline_strat = signal * r
    baseline_strat['EW'] = baseline_strat[FACTOR_NAMES].mean(axis=1)

    # per-fold, per-factor min-max normalize uncertainty to a [0, 1] confidence weight, using
    # each row's own fold's *validation-set* std min/max -- never test or the pooled OOS series,
    # which would leak the full-sample uncertainty distribution into early test years.
    std_vals = oos_std.rename(columns=lambda c: c.replace('_std', ''))
    bounds_min = oos_std_val_min.rename(columns=lambda c: c.replace('_std', ''))
    bounds_max = oos_std_val_max.rename(columns=lambda c: c.replace('_std', ''))
    value_range = (bounds_max - bounds_min).replace(0, 1)
    norm_std = (std_vals - bounds_min) / value_range
    confidence_weight = (1 - norm_std).clip(0, 1)

    scaled_strat = signal * confidence_weight * r
    scaled_strat['EW'] = scaled_strat[FACTOR_NAMES].mean(axis=1)

    flag_vals = oos_flag.rename(columns=lambda c: c.replace('_uncertain', '')).astype(bool)
    abstain_strat = signal * (~flag_vals).astype(int) * r
    abstain_strat['EW'] = abstain_strat[FACTOR_NAMES].mean(axis=1)

    oos_vol_flag = _vol_regime_flag(data, feature_cols, response_factors)
    vol_regime_strat = signal * (~oos_vol_flag).astype(int) * r
    vol_regime_strat['EW'] = vol_regime_strat[FACTOR_NAMES].mean(axis=1)

    buy_ew = response.shift(-1).loc[oos_mean.index, FACTOR_NAMES].mean(axis=1)

    y_true = pd.DataFrame({f: data.loc[oos_mean.index, f'{f}_label'] for f in FACTOR_NAMES})
    correct = (y_true == signal)

    baseline_position = signal == 1
    scaled_position = (signal == 1) & (confidence_weight > POSITION_WEIGHT_THRESHOLD)
    abstain_position = (signal == 1) & (~flag_vals)
    vol_position = (signal == 1) & (~oos_vol_flag)

    def accuracy_when_positioned(position_mask):
        return correct.values[position_mask.values].mean() * 100

    perf = pd.DataFrame({
        'baseline (no uncertainty)': [
            est.annualized_sharpe(baseline_strat['EW']), *est.spanning_regression(baseline_strat['EW'], buy_ew),
            accuracy_when_positioned(baseline_position),
        ],
        'confidence-scaled': [
            est.annualized_sharpe(scaled_strat['EW']), *est.spanning_regression(scaled_strat['EW'], buy_ew),
            accuracy_when_positioned(scaled_position),
        ],
        'abstain on flagged': [
            est.annualized_sharpe(abstain_strat['EW']), *est.spanning_regression(abstain_strat['EW'], buy_ew),
            accuracy_when_positioned(abstain_position),
        ],
        'vol-regime abstain': [
            est.annualized_sharpe(vol_regime_strat['EW']), *est.spanning_regression(vol_regime_strat['EW'], buy_ew),
            accuracy_when_positioned(vol_position),
        ],
    }, index=PERF_METRICS)

    return dict(
        oos_mean=oos_mean, oos_std=oos_std, oos_flag=oos_flag,
        oos_vol_flag=oos_vol_flag, signal=signal, confidence_weight=confidence_weight, flag_vals=flag_vals,
        correct=correct, baseline_strat=baseline_strat, scaled_strat=scaled_strat, abstain_strat=abstain_strat,
        vol_regime_strat=vol_regime_strat, buy_ew=buy_ew, perf=perf,
    )


def run_stability(data, feature_cols, response_factors, n_runs=5):
    """Repeat run_walkforward n_runs times -- same code, same fixed seed=0 every time, so the
    spread across runs isn't seed variation, it's TensorFlow's inherent run-to-run
    non-determinism (see the README's documented caveat). A single run's perf numbers can land
    anywhere in that spread; this reports the range instead, which is the number worth trusting.

    Returns (runs, all_perf, summary): `runs` is the list of n_runs raw run_walkforward() dicts
    (runs[0] is a perfectly good single run for one-off diagnostics, e.g. did-it-catch-this-
    specific-month checks, that don't make sense averaged across runs); `all_perf` stacks every
    run's perf table; `summary` is the min/mean/max of each metric across runs.
    """
    runs = [run_walkforward(data, feature_cols, response_factors) for _ in range(n_runs)]
    all_perf = pd.concat([r['perf'] for r in runs], keys=range(1, n_runs + 1), names=['run', 'metric'])
    summary = all_perf.groupby('metric', sort=False).agg(['min', 'mean', 'max']).reindex(PERF_METRICS)
    return runs, all_perf, summary
