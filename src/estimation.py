"""Shared walk-forward estimation and evaluation helpers for Cotturo, Liu, and Proner (2025).

Used by the MT notebook, the off-the-shelf benchmark notebook, and the MC Dropout
extension notebook so the OOS split, standardization, and scoring logic (paper Sections
3.1, 3.2.4, 4.2, 4.3) stay identical across all three.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_NAMES = ['SMB', 'HML', 'RMW', 'CMA', 'MOM']

# paper's OOS window: Jan 1990 - Dec 2021 (Section 3.2.4)
OOS_TEST_YEARS = range(1990, 2022)


def build_labels_and_panel(response_factors, macro_predictors, financial_predictors):
    """One row per month: macro + financial predictors, plus next-month up/down labels per factor."""
    response = response_factors.rename(columns={'Mom': 'MOM'})
    labels = (response.shift(-1) > 0).astype(int)
    labels.columns = [f'{c}_label' for c in labels.columns]
    labels = labels.iloc[:-1]

    feature_cols = macro_predictors.columns.tolist() + financial_predictors.columns.tolist()
    data = macro_predictors.join(financial_predictors, how='inner').join(labels, how='inner')
    return data, feature_cols


def fold_years(test_year):
    """Paper Sec 3.2.4: train on everything through test_year-3, validate on the 2 years
    immediately before test_year, test on test_year. Both windows shift forward each year,
    with the training window expanding."""
    train_end = f'{test_year - 3}-12-31'
    val_start = f'{test_year - 2}-01-01'
    val_end = f'{test_year - 1}-12-31'
    test_start = f'{test_year}-01-01'
    test_end = f'{test_year}-12-31'
    return train_end, val_start, val_end, test_start, test_end


def prepare_fold(data, feature_cols, test_year):
    """Split one walk-forward fold and standardize/impute using training-sample statistics only
    (paper Sec 3.1: expanding training-set mean for missing values; Sec 3.2.4: train-sample
    mean/variance for standardization) — val/test are transformed with train-only stats, never
    their own, to avoid leakage.
    """
    train_end, val_start, val_end, test_start, test_end = fold_years(test_year)

    train = data.loc[:train_end]
    val = data.loc[val_start:val_end]
    test = data.loc[test_start:test_end]

    X_train, X_val, X_test = train[feature_cols].copy(), val[feature_cols].copy(), test[feature_cols].copy()

    train_mean = X_train.mean()
    X_train = X_train.fillna(train_mean)
    X_val = X_val.fillna(train_mean)
    X_test = X_test.fillna(train_mean)

    mu, sigma = X_train.mean(), X_train.std().replace(0, 1)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    return train, val, test, X_train, X_val, X_test


def classification_accuracy(pred_prob, data):
    """Per-factor + mean OOS classification accuracy (paper Table 1)."""
    acc = {}
    for f in FACTOR_NAMES:
        y_true = data.loc[pred_prob.index, f'{f}_label']
        y_pred = (pred_prob[f'{f}_prob'] > 0.5).astype(int)
        acc[f] = (y_true == y_pred).mean()
    result = pd.Series(acc)
    result['Mean'] = result.mean()
    return result


def buy_and_hold_accuracy(data, pred_index):
    """BUY benchmark: always predicts positive (paper Table 1)."""
    acc = {}
    for f in FACTOR_NAMES:
        y_true = data.loc[pred_index, f'{f}_label']
        acc[f] = (y_true == 1).mean()
    result = pd.Series(acc)
    result['Mean'] = result.mean()
    return result


def strategy_returns(pred_prob, response_factors):
    """Equation (3): long factor i when predicted prob > 0.5, flat otherwise; EW across factors."""
    response = response_factors.rename(columns={'Mom': 'MOM'})
    r = response.loc[pred_prob.index, FACTOR_NAMES]
    signal = pd.DataFrame(
        {f: (pred_prob[f'{f}_prob'] > 0.5).astype(int) for f in FACTOR_NAMES}, index=pred_prob.index
    )
    strat = signal * r
    strat['EW'] = strat[FACTOR_NAMES].mean(axis=1)
    return strat


def annualized_sharpe(returns):
    return returns.mean() / returns.std() * np.sqrt(12)


def spanning_regression(strategy_ew, buy_ew, maxlags=6):
    """Time-series regression of strategy EW returns on the multi-factor BUY EW benchmark,
    with Newey-West standard errors (paper Sec 4.3)."""
    idx = strategy_ew.index.intersection(buy_ew.index)
    y = strategy_ew.loc[idx]
    X = sm.add_constant(buy_ew.loc[idx].rename('BUY'))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return pd.Series({
        'alpha (annualized %)': model.params['const'] * 12 * 100,
        't(alpha)': model.tvalues['const'],
        'beta': model.params['BUY'],
        'R2 (%)': model.rsquared * 100,
    })


def walk_forward_predict_single_task(data, feature_cols, factor, fit_predict_fn, test_years=OOS_TEST_YEARS):
    """Off-the-shelf models (paper Sec 3.2.3) estimate a separate functional form per factor,
    unlike MT. Loops the same walk-forward folds as MT, but for one factor at a time.

    fit_predict_fn(X_train, y_train, X_val, y_val, X_test) -> array of predicted probabilities
    for X_test, in row order. X_* are the standardized/imputed frames from prepare_fold; y_* are
    1-D arrays of 0/1 labels for `factor`.
    """
    records = []
    for test_year in test_years:
        train, val, test, X_train, X_val, X_test = prepare_fold(data, feature_cols, test_year)
        y_train = train[f'{factor}_label'].values.astype('float32')
        y_val = val[f'{factor}_label'].values.astype('float32')
        prob = fit_predict_fn(X_train, y_train, X_val, y_val, X_test)
        records.append(pd.Series(np.asarray(prob).flatten(), index=test.index))
    return pd.concat(records).sort_index()


def make_sequences(X, lookback):
    """Sliding windows of length `lookback` ending at each row of X (must be chronologically
    sorted). Rows with less than `lookback` months of history are left-zero-padded — after
    standardization, 0 represents the training-sample mean, a neutral fill for missing history.
    Used to feed the single-task LSTM benchmark a fixed window of predictor history per month,
    since the paper's Internet Appendix describes the LSTM's gating mechanics but not an explicit
    lookback length; 12 months (one macro/financial cycle) is this notebook's choice.
    """
    values = X.values.astype('float32')
    n, p = values.shape
    seqs = np.zeros((n, lookback, p), dtype='float32')
    for i in range(n):
        start = max(0, i - lookback + 1)
        window = values[start:i + 1]
        seqs[i, -window.shape[0]:] = window
    return seqs


def benchmark_summary(pred_prob, data, response_factors, model_name):
    """One-stop paper-style summary: accuracy (Table 1) + Sharpe/alpha/beta (Table 3) for a model."""
    acc = classification_accuracy(pred_prob, data)
    strat = strategy_returns(pred_prob, response_factors)
    buy_ew = response_factors.rename(columns={'Mom': 'MOM'}).loc[pred_prob.index, FACTOR_NAMES].mean(axis=1)
    sr = annualized_sharpe(strat['EW'])
    span = spanning_regression(strat['EW'], buy_ew)
    out = pd.Series({'Model': model_name, 'Mean Accuracy': acc['Mean'], 'Sharpe Ratio': sr})
    return pd.concat([out, span])
