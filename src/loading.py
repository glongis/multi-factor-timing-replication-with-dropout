import pathlib

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader.famafrench as famafrench
import openassetpricing as oap

# pandas_datareader hardcodes plain http:// for Ken French's data library. Some networks
# (e.g. ones behind a Meraki captive portal) intercept unauthenticated http:// requests and
# redirect them to a portal splash page instead, which breaks the zip download. Force https.
famafrench._URL = famafrench._URL.replace('http://', 'https://')

# baseline sample from Cotturo, Liu, and Proner (2025), "Multi-Factor Timing with Deep Learning"
SAMPLE_START = '1965-01-01'
SAMPLE_END = '2021-12-31'
OOS_CUTOFF = '1989-12-31'  # predictors missing any observation on/before this date are dropped (paper's OOS period starts 1990)

# Ken French / FRED-MD / OpenAP pulls are either slow or (for OpenAP, hosted on a shared
# Google Drive file) subject to a shared download quota that can fail transiently for
# everyone using the package, independent of anything in this repo. Cache each cleaned
# series to data/ on first successful pull so repeated notebook runs and kernel restarts
# don't need to re-hit these sources.
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def _load_or_fetch(name, fetch_fn):
    path = DATA_DIR / f'{name}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_fn()
    df.to_parquet(path)
    return df


def _fetch_response_factors():
    # 5 factors: Mkt-RF, SMB, HML, RMW, CMA, RF
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=SAMPLE_START)[0]

    # Momentum factor (separate file)
    mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=SAMPLE_START)[0]
    mom.columns = mom.columns.str.strip()  # this file's column name often has extra whitespace

    factors = ff5.join(mom, how='inner')
    factors = factors / 100  # convert from percent to decimal — easy to forget

    # PeriodIndex -> DatetimeIndex. to_timestamp() defaults to the *start* of each period;
    # realign to month-end so this index matches macro_predictors/financial_predictors below
    # (both of which are explicitly month-end) — otherwise every downstream join is empty.
    # Cast to a fixed ns resolution too: sources vary (some produce ms-resolution timestamps),
    # and joining mixed-resolution DatetimeIndexes is fragile across pandas versions.
    factors.index = (factors.index.to_timestamp() + pd.offsets.MonthEnd(0)).astype('datetime64[ns]')

    factors = factors.loc[SAMPLE_START:SAMPLE_END]

    # paper's 5 response factors (Mkt-RF is deliberately excluded so results aren't driven by market timing)
    return factors[['SMB', 'HML', 'RMW', 'CMA', 'Mom']]


def _apply_fredmd_tcode(series, code):
    if code == 1:
        return series
    if code == 2:
        return series.diff()
    if code == 3:
        return series.diff().diff()
    if code == 4:
        return np.log(series)
    if code == 5:
        return np.log(series).diff()
    if code == 6:
        return np.log(series).diff().diff()
    if code == 7:
        return series.pct_change().diff()
    raise ValueError(f'unknown FRED-MD transform code: {code}')


def _fetch_macro_predictors():
    # fred-md macro data cleaning, following McCracken and Ng (2016) as described in the paper:
    #   1. apply each series' McCracken-Ng transformation code to induce stationarity
    #   2. lag by one additional month (t-1) to account for data announcement/release delays
    #   3. drop any predictor with a missing value on/before OOS_CUTOFF — the paper's filter,
    #      which keeps 122 of the raw file's 126 series
    # remaining missing values (after 1990) are left as NaN here and imputed with the *expanding*
    # training-set mean inside the walk-forward estimation loop, since imputing them globally now
    # would leak future information into the training folds.
    # this is the *current* release; McCracken and Ng revise/update it monthly, so it reflects
    # revisions made since the paper's data as of writing.
    fredmd_url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
    fredmd_raw = pd.read_csv(fredmd_url)

    fredmd_tcodes = fredmd_raw.iloc[0, 1:].astype(int)
    fredmd = fredmd_raw.iloc[1:].copy()
    fredmd['sasdate'] = pd.to_datetime(fredmd['sasdate'])
    fredmd = fredmd.set_index('sasdate').astype(float)

    fredmd_stationary = pd.DataFrame(
        {col: _apply_fredmd_tcode(fredmd[col], fredmd_tcodes[col]) for col in fredmd.columns}
    )
    fredmd_stationary = fredmd_stationary.shift(1)
    # align to month-end and a fixed ns resolution, like the predictors below
    fredmd_stationary.index = (fredmd_stationary.index + pd.offsets.MonthEnd(0)).astype('datetime64[ns]')
    fredmd_stationary = fredmd_stationary.loc[SAMPLE_START:SAMPLE_END]

    fredmd_keep = fredmd_stationary.loc[:OOS_CUTOFF].notna().all()
    return fredmd_stationary.loc[:, fredmd_keep]


def _fetch_financial_predictors():
    # financial predictors: keep only the 'LS' (long-short) leg of each anomaly's decile-sorted
    # portfolio and reshape to one return column per anomaly.
    # 'op' is openassetpricing's decile + long-short anomaly portfolio return dataset — this is what
    # the paper actually uses ("long–short anomaly portfolio returns"), not firm-level characteristics
    openap = oap.OpenAP(2023)  # August 2023 release ("Version 1.30"); this package version keys it by year, not YYYYMM
    predictors_raw = openap.dl_port('op', 'pandas')

    ls_returns = predictors_raw[predictors_raw['port'] == 'LS'].copy()
    # OpenAP's 'date' column comes back at ms resolution rather than the ns resolution used by
    # macro_predictors/response_factors above; cast explicitly so the three indexes share a
    # resolution — mixed-resolution DatetimeIndex joins are fragile across pandas versions.
    ls_returns['date'] = (pd.to_datetime(ls_returns['date']) + pd.offsets.MonthEnd(0)).astype('datetime64[ns]')

    financial = ls_returns.pivot(index='date', columns='signalname', values='ret') / 100  # percent -> decimal
    financial = financial.loc[SAMPLE_START:SAMPLE_END]

    # same missing-value filter as fred-md: drop anomalies missing any observation on/before OOS_CUTOFF,
    # which keeps the paper's 137 of 212 available signals; remaining gaps are imputed later with the
    # expanding training-set mean inside the estimation loop
    financial_keep = financial.loc[:OOS_CUTOFF].notna().all()
    return financial.loc[:, financial_keep]


response_factors = _load_or_fetch('response_factors', _fetch_response_factors)
macro_predictors = _load_or_fetch('macro_predictors', _fetch_macro_predictors)
financial_predictors = _load_or_fetch('financial_predictors', _fetch_financial_predictors)

# A bad/partial cache (e.g. an interrupted fetch, or a stale file left over from a prior schema)
# silently produces a DataFrame that joins to zero rows several cells downstream, surfacing only
# as a cryptic Keras error deep inside model.predict() on an empty test set. Fail here instead,
# right at import time, with a message that points at the actual cause.
_overlap = response_factors.index.intersection(macro_predictors.index).intersection(financial_predictors.index)
if len(_overlap) < 500:
    raise RuntimeError(
        f'response_factors/macro_predictors/financial_predictors only overlap on {len(_overlap)} months '
        f'(expected ~650+). One of the cached data/*.parquet files is likely stale or from a partial '
        f'fetch. Delete the bad file(s) in data/ and re-import to force a fresh fetch.'
    )
