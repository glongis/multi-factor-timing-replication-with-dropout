"""Regression test for the t -> t+1 alignment bug in strategy_returns().

Run with: python -m unittest tests.test_estimation -v   (from the repo root)
"""
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd

import estimation as est


class TestStrategyReturnsAlignment(unittest.TestCase):
    def test_signal_at_t_is_graded_against_return_at_t_plus_1(self):
        """A signal formed at t predicts sign(r_{t+1}) (build_labels_and_panel's shift(-1)), so
        it must be graded against the return realized in t+1, not the return already known at t.

        Fixture: the real Ken French momentum factor return for April 2009 -- the momentum crash
        -- is -0.3436, recorded on the 2009-04-30 row. Before the fix, a long-MOM signal formed
        at 2009-03-31 was graded against MOM's own 2009-03-31 return; after the fix, it must be
        graded against 2009-04-30's -0.3436.
        """
        dates = pd.to_datetime(['2009-01-31', '2009-02-28', '2009-03-31', '2009-04-30', '2009-05-31'])
        response_factors = pd.DataFrame({
            'SMB': [0.0100, 0.0100, 0.0059, 0.0714, -0.0224],
            'HML': [0.0100, 0.0100, 0.0348, 0.0538, 0.0000],
            'RMW': [0.0100, 0.0100, -0.0258, 0.0134, -0.0081],
            'CMA': [0.0100, 0.0100, -0.0200, 0.0050, -0.0000],
            'Mom': [0.0100, 0.0100, -0.1180, -0.3436, -0.1254],
        }, index=dates)

        # long every factor at every signal date -- keeps the test about date alignment, not
        # about any particular model's predictions. Signal dates exclude the last response date
        # (2009-05-31) since a signal formed there has no following-month return to grade against.
        signal_dates = dates[:-1]
        pred_prob = pd.DataFrame(
            {f'{f}_prob': 1.0 for f in est.FACTOR_NAMES}, index=signal_dates
        )

        strat = est.strategy_returns(pred_prob, response_factors)

        # signal formed 2009-03-31 -> graded against 2009-04-30's return, not 2009-03-31's own.
        self.assertAlmostEqual(strat.loc['2009-03-31', 'MOM'], -0.3436, places=6)
        self.assertNotAlmostEqual(strat.loc['2009-03-31', 'MOM'], -0.1180, places=6)

        # sanity check on an unambiguous, non-repeated value for a second factor/date pair.
        self.assertAlmostEqual(strat.loc['2009-03-31', 'SMB'], 0.0714, places=6)

    def test_nan_guard_trips_when_signal_date_has_no_following_return(self):
        """strategy_returns() should refuse to silently return a NaN-laced result when a signal
        date is at or after the last date in response_factors."""
        dates = pd.to_datetime(['2009-01-31', '2009-02-28'])
        response_factors = pd.DataFrame(
            {f: [0.01, 0.01] for f in ['SMB', 'HML', 'RMW', 'CMA', 'Mom']}, index=dates
        )
        pred_prob = pd.DataFrame({f'{f}_prob': 1.0 for f in est.FACTOR_NAMES}, index=dates)  # includes the last date

        with self.assertRaises(AssertionError):
            est.strategy_returns(pred_prob, response_factors)


if __name__ == '__main__':
    unittest.main()
