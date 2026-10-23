"""
Known-answer and edge-case unit tests for quantstats.utils.

Every input is a small hand-built Series/DataFrame and every expected value is
derived by hand in the comment next to it, so the tests check the formula
rather than whatever the implementation currently returns. Nothing here
touches the network: anything that would download is monkeypatched.
"""

import datetime as dt
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from quantstats import utils
from quantstats.utils import DataValidationError


def _daily(values, start="2024-01-01", name=None):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float, name=name)


def _daily_frame(columns, start="2024-01-01"):
    n = len(next(iter(columns.values())))
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(columns, index=idx, dtype=float)


@pytest.fixture(autouse=True)
def _quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(autouse=True)
def _clear_prepare_returns_cache():
    utils._PREPARE_RETURNS_CACHE.clear()
    yield
    utils._PREPARE_RETURNS_CACHE.clear()


class TestMakeIndex:
    """utils.make_index with returns supplied, so nothing is downloaded."""

    @pytest.fixture
    def returns(self):
        return _daily_frame(
            {
                "A": [0.10, 0.00, -0.05, 0.02, 0.04],
                "B": [0.00, 0.02, 0.05, -0.01, 0.01],
            }
        )

    def test_weighted_sum_known_answer(self, returns):
        result = utils.make_index({"A": 0.6, "B": 0.4}, returns=returns)
        # Each day = 0.6 * A + 0.4 * B
        #   d1: 0.6 * 0.10 + 0.4 * 0.00  =  0.060 + 0.000 =  0.060
        #   d2: 0.6 * 0.00 + 0.4 * 0.02  =  0.000 + 0.008 =  0.008
        #   d3: 0.6 * -0.05 + 0.4 * 0.05 = -0.030 + 0.020 = -0.010
        #   d4: 0.6 * 0.02 + 0.4 * -0.01 =  0.012 - 0.004 =  0.008
        #   d5: 0.6 * 0.04 + 0.4 * 0.01  =  0.024 + 0.004 =  0.028
        assert isinstance(result, pd.Series)
        np.testing.assert_allclose(result, [0.060, 0.008, -0.010, 0.008, 0.028])
        assert result.index.equals(returns.index)

    def test_only_weighted_tickers_used(self, returns):
        returns = returns.assign(C=[1.0, 1.0, 1.0, 1.0, 1.0])
        result = utils.make_index({"A": 1.0}, returns=returns)
        # Column C is not in the weights, so the index is exactly 1.0 * A
        np.testing.assert_allclose(result, [0.10, 0.00, -0.05, 0.02, 0.04])

    def test_does_not_mutate_input(self, returns):
        before = returns.copy()
        utils.make_index({"A": 0.6, "B": 0.4}, returns=returns)
        pd.testing.assert_frame_equal(returns, before)

    def test_match_dates_starts_at_latest_first_nonzero(self):
        returns = _daily_frame(
            {"A": [0.0, 0.10, 0.20, 0.10], "B": [0.0, 0.0, 0.10, 0.20]}
        )
        result = utils.make_index(
            {"A": 0.6, "B": 0.4}, returns=returns, match_dates=True
        )
        # First non-zero: A on d2, B on d3 -> start at the later one, d3.
        #   d3: 0.6 * 0.20 + 0.4 * 0.10 = 0.12 + 0.04 = 0.16
        #   d4: 0.6 * 0.10 + 0.4 * 0.20 = 0.06 + 0.08 = 0.14
        np.testing.assert_allclose(result, [0.16, 0.14])
        assert result.index[0] == pd.Timestamp("2024-01-03")

    def test_rows_with_nan_are_dropped(self):
        returns = _daily_frame({"A": [0.01, np.nan, 0.03], "B": [0.02, 0.02, np.nan]})
        result = utils.make_index({"A": 0.5, "B": 0.5}, returns=returns)
        # d2 and d3 each have a NaN member and are dropped by dropna().
        #   d1: 0.5 * 0.01 + 0.5 * 0.02 = 0.005 + 0.010 = 0.015
        np.testing.assert_allclose(result, [0.015])
        assert list(result.index) == [pd.Timestamp("2024-01-01")]

    def test_missing_ticker_falls_back_to_download(self, returns, monkeypatch):
        calls = []

        def fake_download(ticker, period="max", proxy=None):
            calls.append((ticker, period))
            return pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=returns.index)

        monkeypatch.setattr(utils, "download_returns", fake_download)
        result = utils.make_index({"A": 0.5, "ZZZ": 0.5}, returns=returns, period="1y")
        # Only ZZZ is missing from `returns`, so only it is "downloaded".
        #   d1: 0.5 * 0.10 + 0.5 * 0.01  = 0.050 + 0.005 =  0.055
        #   d2: 0.5 * 0.00 + 0.5 * 0.01  = 0.000 + 0.005 =  0.005
        #   d3: 0.5 * -0.05 + 0.5 * 0.01 = -0.025 + 0.005 = -0.020
        #   d4: 0.5 * 0.02 + 0.5 * 0.01  = 0.010 + 0.005 =  0.015
        #   d5: 0.5 * 0.04 + 0.5 * 0.01  = 0.020 + 0.005 =  0.025
        assert calls == [("ZZZ", "1y")]
        np.testing.assert_allclose(result, [0.055, 0.005, -0.020, 0.015, 0.025])

    def test_single_row(self):
        returns = _daily_frame({"A": [0.02], "B": [0.04]})
        result = utils.make_index({"A": 0.5, "B": 0.5}, returns=returns)
        # 0.5 * 0.02 + 0.5 * 0.04 = 0.01 + 0.02 = 0.03
        np.testing.assert_allclose(result, [0.03])

    def test_constant_returns(self):
        returns = _daily_frame({"A": [0.01] * 5, "B": [0.03] * 5})
        result = utils.make_index({"A": 0.25, "B": 0.75}, returns=returns)
        # 0.25 * 0.01 + 0.75 * 0.03 = 0.0025 + 0.0225 = 0.025 every day
        np.testing.assert_allclose(result, [0.025] * 5)

    def test_rebalance_none_constant_returns(self):
        returns = _daily_frame({"A": [0.01] * 3, "B": [0.03] * 3})
        result = utils.make_index(
            {"A": 0.25, "B": 0.75}, rebalance=None, returns=returns
        )
        # Only d1 is asserted: from d2 on, buy-and-hold and daily rebalancing
        # differ, and that case is the xfail below.
        #   d1: 0.25 * 0.01 + 0.75 * 0.03 = 0.0025 + 0.0225 = 0.025
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(0.025)

    def test_all_nan_raises(self):
        returns = _daily_frame({"A": [np.nan] * 3, "B": [np.nan] * 3})
        # Every row is dropped, leaving no last day to slice up to.
        with pytest.raises(IndexError):
            utils.make_index({"A": 0.5, "B": 0.5}, returns=returns)

    def test_empty_raises(self):
        returns = pd.DataFrame(
            {"A": [], "B": []}, index=pd.DatetimeIndex([]), dtype=float
        )
        with pytest.raises(IndexError):
            utils.make_index({"A": 0.5, "B": 0.5}, returns=returns)

    @pytest.mark.xfail(
        strict=True,
        reason="rebalance is ignored: rebalance=None still rebalances daily",
    )
    def test_rebalance_none_is_buy_and_hold(self):
        returns = _daily_frame({"A": [0.10, 0.10], "B": [0.00, 0.00]})
        result = utils.make_index({"A": 0.5, "B": 0.5}, rebalance=None, returns=returns)
        # Buy and hold from 0.5 / 0.5, never rebalanced:
        #   after d1: A = 0.5 * 1.10 = 0.55, B = 0.50, total 1.05 -> r1 = 0.05
        #   after d2: A = 0.55 * 1.10 = 0.605, B = 0.50, total 1.105
        #             r2 = 1.105 / 1.05 - 1 = 0.0523809...
        # Daily rebalancing would instead give 0.5 * 0.10 = 0.05 on d2.
        np.testing.assert_allclose(result, [0.05, 1.105 / 1.05 - 1])


class TestAggregateReturns:
    """utils.aggregate_returns across every period keyword."""

    @pytest.fixture
    def returns(self):
        # 2023-12-30 Sat and 2023-12-31 Sun are ISO week 52 of 2023,
        # 2024-01-01 Mon is ISO week 1, 2024-02-15 Thu is ISO week 7,
        # 2024-04-01 Mon is ISO week 14.
        idx = pd.DatetimeIndex(
            ["2023-12-30", "2023-12-31", "2024-01-01", "2024-02-15", "2024-04-01"]
        )
        return pd.Series([0.10, -0.05, 0.02, 0.03, 0.01], index=idx)

    # Compounded building blocks used below:
    #   Dec 2023: 1.10 * 0.95 - 1 = 1.045 - 1 = 0.045
    #   Q1 2024:  1.02 * 1.03 - 1 = 1.0506 - 1 = 0.0506
    #   2024:     1.02 * 1.03 * 1.01 - 1 = 1.0506 * 1.01 - 1 = 1.061106 - 1 = 0.061106

    @pytest.mark.parametrize("period", [None, "day", "daily"])
    def test_no_aggregation(self, returns, period):
        # None and anything containing "day" return the input unchanged
        pd.testing.assert_series_equal(
            utils.aggregate_returns(returns, period), returns
        )

    def test_unknown_string_returns_input(self, returns):
        pd.testing.assert_series_equal(
            utils.aggregate_returns(returns, "fortnight"), returns
        )

    def test_month_groups_by_month_number(self, returns):
        result = utils.aggregate_returns(returns, "month")
        # Month numbers only (years merged): 1 -> 0.02, 2 -> 0.03, 4 -> 0.01,
        # 12 -> 0.045 (Dec 2023, compounded above)
        assert list(result.index) == [1, 2, 4, 12]
        np.testing.assert_allclose(result, [0.02, 0.03, 0.01, 0.045])

    def test_quarter(self, returns):
        result = utils.aggregate_returns(returns, "quarter")
        # Q1 -> 0.0506, Q2 -> 0.01, Q4 -> 0.045
        assert list(result.index) == [1, 2, 4]
        np.testing.assert_allclose(result, [0.0506, 0.01, 0.045])

    @pytest.mark.parametrize("period", ["year", "YE", "eoy", "yoy"])
    def test_year_compounded(self, returns, period):
        result = utils.aggregate_returns(returns, period)
        # 2023 -> 0.045, 2024 -> 0.061106
        assert list(result.index) == [2023, 2024]
        np.testing.assert_allclose(result, [0.045, 0.061106])

    def test_year_simple(self, returns):
        result = utils.aggregate_returns(returns, "year", compounded=False)
        # 2023: 0.10 - 0.05 = 0.05
        # 2024: 0.02 + 0.03 + 0.01 = 0.06
        np.testing.assert_allclose(result, [0.05, 0.06])

    def test_week_groups_by_iso_week_number(self, returns):
        result = utils.aggregate_returns(returns, "week")
        # ISO week 1 -> 0.02, 7 -> 0.03, 14 -> 0.01, 52 -> 0.045
        assert list(result.index) == [1, 7, 14, 52]
        np.testing.assert_allclose(result, [0.02, 0.03, 0.01, 0.045])

    @pytest.mark.parametrize("period", ["eow", "W"])
    def test_end_of_week(self, returns, period):
        result = utils.aggregate_returns(returns, period)
        # (2023, 52) -> 0.045, (2024, 1) -> 0.02, (2024, 7) -> 0.03,
        # (2024, 14) -> 0.01
        assert list(result.index) == [(2023, 52), (2024, 1), (2024, 7), (2024, 14)]
        np.testing.assert_allclose(result, [0.045, 0.02, 0.03, 0.01])

    @pytest.mark.parametrize("period", ["eom", "ME"])
    def test_end_of_month(self, returns, period):
        result = utils.aggregate_returns(returns, period)
        # (2023, 12) -> 0.045, (2024, 1) -> 0.02, (2024, 2) -> 0.03,
        # (2024, 4) -> 0.01
        assert list(result.index) == [(2023, 12), (2024, 1), (2024, 2), (2024, 4)]
        np.testing.assert_allclose(result, [0.045, 0.02, 0.03, 0.01])

    @pytest.mark.parametrize("period", ["eoq", "QE"])
    def test_end_of_quarter(self, returns, period):
        result = utils.aggregate_returns(returns, period)
        # (2023, 4) -> 0.045, (2024, 1) -> 0.0506, (2024, 2) -> 0.01
        assert list(result.index) == [(2023, 4), (2024, 1), (2024, 2)]
        np.testing.assert_allclose(result, [0.045, 0.0506, 0.01])

    def test_custom_non_string_grouper(self, returns):
        result = utils.aggregate_returns(returns, ["a", "a", "b", "b", "b"])
        # a: 1.10 * 0.95 - 1 = 0.045
        # b: 1.02 * 1.03 * 1.01 - 1 = 0.061106
        np.testing.assert_allclose(result, [0.045, 0.061106])

    def test_tz_aware_index_grouped_in_utc(self):
        idx = pd.DatetimeIndex(
            ["2024-01-15 12:00", "2024-01-31 22:00"], tz="US/Eastern"
        )
        returns = pd.Series([0.01, 0.02], index=idx)
        result = utils.aggregate_returns(returns, "eom")
        # 2024-01-15 12:00 EST = 17:00 UTC -> January
        # 2024-01-31 22:00 EST = 2024-02-01 03:00 UTC -> February
        assert list(result.index) == [(2024, 1), (2024, 2)]
        np.testing.assert_allclose(result, [0.01, 0.02])

    def test_nan_skipped_within_group(self):
        returns = _daily([0.10, np.nan, 0.05])
        result = utils.aggregate_returns(returns, "year")
        # NaN means "no observation": 1.10 * 1.05 - 1 = 1.155 - 1 = 0.155
        np.testing.assert_allclose(result, [0.155])

    def test_dataframe_per_column(self):
        df = _daily_frame({"A": [0.10, -0.05], "B": [0.02, 0.03]}, start="2024-01-30")
        result = utils.aggregate_returns(df, "eom")
        # 2024-01-30 / 2024-01-31 are both January 2024:
        #   A: 1.10 * 0.95 - 1 = 0.045
        #   B: 1.02 * 1.03 - 1 = 0.0506
        assert list(result.index) == [(2024, 1)]
        np.testing.assert_allclose(result["A"], [0.045])
        np.testing.assert_allclose(result["B"], [0.0506])

    def test_single_value(self):
        result = utils.aggregate_returns(_daily([0.07]), "year")
        # One observation compounds to itself: 1.07 - 1 = 0.07
        np.testing.assert_allclose(result, [0.07])

    def test_constant_returns(self):
        result = utils.aggregate_returns(_daily([0.01] * 3), "year")
        # 1.01 ** 3 - 1 = 1.030301 - 1 = 0.030301
        np.testing.assert_allclose(result, [0.030301])

    def test_empty(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        result = utils.aggregate_returns(empty, "year")
        assert len(result) == 0

    @pytest.mark.xfail(
        strict=True,
        reason="eow pairs calendar year with ISO week, merging late Dec into week 1",
    )
    def test_end_of_week_across_iso_year_boundary(self):
        # 2024-01-01 is ISO (2024, week 1); 2024-12-30 is ISO (2025, week 1),
        # but its calendar year is 2024, so [year, isoweek] gives (2024, 1)
        # for both. They are 52 weeks apart and should stay separate:
        #   expected: two groups, 0.01 and 0.02
        #   actual:   one group (2024, 1) = 1.01 * 1.02 - 1 = 0.0302
        returns = pd.Series(
            [0.01, 0.02], index=pd.DatetimeIndex(["2024-01-01", "2024-12-30"])
        )
        result = utils.aggregate_returns(returns, "eow")
        np.testing.assert_allclose(result, [0.01, 0.02])

    @pytest.mark.xfail(
        strict=True,
        reason="a period with only NaN returns is reported as a 0% return",
    )
    def test_all_nan_period_is_nan(self):
        # No observations in 2024 at all; "no data" should stay NaN rather
        # than become 0.0 (the module keeps missing days as NaN elsewhere).
        result = utils.aggregate_returns(_daily([np.nan, np.nan]), "year")
        assert np.isnan(result.iloc[0])


class TestValidateInput:
    """utils.validate_input type, emptiness, NaN and index checks."""

    def test_valid_series_returns_true(self):
        assert utils.validate_input(_daily([0.01, -0.02, 0.03])) is True

    @pytest.mark.parametrize("bad", [[0.01, 0.02], np.array([0.01]), 0.5, "x"])
    def test_non_pandas_raises(self, bad):
        with pytest.raises(DataValidationError, match="must be pandas Series"):
            utils.validate_input(bad)

    def test_empty_raises_unless_allowed(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        with pytest.raises(DataValidationError, match="cannot be empty"):
            utils.validate_input(empty)
        assert utils.validate_input(empty, allow_empty=True) is True

    def test_all_nan_raises_unless_allowed(self):
        nans = _daily([np.nan, np.nan, np.nan])
        with pytest.raises(DataValidationError, match="only NaN"):
            utils.validate_input(nans)
        assert utils.validate_input(nans, allow_empty=True) is True

    def test_all_nan_dataframe_raises(self):
        df = _daily_frame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
        with pytest.raises(DataValidationError, match="only NaN"):
            utils.validate_input(df)

    def test_dataframe_with_partial_nan_rows_passes(self):
        # dropna() drops any row with a NaN; d2 has no NaN so data remains
        df = _daily_frame({"A": [np.nan, 0.01], "B": [0.02, 0.03]})
        assert utils.validate_input(df) is True

    def test_single_value_and_constant(self):
        assert utils.validate_input(_daily([0.01])) is True
        assert utils.validate_input(_daily([0.0] * 5)) is True

    def test_range_index_left_alone(self):
        s = pd.Series([0.01, 0.02, 0.03])
        assert utils.validate_input(s) is True
        assert isinstance(s.index, pd.RangeIndex)

    def test_string_date_index_converted_in_place(self):
        s = pd.Series([0.01, 0.02], index=["2024-01-01", "2024-01-02"])
        assert utils.validate_input(s) is True
        # The string labels are parsed onto the caller's object
        assert isinstance(s.index, pd.DatetimeIndex)
        assert list(s.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]

    def test_unparseable_index_raises(self):
        s = pd.Series([0.01, 0.02], index=["alpha", "beta"])
        with pytest.raises(DataValidationError, match="valid datetime index"):
            utils.validate_input(s)

    @pytest.mark.xfail(
        strict=True,
        reason="an integer index is accepted as nanoseconds since 1970",
    )
    def test_integer_index_is_rejected(self):
        # Index([10, 20, 30]) is not a RangeIndex, so it goes through
        # pd.to_datetime and silently becomes 1970-01-01 00:00:00.000000010,
        # .000000020 and .000000030: three "dates" 10 nanoseconds apart.
        s = pd.Series([0.01, 0.02, 0.03], index=[10, 20, 30])
        with pytest.raises(DataValidationError):
            utils.validate_input(s)


class TestMakePortfolio:
    """utils.make_portfolio balance paths for every mode."""

    @pytest.fixture
    def returns(self):
        return _daily([0.10, -0.05, 0.02])

    def test_comp_known_answer(self, returns):
        result = utils.make_portfolio(returns, start_balance=100)
        # Prepends the start balance one day before the first return:
        #   2023-12-31: 100
        #   d1: 100 * 1.10 = 110
        #   d2: 110 * 0.95 = 104.5
        #   d3: 104.5 * 1.02 = 106.59
        np.testing.assert_allclose(result, [100, 110, 104.5, 106.59])
        assert result.index[0] == pd.Timestamp("2023-12-31")
        assert list(result.index[1:]) == list(returns.index)

    def test_mode_is_case_insensitive(self, returns):
        result = utils.make_portfolio(returns, start_balance=100, mode="COMPSUM")
        # Same compounding path as above
        np.testing.assert_allclose(result, [100, 110, 104.5, 106.59])

    @pytest.mark.parametrize("mode", ["cumsum", "sum"])
    def test_cumsum_known_answer(self, returns, mode):
        result = utils.make_portfolio(returns, start_balance=100, mode=mode)
        # 100 + 100 * cumsum(r): cumsum = 0.10, 0.05, 0.07
        #   -> 110, 105, 107
        np.testing.assert_allclose(result, [100, 110, 105, 107])

    def test_round_to(self, returns):
        result = utils.make_portfolio(returns, start_balance=100, round_to=1)
        # 106.59 rounds to 106.6; 100, 110, 104.5 are already 1-dp
        np.testing.assert_allclose(result, [100, 110, 104.5, 106.6])

    def test_dataframe(self):
        df = _daily_frame({"A": [0.10, -0.05, 0.02], "B": [0.0, 0.0, 0.0]})
        result = utils.make_portfolio(df, start_balance=100)
        # A: 100, 110, 104.5, 106.59 (as above); B never moves: 100 x 4.
        # The helper column 0 from the start-balance Series is dropped.
        assert list(result.columns) == ["A", "B"]
        np.testing.assert_allclose(result["A"], [100, 110, 104.5, 106.59])
        np.testing.assert_allclose(result["B"], [100, 100, 100, 100])
        assert result.index[0] == pd.Timestamp("2023-12-31")

    def test_single_value(self):
        result = utils.make_portfolio(_daily([0.05]), start_balance=100)
        # 100, then 100 * 1.05 = 105
        np.testing.assert_allclose(result, [100, 105])

    def test_constant_returns(self):
        result = utils.make_portfolio(_daily([0.01] * 3), start_balance=100)
        # 100, 101, 101 * 1.01 = 102.01, 102.01 * 1.01 = 103.0301
        np.testing.assert_allclose(result, [100, 101, 102.01, 103.0301])

    def test_all_nan_stays_at_start_balance(self):
        result = utils.make_portfolio(_daily([np.nan] * 3), start_balance=100)
        # Missing days are filled as flat when the price path is built
        np.testing.assert_allclose(result, [100, 100, 100, 100])

    def test_empty(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        result = utils.make_portfolio(empty, start_balance=100)
        assert len(result) == 0

    @pytest.mark.xfail(
        strict=True,
        reason="'fixed amount' mode scales each day by the previous day's return",
    )
    def test_fixed_amount_mode(self, returns):
        result = utils.make_portfolio(returns, start_balance=100, mode="fixed")
        # Docstring: "Fixed amount every day" -> 100 is at stake each day:
        #   P&L = 100 * r = 10, -5, 2; balance = 110, 105, 107
        # Actual P&L is (100 + 100 * r[t-1]) * r[t]:
        #   d1: 100 * 0.10 = 10
        #   d2: (100 + 10) * -0.05 = -5.5
        #   d3: (100 - 5) * 0.02 = 1.9
        #   balance = 110, 104.5, 106.4
        np.testing.assert_allclose(result, [100, 110, 105, 107])


class TestToExcessReturns:
    """utils.to_excess_returns with scalar, int, deannualized and Series rf."""

    @pytest.fixture
    def returns(self):
        return _daily([0.03, 0.01, -0.02])

    def test_float_rf(self, returns):
        result = utils.to_excess_returns(returns, 0.01)
        # 0.03 - 0.01 = 0.02, 0.01 - 0.01 = 0.0, -0.02 - 0.01 = -0.03
        np.testing.assert_allclose(result, [0.02, 0.0, -0.03], atol=1e-15)

    def test_int_rf(self, returns):
        result = utils.to_excess_returns(returns, 1)
        # int 1 -> 1.0: 0.03 - 1 = -0.97, 0.01 - 1 = -0.99, -0.02 - 1 = -1.02
        assert result.dtype == float
        np.testing.assert_allclose(result, [-0.97, -0.99, -1.02])

    def test_nperiods_deannualizes(self):
        returns = _daily([0.15, 0.10, 0.05])
        result = utils.to_excess_returns(returns, 0.21, nperiods=2)
        # per-period rf = (1 + 0.21) ** (1 / 2) - 1 = 1.1 - 1 = 0.10
        # 0.15 - 0.10 = 0.05, 0.10 - 0.10 = 0.0, 0.05 - 0.10 = -0.05
        np.testing.assert_allclose(result, [0.05, 0.0, -0.05], atol=1e-12)

    def test_series_rf_is_aligned_and_forward_filled(self, returns):
        rf = pd.Series(
            [0.01, 0.02], index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"])
        )
        result = utils.to_excess_returns(returns, rf)
        # rf reindexed to d1..d3 = [0.01, NaN, 0.02] -> ffill -> [0.01, 0.01, 0.02]
        # 0.03 - 0.01 = 0.02, 0.01 - 0.01 = 0.0, -0.02 - 0.02 = -0.04
        np.testing.assert_allclose(result, [0.02, 0.0, -0.04], atol=1e-15)

    def test_series_rf_leading_gap_is_back_filled(self, returns):
        rf = pd.Series([0.01], index=pd.DatetimeIndex(["2024-01-02"]))
        result = utils.to_excess_returns(returns, rf)
        # rf reindexed = [NaN, 0.01, NaN] -> ffill [NaN, 0.01, 0.01]
        #   -> bfill [0.01, 0.01, 0.01]
        # 0.03 - 0.01 = 0.02, 0.01 - 0.01 = 0.0, -0.02 - 0.01 = -0.03
        np.testing.assert_allclose(result, [0.02, 0.0, -0.03], atol=1e-15)

    def test_series_rf_with_no_overlap_is_zero(self, returns):
        rf = pd.Series([0.05], index=pd.DatetimeIndex(["2030-01-01"]))
        result = utils.to_excess_returns(returns, rf)
        # No rf date falls on a return date: all NaN after reindex,
        # ffill/bfill have nothing to copy, fillna(0.0) -> rf = 0
        np.testing.assert_allclose(result, [0.03, 0.01, -0.02])

    def test_dataframe_with_series_rf_subtracts_row_wise(self):
        df = _daily_frame({"A": [0.03, 0.01, -0.02], "B": [0.00, 0.02, 0.04]})
        rf = _daily([0.01, 0.02, 0.03])
        result = utils.to_excess_returns(df, rf)
        # A: 0.03 - 0.01 = 0.02, 0.01 - 0.02 = -0.01, -0.02 - 0.03 = -0.05
        # B: 0.00 - 0.01 = -0.01, 0.02 - 0.02 = 0.0, 0.04 - 0.03 = 0.01
        assert list(result.columns) == ["A", "B"]
        np.testing.assert_allclose(result["A"], [0.02, -0.01, -0.05], atol=1e-15)
        np.testing.assert_allclose(result["B"], [-0.01, 0.0, 0.01], atol=1e-15)

    def test_dataframe_with_scalar_rf(self):
        df = _daily_frame({"A": [0.03, 0.01], "B": [0.00, 0.02]})
        result = utils.to_excess_returns(df, 0.01)
        # A: 0.02, 0.0   B: -0.01, 0.01
        np.testing.assert_allclose(result["A"], [0.02, 0.0], atol=1e-15)
        np.testing.assert_allclose(result["B"], [-0.01, 0.01], atol=1e-15)

    def test_tz_aware_index_made_naive_utc(self):
        idx = pd.DatetimeIndex(
            ["2024-01-01 20:00", "2024-01-02 20:00"], tz="US/Eastern"
        )
        returns = pd.Series([0.03, 0.01], index=idx)
        result = utils.to_excess_returns(returns, 0.01)
        # 20:00 EST = 01:00 UTC next day; values 0.02, 0.0
        assert result.index.tz is None
        assert list(result.index) == [
            pd.Timestamp("2024-01-02 01:00"),
            pd.Timestamp("2024-01-03 01:00"),
        ]
        np.testing.assert_allclose(result, [0.02, 0.0], atol=1e-15)

    def test_single_and_constant(self):
        # 0.05 - 0.01 = 0.04;  [0.02] * 3 - 0.02 = [0, 0, 0]
        np.testing.assert_allclose(
            utils.to_excess_returns(_daily([0.05]), 0.01), [0.04]
        )
        np.testing.assert_allclose(
            utils.to_excess_returns(_daily([0.02] * 3), 0.02), [0.0] * 3
        )

    def test_empty_and_all_nan(self):
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils.to_excess_returns(empty, 0.01)) == 0
        # NaN - 0.01 stays NaN
        assert utils.to_excess_returns(_daily([np.nan] * 2), 0.01).isna().all()


class TestDownloadReturns:
    """utils.download_returns with yfinance replaced by a fake (no network)."""

    @pytest.fixture
    def fake_yf(self, monkeypatch):
        calls = []
        state = {"close": None}

        def fake_download(proxy=None, **params):
            calls.append({"proxy": proxy, **params})
            return pd.DataFrame({"Close": state["close"]})

        monkeypatch.setattr(utils, "safe_yfinance_download", fake_download)
        return calls, state

    def test_known_answer_and_params(self, fake_yf):
        calls, state = fake_yf
        state["close"] = _daily([100.0, 110.0, 99.0, 99.0])
        result = utils.download_returns("SPY", period="1y", proxy="http://p")
        # pct_change, first day filled with 0:
        #   d1: NaN -> 0
        #   d2: 110 / 100 - 1 = 0.10
        #   d3: 99 / 110 - 1 = -0.10
        #   d4: 99 / 99 - 1 = 0.0
        np.testing.assert_allclose(result, [0.0, 0.10, -0.10, 0.0])
        assert len(calls) == 1
        assert calls[0]["tickers"] == "SPY"
        assert calls[0]["period"] == "1y"
        assert calls[0]["proxy"] == "http://p"
        assert calls[0]["auto_adjust"] is True
        assert "start" not in calls[0]

    def test_datetimeindex_period_becomes_start(self, fake_yf):
        calls, state = fake_yf
        state["close"] = _daily([10.0, 11.0])
        period = pd.date_range("2024-01-01", periods=2, freq="D")
        result = utils.download_returns("SPY", period=period)
        # 11 / 10 - 1 = 0.10
        np.testing.assert_allclose(result, [0.0, 0.10])
        assert calls[0]["start"] == pd.Timestamp("2024-01-01")
        assert "period" not in calls[0]

    def test_tz_aware_index_made_naive(self, fake_yf):
        _, state = fake_yf
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
        state["close"] = pd.Series([50.0, 55.0], index=idx)
        result = utils.download_returns("SPY")
        # 55 / 50 - 1 = 0.10; local wall time kept, tz dropped
        assert result.index.tz is None
        assert list(result.index) == [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
        ]
        np.testing.assert_allclose(result, [0.0, 0.10])

    def test_single_and_constant_prices(self, fake_yf):
        _, state = fake_yf
        state["close"] = _daily([42.0])
        # One price -> one NaN return -> filled to 0
        np.testing.assert_allclose(utils.download_returns("X"), [0.0])
        state["close"] = _daily([42.0] * 4)
        # 42 / 42 - 1 = 0 every day
        np.testing.assert_allclose(utils.download_returns("X"), [0.0] * 4)

    def test_empty(self, fake_yf):
        _, state = fake_yf
        state["close"] = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils.download_returns("X")) == 0


class TestToLogReturns:
    """utils.to_log_returns / log_returns: ln(1 + r) after preparation."""

    def test_known_answer(self):
        result = utils.to_log_returns(_daily([0.10, -0.05, 0.0, 0.5]))
        # ln(1.10) = 0.0953101798, ln(0.95) = -0.0512932944,
        # ln(1.00) = 0, ln(1.5) = 0.4054651081
        np.testing.assert_allclose(
            result, [0.0953101798, -0.0512932944, 0.0, 0.4054651081], rtol=1e-9
        )

    def test_log_returns_alias_matches(self):
        r = _daily([0.10, -0.05, 0.0])
        pd.testing.assert_series_equal(utils.log_returns(r), utils.to_log_returns(r))

    def test_minus_100_percent_becomes_nan(self):
        result = utils.to_log_returns(_daily([0.10, -1.0]))
        # ln(1 + -1) = ln(0) = -inf -> replaced with NaN
        assert result.iloc[0] == pytest.approx(np.log(1.10))
        assert np.isnan(result.iloc[1])

    def test_rf_subtracted_before_log(self):
        result = utils.to_log_returns(_daily([0.11, 0.01]), rf=0.01)
        # excess: 0.11 - 0.01 = 0.10, 0.01 - 0.01 = 0.0
        # ln(1.10) = 0.0953101798, ln(1.0) = 0
        np.testing.assert_allclose(result, [0.0953101798, 0.0], rtol=1e-9, atol=1e-15)

    def test_rf_deannualized_with_nperiods(self):
        result = utils.to_log_returns(_daily([0.20, 0.10]), rf=0.21, nperiods=2)
        # per-period rf = 1.21 ** 0.5 - 1 = 0.10
        # excess = 0.10, 0.0 -> ln(1.10) = 0.0953101798, ln(1.0) = 0
        np.testing.assert_allclose(result, [0.0953101798, 0.0], rtol=1e-9, atol=1e-12)

    def test_dataframe(self):
        df = _daily_frame({"A": [0.10, 0.0], "B": [-0.05, 0.5]})
        result = utils.to_log_returns(df)
        # A: ln(1.10) = 0.0953101798, ln(1) = 0
        # B: ln(0.95) = -0.0512932944, ln(1.5) = 0.4054651081
        np.testing.assert_allclose(result["A"], [0.0953101798, 0.0], rtol=1e-9)
        np.testing.assert_allclose(
            result["B"], [-0.0512932944, 0.4054651081], rtol=1e-9
        )

    def test_single_constant_empty_all_nan(self):
        # ln(1.01) = 0.00995033085 for a single value and each constant day
        np.testing.assert_allclose(
            utils.to_log_returns(_daily([0.01])), [0.00995033085]
        )
        np.testing.assert_allclose(
            utils.to_log_returns(_daily([0.01] * 3)), [0.00995033085] * 3
        )
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils.to_log_returns(empty)) == 0
        assert utils.to_log_returns(_daily([np.nan] * 2)).isna().all()

    def test_conversion_error_warns_and_returns_zero(self, monkeypatch):
        # Force the try block to fail: "a" + 1 raises TypeError
        monkeypatch.setattr(
            utils, "_prepare_returns", lambda *a, **k: pd.Series(["a", "b"])
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = utils.to_log_returns(_daily([0.01, 0.02]))
        assert result == 0.0
        assert any("Error converting to log returns" in str(w.message) for w in caught)


class TestExponentialStdev:
    """utils.exponential_stdev: pandas ewm std with span=window."""

    def test_known_answer_span_2(self):
        result = utils.exponential_stdev(_daily([0.01, 0.03, 0.00]), window=2)
        # span 2 -> alpha = 2 / (2 + 1) = 2/3, decay 1 - alpha = 1/3,
        # min_periods = 2 so d1 is NaN.
        # d2: two points -> bias-corrected weighted var equals the sample var:
        #     (0.03 - 0.01)^2 / 2 = 0.0004 / 2 = 0.0002 -> std = 0.0141421356
        # d3: weights (oldest..newest) 1/9, 1/3, 1; sum w = 13/9
        #     mean = (0.01/9 + 0.03/3 + 0) / (13/9) = (0.10/9) / (13/9) = 0.10/13
        #          = 0.0076923077
        #     deviations: 0.03/13, 0.29/13, -0.10/13
        #     biased var = (1/9 * 0.0009 + 1/3 * 0.0841 + 1 * 0.01) / 169 / (13/9)
        #                = (0.0009 + 0.2523 + 0.09) / 9 / 169 * 9 / 13
        #                = 0.3432 / 2197 = 0.000156213
        #     correction = (sum w)^2 / ((sum w)^2 - sum w^2)
        #                = (169/81) / (169/81 - 91/81) = 169 / 78 = 2.1666667
        #     var = 0.000156213 * 2.1666667 = 0.3432 / 1014 = 0.000338462
        #     std = sqrt(0.000338462) = 0.0183973242
        assert np.isnan(result.iloc[0])
        np.testing.assert_allclose(
            result.iloc[1:], [0.0141421356, 0.0183973242], rtol=1e-8
        )

    def test_min_periods_equals_window(self):
        result = utils.exponential_stdev(_daily([0.01, 0.03, 0.00, 0.02]), window=3)
        # First two rows have fewer than 3 observations -> NaN
        assert result.iloc[:2].isna().all()
        assert result.iloc[2:].notna().all()

    def test_dataframe_per_column(self):
        df = _daily_frame({"A": [0.01, 0.03], "B": [0.02, 0.02]})
        result = utils.exponential_stdev(df, window=2)
        # A d2: sqrt((0.03 - 0.01)^2 / 2) = sqrt(0.0002) = 0.0141421356
        # B d2: constant -> 0
        np.testing.assert_allclose(result["A"].iloc[1], 0.0141421356, rtol=1e-8)
        assert result["B"].iloc[1] == pytest.approx(0.0)

    def test_constant_is_zero(self):
        result = utils.exponential_stdev(_daily([0.01] * 4), window=2)
        # No dispersion: std = 0 once min_periods is met
        np.testing.assert_allclose(result.iloc[1:], [0.0, 0.0, 0.0], atol=1e-12)

    def test_single_empty_all_nan(self):
        # Fewer observations than the window -> NaN; empty stays empty
        assert utils.exponential_stdev(_daily([0.01]), window=2).isna().all()
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils.exponential_stdev(empty, window=2)) == 0
        assert utils.exponential_stdev(_daily([np.nan] * 3), window=2).isna().all()

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason="is_halflife=True passes span and halflife together; pandas rejects it",
    )
    def test_halflife(self):
        result = utils.exponential_stdev(
            _daily([0.01, 0.03]), window=2, is_halflife=True
        )
        # With two points the bias-corrected weighted variance is the sample
        # variance whatever the weights: (0.03 - 0.01)^2 / 2 = 0.0002,
        # std = 0.0141421356. Actual: ValueError "comass, span, halflife, and
        # alpha are mutually exclusive".
        np.testing.assert_allclose(result.iloc[1], 0.0141421356, rtol=1e-8)


class TestMultiShift:
    """utils.multi_shift: the frame plus its lags 1..shift-1."""

    def test_series_known_answer(self):
        result = utils.multi_shift(_daily([0.01, 0.02, 0.03, 0.04], name="r"), shift=3)
        # Lags 1 and 2 are appended as r1, r2:
        #   r  = 0.01, 0.02, 0.03, 0.04
        #   r1 = NaN,  0.01, 0.02, 0.03
        #   r2 = NaN,  NaN,  0.01, 0.02
        assert list(result.columns) == ["r", "r1", "r2"]
        np.testing.assert_allclose(result["r"], [0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(result["r1"], [np.nan, 0.01, 0.02, 0.03])
        np.testing.assert_allclose(result["r2"], [np.nan, np.nan, 0.01, 0.02])

    def test_dataframe_input(self):
        df = _daily_frame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})
        result = utils.multi_shift(df, shift=2)
        # One lag per column: A1 = NaN, 1, 2; B1 = NaN, 10, 20
        assert list(result.columns) == ["A", "B", "A1", "B1"]
        np.testing.assert_allclose(result["A1"], [np.nan, 1.0, 2.0])
        np.testing.assert_allclose(result["B1"], [np.nan, 10.0, 20.0])
        pd.testing.assert_frame_equal(result[["A", "B"]], df)

    def test_shift_one_is_a_copy(self):
        df = _daily_frame({"A": [1.0, 2.0]})
        result = utils.multi_shift(df, shift=1)
        # range(1, 1) is empty: no lag columns
        pd.testing.assert_frame_equal(result, df)
        assert result is not df

    def test_single_constant_empty_all_nan(self):
        single = utils.multi_shift(_daily([0.05], name="r"), shift=2)
        # One row: r = 0.05, r1 = NaN
        assert single["r"].iloc[0] == 0.05 and np.isnan(single["r1"].iloc[0])

        const = utils.multi_shift(_daily([0.01] * 3, name="r"), shift=2)
        # r1 = NaN, 0.01, 0.01
        np.testing.assert_allclose(const["r1"], [np.nan, 0.01, 0.01])

        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float, name="r")
        result = utils.multi_shift(empty, shift=3)
        assert list(result.columns) == ["r", "r1", "r2"] and len(result) == 0

        nan = utils.multi_shift(_daily([np.nan] * 2, name="r"), shift=2)
        assert nan.isna().all().all()


class TestPrepareBenchmark:
    """utils._prepare_benchmark with Series/DataFrame/ticker (ticker faked)."""

    def test_none_returns_none(self):
        assert utils._prepare_benchmark(None) is None

    def test_series_passthrough_when_dates_match(self):
        bench = _daily([0.01, -0.02, 0.03])
        result = utils._prepare_benchmark(bench, period=bench.index)
        # Same dates as the period: no realignment, returns unchanged
        np.testing.assert_allclose(result, [0.01, -0.02, 0.03])

    def test_dataframe_uses_first_column(self):
        df = _daily_frame({"A": [0.01, 0.02], "B": [0.50, 0.50]})
        result = utils._prepare_benchmark(df)
        # Only column A is kept
        assert isinstance(result, pd.Series)
        np.testing.assert_allclose(result, [0.01, 0.02])

    def test_ticker_is_downloaded(self, monkeypatch):
        calls = []

        def fake_download(ticker, period="max", proxy=None):
            calls.append(ticker)
            return _daily([0.01, 0.02])

        monkeypatch.setattr(utils, "download_returns", fake_download)
        result = utils._prepare_benchmark("SPY")
        assert calls == ["SPY"]
        np.testing.assert_allclose(result, [0.01, 0.02])

    def test_prices_converted_to_returns(self):
        result = utils._prepare_benchmark(_daily([100.0, 110.0, 99.0]))
        # Values > 1 are read as prices: NaN, 110/100 - 1 = 0.10, 99/110 - 1 = -0.10
        assert np.isnan(result.iloc[0])
        np.testing.assert_allclose(result.iloc[1:], [0.10, -0.10])

    def test_prepare_returns_false_keeps_raw_values(self):
        result = utils._prepare_benchmark(
            _daily([100.0, np.nan, 99.0]), prepare_returns=False
        )
        # Raw values, only NaN rows dropped: 100, 99
        np.testing.assert_allclose(result, [100.0, 99.0])

    def test_rf_is_not_applied(self):
        bench = _daily([0.05, 0.03])
        result = utils._prepare_benchmark(bench, rf=0.01)
        # Benchmark is never converted to excess returns here: 0.05, 0.03
        np.testing.assert_allclose(result, [0.05, 0.03])

    def test_tz_aware_made_naive(self):
        idx = pd.DatetimeIndex(
            ["2024-01-01 20:00", "2024-01-02 20:00"], tz="US/Eastern"
        )
        result = utils._prepare_benchmark(pd.Series([0.01, 0.02], index=idx))
        # 20:00 EST = 01:00 UTC the next day
        assert result.index.tz is None
        assert list(result.index) == [
            pd.Timestamp("2024-01-02 01:00"),
            pd.Timestamp("2024-01-03 01:00"),
        ]

    def test_realigned_to_period_subset(self):
        bench = _daily([0.10, 0.10, 0.10])
        period = pd.date_range("2024-01-02", periods=2, freq="D")
        result = utils._prepare_benchmark(bench, period=period)
        # prices from returns (base 1): d1 1.1, d2 1.21, d3 1.331
        # restricted to d2..d3: 1.21, 1.331 -> pct_change: NaN -> 0, 1.331/1.21 - 1 = 0.10
        # d3 is asserted here; d2 is the xfail below.
        assert list(result.index) == list(period)
        assert result.iloc[1] == pytest.approx(0.10)

    def test_single_empty_all_nan(self):
        np.testing.assert_allclose(utils._prepare_benchmark(_daily([0.02])), [0.02])
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils._prepare_benchmark(empty)) == 0
        # NaN rows are dropped before preparation
        assert len(utils._prepare_benchmark(_daily([np.nan] * 3))) == 0
        np.testing.assert_allclose(
            utils._prepare_benchmark(_daily([0.01] * 3)), [0.01] * 3
        )

    @pytest.mark.xfail(
        strict=True,
        reason="realignment zeroes the first period's benchmark return",
    )
    def test_realigned_first_period_keeps_its_return(self):
        bench = _daily([0.10, 0.10, 0.10])
        period = pd.date_range("2024-01-02", periods=2, freq="D")
        result = utils._prepare_benchmark(bench, period=period)
        # d2's return is known from the data (1.21 / 1.1 - 1 = 0.10), but the
        # price index is cut at period[0] before pct_change, so it becomes 0.
        np.testing.assert_allclose(result, [0.10, 0.10])


def _freeze_now(monkeypatch, now):
    """Point utils' clock at a fixed datetime."""

    class _Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(now.year, now.month, now.day, now.hour, now.minute)

    monkeypatch.setattr(
        utils, "_dt", types.SimpleNamespace(datetime=_Frozen, date=dt.date)
    )


class TestQtd:
    """utils._qtd quarter-to-date filter with a frozen clock."""

    @pytest.fixture
    def returns(self):
        idx = pd.DatetimeIndex(
            [
                "2023-12-31",
                "2024-01-01",
                "2024-03-31",
                "2024-04-01",
                "2024-05-15",
                "2024-10-01",
                "2024-11-15",
            ]
        )
        return pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], index=idx)

    @pytest.mark.parametrize(
        "today, first_kept",
        [
            # First month of a quarter: quarter start = that month's 1st
            (dt.datetime(2024, 1, 20), "2024-01-01"),
            (dt.datetime(2024, 4, 10), "2024-04-01"),
            (dt.datetime(2024, 10, 2), "2024-10-01"),
        ],
    )
    def test_first_month_of_quarter(self, returns, monkeypatch, today, first_kept):
        _freeze_now(monkeypatch, today)
        result = utils._qtd(returns)
        # Everything on or after the quarter's first day
        expected = returns[returns.index >= first_kept]
        pd.testing.assert_series_equal(result, expected)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "today, first_kept",
        [
            # 15 May is in Q2 -> keep from 1 Apr: 2024-04-01, 2024-05-15, ...
            pytest.param(
                dt.datetime(2024, 5, 15),
                "2024-04-01",
                marks=pytest.mark.xfail(
                    strict=True, reason="_qtd picks the NEXT quarter start (1 Jul)"
                ),
            ),
            # 15 Nov is in Q4 -> keep from 1 Oct: 2024-10-01, 2024-11-15
            pytest.param(
                dt.datetime(2024, 11, 15),
                "2024-10-01",
                marks=pytest.mark.xfail(
                    strict=True, reason="_qtd falls back to month start (1 Nov)"
                ),
            ),
        ],
    )
    def test_later_month_of_quarter(self, returns, monkeypatch, today, first_kept):
        _freeze_now(monkeypatch, today)
        result = utils._qtd(returns)
        expected = returns[returns.index >= first_kept]
        pd.testing.assert_series_equal(result, expected)

    def test_empty_and_dataframe(self, monkeypatch):
        _freeze_now(monkeypatch, dt.datetime(2024, 4, 10))
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils._qtd(empty)) == 0
        df = _daily_frame({"A": [0.01, 0.02, 0.03]}, start="2024-03-31")
        # Keeps 2024-04-01 and 2024-04-02 (A = 0.02, 0.03)
        np.testing.assert_allclose(utils._qtd(df)["A"], [0.02, 0.03])


class TestMtdYtd:
    """utils._mtd / utils._ytd with a frozen clock of 2024-05-15."""

    @pytest.fixture
    def returns(self):
        idx = pd.DatetimeIndex(
            ["2023-12-31", "2024-01-01", "2024-04-30", "2024-05-01", "2024-05-15"]
        )
        return pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)

    def test_mtd(self, returns, monkeypatch):
        _freeze_now(monkeypatch, dt.datetime(2024, 5, 15))
        # Month start 2024-05-01 -> keeps 2024-05-01, 2024-05-15 (0.04, 0.05)
        np.testing.assert_allclose(utils._mtd(returns), [0.04, 0.05])

    def test_ytd(self, returns, monkeypatch):
        _freeze_now(monkeypatch, dt.datetime(2024, 5, 15))
        # Year start 2024-01-01 -> drops only 2023-12-31: 0.02, 0.03, 0.04, 0.05
        np.testing.assert_allclose(utils._ytd(returns), [0.02, 0.03, 0.04, 0.05])

    def test_empty_inputs(self, monkeypatch):
        _freeze_now(monkeypatch, dt.datetime(2024, 5, 15))
        empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        assert len(utils._mtd(empty)) == 0
        assert len(utils._ytd(empty)) == 0


class TestPandasDateFilters:
    """utils._pandas_date / utils._pandas_current_month."""

    @pytest.fixture
    def returns(self):
        return _daily([0.01, 0.02, 0.03, 0.04], start="2024-05-30")

    def test_pandas_date_single_and_list(self, returns):
        # Dates: 05-30 0.01, 05-31 0.02, 06-01 0.03, 06-02 0.04
        ts = pd.Timestamp
        np.testing.assert_allclose(
            utils._pandas_date(returns, ts("2024-05-31")), [0.02]
        )
        np.testing.assert_allclose(
            utils._pandas_date(returns, [ts("2024-05-30"), ts("2024-06-02")]),
            [0.01, 0.04],
        )
        assert len(utils._pandas_date(returns, ts("2030-01-01"))) == 0

    @pytest.mark.xfail(
        strict=True,
        reason="string dates silently match nothing (DatetimeIndex.isin on pandas 3)",
    )
    def test_pandas_date_accepts_date_strings(self, returns):
        # "2024-05-31" names the second row (0.02); actual result is empty
        np.testing.assert_allclose(utils._pandas_date(returns, "2024-05-31"), [0.02])

    def test_current_month(self, returns, monkeypatch):
        _freeze_now(monkeypatch, dt.datetime(2024, 6, 2, 12, 0))
        # date_range(2024-06-01, 2024-06-02 12:00) = 06-01 00:00, 06-02 00:00
        # -> keeps 06-01 0.03 and 06-02 0.04
        np.testing.assert_allclose(utils._pandas_current_month(returns), [0.03, 0.04])


class TestRoundToClosest:
    """utils._round_to_closest: round val to the nearest multiple of res."""

    @pytest.mark.parametrize(
        "val, res, decimals, expected",
        [
            # 12.3 / 5 = 2.46 -> round 2 -> 2 * 5 = 10
            (12.3, 5, None, 10),
            # 13.0 / 5 = 2.6 -> round 3 -> 3 * 5 = 15
            (13.0, 5, None, 15),
            # -12.3 / 5 = -2.46 -> round -2 -> -10
            (-12.3, 5, None, -10),
            # 0 / 5 = 0 -> 0
            (0.0, 5, None, 0),
            # res 0.05 -> 2 decimals auto-detected;
            # 0.123 / 0.05 = 2.46 -> 2 -> 2 * 0.05 = 0.1 -> round(0.1, 2) = 0.1
            (0.123, 0.05, None, 0.1),
            # 0.126 / 0.01 = 12.6 -> 13 -> 0.13 -> round(0.13, 1) = 0.1
            (0.126, 0.01, 1, 0.1),
        ],
    )
    def test_known_answers(self, val, res, decimals, expected):
        assert utils._round_to_closest(val, res, decimals) == pytest.approx(expected)

    def test_nan_raises(self):
        # round(NaN) cannot convert to an integer
        with pytest.raises(ValueError):
            utils._round_to_closest(float("nan"), 5)


class TestScoreStr:
    """utils._score_str: prefix '+' unless the string already has a '-'."""

    @pytest.mark.parametrize(
        "val, expected",
        [("1.5", "+1.5"), ("0", "+0"), ("-2.25", "-2.25"), ("12%", "+12%")],
    )
    def test_strings(self, val, expected):
        assert utils._score_str(val) == expected

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="docstring says 'str or numeric' but numbers raise TypeError",
    )
    def test_numeric(self):
        # 1.5 -> "+1.5"; actual: TypeError ("argument of type 'float' is not iterable")
        assert utils._score_str(1.5) == "+1.5"


class TestFlattenDataframe:
    """utils._flatten_dataframe: MultiIndex rows -> plain columns via CSV."""

    @pytest.fixture
    def nested(self):
        idx = pd.MultiIndex.from_tuples(
            [(2024, 1), (2024, 2), (2025, 1)], names=["year", "month"]
        )
        return pd.DataFrame({"r": [0.01, 0.02, 0.03]}, index=idx)

    def test_multiindex_rows_become_columns(self, nested):
        result = utils._flatten_dataframe(nested)
        # Index levels written out as ordinary columns, RangeIndex 0..2
        assert list(result.columns) == ["year", "month", "r"]
        assert list(result["year"]) == [2024, 2024, 2025]
        assert list(result["month"]) == [1, 2, 1]
        np.testing.assert_allclose(result["r"], [0.01, 0.02, 0.03])
        assert list(result.index) == [0, 1, 2]

    def test_set_index(self, nested):
        result = utils._flatten_dataframe(nested, set_index="year")
        assert result.index.name == "year"
        assert list(result.index) == [2024, 2024, 2025]
        assert list(result.columns) == ["month", "r"]

    def test_empty(self):
        empty = pd.DataFrame({"r": []}, index=pd.Index([], name="k"))
        result = utils._flatten_dataframe(empty)
        assert list(result.columns) == ["k", "r"] and len(result) == 0
