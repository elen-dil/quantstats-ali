"""
Known-answer and edge-case tests for quantstats.stats.

Every expected value below is derived by hand from a tiny input; the
arithmetic is shown in the comment next to the assertion.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quantstats import stats
from quantstats.utils import DataValidationError

# Expected-shortfall constant for a normal distribution at alpha = 0.05:
# pdf(ppf(0.05)) / 0.05 = 0.1031356404 / 0.05 = 2.0627128075
ES_K_95 = 2.0627128075074253


def _series(values, start="2021-01-01", name=None):
    """Build a daily float Series from a list of values."""
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float, name=name)


def _frame(start="2021-01-01", **columns):
    """Build a daily float DataFrame from equal-length lists."""
    n = len(next(iter(columns.values())))
    index = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(columns, index=index, dtype=float)


EMPTY = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

# r = [0.01, -0.02, 0.03, -0.01, 0.02]; mean = 0.03 / 5 = 0.006
# deviations = [0.004, -0.026, 0.024, -0.016, 0.014]
# sum of squares = 0.000016 + 0.000676 + 0.000576 + 0.000256 + 0.000196 = 0.00172
# sample std = sqrt(0.00172 / 4) = sqrt(0.00043) = 0.0207364414
FIVE = [0.01, -0.02, 0.03, -0.01, 0.02]


@pytest.fixture(autouse=True)
def _silence_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestCompare:
    """compare() with hand-built Series benchmarks (never a ticker)."""

    @pytest.fixture
    def returns(self):
        # 2021-01-27 .. 2021-02-01: five January days, one February day
        return _series([0.01, 0.02, -0.01, 0.0, 0.03, 0.02], start="2021-01-27")

    @pytest.fixture
    def benchmark(self):
        return _series([0.005, 0.02, 0.01, 0.0, 0.01, -0.01], start="2021-01-27")

    def test_daily_known_answer(self, returns, benchmark):
        result = stats.compare(returns, benchmark)
        # Columns are in percent: 0.01 -> 1.0, 0.005 -> 0.5, ...
        assert list(result.columns) == ["Benchmark", "Returns", "Multiplier", "Won"]
        np.testing.assert_allclose(result["Returns"], [1.0, 2.0, -1.0, 0.0, 3.0, 2.0])
        np.testing.assert_allclose(result["Benchmark"], [0.5, 2.0, 1.0, 0.0, 1.0, -1.0])
        # Multiplier = Returns / Benchmark: 1/0.5=2, 2/2=1, -1/1=-1,
        # 0/0 -> NaN (zero benchmark guarded), 3/1=3, 2/-1=-2
        np.testing.assert_allclose(
            result["Multiplier"], [2.0, 1.0, -1.0, np.nan, 3.0, -2.0]
        )
        # Won is "+" when Returns >= Benchmark (ties count as a win: 2 >= 2, 0 >= 0)
        assert result["Won"].tolist() == ["+", "+", "-", "+", "+", "+"]

    def test_monthly_compounded(self, returns, benchmark):
        result = stats.compare(returns, benchmark, aggregate="month")
        # Jan returns: 1.01 * 1.02 * 0.99 * 1.00 * 1.03 - 1
        #   = 1.0302 * 0.99 = 1.019898 * 1.03 = 1.05049494 - 1 = 5.049494 %
        # Jan bench:  1.005 * 1.02 * 1.01 * 1.00 * 1.01 - 1
        #   = 1.0251 * 1.01 = 1.035351 * 1.01 = 1.04570451 - 1 = 4.570451 %
        # Feb: returns 2 %, bench -1 %
        np.testing.assert_allclose(result["Returns"], [5.049494, 2.0], rtol=1e-9)
        np.testing.assert_allclose(result["Benchmark"], [4.570451, -1.0], rtol=1e-9)
        # Jan multiplier = 5.049494 / 4.570451 = 1.1048129; Feb = 2 / -1 = -2
        np.testing.assert_allclose(
            result["Multiplier"], [5.049494 / 4.570451, -2.0], rtol=1e-9
        )
        assert result["Won"].tolist() == ["+", "+"]

    def test_monthly_simple_sum(self, returns, benchmark):
        result = stats.compare(returns, benchmark, aggregate="month", compounded=False)
        # Jan returns: 1 + 2 - 1 + 0 + 3 = 5 %; Jan bench: 0.5 + 2 + 1 + 0 + 1 = 4.5 %
        np.testing.assert_allclose(result["Returns"], [5.0, 2.0])
        np.testing.assert_allclose(result["Benchmark"], [4.5, -1.0])
        # 5 / 4.5 = 1.1111
        np.testing.assert_allclose(result["Multiplier"], [5.0 / 4.5, -2.0])

    def test_round_vals(self, returns, benchmark):
        result = stats.compare(returns, benchmark, aggregate="ME", round_vals=2)
        # 5.049494 -> 5.05, 4.570451 -> 4.57, multiplier 1.1048 -> 1.1
        assert result["Returns"].tolist() == [5.05, 2.0]
        assert result["Benchmark"].tolist() == [4.57, -1.0]
        assert result["Multiplier"].tolist() == [1.1, -2.0]

    def test_tz_aware_inputs_match_naive(self, returns, benchmark):
        naive = stats.compare(returns, benchmark)
        aware_r = returns.tz_localize("US/Eastern")
        aware_b = benchmark.tz_localize("US/Eastern")
        result = stats.compare(aware_r, aware_b)
        # Same values in the same order; only the index is shifted to UTC-naive
        assert result.index.tz is None
        np.testing.assert_allclose(result["Returns"], naive["Returns"])
        np.testing.assert_allclose(result["Benchmark"], naive["Benchmark"])

    def test_dataframe_benchmark_uses_first_column(self, returns, benchmark):
        bench_df = pd.DataFrame({"bm": benchmark, "other": returns})
        result = stats.compare(returns, bench_df)
        # First column is the benchmark: [0.5, 2, 1, 0, 1, -1] %
        np.testing.assert_allclose(result["Benchmark"], [0.5, 2.0, 1.0, 0.0, 1.0, -1.0])

    def test_dataframe_returns(self, returns, benchmark):
        strategies = pd.DataFrame({"s1": returns, "s2": benchmark})
        result = stats.compare(strategies, benchmark, aggregate="month")
        assert list(result.columns) == ["Benchmark", "Returns_0", "Returns_1"]
        # s1 Jan = 5.049494 %, s2 (== benchmark) Jan = 4.570451 %
        np.testing.assert_allclose(result["Returns_0"], [5.049494, 2.0], rtol=1e-9)
        np.testing.assert_allclose(result["Returns_1"], [4.570451, -1.0], rtol=1e-9)
        np.testing.assert_allclose(result["Benchmark"], [4.570451, -1.0], rtol=1e-9)

    def test_constant_returns_equal_benchmark(self):
        flat = _series([0.01] * 6, start="2021-01-27")
        result = stats.compare(flat, flat.copy(), aggregate="month")
        # Jan: 1.01^5 - 1 = 5.1010050 %; Feb: 1 %; identical -> multiplier 1, won "+"
        np.testing.assert_allclose(result["Returns"], [5.10100501, 1.0], rtol=1e-8)
        np.testing.assert_allclose(result["Multiplier"], [1.0, 1.0])
        assert result["Won"].tolist() == ["+", "+"]

    def test_single_value(self):
        result = stats.compare(_series([0.01]), _series([0.02]))
        # 1 % vs 2 %: multiplier 0.5, lost
        assert result["Multiplier"].tolist() == [0.5]
        assert result["Won"].tolist() == ["-"]

    def test_empty(self):
        result = stats.compare(EMPTY, EMPTY.copy())
        assert len(result) == 0
        assert list(result.columns) == ["Benchmark", "Returns", "Multiplier", "Won"]

    def test_all_nan_returns_have_nan_values(self, benchmark):
        result = stats.compare(_series([np.nan] * 6, start="2021-01-27"), benchmark)
        assert result["Returns"].isna().all()
        assert result["Multiplier"].isna().all()

    @pytest.mark.xfail(
        reason="days with no strategy return are scored as losses ('-')", strict=True
    )
    def test_missing_strategy_day_not_scored_as_loss(self, returns):
        # Benchmark has one extra day (2021-02-02) that the strategy never traded
        bench = _series([0.005, 0.02, 0.01, 0.0, 0.01, -0.01, 0.04], start="2021-01-27")
        result = stats.compare(returns, bench)
        missing = result["Returns"].isna()
        assert missing.sum() == 1
        assert (result.loc[missing, "Won"] != "-").all()


class TestSerenityIndex:
    def test_known_answer(self):
        r = _series([0.10, -0.10, 0.05, 0.0, 0.0])
        # equity = [1.1, 0.99, 1.0395, 1.0395, 1.0395], peak = 1.1 throughout
        # dd = [0, -0.1, -0.055, -0.055, -0.055]
        # ulcer = sqrt((0.01 + 3 * 0.003025) / 4) = sqrt(0.019075 / 4) = 0.0690561
        # dd mean = -0.265 / 5 = -0.053; dd sample std = 0.0354612
        # cvar(dd) = -0.053 - 0.0354612 * 2.0627128 = -0.1261464
        # std(r) = sqrt((0.0081 + 0.0121 + 0.0016 + 0.0001 + 0.0001) / 4) = 0.0741620
        # pitfall = 0.1261464 / 0.0741620 = 1.7009573
        # serenity = sum(r) / (ulcer * pitfall) = 0.05 / 0.1174615 = 0.4256713
        assert stats.serenity_index(r) == pytest.approx(0.4256712544, rel=1e-9)

    def test_rf_is_subtracted_from_total_return(self):
        r = _series([0.10, -0.10, 0.05, 0.0, 0.0])
        # (0.05 - 0.01) / 0.1174615 = 0.3405370
        assert stats.serenity_index(r, rf=0.01) == pytest.approx(0.3405370035, rel=1e-9)

    def test_constant_returns_nan(self):
        # std = 0 -> undefined
        assert np.isnan(stats.serenity_index(_series([0.01] * 5)))

    def test_no_drawdown_nan(self):
        # Only gains: dd = 0 everywhere, ulcer = 0, cvar(dd) = 0 -> 0 denominator
        assert np.isnan(stats.serenity_index(_series([0.01, 0.02, 0.03, 0.01, 0.02])))

    def test_single_value_nan(self):
        # std of one observation is NaN
        assert np.isnan(stats.serenity_index(_series([-0.05])))

    def test_empty_raises(self):
        with pytest.raises(DataValidationError):
            stats.serenity_index(EMPTY)

    def test_all_nan_raises(self):
        with pytest.raises(DataValidationError):
            stats.serenity_index(_series([np.nan] * 5))

    def test_dataframe(self):
        df = _frame(a=[0.10, -0.10, 0.05, 0.0, 0.0], b=[0.01] * 5)
        result = stats.serenity_index(df)
        # column a matches the Series known answer; b is constant -> NaN
        assert result["a"] == pytest.approx(0.4256712544, rel=1e-9)
        assert np.isnan(result["b"])


class TestOmega:
    R = [0.02, -0.01, 0.03, -0.02, 0.01]

    def test_known_answer_zero_threshold(self):
        # gains above 0: 0.02 + 0.03 + 0.01 = 0.06; losses: 0.01 + 0.02 = 0.03
        # omega = 0.06 / 0.03 = 2.0
        assert stats.omega(_series(self.R), periods=1) == pytest.approx(2.0)

    def test_annual_zero_threshold_is_unchanged(self):
        # (1 + 0)^(1/252) - 1 = 0, so the default periods gives the same 2.0
        assert stats.omega(_series(self.R)) == pytest.approx(2.0)

    def test_known_answer_with_required_return(self):
        # periods=1 -> threshold 0.01; deviations [0.01, -0.02, 0.02, -0.03, 0]
        # gains 0.01 + 0.02 = 0.03; losses 0.02 + 0.03 = 0.05; omega = 0.6
        result = stats.omega(_series(self.R), required_return=0.01, periods=1)
        assert result == pytest.approx(0.6)

    def test_required_return_converted_per_period(self):
        # annual 10.25 % over periods=2 -> per period sqrt(1.1025) - 1 = 0.05
        # r = [0.10, 0.0, 0.07, 0.02]; deviations [0.05, -0.05, 0.02, -0.03]
        # gains 0.07; losses 0.08; omega = 0.875
        r = _series([0.10, 0.0, 0.07, 0.02])
        assert stats.omega(r, required_return=0.1025, periods=2) == pytest.approx(0.875)

    def test_invalid_required_return_nan(self):
        assert np.isnan(stats.omega(_series(self.R), required_return=-1))

    def test_constant_positive_returns_nan(self):
        # No losses -> zero denominator -> NaN
        assert np.isnan(stats.omega(_series([0.01] * 5)))

    def test_single_value_nan(self):
        assert np.isnan(stats.omega(_series([0.01])))

    def test_empty_raises(self):
        with pytest.raises(DataValidationError):
            stats.omega(EMPTY)

    def test_all_nan_raises(self):
        with pytest.raises(DataValidationError):
            stats.omega(_series([np.nan] * 5))

    def test_dataframe(self):
        df = _frame(a=self.R, b=[-0.01, -0.02, 0.01, -0.03, 0.02], c=[0.01] * 5)
        result = stats.omega(df, periods=1)
        # a: 0.06 / 0.03 = 2.0
        # b: gains 0.01 + 0.02 = 0.03; losses 0.01 + 0.02 + 0.03 = 0.06 -> 0.5
        # c: no losses -> NaN
        assert result["a"] == pytest.approx(2.0)
        assert result["b"] == pytest.approx(0.5)
        assert np.isnan(result["c"])


class TestTailRatio:
    def test_known_answer_default_cutoff(self):
        r = _series([0.01, -0.04, 0.02, 0.0, -0.01])
        # sorted = [-0.04, -0.01, 0, 0.01, 0.02], n = 5 (linear interpolation)
        # q95: pos 0.95 * 4 = 3.8 -> 0.01 + 0.8 * 0.01 = 0.018
        # q05: pos 0.05 * 4 = 0.2 -> -0.04 + 0.2 * 0.03 = -0.034
        # |0.018 / -0.034| = 0.5294118
        assert stats.tail_ratio(r) == pytest.approx(0.018 / 0.034)

    def test_known_answer_custom_cutoff(self):
        r = _series([0.04, -0.02, 0.0, 0.06, -0.01, 0.02, 0.01])
        # sorted = [-0.02, -0.01, 0, 0.01, 0.02, 0.04, 0.06], n = 7
        # q75: pos 4.5 -> (0.02 + 0.04) / 2 = 0.03
        # q25: pos 1.5 -> (-0.01 + 0) / 2 = -0.005
        # |0.03 / -0.005| = 6.0
        assert stats.tail_ratio(r, cutoff=0.75) == pytest.approx(6.0)

    def test_constant_nonzero_returns(self):
        # both quantiles equal 0.01 -> |0.01 / 0.01| = 1
        assert stats.tail_ratio(_series([0.01] * 5)) == pytest.approx(1.0)

    def test_zero_lower_tail_nan(self):
        assert np.isnan(stats.tail_ratio(_series([0.0] * 5)))

    def test_single_value(self):
        # one observation is both quantiles: |0.02 / 0.02| = 1
        assert stats.tail_ratio(_series([0.02])) == pytest.approx(1.0)

    def test_empty_nan(self):
        assert np.isnan(stats.tail_ratio(EMPTY))

    def test_all_nan(self):
        assert np.isnan(stats.tail_ratio(_series([np.nan] * 5)))

    def test_dataframe(self):
        df = _frame(a=[0.01, -0.04, 0.02, 0.0, -0.01], z=[0.0] * 5)
        result = stats.tail_ratio(df)
        assert result["a"] == pytest.approx(0.018 / 0.034)
        assert np.isnan(result["z"])


class TestPctRank:
    def test_known_answer(self):
        prices = _series([10, 11, 12, 11, 10, 10, 10], name="Close")
        result = stats.pct_rank(prices, window=3)
        # Rank of today's price within [today, t-1, t-2] (average rank for ties):
        # t2 [12, 11, 10] -> 3/3 = 100
        # t3 [11, 12, 11] -> ties at 11 share (1+2)/2 = 1.5 -> 1.5/3 = 50
        # t4 [10, 11, 12] -> 1/3 = 33.33
        # t5 [10, 10, 11] -> 1.5/3 = 50
        # t6 [10, 10, 10] -> (1+2+3)/3 = 2 -> 2/3 = 66.67
        np.testing.assert_allclose(
            result.iloc[2:].to_numpy(), [100.0, 50.0, 100 / 3, 50.0, 200 / 3]
        )

    def test_unnamed_series_matches_named(self):
        values = [10, 11, 12, 11, 10, 10, 10]
        named = stats.pct_rank(_series(values, name="Close"), window=3)
        unnamed = stats.pct_rank(_series(values), window=3)
        np.testing.assert_allclose(unnamed.to_numpy(), named.to_numpy())

    def test_constant_prices(self):
        # every full window is three ties -> average rank 2 of 3 = 66.67
        result = stats.pct_rank(_series([5.0] * 5), window=3)
        np.testing.assert_allclose(result.iloc[2:].to_numpy(), [200 / 3] * 3)

    def test_single_value(self):
        # one observation ranks 1 of 1 -> 100
        assert stats.pct_rank(_series([5.0]), window=3).tolist() == [100.0]

    def test_empty(self):
        assert len(stats.pct_rank(pd.Series([], dtype=float), window=3)) == 0

    def test_all_nan(self):
        assert stats.pct_rank(_series([np.nan] * 4), window=3).isna().all()


class TestImpliedVolatility:
    def test_known_answer_not_annualized(self):
        # ln(1 + r) = [0.0099503, -0.0202027, 0.0295588, -0.0100503, 0.0198026]
        # mean = 0.0058117; squared deviations sum = 0.00170515
        # sample std = sqrt(0.00170515 / 4) = 0.0206467
        result = stats.implied_volatility(_series(FIVE), annualize=False)
        assert result == pytest.approx(0.0206467476, rel=1e-8)

    def test_known_answer_rolling(self):
        result = stats.implied_volatility(_series(FIVE), periods=3)
        # window of 3 log returns, std * sqrt(3):
        # [ln1.01, ln0.98, ln1.03] std 0.0250663 * 1.7320508 = 0.0434160
        # [ln0.98, ln1.03, ln0.99] std 0.0262937 * 1.7320508 = 0.0455421
        # [ln1.03, ln0.99, ln1.02] std 0.0206368 * 1.7320508 = 0.0357440
        assert result.iloc[:2].isna().all()
        np.testing.assert_allclose(
            result.iloc[2:].to_numpy(), [0.0434160437, 0.0455420811, 0.0357439938]
        )

    def test_constant_returns_zero(self):
        assert stats.implied_volatility(
            _series([0.01] * 5), annualize=False
        ) == pytest.approx(0.0)

    def test_single_value_nan(self):
        assert np.isnan(stats.implied_volatility(_series([0.01]), annualize=False))

    def test_empty_nan(self):
        assert np.isnan(stats.implied_volatility(EMPTY, annualize=False))

    def test_all_nan(self):
        assert np.isnan(
            stats.implied_volatility(_series([np.nan] * 5), annualize=False)
        )

    def test_dataframe(self):
        result = stats.implied_volatility(_frame(a=FIVE, b=[0.01] * 5), annualize=False)
        assert result["a"] == pytest.approx(0.0206467476, rel=1e-8)
        assert result["b"] == pytest.approx(0.0)


class TestR2:
    def test_known_answer(self):
        bench = _series([0.01, 0.02, 0.03, 0.04, 0.05])
        rets = _series([0.02, 0.01, 0.04, 0.03, 0.05])
        # dx = [-2, -1, 0, 1, 2]e-2, dy = [-1, -2, 1, 0, 2]e-2
        # sum dx*dy = 2 + 2 + 0 + 0 + 4 = 8; sum dx^2 = 10; sum dy^2 = 10
        # r = 8 / sqrt(10 * 10) = 0.8; r^2 = 0.64
        assert stats.r2(rets, bench) == pytest.approx(0.64)

    def test_exact_linear_relation(self):
        bench = _series([0.01, -0.02, 0.03, 0.0, 0.015])
        # r = 2 * b + 0.001 is perfectly linear -> r^2 = 1
        assert stats.r2(2 * bench + 0.001, bench) == pytest.approx(1.0)

    def test_gap_is_dropped_pairwise(self):
        rets = _series([0.02, 0.01, np.nan, 0.04, 0.03, 0.05])
        bench = _series([0.01, 0.02, 0.09, 0.03, 0.04, 0.05])
        # The NaN day is dropped from both sides, leaving the 0.64 example
        assert stats.r2(rets, bench) == pytest.approx(0.64)

    def test_constant_benchmark_nan(self):
        rets = _series([0.02, 0.01, 0.04, 0.03, 0.05])
        assert np.isnan(stats.r2(rets, _series([0.01] * 5)))

    def test_single_value_nan(self):
        assert np.isnan(stats.r2(_series([0.01]), _series([0.02])))

    def test_empty_nan(self):
        assert np.isnan(stats.r2(EMPTY, EMPTY.copy()))

    def test_all_nan_nan(self):
        bench = _series([0.01, 0.02, 0.03, 0.04, 0.05])
        assert np.isnan(stats.r2(_series([np.nan] * 5), bench))


class TestUpi:
    def test_known_answer(self):
        r = _series([0.10, -0.10, 0.0, 0.0, 0.0])
        # comp = 1.1 * 0.9 - 1 = -0.01
        # equity = [1.1, 0.99, 0.99, 0.99, 0.99]; dd = [0, -0.1, -0.1, -0.1, -0.1]
        # ulcer = sqrt(4 * 0.01 / (5 - 1)) = 0.1
        # upi = -0.01 / 0.1 = -0.1
        assert stats.upi(r) == pytest.approx(-0.1)

    def test_known_answer_with_rf(self):
        r = _series([0.10, -0.10, 0.0, 0.0, 0.0])
        # (-0.01 - 0.01) / 0.1 = -0.2
        assert stats.upi(r, rf=0.01) == pytest.approx(-0.2)

    def test_matches_ulcer_performance_index(self):
        r = _series(FIVE)
        assert stats.upi(r) == pytest.approx(stats.ulcer_performance_index(r))

    def test_no_drawdown_nan(self):
        # ulcer index 0 -> NaN instead of a division by zero
        assert np.isnan(stats.upi(_series([0.01] * 5)))

    @pytest.mark.xfail(
        reason="ulcer_index divides by n - 1 = 0, so upi silently returns -0.0",
        strict=True,
    )
    def test_single_value_nan(self):
        assert np.isnan(stats.upi(_series([-0.05])))

    def test_empty_raises(self):
        with pytest.raises(DataValidationError):
            stats.upi(EMPTY)

    def test_all_nan_raises(self):
        with pytest.raises(DataValidationError):
            stats.upi(_series([np.nan] * 5))

    def test_dataframe(self):
        result = stats.upi(_frame(a=[0.10, -0.10, 0.0, 0.0, 0.0], b=[0.01] * 5))
        assert result["a"] == pytest.approx(-0.1)
        assert np.isnan(result["b"])


class TestRor:
    def test_known_answer(self):
        # win rate = 3 wins / 5 non-zero = 0.6
        # ((1 - 0.6) / (1 + 0.6)) ^ 5 = 0.25 ^ 5 = 0.0009765625
        assert stats.ror(_series(FIVE)) == pytest.approx(0.0009765625)

    def test_matches_risk_of_ruin(self):
        r = _series(FIVE)
        assert stats.ror(r) == pytest.approx(stats.risk_of_ruin(r))

    def test_all_wins(self):
        # win rate 1 -> (0 / 2) ^ 5 = 0
        assert stats.ror(_series([0.01] * 5)) == pytest.approx(0.0)

    def test_all_losses(self):
        # win rate 0 -> (1 / 1) ^ 5 = 1
        assert stats.ror(_series([-0.01] * 5)) == pytest.approx(1.0)

    def test_single_win(self):
        # (0 / 2) ^ 1 = 0
        assert stats.ror(_series([0.01])) == pytest.approx(0.0)

    def test_dataframe(self):
        result = stats.ror(_frame(a=FIVE, b=[-0.01] * 5))
        assert result["a"] == pytest.approx(0.0009765625)
        assert result["b"] == pytest.approx(1.0)

    @pytest.mark.xfail(
        reason="win_rate falls back to 0.0 for no data, so ruin is reported as certain",
        strict=True,
    )
    def test_all_nan_is_nan(self):
        assert np.isnan(stats.ror(_series([np.nan] * 5)))

    @pytest.mark.xfail(
        reason="empty input reports risk of ruin 1.0 instead of NaN", strict=True
    )
    def test_empty_is_nan(self):
        assert np.isnan(stats.ror(EMPTY))

    @pytest.mark.xfail(
        reason="a flat strategy that never loses is reported as certain ruin",
        strict=True,
    )
    def test_flat_returns_not_certain_ruin(self):
        assert stats.ror(_series([0.0] * 5)) < 1.0


class TestExpectedShortfall:
    def test_known_answer_parametric(self):
        # mean 0.006, sample std 0.0207364 (see FIVE)
        # ES = 0.006 - 0.0207364 * 2.0627128 = 0.006 - 0.0427733 = -0.0367733
        expected = 0.006 - np.sqrt(0.00043) * ES_K_95
        assert stats.expected_shortfall(_series(FIVE)) == pytest.approx(expected)
        assert expected == pytest.approx(-0.0367733232, rel=1e-8)

    def test_confidence_as_percent(self):
        r = _series(FIVE)
        assert stats.expected_shortfall(r, confidence=95) == pytest.approx(
            0.006 - np.sqrt(0.00043) * ES_K_95
        )

    def test_sigma_multiplier(self):
        # 0.006 - 2 * 0.0207364 * 2.0627128 = -0.0795466
        result = stats.expected_shortfall(_series(FIVE), sigma=2)
        assert result == pytest.approx(0.006 - 2 * np.sqrt(0.00043) * ES_K_95)

    def test_known_answer_historical(self):
        # sorted [-0.02, -0.01, 0.01, 0.02, 0.03]; q05: pos 0.2 -> -0.018
        # tail = values <= -0.018 = [-0.02] -> mean -0.02
        result = stats.expected_shortfall(_series(FIVE), method="historical")
        assert result == pytest.approx(-0.02)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            stats.expected_shortfall(_series(FIVE), method="bogus")

    def test_constant_returns(self):
        # zero dispersion: the whole distribution is 0.01
        assert stats.expected_shortfall(_series([0.01] * 5)) == pytest.approx(0.01)

    def test_empty_nan(self):
        assert np.isnan(stats.expected_shortfall(EMPTY))

    def test_all_nan(self):
        assert np.isnan(stats.expected_shortfall(_series([np.nan] * 5)))

    @pytest.mark.xfail(
        reason="undefined std of one observation is treated as zero risk", strict=True
    )
    def test_single_value_nan(self):
        assert np.isnan(stats.expected_shortfall(_series([-0.03])))

    def test_dataframe(self):
        result = stats.expected_shortfall(_frame(a=FIVE, b=[0.01] * 5))
        assert result["a"] == pytest.approx(0.006 - np.sqrt(0.00043) * ES_K_95)
        assert result["b"] == pytest.approx(0.01)


class TestMontecarloSharpe:
    def test_seed_is_reproducible(self):
        r = _series(FIVE)
        first = stats.montecarlo_sharpe(r, sims=50, seed=7)
        second = stats.montecarlo_sharpe(r, sims=50, seed=7)
        assert first == pytest.approx(second)

    def test_rf_lowers_every_statistic(self):
        r = _series(FIVE)
        base = stats.montecarlo_sharpe(r, sims=50, seed=0)
        with_rf = stats.montecarlo_sharpe(r, sims=50, seed=0, rf=0.0252)
        # rf / periods = 0.0001 per day is subtracted from each path's mean,
        # so the location statistics of the distribution all fall
        for key in ("min", "max", "mean", "median"):
            assert with_rf[key] < base[key]

    def test_dataframe_uses_first_column(self):
        r = _series(FIVE)
        df = pd.DataFrame({"a": r, "b": r * 2})
        assert stats.montecarlo_sharpe(df, sims=50, seed=0) == pytest.approx(
            stats.montecarlo_sharpe(r, sims=50, seed=0)
        )

    def test_constant_returns_all_nan(self):
        # zero volatility on every path -> no Sharpe values -> NaN summary
        result = stats.montecarlo_sharpe(_series([0.01] * 5), sims=10, seed=0)
        assert all(np.isnan(v) for v in result.values())

    def test_single_value_all_nan(self):
        result = stats.montecarlo_sharpe(_series([0.01]), sims=10, seed=0)
        assert all(np.isnan(v) for v in result.values())

    def test_empty_raises(self):
        with pytest.raises(DataValidationError):
            stats.montecarlo_sharpe(EMPTY, sims=10, seed=0)

    def test_all_nan_raises(self):
        with pytest.raises(DataValidationError):
            stats.montecarlo_sharpe(_series([np.nan] * 5), sims=10, seed=0)

    @pytest.mark.xfail(
        reason="pct_change on the cumulative path drops each path's first return",
        strict=True,
    )
    def test_known_answer_shuffles_preserve_sharpe(self):
        # Mean and std are permutation invariant, so every shuffled path has
        # the same Sharpe as the input:
        # 0.006 / 0.0207364 * sqrt(252) = 0.2893453 * 15.8745079 = 4.5932205
        result = stats.montecarlo_sharpe(_series(FIVE), sims=50, seed=0)
        expected = 0.006 / np.sqrt(0.00043) * np.sqrt(252)
        assert result["min"] == pytest.approx(expected)
        assert result["max"] == pytest.approx(expected)


class TestMontecarloCagr:
    def test_known_answer(self):
        # Terminal wealth is a product, so every shuffle ends at the same value:
        # 1.01 * 0.98 * 1.03 * 0.99 * 1.02 = 1.0294850412
        # years = 5 / 252 -> CAGR = 1.0294850412 ^ 50.4 - 1 = 3.3256367
        result = stats.montecarlo_cagr(_series(FIVE), sims=50, seed=0)
        expected = 1.0294850412 ** (252 / 5) - 1
        for key in ("min", "max", "mean", "median", "percentile_5", "percentile_95"):
            assert result[key] == pytest.approx(expected, rel=1e-9)
        assert result["std"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_returns(self):
        # 1.01^5 over 5/252 years -> 1.01^252 - 1 = 11.2740021
        result = stats.montecarlo_cagr(_series([0.01] * 5), sims=10, seed=0)
        assert result["mean"] == pytest.approx(1.01**252 - 1, rel=1e-9)

    def test_single_value(self):
        # one 1 % day over 1/252 years -> 1.01^252 - 1
        result = stats.montecarlo_cagr(_series([0.01]), sims=10, seed=0)
        assert result["median"] == pytest.approx(1.01**252 - 1, rel=1e-9)

    def test_total_loss_paths_are_excluded(self):
        # a -100 % day makes terminal wealth 0 on every path -> no valid CAGR
        result = stats.montecarlo_cagr(_series([0.1, -1.0, 0.2]), sims=10, seed=0)
        assert all(np.isnan(v) for v in result.values())

    def test_dataframe_uses_first_column(self):
        df = _frame(a=FIVE, b=[0.01] * 5)
        result = stats.montecarlo_cagr(df, sims=10, seed=0)
        assert result["mean"] == pytest.approx(1.0294850412 ** (252 / 5) - 1)

    def test_empty_raises(self):
        with pytest.raises(DataValidationError):
            stats.montecarlo_cagr(EMPTY, sims=10, seed=0)

    def test_all_nan_raises(self):
        with pytest.raises(DataValidationError):
            stats.montecarlo_cagr(_series([np.nan] * 5), sims=10, seed=0)


class TestMontecarloDrawdown:
    def test_gains_only_never_draw_down(self):
        result = stats.montecarlo_drawdown(_series([0.01, 0.02, 0.03]), sims=20, seed=0)
        assert result["min"] == pytest.approx(0.0)
        assert result["max"] == pytest.approx(0.0)

    @pytest.mark.xfail(
        reason="a loss on a path's first day is ignored (no 1.0 starting baseline)",
        strict=True,
    )
    def test_known_answer_first_day_loss(self):
        # Paths are [-0.10, 0.05] or [0.05, -0.10]:
        # 0.90 -> 0.945 has max drawdown 0.90 / 1.00 - 1 = -0.10
        # 1.05 -> 0.945 has max drawdown 0.945 / 1.05 - 1 = -0.10
        result = stats.montecarlo_drawdown(_series([-0.10, 0.05]), sims=20, seed=0)
        assert result["max"] == pytest.approx(-0.10)


class TestSmallUncalledFunctions:
    def test_geometric_mean(self):
        r = _series([0.10, -0.05, 0.02, 0.0, 0.03])
        # 1.10 * 0.95 * 1.02 * 1.00 * 1.03 = 1.097877; 1.097877^(1/5) - 1 = 0.0188511
        assert stats.geometric_mean(r) == pytest.approx(1.097877 ** (1 / 5) - 1)

    def test_outliers(self):
        r = _series([0.01, 0.10, 0.02, 0.04, 0.03])
        # q80: sorted [0.01, 0.02, 0.03, 0.04, 0.10], pos 3.2 -> 0.04 + 0.2 * 0.06
        # = 0.052; only 0.10 is above it
        result = stats.outliers(r, quantile=0.8)
        assert result.tolist() == [0.10]
        assert result.index[0] == pd.Timestamp("2021-01-02")

    def test_probabilistic_sortino_ratio(self):
        r = pd.Series(FIVE)
        # per-period sortino: downside = sqrt((0.02^2 + 0.01^2) / 5) = 0.01
        #   -> 0.006 / 0.01 = 0.6
        # sigma = sqrt((1 - skew * 0.6 + (kurt + 3 - 1) / 4 * 0.36) / 4)
        # with pandas skew -0.2355139 and excess kurt -1.9632234 -> 0.5349342
        # PSR = Phi(0.6 / 0.5349342) = Phi(1.1216331) = 0.8689908
        sk, ku = r.skew(), r.kurt()
        sigma = np.sqrt((1 - sk * 0.6 + ((ku + 2) / 4) * 0.36) / 4)
        expected = norm.cdf(0.6 / sigma)
        assert expected == pytest.approx(0.8689908157, rel=1e-8)
        assert stats.probabilistic_sortino_ratio(_series(FIVE)) == pytest.approx(
            expected
        )

    def test_probabilistic_adjusted_sortino_ratio(self):
        r = pd.Series(FIVE)
        # adjusted sortino = 0.6 / sqrt(2) = 0.4242641
        base = 0.6 / np.sqrt(2)
        sk, ku = r.skew(), r.kurt()
        sigma = np.sqrt((1 - sk * base + ((ku + 2) / 4) * base**2) / 4)
        expected = norm.cdf(base / sigma)
        assert expected == pytest.approx(0.7905874446, rel=1e-8)
        assert stats.probabilistic_adjusted_sortino_ratio(
            _series(FIVE)
        ) == pytest.approx(expected)
