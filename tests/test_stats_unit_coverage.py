"""
Known-answer and edge-case unit tests for quantstats.stats.

Every input is a small hand-built Series/DataFrame and every expected value is
derived by hand in the comment next to it, so the tests check the formula
rather than whatever the implementation currently returns.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from quantstats import stats
from quantstats.utils import DataValidationError


def _daily(values, start="2024-01-01", name=None):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float, name=name)


@pytest.fixture(autouse=True)
def _quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestCompare:
    """stats.compare with Series/DataFrame benchmarks (never a ticker)."""

    @pytest.fixture
    def pair(self):
        strategy = _daily([0.01, 0.02, -0.01, 0.03, 0.00])
        benchmark = _daily([0.005, 0.02, -0.02, 0.00, 0.01])
        return strategy, benchmark

    def test_daily_known_answer(self, pair):
        strategy, benchmark = pair
        result = stats.compare(strategy, benchmark)
        # Values are returns * 100.
        # Multiplier = Returns / Benchmark:
        #   1/0.5 = 2, 2/2 = 1, -1/-2 = 0.5, 3/0 -> NaN (zero guard), 0/1 = 0
        # Won = "+" when Returns >= Benchmark:
        #   1>=0.5 +, 2>=2 +, -1>=-2 +, 3>=0 +, 0>=1 -
        assert list(result.columns) == ["Benchmark", "Returns", "Multiplier", "Won"]
        np.testing.assert_allclose(result["Benchmark"], [0.5, 2.0, -2.0, 0.0, 1.0])
        np.testing.assert_allclose(result["Returns"], [1.0, 2.0, -1.0, 3.0, 0.0])
        np.testing.assert_allclose(
            result["Multiplier"], [2.0, 1.0, 0.5, np.nan, 0.0], equal_nan=True
        )
        assert list(result["Won"]) == ["+", "+", "+", "+", "-"]

    def test_monthly_compounded_and_simple(self):
        idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
        strategy = pd.Series([0.10, -0.05, 0.02, 0.03], index=idx)
        benchmark = pd.Series([0.01, 0.01, 0.00, 0.05], index=idx)

        comp = stats.compare(strategy, benchmark, aggregate="ME")
        # Compounded:
        #   strategy Jan = 1.10 * 0.95 - 1 = 0.045    -> 4.5
        #   strategy Feb = 1.02 * 1.03 - 1 = 0.0506   -> 5.06
        #   bench    Jan = 1.01 * 1.01 - 1 = 0.0201   -> 2.01
        #   bench    Feb = 1.00 * 1.05 - 1 = 0.05     -> 5.0
        np.testing.assert_allclose(comp["Returns"], [4.5, 5.06])
        np.testing.assert_allclose(comp["Benchmark"], [2.01, 5.0])
        # Multiplier: 4.5 / 2.01 = 2.238806..., 5.06 / 5 = 1.012
        np.testing.assert_allclose(comp["Multiplier"], [4.5 / 2.01, 1.012])
        assert list(comp["Won"]) == ["+", "+"]

        simple = stats.compare(strategy, benchmark, aggregate="ME", compounded=False)
        # Simple sums: strategy Jan 0.10 - 0.05 = 0.05 -> 5, Feb 0.05 -> 5
        #              bench    Jan 0.02 -> 2,            Feb 0.05 -> 5
        np.testing.assert_allclose(simple["Returns"], [5.0, 5.0])
        np.testing.assert_allclose(simple["Benchmark"], [2.0, 5.0])
        # Feb ties (5 >= 5) count as a win
        assert list(simple["Won"]) == ["+", "+"]

    def test_benchmark_days_outside_strategy_index_are_aggregated(self):
        # Strategy trades Tue-Fri; the benchmark also has Saturday 2024-02-03.
        idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
        strategy = pd.Series([0.10, -0.05, 0.02, 0.03], index=idx)
        bidx = idx.append(pd.DatetimeIndex(["2024-02-03"]))
        benchmark = pd.Series([0.01, 0.01, 0.00, 0.05, 0.10], index=bidx)
        result = stats.compare(strategy, benchmark, aggregate="ME")
        # Benchmark Feb = 1.00 * 1.05 * 1.10 - 1 = 0.155 -> 15.5 (Saturday kept)
        np.testing.assert_allclose(result["Benchmark"], [2.01, 15.5])
        # Strategy Feb 5.06 < 15.5 -> "-"
        assert list(result["Won"]) == ["+", "-"]

    def test_round_vals(self, pair):
        strategy, benchmark = pair
        strategy = strategy + 0.000123
        result = stats.compare(strategy, benchmark, round_vals=1)
        # (0.01 + 0.000123) * 100 = 1.0123 -> 1.0 ; 3.0123 -> 3.0
        np.testing.assert_allclose(result["Returns"], [1.0, 2.0, -1.0, 3.0, 0.0])

    def test_tz_aware_inputs_match_naive(self, pair):
        strategy, benchmark = pair
        naive = stats.compare(strategy, benchmark)
        aware = stats.compare(strategy.tz_localize("UTC"), benchmark.tz_localize("UTC"))
        assert aware.index.tz is None
        np.testing.assert_allclose(aware["Returns"], naive["Returns"])
        np.testing.assert_allclose(aware["Benchmark"], naive["Benchmark"])
        assert list(aware["Won"]) == list(naive["Won"])

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: tz-aware DataFrame benchmark keeps its tz and fails",
    )
    def test_tz_aware_dataframe_benchmark(self, pair):
        strategy, benchmark = pair
        bench_df = benchmark.tz_localize("UTC").to_frame("SPY")
        result = stats.compare(strategy, bench_df)
        # The first benchmark column is used: 0.5, 2, -2, 0, 1
        np.testing.assert_allclose(result["Benchmark"], [0.5, 2.0, -2.0, 0.0, 1.0])

    def test_dataframe_strategies(self, pair):
        strategy, benchmark = pair
        frame = pd.DataFrame({"a": strategy, "b": -strategy})
        result = stats.compare(frame, benchmark)
        # One Returns_<i> column per strategy, no Multiplier/Won
        assert list(result.columns) == ["Benchmark", "Returns_0", "Returns_1"]
        np.testing.assert_allclose(result["Returns_0"], [1.0, 2.0, -1.0, 3.0, 0.0])
        np.testing.assert_allclose(result["Returns_1"], [-1.0, -2.0, 1.0, -3.0, 0.0])
        np.testing.assert_allclose(result["Benchmark"], [0.5, 2.0, -2.0, 0.0, 1.0])

    def test_single_value(self, pair):
        strategy, benchmark = pair
        result = stats.compare(strategy.iloc[:1], benchmark.iloc[:1])
        # 1.0 / 0.5 = 2.0 and 1.0 >= 0.5
        assert result["Multiplier"].tolist() == [2.0]
        assert result["Won"].tolist() == ["+"]

    def test_constant_returns(self):
        strategy = _daily([0.01] * 5)
        benchmark = _daily([0.01] * 5)
        result = stats.compare(strategy, benchmark)
        # Identical series: every Multiplier is 1/1 = 1, every day is a tie (+)
        np.testing.assert_allclose(result["Multiplier"], [1.0] * 5)
        assert set(result["Won"]) == {"+"}

    @pytest.mark.xfail(
        strict=True, reason="possible bug: days with no strategy data are marked lost"
    )
    def test_all_nan_strategy_not_counted_as_losses(self, pair):
        _, benchmark = pair
        result = stats.compare(_daily([np.nan] * 5), benchmark)
        assert not (result["Won"] == "-").any()

    @pytest.mark.xfail(
        strict=True, reason="possible bug: empty strategy raises IndexError"
    )
    def test_empty_strategy(self, pair):
        _, benchmark = pair
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        assert stats.compare(empty, benchmark).empty

    def test_naive_dataframe_benchmark_uses_first_column(self, pair):
        strategy, benchmark = pair
        bench_df = pd.DataFrame({"SPY": benchmark, "QQQ": benchmark * 10})
        result = stats.compare(strategy, bench_df)
        # Only "SPY" is used: 0.5, 2, -2, 0, 1 (QQQ would be 5, 20, ...)
        np.testing.assert_allclose(result["Benchmark"], [0.5, 2.0, -2.0, 0.0, 1.0])


# Gaussian ES multiplier at 95%: pdf(ppf(0.05)) / 0.05 = 0.103136 / 0.05 = 2.062713
_ES95 = 2.0627128


class TestSerenityIndex:
    def test_known_answer(self):
        r = _daily([0.0, -0.5, 1.0, 0.0])
        # equity 1, 0.5, 1.0, 1.0 (start 1) -> drawdowns 0, -0.5, 0, 0
        # sum(r) = 0.5
        # std(r): mean 0.125, sq dev 0.015625+0.390625+0.765625+0.015625 = 1.1875
        #         /3 = 0.395833 -> 0.629153
        # ulcer = sqrt(0.25 / (4 - 1)) = 0.288675 (ulcer_index's n-1, see TestUpi)
        # cvar(dd) parametric: mean -0.125, std sqrt(0.1875/3) = 0.25
        #         = -0.125 - 0.25 * 2.062713 = -0.640678
        # pitfall = 0.640678 / 0.629153 = 1.018318
        # serenity = 0.5 / (0.288675 * 1.018318) = 1.700898
        std_r = np.sqrt(1.1875 / 3)
        pitfall = (0.125 + 0.25 * _ES95) / std_r
        expected = 0.5 / (np.sqrt(0.25 / 3) * pitfall)
        assert stats.serenity_index(r) == pytest.approx(expected, rel=1e-6)
        assert expected == pytest.approx(1.700898, rel=1e-5)

    def test_rf_is_subtracted_from_sum(self):
        r = _daily([0.0, -0.5, 1.0, 0.0])
        # numerator becomes 0.5 - 0.1 = 0.4, so result scales by 0.4 / 0.5
        base = stats.serenity_index(r)
        assert stats.serenity_index(r, rf=0.1) == pytest.approx(base * 0.8)

    def test_dataframe_matches_columns(self):
        r = _daily([0.0, -0.5, 1.0, 0.0])
        frame = pd.DataFrame({"a": r, "flat": _daily([0.01] * 4)})
        result = stats.serenity_index(frame)
        assert isinstance(result, pd.Series)
        assert result["a"] == pytest.approx(stats.serenity_index(r), rel=1e-9)
        # constant column has std 0 -> guarded to NaN
        assert np.isnan(result["flat"])

    def test_constant_returns_nan(self):
        # std == 0 -> pitfall undefined
        assert np.isnan(stats.serenity_index(_daily([0.01] * 5)))

    def test_no_drawdown_nan(self):
        # strictly rising equity: every drawdown is 0, cvar(dd) = 0, so the
        # denominator ulcer * pitfall = 0 -> NaN
        assert np.isnan(stats.serenity_index(_daily([0.01, 0.02, 0.03, 0.04])))

    def test_empty_and_all_nan_rejected(self):
        with pytest.raises(DataValidationError):
            stats.serenity_index(_daily([]))
        with pytest.raises(DataValidationError):
            stats.serenity_index(_daily([np.nan] * 5))

    def test_single_value(self):
        # one observation has no sample std
        assert np.isnan(stats.serenity_index(_daily([-0.05])))


class TestOmega:
    @pytest.fixture
    def r(self):
        return _daily([0.02, -0.01, 0.03, -0.02, 0.01])

    def test_known_answer_zero_threshold(self, r):
        # gains 0.02 + 0.03 + 0.01 = 0.06, losses 0.01 + 0.02 = 0.03 -> 2.0
        assert stats.omega(r) == pytest.approx(2.0)

    def test_per_period_threshold(self, r):
        # periods=1 uses required_return as-is: threshold 0.01
        # deviations 0.01, -0.02, 0.02, -0.03, 0.00
        # gains 0.03, losses 0.05 -> 0.6
        assert stats.omega(r, required_return=0.01, periods=1) == pytest.approx(0.6)

    def test_annual_threshold_is_deannualized(self, r):
        # (1 + (1.01**252 - 1)) ** (1/252) - 1 = 0.01 per period -> same 0.6
        annual = 1.01**252 - 1
        assert stats.omega(r, required_return=annual) == pytest.approx(0.6)

    def test_rf_shifts_returns(self, r):
        # periods=1: rf 0.01 is subtracted as-is -> same deviations as above
        assert stats.omega(r, rf=0.01, periods=1) == pytest.approx(0.6)

    def test_no_losses_nan(self):
        # nothing below the threshold -> denominator 0 -> NaN
        assert np.isnan(stats.omega(_daily([0.01, 0.02, 0.03])))

    def test_constant_returns_nan(self):
        # constant 0 returns: no gains and no losses -> NaN
        assert np.isnan(stats.omega(_daily([0.0] * 5)))

    def test_single_value_nan(self):
        assert np.isnan(stats.omega(_daily([0.01])))

    def test_invalid_required_return_nan(self, r):
        assert np.isnan(stats.omega(r, required_return=-1.0))

    def test_empty_and_all_nan_rejected(self):
        with pytest.raises(DataValidationError):
            stats.omega(_daily([]))
        with pytest.raises(DataValidationError):
            stats.omega(_daily([np.nan] * 5))

    def test_dataframe(self, r):
        frame = pd.DataFrame({"a": r, "up": _daily([0.01, 0.02, 0.03, 0.01, 0.02])})
        result = stats.omega(frame)
        assert isinstance(result, pd.Series)
        # column a as above = 2.0; column "up" has no losses -> NaN
        assert result["a"] == pytest.approx(2.0)
        assert np.isnan(result["up"])


class TestTailRatio:
    @pytest.fixture
    def r(self):
        return _daily([-0.04, -0.01, 0.00, 0.03, 0.06])

    def test_known_answer_default_cutoff(self, r):
        # linear quantiles on 5 sorted points (positions 0..4):
        #   q95 at 3.8 -> 0.03 + 0.8 * (0.06 - 0.03) = 0.054
        #   q05 at 0.2 -> -0.04 + 0.2 * (-0.01 + 0.04) = -0.034
        #   |0.054 / -0.034| = 1.588235
        assert stats.tail_ratio(r) == pytest.approx(0.054 / 0.034)

    def test_known_answer_custom_cutoff(self, r):
        # q75 at position 3 -> 0.03, q25 at position 1 -> -0.01 -> 3.0
        assert stats.tail_ratio(r, cutoff=0.75) == pytest.approx(3.0)

    def test_zero_lower_tail_nan(self):
        # q25 of [0, 0, 0, 0.01, 0.02] is 0 -> division guarded -> NaN
        assert np.isnan(stats.tail_ratio(_daily([0, 0, 0, 0.01, 0.02]), cutoff=0.75))

    def test_constant_returns(self):
        # both quantiles equal 0.01 -> 1.0; constant 0 -> zero denominator -> NaN
        assert stats.tail_ratio(_daily([0.01] * 5)) == pytest.approx(1.0)
        assert np.isnan(stats.tail_ratio(_daily([0.0] * 5)))

    def test_single_value(self):
        # one observation is every quantile -> |-0.02 / -0.02| = 1.0
        assert stats.tail_ratio(_daily([-0.02])) == pytest.approx(1.0)

    def test_empty_and_all_nan(self):
        assert np.isnan(stats.tail_ratio(_daily([])))
        assert np.isnan(stats.tail_ratio(_daily([np.nan] * 5)))

    def test_dataframe(self, r):
        frame = pd.DataFrame(
            {"a": r, "zero": _daily([0.0] * 5), "gap": _daily([np.nan] * 5)}
        )
        result = stats.tail_ratio(frame, cutoff=0.75)
        assert isinstance(result, pd.Series)
        # a -> 3.0 as above; zero lower tail -> NaN; all-NaN column -> NaN
        assert result["a"] == pytest.approx(3.0)
        assert np.isnan(result["zero"])
        assert np.isnan(result["gap"])


class TestPctRank:
    def test_known_answer(self):
        prices = _daily([100, 105, 110, 95, 120])
        result = stats.pct_rank(prices, window=3)
        # Rank of today's price among the last 3 (today, t-1, t-2), as % of 3:
        #   t2: [110, 105, 100] -> rank 3/3 = 100
        #   t3: [ 95, 110, 105] -> rank 1/3 = 33.33
        #   t4: [120,  95, 110] -> rank 3/3 = 100
        np.testing.assert_allclose(result.iloc[2:], [100.0, 100 / 3, 100.0])
        assert result.index.equals(prices.index)

    def test_ties_use_average_rank(self):
        result = stats.pct_rank(_daily([5.0] * 4), window=3)
        # three equal prices share the average rank (1+2+3)/3 = 2 -> 2/3 = 66.67
        np.testing.assert_allclose(result.iloc[2:], [200 / 3, 200 / 3])

    def test_single_value(self):
        # one price is the only member of its window -> rank 1/1 = 100
        assert stats.pct_rank(_daily([42.0]), window=3).tolist() == [100.0]

    def test_empty_and_all_nan(self):
        assert stats.pct_rank(_daily([]), window=3).empty
        assert stats.pct_rank(_daily([np.nan] * 4), window=3).isna().all()


class TestImpliedVolatility:
    @pytest.fixture
    def r(self):
        # simple returns whose log returns are exactly 0.01, -0.01, 0.02, -0.02, 0
        return _daily(np.expm1([0.01, -0.01, 0.02, -0.02, 0.0]))

    def test_known_answer_not_annualized(self, r):
        # log returns mean 0; squares 1e-4 + 1e-4 + 4e-4 + 4e-4 + 0 = 1e-3
        # sample var = 1e-3 / 4 = 2.5e-4 -> std = 0.0158114
        assert stats.implied_volatility(r, annualize=False) == pytest.approx(
            np.sqrt(2.5e-4)
        )

    def test_known_answer_rolling_annualized(self, r):
        result = stats.implied_volatility(r, periods=5)
        # rolling(5) is defined only on the last row: 0.0158114 * sqrt(5) = 0.0353553
        assert result.iloc[:4].isna().all()
        assert result.iloc[-1] == pytest.approx(np.sqrt(2.5e-4) * np.sqrt(5))

    def test_constant_returns_zero(self):
        assert stats.implied_volatility(
            _daily([0.01] * 5), annualize=False
        ) == pytest.approx(0.0)

    def test_single_empty_all_nan(self):
        for values in ([0.01], [], [np.nan] * 3):
            assert np.isnan(stats.implied_volatility(_daily(values), annualize=False))
            assert stats.implied_volatility(_daily(values), periods=2).isna().all()

    def test_dataframe(self, r):
        frame = pd.DataFrame({"a": r, "flat": _daily([0.01] * 5)})
        result = stats.implied_volatility(frame, annualize=False)
        # per column: a -> 0.0158114, flat -> 0
        np.testing.assert_allclose(result[["a", "flat"]], [np.sqrt(2.5e-4), 0.0])


class TestR2:
    @pytest.fixture
    def bench(self):
        return _daily([0.01, 0.02, 0.03, 0.04, 0.05])

    def test_known_answer(self, bench):
        strategy = _daily([0.02, 0.01, 0.04, 0.03, 0.05])
        # deviations (x0.01): bench -2,-1,0,1,2 ; strategy -1,-2,1,0,2
        # Sxy = 2+2+0+0+4 = 8, Sxx = 10, Syy = 1+4+1+0+4 = 10
        # r = 8 / sqrt(10*10) = 0.8 -> r^2 = 0.64
        assert stats.r2(strategy, bench) == pytest.approx(0.64)
        assert stats.r2(strategy, bench) == pytest.approx(
            stats.r_squared(strategy, bench)
        )

    def test_perfect_linear_fit(self, bench):
        # strategy = 2 * bench - 0.01 -> r^2 = 1
        assert stats.r2(2 * bench - 0.01, bench) == pytest.approx(1.0)

    def test_gap_uses_paired_days_only(self, bench):
        strategy = _daily([0.02, np.nan, 0.04, 0.03, 0.05])
        # drop day 2 -> bench 0.01,0.03,0.04,0.05 (mean 0.0325),
        # strategy 0.02,0.04,0.03,0.05 (mean 0.035); deviations x0.0001:
        #   bench -225,-25,75,175 ; strategy -150,50,-50,150
        # Sxy = 33750-1250-3750+26250 = 55000 (x1e-8)
        # Sxx = 50625+625+5625+30625 = 87500 ; Syy = 22500+2500+2500+22500 = 50000
        # r^2 = 55000^2 / (87500 * 50000) = 0.691429
        assert stats.r2(strategy, bench) == pytest.approx(55000**2 / (87500 * 50000))

    def test_degenerate_inputs_nan(self, bench):
        # constant strategy, single pair, all-NaN strategy: correlation undefined
        assert np.isnan(stats.r2(_daily([0.01] * 5), bench))
        assert np.isnan(stats.r2(_daily([0.01]), _daily([0.02])))
        assert np.isnan(stats.r2(_daily([np.nan] * 5), bench))
        assert np.isnan(stats.r2(_daily([]), _daily([])))


class TestUpi:
    @pytest.fixture
    def r(self):
        # equity 1.1, 0.99, 1.089 -> peak 1.1 -> drawdowns 0, -0.1, -0.01
        # comp = 1.1 * 0.9 * 1.1 - 1 = 0.089
        return _daily([0.10, -0.10, 0.10])

    def test_numerator_is_compounded_excess_return(self, r):
        # upi * ulcer = comp(r) - rf = 0.089 - 0.0 ; with rf 0.089 -> 0
        assert stats.upi(r) * stats.ulcer_index(r) == pytest.approx(0.089)
        assert stats.upi(r, rf=0.089) == pytest.approx(0.0, abs=1e-12)
        assert stats.upi(r) == stats.ulcer_performance_index(r)

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: ulcer_index divides by n-1, docstring says mean",
    )
    def test_known_answer_mean_squared_drawdown(self, r):
        # Ulcer = sqrt(mean(dd^2)) = sqrt((0 + 0.01 + 0.0001) / 3) = 0.0580230
        # UPI = 0.089 / 0.0580230 = 1.533874
        assert stats.upi(r) == pytest.approx(0.089 / np.sqrt(0.0101 / 3))

    def test_no_drawdown_nan(self):
        # rising equity: ulcer 0 -> guarded to NaN
        assert np.isnan(stats.upi(_daily([0.01, 0.02, 0.03])))
        assert np.isnan(stats.upi(_daily([0.01] * 5)))

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: single observation divides by n-1 = 0 (ulcer inf)",
    )
    def test_single_value(self):
        # one -5% day: dd = -0.05, ulcer = sqrt(0.0025 / 1) = 0.05, UPI = -1.0
        assert stats.upi(_daily([-0.05])) == pytest.approx(-1.0)

    def test_empty_and_all_nan_rejected(self):
        with pytest.raises(DataValidationError):
            stats.upi(_daily([]))
        with pytest.raises(DataValidationError):
            stats.upi(_daily([np.nan] * 3))

    def test_dataframe(self, r):
        frame = pd.DataFrame({"a": r, "up": _daily([0.01, 0.02, 0.03])})
        result = stats.upi(frame)
        assert isinstance(result, pd.Series)
        assert result["a"] == pytest.approx(stats.upi(r))
        # no drawdown in "up" -> NaN
        assert np.isnan(result["up"])


class TestRor:
    def test_known_answer(self):
        r = _daily([0.01, -0.01, 0.02, 0.03])
        # win rate 3/4 = 0.75 -> ((1 - 0.75) / (1 + 0.75)) ** 4 = (1/7) ** 4 = 1/2401
        assert stats.ror(r) == pytest.approx(1 / 2401)
        assert stats.ror(r) == stats.risk_of_ruin(r)

    def test_all_wins_and_all_losses(self):
        # win rate 1 -> 0 ** n = 0 ; win rate 0 -> 1 ** n = 1
        assert stats.ror(_daily([0.01] * 5)) == 0.0
        assert stats.ror(_daily([-0.01] * 5)) == 1.0

    def test_single_value(self):
        # one winning day: ((1 - 1) / 2) ** 1 = 0
        assert stats.ror(_daily([0.02])) == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: no data / no losses reported as certain ruin (1.0)",
    )
    @pytest.mark.parametrize(
        "values", [[], [np.nan] * 3, [0.0] * 3], ids=["empty", "all_nan", "flat"]
    )
    def test_undefined_input_not_certain_ruin(self, values):
        assert np.isnan(stats.ror(_daily(values)))

    def test_dataframe(self):
        frame = pd.DataFrame(
            {"a": _daily([0.01, -0.01, 0.02, 0.03]), "b": _daily([-0.01] * 4)}
        )
        result = stats.ror(frame)
        # a -> 1/2401 as above ; b never wins -> 1.0
        np.testing.assert_allclose(result[["a", "b"]], [1 / 2401, 1.0])


class TestExpectedShortfall:
    @pytest.fixture
    def r(self):
        return _daily([0.01, -0.02, 0.03, -0.01, 0.02])

    # mean 0.006; deviations 0.004,-0.026,0.024,-0.016,0.014
    # squares sum 0.00172 -> sample var 0.00043 -> sd 0.0207364
    _SD = np.sqrt(0.00043)

    def test_parametric_known_answer(self, r):
        # 0.006 - 0.0207364 * 2.062713 = -0.0367732
        expected = 0.006 - self._SD * _ES95
        assert stats.expected_shortfall(r) == pytest.approx(expected, rel=1e-6)
        assert expected == pytest.approx(-0.0367732, rel=1e-5)

    def test_sigma_and_percent_confidence(self, r):
        # sigma=2 doubles the sd term: 0.006 - 2 * 0.0207364 * 2.062713
        assert stats.expected_shortfall(r, sigma=2) == pytest.approx(
            0.006 - 2 * self._SD * _ES95, rel=1e-6
        )
        # confidence 95 is read as 0.95
        assert stats.expected_shortfall(r, confidence=95) == pytest.approx(
            stats.expected_shortfall(r, confidence=0.95)
        )

    def test_historical_known_answer(self, r):
        # sorted -0.02,-0.01,0.01,0.02,0.03 ; q05 at position 0.2:
        #   -0.02 + 0.2 * 0.01 = -0.018 -> tail [-0.02] -> mean -0.02
        assert stats.expected_shortfall(r, method="historical") == pytest.approx(-0.02)

    def test_invalid_method(self, r):
        with pytest.raises(ValueError):
            stats.expected_shortfall(r, method="cornish-fisher")

    def test_constant_returns(self):
        # sd 0 -> the distribution is a point mass at the mean
        assert stats.expected_shortfall(_daily([0.01] * 5)) == pytest.approx(0.01)

    def test_empty_and_all_nan(self):
        assert np.isnan(stats.expected_shortfall(_daily([])))
        assert np.isnan(stats.expected_shortfall(_daily([np.nan] * 5)))

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: one observation has no sd, but ES returns the value",
    )
    def test_single_value_nan(self):
        assert np.isnan(stats.expected_shortfall(_daily([0.03])))

    def test_dataframe(self, r):
        frame = pd.DataFrame({"a": r, "gap": _daily([np.nan] * 5)})
        result = stats.expected_shortfall(frame)
        assert result["a"] == pytest.approx(0.006 - self._SD * _ES95, rel=1e-6)
        assert np.isnan(result["gap"])


class TestMontecarloSharpe:
    # Shuffled paths are permutations of the same returns, so mean and std -
    # and therefore Sharpe - are identical on every path.
    @pytest.fixture
    def r(self):
        return _daily([0.01, -0.02, 0.03, -0.01, 0.02])

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: pct_change on each path drops its first return",
    )
    def test_known_answer_path_invariant(self, r):
        # mean 0.006, sd sqrt(0.00043) = 0.0207364
        # Sharpe = 0.006 / 0.0207364 * sqrt(252) = 4.593243 on every path
        expected = 0.006 / np.sqrt(0.00043) * np.sqrt(252)
        result = stats.montecarlo_sharpe(r, sims=50, seed=7)
        assert result["min"] == pytest.approx(expected)
        assert result["max"] == pytest.approx(expected)

    def test_seed_is_reproducible(self, r):
        a = stats.montecarlo_sharpe(r, sims=50, seed=7)
        b = stats.montecarlo_sharpe(r, sims=50, seed=7)
        assert a == b
        assert set(a) == {
            "min",
            "max",
            "mean",
            "median",
            "std",
            "percentile_5",
            "percentile_95",
        }
        # ordering invariants of the summary
        assert a["min"] <= a["percentile_5"] <= a["median"] <= a["percentile_95"]
        assert a["percentile_95"] <= a["max"]

    def test_constant_returns_nan(self):
        # std 0 on every path -> Sharpe undefined -> no values -> NaN summary
        result = stats.montecarlo_sharpe(_daily([0.01] * 5), sims=10, seed=1)
        assert all(np.isnan(v) for v in result.values())

    def test_single_value_nan(self):
        # one observation has no std -> NaN summary
        result = stats.montecarlo_sharpe(_daily([0.01]), sims=10, seed=1)
        assert np.isnan(result["mean"])

    def test_empty_and_all_nan_rejected(self):
        with pytest.raises(DataValidationError):
            stats.montecarlo_sharpe(_daily([]), sims=10, seed=1)
        with pytest.raises(DataValidationError):
            stats.montecarlo_sharpe(_daily([np.nan] * 5), sims=10, seed=1)

    def test_dataframe_uses_first_column(self, r):
        frame = pd.DataFrame({"a": r, "b": -r})
        assert stats.montecarlo_sharpe(frame, sims=20, seed=3) == (
            stats.montecarlo_sharpe(r, sims=20, seed=3)
        )


class TestMontecarloCagr:
    # Every shuffled path compounds the same returns, so the terminal value
    # and the CAGR are identical on every path.
    @pytest.fixture
    def r(self):
        return _daily([0.01, -0.02, 0.03, -0.01, 0.02])

    def test_known_answer_path_invariant(self, r):
        # terminal growth = 1.01 * 0.98 * 1.03 * 0.99 * 1.02 = 1.0294850412
        # years = 5 / 252 -> CAGR = 1.0294850412 ** 50.4 - 1 = 3.3253...
        expected = 1.0294850412 ** (252 / 5) - 1
        result = stats.montecarlo_cagr(r, sims=50, seed=11)
        for key in ("min", "max", "mean", "median", "percentile_5", "percentile_95"):
            assert result[key] == pytest.approx(expected, rel=1e-9)
        assert result["std"] == pytest.approx(0.0, abs=1e-9)

    def test_constant_and_single_value(self):
        # constant 1% for 5 days: (1.01 ** 5) ** (252/5) - 1 = 1.01 ** 252 - 1
        # single 1% day: 1.01 ** (252/1) - 1 -> the same number
        expected = 1.01**252 - 1
        assert stats.montecarlo_cagr(_daily([0.01] * 5), sims=5, seed=0)[
            "mean"
        ] == pytest.approx(expected)
        assert stats.montecarlo_cagr(_daily([0.01]), sims=5, seed=0)[
            "mean"
        ] == pytest.approx(expected)

    @pytest.mark.xfail(
        strict=True,
        reason="possible bug: a total-loss path (terminal == -1) is dropped",
    )
    def test_total_loss_path_is_minus_one(self):
        # 1.10 * 0.00 = 0 -> terminal -1 -> CAGR = 0 ** (1/years) - 1 = -1
        result = stats.montecarlo_cagr(_daily([0.10, -1.0]), sims=5, seed=0)
        assert result["mean"] == pytest.approx(-1.0)

    def test_empty_and_all_nan_rejected(self):
        with pytest.raises(DataValidationError):
            stats.montecarlo_cagr(_daily([]), sims=5, seed=0)
        with pytest.raises(DataValidationError):
            stats.montecarlo_cagr(_daily([np.nan] * 5), sims=5, seed=0)

    def test_dataframe_uses_first_column(self, r):
        frame = pd.DataFrame({"a": r, "b": -r})
        expected = 1.0294850412 ** (252 / 5) - 1
        assert stats.montecarlo_cagr(frame, sims=10, seed=2)["mean"] == (
            pytest.approx(expected)
        )
