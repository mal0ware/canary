"""VaR and expected shortfall: closed forms, order statistics, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from canary import backtest as bt
from canary import garch, var


def test_empirical_tail_is_order_statistic() -> None:
    x = np.arange(1.0, 1001.0)
    rng = np.random.default_rng(0)
    rng.shuffle(x)
    q, tail = var.empirical_tail(x, 0.01)
    assert q == 10.0  # the 10th smallest of 1000
    assert tail == pytest.approx(5.5)  # mean of 1..10
    q5, _ = var.empirical_tail(x, 0.05)
    assert q5 == 50.0


def test_empirical_tail_small_sample_uses_minimum() -> None:
    q, tail = var.empirical_tail(np.array([3.0, -1.0, 2.0]), 0.01)
    assert q == -1.0 and tail == -1.0


def test_normal_var_known_values() -> None:
    v, e = var.normal_var_es(0.0, 1.0, 0.01)
    assert v == pytest.approx(2.326348, abs=1e-6)
    assert e == pytest.approx(2.665214, abs=1e-6)
    v5, e5 = var.normal_var_es(0.0, 1.0, 0.05)
    assert v5 == pytest.approx(1.644854, abs=1e-6)
    assert e5 == pytest.approx(2.062713, abs=1e-6)


def test_normal_es_matches_numerical_integral() -> None:
    mu, sigma, alpha = 0.001, 0.02, 0.025
    _, es = var.normal_var_es(mu, sigma, alpha)
    q = stats.norm.ppf(alpha, mu, sigma)
    integral, _ = integrate.quad(lambda x: x * stats.norm.pdf(x, mu, sigma), -np.inf, q)
    assert es == pytest.approx(-integral / alpha, rel=1e-8)


def test_es_is_at_least_var() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_t(4, size=1000)
    for a in (0.01, 0.05):
        v, e = var.hs_var_es(x, a)
        assert e >= v


def test_hs_coverage_is_exact_for_exchangeable_data() -> None:
    # P(new draw < k-th smallest of n) = k / (n + 1) for continuous iid data.
    rng = np.random.default_rng(2)
    n, alpha, reps = 200, 0.05, 40_000
    x = rng.standard_normal((reps, n + 1))
    k = int(np.ceil(alpha * n))
    q = np.partition(x[:, :n], k - 1, axis=1)[:, k - 1]
    assert np.mean(x[:, n] < q) == pytest.approx(k / (n + 1), abs=0.004)


def test_fhs_reduces_to_hs_under_constant_volatility() -> None:
    rng = np.random.default_rng(3)
    win = rng.standard_normal(1000) * 0.01
    mu = float(win.mean())
    eps = win - mu
    sigma = 0.01
    v_fhs, e_fhs = var.fhs_var_es(mu, sigma, eps / sigma, 0.01)
    v_hs, e_hs = var.hs_var_es(win, 0.01)
    assert v_fhs == pytest.approx(v_hs, rel=1e-12)
    assert e_fhs == pytest.approx(e_hs, rel=1e-12)


def test_fhs_scales_with_forecast_volatility() -> None:
    rng = np.random.default_rng(4)
    z = rng.standard_normal(500)
    v1, e1 = var.fhs_var_es(0.0, 1.0, z, 0.05)
    v2, e2 = var.fhs_var_es(0.0, 2.5, z, 0.05)
    assert v2 == pytest.approx(2.5 * v1)
    assert e2 == pytest.approx(2.5 * e1)


@pytest.fixture(scope="module")
def garch_returns() -> pd.Series:
    params = garch.GarchParams(omega=2e-6, alpha=0.10, beta=0.88)
    eps, _ = garch.simulate(2600, params, np.random.default_rng(5))
    return pd.Series(eps, index=pd.bdate_range("2000-01-03", periods=eps.size))


@pytest.fixture(scope="module")
def forecasts(garch_returns: pd.Series) -> pd.DataFrame:
    return var.rolling_forecasts(garch_returns, var.BacktestConfig(window=500, refit=50))


def test_rolling_forecasts_shape_and_columns(
    garch_returns: pd.Series, forecasts: pd.DataFrame
) -> None:
    assert len(forecasts) == len(garch_returns) - 500
    assert forecasts.index[0] == garch_returns.index[500]
    for model in var.MODELS:
        for a in (0.01, 0.05):
            assert (forecasts[f"var_{model}_{a}"] > 0).all()
            assert (forecasts[f"es_{model}_{a}"] >= forecasts[f"var_{model}_{a}"] - 1e-15).all()


def test_first_forecast_matches_direct_calculation(
    garch_returns: pd.Series, forecasts: pd.DataFrame
) -> None:
    win = garch_returns.to_numpy()[:500]
    v, e = var.hs_var_es(win, 0.01)
    assert forecasts["var_hs_0.01"].iloc[0] == pytest.approx(v)
    assert forecasts["es_hs_0.01"].iloc[0] == pytest.approx(e)
    vn, _ = var.normal_var_es(float(win.mean()), float(win.std(ddof=1)), 0.05)
    assert forecasts["var_normal_0.05"].iloc[0] == pytest.approx(vn)


def test_no_look_ahead(garch_returns: pd.Series, forecasts: pd.DataFrame) -> None:
    # Changing returns from day 600 on must not change any forecast up to day 600.
    changed = garch_returns.copy()
    changed.iloc[600:] = changed.iloc[600:] * 3.0
    again = var.rolling_forecasts(changed, var.BacktestConfig(window=500, refit=50))
    cols = [c for c in forecasts.columns if c != "ret"]
    pd.testing.assert_frame_equal(forecasts[cols].iloc[:101], again[cols].iloc[:101])
    assert not np.allclose(forecasts["var_fhs_0.01"].iloc[101:], again["var_fhs_0.01"].iloc[101:])


def test_garch_models_have_independent_breaches_on_garch_data(forecasts: pd.DataFrame) -> None:
    # The data are GARCH with normal shocks, so the GARCH-normal model is right
    # by construction: it passes both tests at 5%. Historical simulation ignores
    # the volatility clustering and its breaches bunch.
    hits = var.breaches(forecasts, "garch_normal", 0.05)
    assert bt.kupiec(hits, 0.05).p_value > 0.01
    assert bt.christoffersen(hits, 0.05).p_ind > 0.01
    hs = bt.christoffersen(var.breaches(forecasts, "hs", 0.05), 0.05)
    gn = bt.christoffersen(hits, 0.05)
    assert hs.pi11 > gn.pi11


def test_shortfall_ratio_is_one_for_perfect_es() -> None:
    frame = pd.DataFrame(
        {
            "ret": [-0.03, 0.01, -0.05, 0.0],
            "var_hs_0.01": [0.02, 0.02, 0.02, 0.02],
            "es_hs_0.01": [0.04, 0.04, 0.04, 0.04],
        }
    )
    assert var.shortfall_ratio(frame, "hs", 0.01) == pytest.approx(1.0)
    assert var.breaches(frame, "hs", 0.01).tolist() == [1, 0, 1, 0]
