"""Realised variance, HAR, QLIKE and the Diebold-Mariano test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canary import volforecast as vf


def test_realized_variance_is_sum_of_squares() -> None:
    assert vf.realized_variance(np.array([0.01, -0.02, 0.03])) == pytest.approx(0.0014)


def test_realized_variance_recovers_constant_volatility() -> None:
    # 288 five-minute returns a day with daily variance 4e-4, over 400 days.
    rng = np.random.default_rng(0)
    r = rng.standard_normal((400, 288)) * np.sqrt(4e-4 / 288)
    rv = (r**2).sum(axis=1)
    assert rv.mean() == pytest.approx(4e-4, rel=0.01)


def _bars(days: int, per_day: int, rng: np.random.Generator) -> pd.Series:
    idx = pd.date_range("2021-01-01", periods=days * per_day, freq=pd.Timedelta(days=1) / per_day)
    logp = np.cumsum(rng.standard_normal(idx.size) * 0.001)
    return pd.Series(np.exp(logp), index=idx)


def test_daily_realized_variance_matches_direct_sum() -> None:
    close = _bars(5, 24, np.random.default_rng(1))
    out = vf.daily_realized_variance(close, bars_per_day=24)
    logp = np.log(close.to_numpy())
    # Day 2 (index 1): returns into bars 24..47.
    direct = np.sum(np.diff(logp)[23:47] ** 2)
    assert out["rv"].iloc[0] == pytest.approx(direct)
    assert out["ret"].iloc[0] == pytest.approx(logp[47] - logp[23])
    # The first day has no previous close, so it is dropped.
    assert len(out) == 4


def test_daily_realized_variance_handles_gaps() -> None:
    close = _bars(4, 24, np.random.default_rng(2))
    # Remove one bar from day 2 (two returns lost) and most of day 3.
    drop = list(close.index[30:31]) + list(close.index[50:70])
    out = vf.daily_realized_variance(close.drop(drop), bars_per_day=24, min_coverage=0.9)
    assert pd.Timestamp("2021-01-03") not in out.index
    day2 = out.loc[pd.Timestamp("2021-01-02")]
    assert day2["n"] == 22
    logp = np.log(close.to_numpy())
    r = np.diff(logp)[23:47]
    kept = np.delete(r, [6, 7])  # returns into bar 30 and out of it
    assert day2["rv"] == pytest.approx(np.sum(kept**2) * 24 / 22)


def test_har_design_windows() -> None:
    rv = np.arange(1.0, 31.0)
    x = vf.har_design(rv)
    assert np.isnan(x[20, 3]) and not np.isnan(x[21, 3])
    assert x[21, 1] == 22.0
    assert x[21, 2] == pytest.approx(np.mean([18, 19, 20, 21, 22]))
    assert x[21, 3] == pytest.approx(np.mean(np.arange(1, 23)))


def test_har_fit_recovers_coefficients() -> None:
    rng = np.random.default_rng(3)
    true = np.array([0.1, 0.35, 0.30, 0.25])
    n = 60_000
    rv = np.ones(n)
    for t in range(21, n - 1):
        x = np.array([1.0, rv[t], rv[t - 4 : t + 1].mean(), rv[t - 21 : t + 1].mean()])
        rv[t + 1] = x @ true + 0.05 * rng.standard_normal()
    np.testing.assert_allclose(vf.har_fit(rv[1000:]), true, atol=0.03)


def test_har_predict_uses_last_rows() -> None:
    rng = np.random.default_rng(4)
    rv = rng.random(100) + 0.5
    beta = np.array([0.1, 0.2, 0.3, 0.4])
    expected = 0.1 + 0.2 * rv[-1] + 0.3 * rv[-5:].mean() + 0.4 * rv[-22:].mean()
    assert vf.har_predict(rv, beta) == pytest.approx(expected)


def test_qlike_zero_at_truth_and_positive_elsewhere() -> None:
    proxy = np.array([1.0, 2.0, 0.5])
    assert np.allclose(vf.qlike(proxy, proxy), 0.0)
    assert (vf.qlike(proxy, proxy * 1.5) > 0).all()
    assert (vf.qlike(proxy, proxy * 0.5) > 0).all()


def test_qlike_penalises_low_forecasts_more() -> None:
    proxy = np.array([1.0])
    assert vf.qlike(proxy, proxy / 2)[0] > vf.qlike(proxy, proxy * 2)[0]


def test_qlike_expected_loss_is_minimised_at_true_variance_with_noisy_proxy() -> None:
    # The proxy is true variance times a mean-one chi-squared shock.
    rng = np.random.default_rng(5)
    true = 2.0
    proxy = true * rng.chisquare(1, size=400_000)
    grid = np.linspace(1.0, 3.0, 41)
    losses = [np.mean(proxy / h + np.log(h)) for h in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(true, abs=0.051)


def test_qlike_difference_equals_hansen_lunde_form() -> None:
    rng = np.random.default_rng(6)
    proxy, a, b = rng.random(50) + 0.5, rng.random(50) + 0.5, rng.random(50) + 0.5
    mine = vf.qlike(proxy, a) - vf.qlike(proxy, b)
    other = (np.log(a) + proxy / a) - (np.log(b) + proxy / b)
    np.testing.assert_allclose(mine, other, atol=1e-12)


def test_long_run_variance_lag_zero_is_variance() -> None:
    rng = np.random.default_rng(7)
    d = rng.standard_normal(500)
    assert vf.long_run_variance(d, 0) == pytest.approx(np.var(d))


def test_long_run_variance_of_ma1() -> None:
    # d_t = e_t + 0.5 e_{t-1}: long-run variance (1 + 0.5)^2 = 2.25.
    rng = np.random.default_rng(8)
    e = rng.standard_normal(400_001)
    d = e[1:] + 0.5 * e[:-1]
    assert vf.long_run_variance(d, 1) == pytest.approx(2.25, rel=0.02)


def test_dm_hand_value() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 2.5, 2.0, 3.0])
    d = a - b
    expected = d.mean() / np.sqrt(np.var(d) / 4)
    assert vf.diebold_mariano(a, b).stat == pytest.approx(expected)


def test_dm_sign_and_symmetry() -> None:
    rng = np.random.default_rng(9)
    a = rng.random(300)
    b = a + 0.1 + 0.05 * rng.standard_normal(300)
    res = vf.diebold_mariano(a, b)
    assert res.stat < 0 and res.p_value < 0.001
    assert vf.diebold_mariano(b, a).stat == pytest.approx(-res.stat)


def test_dm_size_under_null() -> None:
    rng = np.random.default_rng(10)
    reps = 3000
    rejections = 0
    for _ in range(reps):
        a = rng.standard_normal(300) ** 2
        b = rng.standard_normal(300) ** 2
        rejections += vf.diebold_mariano(a, b).p_value < 0.05
    assert 0.035 < rejections / reps < 0.065


def test_dm_identical_forecasts() -> None:
    a = np.array([1.0, 2.0, 3.0])
    res = vf.diebold_mariano(a, a)
    assert res.mean_diff == 0.0 and np.isnan(res.stat)


def test_rolling_vol_forecasts_no_look_ahead() -> None:
    rng = np.random.default_rng(11)
    n = 400
    rv = (rng.random(n) + 0.5) * 1e-4
    ret = rng.standard_normal(n) * np.sqrt(rv)
    daily = pd.DataFrame({"rv": rv, "ret": ret}, index=pd.date_range("2020-01-01", periods=n))
    base = vf.rolling_vol_forecasts(daily, window=300, refit=20)
    assert len(base) == 100
    changed = daily.copy()
    changed.iloc[350:] = changed.iloc[350:] * 4.0
    again = vf.rolling_vol_forecasts(changed, window=300, refit=20)
    np.testing.assert_allclose(base["har"].iloc[:51], again["har"].iloc[:51])
    np.testing.assert_allclose(base["garch"].iloc[:51], again["garch"].iloc[:51])
    assert (base["har"] > 0).all() and (base["garch"] > 0).all()
    # First HAR forecast equals a direct fit on the first window.
    beta = vf.har_fit(rv[:300])
    assert base["har"].iloc[0] == pytest.approx(max(vf.har_predict(rv[:300], beta), rv[:300].min()))
