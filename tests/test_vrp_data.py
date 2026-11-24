"""Variance risk premium units and the data parsers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canary import data, vrp


def test_realized_variance_units() -> None:
    # A constant 1% move every calendar day: 30 squared returns in 30 days,
    # annualised by 365 / 30, is a variance of 365 in percent squared.
    idx = pd.date_range("2020-01-01", periods=100)
    trailing, forward = vrp.realized_variance_calendar(pd.Series(np.full(100, 0.01), index=idx))
    assert trailing.iloc[-1] == pytest.approx(365.0)
    assert forward.iloc[0] == pytest.approx(365.0)
    # No value where the series does not reach across the 30 days.
    assert trailing.iloc[:30].isna().all() and trailing.iloc[30:].notna().all()
    assert forward.iloc[-30:].isna().all() and forward.iloc[:-30].notna().all()


def test_realized_variance_counts_trading_days_in_the_calendar_span() -> None:
    # Business days only. The span is 30 calendar days whatever the number of
    # trading days inside it, which is what VIX covers.
    idx = pd.bdate_range("2020-01-01", periods=120)
    r = pd.Series(np.random.default_rng(1).standard_normal(120) * 0.01, index=idx)
    trailing, forward = vrp.realized_variance_calendar(r)
    day = idx[60]
    back = r[(idx > day - pd.Timedelta(days=30)) & (idx <= day)]
    ahead = r[(idx > day) & (idx <= day + pd.Timedelta(days=30))]
    assert len(back) in (20, 21, 22) and len(ahead) in (20, 21, 22)
    assert trailing[day] == pytest.approx(float((back**2).sum()) * 365 / 30 * 1e4)
    assert forward[day] == pytest.approx(float((ahead**2).sum()) * 365 / 30 * 1e4)


def test_vrp_is_zero_when_vix_equals_realized_volatility() -> None:
    idx = pd.date_range("2020-01-01", periods=200)
    r = pd.Series(np.tile([0.01, -0.01], 100), index=idx)
    vix = pd.Series(np.sqrt(365.0), index=idx)
    table = vrp.vrp_table(vix, r)
    assert len(table) > 3
    assert np.allclose(table["vrp_trailing"], 0.0)
    assert np.allclose(table["vrp_forward"], 0.0)


def test_vrp_table_alignment() -> None:
    idx = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(120) * 0.01, index=idx)
    vix = pd.Series(20.0, index=idx)
    table = vrp.vrp_table(vix, r)
    day = table.index[1]
    ahead = r[(idx > day) & (idx <= day + pd.Timedelta(days=30))]
    assert table.loc[day, "rv_forward"] == pytest.approx(float((ahead**2).sum()) * 365 / 30 * 1e4)
    assert table.loc[day, "iv"] == 400.0
    # one row per month, each the last trading day of its month
    assert table.index.to_period("M").is_unique
    assert day == idx[idx.to_period("M") == day.to_period("M")][-1]


def test_newey_west_t_statistic() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(400) + 0.2
    iid = x.mean() / (x.std(ddof=0) / np.sqrt(x.size))
    assert vrp.newey_west_t(x, 0) == pytest.approx(iid)
    # A persistent series: the iid standard error is too small.
    e = rng.standard_normal(5000)
    y = np.empty(5000)
    y[0] = e[0]
    for t in range(1, 5000):
        y[t] = 0.7 * y[t - 1] + e[t]
    y += 0.3
    assert abs(vrp.newey_west_t(y, 12)) < 0.6 * abs(vrp.newey_west_t(y, 0))


def test_newey_west_size_on_a_persistent_series() -> None:
    # AR(1) with coefficient 0.5 and mean zero, 440 points like the monthly
    # sample. The iid t-test rejects far too often. Newey-West is closer to 5%.
    rng = np.random.default_rng(3)
    reps, n, lag = 1500, 440, 5
    e = rng.standard_normal((reps, n + 50))
    y = np.zeros_like(e)
    for t in range(1, n + 50):
        y[:, t] = 0.5 * y[:, t - 1] + e[:, t]
    y = y[:, 50:]
    iid = np.mean([abs(vrp.newey_west_t(row, 0)) > 1.96 for row in y])
    nw = np.mean([abs(vrp.newey_west_t(row, lag)) > 1.96 for row in y])
    assert iid > 0.20
    assert 0.04 < nw < 0.11


def test_vrp_summary() -> None:
    table = pd.DataFrame(
        {
            "iv": [4.0, 4.0, 4.0, 4.0],
            "rv_trailing": [3.0, 2.0, 1.0, 6.0],
            "rv_forward": [3.0, 3.0, 3.0, -1.0],
            "vrp_trailing": [1.0, 2.0, 3.0, -2.0],
            "vrp_forward": [1.0, 1.0, 1.0, 5.0],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]),
    )
    s = vrp.vrp_summary(table).set_index("version")
    assert s.loc["vrp_trailing", "mean"] == pytest.approx(1.0)
    assert s.loc["vrp_trailing", "mean_iv"] - s.loc["vrp_trailing", "mean_rv"] == pytest.approx(1.0)
    assert s.loc["vrp_trailing", "share_positive"] == pytest.approx(0.75)
    assert s.loc["vrp_trailing", "min_month"] == "2020-04"
    assert s.loc["vrp_trailing", "nw_lag"] == 1
    assert np.isfinite(s.loc["vrp_trailing", "t_stat_newey_west"])


FRENCH_SAMPLE = """This file was created by using the 202609 CRSP database.
The Tbill return is the simple daily rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01
19260706,   -1.17,    0.29,   -0.38,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_daily() -> None:
    frame = data.parse_french_daily(FRENCH_SAMPLE)
    assert len(frame) == 3
    assert frame.index[0] == pd.Timestamp("1926-07-01")
    assert frame["mkt_rf"].iloc[2] == pytest.approx(-0.0117)
    assert frame["rf"].iloc[0] == pytest.approx(0.0001)


def test_parse_vix() -> None:
    text = (
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "01/03/1990,18.19,18.19,18.19,18.19\n"
        "01/02/1990,17.24,17.24,17.24,17.24\n"
    )
    s = data.parse_vix(text)
    assert s.index[0] == pd.Timestamp("1990-01-02")
    assert s.iloc[0] == 17.24


def test_parse_fred_drops_missing() -> None:
    text = "observation_date,SP500\n2016-11-25,2213.35\n2016-11-26,\n2016-11-28,2201.72\n"
    s = data.parse_fred(text)
    assert len(s) == 2 and s.iloc[1] == 2201.72


def test_parse_binance_klines_milliseconds_and_microseconds() -> None:
    ms = b"1583020800000,8523.61,8575.80,8511.11,8575.73,373.0,1583021099999,1.0,1,1.0,1.0,0\n"
    us = b"1735689600000000,93576.0,93702.1,93489.0,93656.2,1.0,1735689899999999,1.0,1,1.0,1.0,0\n"
    a = data.parse_binance_klines(ms)
    b = data.parse_binance_klines(us)
    assert a.index[0] == pd.Timestamp("2020-03-01 00:00:00")
    assert a.iloc[0] == 8575.73
    assert b.index[0] == pd.Timestamp("2025-01-01 00:00:00")
