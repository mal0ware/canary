"""Volatility forecasts and their comparison.

* realised variance from intraday returns
* the HAR regression of Corsi (2009), here on realised variance
* rolling one-day-ahead forecasts from HAR and from GARCH(1,1)
* the QLIKE loss and the Diebold-Mariano test of equal predictive accuracy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from canary import garch

FloatArray = npt.NDArray[np.float64]


def realized_variance(intraday_returns: FloatArray) -> float:
    """Sum of squared intraday returns."""
    return float(np.sum(intraday_returns**2))


def daily_realized_variance(
    close: pd.Series, bars_per_day: int, min_coverage: float = 0.95
) -> pd.DataFrame:
    """Daily realised variance and close-to-close return from evenly spaced intraday closes.

    ``close`` is indexed by bar open time (UTC). Returns are log differences of
    consecutive bars; a return is dropped when the two bars are not adjacent.
    A day is kept when it has at least ``min_coverage`` of its ``bars_per_day``
    returns; the sum of squares is scaled up by bars_per_day / (returns present).

    Returns a frame indexed by day with ``rv`` (realised variance), ``ret``
    (log return from the previous day's last bar to this day's last bar) and
    ``n`` (returns present).
    """
    close = close.sort_index()
    step = pd.Timedelta(days=1) / bars_per_day
    logp = pd.Series(np.log(close.to_numpy(dtype=np.float64)), index=close.index)
    r = logp.diff()
    adjacent = pd.Series(close.index, index=close.index).diff() == step
    r = r[adjacent.to_numpy()]
    day = pd.DatetimeIndex(r.index).normalize()
    grouped = (r**2).groupby(day)
    count = grouped.count()
    total = grouped.sum()
    rv = total.to_numpy(dtype=np.float64) * bars_per_day / count.to_numpy(dtype=np.float64)
    last = logp.groupby(pd.DatetimeIndex(logp.index).normalize()).last()
    out = pd.DataFrame(
        {"rv": rv, "ret": last.diff().reindex(total.index).to_numpy(), "n": count.to_numpy()},
        index=total.index,
    )
    out = out[out["n"] >= min_coverage * bars_per_day]
    return out.dropna()


def har_design(rv: FloatArray, lags: tuple[int, int, int] = (1, 5, 22)) -> FloatArray:
    """HAR regressors at each day t: [1, mean of rv over the last 1, 5 and 22 days].

    Row ``t`` uses rv[t], rv[t-1], ... only. Rows before the longest window is
    full are NaN.
    """
    n = rv.size
    x = np.full((n, 1 + len(lags)), np.nan)
    x[:, 0] = 1.0
    csum = np.concatenate(([0.0], np.cumsum(rv)))
    for j, k in enumerate(lags):
        x[k - 1 :, j + 1] = (csum[k:] - csum[:-k]) / k
    return x


def har_fit(rv: FloatArray, lags: tuple[int, int, int] = (1, 5, 22)) -> FloatArray:
    """OLS of rv[t+1] on the HAR regressors at t. Returns [c, b_day, b_week, b_month]."""
    x = har_design(rv, lags)
    start = max(lags) - 1
    beta, *_ = np.linalg.lstsq(x[start:-1], rv[start + 1 :], rcond=None)
    return np.asarray(beta, dtype=np.float64)


def har_predict(rv: FloatArray, beta: FloatArray, lags: tuple[int, int, int] = (1, 5, 22)) -> float:
    """Forecast of the next day's rv from the end of ``rv``."""
    x = har_design(rv[-max(lags) :], lags)[-1]
    return float(x @ beta)


def qlike(proxy: FloatArray, forecast: FloatArray) -> FloatArray:
    """QLIKE loss: proxy/forecast - log(proxy/forecast) - 1.

    Zero when the forecast equals the proxy, positive otherwise. It differs
    from log(forecast) + proxy/forecast by a term that does not depend on the
    forecast, so loss differences between two forecasts are the same.
    """
    ratio = proxy / forecast
    out: FloatArray = ratio - np.log(ratio) - 1.0
    return out


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test. ``mean_diff`` is the mean of loss_a - loss_b."""

    mean_diff: float
    stat: float
    p_value: float
    lag: int
    n: int


def long_run_variance(d: FloatArray, lag: int, kernel: str = "rectangular") -> float:
    """Long-run variance of ``d``: gamma_0 + 2 sum_{k=1..lag} w_k gamma_k.

    ``rectangular`` uses w_k = 1, the truncated estimator of Diebold and
    Mariano (1995). ``bartlett`` uses w_k = 1 - k/(lag+1), which cannot be negative.
    """
    n = d.size
    dc = d - d.mean()
    total = float(dc @ dc) / n
    for k in range(1, lag + 1):
        gamma = float(dc[k:] @ dc[:-k]) / n
        weight = 1.0 if kernel == "rectangular" else 1.0 - k / (lag + 1.0)
        total += 2.0 * weight * gamma
    return total


def diebold_mariano(
    loss_a: FloatArray, loss_b: FloatArray, lag: int = 0, kernel: str = "rectangular"
) -> DMResult:
    """Test of equal expected loss. Negative ``stat`` means forecast a has the lower loss.

    stat = mean(d) / sqrt(long-run variance of d / n), d = loss_a - loss_b,
    compared with the standard normal (two-sided). For a one-step forecast
    Diebold and Mariano use ``lag = 0``.
    """
    d = loss_a - loss_b
    n = d.size
    lrv = long_run_variance(d, lag, kernel)
    if lrv <= 0.0:
        return DMResult(float(d.mean()), float("nan"), float("nan"), lag, n)
    stat = float(d.mean()) / float(np.sqrt(lrv / n))
    return DMResult(float(d.mean()), stat, float(2.0 * stats.norm.sf(abs(stat))), lag, n)


def rolling_vol_forecasts(
    daily: pd.DataFrame,
    window: int = 1000,
    refit: int = 1,
    lags: tuple[int, int, int] = (1, 5, 22),
) -> pd.DataFrame:
    """One-day-ahead variance forecasts from HAR and GARCH(1,1) on a rolling window.

    ``daily`` has columns ``rv`` and ``ret``. The forecast for day t uses days
    t-window .. t-1. HAR is OLS on realised variance; a forecast below the
    smallest realised variance in the window is raised to that value, because
    QLIKE needs a positive forecast. GARCH is fitted to demeaned daily returns
    every ``refit`` days.

    Returns columns ``rv``, ``ret2`` (squared return), ``har``, ``garch`` and
    ``har_floored`` (1 where the floor applied).
    """
    rv = daily["rv"].to_numpy(dtype=np.float64)
    ret = daily["ret"].to_numpy(dtype=np.float64)
    n = rv.size
    if n <= window:
        raise ValueError("series is not longer than the window")
    m = n - window
    har = np.empty(m)
    gar = np.empty(m)
    floored = np.zeros(m, dtype=np.int_)
    params: garch.GarchParams | None = None
    for i in range(m):
        rv_win = rv[i : i + window]
        beta = har_fit(rv_win, lags)
        f = har_predict(rv_win, beta, lags)
        floor = float(rv_win.min())
        if f < floor:
            f = floor
            floored[i] = 1
        har[i] = f
        eps = ret[i : i + window] - ret[i : i + window].mean()
        if params is None or i % refit == 0:
            params = garch.fit(eps, start=params).params
        gar[i] = garch.forecast_variance(eps, params)
    return pd.DataFrame(
        {
            "rv": rv[window:],
            "ret2": ret[window:] ** 2,
            "har": har,
            "garch": gar,
            "har_floored": floored,
        },
        index=daily.index[window:],
    )
