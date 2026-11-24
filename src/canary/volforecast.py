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
    A day with at least ``min_coverage`` of its ``bars_per_day`` returns has
    its sum of squares scaled up by bars_per_day / (returns present). A day
    with fewer has no realised variance.

    Returns a frame with one row for every calendar day after the first, with
    ``rv`` (realised variance, NaN on a low-coverage day), ``ret`` (log return
    from the previous day's last bar to this day's last bar, NaN only when
    either day has no bar at all) and ``n`` (returns present). Keeping the
    low-coverage days as rows is what lets later code tell a gap from a
    consecutive day.
    """
    close = close.sort_index()
    step = pd.Timedelta(days=1) / bars_per_day
    logp = pd.Series(np.log(close.to_numpy(dtype=np.float64)), index=close.index)
    r = logp.diff()
    adjacent = pd.Series(close.index, index=close.index).diff() == step
    r = r[adjacent.to_numpy()]
    bar_day = pd.DatetimeIndex(logp.index).normalize()
    calendar = pd.date_range(bar_day[0], bar_day[-1], freq="D")
    grouped = (r**2).groupby(pd.DatetimeIndex(r.index).normalize())
    count = grouped.count().reindex(calendar, fill_value=0)
    total = grouped.sum().reindex(calendar, fill_value=0.0)
    n = count.to_numpy(dtype=np.float64)
    enough = n >= min_coverage * bars_per_day
    rv = np.full(calendar.size, np.nan)
    rv[enough] = total.to_numpy(dtype=np.float64)[enough] * bars_per_day / n[enough]
    last = logp.groupby(bar_day).last().reindex(calendar)
    out = pd.DataFrame(
        {"rv": rv, "ret": last.diff().to_numpy(), "n": count.to_numpy()}, index=calendar
    )
    return out.iloc[1:]


def har_design(rv: FloatArray, lags: tuple[int, int, int] = (1, 5, 22)) -> FloatArray:
    """HAR regressors at each day t: [1, mean of rv over the last 1, 5 and 22 days].

    Row ``t`` uses rv[t], rv[t-1], ... only. Rows before the longest window is
    full are NaN. ``rv`` is one value per calendar position and may hold NaN
    for a day with no measurement: each mean is then taken over the days
    present in its window, and the row is NaN when rv[t] itself is missing.
    """
    n = rv.size
    x = np.full((n, 1 + len(lags)), np.nan)
    x[:, 0] = 1.0
    present = np.isfinite(rv)
    csum = np.concatenate(([0.0], np.cumsum(np.where(present, rv, 0.0))))
    ccnt = np.concatenate(([0.0], np.cumsum(present)))
    for j, k in enumerate(lags):
        cnt = ccnt[k:] - ccnt[:-k]
        with np.errstate(invalid="ignore", divide="ignore"):
            x[k - 1 :, j + 1] = np.where(cnt > 0, (csum[k:] - csum[:-k]) / cnt, np.nan)
    x[~present, 1:] = np.nan
    return x


def har_fit(rv: FloatArray, lags: tuple[int, int, int] = (1, 5, 22)) -> FloatArray:
    """OLS of rv[t+1] on the HAR regressors at t. Returns [c, b_day, b_week, b_month].

    Pairs with a missing regressor row or a missing target are left out.
    """
    x = har_design(rv, lags)
    start = max(lags) - 1
    xs, ys = x[start:-1], rv[start + 1 :]
    ok = np.isfinite(xs).all(axis=1) & np.isfinite(ys)
    beta, *_ = np.linalg.lstsq(xs[ok], ys[ok], rcond=None)
    return np.asarray(beta, dtype=np.float64)


def har_predict(rv: FloatArray, beta: FloatArray, lags: tuple[int, int, int] = (1, 5, 22)) -> float:
    """Forecast of the next day's rv from the end of ``rv``. NaN when the last day is missing."""
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


def bartlett_lag(n: int) -> int:
    """Truncation lag floor(4 (n/100)^(2/9)), a common rule of thumb."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


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


def rolling_har_forecasts(
    rv: FloatArray, window: int, lags: tuple[int, int, int] = (1, 5, 22)
) -> tuple[FloatArray, npt.NDArray[np.int_]]:
    """One-day-ahead HAR forecasts of realised variance on a rolling window.

    The forecast for day t is an OLS fit on days t-window .. t-1. A forecast
    below the smallest realised variance in the window is raised to that value,
    because QLIKE needs a positive forecast. Returns the forecasts and a 0/1
    array marking where the floor applied. ``rv`` may hold NaN for days with no
    measurement (see :func:`har_design`); the forecast for the day after such
    a day is NaN.
    """
    m = rv.size - window
    if m <= 0:
        raise ValueError("series is not longer than the window")
    out = np.empty(m)
    floored = np.zeros(m, dtype=np.int_)
    for i in range(m):
        win = rv[i : i + window]
        f = har_predict(win, har_fit(win, lags), lags)
        floor = float(np.nanmin(win))
        if f < floor:
            f = floor
            floored[i] = 1
        out[i] = f
    return out, floored


def rolling_garch_forecasts(ret: FloatArray, window: int, refit: int = 1) -> FloatArray:
    """One-day-ahead GARCH(1,1) variance forecasts on a rolling window.

    The forecast for day t uses the demeaned returns of days t-window .. t-1.
    Parameters are re-estimated every ``refit`` days.
    """
    m = ret.size - window
    if m <= 0:
        raise ValueError("series is not longer than the window")
    out = np.empty(m)
    params: garch.GarchParams | None = None
    for i in range(m):
        win = ret[i : i + window]
        eps = win - win.mean()
        if params is None or i % refit == 0:
            params = garch.fit(eps, start=params).params
        out[i] = garch.forecast_variance(eps, params)
    return out


def rolling_vol_forecasts(
    daily: pd.DataFrame,
    window: int = 1000,
    refit: int = 1,
    lags: tuple[int, int, int] = (1, 5, 22),
) -> pd.DataFrame:
    """HAR and GARCH forecasts side by side.

    ``daily`` has columns ``rv`` and ``ret``. Returns columns ``rv``, ``ret2``
    (squared return), ``har``, ``garch`` and ``har_floored``, indexed like
    ``daily[window:]``.
    """
    rv = daily["rv"].to_numpy(dtype=np.float64)
    ret = daily["ret"].to_numpy(dtype=np.float64)
    har, floored = rolling_har_forecasts(rv, window, lags)
    return pd.DataFrame(
        {
            "rv": rv[window:],
            "ret2": ret[window:] ** 2,
            "har": har,
            "garch": rolling_garch_forecasts(ret, window, refit),
            "har_floored": floored,
        },
        index=daily.index[window:],
    )
