"""The variance risk premium from VIX: implied variance minus realised variance.

VIX is an annualised volatility in percent for the next 30 calendar days, so
VIX^2 is an annualised variance in percent squared. Realised variance here
covers the same horizon: the sum of squared daily log returns on the trading
days inside a 30-calendar-day span, annualised by 365 / 30 and put in the same
units.

Two versions are computed at each month end:

``vrp_trailing``  VIX^2 minus realised variance over the 30 calendar days ending
                  that day (the definition in Bollerslev, Tauchen and Zhou 2009,
                  who use five-minute returns for the realised leg)
``vrp_forward``   VIX^2 minus realised variance over the following 30 calendar
                  days (what a seller of variance at VIX^2 went on to earn,
                  before costs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from canary import volforecast as vf

VIX_HORIZON_DAYS = 30


def realized_variance_calendar(
    log_returns: pd.Series, days: int = VIX_HORIZON_DAYS
) -> tuple[pd.Series, pd.Series]:
    """Annualised realised variance, in percent squared, over calendar-day spans.

    Returns ``(trailing, forward)``, indexed like ``log_returns``. For a day t,
    ``trailing`` sums the squared returns dated in (t - days, t] and
    ``forward`` those dated in (t, t + days]. A value is NaN when the series
    does not reach across the whole span.
    """
    idx = pd.DatetimeIndex(log_returns.index)
    stamps = idx.to_numpy()
    span = np.timedelta64(days, "D")
    csum = np.concatenate(([0.0], np.cumsum(log_returns.to_numpy(dtype=np.float64) ** 2)))
    here = np.arange(1, stamps.size + 1)
    lo = np.searchsorted(stamps, stamps - span, side="right")
    hi = np.searchsorted(stamps, stamps + span, side="right")
    scale = 365.0 / days * 1e4
    trailing = (csum[here] - csum[lo]) * scale
    forward = (csum[hi] - csum[here]) * scale
    trailing[stamps - span < stamps[0]] = np.nan
    forward[stamps + span > stamps[-1]] = np.nan
    return pd.Series(trailing, index=idx), pd.Series(forward, index=idx)


def vrp_table(vix: pd.Series, log_returns: pd.Series) -> pd.DataFrame:
    """Month-end implied variance, realised variance and both premia.

    ``vix`` is the VIX close in percent; ``log_returns`` are daily log returns
    of the index it is written on (or a proxy). Rows are the last common
    trading day of each month that has both realised legs.
    """
    frame = pd.DataFrame({"vix": vix, "ret": log_returns}).dropna()
    trailing, forward = realized_variance_calendar(frame["ret"])
    daily = pd.DataFrame(
        {
            "iv": frame["vix"] ** 2,
            "rv_trailing": trailing,
            "rv_forward": forward,
        }
    )
    daily["vrp_trailing"] = daily["iv"] - daily["rv_trailing"]
    daily["vrp_forward"] = daily["iv"] - daily["rv_forward"]
    idx = pd.DatetimeIndex(daily.index)
    month_end = daily.groupby([idx.year, idx.month]).tail(1)
    return month_end.dropna()


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """Mean of ``x`` over its Newey-West standard error with a Bartlett kernel."""
    lrv = vf.long_run_variance(np.asarray(x, dtype=np.float64), lag, kernel="bartlett")
    if lrv <= 0.0:
        return float("nan")
    return float(np.mean(x) / np.sqrt(lrv / x.size))


def vrp_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per version: mean implied and realised variance, then the premium's mean, median,
    Newey-West t-statistic and share of positive months. Units are annualised percent squared.

    The monthly premium is persistent, so the t-statistic uses a Newey-West
    standard error with the lag of :func:`canary.volforecast.bartlett_lag`.
    """
    rows = []
    for col in ("vrp_trailing", "vrp_forward"):
        x = table[col].to_numpy(dtype=np.float64)
        n = x.size
        lag = vf.bartlett_lag(n)
        dc = x - x.mean()
        rows.append(
            {
                "version": col,
                "months": n,
                "mean_iv": float(table["iv"].mean()),
                "mean_rv": float(table[col.replace("vrp", "rv")].mean()),
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "autocorrelation_lag1": float(dc[1:] @ dc[:-1] / (dc @ dc)),
                "nw_lag": lag,
                "t_stat_newey_west": newey_west_t(x, lag),
                "share_positive": float(np.mean(x > 0)),
                "min": float(x.min()),
                "min_month": str(table.index[int(np.argmin(x))])[:7],
            }
        )
    return pd.DataFrame(rows)
