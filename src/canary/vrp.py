"""The variance risk premium from VIX: implied variance minus realised variance.

VIX is an annualised volatility in percent for the next 30 calendar days, so
VIX^2 is an annualised variance in percent squared. Realised variance here is
the sum of squared daily log returns over 21 trading days, annualised by 12 and
put in the same units.

Two versions are computed at each month end:

``vrp_trailing``  VIX^2 minus realised variance over the 21 days ending that day
                  (the definition in Bollerslev, Tauchen and Zhou 2009, who use
                  five-minute returns for the realised leg)
``vrp_forward``   VIX^2 minus realised variance over the following 21 days
                  (what a seller of variance at VIX^2 went on to earn, before costs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21


def realized_variance_annualised(log_returns: pd.Series, days: int) -> pd.Series:
    """Trailing ``days``-day sum of squared returns, annualised, in percent squared."""
    scale = 252.0 / days * 1e4
    return (log_returns**2).rolling(days).sum() * scale


def vrp_table(vix: pd.Series, log_returns: pd.Series) -> pd.DataFrame:
    """Month-end implied variance, realised variance and both premia.

    ``vix`` is the VIX close in percent; ``log_returns`` are daily log returns
    of the index it is written on (or a proxy). Rows are the last common
    trading day of each month; the last month has no forward leg and is dropped.
    """
    d = TRADING_DAYS_PER_MONTH
    frame = pd.DataFrame({"vix": vix, "ret": log_returns}).dropna()
    trailing = realized_variance_annualised(frame["ret"], d)
    forward = trailing.shift(-d)
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


def vrp_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per version: mean implied and realised variance, then the premium's mean, median,
    t-statistic and share of positive months. Units are annualised percent squared."""
    rows = []
    for col in ("vrp_trailing", "vrp_forward"):
        x = table[col].to_numpy(dtype=np.float64)
        n = x.size
        rows.append(
            {
                "version": col,
                "months": n,
                "mean_iv": float(table["iv"].mean()),
                "mean_rv": float(table[col.replace("vrp", "rv")].mean()),
                "mean": float(x.mean()),
                "median": float(np.median(x)),
                "t_stat": float(x.mean() / (x.std(ddof=1) / np.sqrt(n))),
                "share_positive": float(np.mean(x > 0)),
                "min": float(x.min()),
                "min_month": str(table.index[int(np.argmin(x))])[:7],
            }
        )
    return pd.DataFrame(rows)
