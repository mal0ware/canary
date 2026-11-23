"""One-day value-at-risk and expected-shortfall forecasts, and a rolling backtest.

Both numbers are reported as positive losses in the units of the returns. A
forecast for day ``t`` uses only returns before ``t``. A breach is a day whose
return is below ``-VaR``.

Four models share one window of past returns:

``hs``           historical simulation: the empirical quantile of the window
``normal``       normal distribution with the window's mean and standard deviation
``garch_normal`` normal distribution with a GARCH(1,1) variance forecast
``fhs``          filtered historical simulation: the empirical quantile of the
                 GARCH-standardised residuals, scaled by the variance forecast
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from canary import garch

FloatArray = npt.NDArray[np.float64]

MODELS: tuple[str, ...] = ("hs", "normal", "garch_normal", "fhs")
MODEL_LABELS: dict[str, str] = {
    "hs": "Historical simulation",
    "normal": "Normal",
    "garch_normal": "GARCH, normal",
    "fhs": "Filtered historical (GARCH)",
}


def empirical_tail(x: FloatArray, alpha: float) -> tuple[float, float]:
    """Lower-tail quantile and tail mean of a sample.

    The quantile is the order statistic x_(k) with k = ceil(alpha n), the
    smallest value whose empirical distribution function is at least alpha.
    The tail mean is the average of the k smallest values.
    """
    n = x.size
    k = max(int(np.ceil(alpha * n - 1e-12)), 1)
    part = np.partition(x, k - 1)[:k]
    return float(part.max()), float(part.mean())


def normal_tail(alpha: float) -> tuple[float, float]:
    """Quantile and tail mean of the standard normal: z_alpha and -phi(z_alpha)/alpha."""
    z = float(stats.norm.ppf(alpha))
    return z, -float(stats.norm.pdf(z)) / alpha


def hs_var_es(window: FloatArray, alpha: float) -> tuple[float, float]:
    """Historical-simulation VaR and ES from a window of returns."""
    q, tail = empirical_tail(window, alpha)
    return -q, -tail


def normal_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    """VaR and ES of a normal return with mean ``mu`` and standard deviation ``sigma``."""
    z, tail = normal_tail(alpha)
    return -(mu + sigma * z), -(mu + sigma * tail)


def fhs_var_es(
    mu: float, sigma_next: float, standardised: FloatArray, alpha: float
) -> tuple[float, float]:
    """Filtered-historical VaR and ES: empirical tail of z, scaled by the forecast volatility."""
    q, tail = empirical_tail(standardised, alpha)
    return -(mu + sigma_next * q), -(mu + sigma_next * tail)


@dataclass(frozen=True)
class BacktestConfig:
    """Settings of the rolling backtest.

    ``window``  number of past returns each forecast uses
    ``refit``   GARCH parameters are re-estimated every ``refit`` days; between
                refits the variance filter still runs over the current window
    ``alphas``  tail probabilities
    """

    window: int = 1000
    refit: int = 21
    alphas: tuple[float, ...] = (0.01, 0.05)


def rolling_forecasts(returns: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """Out-of-sample VaR and ES for every model and tail probability.

    Returns a frame indexed like ``returns[window:]`` with the realised return
    in ``ret``, the GARCH volatility forecast in ``garch_sigma``, and columns
    ``var_<model>_<alpha>`` and ``es_<model>_<alpha>``.
    """
    r = returns.to_numpy(dtype=np.float64)
    w = config.window
    n = r.size
    if n <= w:
        raise ValueError("series is not longer than the window")
    m = n - w
    cols: dict[str, FloatArray] = {"ret": r[w:].copy(), "garch_sigma": np.empty(m)}
    for a in config.alphas:
        for model in MODELS:
            cols[f"var_{model}_{a}"] = np.empty(m)
            cols[f"es_{model}_{a}"] = np.empty(m)

    params: garch.GarchParams | None = None
    for i in range(m):
        win = r[i : i + w]
        mu = float(win.mean())
        sd = float(win.std(ddof=1))
        eps = win - mu
        if params is None or i % config.refit == 0:
            params = garch.fit(eps, start=params).params
        sig2 = garch.variance_path(eps, params)
        sig_next = float(
            np.sqrt(params.omega + params.alpha * eps[-1] ** 2 + params.beta * sig2[-1])
        )
        z = eps / np.sqrt(sig2)
        cols["garch_sigma"][i] = sig_next
        for a in config.alphas:
            v, e = hs_var_es(win, a)
            cols[f"var_hs_{a}"][i], cols[f"es_hs_{a}"][i] = v, e
            v, e = normal_var_es(mu, sd, a)
            cols[f"var_normal_{a}"][i], cols[f"es_normal_{a}"][i] = v, e
            v, e = normal_var_es(mu, sig_next, a)
            cols[f"var_garch_normal_{a}"][i], cols[f"es_garch_normal_{a}"][i] = v, e
            v, e = fhs_var_es(mu, sig_next, z, a)
            cols[f"var_fhs_{a}"][i], cols[f"es_fhs_{a}"][i] = v, e
    return pd.DataFrame(cols, index=returns.index[w:])


def breaches(forecasts: pd.DataFrame, model: str, alpha: float) -> npt.NDArray[np.int_]:
    """0/1 series: 1 where the return fell below minus the VaR forecast."""
    ret = forecasts["ret"].to_numpy(dtype=np.float64)
    limit = forecasts[f"var_{model}_{alpha}"].to_numpy(dtype=np.float64)
    hit = ret < -limit
    return hit.astype(np.int_)


def shortfall_ratio(forecasts: pd.DataFrame, model: str, alpha: float) -> float:
    """Mean loss on breach days divided by the mean ES forecast on those days.

    A value of 1 means losses beyond the limit were as large as forecast; above
    1 means they were larger. Not a formal test.
    """
    hit = breaches(forecasts, model, alpha).astype(bool)
    if not hit.any():
        return float("nan")
    loss = -forecasts["ret"].to_numpy(dtype=np.float64)[hit]
    es = forecasts[f"es_{model}_{alpha}"].to_numpy(dtype=np.float64)[hit]
    return float(loss.mean() / es.mean())
