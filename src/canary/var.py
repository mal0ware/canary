"""One-day value-at-risk and expected-shortfall forecasts, and a rolling backtest.

Both numbers are reported as positive losses in the units of the returns. A
forecast for day ``t`` uses only returns before ``t``. A breach is a day whose
return is below ``-VaR``.

Six models share one window of past returns:

``hs``           historical simulation: the empirical quantile of the window
``normal``       normal distribution with the window's mean and standard deviation
``garch_normal`` normal distribution with a GARCH(1,1) variance forecast
``garch_t``      Student t distribution with a GARCH(1,1) variance forecast, the
                 degrees of freedom estimated with the GARCH parameters
``fhs``          filtered historical simulation: the empirical quantile of the
                 GARCH-standardised residuals, scaled by the variance forecast
``fhs_gjr``      the same with a GJR-GARCH(1,1) filter, which lets a fall raise
                 the next variance by more than a rise
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from canary import garch

FloatArray = npt.NDArray[np.float64]

MODELS: tuple[str, ...] = ("hs", "normal", "garch_normal", "garch_t", "fhs", "fhs_gjr")
MODEL_LABELS: dict[str, str] = {
    "hs": "Historical simulation",
    "normal": "Normal",
    "garch_normal": "GARCH, normal",
    "garch_t": "GARCH, Student t",
    "fhs": "Filtered historical (GARCH)",
    "fhs_gjr": "Filtered historical (GJR-GARCH)",
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


def student_t_tail(alpha: float, nu: float) -> tuple[float, float]:
    """Quantile and tail mean of a Student t scaled to unit variance, nu > 2.

    With q the alpha-quantile of the ordinary t and f its density, the tail
    mean of the ordinary t is -(f(q) / alpha) (nu + q^2) / (nu - 1). Both
    numbers are then multiplied by sqrt((nu - 2) / nu).
    """
    if nu <= 2.0:
        raise ValueError("nu must exceed 2")
    q = float(stats.t.ppf(alpha, nu))
    tail = -float(stats.t.pdf(q, nu)) / alpha * (nu + q * q) / (nu - 1.0)
    c = math.sqrt((nu - 2.0) / nu)
    return c * q, c * tail


def hs_var_es(window: FloatArray, alpha: float) -> tuple[float, float]:
    """Historical-simulation VaR and ES from a window of returns."""
    q, tail = empirical_tail(window, alpha)
    return -q, -tail


def normal_var_es(mu: float, sigma: float, alpha: float) -> tuple[float, float]:
    """VaR and ES of a normal return with mean ``mu`` and standard deviation ``sigma``."""
    z, tail = normal_tail(alpha)
    return -(mu + sigma * z), -(mu + sigma * tail)


def student_t_var_es(mu: float, sigma: float, nu: float, alpha: float) -> tuple[float, float]:
    """VaR and ES of a Student t return with mean ``mu`` and standard deviation ``sigma``."""
    z, tail = student_t_tail(alpha, nu)
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


@dataclass
class NullSums:
    """Sums from returns simulated out of each model's own forecast distribution.

    For model ``m`` and tail probability ``a``, ``s[(m, a)]`` holds, for each
    simulated history, the sum over days of return / ES on simulated breach
    days, and ``n[(m, a)]`` the number of simulated breaches. They are the
    inputs of :func:`canary.backtest.acerbi_szekely`.
    """

    s: dict[tuple[str, float], FloatArray] = field(default_factory=dict)
    n: dict[tuple[str, float], FloatArray] = field(default_factory=dict)


def rolling_backtest(
    returns: pd.Series,
    config: BacktestConfig,
    null_sims: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, NullSums]:
    """Out-of-sample VaR and ES for every model, and optional simulation under each model.

    The frame is indexed like ``returns[window:]`` with the realised return in
    ``ret``, the GARCH volatility forecast in ``garch_sigma``, and columns
    ``var_<model>_<alpha>`` and ``es_<model>_<alpha>``.

    With ``null_sims > 0`` each day also draws that many returns from every
    model's forecast distribution for the day: the window itself for ``hs``,
    the standardised residuals for the filtered models, a normal or Student t
    otherwise. The forecasts do not depend on the draws. A simulated breach
    is a draw at or below the limit. That is the indicator Acerbi and Szekely
    give for distributions that are not continuous, in the case where alpha
    times the window length is a whole number.
    """
    r = returns.to_numpy(dtype=np.float64)
    w = config.window
    n = r.size
    if n <= w:
        raise ValueError("series is not longer than the window")
    m = n - w
    cols: dict[str, FloatArray] = {"ret": r[w:].copy(), "garch_sigma": np.empty(m)}
    null = NullSums()
    for a in config.alphas:
        for model in MODELS:
            cols[f"var_{model}_{a}"] = np.empty(m)
            cols[f"es_{model}_{a}"] = np.empty(m)
            if null_sims > 0:
                null.s[(model, a)] = np.zeros(null_sims)
                null.n[(model, a)] = np.zeros(null_sims)
    if null_sims > 0 and rng is None:
        raise ValueError("null_sims needs a random generator")

    p_n: garch.GarchParams | None = None
    p_t: garch.GarchParams | None = None
    p_g: garch.GarchParams | None = None
    t_tails: dict[float, tuple[float, float]] = {}
    for i in range(m):
        win = r[i : i + w]
        mu = float(win.mean())
        sd = float(win.std(ddof=1))
        eps = win - mu
        if p_n is None or p_t is None or p_g is None or i % config.refit == 0:
            p_n = garch.fit(eps, start=p_n).params
            p_t = garch.fit(eps, start=p_t, dist="t").params
            p_g = garch.fit(eps, start=p_g, asymmetric=True).params
            t_tails = {a: student_t_tail(a, p_t.nu) for a in config.alphas}
        last = float(eps[-1])
        sig2 = garch.variance_path(eps, p_n)
        sig_n = math.sqrt(garch.next_variance(last, float(sig2[-1]), p_n))
        z = eps / np.sqrt(sig2)
        sig2_t = garch.variance_path(eps, p_t)
        sig_t = math.sqrt(garch.next_variance(last, float(sig2_t[-1]), p_t))
        sig2_g = garch.variance_path(eps, p_g)
        sig_g = math.sqrt(garch.next_variance(last, float(sig2_g[-1]), p_g))
        z_g = eps / np.sqrt(sig2_g)
        cols["garch_sigma"][i] = sig_n

        today: dict[tuple[str, float], tuple[float, float]] = {}
        for a in config.alphas:
            today[("hs", a)] = hs_var_es(win, a)
            today[("normal", a)] = normal_var_es(mu, sd, a)
            today[("garch_normal", a)] = normal_var_es(mu, sig_n, a)
            tz, tt = t_tails[a]
            today[("garch_t", a)] = (-(mu + sig_t * tz), -(mu + sig_t * tt))
            today[("fhs", a)] = fhs_var_es(mu, sig_n, z, a)
            today[("fhs_gjr", a)] = fhs_var_es(mu, sig_g, z_g, a)
        for (model, a), (v, e) in today.items():
            cols[f"var_{model}_{a}"][i] = v
            cols[f"es_{model}_{a}"][i] = e

        if null_sims > 0 and rng is not None:
            pick = rng.integers(0, w, null_sims)
            zn = rng.standard_normal(null_sims)
            zt = rng.standard_t(p_t.nu, null_sims) * math.sqrt((p_t.nu - 2.0) / p_t.nu)
            draws = {
                "hs": win[pick],
                "normal": mu + sd * zn,
                "garch_normal": mu + sig_n * zn,
                "garch_t": mu + sig_t * zt,
                "fhs": mu + sig_n * z[pick],
                "fhs_gjr": mu + sig_g * z_g[pick],
            }
            for (model, a), (v, e) in today.items():
                x = draws[model]
                # At or below the limit: a draw from a window can equal the
                # limit itself, and counting it makes the simulated breach
                # rate k / n, the stated rate, with mean return / ES of -1.
                hit = x <= -v + 1e-12 * abs(v)
                null.s[(model, a)] += np.where(hit, x, 0.0) / e
                null.n[(model, a)] += hit
    return pd.DataFrame(cols, index=returns.index[w:]), null


def rolling_forecasts(returns: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """The forecast frame of :func:`rolling_backtest`, without simulation."""
    frame, _ = rolling_backtest(returns, config)
    return frame


def breaches(forecasts: pd.DataFrame, model: str, alpha: float) -> npt.NDArray[np.int_]:
    """0/1 series: 1 where the return fell below minus the VaR forecast."""
    ret = forecasts["ret"].to_numpy(dtype=np.float64)
    limit = forecasts[f"var_{model}_{alpha}"].to_numpy(dtype=np.float64)
    hit = ret < -limit
    return hit.astype(np.int_)


def shortfall_ratio(forecasts: pd.DataFrame, model: str, alpha: float) -> float:
    """Mean loss on breach days divided by the mean ES forecast on those days.

    A value of 1 means losses beyond the limit were as large as forecast; above
    1 means they were larger. A description, with no test behind it.
    """
    hit = breaches(forecasts, model, alpha).astype(bool)
    if not hit.any():
        return float("nan")
    loss = -forecasts["ret"].to_numpy(dtype=np.float64)[hit]
    es = forecasts[f"es_{model}_{alpha}"].to_numpy(dtype=np.float64)[hit]
    return float(loss.mean() / es.mean())
