"""GARCH(1,1) of Bollerslev (1986), fitted by Gaussian maximum likelihood.

    eps_t = sigma_t z_t,   sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2

The recursion is linear in past squared residuals, so the whole variance path
is one call to :func:`scipy.signal.lfilter`. The first variance is set to the
sample variance of the residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.signal import lfilter

FloatArray = npt.NDArray[np.float64]

_LOG_2PI = float(np.log(2.0 * np.pi))
_PERSISTENCE_CAP = 0.9999


@dataclass(frozen=True)
class GarchParams:
    """GARCH(1,1) parameters."""

    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """omega / (1 - alpha - beta), defined when alpha + beta < 1."""
        if self.persistence >= 1.0:
            return float("inf")
        return self.omega / (1.0 - self.persistence)


@dataclass(frozen=True)
class GarchFit:
    """A fitted model: parameters, log likelihood, and whether the optimiser converged."""

    params: GarchParams
    loglik: float
    converged: bool


def variance_path(eps: FloatArray, params: GarchParams, initial: float | None = None) -> FloatArray:
    """Conditional variances sigma_0^2 .. sigma_{T-1}^2 for residuals eps_0 .. eps_{T-1}.

    ``sigma_0^2`` is ``initial`` (default: the mean of eps^2). Each later value
    uses only earlier residuals.
    """
    n = eps.size
    s0 = float(np.mean(eps**2)) if initial is None else initial
    e2 = eps**2
    # y_t = alpha e2_{t-1} + beta y_{t-1}, y_0 = 0
    y = lfilter([0.0, params.alpha], [1.0, -params.beta], e2)
    powers = params.beta ** np.arange(n)
    if params.beta < 1.0:
        const = params.omega * (1.0 - powers) / (1.0 - params.beta)
    else:
        const = params.omega * np.arange(n, dtype=np.float64)
    out: FloatArray = const + powers * s0 + y
    return out


def forecast_variance(eps: FloatArray, params: GarchParams, initial: float | None = None) -> float:
    """One-step-ahead variance sigma_T^2 given residuals eps_0 .. eps_{T-1}."""
    sig2 = variance_path(eps, params, initial)
    return params.omega + params.alpha * float(eps[-1] ** 2) + params.beta * float(sig2[-1])


def loglik(eps: FloatArray, params: GarchParams, initial: float | None = None) -> float:
    """Gaussian log likelihood: -0.5 sum[log(2 pi) + log sigma_t^2 + eps_t^2 / sigma_t^2]."""
    sig2 = variance_path(eps, params, initial)
    if np.any(sig2 <= 0.0) or not np.all(np.isfinite(sig2)):
        return -float("inf")
    return float(-0.5 * np.sum(_LOG_2PI + np.log(sig2) + eps**2 / sig2))


_START_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 0.90),
    (0.10, 0.85),
    (0.10, 0.70),
    (0.20, 0.70),
    (0.03, 0.95),
)


def fit(eps: FloatArray, start: GarchParams | None = None) -> GarchFit:
    """Fit GARCH(1,1) to mean-zero residuals by maximum likelihood.

    Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1. The data
    are rescaled to unit variance for the optimiser and omega is scaled back.
    ``start`` (in the units of ``eps``) is tried alongside a small grid of
    starting points; the best one seeds a single SLSQP run.
    """
    if eps.ndim != 1 or eps.size < 10:
        raise ValueError("need a one-dimensional series of at least 10 residuals")
    scale = float(np.mean(eps**2))
    if scale <= 0.0:
        raise ValueError("residuals have zero variance")
    x = eps / np.sqrt(scale)

    def nll(theta: FloatArray) -> float:
        value = -loglik(x, GarchParams(float(theta[0]), float(theta[1]), float(theta[2])), 1.0)
        return value if np.isfinite(value) else 1e12

    starts = [np.array([1.0 - a - b, a, b]) for a, b in _START_GRID]
    if start is not None and start.persistence < _PERSISTENCE_CAP:
        starts.insert(0, np.array([start.omega / scale, start.alpha, start.beta]))
    theta0 = min(starts, key=nll)

    res = minimize(  # type: ignore[call-overload]
        nll,
        theta0,
        method="SLSQP",
        bounds=[(1e-8, 10.0), (0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda t: _PERSISTENCE_CAP - t[1] - t[2]}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    theta = res.x if res.fun <= nll(theta0) else theta0
    params = GarchParams(float(theta[0]) * scale, float(theta[1]), float(theta[2]))
    return GarchFit(params=params, loglik=loglik(eps, params), converged=bool(res.success))


def simulate(
    n: int, params: GarchParams, rng: np.random.Generator, burn: int = 500
) -> tuple[FloatArray, FloatArray]:
    """Simulate (eps, sigma^2) of length ``n`` with standard normal shocks."""
    total = n + burn
    z = rng.standard_normal(total)
    eps = np.empty(total)
    sig2 = np.empty(total)
    s = params.unconditional_variance if params.persistence < 1.0 else params.omega
    for t in range(total):
        sig2[t] = s
        eps[t] = np.sqrt(s) * z[t]
        s = params.omega + params.alpha * eps[t] ** 2 + params.beta * s
    return eps[burn:], sig2[burn:]
