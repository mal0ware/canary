"""GARCH(1,1) of Bollerslev (1986) and two extensions, fitted by maximum likelihood.

    eps_t = sigma_t z_t
    sigma_t^2 = omega + (alpha + gamma 1[eps_{t-1} < 0]) eps_{t-1}^2 + beta sigma_{t-1}^2

``gamma = 0`` is the plain GARCH(1,1). A non-zero ``gamma`` is the GJR model of
Glosten, Jagannathan and Runkle (1993), in which a negative return raises the
next variance by more than a positive one. The shock ``z_t`` is standard normal,
or a Student t with ``nu`` degrees of freedom scaled to unit variance
(Bollerslev 1987).

The recursion is linear in past squared residuals, so the whole variance path
is one call to :func:`scipy.signal.lfilter`. The first variance is set to the
sample variance of the residuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

FloatArray = npt.NDArray[np.float64]

_LOG_2PI = float(np.log(2.0 * np.pi))
_PERSISTENCE_CAP = 0.9999
_NU_BOUNDS = (2.05, 200.0)


@dataclass(frozen=True)
class GarchParams:
    """Parameters. ``gamma = 0`` and ``nu = inf`` give the normal GARCH(1,1)."""

    omega: float
    alpha: float
    beta: float
    gamma: float = 0.0
    nu: float = math.inf

    @property
    def persistence(self) -> float:
        """alpha + beta + gamma / 2: the shock is symmetric, so half the days are negative."""
        return self.alpha + self.beta + 0.5 * self.gamma

    @property
    def unconditional_variance(self) -> float:
        """omega / (1 - persistence), defined when the persistence is below one."""
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
    drive = params.alpha * e2
    if params.gamma != 0.0:
        drive = drive + params.gamma * e2 * (eps < 0.0)
    # y_t = drive_{t-1} + beta y_{t-1}, y_0 = 0
    y = lfilter([0.0, 1.0], [1.0, -params.beta], drive)
    powers = params.beta ** np.arange(n)
    if params.beta < 1.0:
        const = params.omega * (1.0 - powers) / (1.0 - params.beta)
    else:
        const = params.omega * np.arange(n, dtype=np.float64)
    out: FloatArray = const + powers * s0 + y
    return out


def next_variance(last_eps: float, last_sig2: float, params: GarchParams) -> float:
    """One step of the recursion from the last residual and its variance."""
    news = params.alpha + (params.gamma if last_eps < 0.0 else 0.0)
    return params.omega + news * last_eps**2 + params.beta * last_sig2


def forecast_variance(eps: FloatArray, params: GarchParams, initial: float | None = None) -> float:
    """One-step-ahead variance sigma_T^2 given residuals eps_0 .. eps_{T-1}."""
    sig2 = variance_path(eps, params, initial)
    return next_variance(float(eps[-1]), float(sig2[-1]), params)


def loglik(eps: FloatArray, params: GarchParams, initial: float | None = None) -> float:
    """Log likelihood of the residuals.

    Normal shocks: -0.5 sum[log(2 pi) + log sigma_t^2 + eps_t^2 / sigma_t^2].
    Student t shocks with unit variance and nu > 2 degrees of freedom:
    sum[lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 log(pi (nu-2)) - 0.5 log sigma_t^2
    - (nu+1)/2 log(1 + eps_t^2 / (sigma_t^2 (nu-2)))].
    """
    sig2 = variance_path(eps, params, initial)
    if np.any(sig2 <= 0.0) or not np.all(np.isfinite(sig2)):
        return -float("inf")
    if math.isinf(params.nu):
        return float(-0.5 * np.sum(_LOG_2PI + np.log(sig2) + eps**2 / sig2))
    nu = params.nu
    if nu <= 2.0:
        return -float("inf")
    const = float(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)) - 0.5 * math.log(
        math.pi * (nu - 2.0)
    )
    terms = -0.5 * np.log(sig2) - 0.5 * (nu + 1.0) * np.log1p(eps**2 / (sig2 * (nu - 2.0)))
    return float(eps.size * const + np.sum(terms))


_START_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 0.90),
    (0.10, 0.85),
    (0.10, 0.70),
    (0.20, 0.70),
    (0.03, 0.95),
)


def fit(
    eps: FloatArray,
    start: GarchParams | None = None,
    asymmetric: bool = False,
    dist: str = "normal",
) -> GarchFit:
    """Fit by maximum likelihood to mean-zero residuals.

    ``asymmetric`` adds the GJR term gamma. ``dist`` is ``"normal"`` or ``"t"``.
    Constraints: omega > 0, alpha >= 0, alpha + gamma >= 0, beta >= 0,
    alpha + beta + gamma / 2 < 1, and 2.05 <= nu <= 200. The data are rescaled
    to unit variance for the optimiser and omega is scaled back. ``start`` (in
    the units of ``eps``) is tried alongside a small grid of starting points;
    the best one seeds a single SLSQP run.
    """
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    if eps.ndim != 1 or eps.size < 10:
        raise ValueError("need a one-dimensional series of at least 10 residuals")
    scale = float(np.mean(eps**2))
    if scale <= 0.0:
        raise ValueError("residuals have zero variance")
    x = eps / np.sqrt(scale)
    use_t = dist == "t"

    def unpack(theta: FloatArray) -> GarchParams:
        k = 3
        gamma = 0.0
        if asymmetric:
            gamma = float(theta[k])
            k += 1
        nu = float(theta[k]) if use_t else math.inf
        return GarchParams(float(theta[0]), float(theta[1]), float(theta[2]), gamma, nu)

    def pack(p: GarchParams, omega_scale: float) -> FloatArray:
        out = [p.omega / omega_scale, p.alpha, p.beta]
        if asymmetric:
            out.append(p.gamma)
        if use_t:
            out.append(min(max(p.nu, _NU_BOUNDS[0]), _NU_BOUNDS[1]))
        return np.array(out)

    def nll(theta: FloatArray) -> float:
        value = -loglik(x, unpack(theta), 1.0)
        return value if np.isfinite(value) else 1e12

    starts = []
    for a, b in _START_GRID:
        gamma0 = 0.6 * a if asymmetric else 0.0
        starts.append(pack(GarchParams(1.0 - a - b, a - 0.5 * gamma0, b, gamma0, 8.0), 1.0))
    if start is not None and start.persistence < _PERSISTENCE_CAP:
        nu0 = start.nu if math.isfinite(start.nu) else 8.0
        seed = GarchParams(start.omega, start.alpha, start.beta, start.gamma, nu0)
        starts.insert(0, pack(seed, scale))
    theta0 = min(starts, key=nll)

    bounds = [(1e-8, 10.0), (0.0, 1.0), (0.0, 1.0)]
    constraints = [{"type": "ineq", "fun": lambda t: _PERSISTENCE_CAP - unpack(t).persistence}]
    if asymmetric:
        bounds.append((-1.0, 2.0))
        constraints.append({"type": "ineq", "fun": lambda t: t[1] + t[3]})
    if use_t:
        bounds.append(_NU_BOUNDS)

    res = minimize(  # type: ignore[call-overload]
        nll,
        theta0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    theta = res.x if res.fun <= nll(theta0) else theta0
    fitted = unpack(theta)
    params = GarchParams(fitted.omega * scale, fitted.alpha, fitted.beta, fitted.gamma, fitted.nu)
    return GarchFit(params=params, loglik=loglik(eps, params), converged=bool(res.success))


def simulate(
    n: int, params: GarchParams, rng: np.random.Generator, burn: int = 500
) -> tuple[FloatArray, FloatArray]:
    """Simulate (eps, sigma^2) of length ``n`` with the shock distribution of ``params``."""
    total = n + burn
    if math.isinf(params.nu):
        z = rng.standard_normal(total)
    else:
        z = rng.standard_t(params.nu, total) * math.sqrt((params.nu - 2.0) / params.nu)
    eps = np.empty(total)
    sig2 = np.empty(total)
    s = params.unconditional_variance if params.persistence < 1.0 else params.omega
    for t in range(total):
        sig2[t] = s
        eps[t] = np.sqrt(s) * z[t]
        s = next_variance(float(eps[t]), s, params)
    return eps[burn:], sig2[burn:]
