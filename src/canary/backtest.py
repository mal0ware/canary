"""Coverage tests for value-at-risk breaches.

A breach series is a 0/1 array: 1 on a day the loss exceeded the forecast limit.

* :func:`kupiec` tests whether the breach rate equals the stated rate
  (Kupiec 1995; restated as LR_uc in Christoffersen 1998).
* :func:`christoffersen` tests whether a breach today changes the chance of a
  breach tomorrow (LR_ind), and both properties jointly (LR_cc).
* :func:`kupiec_power` gives the exact power of Kupiec's test from the binomial
  distribution. :func:`independence_power` simulates the power of LR_ind
  against a first-order Markov alternative.

Formulas and their locations in the sources are listed in ``PAPER.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.special import xlogy

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


@dataclass(frozen=True)
class KupiecResult:
    """Result of the unconditional coverage test."""

    n: int
    breaches: int
    rate: float
    lr: float
    p_value: float


@dataclass(frozen=True)
class ChristoffersenResult:
    """Result of the independence and conditional coverage tests.

    ``n00, n01, n10, n11`` count transitions from yesterday's state to today's.
    ``pi01`` is the breach rate after a quiet day, ``pi11`` after a breach day.
    ``lr_uc`` is the unconditional statistic computed on the same ``n - 1``
    transitions, so that ``lr_cc == lr_uc + lr_ind`` exactly.
    """

    n00: int
    n01: int
    n10: int
    n11: int
    pi01: float
    pi11: float
    pi: float
    lr_uc: float
    lr_ind: float
    lr_cc: float
    p_ind: float
    p_cc: float


def _as_hits(hits: npt.ArrayLike) -> IntArray:
    h = np.asarray(hits)
    if h.ndim != 1:
        raise ValueError("hits must be one-dimensional")
    if h.size == 0:
        raise ValueError("hits is empty")
    if not np.isin(h, (0, 1)).all():
        raise ValueError("hits must contain only 0 and 1")
    return h.astype(np.int_)


def _bernoulli_loglik(successes: float, failures: float, p: float) -> float:
    """log of p^successes (1-p)^failures, with 0 log 0 = 0."""
    return float(xlogy(successes, p) + xlogy(failures, 1.0 - p))


def kupiec_lr(n: int, x: int | IntArray, p: float) -> FloatArray:
    """Kupiec's likelihood ratio for ``x`` breaches in ``n`` days at stated rate ``p``.

    LR = -2 log[(1-p)^(n-x) p^x] + 2 log[(1-x/n)^(n-x) (x/n)^x].
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    xa = np.asarray(x, dtype=np.float64)
    phat = xa / n
    null = xlogy(xa, p) + xlogy(n - xa, 1.0 - p)
    alt = xlogy(xa, phat) + xlogy(n - xa, 1.0 - phat)
    out: FloatArray = np.maximum(-2.0 * (null - alt), 0.0)
    return out


def kupiec(hits: npt.ArrayLike, p: float) -> KupiecResult:
    """Unconditional coverage test. Under the null the statistic is chi-squared(1)."""
    h = _as_hits(hits)
    n = int(h.size)
    x = int(h.sum())
    lr = float(kupiec_lr(n, x, p))
    return KupiecResult(n=n, breaches=x, rate=x / n, lr=lr, p_value=float(stats.chi2.sf(lr, 1)))


def christoffersen(hits: npt.ArrayLike, p: float) -> ChristoffersenResult:
    """Independence and conditional coverage tests of Christoffersen (1998).

    The likelihoods condition on the first observation, so they use the
    ``n - 1`` transitions. LR_ind is chi-squared(1) and LR_cc chi-squared(2)
    under their nulls.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    h = _as_hits(hits)
    if h.size < 2:
        raise ValueError("need at least two observations")
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else float("nan")
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else float("nan")
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = _bernoulli_loglik(n01 + n11, n00 + n10, p)
    ll_pooled = _bernoulli_loglik(n01 + n11, n00 + n10, pi)
    ll_markov = 0.0
    if n00 + n01 > 0:
        ll_markov += _bernoulli_loglik(n01, n00, pi01)
    if n10 + n11 > 0:
        ll_markov += _bernoulli_loglik(n11, n10, pi11)

    lr_uc = max(-2.0 * (ll_null - ll_pooled), 0.0)
    lr_ind = max(-2.0 * (ll_pooled - ll_markov), 0.0)
    lr_cc = max(-2.0 * (ll_null - ll_markov), 0.0)
    return ChristoffersenResult(
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi01=pi01,
        pi11=pi11,
        pi=pi,
        lr_uc=lr_uc,
        lr_ind=lr_ind,
        lr_cc=lr_cc,
        p_ind=float(stats.chi2.sf(lr_ind, 1)),
        p_cc=float(stats.chi2.sf(lr_cc, 2)),
    )


def kupiec_power(n: int, p_null: float, p_true: float, size: float = 0.05) -> float:
    """Exact probability that Kupiec's test rejects when the true breach rate is ``p_true``.

    The breach count is binomial(n, p_true); the test rejects when the statistic
    exceeds the chi-squared(1) critical value. With ``p_true == p_null`` this is
    the true size of the test, which differs from ``size`` because counts are discrete.
    """
    crit = float(stats.chi2.ppf(1.0 - size, 1))
    x = np.arange(n + 1)
    reject = kupiec_lr(n, x, p_null) > crit
    pmf = stats.binom.pmf(x, n, p_true)
    return float(np.sum(pmf[reject]))


def kupiec_acceptance(n: int, p: float, size: float = 0.05) -> tuple[int, int]:
    """Smallest and largest breach counts that Kupiec's test does not reject."""
    crit = float(stats.chi2.ppf(1.0 - size, 1))
    x = np.arange(n + 1)
    ok = np.flatnonzero(kupiec_lr(n, x, p) <= crit)
    return int(ok[0]), int(ok[-1])


def simulate_markov_hits(
    n: int, pi01: float, pi11: float, reps: int, rng: np.random.Generator
) -> npt.NDArray[np.int8]:
    """Simulate ``reps`` breach series of length ``n`` from a two-state Markov chain.

    The first day is drawn from the stationary distribution, whose breach rate
    is pi01 / (1 + pi01 - pi11).
    """
    stationary = pi01 / (1.0 + pi01 - pi11)
    out = np.empty((n, reps), dtype=np.int8)
    state = rng.random(reps) < stationary
    out[0] = state
    for t in range(1, n):
        state = rng.random(reps) < np.where(state, pi11, pi01)
        out[t] = state
    return out


def markov_pi01(rate: float, pi11: float) -> float:
    """The pi01 that keeps the long-run breach rate at ``rate`` for a given ``pi11``."""
    return rate * (1.0 - pi11) / (1.0 - rate)


def independence_power(
    n: int,
    rate: float,
    pi11: float,
    reps: int,
    rng: np.random.Generator,
    size: float = 0.05,
) -> tuple[float, float]:
    """Simulated rejection rates of (LR_ind, Kupiec) under a Markov alternative.

    The chain has long-run breach rate ``rate`` and probability ``pi11`` of a
    breach the day after a breach. ``pi11 == rate`` is the independent case.
    """
    hits = simulate_markov_hits(n, markov_pi01(rate, pi11), pi11, reps, rng)
    crit = float(stats.chi2.ppf(1.0 - size, 1))
    rej_ind = 0
    rej_uc = 0
    for j in range(reps):
        col = hits[:, j]
        rej_ind += christoffersen(col, rate).lr_ind > crit
        rej_uc += float(kupiec_lr(n, int(col.sum()), rate)) > crit
    return rej_ind / reps, rej_uc / reps


def followed_within(hits: npt.ArrayLike, days: int = 5) -> tuple[float, float]:
    """Share of breaches followed by another breach within ``days`` trading days.

    Returns ``(observed, expected)``, where ``expected`` is 1 - (1 - rate)^days,
    the share for independent breaches at the observed rate. Breaches in the
    last ``days`` observations are left out because their follow-up is cut off.
    """
    h = _as_hits(hits)
    idx = np.flatnonzero(h[: h.size - days])
    if idx.size == 0:
        return float("nan"), float("nan")
    csum = np.concatenate(([0], np.cumsum(h)))
    later = csum[idx + days + 1] - csum[idx + 1]
    rate = float(h.mean())
    return float(np.mean(later > 0)), 1.0 - (1.0 - rate) ** days
