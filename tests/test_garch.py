"""GARCH(1,1): recursion, likelihood, estimation against a reference implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from arch import arch_model
from scipy import stats

from canary import garch

TRUE = garch.GarchParams(omega=0.02, alpha=0.08, beta=0.90)
FloatArray = npt.NDArray[np.float64]


def _naive_path(eps: FloatArray, p: garch.GarchParams, s0: float) -> FloatArray:
    out = np.empty(eps.size)
    out[0] = s0
    for t in range(1, eps.size):
        out[t] = p.omega + p.alpha * eps[t - 1] ** 2 + p.beta * out[t - 1]
    return out


def test_variance_path_matches_naive_loop() -> None:
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(500)
    np.testing.assert_allclose(
        garch.variance_path(eps, TRUE, 1.3), _naive_path(eps, TRUE, 1.3), rtol=1e-11
    )


def test_variance_path_default_start_is_sample_variance() -> None:
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(200)
    assert garch.variance_path(eps, TRUE)[0] == pytest.approx(np.mean(eps**2))


def test_forecast_is_next_step_of_recursion() -> None:
    rng = np.random.default_rng(2)
    eps = rng.standard_normal(300)
    longer = np.append(eps, 0.0)
    assert garch.forecast_variance(eps, TRUE, 1.0) == pytest.approx(
        _naive_path(longer, TRUE, 1.0)[-1]
    )


def test_unconditional_variance() -> None:
    assert TRUE.unconditional_variance == pytest.approx(0.02 / (1 - 0.98))
    eps, _ = garch.simulate(400_000, TRUE, np.random.default_rng(3))
    assert np.var(eps) == pytest.approx(TRUE.unconditional_variance, rel=0.08)


def test_constant_variance_limit() -> None:
    # alpha = beta = 0 is a constant variance omega; the likelihood is the normal one.
    rng = np.random.default_rng(4)
    eps = rng.standard_normal(100)
    p = garch.GarchParams(omega=1.7, alpha=0.0, beta=0.0)
    np.testing.assert_allclose(garch.variance_path(eps, p, 1.7), 1.7)
    expected = -0.5 * np.sum(np.log(2 * np.pi * 1.7) + eps**2 / 1.7)
    assert garch.loglik(eps, p, 1.7) == pytest.approx(expected)


def test_fit_recovers_simulated_parameters() -> None:
    eps, _ = garch.simulate(20_000, TRUE, np.random.default_rng(5))
    est = garch.fit(eps).params
    assert est.alpha == pytest.approx(TRUE.alpha, abs=0.015)
    assert est.beta == pytest.approx(TRUE.beta, abs=0.02)
    assert est.omega == pytest.approx(TRUE.omega, rel=0.3)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_fit_matches_arch_package(seed: int) -> None:
    eps, _ = garch.simulate(3000, TRUE, np.random.default_rng(seed))
    mine = garch.fit(eps)
    ref = arch_model(eps, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
        disp="off", backcast=float(np.mean(eps**2))
    )
    # arch applies the recursion to the backcast for the first variance; this
    # code uses the backcast itself. The likelihoods differ in the third decimal.
    assert mine.loglik == pytest.approx(float(ref.loglikelihood), abs=0.01)
    assert mine.params.omega == pytest.approx(float(ref.params["omega"]), rel=0.005)
    assert mine.params.alpha == pytest.approx(float(ref.params["alpha[1]"]), abs=5e-4)
    assert mine.params.beta == pytest.approx(float(ref.params["beta[1]"]), abs=5e-4)


def test_fit_is_scale_equivariant() -> None:
    eps, _ = garch.simulate(3000, TRUE, np.random.default_rng(6))
    a = garch.fit(eps).params
    b = garch.fit(eps * 0.01).params
    assert b.alpha == pytest.approx(a.alpha, abs=1e-4)
    assert b.beta == pytest.approx(a.beta, abs=1e-4)
    assert b.omega == pytest.approx(a.omega * 1e-4, rel=1e-3)


def test_fit_respects_constraints() -> None:
    rng = np.random.default_rng(7)
    est = garch.fit(rng.standard_normal(500)).params
    assert est.omega > 0 and est.alpha >= 0 and est.beta >= 0
    assert est.persistence < 1


GJR = garch.GarchParams(omega=0.02, alpha=0.03, beta=0.90, gamma=0.10)
STUDENT = garch.GarchParams(omega=0.02, alpha=0.08, beta=0.90, nu=6.0)


def test_gjr_variance_path_matches_naive_loop() -> None:
    rng = np.random.default_rng(20)
    eps = rng.standard_normal(400)
    naive = np.empty(eps.size)
    naive[0] = 1.1
    for t in range(1, eps.size):
        news = GJR.alpha + (GJR.gamma if eps[t - 1] < 0 else 0.0)
        naive[t] = GJR.omega + news * eps[t - 1] ** 2 + GJR.beta * naive[t - 1]
    np.testing.assert_allclose(garch.variance_path(eps, GJR, 1.1), naive, rtol=1e-11)
    # A fall raises the next variance by more than a rise of the same size.
    up = garch.next_variance(1.0, 1.0, GJR)
    down = garch.next_variance(-1.0, 1.0, GJR)
    assert down - up == pytest.approx(GJR.gamma)


def test_gjr_persistence_and_unconditional_variance() -> None:
    assert GJR.persistence == pytest.approx(0.03 + 0.90 + 0.05)
    eps, _ = garch.simulate(400_000, GJR, np.random.default_rng(21))
    assert np.var(eps) == pytest.approx(GJR.unconditional_variance, rel=0.08)


def test_student_t_loglik_matches_scipy() -> None:
    # A unit-variance t is scipy's t with scale sqrt((nu - 2) / nu).
    rng = np.random.default_rng(22)
    eps = rng.standard_normal(300)
    sig2 = garch.variance_path(eps, STUDENT, 1.2)
    scale = np.sqrt(sig2 * (STUDENT.nu - 2.0) / STUDENT.nu)
    expected = float(np.sum(stats.t.logpdf(eps, STUDENT.nu, scale=scale)))
    assert garch.loglik(eps, STUDENT, 1.2) == pytest.approx(expected, rel=1e-12)


def test_student_t_simulation_has_unit_variance_shocks() -> None:
    eps, sig2 = garch.simulate(300_000, STUDENT, np.random.default_rng(23))
    z = eps / np.sqrt(sig2)
    assert np.var(z) == pytest.approx(1.0, abs=0.03)
    assert stats.kurtosis(z) > 1.0  # the normal has zero


@pytest.mark.parametrize("seed", [30, 31])
def test_gjr_fit_matches_arch_package(seed: int) -> None:
    eps, _ = garch.simulate(4000, GJR, np.random.default_rng(seed))
    mine = garch.fit(eps, asymmetric=True)
    ref = arch_model(eps, mean="Zero", vol="GARCH", p=1, o=1, q=1, rescale=False).fit(
        disp="off", backcast=float(np.mean(eps**2))
    )
    assert mine.loglik == pytest.approx(float(ref.loglikelihood), abs=0.02)
    assert mine.params.alpha == pytest.approx(float(ref.params["alpha[1]"]), abs=2e-3)
    assert mine.params.gamma == pytest.approx(float(ref.params["gamma[1]"]), abs=2e-3)
    assert mine.params.beta == pytest.approx(float(ref.params["beta[1]"]), abs=2e-3)


@pytest.mark.parametrize("seed", [40, 41])
def test_student_t_fit_matches_arch_package(seed: int) -> None:
    eps, _ = garch.simulate(4000, STUDENT, np.random.default_rng(seed))
    mine = garch.fit(eps, dist="t")
    ref = arch_model(eps, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(
        disp="off", backcast=float(np.mean(eps**2))
    )
    assert mine.loglik == pytest.approx(float(ref.loglikelihood), abs=0.02)
    assert mine.params.alpha == pytest.approx(float(ref.params["alpha[1]"]), abs=2e-3)
    assert mine.params.beta == pytest.approx(float(ref.params["beta[1]"]), abs=2e-3)
    assert mine.params.nu == pytest.approx(float(ref.params["nu"]), rel=0.02)


def test_extended_fits_recover_simulated_parameters() -> None:
    eps, _ = garch.simulate(30_000, GJR, np.random.default_rng(50))
    est = garch.fit(eps, asymmetric=True).params
    assert est.gamma == pytest.approx(GJR.gamma, abs=0.02)
    assert est.alpha == pytest.approx(GJR.alpha, abs=0.015)
    eps, _ = garch.simulate(30_000, STUDENT, np.random.default_rng(51))
    est = garch.fit(eps, dist="t").params
    assert est.nu == pytest.approx(STUDENT.nu, rel=0.12)
    assert est.gamma == 0.0


def test_plain_fit_is_unchanged_by_the_extensions() -> None:
    eps, _ = garch.simulate(2000, TRUE, np.random.default_rng(52))
    est = garch.fit(eps).params
    assert est.gamma == 0.0 and np.isinf(est.nu)
    with pytest.raises(ValueError):
        garch.fit(eps, dist="cauchy")
