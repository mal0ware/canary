"""Coverage tests: closed forms, published acceptance regions, size and power."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats

from canary import backtest as bt


def test_kupiec_hand_value() -> None:
    # 1000 days, 20 breaches, stated rate 1%.
    n, x, p = 1000, 20, 0.01
    expected = -2 * (x * np.log(p) + (n - x) * np.log(1 - p)) + 2 * (
        x * np.log(x / n) + (n - x) * np.log(1 - x / n)
    )
    hits = np.zeros(n, dtype=int)
    hits[:x] = 1
    res = bt.kupiec(hits, p)
    assert res.lr == pytest.approx(expected, rel=1e-12)
    assert res.p_value == pytest.approx(stats.chi2.sf(expected, 1))


def test_kupiec_zero_when_rate_matches() -> None:
    hits = np.zeros(1000, dtype=int)
    hits[::100] = 1
    assert bt.kupiec(hits, 0.01).lr == pytest.approx(0.0, abs=1e-12)


def test_kupiec_no_breaches_is_finite() -> None:
    # 0 log 0 = 0: the statistic is -2 n log(1 - p).
    res = bt.kupiec(np.zeros(255, dtype=int), 0.01)
    assert res.lr == pytest.approx(-2 * 255 * np.log(0.99))


# Kupiec (1995), FEDS working paper 95-24, Table 2: "Maximum sample size for which
# the null hypothesis p = p* is rejected at the 5 percent level", by number of failures.
KUPIEC_TABLE_2 = {
    0.01: {1: 6, 2: 34, 3: 75, 4: 124},
    0.02: {1: 3, 2: 17, 3: 38, 4: 63},
    0.03: {2: 11, 3: 26, 4: 44},
    0.04: {2: 9, 3: 19, 4: 32},
    0.05: {3: 16, 4: 26},
}


def _lr_reference(n: int, x: int, p: float) -> float:
    """Kupiec's statistic from scipy's binomial log-pmf, sharing no code with canary.

    The binomial coefficient cancels in the ratio, so twice the difference of
    the log-pmf at the sample rate and at the stated rate is expression (6).
    """
    return float(2.0 * (stats.binom.logpmf(x, n, x / n) - stats.binom.logpmf(x, n, p)))


def _lr_code(n: int, x: int, p: float) -> float:
    return float(bt.kupiec_lr(n, x, p))


def _max_rejected_sample(x: int, p: float, lr: Callable[[int, int, float], float]) -> int:
    return max(n for n in range(x, 1000) if x / n > p and lr(n, x, p) > 3.841)


def test_kupiec_lr_matches_independent_reference() -> None:
    for p in (0.01, 0.025, 0.05):
        for n in (50, 125, 255, 1000, 25338):
            for x in (1, 2, 4, 7, 20, 49):
                assert _lr_code(n, x, p) == pytest.approx(
                    max(_lr_reference(n, x, p), 0.0), rel=1e-9, abs=1e-9
                )


def test_kupiec_table_2_of_the_working_paper() -> None:
    # The printed table is compared with the scipy reference, and the code with
    # the reference, so no number here is the code checked against itself.
    differs = {}
    for p, row in KUPIEC_TABLE_2.items():
        for x, printed in row.items():
            reference = _max_rejected_sample(x, p, _lr_reference)
            assert _max_rejected_sample(x, p, _lr_code) == reference, (p, x)
            if reference != printed:
                differs[(p, x)] = (printed, reference)
    # 14 of the 16 printed cells follow from expression (6) with the 3.841 cut-off.
    assert differs == {(0.01, 4): (124, 125), (0.03, 4): (44, 42)}
    # The two that do not: 4 failures in 125 days still reject at 1%, and 4 in
    # 44 days do not reject at 3%.
    assert _lr_reference(125, 4, 0.01) == pytest.approx(3.867, abs=5e-4)
    assert _lr_reference(44, 4, 0.03) == pytest.approx(3.68, abs=5e-3)


def test_kupiec_one_year_example() -> None:
    # Kupiec, p. 22: with "three or less failures" in about 250 days, a model whose
    # true rate is 2% "would escape detection only 26.23 percent of the time".
    assert stats.binom.cdf(3, 250, 0.02) == pytest.approx(0.2623, abs=2e-4)


@pytest.mark.parametrize(
    ("n", "p", "low", "high"),
    [
        # Non-rejection regions for the number of failures N at the 5% test level,
        # attributed to Kupiec (1995) and read here from a secondary source
        # (Nieppola 2009, Table 1). The journal article was not reachable.
        (255, 0.01, None, 6),  # N < 7
        (510, 0.01, 2, 10),  # 1 < N < 11
        (1000, 0.01, 5, 16),  # 4 < N < 17
        (255, 0.025, 3, 11),  # 2 < N < 12
        (510, 0.025, 7, 20),  # 6 < N < 21
        (1000, 0.025, 16, 35),  # 15 < N < 36
        (255, 0.05, 7, 20),  # 6 < N < 21
        (510, 0.05, 17, 35),  # 16 < N < 36
        (1000, 0.05, 38, 64),  # 37 < N < 65
    ],
)
def test_kupiec_published_acceptance_regions(n: int, p: float, low: int | None, high: int) -> None:
    lo, hi = bt.kupiec_acceptance(n, p)
    assert hi == high
    if low is not None:
        assert lo == low


def test_christoffersen_hand_value() -> None:
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    res = bt.christoffersen(hits, 0.1)
    assert (res.n00, res.n01, res.n10, res.n11) == (10, 3, 3, 3)
    pi01, pi11, pi = 3 / 13, 3 / 6, 6 / 19
    l_markov = 10 * np.log(1 - pi01) + 3 * np.log(pi01) + 3 * np.log(1 - pi11) + 3 * np.log(pi11)
    l_pooled = 13 * np.log(1 - pi) + 6 * np.log(pi)
    assert res.lr_ind == pytest.approx(-2 * (l_pooled - l_markov), rel=1e-12)
    assert res.pi01 == pytest.approx(pi01)
    assert res.pi11 == pytest.approx(pi11)


def test_conditional_coverage_is_sum() -> None:
    rng = np.random.default_rng(0)
    hits = (rng.random(5000) < 0.03).astype(int)
    res = bt.christoffersen(hits, 0.01)
    assert res.lr_cc == pytest.approx(res.lr_uc + res.lr_ind, rel=1e-10)
    # lr_uc is Kupiec's statistic on the n - 1 transitions
    assert res.lr_uc == pytest.approx(bt.kupiec(hits[1:], 0.01).lr, rel=1e-10)


def test_christoffersen_no_consecutive_breaches() -> None:
    hits = np.zeros(1000, dtype=int)
    hits[50::100] = 1
    res = bt.christoffersen(hits, 0.01)
    assert res.n11 == 0
    assert np.isfinite(res.lr_ind)
    assert res.p_ind > 0.5


def test_correct_model_passes_both_tests_at_nominal_rate() -> None:
    # Independent breaches at exactly the stated rate: each test rejects about 5% of the time.
    rng = np.random.default_rng(1)
    reps, n, p = 2000, 5000, 0.05
    rej_uc = rej_ind = rej_cc = 0
    for _ in range(reps):
        hits = (rng.random(n) < p).astype(int)
        rej_uc += bt.kupiec(hits, p).p_value < 0.05
        c = bt.christoffersen(hits, p)
        rej_ind += c.p_ind < 0.05
        rej_cc += c.p_cc < 0.05
    for rej in (rej_uc, rej_ind, rej_cc):
        assert 0.035 < rej / reps < 0.065


def test_clustered_model_fails_christoffersen_and_passes_kupiec() -> None:
    # Long-run breach rate 5%, but a breach is followed by another 40% of the time.
    rng = np.random.default_rng(2)
    rate, pi11, n = 0.05, 0.40, 20000
    hits = bt.simulate_markov_hits(n, bt.markov_pi01(rate, pi11), pi11, 1, rng)[:, 0]
    assert bt.kupiec(hits, rate).p_value > 0.05
    c = bt.christoffersen(hits, rate)
    assert c.p_ind < 1e-6
    assert c.pi11 == pytest.approx(pi11, abs=0.05)
    assert c.p_cc < 1e-6


def test_markov_simulation_has_stated_rate() -> None:
    rng = np.random.default_rng(3)
    hits = bt.simulate_markov_hits(2000, bt.markov_pi01(0.01, 0.2), 0.2, 500, rng)
    assert hits.mean() == pytest.approx(0.01, abs=0.001)


def test_kupiec_power_exact_matches_simulation() -> None:
    rng = np.random.default_rng(4)
    n, p0, p1 = 1000, 0.01, 0.02
    exact = bt.kupiec_power(n, p0, p1)
    x = rng.binomial(n, p1, size=200_000)
    sim = np.mean(bt.kupiec_lr(n, x, p0) > stats.chi2.ppf(0.95, 1))
    assert exact == pytest.approx(sim, abs=0.005)


def test_kupiec_power_grows_with_sample() -> None:
    powers = [bt.kupiec_power(n, 0.01, 0.02) for n in (250, 1000, 5000)]
    assert powers[0] < powers[1] < powers[2]
    assert powers[2] > 0.99


def test_kupiec_size_is_near_nominal_for_large_n() -> None:
    assert bt.kupiec_power(25000, 0.01, 0.01) == pytest.approx(0.05, abs=0.01)


def test_independence_power_is_ordered() -> None:
    rng = np.random.default_rng(5)
    weak, _ = bt.independence_power(1000, 0.01, 0.01, 400, rng)
    strong, _ = bt.independence_power(1000, 0.01, 0.30, 400, rng)
    assert strong > weak


def test_followed_within() -> None:
    hits = np.zeros(100, dtype=int)
    hits[[10, 12, 50]] = 1
    observed, expected = bt.followed_within(hits, days=5)
    assert observed == pytest.approx(1 / 3)
    assert expected == pytest.approx(1 - 0.97**5)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        bt.kupiec(np.array([0, 2, 1]), 0.01)
    with pytest.raises(ValueError):
        bt.kupiec(np.array([0, 1]), 1.5)


def test_vectorised_lr_ind_matches_scalar() -> None:
    rng = np.random.default_rng(6)
    hits = (rng.random(3000) < 0.05).astype(int)
    c = bt.christoffersen(hits, 0.05)
    lr = bt._lr_ind_from_counts(
        np.array([float(c.n00)]),
        np.array([float(c.n01)]),
        np.array([float(c.n10)]),
        np.array([float(c.n11)]),
    )
    assert lr[0] == pytest.approx(c.lr_ind, rel=1e-10)


def test_monte_carlo_p_values() -> None:
    rng = np.random.default_rng(7)
    # Independent breaches: the simulated p-value is not small.
    iid = (rng.random(4000) < 0.02).astype(int)
    p_lr, p_pairs = bt.independence_monte_carlo(iid, 500, rng)
    assert p_lr > 0.05 and p_pairs > 0.05
    # Clustered breaches: both are at the floor 1 / (reps + 1).
    clustered = bt.simulate_markov_hits(4000, bt.markov_pi01(0.02, 0.4), 0.4, 1, rng)[:, 0]
    p_lr, p_pairs = bt.independence_monte_carlo(clustered, 500, rng)
    assert p_lr == pytest.approx(1 / 501)
    assert p_pairs == pytest.approx(1 / 501)


def test_no_pairs_can_reject_in_a_long_sample() -> None:
    # 25,000 days at 1% expect about 2.5 pairs. A series with none has fewer than
    # chance, and the chi-squared p-value falls below 5% although nothing clusters.
    hits = np.zeros(25000, dtype=int)
    hits[50::100] = 1
    res = bt.christoffersen(hits, 0.01)
    assert res.n11 == 0 and res.p_ind < 0.05
    _, p_pairs = bt.independence_monte_carlo(hits, 300, np.random.default_rng(8))
    assert p_pairs == 1.0


def test_shortfall_sums_hand_value() -> None:
    ret = np.array([-0.03, 0.01, -0.05, 0.0])
    v = np.full(4, 0.02)
    e = np.full(4, 0.04)
    s, n = bt.shortfall_sums(ret, v, e)
    assert n[0] == 2
    assert s[0] == pytest.approx(-0.03 / 0.04 - 0.05 / 0.04)
    z1, z2 = bt.as_statistics(s, n, 4, 0.5)
    assert z1[0] == pytest.approx(-2.0 / 2 + 1.0)  # mean loss on breach days equals ES
    assert z2[0] == pytest.approx(-2.0 / (4 * 0.5) + 1.0)


def _normal_forecasts(n: int, alpha: float) -> tuple[bt.FloatArray, bt.FloatArray]:
    z = float(stats.norm.ppf(alpha))
    return np.full(n, -z), np.full(n, float(stats.norm.pdf(z)) / alpha)


def test_acerbi_szekely_statistics_have_mean_zero_for_a_correct_model() -> None:
    rng = np.random.default_rng(20)
    n, alpha = 1000, 0.025
    v, e = _normal_forecasts(n, alpha)
    s, cnt = bt.shortfall_sums(rng.standard_normal((n, 4000)), v, e)
    z1, z2 = bt.as_statistics(s, cnt, n, alpha)
    assert np.nanmean(z1) == pytest.approx(0.0, abs=0.004)
    assert np.mean(z2) == pytest.approx(0.0, abs=0.012)


def test_acerbi_szekely_size_under_the_null() -> None:
    # A correct model is rejected at the 5% level about 5% of the time.
    rng = np.random.default_rng(21)
    n, alpha, reps, sims = 500, 0.05, 800, 199
    v, e = _normal_forecasts(n, alpha)
    rej1 = rej2 = 0
    for _ in range(reps):
        null_s, null_n = bt.shortfall_sums(rng.standard_normal((n, sims)), v, e)
        res = bt.acerbi_szekely(rng.standard_normal(n), v, e, alpha, null_s, null_n)
        rej1 += res.p_z1 <= 0.05
        rej2 += res.p_z2 <= 0.05
    assert 0.03 < rej1 / reps < 0.075
    assert 0.03 < rej2 / reps < 0.075


def test_acerbi_szekely_rejects_a_normal_model_on_fat_tailed_data() -> None:
    rng = np.random.default_rng(22)
    n, alpha, sims = 5000, 0.01, 499
    v, e = _normal_forecasts(n, alpha)
    null_s, null_n = bt.shortfall_sums(rng.standard_normal((n, sims)), v, e)
    fat = rng.standard_t(3, n) / np.sqrt(3.0)  # unit variance
    res = bt.acerbi_szekely(fat, v, e, alpha, null_s, null_n)
    assert res.z1 < 0 and res.z2 < 0
    assert res.p_z1 == pytest.approx(1 / 500)
    assert res.p_z2 == pytest.approx(1 / 500)
    # Forecasts that overstate the risk are not rejected: the test is one-sided.
    res = bt.acerbi_szekely(0.5 * rng.standard_normal(n), v, e, alpha, null_s, null_n)
    assert res.p_z2 > 0.9


def test_fz0_hand_value() -> None:
    ret = np.array([-0.05, 0.01])
    v = np.array([0.02, 0.02])
    e = np.array([0.04, 0.04])
    loss = bt.fz0_loss(ret, v, e, 0.1)
    assert loss[0] == pytest.approx(0.03 / (0.1 * 0.04) + 0.5 + np.log(0.04) - 1.0)
    assert loss[1] == pytest.approx(0.5 + np.log(0.04) - 1.0)
    with pytest.raises(ValueError):
        bt.fz0_loss(ret, v, np.array([0.04, 0.0]), 0.1)


def test_fz0_expected_loss_at_the_truth_is_log_es() -> None:
    # E[1(y < -VaR)(-VaR - y)] = alpha (ES - VaR), so the expected loss is log(ES).
    rng = np.random.default_rng(23)
    alpha = 0.05
    y = rng.standard_normal(2_000_000)
    v, e = _normal_forecasts(1, alpha)
    loss = bt.fz0_loss(y, np.full(y.size, v[0]), np.full(y.size, e[0]), alpha)
    assert loss.mean() == pytest.approx(np.log(e[0]), abs=0.003)


def test_fz0_is_minimised_at_the_true_var_and_es() -> None:
    rng = np.random.default_rng(24)
    alpha = 0.05
    y = rng.standard_normal(400_000)
    v0, e0 = (float(a[0]) for a in _normal_forecasts(1, alpha))
    ones = np.ones(y.size)
    best = bt.fz0_loss(y, v0 * ones, e0 * ones, alpha).mean()
    for dv, de in ((0.8, 1.0), (1.2, 1.0), (1.0, 0.85), (1.0, 1.2), (1.15, 1.15), (0.9, 0.9)):
        assert bt.fz0_loss(y, dv * v0 * ones, de * e0 * ones, alpha).mean() > best
