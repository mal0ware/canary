"""Coverage tests: closed forms, published acceptance regions, size and power."""

from __future__ import annotations

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
# Two printed cells do not follow from Kupiec's own statistic (his expression 6)
# with the 3.841 cut-off he states. The statistic gives the values on the right.
KUPIEC_TABLE_2_DIFFERS = {(0.01, 4): 125, (0.03, 4): 42}


def _max_rejected_sample(x: int, p: float) -> int:
    return max(n for n in range(x, 1000) if x / n > p and float(bt.kupiec_lr(n, x, p)) > 3.841)


def test_kupiec_table_2_of_the_working_paper() -> None:
    matched = 0
    for p, row in KUPIEC_TABLE_2.items():
        for x, printed in row.items():
            computed = _max_rejected_sample(x, p)
            if (p, x) in KUPIEC_TABLE_2_DIFFERS:
                assert computed == KUPIEC_TABLE_2_DIFFERS[(p, x)]
                assert abs(computed - printed) <= 2
            else:
                assert computed == printed, (p, x)
                matched += 1
    assert matched == 14


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
