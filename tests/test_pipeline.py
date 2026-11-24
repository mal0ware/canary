"""The whole pipeline on synthetic data: it runs, and it gives the same tables twice."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from canary import pipeline, plots, var

TINY = pipeline.Config(
    window=300,
    refit=100,
    sensitivity_windows=(200,),
    vol_window=250,
    vol_refit=50,
    power_reps=50,
    mc_reps=50,
    es_reps=50,
    es_size_reps=3,
    es_size_sims=20,
    power_sizes=(250,),
    power_rates=(0.01, 0.02),
    power_pi11=(0.01, 0.2),
)


@pytest.fixture(scope="module")
def inputs() -> pipeline.Inputs:
    return pipeline.synthetic_inputs(seed=3, days=900)


@pytest.fixture(scope="module")
def results(inputs: pipeline.Inputs) -> pipeline.Results:
    return pipeline.run(inputs, TINY)


def test_tables_are_complete(results: pipeline.Results) -> None:
    cov = results.tables["var_coverage"]
    assert len(cov) == 2 * 2 * len(var.MODELS)  # two data sets, two tail probabilities
    assert set(cov["model"]) == set(var.MODELS)
    assert (cov["breaches"] == (cov["breach_rate"] * cov["days"]).round()).all()
    assert len(results.tables["vol_forecast"]) == 4  # two lag sets, two proxies
    es = results.tables["es_backtest"]
    assert len(es) == len(cov)
    assert es["p_z2"].between(0, 1).all()
    assert es.loc[es["model"] == "fhs", "fz0_dm_stat"].isna().all()
    assert es.loc[es["model"] != "fhs", "fz0_dm_stat"].notna().all()
    size = results.tables["es_test_size"]
    assert len(size) == 4  # two shock distributions, two sample sizes
    assert size["mean_breach_rate"].between(0.003, 0.03).all()
    assert size["reject_z2_at_5pct"].between(0, 1).all()
    assert len(results.tables["vrp_summary"]) == 2  # synthetic data start after 2007
    assert len(results.tables["vrp_proxy_check"]) == 1


def test_window_sensitivity_uses_the_same_days(results: pipeline.Results) -> None:
    sens = results.tables["var_window_sensitivity"]
    assert sens["days"].nunique() == 1
    assert set(sens["window"]) == {200, 300}


def test_synthetic_proxy_check_is_exact(results: pipeline.Results) -> None:
    # The synthetic index level is built from the same returns as the market series.
    check = results.tables["vrp_proxy_check"].iloc[0]
    assert check["return_correlation"] == pytest.approx(1.0)
    assert check["variance_ratio_market_over_sp500"] == pytest.approx(1.0)


def test_two_runs_give_identical_files(
    inputs: pipeline.Inputs, results: pipeline.Results, tmp_path: Path
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    pipeline.write(results, first)
    pipeline.write(pipeline.run(pipeline.synthetic_inputs(seed=3, days=900), TINY), second)
    names = sorted(p.name for p in first.glob("*.csv"))
    assert names == sorted(p.name for p in second.glob("*.csv"))
    assert len(names) >= 8
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_figures_are_written(results: pipeline.Results, tmp_path: Path) -> None:
    plots.write_figures(results, tmp_path)
    for name in (
        "fig1_breaches.png",
        "fig2_breaches_by_decade.png",
        "fig3_power.png",
        "fig4_vol_forecast.png",
        "fig5_vrp.png",
    ):
        assert (tmp_path / name).stat().st_size > 5000


def test_written_tables_round_trip(results: pipeline.Results, tmp_path: Path) -> None:
    pipeline.write(results, tmp_path)
    cov = pd.read_csv(tmp_path / "var_coverage.csv")
    assert cov["p_uc"].between(0, 1).all()
    assert cov["p_ind"].between(0, 1).all()


def test_volatility_study_scores_only_days_with_a_measurement(
    inputs: pipeline.Inputs, results: pipeline.Results
) -> None:
    daily = inputs.btc_daily
    assert daily["rv"].isna().sum() == 4 and daily["ret"].notna().all()
    assert (daily.index == pd.date_range(daily.index[0], periods=len(daily))).all()
    row = results.tables["vol_forecast"].iloc[0]
    # Forecast days start at position 250. Days 400 and 650 have no realised
    # variance and days 401 and 651 have no HAR forecast.
    assert row["calendar_days"] == len(daily) - 250
    assert row["days_without_rv"] == 2 and row["days_without_har_forecast"] == 2
    assert row["days"] == len(daily) - 250 - 4
    assert results.tables["vol_forecast_by_year"]["days"].sum() == row["days"]
