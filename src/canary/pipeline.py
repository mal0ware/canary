"""The full study: every table in ``results/`` is built here.

``run`` takes the prepared inputs and a config and returns the tables as data
frames. ``write`` saves them as CSV. ``synthetic_inputs`` builds a small seeded
data set so the pipeline can be smoke-tested without downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from canary import backtest as bt
from canary import garch, var, vrp
from canary import volforecast as vf

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    """Every choice the study makes. The defaults are the published run."""

    window: int = 1000
    refit: int = 21
    alphas: tuple[float, ...] = (0.01, 0.05)
    sensitivity_windows: tuple[int, ...] = (500, 2000)
    vol_window: int = 1000
    vol_refit: int = 1
    har_lag_sets: tuple[tuple[int, int, int], ...] = ((1, 5, 22), (1, 7, 30))
    power_reps: int = 2000
    mc_reps: int = 2000
    power_sizes: tuple[int, ...] = (250, 1000)
    power_rates: tuple[float, ...] = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03)
    power_pi11: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.40)
    seed: int = 20261123


FAST = Config(
    window=500,
    refit=50,
    sensitivity_windows=(250,),
    vol_window=300,
    vol_refit=20,
    power_reps=200,
    mc_reps=200,
    power_sizes=(250, 1000),
)


@dataclass(frozen=True)
class Inputs:
    """Prepared data.

    ``market``     daily simple returns of the US market, decimal
    ``btc_daily``  frame with ``rv`` (realised variance) and ``ret`` (log return)
    ``vix``        VIX close in percent
    ``sp500``      S&P 500 index level, or None
    """

    market: pd.Series
    btc_daily: pd.DataFrame
    vix: pd.Series
    sp500: pd.Series | None = None


@dataclass
class Results:
    """Tables keyed by file stem, and the forecast frames the figures need."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


def coverage_rows(
    forecasts: pd.DataFrame,
    dataset: str,
    window: int,
    alphas: tuple[float, ...],
    mc_reps: int = 0,
    seed: int = 0,
) -> list[dict[str, object]]:
    """One row per model and tail probability: counts, both tests, clustering, shortfall.

    With ``mc_reps > 0`` the independence test also gets simulated p-values.
    """
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for alpha in alphas:
        for model in var.MODELS:
            hits = var.breaches(forecasts, model, alpha)
            k = bt.kupiec(hits, alpha)
            c = bt.christoffersen(hits, alpha)
            within, within_iid = bt.followed_within(hits, 5)
            p_mc, p_pairs = (
                bt.independence_monte_carlo(hits, mc_reps, rng)
                if mc_reps > 0
                else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "dataset": dataset,
                    "window": window,
                    "alpha": alpha,
                    "model": model,
                    "days": k.n,
                    "breaches": k.breaches,
                    "breach_rate": k.rate,
                    "lr_uc": k.lr,
                    "p_uc": k.p_value,
                    "n01": c.n01,
                    "n11": c.n11,
                    "pi01": c.pi01,
                    "pi11": c.pi11,
                    "lr_ind": c.lr_ind,
                    "p_ind": c.p_ind,
                    "p_ind_simulated": p_mc,
                    "p_pairs_one_sided": p_pairs,
                    "expected_n11_if_independent": k.breaches * k.rate,
                    "lr_cc": c.lr_cc,
                    "p_cc": c.p_cc,
                    "followed_within_5d": within,
                    "followed_within_5d_if_independent": within_iid,
                    "shortfall_ratio": var.shortfall_ratio(forecasts, model, alpha),
                }
            )
    return rows


def decade_rows(forecasts: pd.DataFrame, alpha: float) -> list[dict[str, object]]:
    """Coverage by calendar decade for one tail probability."""
    rows: list[dict[str, object]] = []
    decade = (pd.DatetimeIndex(forecasts.index).year // 10) * 10
    for dec in sorted(set(decade)):
        sub = forecasts[decade == dec]
        if len(sub) < 250:
            continue
        for model in var.MODELS:
            hits = var.breaches(sub, model, alpha)
            k = bt.kupiec(hits, alpha)
            c = bt.christoffersen(hits, alpha)
            rows.append(
                {
                    "decade": f"{dec}s",
                    "model": model,
                    "days": k.n,
                    "breaches": k.breaches,
                    "breach_rate": k.rate,
                    "p_uc": k.p_value,
                    "n11": c.n11,
                    "pi11": c.pi11,
                    "p_ind": c.p_ind,
                }
            )
    return rows


def power_tables(
    config: Config, sample_sizes: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact power of Kupiec's test and simulated power of the independence test, at 1%."""
    sizes = tuple(sorted(set(config.power_sizes) | set(sample_sizes)))
    kup = [
        {
            "days": n,
            "true_rate": rate,
            "power": bt.kupiec_power(n, 0.01, rate),
            "accept_low": bt.kupiec_acceptance(n, 0.01)[0],
            "accept_high": bt.kupiec_acceptance(n, 0.01)[1],
        }
        for n in sizes
        for rate in config.power_rates
    ]
    rng = np.random.default_rng(config.seed)
    ind = []
    for n in sizes:
        for pi11 in config.power_pi11:
            p_ind, p_uc = bt.independence_power(n, 0.01, pi11, config.power_reps, rng)
            ind.append(
                {
                    "days": n,
                    "pi11": pi11,
                    "power_independence": p_ind,
                    "rejection_rate_kupiec": p_uc,
                    "expected_breach_pairs": (n - 1) * 0.01 * pi11,
                    "reps": config.power_reps,
                }
            )
    return pd.DataFrame(kup), pd.DataFrame(ind)


def bartlett_lag(n: int) -> int:
    """Truncation lag floor(4 (n/100)^(2/9)), a common rule of thumb."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def vol_forecast_study(daily: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GARCH(1,1) against HAR on realised variance. Returns (table, forecast frame)."""
    rv = daily["rv"].to_numpy(dtype=np.float64)
    ret = daily["ret"].to_numpy(dtype=np.float64)
    w = config.vol_window
    frame = pd.DataFrame(
        {
            "rv": rv[w:],
            "ret2": ret[w:] ** 2,
            "garch": vf.rolling_garch_forecasts(ret, w, config.vol_refit),
        },
        index=daily.index[w:],
    )
    rows: list[dict[str, object]] = []
    for lags in config.har_lag_sets:
        name = "har_" + "_".join(str(k) for k in lags)
        har, floored = vf.rolling_har_forecasts(rv, w, lags)
        frame[name] = har
        for proxy in ("rv", "ret2"):
            target = frame[proxy].to_numpy(dtype=np.float64)
            ok = target > 0.0
            loss_g = vf.qlike(target[ok], frame["garch"].to_numpy(dtype=np.float64)[ok])
            loss_h = vf.qlike(target[ok], har[ok])
            n = int(ok.sum())
            dm0 = vf.diebold_mariano(loss_g, loss_h, lag=0)
            lag = bartlett_lag(n)
            dmb = vf.diebold_mariano(loss_g, loss_h, lag=lag, kernel="bartlett")
            rows.append(
                {
                    "har_lags": "-".join(str(k) for k in lags),
                    "proxy": proxy,
                    "days": n,
                    "qlike_garch": float(loss_g.mean()),
                    "qlike_har": float(loss_h.mean()),
                    "mean_diff_garch_minus_har": dm0.mean_diff,
                    "dm_stat_lag0": dm0.stat,
                    "dm_p_lag0": dm0.p_value,
                    "bartlett_lag": lag,
                    "dm_stat_bartlett": dmb.stat,
                    "dm_p_bartlett": dmb.p_value,
                    "mean_forecast_over_mean_proxy_garch": float(
                        frame["garch"].to_numpy()[ok].mean() / target[ok].mean()
                    ),
                    "mean_forecast_over_mean_proxy_har": float(har[ok].mean() / target[ok].mean()),
                    "har_forecasts_floored": int(floored.sum()),
                }
            )
    return pd.DataFrame(rows), frame


def vol_forecast_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE on realised variance by calendar year, for GARCH and the first HAR lag set."""
    har_col = next(c for c in frame.columns if c.startswith("har_"))
    rv = frame["rv"].to_numpy(dtype=np.float64)
    loss = pd.DataFrame(
        {
            "qlike_garch": vf.qlike(rv, frame["garch"].to_numpy(dtype=np.float64)),
            "qlike_har": vf.qlike(rv, frame[har_col].to_numpy(dtype=np.float64)),
        },
        index=frame.index,
    )
    year = pd.DatetimeIndex(loss.index).year
    out = loss.groupby(year).mean()
    out.insert(0, "days", loss.groupby(year).size())
    out["mean_diff_garch_minus_har"] = out["qlike_garch"] - out["qlike_har"]
    out.index.name = "year"
    return out.reset_index()


def vrp_study(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Variance risk premium. Returns (summary, monthly table, proxy check)."""
    log_ret = pd.Series(np.log1p(inputs.market.to_numpy()), index=inputs.market.index)
    table = vrp.vrp_table(inputs.vix, log_ret)
    parts = []
    # The second window is the sample of Bollerslev, Tauchen and Zhou (2009).
    for sub in (table, table[table.index <= pd.Timestamp("2007-12-31")]):
        if len(sub) < 2:
            continue
        part = vrp.vrp_summary(sub)
        part.insert(0, "returns", "market")
        part.insert(1, "first_month", str(sub.index[0])[:7])
        part.insert(2, "last_month", str(sub.index[-1])[:7])
        parts.append(part)
    summary = pd.concat(parts, ignore_index=True)
    check_rows: list[dict[str, object]] = []
    if inputs.sp500 is not None:
        sp = pd.Series(
            np.log(inputs.sp500.to_numpy(dtype=np.float64)), index=inputs.sp500.index
        ).diff()
        both = pd.DataFrame({"market": log_ret, "sp500": sp}).dropna()
        if len(both) > 250:
            t_sp = vrp.vrp_table(inputs.vix, both["sp500"])
            t_mk = vrp.vrp_table(inputs.vix, both["market"])
            common = t_sp.index.intersection(t_mk.index)
            check_rows.append(
                {
                    "first_day": str(both.index[0])[:10],
                    "last_day": str(both.index[-1])[:10],
                    "days": len(both),
                    "return_correlation": float(both["market"].corr(both["sp500"])),
                    "variance_ratio_market_over_sp500": float(
                        (both["market"] ** 2).sum() / (both["sp500"] ** 2).sum()
                    ),
                    "months": len(common),
                    "mean_vrp_forward_market": float(t_mk.loc[common, "vrp_forward"].mean()),
                    "mean_vrp_forward_sp500": float(t_sp.loc[common, "vrp_forward"].mean()),
                    "mean_vrp_trailing_market": float(t_mk.loc[common, "vrp_trailing"].mean()),
                    "mean_vrp_trailing_sp500": float(t_sp.loc[common, "vrp_trailing"].mean()),
                }
            )
    return summary, table, pd.DataFrame(check_rows)


def garch_rows(inputs: Inputs) -> pd.DataFrame:
    """Full-sample GARCH(1,1) estimates, for comparison with the usual magnitudes."""
    rows = []
    for name, series in (("us_market", inputs.market), ("btc", inputs.btc_daily["ret"])):
        x = series.to_numpy(dtype=np.float64)
        fit = garch.fit(x - x.mean())
        rows.append(
            {
                "dataset": name,
                "days": x.size,
                "omega": fit.params.omega,
                "alpha": fit.params.alpha,
                "beta": fit.params.beta,
                "persistence": fit.params.persistence,
                "annualised_vol_pct": float(
                    np.sqrt(fit.params.unconditional_variance * 252.0) * 100.0
                ),
                "loglik": fit.loglik,
                "converged": fit.converged,
            }
        )
    return pd.DataFrame(rows)


def run(inputs: Inputs, config: Config) -> Results:
    """Run every study."""
    res = Results()
    main_cfg = var.BacktestConfig(config.window, config.refit, config.alphas)

    us = var.rolling_forecasts(inputs.market, main_cfg)
    btc_simple = pd.Series(
        np.expm1(inputs.btc_daily["ret"].to_numpy(dtype=np.float64)), index=inputs.btc_daily.index
    )
    btc = var.rolling_forecasts(btc_simple, main_cfg)
    res.frames["var_us"] = us
    res.frames["var_btc"] = btc

    rows = coverage_rows(us, "us_market", config.window, config.alphas, config.mc_reps, config.seed)
    rows += coverage_rows(btc, "btc", config.window, config.alphas, config.mc_reps, config.seed + 1)
    res.tables["var_coverage"] = pd.DataFrame(rows)
    res.tables["var_coverage_by_decade"] = pd.DataFrame(decade_rows(us, 0.01))

    by_window = {config.window: us}
    for w in config.sensitivity_windows:
        by_window[w] = var.rolling_forecasts(
            inputs.market, var.BacktestConfig(w, config.refit, (0.01,))
        )
    # Score every window on the same days, so the rows are comparable.
    first = max(frame.index[0] for frame in by_window.values())
    sens: list[dict[str, object]] = []
    for w in sorted(by_window):
        frame = by_window[w]
        sens += coverage_rows(frame[frame.index >= first], "us_market", w, (0.01,))
    res.tables["var_window_sensitivity"] = pd.DataFrame(sens)

    kup, ind = power_tables(config, (len(us), len(btc)))
    res.tables["power_kupiec"] = kup
    res.tables["power_independence"] = ind

    vol_table, vol_frame = vol_forecast_study(inputs.btc_daily, config)
    res.tables["vol_forecast"] = vol_table
    res.tables["vol_forecast_by_year"] = vol_forecast_by_year(vol_frame)
    res.frames["vol"] = vol_frame

    summary, monthly, check = vrp_study(inputs)
    res.tables["vrp_summary"] = summary
    res.tables["vrp_proxy_check"] = check
    res.frames["vrp"] = monthly

    res.tables["garch_full_sample"] = garch_rows(inputs)
    return res


def write(res: Results, out_dir: Path) -> None:
    """Write every non-empty table as CSV with fixed float formatting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in res.tables.items():
        if len(table) > 0:
            table.to_csv(
                out_dir / f"{name}.csv", index=False, float_format="%.6g", lineterminator="\n"
            )


def synthetic_inputs(seed: int = 7, days: int = 2200) -> Inputs:
    """A seeded stand-in for the real data: GARCH returns with fat tails, intraday bars, a VIX.

    Used by ``reproduce.py --fast`` and by the tests. Nothing here is a finding.
    """
    rng = np.random.default_rng(seed)
    params = garch.GarchParams(omega=1.5e-6, alpha=0.09, beta=0.90)
    _, sig2 = garch.simulate(days, params, rng)
    shocks = rng.standard_t(5, size=days) / np.sqrt(5.0 / 3.0)
    idx = pd.bdate_range("2012-01-02", periods=days)
    market = pd.Series(np.sqrt(sig2) * shocks, index=idx, name="market")
    vix = pd.Series(
        np.sqrt(sig2 * 252.0) * 100.0 * 1.1 * np.exp(0.05 * rng.standard_normal(days)),
        index=idx,
        name="vix",
    )
    sp500 = pd.Series(100.0 * np.exp(np.cumsum(np.log1p(market.to_numpy()))), index=idx)

    btc_days = 700
    bars = 48
    log_var = np.empty(btc_days)
    level = np.log(4e-4)
    x = level
    for t in range(btc_days):
        x = level + 0.95 * (x - level) + 0.25 * rng.standard_normal()
        log_var[t] = x
    intraday = rng.standard_normal((btc_days, bars)) * np.sqrt(np.exp(log_var) / bars)[:, None]
    btc_idx = pd.date_range("2022-01-01", periods=btc_days)
    btc_daily = pd.DataFrame(
        {"rv": (intraday**2).sum(axis=1), "ret": intraday.sum(axis=1)}, index=btc_idx
    )
    return Inputs(market=market, btc_daily=btc_daily, vix=vix, sp500=sp500)
