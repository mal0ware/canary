"""Figures for ``results/``. One chart per claim."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from canary import var
from canary import volforecast as vf
from canary.pipeline import Results

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MODEL_COLORS = {
    "hs": "#2a78d6",
    "normal": "#eb6834",
    "garch_normal": "#1baf7a",
    "fhs": "#eda100",
}
BLUE_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _save(fig: Figure, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=150, metadata={"Software": None})
    plt.close(fig)


def fig_breaches(res: Results, out: Path, alpha: float = 0.01) -> None:
    """Headline: breach rate, and breach rate on the day after a breach, per model."""
    table = res.tables["var_coverage"]
    t = table[(table["dataset"] == "us_market") & (table["alpha"] == alpha)].set_index("model")
    models = list(reversed(var.MODELS))
    labels = [var.MODEL_LABELS[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True, layout="constrained")
    panels = (
        ("breach_rate", "Days the limit was breached (%)"),
        ("pi11", "Breached again the next day (%)"),
    )
    for ax, (col, title) in zip(axes, panels, strict=True):
        values = [
            100.0 * float(t[col].to_numpy(dtype=np.float64)[t.index.get_loc(m)]) for m in models
        ]
        ax.barh(labels, values, color=colors, height=0.55)
        ax.axvline(100.0 * alpha, color=INK, linewidth=1.0, linestyle="--")
        for y, v in enumerate(values):
            ax.text(v, y, f"  {v:.2f}", va="center", ha="left", color=INK, fontsize=9)
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_title(title, fontsize=10, loc="left")
        _style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].text(
        100.0 * alpha, 3.45, " stated: 1%", color=INK_2, fontsize=8, va="bottom", ha="left"
    )
    days = int(t["days"].iloc[0])
    first = str(res.frames["var_us"].index[0])[:4]
    last = str(res.frames["var_us"].index[-1])[:4]
    fig.suptitle(
        f"1% one-day value at risk, US stock market, {first} to {last} ({days:,} days)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    _save(fig, out)


def fig_decades(res: Results, out: Path) -> None:
    """Breach rate by decade for each model."""
    t = res.tables["var_coverage_by_decade"]
    fig, ax = plt.subplots(figsize=(9.6, 3.8), layout="constrained")
    for model in var.MODELS:
        sub = t[t["model"] == model]
        ax.plot(
            sub["decade"],
            100.0 * sub["breach_rate"],
            color=MODEL_COLORS[model],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=var.MODEL_LABELS[model],
        )
    ax.axhline(1.0, color=INK, linewidth=1.0, linestyle="--")
    ax.set_ylabel("Days the 1% limit was breached (%)")
    ax.set_xlabel("Decade")
    ax.set_ylim(bottom=0)
    ax.set_title("Breach rate of the 1% limit by decade, US stock market", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncols=2, labelcolor=INK_2)
    _style(ax)
    _save(fig, out)


def fig_power(res: Results, out: Path) -> None:
    """Power of both tests at the 1% level, by sample size."""
    kup = res.tables["power_kupiec"]
    ind = res.tables["power_independence"]
    sizes = sorted(kup["days"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True, layout="constrained")
    for i, n in enumerate(sizes):
        color = BLUE_RAMP[min(i, len(BLUE_RAMP) - 1)]
        k = kup[kup["days"] == n]
        axes[0].plot(
            100.0 * k["true_rate"],
            100.0 * k["power"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"{n:,} days",
        )
        j = ind[ind["days"] == n]
        axes[1].plot(
            100.0 * j["pi11"],
            100.0 * j["power_independence"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"{n:,} days",
        )
    axes[0].set_xlabel("True breach rate (%)")
    axes[0].set_ylabel("Chance the test rejects at 5% (%)")
    axes[0].set_title("Kupiec: is the rate 1%?", fontsize=10, loc="left")
    axes[1].set_xlabel("True chance of a breach the day after a breach (%)")
    axes[1].set_title("Christoffersen: are breaches independent?", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK_2, title="Sample")
    for ax in axes:
        ax.set_ylim(0, 102)
        _style(ax)
    _save(fig, out)


def fig_vol(res: Results, out: Path) -> None:
    """Cumulative QLIKE difference, GARCH minus HAR, on realised variance."""
    frame = res.frames["vol"]
    har_col = next(c for c in frame.columns if c.startswith("har_"))
    rv = frame["rv"].to_numpy(dtype=np.float64)
    diff = vf.qlike(rv, frame["garch"].to_numpy(dtype=np.float64)) - vf.qlike(
        rv, frame[har_col].to_numpy(dtype=np.float64)
    )
    fig, ax = plt.subplots(figsize=(9.6, 3.6), layout="constrained")
    ax.plot(frame.index, np.cumsum(diff), color=MODEL_COLORS["hs"], linewidth=2.0)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_ylabel("Cumulative QLIKE, GARCH minus HAR")
    ax.set_title(
        "One-day variance forecasts for BTC: the line rises when HAR has the smaller loss",
        fontsize=11,
        loc="left",
    )
    _style(ax)
    _save(fig, out)


def fig_vrp(res: Results, out: Path) -> None:
    """Monthly variance risk premium, forward version."""
    table = res.frames["vrp"]
    fig, ax = plt.subplots(figsize=(9.6, 3.6), layout="constrained")
    x = pd.DatetimeIndex(table.index)
    y = table["vrp_forward"].to_numpy(dtype=np.float64)
    ax.bar(x, y, width=25, color=MODEL_COLORS["hs"], linewidth=0)
    ax.axhline(0.0, color=AXIS, linewidth=1.0)
    ax.set_yscale("symlog", linthresh=500)
    ax.set_ylabel("VIX² minus realised variance\n(annualised, percent², symlog)")
    ax.set_title(
        "Variance risk premium by month: implied variance minus what the next 21 days delivered",
        fontsize=11,
        loc="left",
    )
    _style(ax)
    _save(fig, out)


def write_figures(res: Results, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_breaches(res, out_dir / "fig1_breaches.png")
    fig_decades(res, out_dir / "fig2_breaches_by_decade.png")
    fig_power(res, out_dir / "fig3_power.png")
    fig_vol(res, out_dir / "fig4_vol_forecast.png")
    fig_vrp(res, out_dir / "fig5_vrp.png")
