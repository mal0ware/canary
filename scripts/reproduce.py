"""Rebuild every table and figure.

    uv run python scripts/fetch_data.py     # once: downloads into data/
    uv run python scripts/reproduce.py      # writes results/
    uv run python scripts/reproduce.py --fast

``--fast`` needs no data. It runs the whole pipeline on a small seeded
synthetic data set and writes to results_fast/ (gitignored). It checks that
the code runs end to end; its numbers are not findings.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd

from canary import data, pipeline, plots
from canary import volforecast as vf

ROOT = Path(__file__).resolve().parents[1]

# The published run stops at these dates, so later downloads give the same sample.
MARKET_END = "2026-09-30"
BTC_END = "2026-10-31"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(data_dir: Path) -> tuple[pipeline.Inputs, pd.DataFrame]:
    """Load the downloaded files and describe them."""
    market = data.load_market_returns(data_dir)[:MARKET_END]
    vix = data.load_vix(data_dir)[:MARKET_END]
    sp500 = data.load_sp500(data_dir)[:MARKET_END]
    bars = data.load_btc_5m(data_dir)[:BTC_END]
    btc_daily = vf.daily_realized_variance(bars, data.BARS_PER_DAY_5M)

    digest = hashlib.sha256()
    for path in sorted((data_dir / data.BTC_DIR).glob("*.zip")):
        digest.update(sha256(path).encode())
    calendar = pd.date_range(btc_daily.index[0], btc_daily.index[-1])
    manifest = pd.DataFrame(
        [
            {
                "file": data.FRENCH_FILE,
                "sha256": sha256(data_dir / data.FRENCH_FILE),
                "rows_used": len(market),
                "first": str(market.index[0])[:10],
                "last": str(market.index[-1])[:10],
                "note": "",
            },
            {
                "file": data.VIX_FILE,
                "sha256": sha256(data_dir / data.VIX_FILE),
                "rows_used": len(vix),
                "first": str(vix.index[0])[:10],
                "last": str(vix.index[-1])[:10],
                "note": "",
            },
            {
                "file": data.SP500_FILE,
                "sha256": sha256(data_dir / data.SP500_FILE),
                "rows_used": len(sp500),
                "first": str(sp500.index[0])[:10],
                "last": str(sp500.index[-1])[:10],
                "note": "",
            },
            {
                "file": data.BTC_DIR,
                "sha256": digest.hexdigest(),
                "rows_used": len(btc_daily),
                "first": str(btc_daily.index[0])[:10],
                "last": str(btc_daily.index[-1])[:10],
                "note": (
                    f"{len(bars)} five-minute bars; "
                    f"{len(calendar) - len(btc_daily)} days dropped for missing bars; "
                    f"{int((btc_daily['n'] < data.BARS_PER_DAY_5M).sum())} kept days rescaled; "
                    "sha256 is of the archive hashes in name order"
                ),
            },
        ]
    )
    inputs = pipeline.Inputs(
        market=market, btc_daily=btc_daily[["rv", "ret"]], vix=vix, sp500=sp500
    )
    return inputs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="synthetic smoke run, no data needed")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    start = time.time()
    if args.fast:
        out_dir = args.out_dir or ROOT / "results_fast"
        inputs = pipeline.synthetic_inputs()
        config = pipeline.FAST
        manifest = None
    else:
        out_dir = args.out_dir or ROOT / "results"
        if not (args.data_dir / data.FRENCH_FILE).exists():
            raise SystemExit("data/ is empty: run `uv run python scripts/fetch_data.py` first")
        inputs, manifest = load_inputs(args.data_dir)
        config = pipeline.Config()

    res = pipeline.run(inputs, config)
    pipeline.write(res, out_dir)
    if manifest is not None:
        manifest.to_csv(out_dir / "data_manifest.csv", index=False, lineterminator="\n")
    plots.write_figures(res, out_dir)
    print(f"wrote {out_dir} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
