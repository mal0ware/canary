"""Download the public data into data/ (gitignored). No key or account is needed.

    uv run python scripts/fetch_data.py

Sources:
  Kenneth French data library   daily market return, 1926 on
  Cboe                          daily VIX history, 1990 on
  FRED                          S&P 500 index level, last ten years (a check on the proxy)
  Binance public data archive   BTCUSDT five-minute bars, 2017-08 on

Files already present are skipped. Each Binance archive is checked against the
SHA-256 checksum published beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (research script)"

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
SP500_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
BINANCE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-{ym}.zip"
BTC_FIRST = (2017, 8)
BTC_LAST = (2026, 10)


def get(url: str, tries: int = 4) -> bytes:
    """GET with a few retries."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data: bytes = resp.read()
                return data
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == tries - 1:
                raise
            print(f"  retry {url}: {exc}", file=sys.stderr)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"fetch {dest.name}")
    dest.write_bytes(get(url))


def months(first: tuple[int, int], last: tuple[int, int]) -> list[str]:
    out = []
    y, m = first
    while (y, m) <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_btc(dest_dir: Path, last: tuple[int, int]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ym in months(BTC_FIRST, last):
        dest = dest_dir / f"BTCUSDT-5m-{ym}.zip"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = BINANCE_URL.format(ym=ym)
        print(f"fetch {dest.name}")
        blob = get(url)
        expected = get(url + ".CHECKSUM").decode().split()[0]
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {dest.name}")
        dest.write_bytes(blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    fetch(FRENCH_URL, data_dir / "F-F_Research_Data_Factors_daily_CSV.zip")
    fetch(VIX_URL, data_dir / "VIX_History.csv")
    fetch(SP500_URL, data_dir / "SP500.csv")
    fetch_btc(data_dir / "btcusdt_5m", BTC_LAST)
    (data_dir / "FETCHED.txt").write_text(f"fetched {date.today().isoformat()}\n")
    print("done")


if __name__ == "__main__":
    main()
