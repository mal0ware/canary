"""Loaders for the files that ``scripts/fetch_data.py`` downloads into ``data/``."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRENCH_FILE = "F-F_Research_Data_Factors_daily_CSV.zip"
VIX_FILE = "VIX_History.csv"
SP500_FILE = "SP500.csv"
BTC_DIR = "btcusdt_5m"

BARS_PER_DAY_5M = 288


def parse_french_daily(text: str) -> pd.DataFrame:
    """Parse the Fama-French daily factors CSV. Values are returned as decimals.

    The file has a text header, a line of column names starting with a comma,
    rows keyed by YYYYMMDD, and a copyright footer.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.replace(" ", "").startswith(",Mkt-RF"))
    rows = []
    for line in lines[start + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or len(parts[0]) != 8 or not parts[0].isdigit():
            break
        rows.append(parts[:5])
    frame = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    frame.index = pd.to_datetime(frame.pop("date"), format="%Y%m%d")
    frame.index.name = "date"
    return frame.astype(np.float64) / 100.0


def load_market_returns(data_dir: Path) -> pd.Series:
    """Daily simple return of the US value-weighted market (Mkt-RF plus RF), as a decimal."""
    with zipfile.ZipFile(data_dir / FRENCH_FILE) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")
    frame = parse_french_daily(text)
    out = frame["mkt_rf"] + frame["rf"]
    out.name = "market"
    return out


def parse_vix(text: str) -> pd.Series:
    """Parse the Cboe VIX history CSV (DATE as MM/DD/YYYY, CLOSE in percent)."""
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [c.strip().upper() for c in frame.columns]
    out = pd.Series(
        frame["CLOSE"].to_numpy(dtype=np.float64),
        index=pd.to_datetime(frame["DATE"], format="%m/%d/%Y"),
        name="vix",
    )
    out.index.name = "date"
    return out.sort_index()


def load_vix(data_dir: Path) -> pd.Series:
    return parse_vix((data_dir / VIX_FILE).read_text())


def parse_fred(text: str) -> pd.Series:
    """Parse a FRED graph CSV with two columns: observation_date and the series."""
    frame = pd.read_csv(io.StringIO(text))
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    out = pd.Series(values.to_numpy(dtype=np.float64), index=pd.to_datetime(frame.iloc[:, 0]))
    out.index.name = "date"
    return out.dropna().sort_index()


def load_sp500(data_dir: Path) -> pd.Series:
    return parse_fred((data_dir / SP500_FILE).read_text()).rename("sp500")


def parse_binance_klines(raw: bytes) -> pd.Series:
    """Close prices from one Binance kline CSV, indexed by bar open time (UTC).

    Columns are open time, open, high, low, close, ...; there is no header in
    older files. Open time is in milliseconds, or microseconds in files from
    2025 on.
    """
    frame = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4])
    ts = pd.to_numeric(frame[0], errors="coerce")
    frame = frame[ts.notna()]
    ts = ts.dropna().astype(np.int64)
    ms = np.where(ts > 10**14, ts // 1000, ts)
    out = pd.Series(
        frame[4].to_numpy(dtype=np.float64),
        index=pd.to_datetime(ms, unit="ms"),
        name="close",
    )
    return out


def load_btc_5m(data_dir: Path) -> pd.Series:
    """All downloaded BTCUSDT five-minute closes, in time order."""
    parts = []
    for path in sorted((data_dir / BTC_DIR).glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            parts.append(parse_binance_klines(zf.read(zf.namelist()[0])))
    if not parts:
        raise FileNotFoundError(f"no kline files in {data_dir / BTC_DIR}")
    out = pd.concat(parts).sort_index()
    deduped: pd.Series = out[~out.index.duplicated()]
    return deduped
