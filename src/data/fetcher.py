"""
Data fetcher module - loads local CSV or downloads NQ futures data via yfinance.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

from rich.console import Console

console = Console()

# Local CSV path (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_CSV = _PROJECT_ROOT / "qqq_5m.csv"


def fetch_nq_data(
    symbol: str = "NQ=F",
    period: str = "60d",
    interval: str = "5m",
    retries: int = 3,
    fallback_symbol: str = "^NDX",
) -> pd.DataFrame:
    """
    Download NQ futures (or NDX index as fallback) 5-minute OHLCV data.
    Falls back to local CSV if yfinance is unavailable.
    """
    # Try local CSV first if yfinance not available
    if not _HAS_YF:
        return _load_local_csv()

    for attempt in range(1, retries + 1):
        try:
            console.print(
                f"[cyan]Downloading {symbol} data (attempt {attempt}/{retries})...[/cyan]"
            )
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 100:
                df = _clean_df(df)
                console.print(
                    f"[green]Downloaded {len(df)} bars for {symbol}[/green]"
                )
                return df
        except Exception as exc:
            console.print(f"[yellow]Attempt {attempt} failed: {exc}[/yellow]")
            time.sleep(2)

    # Fallback to local CSV
    console.print(f"[yellow]yfinance failed, loading local CSV...[/yellow]")
    return _load_local_csv()


def _load_local_csv() -> pd.DataFrame:
    """Load the local qqq_5m.csv file."""
    if not _LOCAL_CSV.exists():
        raise RuntimeError(f"Local CSV not found: {_LOCAL_CSV}")

    console.print(f"[cyan]Loading local CSV: {_LOCAL_CSV.name}[/cyan]")

    # Handle yfinance multi-level header (row 0 = Price names, row 1 = Ticker)
    df = pd.read_csv(_LOCAL_CSV, header=[0, 1], index_col=0)

    # Flatten multi-level columns: take first level only
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop any rows where index is not a valid datetime
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df.index.notna()]

    df = _clean_df(df)
    console.print(f"[green]Loaded {len(df)} bars from local CSV[/green]")
    return df


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the dataframe returned by yfinance."""
    # Keep only OHLCV columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # Drop rows with NaN in OHLC
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Ensure the index is a proper DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Localise to US/Eastern if timezone-naive
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    df.sort_index(inplace=True)
    return df


def get_trading_sessions(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split a continuous intraday DataFrame into individual trading days.
    """
    sessions = []
    for date, group in df.groupby(df.index.date):
        session = group.between_time("09:30", "16:00")
        if len(session) >= 10:
            sessions.append(session)
    return sessions


if __name__ == "__main__":
    data = fetch_nq_data()
    print(data.tail())
    sessions = get_trading_sessions(data)
    print(f"Number of trading sessions: {len(sessions)}")
