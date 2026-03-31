"""
Data fetcher module - downloads NQ futures data via yfinance.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from rich.console import Console

console = Console()


def fetch_nq_data(
    symbol: str = "NQ=F",
    period: str = "60d",
    interval: str = "5m",
    retries: int = 3,
    fallback_symbol: str = "^NDX",
) -> pd.DataFrame:
    """
    Download NQ futures (or NDX index as fallback) 5-minute OHLCV data.

    Parameters
    ----------
    symbol      : Primary ticker symbol (default NQ=F)
    period      : History length accepted by yfinance (default 60d)
    interval    : Bar interval (default 5m)
    retries     : Number of download attempts before falling back
    fallback_symbol : Ticker to try if primary fails

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume] and
    a DatetimeIndex in US/Eastern timezone.
    """
    for attempt in range(1, retries + 1):
        try:
            console.print(
                f"[cyan]Downloading {symbol} data (attempt {attempt}/{retries})…[/cyan]"
            )
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 100:
                df = _clean_df(df)
                console.print(
                    f"[green]✓ Downloaded {len(df)} bars for {symbol}[/green]"
                )
                return df
        except Exception as exc:
            console.print(f"[yellow]Attempt {attempt} failed: {exc}[/yellow]")
            time.sleep(2)

    # Fallback
    console.print(
        f"[yellow]Primary symbol {symbol} failed – trying fallback {fallback_symbol}[/yellow]"
    )
    try:
        ticker = yf.Ticker(fallback_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is not None and not df.empty:
            df = _clean_df(df)
            console.print(
                f"[green]✓ Downloaded {len(df)} bars for {fallback_symbol} (fallback)[/green]"
            )
            return df
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download data for {symbol} and {fallback_symbol}: {exc}"
        ) from exc

    raise RuntimeError(f"No data returned for {symbol} or {fallback_symbol}")


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

    Returns a list of per-day DataFrames, each containing only regular
    trading hours (09:30 – 16:00 US/Eastern).
    """
    sessions = []
    for date, group in df.groupby(df.index.date):
        session = group.between_time("09:30", "16:00")
        if len(session) >= 10:  # skip days with too few bars
            sessions.append(session)
    return sessions


if __name__ == "__main__":
    data = fetch_nq_data()
    print(data.tail())
    sessions = get_trading_sessions(data)
    print(f"Number of trading sessions: {len(sessions)}")
