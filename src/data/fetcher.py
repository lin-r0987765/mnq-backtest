"""
Data fetcher module.

Primary use cases:
- load the local QQQ intraday CSV used by the repo
- optionally download a symbol from yfinance
- optionally merge a peer symbol so ICT SMT filters can use real peer highs/lows
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
from src.data.coverage import DEFAULT_DATA_TARGETS, describe_price_coverage

console = Console()

# Local CSV path (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_CSV = _PROJECT_ROOT / "qqq_5m.csv"
_ALPACA_COLUMN_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


def _normalise_timeframe_rule(timeframe: str) -> str:
    rule = str(timeframe).strip().lower()
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    return aliases.get(rule, rule)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV in either single-header or multi-header format."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception:
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df = _normalise_csv_ohlcv_columns(df)
    df = _coerce_csv_timestamp_index(df)
    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(set(df.columns)):
        df = pd.read_csv(csv_path)
        df = _normalise_csv_ohlcv_columns(df)
        df = _coerce_csv_timestamp_index(df)
        if not required_cols.issubset(set(df.columns)):
            df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = _normalise_csv_ohlcv_columns(df)
            df = _coerce_csv_timestamp_index(df)
        if not required_cols.issubset(set(df.columns)):
            raise RuntimeError(
                f"CSV missing OHLC columns after normalization: {csv_path}"
            )

    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df[df.index.notna()]
    return _clean_df(df)


def load_ohlcv_parquet(path: str | Path) -> pd.DataFrame:
    """Load an Alpaca-style parquet file into the repo's OHLCV format."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise RuntimeError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        valid_rows = timestamps.notna()
        df = df.loc[valid_rows].copy()
        df.index = timestamps.loc[valid_rows]
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    else:
        raise RuntimeError(
            f"Parquet must contain a timestamp column or DatetimeIndex: {parquet_path}"
        )

    rename_map = {
        column: mapped
        for column, mapped in _ALPACA_COLUMN_MAP.items()
        if column in df.columns
    }
    df = df.rename(columns=rename_map)
    return _clean_df(df)


def normalise_alpaca_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an Alpaca parquet frame into the normalized CSV schema."""
    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized.loc[:, ["timestamp", "close", "high", "low", "open", "volume"]]
    normalized.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    normalized["Datetime"] = normalized["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    normalized["Datetime"] = normalized["Datetime"].str.replace(
        r"([+-]\d{2})(\d{2})$",
        r"\1:\2",
        regex=True,
    )
    return normalized


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    *,
    label: str = "left",
    closed: str = "left",
) -> pd.DataFrame:
    """
    Resample OHLCV bars.

    The default `label="left", closed="left"` matches the repo's normalized
    Alpaca intraday reference files. Strategy code that needs explicit
    no-lookahead higher-timeframe bars should pass `label="right",
    closed="right"` and only consume the resampled values after the bar close.
    """
    if df.empty:
        return df.copy()

    rule = _normalise_timeframe_rule(timeframe)
    aggregations: dict[str, str] = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        aggregations["Volume"] = "sum"

    resampled = (
        df.loc[:, [column for column in aggregations if column in df.columns]]
        .resample(rule, label=label, closed=closed)
        .agg(aggregations)
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    if "Volume" in resampled.columns:
        resampled["Volume"] = pd.to_numeric(resampled["Volume"], errors="coerce").fillna(0.0)
    return _clean_df(resampled)


def merge_peer_columns(
    primary_df: pd.DataFrame,
    peer_df: pd.DataFrame,
    *,
    prefix: str = "Peer",
) -> pd.DataFrame:
    """
    Merge peer-symbol OHLC data onto the primary frame without forward filling.

    We keep this alignment strict so SMT logic cannot silently look ahead.
    """
    merged = primary_df.copy()
    peer_aligned = peer_df.reindex(primary_df.index)

    if "High" in peer_aligned.columns:
        merged[f"{prefix}High"] = peer_aligned["High"]
    if "Low" in peer_aligned.columns:
        merged[f"{prefix}Low"] = peer_aligned["Low"]
    if "Close" in peer_aligned.columns:
        merged[f"{prefix}Close"] = peer_aligned["Close"]

    return merged


def fetch_peer_data(
    *,
    peer_symbol: str | None = None,
    peer_csv: str | Path | None = None,
    period: str = "60d",
    interval: str = "5m",
    retries: int = 3,
) -> pd.DataFrame:
    """Load peer-symbol intraday data from CSV or yfinance."""
    if peer_csv:
        console.print(f"[cyan]Loading ICT peer CSV: {Path(peer_csv).name}[/cyan]")
        return load_ohlcv_csv(peer_csv)

    if not peer_symbol:
        raise RuntimeError("peer_symbol or peer_csv is required for peer data")

    if not _HAS_YF:
        raise RuntimeError("yfinance unavailable and no peer_csv was supplied")

    for attempt in range(1, retries + 1):
        try:
            console.print(
                f"[cyan]Downloading peer symbol {peer_symbol} (attempt {attempt}/{retries})...[/cyan]"
            )
            ticker = yf.Ticker(peer_symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 100:
                df = _clean_df(df)
                console.print(
                    f"[green]Downloaded {len(df)} peer bars for {peer_symbol}[/green]"
                )
                return df
        except Exception as exc:
            console.print(f"[yellow]Peer attempt {attempt} failed: {exc}[/yellow]")
            time.sleep(2)

    raise RuntimeError(f"Unable to load peer data for {peer_symbol}")


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
    # v13: 優先使用本地 QQQ CSV（所有參數都針對 QQQ 調優）
    if _LOCAL_CSV.exists():
        console.print("[cyan]Using local QQQ CSV (parameter-tuned dataset)...[/cyan]")
        return _load_local_csv()

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

    df = load_ohlcv_csv(_LOCAL_CSV)
    coverage = describe_price_coverage(
        _LOCAL_CSV,
        interval=DEFAULT_DATA_TARGETS["qqq_5m.csv"]["interval"],
        target_years=float(DEFAULT_DATA_TARGETS["qqq_5m.csv"]["target_years"]),
        recommended_provider=str(DEFAULT_DATA_TARGETS["qqq_5m.csv"]["recommended_provider"]),
        notes=str(DEFAULT_DATA_TARGETS["qqq_5m.csv"]["notes"]),
    )
    console.print(f"[green]Loaded {len(df)} bars from local CSV[/green]")
    if not coverage.meets_target:
        console.print(
            "[yellow]"
            f"QQQ 5m coverage only {coverage.span_years:.3f} years "
            f"(target {coverage.target_years:.1f}y). "
            f"Recommended provider: {coverage.recommended_provider}."
            "[/yellow]"
        )
    return df


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the dataframe returned by yfinance."""
    # Keep only OHLCV columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # Convert numeric columns to float (fixes issue with CSV string values)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

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


def _normalise_csv_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common CSV OHLCV header variants into the repo schema."""
    normalized = df.copy()
    rename_map: dict[str, str] = {}
    for column in normalized.columns:
        key = str(column).strip().lower()
        if key in _ALPACA_COLUMN_MAP:
            rename_map[column] = _ALPACA_COLUMN_MAP[key]
    if rename_map:
        normalized = normalized.rename(columns=rename_map)
    return normalized


def _coerce_csv_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use a common timestamp column as the DatetimeIndex when present."""
    normalized = df.copy()
    for column in normalized.columns:
        key = str(column).strip().lower()
        if key in {"timestamp", "datetime"}:
            normalized = normalized.set_index(column)
            break
    return normalized


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
