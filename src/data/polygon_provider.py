from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


@dataclass(frozen=True)
class PolygonInterval:
    multiplier: int
    timespan: str
    rth_start: str
    rth_end: str


POLYGON_INTERVALS = {
    "5m": PolygonInterval(multiplier=5, timespan="minute", rth_start="09:30", rth_end="15:55"),
    "1h": PolygonInterval(multiplier=1, timespan="hour", rth_start="09:30", rth_end="15:30"),
    "1d": PolygonInterval(multiplier=1, timespan="day", rth_start="", rth_end=""),
}


def polygon_aggs_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("results") or []
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame.from_records(rows)
    rename_map = {
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
        "t": "Timestamp",
    }
    df = df.rename(columns=rename_map)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).set_index("Timestamp")
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df = df.sort_index()
    return df


def _append_api_key(next_url: str, api_key: str) -> str:
    if "apiKey=" in next_url:
        return next_url
    joiner = "&" if "?" in next_url else "?"
    return f"{next_url}{joiner}apiKey={api_key}"


def fetch_polygon_aggs(
    symbol: str,
    interval: str,
    start_date: date,
    end_date: date,
    api_key: str,
    *,
    adjusted: bool = True,
    limit: int = 50000,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    if interval not in POLYGON_INTERVALS:
        raise ValueError(f"Unsupported polygon interval: {interval}")

    spec = POLYGON_INTERVALS[interval]
    client = session or requests.Session()
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{spec.multiplier}/{spec.timespan}/{start_date.isoformat()}/{end_date.isoformat()}"
    )
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key,
    }

    frames: list[pd.DataFrame] = []
    next_url: str | None = url
    first_request = True

    while next_url:
        response = client.get(
            next_url,
            params=params if first_request else None,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        frame = polygon_aggs_to_frame(payload)
        if not frame.empty:
            frames.append(frame)
        next_url = payload.get("next_url")
        if next_url:
            next_url = _append_api_key(next_url, api_key)
        first_request = False

    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if interval in {"5m", "1h"}:
        eastern = df.tz_convert("America/New_York")
        eastern = eastern.between_time(spec.rth_start, spec.rth_end)
        df = eastern.tz_convert(timezone.utc)

    return df
