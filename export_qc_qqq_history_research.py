#!/usr/bin/env python3
"""Export adjusted QQQ history from QuantConnect Research.

Paste this file into a QuantConnect Research notebook cell or upload it as a
project file and run it there. It exports QQQ history in the same column layout
used by the local repo CSVs:

- ``qqq_5m_qc.csv``
- ``qqq_1h_qc.csv``
- ``qqq_1d_qc.csv``
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from QuantConnect import DataNormalizationMode, Resolution
    from QuantConnect.Research import QuantBook
except ImportError:  # pragma: no cover - local unit tests import helpers only
    DataNormalizationMode = None
    Resolution = None
    QuantBook = None


SYMBOL = "QQQ"
START = datetime(2017, 4, 3)
END_INCLUSIVE = datetime(2026, 4, 2)
OUTPUT_DIR = Path("qc_history_export")
FRAME_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CSV_EXPORT_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]


def _empty_history_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=FRAME_COLUMNS)
    frame.index = pd.DatetimeIndex([], tz="UTC", name="Datetime")
    return frame


def _infer_time_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return column

    for column in frame.columns:
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if parsed.notna().any():
            return column

    raise ValueError("Unable to infer the timestamp column from QuantConnect history output.")


def normalize_qc_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize QuantConnect history output into repo CSV layout."""
    if history is None or len(history) == 0:
        return _empty_history_frame()

    frame = history.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    else:
        frame = frame.sort_index().reset_index()

    time_column = _infer_time_column(frame)
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame[frame[time_column].notna()].sort_values(time_column).set_index(time_column)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    frame = frame.rename(columns=rename_map)

    missing = [column for column in FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"QuantConnect history is missing expected OHLCV columns: {missing}")

    frame = frame[FRAME_COLUMNS].copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[frame.index.notna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index.name = "Datetime"

    for column in ["Open", "High", "Low", "Close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0).astype("int64")

    return frame


def resample_ohlcv(frame: pd.DataFrame, rule: str, *, offset: str | None = None) -> pd.DataFrame:
    """Resample an OHLCV frame with left-labeled bars."""
    if frame.empty:
        return _empty_history_frame()

    resample_kwargs: dict[str, Any] = {"label": "left", "closed": "left"}
    if offset is not None:
        resample_kwargs["offset"] = offset

    resampled = frame.resample(rule, **resample_kwargs).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
    resampled.index.name = "Datetime"
    resampled["Volume"] = pd.to_numeric(resampled["Volume"], errors="coerce").fillna(0).astype("int64")
    return resampled


def trim_to_inclusive_date_range(
    frame: pd.DataFrame,
    *,
    start: datetime,
    end_inclusive: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    start_ts = pd.Timestamp(start, tz="UTC")
    end_exclusive_ts = pd.Timestamp(end_inclusive + timedelta(days=1), tz="UTC")
    return frame[(frame.index >= start_ts) & (frame.index < end_exclusive_ts)].copy()


def save_history_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    output.index = pd.to_datetime(output.index, utc=True, errors="coerce")
    output.index.name = "Datetime"
    output = output[CSV_EXPORT_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path)


def _summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": int(len(frame)),
        "start": frame.index.min().isoformat(),
        "end": frame.index.max().isoformat(),
    }


def run_export(output_dir: Path | str = OUTPUT_DIR) -> dict[str, dict[str, Any]]:
    """Fetch and save adjusted QQQ history from QuantConnect Research."""
    if QuantBook is None or Resolution is None or DataNormalizationMode is None:
        raise RuntimeError(
            "QuantConnect Research libraries are not available. "
            "Run this script inside QuantConnect Research."
        )

    end_exclusive = END_INCLUSIVE + timedelta(days=1)
    qb = QuantBook()
    security = qb.add_equity(
        SYMBOL,
        Resolution.MINUTE,
        fill_forward=False,
        extended_market_hours=False,
        data_normalization_mode=DataNormalizationMode.ADJUSTED,
    )
    symbol = security.symbol

    minute_history = qb.history(symbol, START, end_exclusive, Resolution.MINUTE)
    daily_history = qb.history(symbol, START, end_exclusive, Resolution.DAILY)

    minute_frame = normalize_qc_history_frame(minute_history)
    daily_frame = trim_to_inclusive_date_range(
        normalize_qc_history_frame(daily_history),
        start=START,
        end_inclusive=END_INCLUSIVE,
    )

    five_minute_frame = trim_to_inclusive_date_range(
        resample_ohlcv(minute_frame, "5min"),
        start=START,
        end_inclusive=END_INCLUSIVE,
    )
    one_hour_frame = trim_to_inclusive_date_range(
        resample_ohlcv(minute_frame, "60min", offset="30min"),
        start=START,
        end_inclusive=END_INCLUSIVE,
    )

    output_root = Path(output_dir)
    outputs = {
        "qqq_5m_qc.csv": five_minute_frame,
        "qqq_1h_qc.csv": one_hour_frame,
        "qqq_1d_qc.csv": daily_frame,
    }

    summary: dict[str, dict[str, Any]] = {}
    for filename, frame in outputs.items():
        destination = output_root / filename
        save_history_csv(frame, destination)
        summary[filename] = {"path": str(destination), **_summarize_frame(frame)}

    print(f"Exported {SYMBOL} history to {output_root.resolve()}")
    for filename, details in summary.items():
        print(
            f"- {filename}: rows={details['rows']} | "
            f"range={details['start']} -> {details['end']}"
        )

    return summary


if __name__ == "__main__":
    run_export()
