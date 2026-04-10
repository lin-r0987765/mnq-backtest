from __future__ import annotations

from pathlib import Path

import pandas as pd


NEW_YORK_TZ = "America/New_York"


def load_daily_market_frame(daily_path: Path) -> pd.DataFrame:
    daily = pd.read_csv(daily_path)
    daily["Datetime"] = pd.to_datetime(daily["Datetime"], utc=True)
    daily = daily.sort_values("Datetime").copy()
    daily["market_date"] = daily["Datetime"].dt.tz_convert(NEW_YORK_TZ).dt.date
    daily["session_date"] = daily["market_date"].shift(-1)
    return daily


def align_features_to_next_session(
    daily: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[object]]:
    aligned = daily[["market_date", "session_date", *feature_columns]].copy()
    aligned = aligned[aligned["session_date"].notna()].copy()
    aligned = aligned.rename(columns={"market_date": "source_date", "session_date": "date"})
    calendar_dates = list(aligned["date"])
    return aligned[["date", "source_date", *feature_columns]].copy(), calendar_dates
