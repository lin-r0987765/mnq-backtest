#!/usr/bin/env python3
"""
Normalize Alpaca parquet files into repo-native CSVs for local research.

This keeps QuantConnect promotion on the existing 10-year QC workflow while
adding a parallel Alpaca-backed local research dataset with longer intraday
coverage than the current qqq_5m.csv snapshot.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ALPACA_ROOT = PROJECT_ROOT / "alpaca"
NORMALIZED_ROOT = ALPACA_ROOT / "normalized"
MANIFEST_PATH = NORMALIZED_ROOT / "alpaca_research_manifest.json"
ANALYSIS_VERSION = "v1_alpaca_research_normalization"

SUPPORTED_SYMBOL_DIRS = {
    "QQQ": (ALPACA_ROOT,),
    "SPY": (ALPACA_ROOT / "SPY", ALPACA_ROOT),
}
TIMEFRAME_TO_OUTPUT = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "1Hour": "1h",
    "4Hour": "4h",
    "1Day": "1d",
}


@dataclass
class OutputEntry:
    symbol: str
    source_file: str
    source_path: str
    output_file: str
    rows: int
    start_utc: str
    end_utc: str
    span_days: float
    span_years: float


def discover_symbol_sources() -> list[tuple[str, Path, str]]:
    discovered: list[tuple[str, Path, str]] = []
    seen_paths: set[Path] = set()
    for symbol, roots in SUPPORTED_SYMBOL_DIRS.items():
        for root in roots:
            if not root.exists():
                continue
            for parquet_path in sorted(root.glob(f"{symbol}_*_latest_iex_raw_regular.parquet")):
                if parquet_path in seen_paths:
                    continue
                matched = re.match(
                    rf"{re.escape(symbol)}_(.+?)_2016-01-01_latest_iex_raw_regular\.parquet$",
                    parquet_path.name,
                )
                if not matched:
                    continue
                timeframe = matched.group(1)
                output_suffix = TIMEFRAME_TO_OUTPUT.get(timeframe)
                if output_suffix is None:
                    continue
                output_name = f"{symbol.lower()}_{output_suffix}_alpaca.csv"
                discovered.append((symbol, parquet_path, output_name))
                seen_paths.add(parquet_path)
    if not discovered:
        raise FileNotFoundError(f"No supported Alpaca parquet files found under {ALPACA_ROOT}")
    return discovered


def normalize_alpaca_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")

    for column in ["open", "high", "low", "close", "volume"]:
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


def build_entry(symbol: str, source_path: Path, output_path: Path, normalized: pd.DataFrame) -> OutputEntry:
    timestamps = pd.to_datetime(normalized["Datetime"], utc=True)
    start = timestamps.min()
    end = timestamps.max()
    span_days = max((end - start).total_seconds() / 86400.0, 0.0)
    return OutputEntry(
        symbol=symbol,
        source_file=source_path.name,
        source_path=str(source_path),
        output_file=output_path.name,
        rows=len(normalized),
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        span_days=round(span_days, 3),
        span_years=round(span_days / 365.25, 3),
    )


def main() -> int:
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for symbol, source_path, output_name in discover_symbol_sources():
        frame = pd.read_parquet(source_path)
        normalized = normalize_alpaca_frame(frame)
        output_path = NORMALIZED_ROOT / output_name
        normalized.to_csv(output_path, index=False)
        entries.append(asdict(build_entry(symbol, source_path, output_path, normalized)))

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "alpaca_root": str(ALPACA_ROOT),
        "normalized_root": str(NORMALIZED_ROOT),
        "entries": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(entries)} normalized Alpaca files to {NORMALIZED_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
