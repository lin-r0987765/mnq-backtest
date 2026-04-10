#!/usr/bin/env python3
"""取得 QQQ 多時間框架市場數據，並檢查是否滿足最少 8 年覆蓋。"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

from src.data.coverage import DEFAULT_DATA_TARGETS, DEFAULT_TARGET_YEARS, build_coverage_report
from src.data.polygon_provider import fetch_polygon_aggs


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "data_coverage_report.json"


@dataclass(frozen=True)
class DownloadJob:
    interval: str
    filename: str
    provider: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or audit QQQ market data coverage")
    parser.add_argument("--symbol", default="QQQ", help="Ticker symbol (default: QQQ)")
    parser.add_argument(
        "--provider",
        choices=["auto", "yfinance", "polygon"],
        default="auto",
        help="Data provider strategy",
    )
    parser.add_argument(
        "--target-years",
        type=float,
        default=DEFAULT_TARGET_YEARS,
        help="Minimum target coverage in years (default: 8)",
    )
    parser.add_argument(
        "--polygon-api-key-env",
        default="POLYGON_API_KEY",
        help="Environment variable name for Polygon API key",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only write coverage report and skip downloads",
    )
    return parser.parse_args()


def _target_years_str(target_years: float) -> str:
    return str(int(math.ceil(target_years)))


def _coverage_targets(target_years: float) -> dict:
    targets = {}
    for filename, spec in DEFAULT_DATA_TARGETS.items():
        targets[filename] = {**spec, "target_years": float(target_years)}
    return targets


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out.index.name = "Datetime"
    out.to_csv(path)


def _promote_download(filename: str) -> None:
    dest = PROJECT_ROOT / filename
    new_path = PROJECT_ROOT / f"{Path(filename).stem}_new.csv"
    backup_path = PROJECT_ROOT / f"{Path(filename).stem}_backup.csv"
    if not new_path.exists():
        return
    if dest.exists():
        shutil.copy2(dest, backup_path)
    shutil.move(str(new_path), str(dest))


def _download_yfinance(symbol: str, interval: str, target_years: float) -> pd.DataFrame:
    if not _HAS_YFINANCE:
        raise RuntimeError("yfinance is not installed.")

    expanded_years = max(int(math.ceil(target_years)) + 1, int(math.ceil(DEFAULT_TARGET_YEARS)) + 1)
    if interval == "5m":
        periods = [f"{expanded_years}y", "730d", "60d"]
    elif interval == "1h":
        periods = [f"{expanded_years}y", "730d", "2y"]
    elif interval == "1d":
        periods = [f"{expanded_years}y", f"{_target_years_str(max(target_years, DEFAULT_TARGET_YEARS))}y", "10y"]
    else:
        raise ValueError(f"Unsupported yfinance interval: {interval}")

    last_error = "yfinance returned empty dataframe"
    for period in periods:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[df.index.notna()]
            df.attrs["source_period"] = period
            return df
        last_error = f"yfinance returned empty dataframe for period={period}"

    raise RuntimeError(f"{last_error} ({symbol} {interval})")


def _download_polygon(symbol: str, interval: str, target_years: float, api_key: str) -> pd.DataFrame:
    end_date = date.today()
    start_date = end_date - timedelta(days=int(math.ceil(target_years * 365.25)))
    df = fetch_polygon_aggs(symbol, interval, start_date, end_date, api_key)
    if df.empty:
        raise RuntimeError(f"Polygon returned empty dataframe for {symbol} {interval}")
    return df


def _choose_jobs(provider: str, has_polygon_key: bool) -> list[DownloadJob]:
    if provider == "polygon":
        chosen = "polygon"
    elif provider == "yfinance":
        chosen = "yfinance"
    else:
        chosen = "polygon" if has_polygon_key else "yfinance"

    jobs = [
        DownloadJob("5m", "qqq_5m.csv", chosen if chosen == "polygon" else "yfinance", "5 分鐘線"),
        DownloadJob("1h", "qqq_1h.csv", chosen if chosen == "polygon" else "yfinance", "1 小時線"),
        DownloadJob("1d", "qqq_1d.csv", "yfinance" if provider == "auto" else chosen, "日線"),
    ]
    return jobs


def _print_report(report: dict) -> None:
    print("資料覆蓋檢查：")
    for filename, entry in report["files"].items():
        status = "OK" if entry["meets_target"] else "不足"
        print(
            f"- {filename}: {status} | rows={entry['rows']} | "
            f"span_years={entry['span_years']:.3f} / target={entry['target_years']:.1f}"
        )
    if report["all_targets_met"]:
        print("所有覆蓋目標已達成。")
    else:
        missing = ", ".join(report["missing_targets"])
        print(f"尚未達成 8 年目標: {missing}")


def main() -> int:
    args = parse_args()
    polygon_api_key = os.getenv(args.polygon_api_key_env, "").strip()
    has_polygon_key = bool(polygon_api_key)

    if args.audit_only:
        report = build_coverage_report(PROJECT_ROOT, targets=_coverage_targets(args.target_years))
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _print_report(report)
        return 0

    jobs = _choose_jobs(args.provider, has_polygon_key)
    failures: list[str] = []

    for job in jobs:
        print(f"下載 {job.description} ({job.provider})...")
        try:
            if job.provider == "polygon":
                if not polygon_api_key:
                    raise RuntimeError(f"缺少 {args.polygon_api_key_env}")
                df = _download_polygon(args.symbol, job.interval, args.target_years, polygon_api_key)
            else:
                df = _download_yfinance(args.symbol, job.interval, args.target_years)

            new_path = PROJECT_ROOT / f"{Path(job.filename).stem}_new.csv"
            _save_csv(df, new_path)
            _promote_download(job.filename)
            source_period = df.attrs.get("source_period", "")
            source_suffix = f", source_period={source_period}" if source_period else ""
            print(
                f"  完成 {job.filename}: rows={len(df)}, "
                f"range={df.index.min()} ~ {df.index.max()}{source_suffix}"
            )
        except Exception as exc:
            failures.append(f"{job.filename} ({job.provider}): {exc}")
            print(f"  失敗 {job.filename}: {exc}")

    report = build_coverage_report(PROJECT_ROOT, targets=_coverage_targets(args.target_years))
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(report)

    if args.provider != "polygon" and not has_polygon_key and not report["all_targets_met"]:
        print(
            "提示: 若要補足 8 年以上的 QQQ 5m / 1h，建議提供 Polygon API key，"
            "再用 `python fetch_data.py --provider polygon --target-years 8` 回補。"
        )

    if failures:
        print("下載失敗摘要：")
        for item in failures:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
