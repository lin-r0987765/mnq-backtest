#!/usr/bin/env python3
"""
Run a local reference analysis for the accepted v26 baseline on normalized Alpaca data.

This adds a broader local evidence lane without changing the rule that QC
promotion decisions stay on the 10-year QuantConnect dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.orb.analyze_local_orb_v25_profit_lock import (
    BASE_BE_GATE_MIN,
    BASE_BE_TRIGGER,
    BASE_PARAMS,
    MAX_ENTRIES_PER_SESSION,
    compute_metrics,
    load_csv_5m,
    simulate_orb_v25_profit_lock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTRADAY_PATH = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
MANIFEST_PATH = PROJECT_ROOT / "alpaca" / "normalized" / "alpaca_research_manifest.json"
LEGACY_INTRADAY_PATH = PROJECT_ROOT / "qqq_5m.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "alpaca_v26_reference_analysis.json"
ANALYSIS_VERSION = "v1_alpaca_v26_reference"
BASE_V26_PROFIT_LOCK_TRIGGER = 1.50
BASE_V26_PROFIT_LOCK_LEVEL = 0.25


def describe_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["Datetime"], utc=True)
    span_days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 0.0)
    return {
        "path": str(path),
        "rows": int(len(df)),
        "start_utc": ts.min().isoformat(),
        "end_utc": ts.max().isoformat(),
        "span_days": round(span_days, 3),
        "span_years": round(span_days / 365.25, 3),
    }


def compute_period_rows(trades: list, dates: list[pd.Timestamp.date], period_name: str) -> list[dict]:
    rows = []
    for date_value in dates:
        subset = [trade for trade in trades if trade.entry_time.date() == date_value]
        if not subset:
            continue
        metrics = compute_metrics(subset)
        rows.append(
            {
                period_name: str(date_value),
                "trades": metrics["trades"],
                "total_pnl": metrics["total_pnl"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "avg_duration_min": metrics["avg_duration_min"],
                "eod_exits": metrics["eod_exits"],
            }
        )
    return rows


def build_time_folds(df: pd.DataFrame, fold_count: int = 4) -> list[dict]:
    sessions = sorted(pd.Index(df.index.normalize().unique()))
    if not sessions:
        return []
    fold_size = max(len(sessions) // fold_count, 1)
    folds = []
    for idx in range(fold_count):
        start_pos = idx * fold_size
        end_pos = len(sessions) if idx == fold_count - 1 else min((idx + 1) * fold_size, len(sessions))
        if start_pos >= len(sessions):
            break
        fold_sessions = sessions[start_pos:end_pos]
        folds.append(
            {
                "fold": idx + 1,
                "start": fold_sessions[0].date(),
                "end": fold_sessions[-1].date(),
            }
        )
    return folds


def compute_fold_metrics(trades: list, folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        subset = [
            trade
            for trade in trades
            if fold["start"] <= trade.entry_time.date() <= fold["end"]
        ]
        metrics = compute_metrics(subset)
        rows.append(
            {
                "fold": fold["fold"],
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                **metrics,
            }
        )
    return rows


def main() -> int:
    if not INTRADAY_PATH.exists():
        raise FileNotFoundError(
            f"Normalized Alpaca intraday CSV missing: {INTRADAY_PATH}. Run prepare_alpaca_research_data.py first."
        )

    intraday = load_csv_5m(INTRADAY_PATH)
    trades = simulate_orb_v25_profit_lock(
        intraday,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER,
        profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL,
    )
    metrics = compute_metrics(trades)

    annual_metrics = []
    for year in sorted({trade.entry_time.year for trade in trades}):
        subset = [trade for trade in trades if trade.entry_time.year == year]
        annual_metrics.append({"year": year, **compute_metrics(subset)})

    folds = build_time_folds(intraday, fold_count=4)
    fold_metrics = compute_fold_metrics(trades, folds)

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "baseline_reference": "v26-profit-lock",
        "qc_promotion_policy": "unchanged_use_10y_qc_only",
        "alpaca_manifest": json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else None,
        "legacy_local_intraday": describe_csv(LEGACY_INTRADAY_PATH) if LEGACY_INTRADAY_PATH.exists() else None,
        "alpaca_intraday": describe_csv(INTRADAY_PATH),
        "baseline_params": {
            "max_entries_per_session": MAX_ENTRIES_PER_SESSION,
            "breakeven_trigger_mult": BASE_BE_TRIGGER,
            "breakeven_active_minutes": BASE_BE_GATE_MIN,
            "profit_lock_trigger_mult": BASE_V26_PROFIT_LOCK_TRIGGER,
            "profit_lock_level_mult": BASE_V26_PROFIT_LOCK_LEVEL,
        },
        "v26_alpaca_metrics": metrics,
        "annual_metrics": annual_metrics,
        "time_folds": fold_metrics,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote Alpaca v26 reference analysis to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
