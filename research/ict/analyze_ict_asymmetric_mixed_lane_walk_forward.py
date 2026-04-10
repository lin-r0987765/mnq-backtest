from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ict.analyze_ict_asymmetric_mixed_lane import (
    DEFAULT_INTRADAY,
    DEFAULT_PEER_CSV,
    _profile_specs,
    _run_profile,
)
from research.ict.analyze_ict_lite_reversal_baseline import RESEARCH_STANDARD
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_asymmetric_mixed_lane_walk_forward.json"
)


def _slice_by_dates(df, dates: list[Any], start: int, end: int):
    selected = set(dates[start:end])
    mask = np.isin(df.index.date, list(selected))
    return df[mask].copy()


def _windows(dates: list[Any]) -> list[dict[str, int]]:
    definitions = [
        ("train", "2020-01-01", "2022-12-31"),
        ("validation_2023", "2023-01-01", "2023-12-31"),
        ("validation_2024", "2024-01-01", "2024-12-31"),
        ("holdout_2025", "2025-01-01", "2025-12-31"),
    ]
    normalized_dates = [trading_day if isinstance(trading_day, date) else trading_day.date() for trading_day in dates]

    def _first_index_on_or_after(target: date) -> int | None:
        for idx, trading_day in enumerate(normalized_dates):
            if trading_day >= target:
                return idx
        return None

    def _first_index_after(target: date) -> int | None:
        for idx, trading_day in enumerate(normalized_dates):
            if trading_day > target:
                return idx
        return None

    boundaries: dict[str, tuple[int, int]] = {}
    for label, start_date, end_date in definitions:
        start_idx = _first_index_on_or_after(date.fromisoformat(start_date))
        end_idx = _first_index_after(date.fromisoformat(end_date))
        if start_idx is None:
            return []
        if end_idx is None:
            end_idx = len(normalized_dates)
        if start_idx >= end_idx:
            return []
        boundaries[label] = (start_idx, end_idx)
    return [
        {
            "train_start": boundaries["train"][0],
            "train_end": boundaries["train"][1],
            "validation_start": boundaries["validation_2023"][0],
            "validation_end": boundaries["validation_2023"][1],
            "holdout_start": boundaries["validation_2024"][0],
            "holdout_end": boundaries["validation_2024"][1],
        },
        {
            "train_start": boundaries["train"][0],
            "train_end": boundaries["validation_2023"][1],
            "validation_start": boundaries["validation_2024"][0],
            "validation_end": boundaries["validation_2024"][1],
            "holdout_start": boundaries["holdout_2025"][0],
            "holdout_end": boundaries["holdout_2025"][1],
        },
    ]


def _segment(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_pct": round(float(metrics["total_return_pct"]), 4),
        "profit_factor": round(float(metrics["profit_factor"]), 4),
        "win_rate_pct": round(float(metrics["win_rate_pct"]), 4),
        "total_trades": int(metrics["total_trades"]),
        "max_drawdown_pct": round(float(metrics["max_drawdown_pct"]), 4),
    }


def _summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if not folds:
        return {
            "folds": 0,
            "avg_validation_return_pct": 0.0,
            "avg_holdout_return_pct": 0.0,
            "positive_holdout_fold_pct": 0.0,
            "holdout_trade_total": 0,
        }
    validation_returns = [float(fold["validation"]["total_return_pct"]) for fold in folds]
    holdout_returns = [float(fold["holdout"]["total_return_pct"]) for fold in folds]
    holdout_trades = [int(fold["holdout"]["total_trades"]) for fold in folds]
    return {
        "folds": len(folds),
        "avg_validation_return_pct": round(float(np.mean(validation_returns)), 4),
        "avg_holdout_return_pct": round(float(np.mean(holdout_returns)), 4),
        "positive_holdout_fold_pct": round(
            float(sum(1 for value in holdout_returns if value > 0) / len(holdout_returns) * 100.0),
            4,
        ),
        "holdout_trade_total": int(sum(holdout_trades)),
    }


def _data_window(df) -> dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "trading_days": 0}
    return {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "trading_days": int(len(set(df.index.date))),
    }


def _period_label(df) -> str:
    window = _data_window(df)
    if window["start"] is None or window["end"] is None:
        return "empty"
    return f'{window["start"]} -> {window["end"]}'


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _resolve_output_path(requested_output: str, *, profile_label: str) -> Path:
    requested_path = Path(requested_output)
    if requested_path != DEFAULT_OUTPUT:
        return requested_path
    slug = _sanitize_label(profile_label)
    return requested_path.with_name(f"ict_asymmetric_mixed_lane_{slug}_walk_forward.json")


def main() -> None:
    profile_specs = _profile_specs()
    profile_labels = [str(spec["label"]) for spec in profile_specs]
    default_profile = (
        "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_plus_short_structure_refined_recovery_sl135_dailybiaslb8"
        if "rev_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_plus_short_structure_refined_recovery_sl135_dailybiaslb8" in profile_labels
        else profile_labels[0]
    )

    parser = argparse.ArgumentParser(
        description="Run walk-forward evaluation for the asymmetric ICT mixed-lane candidates."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="full_local")
    parser.add_argument("--profile", choices=profile_labels, default=default_profile)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    dates = sorted(set(merged.index.date))
    windows = _windows(dates)
    profile_spec = next(spec for spec in profile_specs if str(spec["label"]) == args.profile)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    output_path = _resolve_output_path(args.output, profile_label=str(profile_spec["label"]))

    folds: list[dict[str, Any]] = []
    for fold_number, window in enumerate(windows, start=1):
        train_df = _slice_by_dates(merged, dates, window["train_start"], window["train_end"])
        validation_df = _slice_by_dates(merged, dates, window["validation_start"], window["validation_end"])
        holdout_df = _slice_by_dates(merged, dates, window["holdout_start"], window["holdout_end"])
        if len(train_df) < 1000 or len(validation_df) < 500 or len(holdout_df) < 500:
            continue

        train_result = _run_profile(
            train_df,
            label=str(profile_spec["label"]),
            long_params=dict(profile_spec["long_params"]),
            short_params=dict(profile_spec["short_params"]),
            engine=engine,
        )
        validation_result = _run_profile(
            validation_df,
            label=str(profile_spec["label"]),
            long_params=dict(profile_spec["long_params"]),
            short_params=dict(profile_spec["short_params"]),
            engine=engine,
        )
        holdout_result = _run_profile(
            holdout_df,
            label=str(profile_spec["label"]),
            long_params=dict(profile_spec["long_params"]),
            short_params=dict(profile_spec["short_params"]),
            engine=engine,
        )
        folds.append(
            {
                "fold": fold_number,
                "train": _segment(train_result["metrics"]),
                "validation": _segment(validation_result["metrics"]),
                "holdout": _segment(holdout_result["metrics"]),
            }
        )

    output = {
        "analysis": "ict_asymmetric_mixed_lane_walk_forward",
        "profile": str(profile_spec["label"]),
        "params": {
            "long_profile": profile_spec["long_params"],
            "short_profile": profile_spec["short_params"],
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": _period_label(intraday_df),
        "peer_fetch_period": args.period if not args.peer_csv else None,
        "data_window": _data_window(intraday_df),
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "folds": folds,
        "summary": _summary(folds),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
