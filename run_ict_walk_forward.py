from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_walk_forward_results.json"


def _fold_windows(
    dates: list[Any],
    *,
    train_days: int,
    validation_days: int,
    holdout_days: int,
    step_days: int,
) -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    start = 0
    fold = 0
    total_needed = train_days + validation_days + holdout_days
    while start + total_needed <= len(dates):
        fold += 1
        windows.append(
            {
                "fold": fold,
                "train_start": start,
                "train_end": start + train_days,
                "validation_start": start + train_days,
                "validation_end": start + train_days + validation_days,
                "holdout_start": start + train_days + validation_days,
                "holdout_end": start + total_needed,
            }
        )
        start += step_days
    return windows


def _slice_by_dates(df, dates: list[Any], start: int, end: int):
    selected = set(dates[start:end])
    mask = np.isin(df.index.date, list(selected))
    return df[mask].copy()


def _segment_payload(result) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "total_return_pct": _serialise_metric(metrics["total_return_pct"]),
        "sharpe_ratio": _serialise_metric(metrics["sharpe_ratio"]),
        "max_drawdown_pct": _serialise_metric(metrics["max_drawdown_pct"]),
        "profit_factor": _serialise_metric(metrics["profit_factor"]),
        "total_trades": int(metrics["total_trades"]),
        "win_rate_pct": _serialise_metric(metrics["win_rate_pct"]),
        "avg_trade_pct": _serialise_metric(metrics["avg_trade_pct"]),
    }


def _serialise_metric(value: Any) -> float | str:
    numeric = float(value)
    if np.isposinf(numeric):
        return "Infinity"
    if np.isneginf(numeric):
        return "-Infinity"
    return round(numeric, 4)


def _summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if not folds:
        return {
            "folds": 0,
            "avg_validation_return_pct": 0.0,
            "avg_holdout_return_pct": 0.0,
            "avg_holdout_sharpe_ratio": 0.0,
            "positive_holdout_fold_pct": 0.0,
            "holdout_trade_total": 0,
            "best_holdout_fold": None,
        }

    validation_returns = [float(fold["validation"]["total_return_pct"]) for fold in folds]
    holdout_returns = [float(fold["holdout"]["total_return_pct"]) for fold in folds]
    holdout_sharpes = [float(fold["holdout"]["sharpe_ratio"]) for fold in folds]
    holdout_trades = [int(fold["holdout"]["total_trades"]) for fold in folds]
    finite_holdout_sharpes = [value for value in holdout_sharpes if np.isfinite(value)]
    best_fold = max(
        folds,
        key=lambda fold: (
            float(fold["holdout"]["total_return_pct"]),
            float(fold["holdout"]["profit_factor"]),
            int(fold["holdout"]["total_trades"]),
        ),
    )
    return {
        "folds": len(folds),
        "avg_validation_return_pct": round(float(np.mean(validation_returns)), 4),
        "avg_holdout_return_pct": round(float(np.mean(holdout_returns)), 4),
        "avg_holdout_sharpe_ratio": round(float(np.mean(finite_holdout_sharpes)), 4) if finite_holdout_sharpes else 0.0,
        "positive_holdout_fold_pct": round(
            float(sum(1 for value in holdout_returns if value > 0) / len(holdout_returns) * 100.0),
            4,
        ),
        "holdout_trade_total": int(sum(holdout_trades)),
        "best_holdout_fold": {
            "fold": int(best_fold["fold"]),
            "holdout_return_pct": _serialise_metric(best_fold["holdout"]["total_return_pct"]),
            "holdout_profit_factor": best_fold["holdout"]["profit_factor"],
            "holdout_trades": int(best_fold["holdout"]["total_trades"]),
        },
    }


def _verdict(summary: dict[str, Any]) -> tuple[str, str]:
    if summary["folds"] == 0:
        return (
            "ICT_WALK_FORWARD_INSUFFICIENT_FOLDS",
            "The current split configuration did not produce enough train-validation-holdout folds to evaluate the active lite ICT frontier.",
        )
    if summary["positive_holdout_fold_pct"] >= 60.0 and summary["avg_holdout_return_pct"] > 0:
        return (
            "ICT_WALK_FORWARD_POSITIVE_OOS_STABILITY_CONFIRMED",
            "The active lite ICT frontier stays positive across most holdout folds, so the roadmap can keep improving density without discarding the current reversal lane.",
        )
    if summary["avg_holdout_return_pct"] > 0:
        return (
            "ICT_WALK_FORWARD_MIXED_BUT_POSITIVE",
            "The active lite ICT frontier stays positive on average in holdout, but fold-level stability is still mixed and should not be treated as promotion-ready.",
        )
    return (
        "ICT_WALK_FORWARD_OOS_REJECTED",
        "The active lite ICT frontier fails to stay positive on average in holdout, so density work should pause until the lane regains out-of-sample stability.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train-validation-holdout walk-forward checks on the active lite ICT frontier."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--validation-days", type=int, default=20)
    parser.add_argument("--holdout-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=20)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    dates = sorted(set(merged.index.date))
    windows = _fold_windows(
        dates,
        train_days=args.train_days,
        validation_days=args.validation_days,
        holdout_days=args.holdout_days,
        step_days=args.step_days,
    )

    params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(enable_smt=True)
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    folds: list[dict[str, Any]] = []

    for window in windows:
        train_df = _slice_by_dates(merged, dates, window["train_start"], window["train_end"])
        validation_df = _slice_by_dates(merged, dates, window["validation_start"], window["validation_end"])
        holdout_df = _slice_by_dates(merged, dates, window["holdout_start"], window["holdout_end"])
        if len(train_df) < 100 or len(validation_df) < 50 or len(holdout_df) < 50:
            continue

        train_result = engine.run(ICTEntryModelStrategy(params=dict(params)), train_df)
        validation_result = engine.run(ICTEntryModelStrategy(params=dict(params)), validation_df)
        holdout_result = engine.run(ICTEntryModelStrategy(params=dict(params)), holdout_df)

        folds.append(
            {
                "fold": int(window["fold"]),
                "train_period": {
                    "start": str(dates[window["train_start"]]),
                    "end": str(dates[window["train_end"] - 1]),
                },
                "validation_period": {
                    "start": str(dates[window["validation_start"]]),
                    "end": str(dates[window["validation_end"] - 1]),
                },
                "holdout_period": {
                    "start": str(dates[window["holdout_start"]]),
                    "end": str(dates[window["holdout_end"] - 1]),
                },
                "train": _segment_payload(train_result),
                "validation": _segment_payload(validation_result),
                "holdout": _segment_payload(holdout_result),
            }
        )

    summary = _summarize_folds(folds)
    verdict, interpretation = _verdict(summary)
    output = {
        "analysis": "ict_walk_forward",
        "profile": "lite_ict_reversal_relaxed_smt_looser_sweep_faster_retest_frontier",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "walk_forward_config": {
            "train_days": int(args.train_days),
            "validation_days": int(args.validation_days),
            "holdout_days": int(args.holdout_days),
            "step_days": int(args.step_days),
        },
        "frontier_params": {
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "smt_threshold": float(params["smt_threshold"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "displacement_body_min_pct": float(params["displacement_body_min_pct"]),
        },
        "folds": folds,
        "summary": summary,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
