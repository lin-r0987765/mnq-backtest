from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from research.qc.analyze_qc_regime_prototypes import compute_profit_factor
from research.shared.analyze_single_factor_robustness import build_feature_frame
from src.backtest.engine import BacktestEngine
from src.strategies.orb import ORBStrategy


def run_local_backtest(intraday_csv: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    bars = pd.read_csv(intraday_csv)
    bars["Datetime"] = pd.to_datetime(bars["Datetime"], utc=True)
    bars = bars.set_index("Datetime").sort_index()

    result = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0).run(ORBStrategy(), bars)
    trades = pd.DataFrame(result.trades)
    if trades.empty:
        return trades, result.metrics, {}

    trades["Entry Timestamp"] = pd.to_datetime(trades["Entry Timestamp"], utc=True)
    trades["Exit Timestamp"] = pd.to_datetime(trades["Exit Timestamp"], utc=True)
    trades["entry_date"] = trades["Entry Timestamp"].dt.tz_convert("America/New_York").dt.date
    trades["entry_month"] = trades["Entry Timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    trades["net_pnl"] = pd.to_numeric(trades["PnL"], errors="coerce").fillna(0.0)
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)

    meta = {
        "first_bar_utc": str(bars.index.min()),
        "last_bar_utc": str(bars.index.max()),
        "session_count": int(pd.Index(bars.index.tz_convert("America/New_York").date).nunique()),
    }
    return trades, result.metrics, meta


def safe_profit_factor(value: float) -> float:
    if np.isfinite(value):
        return round(float(value), 3)
    return 999.0 if value > 0 else 0.0


def stats_for_subset(subset: pd.DataFrame) -> dict[str, float]:
    if subset.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
        }
    return {
        "trades": int(len(subset)),
        "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2),
        "profit_factor": safe_profit_factor(float(compute_profit_factor(subset["net_pnl"]))),
        "net_pnl": round(float(subset["net_pnl"].sum()), 2),
    }


def half_split(subset: pd.DataFrame) -> dict[str, dict[str, float]]:
    ordered = subset.sort_values("Entry Timestamp").reset_index(drop=True)
    midpoint = len(ordered) // 2
    first_half = ordered.iloc[:midpoint].copy()
    second_half = ordered.iloc[midpoint:].copy()
    return {
        "first_half": stats_for_subset(first_half),
        "second_half": stats_for_subset(second_half),
    }


def monthly_stats(subset: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for month, part in subset.groupby("entry_month"):
        out[str(month)] = stats_for_subset(part)
    return out


def leave_one_month_out(subset: pd.DataFrame) -> dict[str, object]:
    nets: dict[str, float] = {}
    for month in sorted(subset["entry_month"].dropna().unique()):
        part = subset[subset["entry_month"] != month]
        nets[str(month)] = round(float(part["net_pnl"].sum()), 2)
    min_net = min(nets.values()) if nets else 0.0
    return {
        "nets": nets,
        "min_net_pnl": round(float(min_net), 2),
        "all_positive": bool(nets) and all(value > 0 for value in nets.values()),
    }


def local_verdict(candidate: dict[str, object], baseline: dict[str, float]) -> str:
    kept = candidate["kept"]
    excluded = candidate["excluded"]
    halves = candidate["half_split"]
    loo = candidate["leave_one_month_out"]
    if (
        kept["net_pnl"] > baseline["net_pnl"]
        and excluded["net_pnl"] < 0
        and halves["first_half"]["net_pnl"] > 0
        and halves["second_half"]["net_pnl"] > 0
        and loo["all_positive"]
    ):
        return "LOCAL_CONFIRMATION"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0:
        return "LOCAL_POSITIVE_BUT_MIXED"
    return "LOCAL_WEAK_OR_MIXED"


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    candidates: list[str],
    base_filter: str | None = None,
) -> dict[str, object]:
    features, _ = build_feature_frame(daily_csv)
    trades, engine_metrics, sample_meta = run_local_backtest(intraday_csv)
    merged = trades.merge(features, left_on="entry_date", right_on="date", how="left")
    if base_filter:
        merged = merged[merged[base_filter].fillna(False)].copy()

    baseline = stats_for_subset(merged)
    results: dict[str, object] = {}
    for name in candidates:
        mask = merged[name].fillna(False)
        kept = merged[mask].copy()
        excluded = merged[~mask].copy()
        candidate = {
            "kept": stats_for_subset(kept),
            "excluded": stats_for_subset(excluded),
            "half_split": half_split(kept),
            "leave_one_month_out": leave_one_month_out(kept),
            "monthly_kept": monthly_stats(kept),
        }
        candidate["verdict"] = local_verdict(candidate, baseline)
        results[name] = candidate

    return {
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "base_filter": base_filter,
        "baseline_backtest_metrics": engine_metrics,
        "sample_meta": sample_meta,
        "baseline": baseline,
        "candidates": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local ORB daily-filter confirmation on qqq_5m sample.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument(
        "--candidates",
        default="prev_day_up,mom3_positive,mom10_positive,close_above_sma10,close_above_sma20",
    )
    parser.add_argument("--base-filter", default="")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    base_filter = args.base_filter.strip() or None
    result = evaluate(Path(args.intraday_csv), Path(args.daily_csv), candidates, base_filter=base_filter)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "local_orb_2026_daily_factor_confirmation.json"
    if base_filter:
        out_name = f"local_orb_{base_filter}_daily_factor_confirmation.json"
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
