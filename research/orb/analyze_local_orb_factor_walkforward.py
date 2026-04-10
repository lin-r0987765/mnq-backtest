from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.orb.analyze_local_orb_daily_factors import stats_for_subset
from research.shared.analyze_single_factor_robustness import build_feature_frame
from src.backtest.engine import BacktestEngine
from src.strategies.orb import ORBStrategy


def load_bars(intraday_csv: Path) -> pd.DataFrame:
    bars = pd.read_csv(intraday_csv)
    bars["Datetime"] = pd.to_datetime(bars["Datetime"], utc=True)
    bars = bars.set_index("Datetime").sort_index()
    bars["et_date"] = bars.index.tz_convert("America/New_York").date
    return bars


def load_folds(walk_forward_json: Path) -> list[dict[str, object]]:
    obj = json.loads(walk_forward_json.read_text(encoding="utf-8"))
    return list(obj["orb"]["folds"])


def run_fold_backtest(bars: pd.DataFrame, start_date: object, end_date: object) -> pd.DataFrame:
    subset = bars[(bars["et_date"] >= start_date) & (bars["et_date"] <= end_date)].drop(columns=["et_date"])
    result = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0).run(ORBStrategy(), subset)
    trades = pd.DataFrame(result.trades)
    if trades.empty:
        return trades

    trades["Entry Timestamp"] = pd.to_datetime(trades["Entry Timestamp"], utc=True)
    trades["Exit Timestamp"] = pd.to_datetime(trades["Exit Timestamp"], utc=True)
    trades["entry_date"] = trades["Entry Timestamp"].dt.tz_convert("America/New_York").dt.date
    trades["net_pnl"] = pd.to_numeric(trades["PnL"], errors="coerce").fillna(0.0)
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)
    return trades


def summarize_candidate_folds(folds: list[dict[str, object]]) -> dict[str, object]:
    baseline_net = round(float(sum(item["baseline"]["net_pnl"] for item in folds)), 2)
    kept_net = round(float(sum(item["kept"]["net_pnl"] for item in folds)), 2)
    excluded_net = round(float(sum(item["excluded"]["net_pnl"] for item in folds)), 2)
    improved = sum(1 for item in folds if item["kept"]["net_pnl"] > item["baseline"]["net_pnl"])
    positive_kept = sum(1 for item in folds if item["kept"]["net_pnl"] > 0)
    excluded_negative = sum(1 for item in folds if item["excluded"]["net_pnl"] < 0)
    no_trade_folds = sum(1 for item in folds if item["kept"]["trades"] == 0)
    if improved >= 3 and positive_kept >= 3 and excluded_negative >= 3:
        verdict = "WALKFORWARD_LEADER"
    elif improved >= 2:
        verdict = "WALKFORWARD_MIXED"
    else:
        verdict = "WALKFORWARD_WEAK"
    return {
        "baseline_net_pnl": baseline_net,
        "kept_net_pnl": kept_net,
        "excluded_net_pnl": excluded_net,
        "improved_vs_baseline_folds": int(improved),
        "positive_kept_folds": int(positive_kept),
        "excluded_negative_folds": int(excluded_negative),
        "no_trade_folds": int(no_trade_folds),
        "verdict": verdict,
    }


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    candidates: list[str],
    base_filter: str | None = None,
) -> dict[str, object]:
    bars = load_bars(intraday_csv)
    features, _ = build_feature_frame(daily_csv)
    folds = load_folds(walk_forward_json)

    candidate_folds: dict[str, list[dict[str, object]]] = {name: [] for name in candidates}
    fold_rows: list[dict[str, object]] = []

    for fold in folds:
        start_text, end_text = [part.strip() for part in str(fold["test_period"]).split("~")]
        start_date = pd.Timestamp(start_text).date()
        end_date = pd.Timestamp(end_text).date()
        trades = run_fold_backtest(bars, start_date, end_date)
        merged = trades.merge(features, left_on="entry_date", right_on="date", how="left") if len(trades) else trades
        if base_filter and len(merged):
            merged = merged[merged[base_filter].fillna(False)].copy()
        baseline = stats_for_subset(merged)
        row = {
            "fold": int(fold["fold"]),
            "test_period": str(fold["test_period"]),
            "baseline": baseline,
            "candidates": {},
        }
        for name in candidates:
            mask = merged[name].fillna(False) if len(merged) else pd.Series(dtype=bool)
            kept = merged[mask].copy() if len(merged) else merged
            excluded = merged[~mask].copy() if len(merged) else merged
            candidate_row = {
                "kept": stats_for_subset(kept),
                "excluded": stats_for_subset(excluded),
                "delta_vs_baseline": round(float(stats_for_subset(kept)["net_pnl"] - baseline["net_pnl"]), 2),
            }
            row["candidates"][name] = candidate_row
            candidate_folds[name].append(
                {
                    "fold": int(fold["fold"]),
                    "test_period": str(fold["test_period"]),
                    "baseline": baseline,
                    **candidate_row,
                }
            )
        fold_rows.append(row)

    summaries = {
        name: {
            "folds": candidate_folds[name],
            "summary": summarize_candidate_folds(candidate_folds[name]),
        }
        for name in candidates
    }

    return {
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": base_filter,
        "fold_count": len(folds),
        "folds": fold_rows,
        "candidates": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare daily filters on local ORB walk-forward test folds.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--candidates", default="prev_day_up,mom3_positive,close_above_sma10")
    parser.add_argument("--base-filter", default="")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    base_filter = args.base_filter.strip() or None
    result = evaluate(
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
        candidates=candidates,
        base_filter=base_filter,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "local_orb_walkforward_factor_tiebreak.json"
    if base_filter:
        out_name = f"local_orb_{base_filter}_walkforward_factor_tiebreak.json"
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
