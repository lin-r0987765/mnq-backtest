from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.orb.analyze_local_orb_v18_return_cap import (
    apply_base_filter,
    leave_one_month_out,
    load_bars,
    merge_features,
    parse_folds,
    run_local_backtest,
    stats_for_subset,
)
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"


def build_feature_frame(daily_csv: Path) -> tuple[object, list[str]]:
    daily = load_daily_market_frame(daily_csv)
    close = daily["Close"]

    daily["prev_day_up"] = close.pct_change() > 0.0
    daily["mom3_positive"] = (close / close.shift(3) - 1.0) > 0.0

    feature_columns: list[str] = []
    for window in (4, 5, 6, 7, 8):
        label = f"mom{window}_positive"
        daily[label] = (close / close.shift(window) - 1.0) > 0.0
        feature_columns.append(label)

    for window in (4, 5, 6, 7, 8, 9):
        label = f"close_above_sma{window}"
        daily[label] = close > close.rolling(window).mean()
        feature_columns.append(label)

    aligned, _ = align_features_to_next_session(daily, ["prev_day_up", "mom3_positive", *feature_columns])
    return aligned, feature_columns


def evaluate_walkforward(bars, features, folds, candidate_name: str) -> dict[str, object]:
    improved = 0
    positive_kept = 0
    excluded_negative = 0
    kept_total = 0.0
    baseline_total = 0.0

    rows: list[dict[str, object]] = []
    for fold in folds:
        start_text, end_text = [part.strip() for part in str(fold["test_period"]).split("~")]
        merged = merge_features(
            run_local_backtest(
                bars,
                start_date=pd.Timestamp(start_text).date(),
                end_date=pd.Timestamp(end_text).date(),
            ),
            features,
        )
        baseline = apply_base_filter(merged)
        kept = baseline[baseline[candidate_name].fillna(False)].copy()
        excluded = baseline[~baseline[candidate_name].fillna(False)].copy()

        base_stats = stats_for_subset(baseline)
        kept_stats = stats_for_subset(kept)
        excluded_stats = stats_for_subset(excluded)

        if kept_stats["net_pnl"] > base_stats["net_pnl"]:
            improved += 1
        if kept_stats["net_pnl"] > 0:
            positive_kept += 1
        if excluded_stats["net_pnl"] < 0:
            excluded_negative += 1

        kept_total += kept_stats["net_pnl"]
        baseline_total += base_stats["net_pnl"]

        rows.append(
            {
                "fold": int(fold["fold"]),
                "test_period": str(fold["test_period"]),
                "baseline": base_stats,
                "kept": kept_stats,
                "excluded": excluded_stats,
                "delta_vs_baseline": round(float(kept_stats["net_pnl"] - base_stats["net_pnl"]), 2),
            }
        )

    return {
        "folds": rows,
        "summary": {
            "improved_vs_baseline_folds": int(improved),
            "positive_kept_folds": int(positive_kept),
            "excluded_negative_folds": int(excluded_negative),
            "kept_net_pnl": round(float(kept_total), 2),
            "baseline_net_pnl": round(float(baseline_total), 2),
            "delta_vs_baseline": round(float(kept_total - baseline_total), 2),
        },
    }


def verdict(candidate: dict[str, object], baseline: dict[str, float]) -> str:
    kept = candidate["kept"]
    excluded = candidate["excluded"]
    walk = candidate["walkforward"]["summary"]
    loo = candidate["leave_one_month_out"]

    if (
        kept["net_pnl"] > baseline["net_pnl"]
        and excluded["net_pnl"] < 0
        and walk["delta_vs_baseline"] >= 0
        and walk["improved_vs_baseline_folds"] >= 2
        and loo["all_positive"]
    ):
        return "LOCAL_RESEARCH_LEADER"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0:
        return "LOCAL_PROMISING"
    return "LOCAL_WEAK_OR_MIXED"


def evaluate(intraday_csv: Path, daily_csv: Path, walk_forward_json: Path) -> dict[str, object]:
    bars = load_bars(intraday_csv)
    features, candidate_names = build_feature_frame(daily_csv)
    folds = parse_folds(walk_forward_json)

    merged = merge_features(run_local_backtest(bars), features)
    baseline = apply_base_filter(merged)
    baseline_stats = stats_for_subset(baseline)

    candidates: dict[str, object] = {}
    for name in candidate_names:
        kept = baseline[baseline[name].fillna(False)].copy()
        excluded = baseline[~baseline[name].fillna(False)].copy()
        candidate = {
            "kept": stats_for_subset(kept),
            "excluded": stats_for_subset(excluded),
            "leave_one_month_out": leave_one_month_out(kept),
            "walkforward": evaluate_walkforward(bars, features, folds, name),
        }
        candidate["delta_vs_baseline_net_pnl"] = round(float(candidate["kept"]["net_pnl"] - baseline_stats["net_pnl"]), 2)
        candidate["verdict"] = verdict(candidate, baseline_stats)
        candidates[name] = candidate

    def rank_key(item: tuple[str, dict[str, object]]) -> tuple[int, float, float]:
        value = item[1]
        verdict_rank = {
            "LOCAL_RESEARCH_LEADER": 2,
            "LOCAL_PROMISING": 1,
            "LOCAL_WEAK_OR_MIXED": 0,
        }[value["verdict"]]
        return (
            verdict_rank,
            value["walkforward"]["summary"]["delta_vs_baseline"],
            value["kept"]["net_pnl"],
        )

    best_name, best_value = max(candidates.items(), key=rank_key)

    return {
        "research_scope": "local_v18_adjacent_trend_scan",
        "analysis_version": "v1_adjacent_momentum_and_sma_local_only",
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": BASE_FILTER_LABEL,
        "baseline": baseline_stats,
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": {
                "label": best_name,
                "kept_net_pnl": best_value["kept"]["net_pnl"],
                "excluded_net_pnl": best_value["excluded"]["net_pnl"],
                "walkforward_delta_vs_baseline": best_value["walkforward"]["summary"]["delta_vs_baseline"],
                "improved_vs_baseline_folds": best_value["walkforward"]["summary"]["improved_vs_baseline_folds"],
                "verdict": best_value["verdict"],
            },
            "interpretation": (
                "This local-only scan searches for a lighter single-hypothesis slow-trend gate than the rejected "
                "v21 combo. It is exploratory only and cannot become a QC candidate until accepted baseline raw "
                "trades are restored in the workspace."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local-only adjacent trend scan around the rejected v21 idea.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    result = evaluate(
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_orb_v18_adjacent_trend_scan.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
