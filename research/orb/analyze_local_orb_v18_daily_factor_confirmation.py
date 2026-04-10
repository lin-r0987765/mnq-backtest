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
from research.shared.analyze_single_factor_robustness import build_feature_frame


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"


def monthly_stats(subset: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if subset.empty:
        return out
    for month, part in subset.groupby("entry_month"):
        out[str(month)] = stats_for_subset(part)
    return out


def evaluate_walkforward(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    folds: list[dict[str, object]],
    candidate_name: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    improved = 0
    positive_kept = 0
    excluded_negative = 0
    kept_total = 0.0
    baseline_total = 0.0

    for fold in folds:
        start_text, end_text = [part.strip() for part in str(fold["test_period"]).split("~")]
        start_date = pd.Timestamp(start_text).date()
        end_date = pd.Timestamp(end_text).date()
        merged = merge_features(run_local_backtest(bars, start_date=start_date, end_date=end_date), features)
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
        and walk["improved_vs_baseline_folds"] >= 2
        and walk["positive_kept_folds"] >= 2
        and loo["all_positive"]
    ):
        return "LOCAL_CONFIRMATION"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0:
        return "LOCAL_POSITIVE_BUT_MIXED"
    return "LOCAL_WEAK_OR_MIXED"


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    candidates: list[str],
) -> dict[str, object]:
    bars = load_bars(intraday_csv)
    features, _ = build_feature_frame(daily_csv)
    folds = parse_folds(walk_forward_json)

    merged = merge_features(run_local_backtest(bars), features)
    baseline = apply_base_filter(merged)
    baseline_stats = stats_for_subset(baseline)

    results: dict[str, object] = {}
    for name in candidates:
        kept = baseline[baseline[name].fillna(False)].copy()
        excluded = baseline[~baseline[name].fillna(False)].copy()
        candidate = {
            "kept": stats_for_subset(kept),
            "excluded": stats_for_subset(excluded),
            "monthly_kept": monthly_stats(kept),
            "leave_one_month_out": leave_one_month_out(kept),
            "walkforward": evaluate_walkforward(bars, features, folds, name),
        }
        candidate["verdict"] = verdict(candidate, baseline_stats)
        results[name] = candidate

    return {
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": BASE_FILTER_LABEL,
        "baseline": baseline_stats,
        "candidates": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Confirm adjacent slow-trend daily factors on top of local v18 baseline.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--candidates", default="mom10_positive,close_above_sma10,close_above_sma20")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    result = evaluate(
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
        candidates=candidates,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_orb_v18_daily_factor_confirmation.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
