from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.orb.analyze_local_orb_v18_return_cap import (
    apply_base_filter,
    load_bars,
    merge_features,
    parse_folds,
    run_local_backtest,
    stats_for_subset as local_stats_for_subset,
)
from research.orb.analyze_v18_calmness_bridge_family import (
    combined_verdict,
    evaluate_local_walkforward,
    load_qc_trades,
    local_verdict,
    qc_stats_for_subset,
    qc_verdict,
)
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"
CANDIDATES = [
    "range_expansion_prev",
    "bull_displacement_prev",
    "marubozu_bull_prev",
    "outside_up_close_high_prev",
    "large_body_prev",
]


def build_feature_frame(daily_csv: Path) -> tuple[pd.DataFrame, list[object]]:
    daily = load_daily_market_frame(daily_csv)
    close = daily["Close"].astype(float)
    open_ = daily["Open"].astype(float)
    high = daily["High"].astype(float)
    low = daily["Low"].astype(float)
    prev_close = close.shift(1)

    daily["prev_day_up"] = close.pct_change() > 0.0
    daily["mom3_positive"] = close.pct_change(3) > 0.0

    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr20 = true_range.rolling(20, min_periods=20).mean()

    day_range = (high - low).replace(0.0, np.nan)
    body_ratio = (close - open_).abs() / day_range
    close_location = (close - low) / day_range

    daily["range_expansion_prev"] = true_range > atr20
    daily["bull_displacement_prev"] = (
        (close > open_)
        & (true_range > atr20)
        & (close_location >= 0.67)
    )
    daily["marubozu_bull_prev"] = (
        (close > open_)
        & (body_ratio >= 0.70)
        & (close_location >= 0.80)
    )
    daily["outside_up_close_high_prev"] = (
        (high > high.shift(1))
        & (low < low.shift(1))
        & (close > open_)
        & (close_location >= 0.67)
    )
    daily["large_body_prev"] = body_ratio >= body_ratio.rolling(60, min_periods=20).quantile(0.70)

    aligned, calendar_dates = align_features_to_next_session(
        daily,
        ["prev_day_up", "mom3_positive", *CANDIDATES],
    )
    return aligned, calendar_dates


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    result_dir: Path,
) -> dict[str, object]:
    features, calendar_dates = build_feature_frame(daily_csv)

    bars = load_bars(intraday_csv)
    folds = parse_folds(walk_forward_json)
    local_merged = merge_features(run_local_backtest(bars), features)
    local_baseline = apply_base_filter(local_merged)
    local_baseline_stats = local_stats_for_subset(local_baseline)

    bundle, qc_trades = load_qc_trades(result_dir)
    qc_merged = qc_trades.merge(features, left_on="entry_date", right_on="date", how="left")
    qc_baseline_stats = qc_stats_for_subset(qc_merged, calendar_dates)

    candidates: dict[str, object] = {}
    for name in CANDIDATES:
        local_kept = local_baseline[local_baseline[name].fillna(False)].copy()
        local_excluded = local_baseline[~local_baseline[name].fillna(False)].copy()
        local_row = {
            "kept": local_stats_for_subset(local_kept),
            "excluded": local_stats_for_subset(local_excluded),
            "walkforward": evaluate_local_walkforward(bars, features, folds, name),
        }
        local_row["verdict"] = local_verdict(local_row, local_baseline_stats)

        qc_kept = qc_merged[qc_merged[name].fillna(False)].copy()
        qc_excluded = qc_merged[~qc_merged[name].fillna(False)].copy()
        qc_row = {
            "kept": qc_stats_for_subset(qc_kept, calendar_dates),
            "excluded": qc_stats_for_subset(qc_excluded, calendar_dates),
        }
        qc_row["delta_vs_baseline_net_pnl"] = round(float(qc_row["kept"]["net_pnl"] - qc_baseline_stats["net_pnl"]), 2)
        qc_row["delta_vs_baseline_6m_positive_pct"] = round(
            float(qc_row["kept"]["rolling_6m"]["positive_sharpe_pct"] - qc_baseline_stats["rolling_6m"]["positive_sharpe_pct"]),
            1,
        )
        qc_row["delta_vs_baseline_12m_positive_pct"] = round(
            float(qc_row["kept"]["rolling_12m"]["positive_sharpe_pct"] - qc_baseline_stats["rolling_12m"]["positive_sharpe_pct"]),
            1,
        )
        qc_row["verdict"] = qc_verdict(qc_row, qc_baseline_stats)

        combined = combined_verdict(local_row["verdict"], qc_row["verdict"])
        candidates[name] = {
            "local": local_row,
            "qc_proxy": qc_row,
            "combined_verdict": combined,
        }

    best_name, best_row = max(
        candidates.items(),
        key=lambda item: (
            item[1]["combined_verdict"] == "ORTHOGONAL_RESEARCH_LEADER",
            item[1]["combined_verdict"] == "ORTHOGONAL_PROMISING",
            item[1]["qc_proxy"]["kept"]["net_pnl"],
            item[1]["local"]["walkforward"]["summary"]["delta_vs_baseline"],
        ),
    )

    return {
        "research_scope": "v18_displacement_bridge_family",
        "analysis_version": "v1_daily_displacement_and_candle_anatomy",
        "source_bundle": bundle.json_path.stem,
        "source_trades_csv": bundle.trades_path.name,
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": BASE_FILTER_LABEL,
        "baseline": {
            "local": local_baseline_stats,
            "qc_proxy": qc_baseline_stats,
        },
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": {
                "label": best_name,
                "combined_verdict": best_row["combined_verdict"],
                "local_verdict": best_row["local"]["verdict"],
                "qc_verdict": best_row["qc_proxy"]["verdict"],
                "local_walkforward_delta_vs_baseline": best_row["local"]["walkforward"]["summary"]["delta_vs_baseline"],
                "qc_delta_vs_baseline_net_pnl": best_row["qc_proxy"]["delta_vs_baseline_net_pnl"],
                "qc_delta_vs_baseline_6m_positive_pct": best_row["qc_proxy"]["delta_vs_baseline_6m_positive_pct"],
                "qc_delta_vs_baseline_12m_positive_pct": best_row["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
            },
            "interpretation": (
                "This family tests whether v18 benefits from prior-day displacement / candle-anatomy filters. "
                "It is intended as a fresh orthogonal branch after both adjacent slow-trend and calmness/compression "
                "families failed."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate daily displacement / candle-anatomy bridge candidates on top of v18.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    result = evaluate(
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
        result_dir=Path(args.result_dir),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v18_displacement_bridge_family.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
