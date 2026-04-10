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
from research.qc.analyze_qc_regime_prototypes import compute_profit_factor, rolling_summary
from research.qc.analyze_qc_webide_result import resolve_bundle
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"
CANDIDATES = [
    "low_gap_abs",
    "narrow_prev_day_range",
    "atr5_below_atr20",
    "inside_day_prev",
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

    gap_abs = (open_ / prev_close - 1.0).abs()
    day_range_pct = (high - low) / close.replace(0.0, np.nan)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr5 = tr.rolling(5, min_periods=5).mean()
    atr20 = tr.rolling(20, min_periods=20).mean()

    daily["low_gap_abs"] = gap_abs < gap_abs.rolling(60, min_periods=20).median()
    daily["narrow_prev_day_range"] = day_range_pct < day_range_pct.rolling(20, min_periods=10).median()
    daily["atr5_below_atr20"] = atr5 < atr20
    daily["inside_day_prev"] = (high < high.shift(1)) & (low > low.shift(1))

    aligned, calendar_dates = align_features_to_next_session(
        daily,
        ["prev_day_up", "mom3_positive", *CANDIDATES],
    )
    return aligned, calendar_dates


def evaluate_local_walkforward(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    folds: list[dict[str, object]],
    candidate_name: str,
) -> dict[str, object]:
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

        baseline_stats = local_stats_for_subset(baseline)
        kept_stats = local_stats_for_subset(kept)
        excluded_stats = local_stats_for_subset(excluded)

        if kept_stats["net_pnl"] > baseline_stats["net_pnl"]:
            improved += 1
        if kept_stats["net_pnl"] > 0:
            positive_kept += 1
        if excluded_stats["net_pnl"] < 0:
            excluded_negative += 1

        kept_total += kept_stats["net_pnl"]
        baseline_total += baseline_stats["net_pnl"]

        rows.append(
            {
                "fold": int(fold["fold"]),
                "test_period": str(fold["test_period"]),
                "baseline": baseline_stats,
                "kept": kept_stats,
                "excluded": excluded_stats,
                "delta_vs_baseline": round(float(kept_stats["net_pnl"] - baseline_stats["net_pnl"]), 2),
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


def local_verdict(candidate: dict[str, object], baseline: dict[str, float]) -> str:
    kept = candidate["kept"]
    excluded = candidate["excluded"]
    walk = candidate["walkforward"]["summary"]
    if (
        kept["net_pnl"] > baseline["net_pnl"]
        and excluded["net_pnl"] < 0
        and walk["improved_vs_baseline_folds"] >= 2
    ):
        return "LOCAL_RESEARCH_LEADER"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0:
        return "LOCAL_PROMISING"
    return "LOCAL_WEAK_OR_MIXED"


def load_qc_trades(result_dir: Path) -> tuple[object, pd.DataFrame]:
    bundle = resolve_bundle(result_dir)
    trades = pd.read_csv(bundle.trades_path)
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
    trades["entry_date"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.date
    trades["entry_year"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.year
    trades["exit_date"] = trades["Exit Time"].dt.tz_convert("America/New_York").dt.date
    trades["net_pnl"] = pd.to_numeric(trades["P&L"], errors="coerce").fillna(0.0) - pd.to_numeric(
        trades["Fees"], errors="coerce"
    ).fillna(0.0)
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)
    return bundle, trades


def qc_stats_for_subset(subset: pd.DataFrame, calendar_dates: list[object]) -> dict[str, object]:
    if subset.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "positive_years": 0,
            "negative_years": 0,
            "year_net_pnl": {},
            "rolling_6m": rolling_summary(calendar_dates, subset, 6),
            "rolling_12m": rolling_summary(calendar_dates, subset, 12),
        }

    pf = compute_profit_factor(subset["net_pnl"])
    by_year = subset.groupby("entry_year")["net_pnl"].sum()
    return {
        "trades": int(len(subset)),
        "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2),
        "profit_factor": round(float(pf), 3) if np.isfinite(pf) else float("inf"),
        "net_pnl": round(float(subset["net_pnl"].sum()), 2),
        "avg_trade_pnl": round(float(subset["net_pnl"].mean()), 2),
        "positive_years": int((by_year > 0).sum()),
        "negative_years": int((by_year < 0).sum()),
        "year_net_pnl": {str(int(year)): round(float(value), 2) for year, value in by_year.items()},
        "rolling_6m": rolling_summary(calendar_dates, subset, 6),
        "rolling_12m": rolling_summary(calendar_dates, subset, 12),
    }


def qc_verdict(candidate: dict[str, object], baseline: dict[str, object]) -> str:
    kept = candidate["kept"]
    excluded = candidate["excluded"]
    delta_6m = candidate["delta_vs_baseline_6m_positive_pct"]
    delta_12m = candidate["delta_vs_baseline_12m_positive_pct"]
    if (
        kept["net_pnl"] > baseline["net_pnl"]
        and excluded["net_pnl"] < 0
        and kept["positive_years"] >= baseline["positive_years"]
        and delta_12m >= 0.0
        and delta_6m >= -2.0
    ):
        return "QC_PROXY_FRONT_RUNNER"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0:
        return "QC_PROXY_PROMISING"
    return "QC_PROXY_WEAK_OR_MIXED"


def combined_verdict(local_state: str, qc_state: str) -> str:
    if local_state == "LOCAL_RESEARCH_LEADER" and qc_state == "QC_PROXY_FRONT_RUNNER":
        return "ORTHOGONAL_RESEARCH_LEADER"
    if local_state in {"LOCAL_RESEARCH_LEADER", "LOCAL_PROMISING"} and qc_state in {
        "QC_PROXY_FRONT_RUNNER",
        "QC_PROXY_PROMISING",
    }:
        return "ORTHOGONAL_PROMISING"
    return "ORTHOGONAL_WEAK_OR_MIXED"


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
            float(
                qc_row["kept"]["rolling_12m"]["positive_sharpe_pct"] - qc_baseline_stats["rolling_12m"]["positive_sharpe_pct"]
            ),
            1,
        )
        qc_row["verdict"] = qc_verdict(qc_row, qc_baseline_stats)

        combo = combined_verdict(local_row["verdict"], qc_row["verdict"])
        candidates[name] = {
            "label": name,
            "local": local_row,
            "qc_proxy": qc_row,
            "combined_verdict": combo,
        }

    best = max(
        candidates.values(),
        key=lambda item: (
            item["combined_verdict"] == "ORTHOGONAL_RESEARCH_LEADER",
            item["combined_verdict"] == "ORTHOGONAL_PROMISING",
            item["qc_proxy"]["kept"]["net_pnl"],
            item["local"]["walkforward"]["summary"]["delta_vs_baseline"],
        ),
    )

    return {
        "research_scope": "v18_calmness_bridge_family",
        "analysis_version": "v1_post_v22_orthogonal_daily_calmness_scan",
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "source_qc_bundle": bundle.json_path.stem,
        "base_filter": BASE_FILTER_LABEL,
        "local_baseline": local_baseline_stats,
        "qc_baseline": qc_baseline_stats,
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": {
                "label": best["label"],
                "combined_verdict": best["combined_verdict"],
                "local_verdict": best["local"]["verdict"],
                "qc_verdict": best["qc_proxy"]["verdict"],
                "local_walkforward_delta_vs_baseline": best["local"]["walkforward"]["summary"]["delta_vs_baseline"],
                "qc_delta_vs_baseline_net_pnl": best["qc_proxy"]["delta_vs_baseline_net_pnl"],
                "qc_delta_vs_baseline_6m_positive_pct": best["qc_proxy"]["delta_vs_baseline_6m_positive_pct"],
                "qc_delta_vs_baseline_12m_positive_pct": best["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
            },
            "interpretation": (
                "This scan intentionally pivots away from the fragile adjacent slow-trend branch that produced rejected "
                "v22. It searches orthogonal single-hypothesis calmness/compression filters before any future QC "
                "candidate launch."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge-scan orthogonal daily calmness families after the rejected v22 branch.")
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
    out_path = out_dir / "v18_calmness_bridge_family.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
