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


def build_feature_frame(daily_path: Path) -> pd.DataFrame:
    daily = load_daily_market_frame(daily_path)

    close = daily["Close"]
    daily["prev_day_up"] = close.pct_change() > 0
    daily["mom3_positive"] = (close / close.shift(3) - 1.0) > 0
    daily["mom5_positive"] = (close / close.shift(5) - 1.0) > 0
    daily["mom7_positive"] = (close / close.shift(7) - 1.0) > 0
    daily["mom10_positive"] = (close / close.shift(10) - 1.0) > 0
    daily["close_above_sma5"] = close > close.rolling(5).mean()
    daily["close_above_sma8"] = close > close.rolling(8).mean()
    daily["close_above_sma10"] = close > close.rolling(10).mean()
    daily["close_above_sma15"] = close > close.rolling(15).mean()
    daily["close_above_sma20"] = close > close.rolling(20).mean()
    aligned, _ = align_features_to_next_session(
        daily,
        [
            "prev_day_up",
            "mom3_positive",
            "mom5_positive",
            "mom7_positive",
            "mom10_positive",
            "close_above_sma5",
            "close_above_sma8",
            "close_above_sma10",
            "close_above_sma15",
            "close_above_sma20",
        ],
    )
    return aligned


def load_qc_trades(result_dir: Path) -> tuple[pd.DataFrame, str]:
    bundle = resolve_bundle(result_dir)
    trades = pd.read_csv(bundle.trades_path)
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
    trades["entry_date"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.date
    trades["exit_date"] = trades["Exit Time"].dt.tz_convert("America/New_York").dt.date
    trades["entry_year"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.year
    trades["net_pnl"] = pd.to_numeric(trades["P&L"], errors="coerce").fillna(0.0) - pd.to_numeric(
        trades["Fees"], errors="coerce"
    ).fillna(0.0)
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)
    return trades, bundle.trades_path.name


def qc_stats(subset: pd.DataFrame, calendar_dates: list[object]) -> dict[str, object]:
    if subset.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "positive_years": 0,
            "negative_years": 0,
            "rolling_6m_positive_pct": 0.0,
            "rolling_12m_positive_pct": 0.0,
        }
    by_year = subset.groupby("entry_year")["net_pnl"].sum()
    pf = compute_profit_factor(subset["net_pnl"])
    return {
        "trades": int(len(subset)),
        "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2),
        "profit_factor": round(float(pf), 3) if np.isfinite(pf) else float("inf"),
        "net_pnl": round(float(subset["net_pnl"].sum()), 2),
        "positive_years": int((by_year > 0).sum()),
        "negative_years": int((by_year < 0).sum()),
        "rolling_6m_positive_pct": rolling_summary(calendar_dates, subset, 6)["positive_sharpe_pct"],
        "rolling_12m_positive_pct": rolling_summary(calendar_dates, subset, 12)["positive_sharpe_pct"],
    }


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

    for fold in folds:
        start_text, end_text = [part.strip() for part in str(fold["test_period"]).split("~")]
        start_date = pd.Timestamp(start_text).date()
        end_date = pd.Timestamp(end_text).date()
        merged = merge_features(run_local_backtest(bars, start_date=start_date, end_date=end_date), features)
        baseline = apply_base_filter(merged)
        kept = baseline[baseline[candidate_name].fillna(False)].copy()
        excluded = baseline[~baseline[candidate_name].fillna(False)].copy()

        base_stats = local_stats_for_subset(baseline)
        kept_stats = local_stats_for_subset(kept)
        excluded_stats = local_stats_for_subset(excluded)
        if kept_stats["net_pnl"] > base_stats["net_pnl"]:
            improved += 1
        if kept_stats["net_pnl"] > 0:
            positive_kept += 1
        if excluded_stats["net_pnl"] < 0:
            excluded_negative += 1
        kept_total += kept_stats["net_pnl"]
        baseline_total += base_stats["net_pnl"]

    return {
        "improved_vs_baseline_folds": int(improved),
        "positive_kept_folds": int(positive_kept),
        "excluded_negative_folds": int(excluded_negative),
        "kept_net_pnl": round(float(kept_total), 2),
        "baseline_net_pnl": round(float(baseline_total), 2),
        "delta_vs_baseline": round(float(kept_total - baseline_total), 2),
    }


def local_verdict(candidate: dict[str, object], baseline: dict[str, float]) -> str:
    if (
        candidate["kept"]["net_pnl"] > baseline["net_pnl"]
        and candidate["excluded"]["net_pnl"] < 0
        and candidate["walkforward"]["delta_vs_baseline"] >= 0
    ):
        return "LOCAL_CONFIRMATION"
    if candidate["kept"]["net_pnl"] > 0:
        return "LOCAL_WEAK_OR_MIXED"
    return "LOCAL_NEGATIVE"


def qc_verdict(candidate: dict[str, object], baseline: dict[str, object]) -> str:
    if (
        candidate["kept"]["net_pnl"] > baseline["net_pnl"]
        and candidate["excluded"]["net_pnl"] < 0
        and candidate["kept"]["rolling_6m_positive_pct"] >= baseline["rolling_6m_positive_pct"]
        and candidate["kept"]["rolling_12m_positive_pct"] >= baseline["rolling_12m_positive_pct"]
    ):
        return "QC_STRONG"
    if candidate["kept"]["net_pnl"] > baseline["net_pnl"] and candidate["excluded"]["net_pnl"] < 0:
        return "QC_POSITIVE_BUT_MIXED"
    return "QC_WEAK_OR_MIXED"


def bridge_verdict(qc_result: str, local_result: str) -> str:
    if qc_result == "QC_STRONG" and local_result == "LOCAL_CONFIRMATION":
        return "BRIDGE_CONFIRMED"
    if qc_result == "QC_STRONG" and local_result == "LOCAL_WEAK_OR_MIXED":
        return "QC_ONLY_LEADER"
    if qc_result == "QC_POSITIVE_BUT_MIXED" and local_result == "LOCAL_CONFIRMATION":
        return "LOCAL_AHEAD_OF_QC"
    return "NO_BRIDGE"


def evaluate(
    result_dir: Path,
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    candidates: list[str],
) -> dict[str, object]:
    features = build_feature_frame(daily_csv)
    calendar_dates = sorted(set(features["date"]))
    qc_trades, qc_source_file = load_qc_trades(result_dir)
    qc_merged = qc_trades.merge(features, left_on="entry_date", right_on="date", how="left")
    qc_baseline = qc_stats(qc_merged, calendar_dates)

    bars = load_bars(intraday_csv)
    local_merged = merge_features(run_local_backtest(bars), features)
    local_baseline_frame = apply_base_filter(local_merged)
    local_baseline = local_stats_for_subset(local_baseline_frame)
    folds = parse_folds(walk_forward_json)

    results: dict[str, object] = {}
    for name in candidates:
        qc_kept = qc_merged[qc_merged[name].fillna(False)].copy()
        qc_excluded = qc_merged[~qc_merged[name].fillna(False)].copy()
        local_kept = local_baseline_frame[local_baseline_frame[name].fillna(False)].copy()
        local_excluded = local_baseline_frame[~local_baseline_frame[name].fillna(False)].copy()

        row = {
            "qc": {
                "kept": qc_stats(qc_kept, calendar_dates),
                "excluded": qc_stats(qc_excluded, calendar_dates),
            },
            "local": {
                "kept": local_stats_for_subset(local_kept),
                "excluded": local_stats_for_subset(local_excluded),
                "walkforward": evaluate_local_walkforward(bars, features, folds, name),
            },
        }
        row["qc"]["delta_vs_baseline_net_pnl"] = round(float(row["qc"]["kept"]["net_pnl"] - qc_baseline["net_pnl"]), 2)
        row["local"]["delta_vs_baseline_net_pnl"] = round(
            float(row["local"]["kept"]["net_pnl"] - local_baseline["net_pnl"]), 2
        )
        row["qc_verdict"] = qc_verdict(row["qc"], qc_baseline)
        row["local_verdict"] = local_verdict(row["local"], local_baseline)
        row["bridge_verdict"] = bridge_verdict(row["qc_verdict"], row["local_verdict"])
        results[name] = row

    ranked = sorted(
        results.items(),
        key=lambda item: (
            item[1]["bridge_verdict"] == "BRIDGE_CONFIRMED",
            item[1]["bridge_verdict"] == "QC_ONLY_LEADER",
            item[1]["qc"]["kept"]["net_pnl"],
            item[1]["local"]["walkforward"]["delta_vs_baseline"],
        ),
        reverse=True,
    )

    best_qc = max(results.items(), key=lambda item: item[1]["qc"]["kept"]["net_pnl"])
    confirmed = [item for item in ranked if item[1]["bridge_verdict"] == "BRIDGE_CONFIRMED"]

    return {
        "research_scope": "v18_slow_trend_bridge_family",
        "analysis_version": "v2_qc_local_bridge_family_next_session_alignment",
        "source_qc_trades_file": qc_source_file,
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": BASE_FILTER_LABEL,
        "qc_baseline": qc_baseline,
        "local_baseline": local_baseline,
        "candidates": results,
        "candidate_summary": {
            "best_qc_leader": {
                "label": best_qc[0],
                "qc_kept_net_pnl": best_qc[1]["qc"]["kept"]["net_pnl"],
                "qc_excluded_net_pnl": best_qc[1]["qc"]["excluded"]["net_pnl"],
                "qc_verdict": best_qc[1]["qc_verdict"],
                "local_verdict": best_qc[1]["local_verdict"],
                "bridge_verdict": best_qc[1]["bridge_verdict"],
            },
            "bridge_confirmed_count": len(confirmed),
            "top_bridge_candidates": [
                {
                    "label": name,
                    "qc_verdict": payload["qc_verdict"],
                    "local_verdict": payload["local_verdict"],
                    "bridge_verdict": payload["bridge_verdict"],
                    "qc_kept_net_pnl": payload["qc"]["kept"]["net_pnl"],
                    "local_walkforward_delta": payload["local"]["walkforward"]["delta_vs_baseline"],
                }
                for name, payload in ranked[:5]
            ],
            "interpretation": (
                "This bridge scan searches an adjacent slow-trend family on top of v18. "
                "A candidate is only launch-ready if accepted QC trades and the local v18 sample both point in the same direction."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge accepted QC trades and local v18 sample for a slow-trend factor family.")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument(
        "--candidates",
        default="mom5_positive,mom7_positive,mom10_positive,close_above_sma5,close_above_sma8,close_above_sma10,close_above_sma15,close_above_sma20",
    )
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    result = evaluate(
        result_dir=Path(args.result_dir),
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
        candidates=candidates,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v18_slow_trend_bridge_family.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
