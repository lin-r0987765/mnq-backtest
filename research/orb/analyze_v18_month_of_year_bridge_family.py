"""Family 18: Month-of-Year / Quarter Seasonality gates.

This family tests whether the ORB strategy's performance varies by calendar
month or quarter.  It is orthogonal to all 17 prior families because it
uses only the *calendar month/quarter* of the trading session — not any
price-derived signal, intraday feature, trade history, or weekday.

Note: `skip_feb_mar` was previously disqualified as a research trap
(cherry-picked months). This family is different because it tests a
comprehensive set of *all* monthly/quarterly groups to identify any
broad seasonal pattern, not target specific months.

Candidates:
  q1_jan_mar            – trade only in January through March
  q2_apr_jun            – trade only in April through June
  q3_jul_sep            – trade only in July through September
  q4_oct_dec            – trade only in October through December
  skip_q1               – trade every quarter except Q1
  skip_q3               – trade every quarter except Q3 (summer doldrums)
  h1_jan_jun            – trade only in first half of year
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.orb.analyze_local_orb_v18_return_cap import (
    apply_base_filter,
    build_feature_frame,
    load_bars,
    merge_features,
    parse_folds,
    run_local_backtest,
    stats_for_subset as local_stats_for_subset,
)
from research.orb.analyze_v18_calmness_bridge_family import (
    combined_verdict,
    load_qc_trades,
    local_verdict,
    qc_stats_for_subset,
    qc_verdict,
)
from daily_session_alignment import NEW_YORK_TZ

BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"

CANDIDATE_SPECS = {
    "q1_jan_mar":   {"months": {1, 2, 3}},
    "q2_apr_jun":   {"months": {4, 5, 6}},
    "q3_jul_sep":   {"months": {7, 8, 9}},
    "q4_oct_dec":   {"months": {10, 11, 12}},
    "skip_q1":      {"months": {4, 5, 6, 7, 8, 9, 10, 11, 12}},
    "skip_q3":      {"months": {1, 2, 3, 4, 5, 6, 10, 11, 12}},
    "h1_jan_jun":   {"months": {1, 2, 3, 4, 5, 6}},
}

CANDIDATE_LABELS = list(CANDIDATE_SPECS.keys())


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def enrich_month_of_year(
    trades: pd.DataFrame,
    *,
    date_col: str,
) -> pd.DataFrame:
    """Add month-of-year boolean columns to a trade DataFrame."""
    if trades.empty:
        enriched = trades.copy()
        for label in CANDIDATE_LABELS:
            enriched[label] = False
        return enriched

    enriched = trades.copy()
    dates = enriched[date_col]
    if hasattr(dates.iloc[0], "month"):
        month = dates.apply(lambda d: d.month)
    else:
        month = pd.to_datetime(dates).dt.month

    for label, spec in CANDIDATE_SPECS.items():
        enriched[label] = month.isin(spec["months"])

    return enriched


def build_local_baseline_frame(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    trades = merge_features(
        run_local_backtest(bars, start_date=start_date, end_date=end_date),
        features,
    )
    baseline = apply_base_filter(trades)
    if baseline.empty:
        return baseline
    return enrich_month_of_year(baseline, date_col="date")


def evaluate_local_walkforward(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    folds: list[dict],
    candidate_name: str,
) -> dict:
    improved = 0
    positive_kept = 0
    excluded_negative = 0
    kept_total = 0.0
    baseline_total = 0.0
    rows = []

    for fold in folds:
        start_text, end_text = [p.strip() for p in str(fold["test_period"]).split("~")]
        baseline = build_local_baseline_frame(
            bars, features,
            start_date=pd.Timestamp(start_text).date(),
            end_date=pd.Timestamp(end_text).date(),
        )
        kept = baseline[_bool(baseline[candidate_name])].copy()
        excluded = baseline[~_bool(baseline[candidate_name])].copy()

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

        rows.append({
            "fold": int(fold["fold"]),
            "test_period": str(fold["test_period"]),
            "baseline": baseline_stats,
            "kept": kept_stats,
            "excluded": excluded_stats,
            "delta_vs_baseline": round(float(kept_stats["net_pnl"] - baseline_stats["net_pnl"]), 2),
        })

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


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    result_dir: Path,
) -> dict:
    features = build_feature_frame(daily_csv)
    bars = load_bars(intraday_csv)
    folds = parse_folds(walk_forward_json)

    local_baseline = build_local_baseline_frame(bars, features)
    local_baseline_stats = local_stats_for_subset(local_baseline)

    bundle, qc_trades = load_qc_trades(result_dir)
    qc_trades["entry_date"] = pd.to_datetime(qc_trades["Entry Time"], utc=True).dt.tz_convert(NEW_YORK_TZ).dt.date
    qc_merged = qc_trades.merge(features, left_on="entry_date", right_on="date", how="left")
    qc_merged = enrich_month_of_year(qc_merged, date_col="entry_date")

    calendar_dates = list(features["date"])
    qc_baseline_stats = qc_stats_for_subset(qc_merged, calendar_dates)

    candidates = {}
    for label in CANDIDATE_LABELS:
        local_mask = _bool(local_baseline[label])
        local_kept = local_baseline[local_mask].copy()
        local_excluded = local_baseline[~local_mask].copy()
        local_row = {
            "kept": local_stats_for_subset(local_kept),
            "excluded": local_stats_for_subset(local_excluded),
            "walkforward": evaluate_local_walkforward(bars, features, folds, label),
        }
        local_row["verdict"] = local_verdict(local_row, local_baseline_stats)

        qc_mask = _bool(qc_merged[label])
        qc_kept = qc_merged[qc_mask].copy()
        qc_excluded = qc_merged[~qc_mask].copy()
        qc_row = {
            "kept": qc_stats_for_subset(qc_kept, calendar_dates),
            "excluded": qc_stats_for_subset(qc_excluded, calendar_dates),
        }
        qc_row["delta_vs_baseline_net_pnl"] = round(float(qc_row["kept"]["net_pnl"] - qc_baseline_stats["net_pnl"]), 2)
        qc_row["delta_vs_baseline_6m_positive_pct"] = round(
            float(qc_row["kept"]["rolling_6m"]["positive_sharpe_pct"] - qc_baseline_stats["rolling_6m"]["positive_sharpe_pct"]), 1,
        )
        qc_row["delta_vs_baseline_12m_positive_pct"] = round(
            float(qc_row["kept"]["rolling_12m"]["positive_sharpe_pct"] - qc_baseline_stats["rolling_12m"]["positive_sharpe_pct"]), 1,
        )
        qc_row["verdict"] = qc_verdict(qc_row, qc_baseline_stats)

        candidates[label] = {
            "spec": label,
            "local": local_row,
            "qc_proxy": qc_row,
            "combined_verdict": combined_verdict(local_row["verdict"], qc_row["verdict"]),
        }

    def rank_key(item):
        row = item[1]
        combined_rank = {
            "ORTHOGONAL_RESEARCH_LEADER": 2,
            "ORTHOGONAL_PROMISING": 1,
            "ORTHOGONAL_WEAK_OR_MIXED": 0,
        }[row["combined_verdict"]]
        return (
            combined_rank,
            row["qc_proxy"]["delta_vs_baseline_net_pnl"],
            row["local"]["walkforward"]["summary"]["delta_vs_baseline"],
        )

    def summarize(label, row):
        return {
            "label": label,
            "spec": row["spec"],
            "combined_verdict": row["combined_verdict"],
            "local_verdict": row["local"]["verdict"],
            "qc_verdict": row["qc_proxy"]["verdict"],
            "local_walkforward_delta_vs_baseline": row["local"]["walkforward"]["summary"]["delta_vs_baseline"],
            "local_total_delta_vs_baseline": round(float(row["local"]["kept"]["net_pnl"] - local_baseline_stats["net_pnl"]), 2),
            "qc_delta_vs_baseline_net_pnl": row["qc_proxy"]["delta_vs_baseline_net_pnl"],
            "qc_delta_vs_baseline_6m_positive_pct": row["qc_proxy"]["delta_vs_baseline_6m_positive_pct"],
            "qc_delta_vs_baseline_12m_positive_pct": row["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
        }

    best_name, best_row = max(candidates.items(), key=rank_key)

    strongest_qc_only = max(
        (
            (label, row) for label, row in candidates.items()
            if row["qc_proxy"]["delta_vs_baseline_net_pnl"] > 0
            and row["local"]["walkforward"]["summary"]["delta_vs_baseline"] <= 0
        ),
        key=lambda item: (
            item[1]["qc_proxy"]["delta_vs_baseline_net_pnl"],
            item[1]["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
        ),
        default=None,
    )
    strongest_local_only = max(
        (
            (label, row) for label, row in candidates.items()
            if row["local"]["walkforward"]["summary"]["delta_vs_baseline"] > 0
            and row["qc_proxy"]["delta_vs_baseline_net_pnl"] <= 0
        ),
        key=lambda item: (
            item[1]["local"]["walkforward"]["summary"]["delta_vs_baseline"],
            item[1]["qc_proxy"]["delta_vs_baseline_net_pnl"],
        ),
        default=None,
    )
    strongest_balanced = max(
        (
            (label, row) for label, row in candidates.items()
            if row["local"]["walkforward"]["summary"]["delta_vs_baseline"] > 0
            and row["qc_proxy"]["delta_vs_baseline_net_pnl"] > 0
        ),
        key=lambda item: (
            item[1]["qc_proxy"]["delta_vs_baseline_net_pnl"],
            item[1]["local"]["walkforward"]["summary"]["delta_vs_baseline"],
        ),
        default=None,
    )

    # Per-quarter QC breakdown for interpretation
    per_quarter_pnl = {}
    for qname in ["q1_jan_mar", "q2_apr_jun", "q3_jul_sep", "q4_oct_dec"]:
        qmask = _bool(qc_merged[qname])
        per_quarter_pnl[qname] = {
            "trades": int(qmask.sum()),
            "net_pnl": round(float(qc_merged.loc[qmask, "net_pnl"].sum()), 2),
        }

    return {
        "research_scope": "v18_month_of_year_bridge_family",
        "analysis_version": "v1_quarterly_and_half_year_seasonality",
        "source_bundle": bundle.json_path.stem,
        "source_trades_csv": bundle.trades_path.name,
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": BASE_FILTER_LABEL,
        "per_quarter_qc_breakdown": per_quarter_pnl,
        "baseline": {
            "local": local_baseline_stats,
            "qc_proxy": qc_baseline_stats,
        },
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": summarize(best_name, best_row),
            "strongest_qc_only_near_miss": summarize(*strongest_qc_only) if strongest_qc_only else None,
            "strongest_local_only_near_miss": summarize(*strongest_local_only) if strongest_local_only else None,
            "strongest_balanced_near_miss": summarize(*strongest_balanced) if strongest_balanced else None,
            "interpretation": (
                "This family tests whether the ORB strategy's performance varies by calendar month or "
                "quarter. It is orthogonal to all prior families because it uses only the month/quarter "
                "of the trading session — not any price-derived signal, intraday feature, trade history, "
                "or weekday. This differs from the disqualified skip_feb_mar hypothesis because it "
                "comprehensively tests all quarterly/half-year groups rather than cherry-picking months."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze v18 month-of-year / quarter seasonality candidates."
    )
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--output", default="results/qc_regime_prototypes/v18_month_of_year_bridge_family.json")
    args = parser.parse_args()

    result = evaluate(
        Path(args.intraday_csv),
        Path(args.daily_csv),
        Path(args.walk_forward_json),
        Path(args.result_dir),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
