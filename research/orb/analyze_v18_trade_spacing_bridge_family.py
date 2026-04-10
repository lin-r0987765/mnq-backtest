"""Family 17: Trade-Spacing / Regime-Gap gates.

This family tests whether the *number of sessions the regime filter blocked
before allowing each trade* carries predictive power.  It is orthogonal to
all 16 prior families because it uses the regime filter's own blocking
pattern as a signal — not market data, daily features, intraday structure,
trade outcomes, or calendar weekday.

Hypothesis:  When the v18 filter (prev_day_up AND mom3_positive) blocks
many consecutive sessions then allows one, the market may have just
transitioned into a favorable regime, making that first-after-gap trade
higher quality.  Conversely, clustered trades (many consecutive allowed
days) may indicate the filter is too permissive.

Candidates:
  gap_eq_0              – trade immediately follows another trade (consecutive)
  gap_eq_1              – exactly 1 blocked session before this trade
  gap_ge_2              – ≥2 blocked sessions before this trade
  gap_ge_3              – ≥3 blocked sessions (stricter regime transition)
  gap_le_1              – 0-1 blocked sessions (clustered trades)
  cluster_trade         – in a run of ≥3 consecutive allowed sessions
  isolated_trade        – NOT in a run of ≥3 consecutive allowed sessions
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

CANDIDATE_LABELS = [
    "gap_eq_0",
    "gap_eq_1",
    "gap_ge_2",
    "gap_ge_3",
    "gap_le_1",
    "cluster_trade",
    "isolated_trade",
]


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def enrich_trade_spacing(
    trades: pd.DataFrame,
    calendar_dates: list,
    *,
    date_col: str,
) -> pd.DataFrame:
    """Add trade-spacing features using the calendar of all trading sessions."""
    if trades.empty or len(trades) < 2:
        enriched = trades.copy()
        for label in CANDIDATE_LABELS:
            enriched[label] = False
        return enriched

    enriched = trades.copy()

    # Build a set of all trade dates for fast lookup
    trade_dates_sorted = sorted(set(enriched[date_col]))
    calendar_sorted = sorted(set(calendar_dates))

    # Build calendar index for fast position lookup
    cal_index = {d: i for i, d in enumerate(calendar_sorted)}

    # For each trade, compute the gap (number of blocked sessions since last trade)
    trade_date_list = enriched[date_col].tolist()
    gaps = []
    for i, d in enumerate(trade_date_list):
        if i == 0:
            gaps.append(np.nan)  # First trade has no prior reference
            continue
        prev_trade_date = trade_date_list[i - 1]
        prev_pos = cal_index.get(prev_trade_date)
        curr_pos = cal_index.get(d)
        if prev_pos is not None and curr_pos is not None:
            gap = curr_pos - prev_pos - 1  # Number of sessions between
            gaps.append(gap)
        else:
            gaps.append(np.nan)

    enriched["regime_gap"] = gaps

    # Compute cluster runs: identify runs of consecutive trading days
    trade_date_set = set(trade_dates_sorted)
    run_lengths = {}
    # Walk through calendar and find consecutive trade-day runs
    current_run = []
    for d in calendar_sorted:
        if d in trade_date_set:
            current_run.append(d)
        else:
            if current_run:
                run_len = len(current_run)
                for rd in current_run:
                    run_lengths[rd] = run_len
                current_run = []
    if current_run:
        run_len = len(current_run)
        for rd in current_run:
            run_lengths[rd] = run_len

    enriched["trade_run_length"] = enriched[date_col].map(run_lengths).fillna(1)

    gap = pd.to_numeric(enriched["regime_gap"], errors="coerce")
    valid = gap.notna()

    enriched["gap_eq_0"] = valid & (gap == 0)
    enriched["gap_eq_1"] = valid & (gap == 1)
    enriched["gap_ge_2"] = valid & (gap >= 2)
    enriched["gap_ge_3"] = valid & (gap >= 3)
    enriched["gap_le_1"] = valid & (gap <= 1)

    # cluster vs isolated based on consecutive run length
    run_len = pd.to_numeric(enriched["trade_run_length"], errors="coerce")
    enriched["cluster_trade"] = run_len >= 3
    enriched["isolated_trade"] = run_len < 3

    return enriched


def build_local_baseline_frame(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    calendar_dates: list,
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
    # Filter calendar dates to the same window
    if start_date is not None and end_date is not None:
        cal = [d for d in calendar_dates if start_date <= d <= end_date]
    else:
        cal = calendar_dates
    return enrich_trade_spacing(baseline, cal, date_col="date")


def evaluate_local_walkforward(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    calendar_dates: list,
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
            bars, features, calendar_dates,
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

    # Build calendar dates from features
    calendar_dates = sorted(features["date"].tolist())

    local_baseline = build_local_baseline_frame(bars, features, calendar_dates)
    local_baseline_stats = local_stats_for_subset(local_baseline)

    bundle, qc_trades = load_qc_trades(result_dir)
    qc_trades["entry_date"] = pd.to_datetime(qc_trades["Entry Time"], utc=True).dt.tz_convert(NEW_YORK_TZ).dt.date
    qc_merged = qc_trades.merge(features, left_on="entry_date", right_on="date", how="left")
    # Sort by entry time for sequential spacing computation
    qc_merged = qc_merged.sort_values("Entry Time").reset_index(drop=True)
    qc_merged = enrich_trade_spacing(qc_merged, calendar_dates, date_col="entry_date")

    qc_baseline_stats = qc_stats_for_subset(qc_merged, calendar_dates)

    candidates = {}
    for label in CANDIDATE_LABELS:
        local_mask = _bool(local_baseline[label])
        local_kept = local_baseline[local_mask].copy()
        local_excluded = local_baseline[~local_mask].copy()
        local_row = {
            "kept": local_stats_for_subset(local_kept),
            "excluded": local_stats_for_subset(local_excluded),
            "walkforward": evaluate_local_walkforward(bars, features, calendar_dates, folds, label),
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

    return {
        "research_scope": "v18_trade_spacing_bridge_family",
        "analysis_version": "v1_regime_gap_and_trade_clustering",
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
            "best_overall": summarize(best_name, best_row),
            "strongest_qc_only_near_miss": summarize(*strongest_qc_only) if strongest_qc_only else None,
            "strongest_local_only_near_miss": summarize(*strongest_local_only) if strongest_local_only else None,
            "strongest_balanced_near_miss": summarize(*strongest_balanced) if strongest_balanced else None,
            "interpretation": (
                "This family tests whether the regime filter's own blocking pattern carries predictive "
                "power. It measures how many sessions were blocked before each allowed trade (regime gap) "
                "and whether trades in clusters vs isolation perform differently. It is orthogonal to all "
                "prior families because it uses the filter's behavior itself — not market data, daily "
                "features, intraday structure, trade outcomes, or calendar weekday."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze v18 trade-spacing / regime-gap candidates."
    )
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--output", default="results/qc_regime_prototypes/v18_trade_spacing_bridge_family.json")
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
