"""Family 15: Trade-Autocorrelation / Recent-Performance gates.

This family tests whether the *strategy's own recent track record* can
improve entry selection.  It is fundamentally different from all prior
families because it uses the *trade sequence itself* as a filter signal —
not market data, daily features, intraday structure, or timing.

Hypothesis: Many strategies exhibit serial autocorrelation in returns.
If recent baseline trades lost money, the regime may be hostile and
skipping sessions after streaks of losses (or wins) could improve
overall performance.

Candidates:
  prev_trade_winner       – only trade if the most recent baseline trade won
  prev_2_of_3_winners     – only trade if ≥2 of last 3 baseline trades won
  recent_3_trades_net_pos – only trade if last 3 baseline trades had positive
                            cumulative PnL
  no_loss_streak_2        – skip entry if the last 2 baseline trades both lost
  no_loss_streak_3        – skip entry if the last 3 baseline trades all lost
  recent_5_winrate_ge_50  – only trade if ≥3 of last 5 baseline trades won
  recent_5_net_positive   – only trade if last 5 baseline trades had positive
                            cumulative PnL

Note: For the QC proxy, we apply these filters to the QC trade sequence
directly.  For local, we apply to the local trade sequence.
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
    "prev_trade_winner",
    "prev_2_of_3_winners",
    "recent_3_trades_net_pos",
    "no_loss_streak_2",
    "no_loss_streak_3",
    "recent_5_winrate_ge_50",
    "recent_5_net_positive",
]


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def enrich_trade_autocorrelation(
    trades: pd.DataFrame,
    *,
    pnl_col: str,
    is_win_col: str,
) -> pd.DataFrame:
    """Add trade-autocorrelation boolean columns using sequential trade history."""
    if trades.empty or len(trades) < 2:
        enriched = trades.copy()
        for label in CANDIDATE_LABELS:
            enriched[label] = False
        return enriched

    enriched = trades.copy()
    pnl = pd.to_numeric(enriched[pnl_col], errors="coerce").fillna(0.0)
    is_win = pd.to_numeric(enriched[is_win_col], errors="coerce").fillna(0).astype(int)

    # prev_trade_winner: previous baseline trade was a win
    enriched["prev_trade_winner"] = is_win.shift(1).fillna(0).astype(bool)

    # prev_2_of_3_winners: ≥2 of last 3 trades were winners
    rolling_3_wins = is_win.shift(1).rolling(3, min_periods=3).sum()
    enriched["prev_2_of_3_winners"] = (rolling_3_wins >= 2).fillna(False)

    # recent_3_trades_net_pos: last 3 trades cumulative PnL > 0
    rolling_3_pnl = pnl.shift(1).rolling(3, min_periods=3).sum()
    enriched["recent_3_trades_net_pos"] = (rolling_3_pnl > 0).fillna(False)

    # no_loss_streak_2: NOT (last 2 trades both lost)
    prev1_loss = (is_win.shift(1) == 0)
    prev2_loss = (is_win.shift(2) == 0)
    enriched["no_loss_streak_2"] = ~(prev1_loss & prev2_loss)
    # Ensure first 2 trades default to True (no streak yet)
    enriched.loc[enriched.index[:2], "no_loss_streak_2"] = True

    # no_loss_streak_3: NOT (last 3 trades all lost)
    prev3_loss = (is_win.shift(3) == 0)
    enriched["no_loss_streak_3"] = ~(prev1_loss & prev2_loss & prev3_loss)
    enriched.loc[enriched.index[:3], "no_loss_streak_3"] = True

    # recent_5_winrate_ge_50: ≥3 of last 5 trades won
    rolling_5_wins = is_win.shift(1).rolling(5, min_periods=5).sum()
    enriched["recent_5_winrate_ge_50"] = (rolling_5_wins >= 3).fillna(False)

    # recent_5_net_positive: last 5 trades cumulative PnL > 0
    rolling_5_pnl = pnl.shift(1).rolling(5, min_periods=5).sum()
    enriched["recent_5_net_positive"] = (rolling_5_pnl > 0).fillna(False)

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
    return enrich_trade_autocorrelation(
        baseline,
        pnl_col="net_pnl",
        is_win_col="is_win_net",
    )


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

    # Enrich QC trades with trade-autocorrelation features
    # Sort by entry time to ensure sequential ordering
    qc_merged = qc_merged.sort_values("Entry Time").reset_index(drop=True)
    qc_merged = enrich_trade_autocorrelation(
        qc_merged,
        pnl_col="net_pnl",
        is_win_col="is_win_net",
    )

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

    return {
        "research_scope": "v18_trade_autocorrelation_bridge_family",
        "analysis_version": "v1_sequential_trade_performance_gates",
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
                "This family tests whether the strategy's own recent trade sequence carries predictive "
                "signal for future entry quality. It is orthogonal to all prior families because it uses "
                "the trade outcome history itself as a filter — not market data, daily features, intraday "
                "structure, or timing. It tests serial autocorrelation in strategy returns at the trade level."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze v18 trade-autocorrelation / recent-performance candidates."
    )
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--output", default="results/qc_regime_prototypes/v18_trade_autocorrelation_bridge_family.json")
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
