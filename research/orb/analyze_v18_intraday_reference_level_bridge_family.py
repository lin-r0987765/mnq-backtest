from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    load_qc_trades,
    local_verdict,
    qc_stats_for_subset,
    qc_verdict,
)
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"
CANDIDATES = [
    "entry_above_prev_close",
    "entry_above_prev_close_0p10pct",
    "entry_above_prev_high",
    "entry_above_prev_high_0p10pct",
    "entry_at_or_below_prev_close",
    "entry_at_or_below_prev_high",
    "entry_inside_prev_range",
]


def _bool_mask(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def build_feature_frame(daily_csv: Path) -> tuple[pd.DataFrame, list[object]]:
    daily = load_daily_market_frame(daily_csv)
    close = daily["Close"].astype(float)
    high = daily["High"].astype(float)
    low = daily["Low"].astype(float)

    daily["prev_day_up"] = close.pct_change() > 0.0
    daily["mom3_positive"] = close.pct_change(3) > 0.0
    daily["prev_close"] = close
    daily["prev_high"] = high
    daily["prev_low"] = low
    daily["prev_mid"] = (high + low) / 2.0

    aligned, calendar_dates = align_features_to_next_session(
        daily,
        ["prev_day_up", "mom3_positive", "prev_close", "prev_high", "prev_low", "prev_mid"],
    )
    return aligned, calendar_dates


def enrich_reference_state(trades: pd.DataFrame, *, entry_price_col: str) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    enriched = trades.copy()
    entry_price = pd.to_numeric(enriched[entry_price_col], errors="coerce")
    prev_close = pd.to_numeric(enriched["prev_close"], errors="coerce")
    prev_high = pd.to_numeric(enriched["prev_high"], errors="coerce")
    prev_low = pd.to_numeric(enriched["prev_low"], errors="coerce")

    enriched["entry_price"] = entry_price
    valid = entry_price.notna() & prev_close.notna() & prev_high.notna() & prev_low.notna()

    enriched["entry_above_prev_close"] = valid & (entry_price > prev_close)
    enriched["entry_above_prev_close_0p10pct"] = valid & (entry_price > prev_close * 1.001)
    enriched["entry_above_prev_high"] = valid & (entry_price > prev_high)
    enriched["entry_above_prev_high_0p10pct"] = valid & (entry_price > prev_high * 1.001)
    enriched["entry_at_or_below_prev_close"] = valid & (entry_price <= prev_close)
    enriched["entry_at_or_below_prev_high"] = valid & (entry_price <= prev_high)
    enriched["entry_inside_prev_range"] = valid & (entry_price >= prev_low) & (entry_price <= prev_high)
    return enriched


def build_local_baseline_frame(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    trades = merge_features(run_local_backtest(bars, start_date=start_date, end_date=end_date), features)
    baseline = apply_base_filter(trades)
    return enrich_reference_state(baseline, entry_price_col="Avg Entry Price")


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
        baseline = build_local_baseline_frame(
            bars,
            features,
            start_date=pd.Timestamp(start_text).date(),
            end_date=pd.Timestamp(end_text).date(),
        )
        kept = baseline[_bool_mask(baseline[candidate_name])].copy()
        excluded = baseline[~_bool_mask(baseline[candidate_name])].copy()

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


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    result_dir: Path,
) -> dict[str, object]:
    features, calendar_dates = build_feature_frame(daily_csv)

    bars = load_bars(intraday_csv)
    folds = parse_folds(walk_forward_json)
    local_baseline = build_local_baseline_frame(bars, features)
    local_baseline_stats = local_stats_for_subset(local_baseline)

    bundle, qc_trades = load_qc_trades(result_dir)
    qc_merged = qc_trades.merge(features, left_on="entry_date", right_on="date", how="left")
    qc_merged = enrich_reference_state(qc_merged, entry_price_col="Entry Price")
    qc_baseline_stats = qc_stats_for_subset(qc_merged, calendar_dates)

    candidates: dict[str, object] = {}
    for name in CANDIDATES:
        local_kept = local_baseline[_bool_mask(local_baseline[name])].copy()
        local_excluded = local_baseline[~_bool_mask(local_baseline[name])].copy()
        local_row = {
            "kept": local_stats_for_subset(local_kept),
            "excluded": local_stats_for_subset(local_excluded),
            "walkforward": evaluate_local_walkforward(bars, features, folds, name),
        }
        local_row["verdict"] = local_verdict(local_row, local_baseline_stats)

        qc_kept = qc_merged[_bool_mask(qc_merged[name])].copy()
        qc_excluded = qc_merged[~_bool_mask(qc_merged[name])].copy()
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
                qc_row["kept"]["rolling_12m"]["positive_sharpe_pct"]
                - qc_baseline_stats["rolling_12m"]["positive_sharpe_pct"]
            ),
            1,
        )
        qc_row["verdict"] = qc_verdict(qc_row, qc_baseline_stats)

        candidates[name] = {
            "local": local_row,
            "qc_proxy": qc_row,
            "combined_verdict": combined_verdict(local_row["verdict"], qc_row["verdict"]),
        }

    def rank_key(item: tuple[str, dict[str, object]]) -> tuple[int, float, float]:
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

    best_name, best_row = max(candidates.items(), key=rank_key)
    strongest_qc_only = max(
        (
            (label, row)
            for label, row in candidates.items()
            if row["qc_proxy"]["delta_vs_baseline_net_pnl"] > 0
            and row["local"]["walkforward"]["summary"]["delta_vs_baseline"] <= 0
        ),
        key=lambda item: (
            item[1]["qc_proxy"]["delta_vs_baseline_net_pnl"],
            item[1]["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
            item[1]["qc_proxy"]["delta_vs_baseline_6m_positive_pct"],
        ),
        default=None,
    )
    strongest_local_only = max(
        (
            (label, row)
            for label, row in candidates.items()
            if row["local"]["walkforward"]["summary"]["delta_vs_baseline"] > 0
            and row["qc_proxy"]["delta_vs_baseline_net_pnl"] <= 0
        ),
        key=lambda item: (
            item[1]["local"]["walkforward"]["summary"]["delta_vs_baseline"],
            item[1]["qc_proxy"]["delta_vs_baseline_net_pnl"],
        ),
        default=None,
    )

    def summarize(label: str, row: dict[str, object]) -> dict[str, object]:
        return {
            "label": label,
            "combined_verdict": row["combined_verdict"],
            "local_verdict": row["local"]["verdict"],
            "qc_verdict": row["qc_proxy"]["verdict"],
            "local_walkforward_delta_vs_baseline": row["local"]["walkforward"]["summary"]["delta_vs_baseline"],
            "qc_delta_vs_baseline_net_pnl": row["qc_proxy"]["delta_vs_baseline_net_pnl"],
            "qc_delta_vs_baseline_6m_positive_pct": row["qc_proxy"]["delta_vs_baseline_6m_positive_pct"],
            "qc_delta_vs_baseline_12m_positive_pct": row["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
        }

    return {
        "research_scope": "v18_intraday_reference_level_bridge_family",
        "analysis_version": "v1_entry_price_relative_to_prior_day_levels",
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
            "interpretation": (
                "This family tests same-day intraday ORB structure by measuring where the actual entry price sits "
                "relative to the previous day's key levels at the moment the baseline v18 trade triggers. It is meant "
                "to extend same-day execution-state research beyond simple session-open stretch thresholds."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test same-day entry-reference-state gates on top of the accepted v18 baseline."
    )
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
    out_path = out_dir / "v18_intraday_reference_level_bridge_family.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
