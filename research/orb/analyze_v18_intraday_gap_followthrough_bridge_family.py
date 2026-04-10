from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from research.orb.analyze_v18_intraday_entry_state_bridge_family import build_local_session_open_map
from daily_session_alignment import NEW_YORK_TZ, load_daily_market_frame


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"
CANDIDATE_SPECS = {
    "entry_continues_overnight_gap": {"mode": "gap_continuation"},
    "entry_reverses_overnight_gap": {"mode": "gap_reversal"},
    "gap_followthrough_ge_0p25x": {"mode": "gap_followthrough_ge", "threshold": 0.25},
    "gap_followthrough_ge_0p50x": {"mode": "gap_followthrough_ge", "threshold": 0.50},
    "gap_followthrough_ge_1p00x": {"mode": "gap_followthrough_ge", "threshold": 1.00},
    "small_gap_continuation_le_0p50pct": {"mode": "small_gap_continuation", "threshold": 0.50},
    "large_gap_continuation_ge_0p60pct": {"mode": "large_gap_continuation", "threshold": 0.60},
}


def _bool_mask(mask: pd.Series) -> pd.Series:
    return mask.fillna(False).astype(bool)


def build_gap_feature_frame(daily_csv: Path) -> pd.DataFrame:
    features = build_feature_frame(daily_csv)
    daily = load_daily_market_frame(daily_csv)[["market_date", "Open", "Close"]].copy()
    daily["prev_close"] = daily["Close"].shift(1)
    daily["overnight_gap_pct"] = ((daily["Open"] / daily["prev_close"]) - 1.0) * 100.0
    daily = daily.rename(columns={"market_date": "date", "Open": "session_open_daily"})
    return features.merge(daily[["date", "session_open_daily", "prev_close", "overnight_gap_pct"]], on="date", how="left")


def enrich_gap_followthrough(
    trades: pd.DataFrame,
    *,
    entry_timestamp_col: str,
    entry_price_col: str,
    session_open_map: pd.Series,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    enriched = trades.copy()
    entry_ts = pd.to_datetime(enriched[entry_timestamp_col], utc=True)
    entry_et = entry_ts.dt.tz_convert(NEW_YORK_TZ)
    session_open = entry_et.dt.date.map(session_open_map)
    entry_price = pd.to_numeric(enriched[entry_price_col], errors="coerce")
    gap_pct = pd.to_numeric(enriched["overnight_gap_pct"], errors="coerce")

    enriched["session_open"] = pd.to_numeric(session_open, errors="coerce")
    enriched["entry_move_from_open_pct"] = ((entry_price / enriched["session_open"]) - 1.0) * 100.0
    enriched["gap_abs_pct"] = gap_pct.abs()
    enriched["gap_same_direction"] = (enriched["entry_move_from_open_pct"] * gap_pct) > 0
    enriched["gap_opposite_direction"] = (enriched["entry_move_from_open_pct"] * gap_pct) < 0
    enriched["gap_followthrough_ratio"] = (
        enriched["entry_move_from_open_pct"].abs() / enriched["gap_abs_pct"].replace(0, pd.NA)
    )
    return enriched


def build_candidate_mask(frame: pd.DataFrame, spec: dict[str, object]) -> pd.Series:
    mode = str(spec["mode"])
    threshold = float(spec["threshold"]) if "threshold" in spec else None

    if mode == "gap_continuation":
        return _bool_mask(frame["gap_same_direction"])
    if mode == "gap_reversal":
        return _bool_mask(frame["gap_opposite_direction"])
    if mode == "gap_followthrough_ge":
        return _bool_mask(frame["gap_same_direction"] & (frame["gap_followthrough_ratio"] >= threshold))
    if mode == "small_gap_continuation":
        return _bool_mask(frame["gap_same_direction"] & (frame["gap_abs_pct"] <= threshold))
    if mode == "large_gap_continuation":
        return _bool_mask(frame["gap_same_direction"] & (frame["gap_abs_pct"] >= threshold))
    raise ValueError(f"Unknown candidate mode: {mode}")


def build_local_baseline_frame(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    *,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    subset = bars
    if start_date is not None and end_date is not None:
        subset = bars[(bars["et_date"] >= start_date) & (bars["et_date"] <= end_date)]
    trades = merge_features(run_local_backtest(bars, start_date=start_date, end_date=end_date), features)
    baseline = apply_base_filter(trades)
    if baseline.empty:
        return baseline
    session_open_map = build_local_session_open_map(subset)
    return enrich_gap_followthrough(
        baseline,
        entry_timestamp_col="Entry Timestamp",
        entry_price_col="Avg Entry Price",
        session_open_map=session_open_map,
    )


def evaluate_local_walkforward(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    folds: list[dict[str, object]],
    spec: dict[str, object],
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
        kept = baseline[build_candidate_mask(baseline, spec)].copy()
        excluded = baseline[~build_candidate_mask(baseline, spec)].copy()

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
    features = build_gap_feature_frame(daily_csv)
    bars = load_bars(intraday_csv)
    folds = parse_folds(walk_forward_json)

    local_baseline = build_local_baseline_frame(bars, features)
    local_baseline_stats = local_stats_for_subset(local_baseline)

    bundle, qc_trades = load_qc_trades(result_dir)
    qc_trades["entry_date"] = pd.to_datetime(qc_trades["Entry Time"], utc=True).dt.tz_convert(NEW_YORK_TZ).dt.date
    qc_merged = qc_trades.merge(features, left_on="entry_date", right_on="date", how="left")
    qc_session_open_map = features.set_index("date")["session_open_daily"]
    qc_merged = enrich_gap_followthrough(
        qc_merged,
        entry_timestamp_col="Entry Time",
        entry_price_col="Entry Price",
        session_open_map=qc_session_open_map,
    )
    calendar_dates = list(features["date"])
    qc_baseline_stats = qc_stats_for_subset(qc_merged, calendar_dates)

    candidates: dict[str, object] = {}
    for label, spec in CANDIDATE_SPECS.items():
        local_mask = build_candidate_mask(local_baseline, spec)
        local_kept = local_baseline[local_mask].copy()
        local_excluded = local_baseline[~local_mask].copy()
        local_row = {
            "kept": local_stats_for_subset(local_kept),
            "excluded": local_stats_for_subset(local_excluded),
            "walkforward": evaluate_local_walkforward(bars, features, folds, spec),
        }
        local_row["verdict"] = local_verdict(local_row, local_baseline_stats)

        qc_mask = build_candidate_mask(qc_merged, spec)
        qc_kept = qc_merged[qc_mask].copy()
        qc_excluded = qc_merged[~qc_mask].copy()
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

        candidates[label] = {
            "spec": spec,
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

    def summarize(label: str, row: dict[str, object]) -> dict[str, object]:
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
    strongest_balanced = max(
        (
            (label, row)
            for label, row in candidates.items()
            if row["local"]["walkforward"]["summary"]["delta_vs_baseline"] > 0
            and row["qc_proxy"]["delta_vs_baseline_net_pnl"] > 0
        ),
        key=lambda item: (
            item[1]["qc_proxy"]["delta_vs_baseline_net_pnl"],
            item[1]["local"]["walkforward"]["summary"]["delta_vs_baseline"],
            item[1]["qc_proxy"]["delta_vs_baseline_12m_positive_pct"],
        ),
        default=None,
    )

    return {
        "research_scope": "v18_intraday_gap_followthrough_bridge_family",
        "analysis_version": "v1_same_day_gap_followthrough_at_entry",
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
                "This family tests same-day intraday ORB execution-state through how much the actual baseline entry is "
                "continuing or fading the overnight gap at the moment of entry. It is intended to move beyond simple "
                "daily gap classification by focusing on execution-time gap follow-through."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze v18 same-day intraday gap-followthrough candidates on top of the accepted baseline."
    )
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--output", default="results/qc_regime_prototypes/v18_intraday_gap_followthrough_bridge_family.json")
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
