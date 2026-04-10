from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.qc.analyze_qc_regime_prototypes import compute_profit_factor, rolling_summary
from research.qc.analyze_qc_webide_result import resolve_bundle
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


ANALYSIS_VERSION = "v1_restored_baseline_priority_proxy"
TARGETS = [
    "mom4_positive",
    "mom5_positive",
    "close_above_sma8",
]
LOCAL_POSITIVE_VERDICTS = {"LOCAL_RESEARCH_LEADER", "LOCAL_PROMISING"}


def load_trades(result_dir: Path) -> tuple[object, pd.DataFrame]:
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
    return bundle, trades


def build_daily_features(daily_path: Path) -> tuple[pd.DataFrame, list[object]]:
    daily = load_daily_market_frame(daily_path)
    close = daily["Close"].astype(float)

    daily["mom4_positive"] = (close / close.shift(4) - 1.0) > 0.0
    daily["mom5_positive"] = (close / close.shift(5) - 1.0) > 0.0
    daily["close_above_sma8"] = close > close.rolling(8).mean()

    feature_columns = list(TARGETS)
    aligned, calendar_dates = align_features_to_next_session(daily, feature_columns)
    return aligned, calendar_dates


def stats_for_subset(subset: pd.DataFrame, calendar_dates: list[object]) -> dict[str, object]:
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


def load_local_context(local_scan_json: Path, local_priority_json: Path) -> tuple[dict[str, object], dict[str, int]]:
    local_scan = json.loads(local_scan_json.read_text(encoding="utf-8"))
    local_priority = json.loads(local_priority_json.read_text(encoding="utf-8"))
    local_lookup = local_scan["candidates"]

    rank_lookup: dict[str, int] = {}
    for idx, row in enumerate(local_priority["ranked_candidates"], start=1):
        rank_lookup[str(row["label"])] = idx
    return local_lookup, rank_lookup


def qc_verdict(row: dict[str, object], baseline: dict[str, object]) -> str:
    kept = row["kept"]
    excluded = row["excluded"]
    delta_6m = row["delta_vs_baseline_6m_positive_pct"]
    delta_12m = row["delta_vs_baseline_12m_positive_pct"]

    if (
        kept["net_pnl"] > baseline["net_pnl"]
        and excluded["net_pnl"] < 0
        and kept["positive_years"] >= baseline["positive_years"]
        and delta_12m >= 0.0
        and delta_6m >= -2.0
    ):
        return "QC_PROXY_FRONT_RUNNER"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0 and kept["positive_years"] >= baseline["positive_years"]:
        return "QC_PROXY_PROMISING"
    return "QC_PROXY_WEAK_OR_MIXED"


def combined_verdict(row: dict[str, object]) -> str:
    local_verdict = str(row["local_context"]["verdict"])
    qc_state = str(row["qc_verdict"])
    if local_verdict in LOCAL_POSITIVE_VERDICTS and qc_state == "QC_PROXY_FRONT_RUNNER":
        return "READY_FOR_QC_CANDIDATE"
    if local_verdict in LOCAL_POSITIVE_VERDICTS and qc_state == "QC_PROXY_PROMISING":
        return "RESEARCH_CONTINUE"
    return "HOLD_BASELINE"


def evaluate(
    result_dir: Path,
    daily_csv: Path,
    local_scan_json: Path,
    local_priority_json: Path,
) -> dict[str, object]:
    bundle, trades = load_trades(result_dir)
    features, calendar_dates = build_daily_features(daily_csv)
    merged = trades.merge(features, left_on="entry_date", right_on="date", how="left")
    baseline_stats = stats_for_subset(merged, calendar_dates)

    local_lookup, rank_lookup = load_local_context(local_scan_json, local_priority_json)

    candidates: dict[str, object] = {}
    for name in TARGETS:
        kept = merged[merged[name].fillna(False)].copy()
        excluded = merged[~merged[name].fillna(False)].copy()
        row = {
            "label": name,
            "priority_rank_before_qc_proxy": rank_lookup.get(name),
            "kept": stats_for_subset(kept, calendar_dates),
            "excluded": stats_for_subset(excluded, calendar_dates),
            "local_context": {
                "verdict": local_lookup[name]["verdict"],
                "kept_net_pnl": local_lookup[name]["kept"]["net_pnl"],
                "excluded_net_pnl": local_lookup[name]["excluded"]["net_pnl"],
                "walkforward_delta_vs_baseline": local_lookup[name]["walkforward"]["summary"]["delta_vs_baseline"],
                "improved_vs_baseline_folds": local_lookup[name]["walkforward"]["summary"]["improved_vs_baseline_folds"],
                "leave_one_month_out_all_positive": local_lookup[name]["leave_one_month_out"]["all_positive"],
            },
        }
        row["delta_vs_baseline_net_pnl"] = round(float(row["kept"]["net_pnl"] - baseline_stats["net_pnl"]), 2)
        row["delta_vs_baseline_6m_positive_pct"] = round(
            float(row["kept"]["rolling_6m"]["positive_sharpe_pct"] - baseline_stats["rolling_6m"]["positive_sharpe_pct"]),
            1,
        )
        row["delta_vs_baseline_12m_positive_pct"] = round(
            float(row["kept"]["rolling_12m"]["positive_sharpe_pct"] - baseline_stats["rolling_12m"]["positive_sharpe_pct"]),
            1,
        )
        row["delta_vs_baseline_positive_years"] = int(row["kept"]["positive_years"] - baseline_stats["positive_years"])
        row["qc_verdict"] = qc_verdict(row, baseline_stats)
        row["combined_verdict"] = combined_verdict(row)
        candidates[name] = row

    best = max(
        candidates.values(),
        key=lambda item: (
            item["combined_verdict"] == "READY_FOR_QC_CANDIDATE",
            item["qc_verdict"] == "QC_PROXY_FRONT_RUNNER",
            item["kept"]["net_pnl"],
            item["kept"]["rolling_12m"]["positive_sharpe_pct"],
            item["kept"]["rolling_6m"]["positive_sharpe_pct"],
        ),
    )

    return {
        "research_scope": "qc_v18_priority_proxy",
        "analysis_version": ANALYSIS_VERSION,
        "source_bundle": bundle.json_path.stem,
        "source_trades_csv": bundle.trades_path.name,
        "source_daily_csv": daily_csv.name,
        "baseline": baseline_stats,
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": {
                "label": best["label"],
                "priority_rank_before_qc_proxy": best["priority_rank_before_qc_proxy"],
                "kept_net_pnl": best["kept"]["net_pnl"],
                "excluded_net_pnl": best["excluded"]["net_pnl"],
                "delta_vs_baseline_net_pnl": best["delta_vs_baseline_net_pnl"],
                "rolling_6m_positive_pct": best["kept"]["rolling_6m"]["positive_sharpe_pct"],
                "rolling_12m_positive_pct": best["kept"]["rolling_12m"]["positive_sharpe_pct"],
                "positive_years": best["kept"]["positive_years"],
                "qc_verdict": best["qc_verdict"],
                "combined_verdict": best["combined_verdict"],
            },
            "interpretation": (
                "This QC proxy pass uses restored accepted baseline trades and next-session daily alignment. "
                "It is the required bridge between local-only ranking and any new live QC candidate launch."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QC proxy validation for the v18 priority queue after baseline trades return.")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--local-scan-json", default="results/qc_regime_prototypes/local_orb_v18_adjacent_trend_scan.json")
    parser.add_argument("--local-priority-json", default="results/qc_regime_prototypes/local_v18_priority_tiebreak.json")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    result = evaluate(
        result_dir=Path(args.result_dir),
        daily_csv=Path(args.daily_csv),
        local_scan_json=Path(args.local_scan_json),
        local_priority_json=Path(args.local_priority_json),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qc_v18_priority_proxy.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
