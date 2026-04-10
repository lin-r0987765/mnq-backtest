from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_session_alignment import align_features_to_next_session, load_daily_market_frame
from src.backtest.engine import BacktestEngine
from src.strategies.orb import ORBStrategy


BASE_FILTER_LABEL = "prev_day_up_and_mom3_positive"


def load_bars(intraday_csv: Path) -> pd.DataFrame:
    bars = pd.read_csv(intraday_csv)
    bars["Datetime"] = pd.to_datetime(bars["Datetime"], utc=True)
    bars = bars.set_index("Datetime").sort_index()
    bars["et_date"] = bars.index.tz_convert("America/New_York").date
    return bars


def build_feature_frame(daily_csv: Path) -> pd.DataFrame:
    daily = load_daily_market_frame(daily_csv)

    close = daily["Close"]
    ret1 = close.pct_change()
    ret3 = close / close.shift(3) - 1.0

    daily["prev_day_return"] = ret1
    daily["prev_day_up"] = ret1 > 0.0
    daily["mom3_positive"] = ret3 > 0.0
    aligned, _ = align_features_to_next_session(daily, ["prev_day_return", "prev_day_up", "mom3_positive"])
    return aligned


def run_local_backtest(bars: pd.DataFrame, *, start_date: object | None = None, end_date: object | None = None) -> pd.DataFrame:
    subset = bars
    if start_date is not None and end_date is not None:
        subset = bars[(bars["et_date"] >= start_date) & (bars["et_date"] <= end_date)]
    result = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0).run(
        ORBStrategy(),
        subset.drop(columns=["et_date"]),
    )
    trades = pd.DataFrame(result.trades)
    if trades.empty:
        return trades

    trades["Entry Timestamp"] = pd.to_datetime(trades["Entry Timestamp"], utc=True)
    trades["Exit Timestamp"] = pd.to_datetime(trades["Exit Timestamp"], utc=True)
    trades["entry_date"] = trades["Entry Timestamp"].dt.tz_convert("America/New_York").dt.date
    trades["entry_month"] = trades["Entry Timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    trades["net_pnl"] = pd.to_numeric(trades["PnL"], errors="coerce").fillna(0.0)
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)
    return trades


def merge_features(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades.merge(features, left_on="entry_date", right_on="date", how="left")


def apply_base_filter(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    mask = trades["prev_day_up"].fillna(False) & trades["mom3_positive"].fillna(False)
    return trades[mask].copy()


def stats_for_subset(subset: pd.DataFrame) -> dict[str, float]:
    if subset.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
        }

    gross_pos = subset.loc[subset["net_pnl"] > 0, "net_pnl"].sum()
    gross_neg = abs(subset.loc[subset["net_pnl"] < 0, "net_pnl"].sum())
    if gross_neg > 0:
        pf = float(gross_pos / gross_neg)
    elif gross_pos > 0:
        pf = 999.0
    else:
        pf = 0.0

    return {
        "trades": int(len(subset)),
        "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2),
        "profit_factor": round(float(pf), 3),
        "net_pnl": round(float(subset["net_pnl"].sum()), 2),
    }


def monthly_stats(subset: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if subset.empty:
        return out
    for month, part in subset.groupby("entry_month"):
        out[str(month)] = stats_for_subset(part)
    return out


def leave_one_month_out(subset: pd.DataFrame) -> dict[str, object]:
    nets: dict[str, float] = {}
    if subset.empty:
        return {"nets": nets, "min_net_pnl": 0.0, "all_positive": False}

    for month in sorted(subset["entry_month"].dropna().unique()):
        part = subset[subset["entry_month"] != month]
        nets[str(month)] = round(float(part["net_pnl"].sum()), 2)
    return {
        "nets": nets,
        "min_net_pnl": round(float(min(nets.values())), 2) if nets else 0.0,
        "all_positive": bool(nets) and all(value > 0 for value in nets.values()),
    }


def parse_folds(walk_forward_json: Path) -> list[dict[str, object]]:
    obj = json.loads(walk_forward_json.read_text(encoding="utf-8"))
    return list(obj["orb"]["folds"])


def evaluate_walkforward(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    folds: list[dict[str, object]],
    return_cap: float,
) -> dict[str, object]:
    fold_rows: list[dict[str, object]] = []
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
        kept = baseline[baseline["prev_day_return"].fillna(np.inf) <= return_cap].copy()
        excluded = baseline[baseline["prev_day_return"].fillna(np.inf) > return_cap].copy()

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

        fold_rows.append(
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
        "folds": fold_rows,
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
        and walk["positive_kept_folds"] >= 3
        and loo["all_positive"]
    ):
        return "LOCAL_RESEARCH_LEADER"
    if kept["net_pnl"] > baseline["net_pnl"] and excluded["net_pnl"] < 0:
        return "LOCAL_PROMISING"
    return "LOCAL_WEAK_OR_MIXED"


def evaluate(
    intraday_csv: Path,
    daily_csv: Path,
    walk_forward_json: Path,
    return_caps: list[float],
) -> dict[str, object]:
    bars = load_bars(intraday_csv)
    features = build_feature_frame(daily_csv)
    folds = parse_folds(walk_forward_json)

    merged = merge_features(run_local_backtest(bars), features)
    baseline = apply_base_filter(merged)
    baseline_stats = stats_for_subset(baseline)

    candidates: dict[str, object] = {}
    for cap in return_caps:
        kept = baseline[baseline["prev_day_return"].fillna(np.inf) <= cap].copy()
        excluded = baseline[baseline["prev_day_return"].fillna(np.inf) > cap].copy()
        label = f"prev_day_return<={cap * 100.0:.2f}%"
        candidate = {
            "label": label,
            "return_cap": round(float(cap), 6),
            "kept": stats_for_subset(kept),
            "excluded": stats_for_subset(excluded),
            "monthly_kept": monthly_stats(kept),
            "leave_one_month_out": leave_one_month_out(kept),
            "walkforward": evaluate_walkforward(bars, features, folds, cap),
        }
        candidate["verdict"] = verdict(candidate, baseline_stats)
        candidates[label] = candidate

    def rank_key(item: tuple[str, dict[str, object]]) -> tuple[int, float, float]:
        value = item[1]
        verdict_rank = {
            "LOCAL_RESEARCH_LEADER": 2,
            "LOCAL_PROMISING": 1,
            "LOCAL_WEAK_OR_MIXED": 0,
        }[value["verdict"]]
        walk_delta = value["walkforward"]["summary"]["delta_vs_baseline"]
        kept_net = value["kept"]["net_pnl"]
        return verdict_rank, walk_delta, kept_net

    best_name, best_value = max(candidates.items(), key=rank_key)

    return {
        "research_scope": "local_v18_return_cap",
        "analysis_version": "v1_prev_day_return_upper_bound",
        "source_intraday_csv": intraday_csv.name,
        "source_daily_csv": daily_csv.name,
        "source_walk_forward_json": walk_forward_json.name,
        "base_filter": BASE_FILTER_LABEL,
        "baseline": baseline_stats,
        "return_caps": [round(float(cap), 6) for cap in return_caps],
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": {
                "label": best_name,
                **{
                    "return_cap": best_value["return_cap"],
                    "kept_net_pnl": best_value["kept"]["net_pnl"],
                    "excluded_net_pnl": best_value["excluded"]["net_pnl"],
                    "improved_vs_baseline_folds": best_value["walkforward"]["summary"]["improved_vs_baseline_folds"],
                    "positive_kept_folds": best_value["walkforward"]["summary"]["positive_kept_folds"],
                    "delta_vs_baseline_walkforward": best_value["walkforward"]["summary"]["delta_vs_baseline"],
                    "verdict": best_value["verdict"],
                },
            },
            "interpretation": (
                "This sweep tests whether v18 performs better when prior-day upside is positive but not overextended. "
                "Any strong candidate here is still local-only evidence and cannot become a QC candidate until a clean "
                "accepted baseline trade source is restored for QC-side validation."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep prior-day return caps on top of local v18 baseline.")
    parser.add_argument("--intraday-csv", default="qqq_5m.csv")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--walk-forward-json", default="walk_forward_results.json")
    parser.add_argument("--return-caps", default="0.005,0.0075,0.01,0.0125,0.015,0.02")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    return_caps = [float(item.strip()) for item in args.return_caps.split(",") if item.strip()]
    result = evaluate(
        intraday_csv=Path(args.intraday_csv),
        daily_csv=Path(args.daily_csv),
        walk_forward_json=Path(args.walk_forward_json),
        return_caps=return_caps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_orb_v18_return_cap.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
