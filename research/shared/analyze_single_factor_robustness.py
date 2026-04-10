from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.qc.analyze_qc_regime_prototypes import compute_profit_factor, rolling_summary
from research.qc.analyze_qc_webide_result import resolve_bundle
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


def build_feature_frame(daily_path: Path) -> tuple[pd.DataFrame, list[object]]:
    daily = load_daily_market_frame(daily_path)

    close = daily["Close"]
    ret1 = close.pct_change()
    daily["prev_day_up"] = ret1 > 0
    daily["mom3_positive"] = (close / close.shift(3) - 1.0) > 0
    daily["mom10_positive"] = (close / close.shift(10) - 1.0) > 0
    daily["close_above_sma10"] = close > close.rolling(10).mean()
    daily["close_above_sma20"] = close > close.rolling(20).mean()

    cols = ["prev_day_up", "mom3_positive", "mom10_positive", "close_above_sma10", "close_above_sma20"]
    return align_features_to_next_session(daily, cols)


def load_trades(result_dir: Path) -> tuple[pd.DataFrame, str]:
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


def stats_for_subset(subset: pd.DataFrame, calendar_dates: list[object]) -> dict[str, float]:
    if subset.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "rolling_6m_positive_pct": 0.0,
            "rolling_12m_positive_pct": 0.0,
        }
    return {
        "trades": int(len(subset)),
        "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2),
        "profit_factor": round(float(compute_profit_factor(subset["net_pnl"])), 3),
        "net_pnl": round(float(subset["net_pnl"].sum()), 2),
        "rolling_6m_positive_pct": rolling_summary(calendar_dates, subset, 6)["positive_sharpe_pct"],
        "rolling_12m_positive_pct": rolling_summary(calendar_dates, subset, 12)["positive_sharpe_pct"],
    }


def split_stats(subset: pd.DataFrame, groups: dict[str, pd.Series]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, mask in groups.items():
        part = subset[mask]
        out[name] = {
            "trades": int(len(part)),
            "net_pnl": round(float(part["net_pnl"].sum()), 2),
            "win_rate_pct": round(float(part["is_win_net"].mean() * 100.0), 2) if len(part) else 0.0,
        }
    return out


def yearly_stats(subset: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for year, part in subset.groupby("entry_year"):
        out[str(int(year))] = {
            "trades": int(len(part)),
            "net_pnl": round(float(part["net_pnl"].sum()), 2),
            "win_rate_pct": round(float(part["is_win_net"].mean() * 100.0), 2) if len(part) else 0.0,
        }
    return out


def leave_one_year_out_stats(subset: pd.DataFrame, years: list[int]) -> dict[str, object]:
    nets: dict[str, float] = {}
    for year in years:
        part = subset[subset["entry_year"] != year]
        nets[str(year)] = round(float(part["net_pnl"].sum()), 2)
    min_net = min(nets.values()) if nets else 0.0
    return {
        "nets": nets,
        "min_net_pnl": round(float(min_net), 2),
        "all_positive": bool(nets) and all(v > 0 for v in nets.values()),
    }


def stress_stats(subset: pd.DataFrame) -> dict[str, object]:
    by_year = subset.groupby("entry_year")["net_pnl"].sum().sort_values()
    if by_year.empty:
        return {"exclude_best_year_net_pnl": 0.0, "exclude_worst_year_net_pnl": 0.0}
    worst_year = int(by_year.index[0])
    best_year = int(by_year.index[-1])
    exclude_best = subset[subset["entry_year"] != best_year]["net_pnl"].sum()
    exclude_worst = subset[subset["entry_year"] != worst_year]["net_pnl"].sum()
    return {
        "best_year": best_year,
        "worst_year": worst_year,
        "exclude_best_year_net_pnl": round(float(exclude_best), 2),
        "exclude_worst_year_net_pnl": round(float(exclude_worst), 2),
    }


def verdict(candidate: dict[str, object]) -> str:
    half = candidate["half_split"]
    parity = candidate["parity_split"]
    loo = candidate["leave_one_year_out"]
    excluded_net = candidate["excluded"]["net_pnl"]
    if (
        half["first_half"]["net_pnl"] > 0
        and half["second_half"]["net_pnl"] > 0
        and parity["odd_years"]["net_pnl"] > 0
        and parity["even_years"]["net_pnl"] > 0
        and loo["all_positive"]
        and excluded_net < 0
    ):
        return "STRONG_EXPLORATORY"
    if half["first_half"]["net_pnl"] > 0 and half["second_half"]["net_pnl"] > 0 and excluded_net < 0:
        return "PROMISING_BUT_NEEDS_MORE_DATA"
    return "UNSTABLE"


def evaluate(result_dir: Path, daily_csv: Path, candidates: list[str]) -> dict[str, object]:
    features, calendar_dates = build_feature_frame(daily_csv)
    trades, trades_file = load_trades(result_dir)
    merged = trades.merge(features, left_on="entry_date", right_on="date", how="left")
    years = sorted(int(y) for y in merged["entry_year"].dropna().unique())

    baseline = stats_for_subset(merged, calendar_dates)
    results: dict[str, object] = {}
    for name in candidates:
        mask = merged[name].fillna(False)
        kept = merged[mask].copy()
        excluded = merged[~mask].copy()
        half_groups = {
            "first_half": kept["entry_year"] <= 2020,
            "second_half": kept["entry_year"] > 2020,
        }
        parity_groups = {
            "odd_years": (kept["entry_year"] % 2) == 1,
            "even_years": (kept["entry_year"] % 2) == 0,
        }
        candidate = {
            "kept": stats_for_subset(kept, calendar_dates),
            "excluded": {
                **stats_for_subset(excluded, calendar_dates),
                "yearly": yearly_stats(excluded),
            },
            "half_split": split_stats(kept, half_groups),
            "parity_split": split_stats(kept, parity_groups),
            "leave_one_year_out": leave_one_year_out_stats(kept, years),
            "stress_test": stress_stats(kept),
            "yearly_kept": yearly_stats(kept),
        }
        candidate["verdict"] = verdict(candidate)
        results[name] = candidate

    return {
        "source_trades_file": trades_file,
        "baseline": baseline,
        "candidates": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stronger robustness checks on single-factor research candidates.")
    parser.add_argument("result_dir", nargs="?", default="QuantConnect results/2017-2026")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--candidates", default="prev_day_up,mom3_positive")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    result = evaluate(Path(args.result_dir), Path(args.daily_csv), candidates)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.result_dir).name.replace(" ", "_")
    out_path = out_dir / f"{stem}_candidate_robustness.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
