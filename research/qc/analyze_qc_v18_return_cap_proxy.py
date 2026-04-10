from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.qc.analyze_qc_regime_prototypes import compute_profit_factor, rolling_summary
from research.qc.analyze_qc_webide_result import resolve_bundle
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


INITIAL_CAPITAL = 100000.0
ANALYSIS_VERSION = "v2_qc_trade_filter_proxy_next_session_alignment"
DEFAULT_RETURN_CAPS = [0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]


def load_trades(result_dir: Path) -> pd.DataFrame:
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
    return trades


def build_daily_features(daily_path: Path) -> pd.DataFrame:
    qqq = load_daily_market_frame(daily_path)
    qqq["close"] = qqq["Close"]
    qqq["prev_day_return"] = qqq["close"].pct_change()
    aligned, _ = align_features_to_next_session(qqq, ["prev_day_return"])
    return aligned


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


def verdict(candidate: dict[str, object], baseline: dict[str, object]) -> str:
    if (
        candidate["kept"]["net_pnl"] > baseline["net_pnl"]
        and candidate["excluded"]["net_pnl"] < 0
        and candidate["kept"]["rolling_6m"]["positive_sharpe_pct"] >= baseline["rolling_6m"]["positive_sharpe_pct"]
        and candidate["kept"]["rolling_12m"]["positive_sharpe_pct"] >= baseline["rolling_12m"]["positive_sharpe_pct"]
    ):
        return "QC_PROXY_LEADER"
    if candidate["kept"]["net_pnl"] > baseline["net_pnl"] and candidate["excluded"]["net_pnl"] < 0:
        return "QC_PROXY_PROMISING"
    return "QC_PROXY_WEAK_OR_MIXED"


def evaluate(result_dir: Path, daily_csv: Path, return_caps: list[float]) -> dict[str, object]:
    trades = load_trades(result_dir)
    features = build_daily_features(daily_csv)
    merged = trades.merge(features, left_on="entry_date", right_on="date", how="left")
    calendar_dates = sorted(features["date"])
    baseline_stats = stats_for_subset(merged, calendar_dates)

    candidates: dict[str, object] = {}
    for cap in sorted(return_caps):
        kept = merged[merged["prev_day_return"].fillna(np.inf) <= cap].copy()
        excluded = merged[merged["prev_day_return"].fillna(np.inf) > cap].copy()
        label = f"prev_day_return<={cap * 100.0:.2f}%"
        row = {
            "label": label,
            "return_cap": round(float(cap), 6),
            "kept": stats_for_subset(kept, calendar_dates),
            "excluded": stats_for_subset(excluded, calendar_dates),
        }
        row["delta_vs_baseline_net_pnl"] = round(float(row["kept"]["net_pnl"] - baseline_stats["net_pnl"]), 2)
        row["delta_vs_baseline_6m_positive_pct"] = round(
            float(row["kept"]["rolling_6m"]["positive_sharpe_pct"] - baseline_stats["rolling_6m"]["positive_sharpe_pct"]),
            1,
        )
        row["delta_vs_baseline_12m_positive_pct"] = round(
            float(
                row["kept"]["rolling_12m"]["positive_sharpe_pct"] - baseline_stats["rolling_12m"]["positive_sharpe_pct"]
            ),
            1,
        )
        row["verdict"] = verdict(row, baseline_stats)
        candidates[label] = row

    best = max(
        candidates.values(),
        key=lambda item: (
            item["verdict"] == "QC_PROXY_LEADER",
            item["kept"]["net_pnl"],
            item["kept"]["rolling_12m"]["positive_sharpe_pct"],
            item["kept"]["rolling_6m"]["positive_sharpe_pct"],
        ),
    )

    bundle = resolve_bundle(result_dir)
    return {
        "research_scope": "qc_v18_return_cap_proxy",
        "analysis_version": ANALYSIS_VERSION,
        "source_file": bundle.trades_path.name,
        "source_bundle": bundle.json_path.stem,
        "baseline": baseline_stats,
        "return_caps": [round(float(cap), 6) for cap in sorted(return_caps)],
        "candidates": candidates,
        "candidate_summary": {
            "best_overall": {
                "label": best["label"],
                "return_cap": best["return_cap"],
                "kept_net_pnl": best["kept"]["net_pnl"],
                "excluded_net_pnl": best["excluded"]["net_pnl"],
                "rolling_6m_positive_pct": best["kept"]["rolling_6m"]["positive_sharpe_pct"],
                "rolling_12m_positive_pct": best["kept"]["rolling_12m"]["positive_sharpe_pct"],
                "delta_vs_baseline_net_pnl": best["delta_vs_baseline_net_pnl"],
                "verdict": best["verdict"],
            },
            "interpretation": (
                "This is a QC-side trade-filter proxy using accepted baseline trades. "
                "Because the hypothesis only removes entry days and does not alter exits, it is materially safer than the "
                "rejected exit-rule proxy path, but it still remains a proxy until a full QC rerun confirms it."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QC proxy analysis for v18 prev-day return cap.")
    parser.add_argument("--result-dir", default="QuantConnect results/2017-2026")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    parser.add_argument("--return-caps", default="0.0075,0.01,0.0125,0.015,0.0175,0.02")
    parser.add_argument("--output-dir", default="results/qc_regime_prototypes")
    args = parser.parse_args()

    return_caps = [float(item.strip()) for item in args.return_caps.split(",") if item.strip()]
    result = evaluate(Path(args.result_dir), Path(args.daily_csv), return_caps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qc_v18_return_cap_proxy.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
