"""Iteration 102: Expanded factor scan after alignment fix reset.

New research directions on top of the v18-prev-day-mom3 baseline:
1. Combinations of the two LOCAL_AHEAD_OF_QC leaders (mom5_positive, close_above_sma8)
2. New factor families: volatility regime, volume regime, day-of-week, gap size
3. All using corrected next-session alignment

The goal is to find a filter that achieves BRIDGE_CONFIRMED or at minimum
QC_STRONG on accepted QC trades.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.qc.analyze_qc_regime_prototypes import compute_profit_factor, rolling_summary
from research.qc.analyze_qc_webide_result import resolve_bundle
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame


def build_expanded_features(daily_path: Path) -> tuple[pd.DataFrame, list[object]]:
    """Build an expanded feature frame with new factor families."""
    daily = load_daily_market_frame(daily_path)

    close = daily["Close"]
    high = daily["High"]
    low = daily["Low"]
    volume = daily["Volume"]
    open_ = daily["Open"]

    # --- Existing slow-trend family (for combinations) ---
    daily["prev_day_up"] = close.pct_change() > 0
    daily["mom3_positive"] = (close / close.shift(3) - 1.0) > 0
    daily["mom5_positive"] = (close / close.shift(5) - 1.0) > 0
    daily["close_above_sma8"] = close > close.rolling(8).mean()

    # --- Volatility regime family ---
    # True range as volatility proxy
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr5 = tr.rolling(5).mean()
    atr10 = tr.rolling(10).mean()
    atr20 = tr.rolling(20).mean()

    # Low-vol = ATR below its own rolling median (calmer market → better ORB?)
    daily["low_atr5_vs_20"] = atr5 < atr20
    daily["low_atr10_vs_20"] = atr10 < atr20
    # Realized vol (close-to-close) below median
    ret = close.pct_change()
    realized_vol_10 = ret.rolling(10).std()
    realized_vol_20 = ret.rolling(20).std()
    realized_vol_60 = ret.rolling(60).std()
    daily["low_rvol10_vs_60"] = realized_vol_10 < realized_vol_60
    daily["low_rvol20_vs_60"] = realized_vol_20 < realized_vol_60
    # Contracting range: 5d average range < 20d average range
    day_range = high - low
    daily["contracting_range"] = day_range.rolling(5).mean() < day_range.rolling(20).mean()

    # --- Volume regime family ---
    vol_sma20 = volume.rolling(20).mean()
    vol_sma50 = volume.rolling(50).mean()
    daily["vol_above_sma20"] = volume > vol_sma20
    daily["vol_below_sma20"] = volume < vol_sma20
    daily["vol_above_sma50"] = volume > vol_sma50
    # Declining volume: 5d avg volume < 20d avg volume
    daily["declining_volume"] = volume.rolling(5).mean() < vol_sma20

    # --- Gap regime family ---
    gap_pct = open_ / prev_close - 1.0
    daily["gap_small"] = gap_pct.abs() < 0.005  # |gap| < 0.5%
    daily["gap_positive"] = gap_pct > 0
    daily["gap_negative"] = gap_pct < 0

    # --- Day-of-week (apply to session_date, not market_date) ---
    # We'll handle this differently since the session date is the next trading day

    # --- Combinations of two leaders ---
    daily["mom5_AND_sma8"] = daily["mom5_positive"] & daily["close_above_sma8"]
    daily["mom5_OR_sma8"] = daily["mom5_positive"] | daily["close_above_sma8"]

    feature_cols = [
        # Existing leaders (for reference)
        "mom5_positive",
        "close_above_sma8",
        # Combinations
        "mom5_AND_sma8",
        "mom5_OR_sma8",
        # Volatility regime
        "low_atr5_vs_20",
        "low_atr10_vs_20",
        "low_rvol10_vs_60",
        "low_rvol20_vs_60",
        "contracting_range",
        # Volume regime
        "vol_above_sma20",
        "vol_below_sma20",
        "vol_above_sma50",
        "declining_volume",
        # Gap regime
        "gap_small",
        "gap_positive",
        "gap_negative",
    ]

    aligned, calendar_dates = align_features_to_next_session(daily, feature_cols)
    return aligned, calendar_dates


def load_qc_trades(result_dir: Path) -> tuple[pd.DataFrame, str]:
    bundle = resolve_bundle(result_dir)
    trades = pd.read_csv(bundle.trades_path)
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
    trades["entry_date"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.date
    trades["exit_date"] = trades["Exit Time"].dt.tz_convert("America/New_York").dt.date
    trades["entry_year"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.year
    trades["entry_dow"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.dayofweek
    trades["net_pnl"] = (
        pd.to_numeric(trades["P&L"], errors="coerce").fillna(0.0)
        - pd.to_numeric(trades["Fees"], errors="coerce").fillna(0.0)
    )
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)
    return trades, bundle.trades_path.name


def stats_for_subset(subset: pd.DataFrame, calendar_dates: list[object]) -> dict:
    if subset.empty:
        return {
            "trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "net_pnl": 0.0, "positive_years": 0, "negative_years": 0,
            "rolling_6m_positive_pct": 0.0, "rolling_12m_positive_pct": 0.0,
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


def yearly_breakdown(subset: pd.DataFrame) -> dict:
    out = {}
    for year, part in subset.groupby("entry_year"):
        out[str(int(year))] = {
            "trades": int(len(part)),
            "net_pnl": round(float(part["net_pnl"].sum()), 2),
            "win_rate_pct": round(float(part["is_win_net"].mean() * 100.0), 2) if len(part) else 0.0,
        }
    return out


def qc_verdict(kept_stats: dict, excluded_stats: dict, baseline_stats: dict) -> str:
    if (
        kept_stats["net_pnl"] > baseline_stats["net_pnl"]
        and excluded_stats["net_pnl"] < 0
        and kept_stats["rolling_6m_positive_pct"] >= baseline_stats["rolling_6m_positive_pct"]
        and kept_stats["rolling_12m_positive_pct"] >= baseline_stats["rolling_12m_positive_pct"]
    ):
        return "QC_STRONG"
    if kept_stats["net_pnl"] > baseline_stats["net_pnl"] and excluded_stats["net_pnl"] < 0:
        return "QC_POSITIVE_BUT_MIXED"
    if kept_stats["net_pnl"] > baseline_stats["net_pnl"]:
        return "QC_MARGINAL_POSITIVE"
    return "QC_WEAK_OR_MIXED"


def half_split_check(subset: pd.DataFrame) -> dict:
    """Check if edge is present in both halves of the dataset."""
    first = subset[subset["entry_year"] <= 2020]
    second = subset[subset["entry_year"] > 2020]
    return {
        "first_half_net": round(float(first["net_pnl"].sum()), 2),
        "second_half_net": round(float(second["net_pnl"].sum()), 2),
        "both_positive": bool(first["net_pnl"].sum() > 0 and second["net_pnl"].sum() > 0),
    }


def evaluate_all(result_dir: Path, daily_csv: Path) -> dict:
    features, calendar_dates = build_expanded_features(daily_csv)
    trades, trades_file = load_qc_trades(result_dir)

    # Merge features onto trades
    merged = trades.merge(features, left_on="entry_date", right_on="date", how="left")

    # Add day-of-week filters directly on trade data
    merged["dow_mon_thu"] = merged["entry_dow"].isin([0, 1, 2, 3])  # Mon-Thu
    merged["dow_tue_thu"] = merged["entry_dow"].isin([1, 2, 3])      # Tue-Thu
    merged["dow_not_friday"] = merged["entry_dow"] != 4               # Not Friday
    merged["dow_not_monday"] = merged["entry_dow"] != 0               # Not Monday

    baseline_stats = stats_for_subset(merged, calendar_dates)

    # All candidate filters to test
    candidate_names = [
        # Leaders from iteration 101
        "mom5_positive", "close_above_sma8",
        # Combinations
        "mom5_AND_sma8", "mom5_OR_sma8",
        # Volatility regime
        "low_atr5_vs_20", "low_atr10_vs_20",
        "low_rvol10_vs_60", "low_rvol20_vs_60",
        "contracting_range",
        # Volume regime
        "vol_above_sma20", "vol_below_sma20", "vol_above_sma50",
        "declining_volume",
        # Gap regime
        "gap_small", "gap_positive", "gap_negative",
        # Day-of-week
        "dow_mon_thu", "dow_tue_thu", "dow_not_friday", "dow_not_monday",
    ]

    results = {}
    for name in candidate_names:
        if name not in merged.columns:
            continue
        mask = merged[name].fillna(False)
        kept = merged[mask]
        excluded = merged[~mask]

        kept_stats = stats_for_subset(kept, calendar_dates)
        excluded_stats = stats_for_subset(excluded, calendar_dates)
        verdict = qc_verdict(kept_stats, excluded_stats, baseline_stats)

        results[name] = {
            "kept": kept_stats,
            "excluded": {
                "trades": excluded_stats["trades"],
                "net_pnl": excluded_stats["net_pnl"],
                "win_rate_pct": excluded_stats["win_rate_pct"],
            },
            "delta_vs_baseline_net": round(kept_stats["net_pnl"] - baseline_stats["net_pnl"], 2),
            "qc_verdict": verdict,
            "half_split": half_split_check(kept),
            "excluded_yearly": yearly_breakdown(excluded),
        }

    # Rank by net PnL improvement with QC_STRONG or QC_POSITIVE_BUT_MIXED first
    rank_order = {"QC_STRONG": 3, "QC_POSITIVE_BUT_MIXED": 2, "QC_MARGINAL_POSITIVE": 1, "QC_WEAK_OR_MIXED": 0}
    ranked = sorted(
        results.items(),
        key=lambda item: (
            rank_order.get(item[1]["qc_verdict"], 0),
            item[1]["kept"]["net_pnl"],
        ),
        reverse=True,
    )

    top5 = [
        {
            "label": name,
            "qc_verdict": payload["qc_verdict"],
            "kept_trades": payload["kept"]["trades"],
            "kept_net_pnl": payload["kept"]["net_pnl"],
            "excluded_net_pnl": payload["excluded"]["net_pnl"],
            "delta_vs_baseline": payload["delta_vs_baseline_net"],
            "kept_rolling_6m": payload["kept"]["rolling_6m_positive_pct"],
            "kept_rolling_12m": payload["kept"]["rolling_12m_positive_pct"],
            "half_split_both_positive": payload["half_split"]["both_positive"],
        }
        for name, payload in ranked[:10]
    ]

    return {
        "research_scope": "v18_expanded_factor_scan_iter102",
        "alignment": "next_session_corrected",
        "source_trades_file": trades_file,
        "baseline": baseline_stats,
        "candidate_count": len(results),
        "candidates": {name: payload for name, payload in ranked},
        "top_candidates": top5,
    }


def main() -> int:
    result_dir = Path("QuantConnect results/2017-2026")
    daily_csv = Path("qqq_1d.csv")

    result = evaluate_all(result_dir, daily_csv)

    out_dir = Path("results/qc_regime_prototypes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v18_expanded_factor_scan_iter102.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
