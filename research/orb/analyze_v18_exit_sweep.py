"""Phase 2: Exit-rule parameter sweep on local 5m data.

Tests multiple exit configurations by modifying ORBStrategy parameters
and measuring impact on net PnL, win rate, and trade outcomes.

Candidates:
1. Trailing stop sensitivity: 0.008, 0.010, 0.013 (baseline), 0.016, 0.020, 0.025, 0.030
2. No trailing stop (trailing_stop=False) — pure ORB boundary + TP + EOD
3. Profit ratio sensitivity: 2.0, 2.5, 3.0, 3.5 (baseline), 4.0, 5.0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.orb.analyze_local_orb_v18_return_cap import (
    apply_base_filter,
    build_feature_frame,
    load_bars,
    merge_features,
    stats_for_subset,
)
from src.backtest.engine import BacktestEngine
from src.strategies.orb import ORBStrategy


def run_exit_variant(
    bars: pd.DataFrame,
    *,
    trailing_pct: float = 0.013,
    trailing_stop: bool = True,
    profit_ratio: float = 3.5,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Run local backtest with modified exit parameters."""
    subset = bars
    if start_date is not None and end_date is not None:
        subset = bars[(bars["et_date"] >= start_date) & (bars["et_date"] <= end_date)]

    strategy = ORBStrategy(params={
        "trailing_pct": trailing_pct,
        "trailing_stop": trailing_stop,
        "profit_ratio": profit_ratio,
    })
    result = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0).run(
        strategy,
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


def main():
    bars = load_bars(Path("qqq_5m.csv"))
    features = build_feature_frame(Path("qqq_1d.csv"))

    # Baseline (v18 defaults)
    baseline_trades = merge_features(run_exit_variant(bars), features)
    baseline_filtered = apply_base_filter(baseline_trades)
    baseline_stats = stats_for_subset(baseline_filtered)

    print("=== V18 BASELINE (local, regime-filtered) ===")
    print(f"  Trades: {baseline_stats['trades']}")
    print(f"  Net PnL: ${baseline_stats['net_pnl']:.2f}")
    print(f"  Win Rate: {baseline_stats['win_rate_pct']:.1f}%")
    print(f"  Profit Factor: {baseline_stats['profit_factor']:.3f}")
    print()

    results = []

    # === TEST 1: Trailing Stop Sensitivity ===
    print("=== TRAILING STOP SENSITIVITY ===")
    for tp in [0.008, 0.010, 0.013, 0.016, 0.020, 0.025, 0.030, 0.040, 0.050]:
        trades = merge_features(run_exit_variant(bars, trailing_pct=tp), features)
        filtered = apply_base_filter(trades)
        stats = stats_for_subset(filtered)
        delta = stats["net_pnl"] - baseline_stats["net_pnl"]
        label = f"trail_{tp*100:.1f}pct"
        marker = " <-- BASELINE" if tp == 0.013 else (" ***" if delta > 0 else "")
        print(f"  {label:18s}: {stats['trades']:3d} trades, ${stats['net_pnl']:>8.2f}, WR {stats['win_rate_pct']:>5.1f}%, PF {stats['profit_factor']:.3f}, delta ${delta:>+8.2f}{marker}")
        results.append({
            "category": "trailing_stop",
            "label": label,
            "trailing_pct": tp,
            "trailing_stop": True,
            "profit_ratio": 3.5,
            **stats,
            "delta_pnl": round(delta, 2),
        })

    # === TEST 2: No Trailing Stop (pure ORB boundary + TP + EOD) ===
    print()
    print("=== NO TRAILING STOP ===")
    trades = merge_features(run_exit_variant(bars, trailing_stop=False), features)
    filtered = apply_base_filter(trades)
    stats = stats_for_subset(filtered)
    delta = stats["net_pnl"] - baseline_stats["net_pnl"]
    print(f"  no_trailing_stop  : {stats['trades']:3d} trades, ${stats['net_pnl']:>8.2f}, WR {stats['win_rate_pct']:>5.1f}%, PF {stats['profit_factor']:.3f}, delta ${delta:>+8.2f}")
    results.append({
        "category": "no_trailing",
        "label": "no_trailing_stop",
        "trailing_pct": 0,
        "trailing_stop": False,
        "profit_ratio": 3.5,
        **stats,
        "delta_pnl": round(delta, 2),
    })

    # === TEST 3: Profit Ratio Sensitivity ===
    print()
    print("=== PROFIT RATIO SENSITIVITY ===")
    for pr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]:
        trades = merge_features(run_exit_variant(bars, profit_ratio=pr), features)
        filtered = apply_base_filter(trades)
        stats = stats_for_subset(filtered)
        delta = stats["net_pnl"] - baseline_stats["net_pnl"]
        label = f"pr_{pr:.1f}x"
        marker = " <-- BASELINE" if pr == 3.5 else (" ***" if delta > 0 else "")
        print(f"  {label:18s}: {stats['trades']:3d} trades, ${stats['net_pnl']:>8.2f}, WR {stats['win_rate_pct']:>5.1f}%, PF {stats['profit_factor']:.3f}, delta ${delta:>+8.2f}{marker}")
        results.append({
            "category": "profit_ratio",
            "label": label,
            "trailing_pct": 0.013,
            "trailing_stop": True,
            "profit_ratio": pr,
            **stats,
            "delta_pnl": round(delta, 2),
        })

    # === TEST 4: Combined best trailing + best profit ratio ===
    # Find best trailing
    trail_results = [r for r in results if r["category"] == "trailing_stop" and r["delta_pnl"] > 0]
    pr_results = [r for r in results if r["category"] == "profit_ratio" and r["delta_pnl"] > 0]

    if trail_results:
        best_trail = max(trail_results, key=lambda r: r["delta_pnl"])
        print(f"\nBest trailing: {best_trail['label']} (delta ${best_trail['delta_pnl']:+.2f})")
    if pr_results:
        best_pr = max(pr_results, key=lambda r: r["delta_pnl"])
        print(f"Best profit ratio: {best_pr['label']} (delta ${best_pr['delta_pnl']:+.2f})")

    # Test combined if both exist
    if trail_results and pr_results:
        bt = best_trail["trailing_pct"]
        bp = best_pr["profit_ratio"]
        print(f"\n=== COMBINED: trail={bt*100:.1f}% + pr={bp:.1f}x ===")
        trades = merge_features(run_exit_variant(bars, trailing_pct=bt, profit_ratio=bp), features)
        filtered = apply_base_filter(trades)
        stats = stats_for_subset(filtered)
        delta = stats["net_pnl"] - baseline_stats["net_pnl"]
        print(f"  combined          : {stats['trades']:3d} trades, ${stats['net_pnl']:>8.2f}, WR {stats['win_rate_pct']:>5.1f}%, PF {stats['profit_factor']:.3f}, delta ${delta:>+8.2f}")
        results.append({
            "category": "combined",
            "label": f"trail_{bt*100:.1f}_pr_{bp:.1f}",
            "trailing_pct": bt,
            "trailing_stop": True,
            "profit_ratio": bp,
            **stats,
            "delta_pnl": round(delta, 2),
        })

    # Also test no-trailing + best PR
    if pr_results:
        bp = best_pr["profit_ratio"]
        print(f"\n=== COMBINED: no_trailing + pr={bp:.1f}x ===")
        trades = merge_features(run_exit_variant(bars, trailing_stop=False, profit_ratio=bp), features)
        filtered = apply_base_filter(trades)
        stats = stats_for_subset(filtered)
        delta = stats["net_pnl"] - baseline_stats["net_pnl"]
        print(f"  no_trail_pr_{bp:.1f}x  : {stats['trades']:3d} trades, ${stats['net_pnl']:>8.2f}, WR {stats['win_rate_pct']:>5.1f}%, PF {stats['profit_factor']:.3f}, delta ${delta:>+8.2f}")

    # Save results
    output = {
        "research_scope": "v18_exit_parameter_sweep",
        "baseline": baseline_stats,
        "results": results,
    }
    out_path = Path("results/qc_regime_prototypes/v18_exit_parameter_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Final summary
    print("\n=== TOP 5 EXIT VARIANTS BY PNL IMPROVEMENT ===")
    sorted_results = sorted(results, key=lambda r: r["delta_pnl"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"  #{i+1} {r['label']:25s}: delta ${r['delta_pnl']:>+8.2f}, PnL ${r['net_pnl']:>8.2f}, WR {r['win_rate_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
