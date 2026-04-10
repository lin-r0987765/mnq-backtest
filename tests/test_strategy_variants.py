#!/usr/bin/env python3
"""
ORB 動態止盈測試 — 根據 HTF bias 調整 TP 倍數
==============================================
HTF bias 與交易方向一致時，止盈延展 20%；
不一致或中性時，維持標準止盈。
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetch_nq_data
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine
from src.portfolio_overlay import combine_results_active_reuse, combine_results_active_reuse_kelly


def compute_metrics(equity):
    arr = np.array(equity)
    if len(arr) < 2:
        return {"return_pct": 0, "sharpe": 0, "sortino": 0, "max_dd_pct": 0, "win_rate": 0, "trades": 0, "pf": 0}
    ret_pct = (arr[-1] / arr[0] - 1) * 100
    returns = pd.Series(arr).pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * (252 * 78) ** 0.5) if returns.std() > 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = float(returns.mean() / downside * (252 * 78) ** 0.5) if downside > 0 else 0.0
    peak = pd.Series(arr).cummax()
    dd = (pd.Series(arr) - peak) / peak
    max_dd = float(dd.min() * 100)
    return {
        "return_pct": round(ret_pct, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 4),
    }


def main():
    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    # Baseline ORB
    orb_base = engine.run(ORBStrategy(), df)
    om = orb_base.metrics
    print(f"ORB Baseline: Ret={om['total_return_pct']:+.3f}%, Sh={om['sharpe_ratio']:.3f}, "
          f"WR={om['win_rate_pct']:.1f}%, Trades={om['total_trades']}, "
          f"DD={om['max_drawdown_pct']:.3f}%, PF={om['profit_factor']:.3f}")

    # Test ORB variants
    print(f"\n{'Config':<45} {'Ret':>8} {'Sharpe':>8} {'WR':>6} {'Trades':>6} {'DD':>8} {'PF':>6}")
    print("-" * 90)

    # Current best
    print(f"{'ORB Baseline (profit_ratio=3.5)':<45} {om['total_return_pct']:>+7.3f}% {om['sharpe_ratio']:>8.3f} "
          f"{om['win_rate_pct']:>5.1f}% {om['total_trades']:>6} {om['max_drawdown_pct']:>7.3f}% {om['profit_factor']:>5.3f}")

    # Test profit_ratio variants (within ±20% = 2.8~4.2)
    for pr in [3.0, 3.2, 3.8, 4.0, 4.2]:
        r = engine.run(ORBStrategy(params={"profit_ratio": pr}), df)
        m = r.metrics
        print(f"{'ORB pr='+str(pr):<45} {m['total_return_pct']:>+7.3f}% {m['sharpe_ratio']:>8.3f} "
              f"{m['win_rate_pct']:>5.1f}% {m['total_trades']:>6} {m['max_drawdown_pct']:>7.3f}% {m['profit_factor']:>5.3f}")

    # Test trailing_pct variants (within ±20% = 0.012~0.018)
    print()
    for tp in [0.012, 0.013, 0.018]:
        r = engine.run(ORBStrategy(params={"trailing_pct": tp}), df)
        m = r.metrics
        print(f"{'ORB trail='+str(tp):<45} {m['total_return_pct']:>+7.3f}% {m['sharpe_ratio']:>8.3f} "
              f"{m['win_rate_pct']:>5.1f}% {m['total_trades']:>6} {m['max_drawdown_pct']:>7.3f}% {m['profit_factor']:>5.3f}")

    # Test close_before_min variants
    print()
    for cbm in [5, 8, 12, 15]:
        r = engine.run(ORBStrategy(params={"close_before_min": cbm}), df)
        m = r.metrics
        print(f"{'ORB cbm='+str(cbm):<45} {m['total_return_pct']:>+7.3f}% {m['sharpe_ratio']:>8.3f} "
              f"{m['win_rate_pct']:>5.1f}% {m['total_trades']:>6} {m['max_drawdown_pct']:>7.3f}% {m['profit_factor']:>5.3f}")

    # Test VWAP variants too
    print("\n--- VWAP Variants ---")
    vwap_base = engine.run(VWAPReversionStrategy(), df)
    vm = vwap_base.metrics
    print(f"{'VWAP Baseline':<45} {vm['total_return_pct']:>+7.3f}% {vm['sharpe_ratio']:>8.3f} "
          f"{vm['win_rate_pct']:>5.1f}% {vm['total_trades']:>6} {vm['max_drawdown_pct']:>7.3f}% {vm['profit_factor']:>5.3f}")

    # k variants (±20% of 1.5 = 1.2~1.8)
    for k_val in [1.3, 1.4, 1.6, 1.7]:
        r = engine.run(VWAPReversionStrategy(params={"k": k_val}), df)
        m = r.metrics
        print(f"{'VWAP k='+str(k_val):<45} {m['total_return_pct']:>+7.3f}% {m['sharpe_ratio']:>8.3f} "
              f"{m['win_rate_pct']:>5.1f}% {m['total_trades']:>6} {m['max_drawdown_pct']:>7.3f}% {m['profit_factor']:>5.3f}")

    # combo impact of best variants
    print("\n--- Combined Impact (Active Reuse + Kelly) ---")
    best_combos = [
        ("Both baseline", {}, {}),
    ]
    
    for label, orb_p, vwap_p in best_combos:
        orb_r = engine.run(ORBStrategy(params=orb_p if orb_p else None), df)
        vwap_r = engine.run(VWAPReversionStrategy(params=vwap_p if vwap_p else None), df)
        combo = combine_results_active_reuse_kelly(orb_r, vwap_r, active_weight=0.8)
        cm = compute_metrics(combo)
        print(f"{label:<45} Combo: Sh={cm['sharpe']:.3f} Sort={cm['sortino']:.3f} DD={cm['max_dd_pct']:.3f}%")

    print("\n-- Test finished --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
