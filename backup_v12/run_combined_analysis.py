#!/usr/bin/env python3
"""
Combined Portfolio Analysis + Out-of-Sample Validation
======================================================
1. Run ORB + VWAP with optimal params on full dataset
2. Compute combined portfolio equity (configurable allocation)
3. Split into in-sample (first 40 days) / out-of-sample (last 20 days)
4. Print comparison table
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetch_nq_data
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine, BacktestResult


def compute_metrics(equity: np.ndarray, trades: list = None) -> dict:
    """Compute standard metrics from equity curve array."""
    ret_pct = (equity[-1] / equity[0] - 1) * 100
    returns = pd.Series(equity).pct_change().dropna()

    sharpe = 0.0
    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * (252 * 78) ** 0.5)

    sortino = 0.0
    downside = returns[returns < 0].std()
    if downside > 0:
        sortino = float(returns.mean() / downside * (252 * 78) ** 0.5)

    peak = pd.Series(equity).cummax()
    dd = (pd.Series(equity) - peak) / peak
    max_dd = float(dd.min() * 100)

    total_trades = len(trades) if trades else 0
    wins = [t for t in (trades or []) if t.get("pnl", 0) > 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    return {
        "return_pct": round(ret_pct, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd, 4),
        "win_rate": round(win_rate, 1),
        "trades": total_trades,
    }


def run_strategy_on_data(strategy_cls, params, df, engine):
    """Run a strategy and return BacktestResult."""
    strategy = strategy_cls(params=params)
    return engine.run(strategy, df)


def combine_equity(eq1, eq2, w1=0.6, w2=0.4):
    """Combine two equity curves with allocation weights."""
    arr1 = np.array(eq1)
    arr2 = np.array(eq2)
    n = min(len(arr1), len(arr2))

    # Normalize to returns
    ret1 = arr1[:n] / arr1[0]
    ret2 = arr2[:n] / arr2[0]

    combined = w1 * ret1 + w2 * ret2
    return combined * 100000  # Scale back to initial capital


def main():
    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=5.0)

    # Run both strategies with best params
    orb_result = run_strategy_on_data(ORBStrategy, None, df, engine)
    vwap_result = run_strategy_on_data(VWAPReversionStrategy, None, df, engine)

    print("\n" + "=" * 70)
    print("FULL DATASET RESULTS")
    print("=" * 70)

    orb_m = orb_result.metrics
    vwap_m = vwap_result.metrics

    print(f"ORB:  Return={orb_m['total_return_pct']:+.4f}%, Sharpe={orb_m['sharpe_ratio']:.3f}, "
          f"WR={orb_m['win_rate_pct']:.1f}%, Trades={orb_m['total_trades']}, DD={orb_m['max_drawdown_pct']:.3f}%")
    print(f"VWAP: Return={vwap_m['total_return_pct']:+.4f}%, Sharpe={vwap_m['sharpe_ratio']:.3f}, "
          f"WR={vwap_m['win_rate_pct']:.1f}%, Trades={vwap_m['total_trades']}, DD={vwap_m['max_drawdown_pct']:.3f}%")

    # Combined portfolio analysis
    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO ANALYSIS")
    print("=" * 70)

    allocations = [(0.6, 0.4), (0.5, 0.5), (0.7, 0.3)]

    for w_orb, w_vwap in allocations:
        combined_eq = combine_equity(orb_result.equity_curve, vwap_result.equity_curve, w_orb, w_vwap)
        cm = compute_metrics(combined_eq)
        print(f"ORB {int(w_orb*100)}% / VWAP {int(w_vwap*100)}%: "
              f"Return={cm['return_pct']:+.4f}%, Sharpe={cm['sharpe']:.3f}, "
              f"Sortino={cm['sortino']:.3f}, MaxDD={cm['max_dd_pct']:.3f}%")

    # OOS Validation
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    # Split by trading days
    dates = sorted(set(df.index.date))
    n_days = len(dates)
    split_idx = int(n_days * 2 / 3)  # ~40 days IS, ~20 days OOS
    split_date = dates[split_idx]

    df_is = df[df.index.date < split_date]
    df_oos = df[df.index.date >= split_date]

    print(f"\nIn-Sample:  {dates[0]} ~ {dates[split_idx-1]} ({split_idx} days, {len(df_is)} bars)")
    print(f"Out-of-Sample: {split_date} ~ {dates[-1]} ({n_days - split_idx} days, {len(df_oos)} bars)")

    # IS results
    orb_is = run_strategy_on_data(ORBStrategy, None, df_is, engine)
    vwap_is = run_strategy_on_data(VWAPReversionStrategy, None, df_is, engine)

    # OOS results
    orb_oos = run_strategy_on_data(ORBStrategy, None, df_oos, engine)
    vwap_oos = run_strategy_on_data(VWAPReversionStrategy, None, df_oos, engine)

    # Buy & Hold
    bh_is = (df_is["Close"].iloc[-1] / df_is["Close"].iloc[0] - 1) * 100
    bh_oos = (df_oos["Close"].iloc[-1] / df_oos["Close"].iloc[0] - 1) * 100

    print(f"\n{'Strategy':<20} {'IS Return':>10} {'IS Sharpe':>10} {'IS WR':>8} {'IS Trades':>10}")
    print("-" * 60)
    om = orb_is.metrics
    vm = vwap_is.metrics
    print(f"{'ORB':<20} {om['total_return_pct']:>+9.4f}% {om['sharpe_ratio']:>10.3f} {om['win_rate_pct']:>7.1f}% {om['total_trades']:>10}")
    print(f"{'VWAP':<20} {vm['total_return_pct']:>+9.4f}% {vm['sharpe_ratio']:>10.3f} {vm['win_rate_pct']:>7.1f}% {vm['total_trades']:>10}")
    print(f"{'Buy & Hold':<20} {bh_is:>+9.4f}%")

    print(f"\n{'Strategy':<20} {'OOS Return':>10} {'OOS Sharpe':>10} {'OOS WR':>8} {'OOS Trades':>10}")
    print("-" * 60)
    om = orb_oos.metrics
    vm = vwap_oos.metrics
    print(f"{'ORB':<20} {om['total_return_pct']:>+9.4f}% {om['sharpe_ratio']:>10.3f} {om['win_rate_pct']:>7.1f}% {om['total_trades']:>10}")
    print(f"{'VWAP':<20} {vm['total_return_pct']:>+9.4f}% {vm['sharpe_ratio']:>10.3f} {vm['win_rate_pct']:>7.1f}% {vm['total_trades']:>10}")
    print(f"{'Buy & Hold':<20} {bh_oos:>+9.4f}%")

    # Combined OOS
    if len(orb_oos.equity_curve) > 0 and len(vwap_oos.equity_curve) > 0:
        combined_oos = combine_equity(orb_oos.equity_curve, vwap_oos.equity_curve, 0.6, 0.4)
        cm_oos = compute_metrics(combined_oos)
        print(f"\n{'Combined 60/40 OOS':<20} {cm_oos['return_pct']:>+9.4f}% {cm_oos['sharpe']:>10.3f} {'':>8} {'':>10}")

    # Save results
    results = {
        "full": {
            "orb": orb_result.metrics,
            "vwap": vwap_result.metrics,
        },
        "in_sample": {
            "orb": orb_is.metrics,
            "vwap": vwap_is.metrics,
            "buy_hold": round(bh_is, 4),
        },
        "out_of_sample": {
            "orb": orb_oos.metrics,
            "vwap": vwap_oos.metrics,
            "buy_hold": round(bh_oos, 4),
        },
        "split_date": str(split_date),
        "is_days": split_idx,
        "oos_days": n_days - split_idx,
    }

    with open(PROJECT_ROOT / "combined_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to combined_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
