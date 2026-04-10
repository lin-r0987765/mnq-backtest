#!/usr/bin/env python3
"""
Equity Curve Momentum Filter 測試
===================================
用組合權益曲線的 N-bar 動量判斷是否處於「良好期」。
良好期：正常倉位；不良期：減半倉位。

同時測試第二個方向：Win-streak sizing
連勝中逐步加碼，連敗時減碼。
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
from src.portfolio_overlay import (
    combine_results_active_reuse,
    combine_results_active_reuse_kelly,
    position_mask_from_result,
    _apply_kelly_to_equity,
)


def compute_metrics(equity):
    arr = np.array(equity)
    if len(arr) < 2:
        return {"return_pct": 0, "sharpe": 0, "sortino": 0, "max_dd_pct": 0}
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


def combine_eq_momentum(
    result1, result2,
    base_weight=0.8,
    momentum_window=200,  # bars (200 = ~2.5 trading days of 5m bars)
    reduced_weight=0.4,
    both_weights=(0.5, 0.5),
    initial_cash=100_000.0,
    kelly_mult=0.0,
    lookback_trades=10,
):
    """
    Equity curve momentum: when combined equity > N-bar SMA → full weight;
    below SMA → reduced weight.
    """
    if kelly_mult > 0:
        eq1 = _apply_kelly_to_equity(
            result1.equity_curve, result1.trades,
            lookback_trades=lookback_trades, kelly_mult=kelly_mult,
        )
        eq2 = _apply_kelly_to_equity(
            result2.equity_curve, result2.trades,
            lookback_trades=lookback_trades, kelly_mult=kelly_mult,
        )
    else:
        eq1 = np.asarray(result1.equity_curve, dtype=float)
        eq2 = np.asarray(result2.equity_curve, dtype=float)

    mask1 = position_mask_from_result(result1)
    mask2 = position_mask_from_result(result2)

    n = min(len(eq1), len(eq2), len(mask1), len(mask2))
    ret1 = eq1[:n] / eq1[0]
    ret2 = eq2[:n] / eq2[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])
    combined_ret = np.ones(n)

    # Running SMA of combined equity
    eq_history = []

    for i in range(1, n):
        eq_history.append(combined_ret[i-1])

        # Compute momentum: is equity above its N-bar SMA?
        if len(eq_history) >= momentum_window:
            sma = np.mean(eq_history[-momentum_window:])
            above_sma = combined_ret[i-1] >= sma
        else:
            above_sma = True  # not enough history, default to full

        active_w = base_weight if above_sma else reduced_weight

        if mask1[i] and mask2[i]:
            w1, w2 = both_weights
        elif mask1[i] and not mask2[i]:
            w1, w2 = active_w, 0.0
        elif mask2[i] and not mask1[i]:
            w1, w2 = 0.0, active_w
        else:
            w1, w2 = both_weights

        combined_ret[i] = combined_ret[i-1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * initial_cash


def combine_streak_sizing(
    result1, result2,
    base_weight=0.8,
    win_streak_bonus=0.1,  # per win, add this to weight
    lose_streak_penalty=0.15, # per loss, subtract this
    max_weight=1.2,
    min_weight=0.3,
    both_weights=(0.5, 0.5),
    initial_cash=100_000.0,
    kelly_mult=0.0,
    lookback_trades=10,
):
    """Win-streak sizing: increase weight after wins, decrease after losses."""
    if kelly_mult > 0:
        eq1 = _apply_kelly_to_equity(
            result1.equity_curve, result1.trades,
            lookback_trades=lookback_trades, kelly_mult=kelly_mult,
        )
        eq2 = _apply_kelly_to_equity(
            result2.equity_curve, result2.trades,
            lookback_trades=lookback_trades, kelly_mult=kelly_mult,
        )
    else:
        eq1 = np.asarray(result1.equity_curve, dtype=float)
        eq2 = np.asarray(result2.equity_curve, dtype=float)

    mask1 = position_mask_from_result(result1)
    mask2 = position_mask_from_result(result2)

    n = min(len(eq1), len(eq2), len(mask1), len(mask2))
    ret1 = eq1[:n] / eq1[0]
    ret2 = eq2[:n] / eq2[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])
    combined_ret = np.ones(n)

    # Track trade-level outcomes from combined equity
    in_position = False
    entry_val = 0.0
    streak = 0  # positive = consecutive wins, negative = consecutive losses
    active_w = base_weight

    for i in range(1, n):
        is_active = mask1[i] or mask2[i]

        if is_active and not in_position:
            in_position = True
            entry_val = combined_ret[i-1]
            # Compute weight based on streak
            active_w = base_weight + streak * (win_streak_bonus if streak > 0 else lose_streak_penalty)
            active_w = max(min_weight, min(max_weight, active_w))

        elif not is_active and in_position:
            in_position = False
            pnl = combined_ret[i-1] - entry_val
            if pnl > 0:
                streak = max(streak, 0) + 1
            else:
                streak = min(streak, 0) - 1

        if mask1[i] and mask2[i]:
            w1, w2 = both_weights
        elif mask1[i] and not mask2[i]:
            w1, w2 = active_w, 0.0
        elif mask2[i] and not mask1[i]:
            w1, w2 = 0.0, active_w
        else:
            w1, w2 = both_weights

        combined_ret[i] = combined_ret[i-1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * initial_cash


def main():
    print("=" * 70)
    print("Equity Curve Momentum + Streak Sizing Test")
    print("=" * 70)

    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    orb_r = engine.run(ORBStrategy(), df)
    vwap_r = engine.run(VWAPReversionStrategy(), df)

    base_eq = combine_results_active_reuse(orb_r, vwap_r, active_weight=0.8)
    base_m = compute_metrics(base_eq)
    kelly_eq = combine_results_active_reuse_kelly(orb_r, vwap_r, active_weight=0.8)
    kelly_m = compute_metrics(kelly_eq)

    print(f"\n{'Config':<55} {'Ret':>8} {'Sharpe':>8} {'Sort':>8} {'MaxDD':>8}")
    print("-" * 90)
    print(f"{'Baseline AR 80%':<55} {base_m['return_pct']:>+7.3f}% {base_m['sharpe']:>8.3f} {base_m['sortino']:>8.3f} {base_m['max_dd_pct']:>7.3f}%")
    print(f"{'Kelly AR 80% (iter 25)':<55} {kelly_m['return_pct']:>+7.3f}% {kelly_m['sharpe']:>8.3f} {kelly_m['sortino']:>8.3f} {kelly_m['max_dd_pct']:>7.3f}%")

    # Equity curve momentum tests
    print("\n-- Equity Curve Momentum --")
    for window in [100, 200, 400, 780]:
        for red_w in [0.4, 0.5]:
            eq = combine_eq_momentum(orb_r, vwap_r, momentum_window=window, reduced_weight=red_w)
            m = compute_metrics(eq)
            label = f"EQ Momentum w={window} red={red_w}"
            print(f"{label:<55} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")

    # Kelly + Momentum
    print("\n-- Kelly + Equity Curve Momentum --")
    for window in [200, 400]:
        for red_w in [0.4, 0.5]:
            eq = combine_eq_momentum(orb_r, vwap_r, momentum_window=window, reduced_weight=red_w, kelly_mult=0.5)
            m = compute_metrics(eq)
            label = f"K+EQ w={window} red={red_w}"
            print(f"{label:<55} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")

    # Streak sizing
    print("\n-- Win-Streak Sizing --")
    for bonus in [0.05, 0.1]:
        for penalty in [0.1, 0.15]:
            eq = combine_streak_sizing(orb_r, vwap_r, win_streak_bonus=bonus, lose_streak_penalty=penalty)
            m = compute_metrics(eq)
            label = f"Streak b={bonus} p={penalty}"
            print(f"{label:<55} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")

    # Kelly + Streak
    print("\n-- Kelly + Streak --")
    for bonus, penalty in [(0.05, 0.1), (0.1, 0.15)]:
        eq = combine_streak_sizing(orb_r, vwap_r, win_streak_bonus=bonus, lose_streak_penalty=penalty, kelly_mult=0.5)
        m = compute_metrics(eq)
        label = f"K+Streak b={bonus} p={penalty}"
        print(f"{label:<55} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")

    print("\n-- Test finished --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
