#!/usr/bin/env python3
"""
Kelly Criterion 動態倉位測試
============================
測試在組合層引入 Kelly sizing 是否改善風險調整後報酬。

核心思路：
- 根據近期 rolling window 的勝率和盈虧比計算 Kelly fraction
- 用 half-Kelly（保守版）調整每筆交易的 size
- 在 equity curve 層面模擬倉位調整效果
"""
from __future__ import annotations
import sys, json
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
    combine_equity_static,
    combine_results_active_reuse,
    position_mask_from_result,
)


def compute_metrics(equity):
    """Compute metrics from equity curve array."""
    arr = np.array(equity)
    if len(arr) < 2:
        return {"return_pct": 0, "sharpe": 0, "sortino": 0, "max_dd_pct": 0}
    ret_pct = (arr[-1] / arr[0] - 1) * 100
    returns = pd.Series(arr).pct_change().dropna()
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * (252 * 78) ** 0.5)
    sortino = 0.0
    downside = returns[returns < 0].std()
    if downside > 0:
        sortino = float(returns.mean() / downside * (252 * 78) ** 0.5)
    peak = pd.Series(arr).cummax()
    dd = (pd.Series(arr) - peak) / peak
    max_dd = float(dd.min() * 100)
    return {
        "return_pct": round(ret_pct, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 4),
    }


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion fraction.
    f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = avg_win / avg_loss
    """
    if avg_loss <= 0 or win_rate <= 0 or avg_win <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss
    f = (p * b - q) / b
    return max(0.0, min(f, 1.0))  # clamp to [0, 1]


def apply_kelly_to_equity(equity_curve, trades, 
                          lookback_trades=10, 
                          kelly_mult=0.5,  # half-Kelly
                          min_size_mult=0.3,
                          max_size_mult=1.5,
                          initial_cash=100_000.0):
    """
    Apply Kelly criterion sizing to an equity curve.
    
    Uses rolling trade history to compute Kelly fraction,
    then scales the return contribution of each trade period.
    
    Parameters:
    - lookback_trades: number of recent trades to compute Kelly from
    - kelly_mult: Kelly fraction multiplier (0.5 = half-Kelly, conservative)
    - min_size_mult: minimum size multiplier (floor)
    - max_size_mult: maximum size multiplier (ceiling)
    """
    eq = np.array(equity_curve, dtype=float)
    n = len(eq)
    if n < 2 or not trades:
        return eq.copy()
    
    # Extract trade PnLs
    trade_pnls = []
    for t in trades:
        pnl = 0.0
        if "pnl" in t:
            pnl = float(t.get("pnl", 0) or 0)
        elif "PnL" in t:
            pnl = float(t.get("PnL", 0) or 0)
        elif "Return" in t:
            pnl = float(t.get("Return", 0) or 0)
        trade_pnls.append(pnl)
    
    if not trade_pnls:
        return eq.copy()
    
    # Compute returns from equity curve
    daily_returns = np.diff(eq) / eq[:-1]
    
    # Identify "trade regions" - periods where equity is changing
    in_trade = np.abs(daily_returns) > 1e-12
    
    # Build kelly sizing schedule
    # For each trade, compute rolling Kelly based on previous trades
    completed_trades = 0
    kelly_sizes = []  # (start_bar, size_mult) pairs
    
    current_trade_start = None
    prev_eq = eq[0]
    
    for i in range(1, n):
        is_active = abs(eq[i] - eq[i-1]) > 0.01  # equity changed = in trade
        
        if is_active and current_trade_start is None:
            # New trade starts
            current_trade_start = i
            
            # Compute Kelly fraction from recent trades
            if completed_trades >= 3:  # need minimum history
                recent = trade_pnls[max(0, completed_trades - lookback_trades):completed_trades]
                wins = [p for p in recent if p > 0]
                losses = [abs(p) for p in recent if p <= 0]
                
                if wins and losses:
                    wr = len(wins) / len(recent)
                    avg_w = np.mean(wins)
                    avg_l = np.mean(losses)
                    kf = kelly_fraction(wr, avg_w, avg_l)
                    size_mult = max(min_size_mult, min(max_size_mult, kf * kelly_mult / 0.5))
                else:
                    size_mult = 1.0
            else:
                size_mult = 1.0  # default until enough history
            
            kelly_sizes.append((i, size_mult))
            
        elif not is_active and current_trade_start is not None:
            # Trade ended
            current_trade_start = None
            completed_trades += 1
    
    # Apply Kelly sizing to equity curve
    if not kelly_sizes:
        return eq.copy()
    
    kelly_eq = np.full(n, initial_cash)
    cash = initial_cash
    
    # Build a per-bar size multiplier array
    size_mults = np.ones(n)
    current_mult = 1.0
    size_idx = 0
    
    for i in range(n):
        while size_idx < len(kelly_sizes) and kelly_sizes[size_idx][0] <= i:
            current_mult = kelly_sizes[size_idx][1]
            size_idx += 1
        size_mults[i] = current_mult
    
    # Re-compute equity with scaled returns
    kelly_eq[0] = initial_cash
    for i in range(1, n):
        base_return = (eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0
        scaled_return = base_return * size_mults[i]
        kelly_eq[i] = kelly_eq[i-1] * (1 + scaled_return)
    
    return kelly_eq


def combine_with_kelly_reuse(
    result1, result2,
    active_weight=0.8,
    kelly_mult=0.5,
    lookback_trades=10,
    min_size_mult=0.3,
    max_size_mult=1.5,
    initial_cash=100_000.0,
):
    """
    Combined portfolio with Active Reuse + Kelly sizing.
    Apply Kelly to individual strategies, then combine.
    """
    eq1_kelly = apply_kelly_to_equity(
        result1.equity_curve, result1.trades,
        lookback_trades=lookback_trades,
        kelly_mult=kelly_mult,
        min_size_mult=min_size_mult,
        max_size_mult=max_size_mult,
        initial_cash=initial_cash,
    )
    eq2_kelly = apply_kelly_to_equity(
        result2.equity_curve, result2.trades,
        lookback_trades=lookback_trades,
        kelly_mult=kelly_mult,
        min_size_mult=min_size_mult,
        max_size_mult=max_size_mult,
        initial_cash=initial_cash,
    )
    
    # Get position masks
    mask1 = position_mask_from_result(result1)
    mask2 = position_mask_from_result(result2)
    
    n = min(len(eq1_kelly), len(eq2_kelly), len(mask1), len(mask2))
    
    ret1 = eq1_kelly[:n] / eq1_kelly[0]
    ret2 = eq2_kelly[:n] / eq2_kelly[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])
    combined_ret = np.ones(n)
    
    for i in range(1, n):
        if mask1[i] and mask2[i]:
            w1, w2 = 0.5, 0.5
        elif mask1[i] and not mask2[i]:
            w1, w2 = active_weight, 0.0
        elif mask2[i] and not mask1[i]:
            w1, w2 = 0.0, active_weight
        else:
            w1, w2 = 0.5, 0.5
        combined_ret[i] = combined_ret[i - 1] + w1 * dret1[i] + w2 * dret2[i]
    
    return combined_ret * initial_cash


def main():
    print("=" * 70)
    print("Kelly Criterion 動態倉位測試")
    print("=" * 70)
    
    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    
    # Run strategies
    orb = ORBStrategy()
    vwap = VWAPReversionStrategy()
    orb_result = engine.run(orb, df)
    vwap_result = engine.run(vwap, df)
    
    orb_m = orb_result.metrics
    vwap_m = vwap_result.metrics
    
    print(f"\nBaseline ORB:  Ret={orb_m['total_return_pct']:+.3f}%, Sh={orb_m['sharpe_ratio']:.3f}, WR={orb_m['win_rate_pct']:.1f}%, MaxDD={orb_m['max_drawdown_pct']:.3f}%")
    print(f"Baseline VWAP: Ret={vwap_m['total_return_pct']:+.3f}%, Sh={vwap_m['sharpe_ratio']:.3f}, WR={vwap_m['win_rate_pct']:.1f}%, MaxDD={vwap_m['max_drawdown_pct']:.3f}%")
    
    # Baseline combined
    baseline_combo = combine_results_active_reuse(orb_result, vwap_result, active_weight=0.8)
    baseline_m = compute_metrics(baseline_combo)
    print(f"\nBaseline Active Reuse 80%: Ret={baseline_m['return_pct']:+.3f}%, Sh={baseline_m['sharpe']:.3f}, Sort={baseline_m['sortino']:.3f}, MaxDD={baseline_m['max_dd_pct']:.3f}%")
    
    # Test Kelly variants
    print("\n" + "=" * 70)
    print("Kelly Criterion 測試結果")
    print("=" * 70)
    
    print(f"\n{'Config':<40} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
    print("-" * 75)
    print(f"{'Baseline Active Reuse 80%':<40} {baseline_m['return_pct']:>+7.3f}% {baseline_m['sharpe']:>8.3f} {baseline_m['sortino']:>8.3f} {baseline_m['max_dd_pct']:>7.3f}%")
    
    results = {}
    
    # Test different Kelly configurations
    configs = [
        {"kelly_mult": 0.25, "lookback": 8, "min_size": 0.3, "max_size": 1.3, "label": "Quarter-Kelly lb=8"},
        {"kelly_mult": 0.5, "lookback": 8, "min_size": 0.3, "max_size": 1.5, "label": "Half-Kelly lb=8"},
        {"kelly_mult": 0.5, "lookback": 10, "min_size": 0.3, "max_size": 1.5, "label": "Half-Kelly lb=10"},
        {"kelly_mult": 0.5, "lookback": 15, "min_size": 0.3, "max_size": 1.5, "label": "Half-Kelly lb=15"},
        {"kelly_mult": 0.5, "lookback": 10, "min_size": 0.5, "max_size": 1.3, "label": "Half-Kelly conservative"},
        {"kelly_mult": 0.75, "lookback": 10, "min_size": 0.3, "max_size": 1.5, "label": "3/4-Kelly lb=10"},
    ]
    
    for cfg in configs:
        combo_kelly = combine_with_kelly_reuse(
            orb_result, vwap_result,
            active_weight=0.8,
            kelly_mult=cfg["kelly_mult"],
            lookback_trades=cfg["lookback"],
            min_size_mult=cfg["min_size"],
            max_size_mult=cfg["max_size"],
        )
        m = compute_metrics(combo_kelly)
        label = cfg["label"]
        print(f"{label:<40} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")
        results[label] = m
    
    # Also test Kelly on individual strategies
    print("\n" + "-" * 75)
    print("Kelly 個別策略效果:")
    print(f"\n{'Config':<40} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
    print("-" * 75)
    
    for strat_name, result in [("ORB", orb_result), ("VWAP", vwap_result)]:
        # Baseline
        eq_base = np.array(result.equity_curve)
        m_base = compute_metrics(eq_base)
        print(f"{strat_name + ' (baseline)':<40} {m_base['return_pct']:>+7.3f}% {m_base['sharpe']:>8.3f} {m_base['sortino']:>8.3f} {m_base['max_dd_pct']:>7.3f}%")
        
        # Half-Kelly
        eq_kelly = apply_kelly_to_equity(
            result.equity_curve, result.trades,
            lookback_trades=10, kelly_mult=0.5,
            min_size_mult=0.3, max_size_mult=1.5,
        )
        m_kelly = compute_metrics(eq_kelly)
        print(f"{strat_name + ' (half-Kelly lb=10)':<40} {m_kelly['return_pct']:>+7.3f}% {m_kelly['sharpe']:>8.3f} {m_kelly['sortino']:>8.3f} {m_kelly['max_dd_pct']:>7.3f}%")
    
    # OOS validation of best Kelly config
    print("\n" + "=" * 70)
    print("OOS 驗證 (Kelly vs Baseline)")
    print("=" * 70)
    
    dates = sorted(set(df.index.date))
    n_days = len(dates)
    split_idx = int(n_days * 2 / 3)
    split_date = dates[split_idx]
    
    df_is = df[df.index.date < split_date]
    df_oos = df[df.index.date >= split_date]
    
    print(f"\nIS: {dates[0]}~{dates[split_idx-1]} ({split_idx}d)")
    print(f"OOS: {split_date}~{dates[-1]} ({n_days - split_idx}d)")
    
    # IS
    orb_is = engine.run(ORBStrategy(), df_is)
    vwap_is = engine.run(VWAPReversionStrategy(), df_is)
    
    # OOS
    orb_oos = engine.run(ORBStrategy(), df_oos)
    vwap_oos = engine.run(VWAPReversionStrategy(), df_oos)
    
    # Baseline OOS combo
    baseline_oos = combine_results_active_reuse(orb_oos, vwap_oos, active_weight=0.8)
    baseline_oos_m = compute_metrics(baseline_oos)
    
    # Kelly OOS combo - test top 2 configs
    print(f"\n{'Config':<40} {'OOS Ret':>8} {'OOS Sh':>8} {'OOS Sort':>8} {'OOS DD':>8}")
    print("-" * 75)
    print(f"{'Baseline Active Reuse 80%':<40} {baseline_oos_m['return_pct']:>+7.3f}% {baseline_oos_m['sharpe']:>8.3f} {baseline_oos_m['sortino']:>8.3f} {baseline_oos_m['max_dd_pct']:>7.3f}%")
    
    for cfg in configs[:4]:  # test first 4 configs
        combo_kelly_oos = combine_with_kelly_reuse(
            orb_oos, vwap_oos,
            active_weight=0.8,
            kelly_mult=cfg["kelly_mult"],
            lookback_trades=cfg["lookback"],
            min_size_mult=cfg["min_size"],
            max_size_mult=cfg["max_size"],
        )
        m = compute_metrics(combo_kelly_oos)
        label = cfg["label"]
        print(f"{label:<40} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")
    
    print("\n-- Kelly 測試完成 --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
