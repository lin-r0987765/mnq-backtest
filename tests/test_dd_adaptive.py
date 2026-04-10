#!/usr/bin/env python3
"""
Drawdown-Adaptive Active Reuse 測試
=====================================
當組合處於回撤時，動態降低 active_weight，保護資金。

思路：
- 正常狀態 (equity >= rolling peak): active_weight = 0.8 (標準)
- 輕微回撤 (equity 低於 peak 0.02%~0.05%): active_weight = 0.5
- 深度回撤 (equity 低於 peak > 0.05%): active_weight = 0.3
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


def combine_dd_adaptive(
    result1, result2,
    base_weight=0.8,
    dd_threshold_light=0.0002,  # 0.02% drawdown
    dd_threshold_deep=0.0005,   # 0.05% drawdown
    weight_light=0.5,
    weight_deep=0.3,
    both_weights=(0.5, 0.5),
    initial_cash=100_000.0,
    kelly_mult=0.0,  # 0 = no Kelly
    lookback_trades=10,
):
    """Drawdown-adaptive active reuse."""
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
    combined_peak = 1.0

    for i in range(1, n):
        # Current drawdown from peak
        dd = (combined_ret[i-1] - combined_peak) / combined_peak if combined_peak > 0 else 0

        # Adaptive weight selection
        if dd < -dd_threshold_deep:
            active_w = weight_deep
        elif dd < -dd_threshold_light:
            active_w = weight_light
        else:
            active_w = base_weight

        if mask1[i] and mask2[i]:
            w1, w2 = both_weights
        elif mask1[i] and not mask2[i]:
            w1, w2 = active_w, 0.0
        elif mask2[i] and not mask1[i]:
            w1, w2 = 0.0, active_w
        else:
            w1, w2 = both_weights

        combined_ret[i] = combined_ret[i-1] + w1 * dret1[i] + w2 * dret2[i]
        combined_peak = max(combined_peak, combined_ret[i])

    return combined_ret * initial_cash


def main():
    print("=" * 70)
    print("Drawdown-Adaptive Active Reuse Test")
    print("=" * 70)

    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    orb_result = engine.run(ORBStrategy(), df)
    vwap_result = engine.run(VWAPReversionStrategy(), df)

    # Baselines
    base_eq = combine_results_active_reuse(orb_result, vwap_result, active_weight=0.8)
    base_m = compute_metrics(base_eq)
    
    kelly_eq = combine_results_active_reuse_kelly(orb_result, vwap_result, active_weight=0.8)
    kelly_m = compute_metrics(kelly_eq)

    print(f"\n{'Config':<50} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
    print("-" * 85)
    print(f"{'Baseline AR 80%':<50} {base_m['return_pct']:>+7.3f}% {base_m['sharpe']:>8.3f} {base_m['sortino']:>8.3f} {base_m['max_dd_pct']:>7.3f}%")
    print(f"{'Kelly AR 80% (iter 25)':<50} {kelly_m['return_pct']:>+7.3f}% {kelly_m['sharpe']:>8.3f} {kelly_m['sortino']:>8.3f} {kelly_m['max_dd_pct']:>7.3f}%")

    # Test DD-adaptive configs
    configs = [
        # (label, dd_light, dd_deep, w_light, w_deep, kelly)
        ("DD-Adaptive (0.02%/0.05%, 0.5/0.3)", 0.0002, 0.0005, 0.5, 0.3, 0.0),
        ("DD-Adaptive (0.03%/0.08%, 0.5/0.3)", 0.0003, 0.0008, 0.5, 0.3, 0.0),
        ("DD-Adaptive (0.02%/0.05%, 0.6/0.4)", 0.0002, 0.0005, 0.6, 0.4, 0.0),
        ("DD-Adaptive tight (0.01%/0.03%, 0.5/0.3)", 0.0001, 0.0003, 0.5, 0.3, 0.0),
        # Kelly + DD-Adaptive
        ("Kelly + DD-Adaptive (0.02%/0.05%)", 0.0002, 0.0005, 0.5, 0.3, 0.5),
        ("Kelly + DD-Adaptive (0.03%/0.08%)", 0.0003, 0.0008, 0.5, 0.3, 0.5),
        ("Kelly + DD-Adaptive tight", 0.0001, 0.0003, 0.5, 0.3, 0.5),
    ]

    for label, dd_l, dd_d, w_l, w_d, km in configs:
        eq = combine_dd_adaptive(
            orb_result, vwap_result,
            dd_threshold_light=dd_l, dd_threshold_deep=dd_d,
            weight_light=w_l, weight_deep=w_d,
            kelly_mult=km,
        )
        m = compute_metrics(eq)
        print(f"{label:<50} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")

    # OOS test for top configs
    print("\n" + "=" * 70)
    print("OOS Validation")
    print("=" * 70)

    dates = sorted(set(df.index.date))
    n_days = len(dates)
    split_idx = int(n_days * 2 / 3)
    split_date = dates[split_idx]
    df_oos = df[df.index.date >= split_date]

    orb_oos = engine.run(ORBStrategy(), df_oos)
    vwap_oos = engine.run(VWAPReversionStrategy(), df_oos)

    base_oos = combine_results_active_reuse(orb_oos, vwap_oos, active_weight=0.8)
    base_oos_m = compute_metrics(base_oos)
    kelly_oos = combine_results_active_reuse_kelly(orb_oos, vwap_oos, active_weight=0.8)
    kelly_oos_m = compute_metrics(kelly_oos)

    print(f"\n{'Config':<50} {'OOS Ret':>8} {'OOS Sh':>8} {'OOS Sort':>8} {'OOS DD':>8}")
    print("-" * 85)
    print(f"{'Baseline AR 80%':<50} {base_oos_m['return_pct']:>+7.3f}% {base_oos_m['sharpe']:>8.3f} {base_oos_m['sortino']:>8.3f} {base_oos_m['max_dd_pct']:>7.3f}%")
    print(f"{'Kelly AR 80%':<50} {kelly_oos_m['return_pct']:>+7.3f}% {kelly_oos_m['sharpe']:>8.3f} {kelly_oos_m['sortino']:>8.3f} {kelly_oos_m['max_dd_pct']:>7.3f}%")

    for label, dd_l, dd_d, w_l, w_d, km in configs:
        eq = combine_dd_adaptive(
            orb_oos, vwap_oos,
            dd_threshold_light=dd_l, dd_threshold_deep=dd_d,
            weight_light=w_l, weight_deep=w_d,
            kelly_mult=km,
        )
        m = compute_metrics(eq)
        print(f"{label:<50} {m['return_pct']:>+7.3f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.3f}%")

    # Walk-forward test for best config
    print("\n" + "=" * 70)
    print("Walk-Forward Test")
    print("=" * 70)

    TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 20, 10, 10
    start = 0
    fold = 0
    
    print(f"\n{'Fold':>4} | {'Test Period':<25} | {'Base Sh':>8} | {'Kelly Sh':>8} | {'DD-Adpt Sh':>10} | {'K+DD Sh':>8}")
    print("-" * 95)

    base_shs, kelly_shs, dd_shs, kdd_shs = [], [], [], []

    while start + TRAIN_DAYS + TEST_DAYS <= n_days:
        test_start = dates[start + TRAIN_DAYS]
        test_end = dates[min(start + TRAIN_DAYS + TEST_DAYS - 1, n_days - 1)]
        df_test = df[(df.index.date >= test_start) & (df.index.date <= test_end)]
        if len(df_test) < 50:
            start += STEP_DAYS
            continue
        fold += 1
        orb_f = engine.run(ORBStrategy(), df_test)
        vwap_f = engine.run(VWAPReversionStrategy(), df_test)

        b_eq = combine_results_active_reuse(orb_f, vwap_f, active_weight=0.8)
        k_eq = combine_results_active_reuse_kelly(orb_f, vwap_f, active_weight=0.8)
        dd_eq = combine_dd_adaptive(orb_f, vwap_f, dd_threshold_light=0.0002, dd_threshold_deep=0.0005)
        kdd_eq = combine_dd_adaptive(orb_f, vwap_f, dd_threshold_light=0.0002, dd_threshold_deep=0.0005, kelly_mult=0.5)

        bm, km2, dm, kdm = compute_metrics(b_eq), compute_metrics(k_eq), compute_metrics(dd_eq), compute_metrics(kdd_eq)
        print(f"{fold:>4} | {str(test_start)+' ~ '+str(test_end):<25} | "
              f"{bm['sharpe']:>8.3f} | {km2['sharpe']:>8.3f} | {dm['sharpe']:>10.3f} | {kdm['sharpe']:>8.3f}")
        base_shs.append(bm['sharpe'])
        kelly_shs.append(km2['sharpe'])
        dd_shs.append(dm['sharpe'])
        kdd_shs.append(kdm['sharpe'])
        start += STEP_DAYS

    if base_shs:
        print("-" * 95)
        print(f"{'Avg':>4} | {'Folds: '+str(len(base_shs)):<25} | "
              f"{np.mean(base_shs):>8.3f} | {np.mean(kelly_shs):>8.3f} | "
              f"{np.mean(dd_shs):>10.3f} | {np.mean(kdd_shs):>8.3f}")
        print(f"{'Std':>4} |                           | "
              f"{np.std(base_shs):>8.3f} | {np.std(kelly_shs):>8.3f} | "
              f"{np.std(dd_shs):>10.3f} | {np.std(kdd_shs):>8.3f}")
        for name, shs in [("Baseline", base_shs), ("Kelly", kelly_shs), ("DD-Adpt", dd_shs), ("K+DD", kdd_shs)]:
            pos = sum(1 for s in shs if s > 0) / len(shs) * 100
            print(f"  {name} positive Sharpe: {pos:.0f}%")

    print("\n-- DD-Adaptive test finished --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
