#!/usr/bin/env python3
"""Quick walk-forward test for Kelly sizing."""
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
from src.portfolio_overlay import combine_results_active_reuse
from test_kelly_sizing import combine_with_kelly_reuse, compute_metrics


def main():
    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    
    dates = sorted(set(df.index.date))
    n_days = len(dates)
    
    TRAIN_DAYS = 20
    TEST_DAYS = 10
    STEP_DAYS = 10
    
    print(f"Data: {dates[0]}~{dates[-1]} ({n_days}d)")
    print(f"Config: train={TRAIN_DAYS}d, test={TEST_DAYS}d, step={STEP_DAYS}d")
    print()
    print(f"{'Fold':>4} | {'Test Period':<25} | {'Base Sh':>8} | {'Base Ret':>8} | {'Kelly Sh':>8} | {'Kelly Ret':>8} | {'Base DD':>8} | {'Kelly DD':>8}")
    print("-" * 110)
    
    base_sharpes = []
    kelly_sharpes = []
    base_dds = []
    kelly_dds = []
    
    start = 0
    fold = 0
    
    while start + TRAIN_DAYS + TEST_DAYS <= n_days:
        test_start = dates[start + TRAIN_DAYS]
        test_end_idx = min(start + TRAIN_DAYS + TEST_DAYS - 1, n_days - 1)
        test_end = dates[test_end_idx]
        
        df_test = df[(df.index.date >= test_start) & (df.index.date <= test_end)]
        
        if len(df_test) < 50:
            start += STEP_DAYS
            continue
        
        fold += 1
        
        orb_r = engine.run(ORBStrategy(), df_test)
        vwap_r = engine.run(VWAPReversionStrategy(), df_test)
        
        # Baseline
        base_eq = combine_results_active_reuse(orb_r, vwap_r, active_weight=0.8)
        base_m = compute_metrics(base_eq)
        
        # Kelly (Half-Kelly lb=10)
        kelly_eq = combine_with_kelly_reuse(
            orb_r, vwap_r,
            active_weight=0.8,
            kelly_mult=0.5,
            lookback_trades=10,
            min_size_mult=0.3,
            max_size_mult=1.5,
        )
        kelly_m = compute_metrics(kelly_eq)
        
        print(f"{fold:>4} | {str(test_start)+' ~ '+str(test_end):<25} | "
              f"{base_m['sharpe']:>8.3f} | {base_m['return_pct']:>+7.3f}% | "
              f"{kelly_m['sharpe']:>8.3f} | {kelly_m['return_pct']:>+7.3f}% | "
              f"{base_m['max_dd_pct']:>7.3f}% | {kelly_m['max_dd_pct']:>7.3f}%")
        
        base_sharpes.append(base_m['sharpe'])
        kelly_sharpes.append(kelly_m['sharpe'])
        base_dds.append(base_m['max_dd_pct'])
        kelly_dds.append(kelly_m['max_dd_pct'])
        
        start += STEP_DAYS
    
    n = len(base_sharpes)
    if n > 0:
        print("-" * 110)
        print(f"{'Avg':>4} | {'Folds: '+str(n):<25} | "
              f"{np.mean(base_sharpes):>8.3f} |          | "
              f"{np.mean(kelly_sharpes):>8.3f} |          | "
              f"{np.mean(base_dds):>7.3f}% | {np.mean(kelly_dds):>7.3f}%")
        print(f"{'Std':>4} |                           | "
              f"{np.std(base_sharpes):>8.3f} |          | "
              f"{np.std(kelly_sharpes):>8.3f} |")
        
        # Positive Sharpe rates
        base_pos = sum(1 for s in base_sharpes if s > 0) / n * 100
        kelly_pos = sum(1 for s in kelly_sharpes if s > 0) / n * 100
        print(f"\nPositive Sharpe rate: Baseline={base_pos:.0f}%, Kelly={kelly_pos:.0f}%")
        
        # Check grade
        avg_pos = kelly_pos
        avg_sh = np.mean(kelly_sharpes)
        if avg_pos >= 80 and avg_sh > 2:
            grade = "A"
        elif avg_pos >= 60 and avg_sh > 1:
            grade = "B"
        elif avg_pos >= 40:
            grade = "C"
        else:
            grade = "D"
        print(f"Kelly WF grade: {grade}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
