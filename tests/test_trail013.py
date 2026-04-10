#!/usr/bin/env python3
"""Quick OOS + WF test for ORB trailing_pct=0.013."""
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
        return {"return_pct": 0, "sharpe": 0, "sortino": 0, "max_dd_pct": 0}
    ret_pct = (arr[-1] / arr[0] - 1) * 100
    returns = pd.Series(arr).pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * (252 * 78) ** 0.5) if returns.std() > 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = float(returns.mean() / downside * (252 * 78) ** 0.5) if downside > 0 else 0.0
    peak = pd.Series(arr).cummax()
    dd = (pd.Series(arr) - peak) / peak
    max_dd = float(dd.min() * 100)
    return {"return_pct": round(ret_pct, 4), "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3), "max_dd_pct": round(max_dd, 4)}


def main():
    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    dates = sorted(set(df.index.date))
    n_days = len(dates)

    # Full dataset comparison
    print("=== Full Dataset ===")
    for label, tp_val in [("Baseline trail=0.015", 0.015), ("New trail=0.013", 0.013)]:
        orb_r = engine.run(ORBStrategy(params={"trailing_pct": tp_val}), df)
        vwap_r = engine.run(VWAPReversionStrategy(), df)
        om = orb_r.metrics
        combo_k = combine_results_active_reuse_kelly(orb_r, vwap_r, active_weight=0.8)
        cm = compute_metrics(combo_k)
        print(f"{label}: ORB Sh={om['sharpe_ratio']:.3f} WR={om['win_rate_pct']:.1f}% Trades={om['total_trades']} | "
              f"Combo: Sh={cm['sharpe']:.3f} Sort={cm['sortino']:.3f} DD={cm['max_dd_pct']:.3f}%")

    # OOS
    print("\n=== OOS ===")
    split_idx = int(n_days * 2 / 3)
    split_date = dates[split_idx]
    df_oos = df[df.index.date >= split_date]
    
    for label, tp_val in [("Baseline trail=0.015", 0.015), ("New trail=0.013", 0.013)]:
        orb_oos = engine.run(ORBStrategy(params={"trailing_pct": tp_val}), df_oos)
        vwap_oos = engine.run(VWAPReversionStrategy(), df_oos)
        om = orb_oos.metrics
        combo_k = combine_results_active_reuse_kelly(orb_oos, vwap_oos, active_weight=0.8)
        cm = compute_metrics(combo_k)
        print(f"{label}: ORB Sh={om['sharpe_ratio']:.3f} WR={om['win_rate_pct']:.1f}% Trades={om['total_trades']} | "
              f"Combo: Sh={cm['sharpe']:.3f} Sort={cm['sortino']:.3f}")

    # Walk-forward
    print("\n=== Walk-Forward ===")
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS = 20, 10, 10
    
    for label, tp_val in [("Baseline trail=0.015", 0.015), ("New trail=0.013", 0.013)]:
        start = 0
        fold = 0
        sharpes = []
        kelly_sharpes = []
        while start + TRAIN_DAYS + TEST_DAYS <= n_days:
            test_start = dates[start + TRAIN_DAYS]
            test_end = dates[min(start + TRAIN_DAYS + TEST_DAYS - 1, n_days - 1)]
            df_test = df[(df.index.date >= test_start) & (df.index.date <= test_end)]
            if len(df_test) < 50:
                start += STEP_DAYS
                continue
            fold += 1
            orb_f = engine.run(ORBStrategy(params={"trailing_pct": tp_val}), df_test)
            vwap_f = engine.run(VWAPReversionStrategy(), df_test)
            combo_base = combine_results_active_reuse(orb_f, vwap_f, active_weight=0.8)
            combo_kelly = combine_results_active_reuse_kelly(orb_f, vwap_f, active_weight=0.8)
            bm = compute_metrics(combo_base)
            km = compute_metrics(combo_kelly)
            sharpes.append(bm['sharpe'])
            kelly_sharpes.append(km['sharpe'])
            start += STEP_DAYS

        avg_sh = np.mean(sharpes)
        avg_ksh = np.mean(kelly_sharpes)
        pos_pct = sum(1 for s in kelly_sharpes if s > 0) / len(kelly_sharpes) * 100
        print(f"{label}: WF avg Combo Sh={avg_sh:.3f}, Kelly Sh={avg_ksh:.3f}, "
              f"Kelly pos={pos_pct:.0f}%, folds={len(sharpes)}")

    print("\n-- Done --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
