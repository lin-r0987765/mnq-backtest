#!/usr/bin/env python3
"""
Walk-Forward Validation Framework
===================================
取代固定 IS/OOS 切割，使用滾動窗口驗證策略穩健性。

方法：
- 將數據切割成多個 (訓練, 測試) 窗口
- 每個窗口獨立計算績效
- 匯總所有 OOS 窗口的績效，評估策略穩定性
"""
from __future__ import annotations
import argparse
import sys, json
from pathlib import Path
from datetime import timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetcher import fetch_nq_data
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine, BacktestResult
from src.portfolio_overlay import (
    DEFAULT_ACTIVE_REUSE_WEIGHT,
    combine_results_active_reuse,
    combine_results_active_reuse_kelly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Framework")
    parser.add_argument(
        "--include-vwap",
        action="store_true",
        help="Enable the optional VWAP module and combined portfolio walk-forward",
    )
    return parser.parse_args()


def compute_metrics(equity, trades=None):
    """Compute metrics from equity curve."""
    arr = np.array(equity)
    if len(arr) < 2:
        return {"return_pct": 0, "sharpe": 0, "sortino": 0, "max_dd_pct": 0,
                "win_rate": 0, "trades": 0, "profit_factor": 0}

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

    def _trade_pnl(trade):
        if not trade:
            return 0.0
        if "pnl" in trade:
            return float(trade.get("pnl", 0) or 0)
        if "PnL" in trade:
            return float(trade.get("PnL", 0) or 0)
        if "Return" in trade and "Size" in trade and "Avg Entry Price" in trade:
            return float(trade["Return"]) * float(trade["Size"]) * float(trade["Avg Entry Price"])
        return 0.0

    total_trades = len(trades) if trades else 0
    wins = [t for t in (trades or []) if _trade_pnl(t) > 0]
    losses = [t for t in (trades or []) if _trade_pnl(t) <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(_trade_pnl(t) for t in wins) if wins else 0
    gross_loss = abs(sum(_trade_pnl(t) for t in losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    return {
        "return_pct": round(ret_pct, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 4),
        "win_rate": round(win_rate, 1),
        "trades": total_trades,
        "profit_factor": round(pf, 3),
    }


def run_strategy(strategy_cls, df, engine):
    """Run strategy and return BacktestResult."""
    strategy = strategy_cls(params=None)
    return engine.run(strategy, df)


def walk_forward(df, strategy_cls, engine, train_days=20, test_days=10, step_days=10):
    """
    Walk-forward validation.
    
    Parameters:
    - train_days: 訓練窗口大小（交易日）
    - test_days: 測試窗口大小（交易日）
    - step_days: 每次前進步數（交易日）
    
    Returns list of fold results.
    """
    dates = sorted(set(df.index.date))
    n_days = len(dates)
    
    folds = []
    fold_num = 0
    start = 0
    
    while start + train_days + test_days <= n_days:
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end_idx = min(start + train_days + test_days - 1, n_days - 1)
        test_end = dates[test_end_idx]
        
        df_train = df[(df.index.date >= train_start) & (df.index.date <= train_end)]
        df_test = df[(df.index.date >= test_start) & (df.index.date <= test_end)]
        
        if len(df_train) < 100 or len(df_test) < 50:
            start += step_days
            continue
        
        fold_num += 1
        
        # Run on train
        try:
            result_train = run_strategy(strategy_cls, df_train, engine)
            m_train = compute_metrics(result_train.equity_curve, result_train.trades)
        except Exception as e:
            m_train = {"return_pct": 0, "sharpe": 0, "trades": 0, "error": str(e)}
        
        # Run on test
        try:
            result_test = run_strategy(strategy_cls, df_test, engine)
            m_test = compute_metrics(result_test.equity_curve, result_test.trades)
        except Exception as e:
            m_test = {"return_pct": 0, "sharpe": 0, "trades": 0, "error": str(e)}
        
        # Buy & hold
        bh_test = 0
        if len(df_test) > 0:
            bh_test = (df_test["Close"].iloc[-1] / df_test["Close"].iloc[0] - 1) * 100
        
        folds.append({
            "fold": fold_num,
            "train_period": f"{train_start} ~ {train_end}",
            "test_period": f"{test_start} ~ {test_end}",
            "train_bars": len(df_train),
            "test_bars": len(df_test),
            "train": m_train,
            "test": m_test,
            "bh_test_pct": round(bh_test, 4),
        })
        
        start += step_days
    
    return folds


def combine_equity_wf(eq1, eq2, w1=0.5, w2=0.5):
    """Combine two equity curves."""
    arr1, arr2 = np.array(eq1), np.array(eq2)
    n = min(len(arr1), len(arr2))
    if n < 2:
        return arr1[:n] if len(arr1) > 0 else arr2[:n]
    ret1, ret2 = arr1[:n] / arr1[0], arr2[:n] / arr2[0]
    return (w1 * ret1 + w2 * ret2) * 100000


def walk_forward_combined(df, engine, train_days=20, test_days=10, step_days=10):
    """Walk-forward for combined portfolio with active capital reuse."""
    dates = sorted(set(df.index.date))
    n_days = len(dates)
    
    folds = []
    fold_num = 0
    start = 0
    
    while start + train_days + test_days <= n_days:
        train_start = dates[start]
        train_end = dates[start + train_days - 1]
        test_start = dates[start + train_days]
        test_end_idx = min(start + train_days + test_days - 1, n_days - 1)
        test_end = dates[test_end_idx]
        
        df_test = df[(df.index.date >= test_start) & (df.index.date <= test_end)]
        
        if len(df_test) < 50:
            start += step_days
            continue
        
        fold_num += 1
        
        try:
            orb_r = run_strategy(ORBStrategy, df_test, engine)
            vwap_r = run_strategy(VWAPReversionStrategy, df_test, engine)
            
            orb_m = compute_metrics(orb_r.equity_curve, orb_r.trades)
            vwap_m = compute_metrics(vwap_r.equity_curve, vwap_r.trades)
            
            combo_eq = combine_results_active_reuse(orb_r, vwap_r, active_weight=DEFAULT_ACTIVE_REUSE_WEIGHT)
            combo_m = compute_metrics(combo_eq)

            kelly_eq = combine_results_active_reuse_kelly(
                orb_r, vwap_r, active_weight=DEFAULT_ACTIVE_REUSE_WEIGHT,
                kelly_mult=0.5, lookback_trades=10,
            )
            kelly_m = compute_metrics(kelly_eq)
            
            bh = (df_test["Close"].iloc[-1] / df_test["Close"].iloc[0] - 1) * 100
            
            folds.append({
                "fold": fold_num,
                "test_period": f"{test_start} ~ {test_end}",
                "test_bars": len(df_test),
                "orb": orb_m,
                "vwap": vwap_m,
                "combined": combo_m,
                "kelly": kelly_m,
                "bh_pct": round(bh, 4),
            })
        except Exception as e:
            folds.append({"fold": fold_num, "error": str(e)})
        
        start += step_days
    
    return folds


def print_wf_summary(strategy_name, folds):
    """Print walk-forward summary table."""
    print(f"\n{'='*80}")
    print(f"Walk-Forward Validation: {strategy_name}")
    print(f"{'='*80}")
    
    if not folds:
        print("  No valid folds.")
        return {}
    
    print(f"{'Fold':>4} | {'Test Period':<25} | {'Return':>8} | {'Sharpe':>8} | {'WR':>6} | {'Trades':>6} | {'MaxDD':>8} | {'PF':>6} | {'B&H':>8}")
    print("-" * 110)
    
    test_sharpes = []
    test_returns = []
    test_wrs = []
    test_dds = []
    test_trades = []
    alpha_count = 0
    
    for f in folds:
        t = f["test"]
        if "error" in t:
            print(f"{f['fold']:>4} | {f['test_period']:<25} | ERROR: {t['error']}")
            continue
        
        alpha = t["return_pct"] - f["bh_test_pct"]
        alpha_str = "+" if alpha > 0 else "-"
        
        print(f"{f['fold']:>4} | {f['test_period']:<25} | {t['return_pct']:>+7.3f}% | "
              f"{t['sharpe']:>8.3f} | {t['win_rate']:>5.1f}% | {t['trades']:>6} | "
              f"{t['max_dd_pct']:>7.3f}% | {t.get('profit_factor',0):>5.2f} | {f['bh_test_pct']:>+7.3f}% {alpha_str}")
        
        test_sharpes.append(t["sharpe"])
        test_returns.append(t["return_pct"])
        test_wrs.append(t["win_rate"])
        test_dds.append(t["max_dd_pct"])
        test_trades.append(t["trades"])
        if alpha > 0:
            alpha_count += 1
    
    n = len(test_sharpes)
    if n == 0:
        return {}
    
    summary = {
        "folds": n,
        "avg_sharpe": round(np.mean(test_sharpes), 3),
        "std_sharpe": round(np.std(test_sharpes), 3),
        "min_sharpe": round(np.min(test_sharpes), 3),
        "max_sharpe": round(np.max(test_sharpes), 3),
        "avg_return_pct": round(np.mean(test_returns), 4),
        "avg_win_rate": round(np.mean(test_wrs), 1),
        "avg_max_dd": round(np.mean(test_dds), 4),
        "total_trades": int(np.sum(test_trades)),
        "positive_sharpe_pct": round(sum(1 for s in test_sharpes if s > 0) / n * 100, 1),
        "alpha_hit_rate": round(alpha_count / n * 100, 1),
    }
    
    print("-" * 110)
    print(f"{'Summary':>4} | {'Folds: ' + str(n):<25} | {summary['avg_return_pct']:>+7.3f}% | "
          f"{summary['avg_sharpe']:>8.3f} | {summary['avg_win_rate']:>5.1f}% | "
          f"{summary['total_trades']:>6} | {summary['avg_max_dd']:>7.3f}% |       |")
    print(f"       | Sharpe StdDev: {summary['std_sharpe']:.3f}    | "
          f"Positive Sharpe: {summary['positive_sharpe_pct']:.0f}% | "
          f"Alpha Hit Rate: {summary['alpha_hit_rate']:.0f}%")
    
    return summary


def print_combined_wf_summary(folds):
    """Print combined walk-forward summary."""
    print(f"\n{'='*80}")
    print(f"Walk-Forward Validation: Combined Portfolio (Active Reuse 80%)")
    print(f"{'='*80}")
    
    if not folds:
        print("  No valid folds.")
        return {}
    
    print(f"{'Fold':>4} | {'Test Period':<25} | {'ORB Sh':>8} | {'VWAP Sh':>8} | {'Combo Sh':>8} | {'Kelly Sh':>8} | {'Combo Ret':>9} | {'B&H':>8}")
    print("-" * 115)
    
    combo_sharpes = []
    combo_returns = []
    kelly_sharpes = []
    orb_sharpes = []
    vwap_sharpes = []
    
    for f in folds:
        if "error" in f:
            print(f"{f['fold']:>4} | ERROR: {f['error']}")
            continue
        
        kelly_sh = f.get('kelly', {}).get('sharpe', 0)
        print(f"{f['fold']:>4} | {f['test_period']:<25} | "
              f"{f['orb']['sharpe']:>8.3f} | {f['vwap']['sharpe']:>8.3f} | "
              f"{f['combined']['sharpe']:>8.3f} | {kelly_sh:>8.3f} | "
              f"{f['combined']['return_pct']:>+8.3f}% | "
              f"{f['bh_pct']:>+7.3f}%")
        
        combo_sharpes.append(f["combined"]["sharpe"])
        combo_returns.append(f["combined"]["return_pct"])
        kelly_sharpes.append(kelly_sh)
        orb_sharpes.append(f["orb"]["sharpe"])
        vwap_sharpes.append(f["vwap"]["sharpe"])
    
    n = len(combo_sharpes)
    if n == 0:
        return {}
    
    summary = {
        "folds": n,
        "avg_combo_sharpe": round(np.mean(combo_sharpes), 3),
        "std_combo_sharpe": round(np.std(combo_sharpes), 3),
        "avg_combo_return": round(np.mean(combo_returns), 4),
        "avg_orb_sharpe": round(np.mean(orb_sharpes), 3),
        "avg_vwap_sharpe": round(np.mean(vwap_sharpes), 3),
        "avg_kelly_sharpe": round(np.mean(kelly_sharpes), 3),
        "std_kelly_sharpe": round(np.std(kelly_sharpes), 3),
        "positive_combo_sharpe_pct": round(sum(1 for s in combo_sharpes if s > 0) / n * 100, 1),
        "positive_kelly_sharpe_pct": round(sum(1 for s in kelly_sharpes if s > 0) / n * 100, 1),
    }
    
    print("-" * 115)
    print(f"{'Avg':>4} | {'Folds: ' + str(n):<25} | "
          f"{summary['avg_orb_sharpe']:>8.3f} | {summary['avg_vwap_sharpe']:>8.3f} | "
          f"{summary['avg_combo_sharpe']:>8.3f} | {summary['avg_kelly_sharpe']:>8.3f} | "
          f"{summary['avg_combo_return']:>+8.3f}% |")
    print(f"       | Combo Sharpe StdDev: {summary['std_combo_sharpe']:.3f} | "
          f"Positive Sharpe: {summary['positive_combo_sharpe_pct']:.0f}% | "
          f"Kelly Pos: {summary['positive_kelly_sharpe_pct']:.0f}%")
    
    return summary


def main():
    args = parse_args()
    vwap_enabled = args.include_vwap

    print("=" * 80)
    print("Walk-Forward Validation Framework")
    print("=" * 80)

    df = fetch_nq_data()
    if df is None or df.empty:
        print("ERROR: No data loaded.")
        return 1

    dates = sorted(set(df.index.date))
    n_days = len(dates)
    print(f"\n數據: {dates[0]} ~ {dates[-1]} ({n_days} 交易日, {len(df)} 根 K 棒)")

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    TRAIN_DAYS = 20
    TEST_DAYS = 10
    STEP_DAYS = 10

    print(f"\n配置: 訓練窗口={TRAIN_DAYS}天, 測試窗口={TEST_DAYS}天, 步進={STEP_DAYS}天")
    print(f"預計 fold 數量: ~{(n_days - TRAIN_DAYS) // STEP_DAYS}")

    orb_folds = walk_forward(df, ORBStrategy, engine, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    orb_summary = print_wf_summary("ORB v19", orb_folds)

    if vwap_enabled:
        vwap_folds = walk_forward(df, VWAPReversionStrategy, engine, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
        vwap_summary = print_wf_summary("VWAP v23", vwap_folds)
        combo_folds = walk_forward_combined(df, engine, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
        combo_summary = print_combined_wf_summary(combo_folds)
    else:
        vwap_folds = []
        vwap_summary = {}
        combo_folds = []
        combo_summary = {}
        print("\nVWAP optional module disabled in default flow; skipping VWAP and combined walk-forward.")

    print(f"\n{'='*80}")
    print("Walk-Forward 綜合評估")
    print(f"{'='*80}")

    all_results = {
        "mode": "orb_plus_vwap" if vwap_enabled else "orb_only_default",
        "vwap_enabled": vwap_enabled,
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "total_days": n_days,
            "combined_active_reuse_weight": DEFAULT_ACTIVE_REUSE_WEIGHT,
        },
        "orb": {"folds": orb_folds, "summary": orb_summary},
        "vwap": {"folds": vwap_folds, "summary": vwap_summary},
        "combined": {"folds": combo_folds, "summary": combo_summary},
    }

    stability_items = []
    if orb_summary:
        orb_stable = orb_summary.get("positive_sharpe_pct", 0)
        print(
            f"  ORB:  平均 Sharpe={orb_summary['avg_sharpe']:.3f} (±{orb_summary['std_sharpe']:.3f}), "
            f"正 Sharpe 比率={orb_stable:.0f}%, Alpha 命中={orb_summary['alpha_hit_rate']:.0f}%"
        )
        stability_items.append(("ORB", orb_stable, orb_summary["avg_sharpe"]))

    if vwap_summary:
        vwap_stable = vwap_summary.get("positive_sharpe_pct", 0)
        print(
            f"  VWAP: 平均 Sharpe={vwap_summary['avg_sharpe']:.3f} (±{vwap_summary['std_sharpe']:.3f}), "
            f"正 Sharpe 比率={vwap_stable:.0f}%, Alpha 命中={vwap_summary['alpha_hit_rate']:.0f}%"
        )
        stability_items.append(("VWAP", vwap_stable, vwap_summary["avg_sharpe"]))

    if combo_summary:
        combo_stable = combo_summary.get("positive_combo_sharpe_pct", 0)
        print(
            f"  組合: 平均 Sharpe={combo_summary['avg_combo_sharpe']:.3f} (±{combo_summary['std_combo_sharpe']:.3f}), "
            f"正 Sharpe 比率={combo_stable:.0f}%"
        )
        stability_items.append(("Combined", combo_stable, combo_summary["avg_combo_sharpe"]))

    if stability_items:
        avg_positive_rate = np.mean([s[1] for s in stability_items])
        avg_sharpe = np.mean([s[2] for s in stability_items])

        if avg_positive_rate >= 80 and avg_sharpe > 2:
            grade = "A（優秀）— 策略在不同時間段表現穩健"
        elif avg_positive_rate >= 60 and avg_sharpe > 1:
            grade = "B（良好）— 策略整體正向但有波動"
        elif avg_positive_rate >= 40:
            grade = "C（一般）— 策略不夠穩定，需進一步優化"
        else:
            grade = "D（較差）— 策略可能過擬合"

        print(f"\n  穩健性評級: {grade}")
        all_results["grade"] = grade
        all_results["avg_positive_sharpe_rate"] = round(avg_positive_rate, 1)

    with open(PROJECT_ROOT / "walk_forward_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nWalk-forward 結果已保存至 walk_forward_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
