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

ACTIVE_REUSE_WEIGHT = 0.8


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


def _position_mask_from_result(result: BacktestResult) -> np.ndarray:
    """Extract a per-bar active-position mask from a backtest result."""
    if result.raw is not None:
        try:
            mask = result.raw.position_mask()
            if hasattr(mask, "to_numpy"):
                return mask.to_numpy().astype(bool).reshape(-1)
            return np.asarray(mask).astype(bool).reshape(-1)
        except Exception:
            pass

    eq = np.asarray(result.equity_curve, dtype=float)
    if len(eq) < 2:
        return np.zeros(len(eq), dtype=bool)
    return (np.abs(np.diff(eq, prepend=eq[0])) > 1e-12).astype(bool)


def combine_equity_active_reuse(
    eq1,
    eq2,
    mask1,
    mask2,
    active_weight: float = ACTIVE_REUSE_WEIGHT,
    both_weights: tuple[float, float] = (0.5, 0.5),
):
    """Reuse idle allocation when only one strategy is holding a position."""
    arr1 = np.array(eq1)
    arr2 = np.array(eq2)
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)
    n = min(len(arr1), len(arr2), len(m1), len(m2))
    if n < 2:
        return arr1[:n] if len(arr1) > 0 else arr2[:n]

    ret1 = arr1[:n] / arr1[0]
    ret2 = arr2[:n] / arr2[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])
    combined_ret = np.ones(n)

    for i in range(1, n):
        if m1[i] and m2[i]:
            w1, w2 = both_weights
        elif m1[i] and not m2[i]:
            w1, w2 = active_weight, 0.0
        elif m2[i] and not m1[i]:
            w1, w2 = 0.0, active_weight
        else:
            w1, w2 = both_weights
        combined_ret[i] = combined_ret[i - 1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * 100000


def combine_results_active_reuse(
    orb_result: BacktestResult,
    vwap_result: BacktestResult,
    active_weight: float = ACTIVE_REUSE_WEIGHT,
    both_weights: tuple[float, float] = (0.5, 0.5),
):
    return combine_equity_active_reuse(
        orb_result.equity_curve,
        vwap_result.equity_curve,
        _position_mask_from_result(orb_result),
        _position_mask_from_result(vwap_result),
        active_weight=active_weight,
        both_weights=both_weights,
    )


def _load_daily_atr(atr_period=14):
    """從 qqq_1d.csv 載入日級別 ATR%，避免 5m 重取樣造成閾值失效。"""
    daily_csv = PROJECT_ROOT / "qqq_1d.csv"
    if not daily_csv.exists():
        return None
    try:
        dfd = pd.read_csv(daily_csv, header=[0, 1], index_col=0)
        if isinstance(dfd.columns, pd.MultiIndex):
            dfd.columns = dfd.columns.get_level_values(0)
    except Exception:
        dfd = pd.read_csv(daily_csv, index_col=0)
    dfd.index = pd.to_datetime(dfd.index, utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in dfd.columns:
            dfd[col] = pd.to_numeric(dfd[col], errors="coerce")
    dfd = dfd.dropna(subset=["Open", "High", "Low", "Close"])
    dfd = dfd[dfd.index.notna()]
    close = dfd["Close"]; high = dfd["High"]; low = dfd["Low"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    atr_pct = atr / close
    # index → date → atr_pct mapping
    return {d.date() if hasattr(d, 'date') else d: v for d, v in atr_pct.dropna().items()}


def combine_equity_adaptive(eq1, eq2, df, atr_period=14, atr_pct_threshold=0.015, mode="absolute"):
    """
    ATR 自適應策略切換（v14: 使用日級別 ATR）：
    - 高 ATR（≥ threshold）→ ORB 權重 70%, VWAP 30%
    - 低 ATR（< threshold）→ ORB 30%, VWAP 70%

    mode:
    - "absolute": 使用固定閾值（適用於固定波動期）
    - "rolling": 使用滾動 20 日中位數作為動態閾值（適應不同波動環境）
    """
    arr1 = np.array(eq1)  # ORB
    arr2 = np.array(eq2)  # VWAP
    n = min(len(arr1), len(arr2))

    # v14: 使用日線 CSV 計算真實日級別 ATR
    daily_atr_map = _load_daily_atr(atr_period)
    if daily_atr_map is None:
        close = df["Close"]; high = df["High"]; low = df["Low"]
        tr = pd.concat([
            high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period * 78, min_periods=78).mean()
        atr_pct = atr / close
        daily_atr_map = dict(atr_pct.groupby(atr_pct.index.date).mean())

    # Build rolling median if mode is "rolling"
    rolling_median = {}
    if mode == "rolling":
        all_dates = sorted(daily_atr_map.keys())
        atr_series = pd.Series({d: daily_atr_map[d] for d in all_dates})
        rolling_med = atr_series.rolling(20, min_periods=5).median()
        rolling_median = dict(rolling_med.dropna())

    ret1 = arr1[:n] / arr1[0]
    ret2 = arr2[:n] / arr2[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])

    idx = df.index[:n]
    combined_ret = np.ones(n)

    for i in range(1, n):
        date = idx[i].date()
        atr_val = daily_atr_map.get(date, atr_pct_threshold)
        if pd.isna(atr_val):
            atr_val = atr_pct_threshold

        if mode == "rolling":
            thresh = rolling_median.get(date, atr_pct_threshold)
        else:
            thresh = atr_pct_threshold

        if atr_val >= thresh:
            w1, w2 = 0.7, 0.3  # 高波動 → ORB 主力
        else:
            w1, w2 = 0.3, 0.7  # 低波動 → VWAP 主力

        combined_ret[i] = combined_ret[i-1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * 100000


def main():
    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

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
    allocation_metrics: dict[str, dict] = {}

    for w_orb, w_vwap in allocations:
        combined_eq = combine_equity(orb_result.equity_curve, vwap_result.equity_curve, w_orb, w_vwap)
        cm = compute_metrics(combined_eq)
        allocation_metrics[f"static_{int(w_orb * 100)}_{int(w_vwap * 100)}"] = cm
        print(f"ORB {int(w_orb*100)}% / VWAP {int(w_vwap*100)}%: "
              f"Return={cm['return_pct']:+.4f}%, Sharpe={cm['sharpe']:.3f}, "
              f"Sortino={cm['sortino']:.3f}, MaxDD={cm['max_dd_pct']:.3f}%")

    active_reuse_eq = combine_results_active_reuse(orb_result, vwap_result)
    active_reuse_m = compute_metrics(active_reuse_eq)
    print(f"Active Reuse 80%: Return={active_reuse_m['return_pct']:+.4f}%, "
          f"Sharpe={active_reuse_m['sharpe']:.3f}, Sortino={active_reuse_m['sortino']:.3f}, "
          f"MaxDD={active_reuse_m['max_dd_pct']:.3f}%")

    # ATR Adaptive allocation (absolute thresholds)
    print("\n[ATR Adaptive - Absolute] 高 ATR→ORB 70%/VWAP 30%, 低 ATR→ORB 30%/VWAP 70%:")
    best_adaptive = {"sharpe": -999}
    for threshold in [0.013, 0.015, 0.017]:
        adaptive_eq = combine_equity_adaptive(
            orb_result.equity_curve, vwap_result.equity_curve, df,
            atr_period=14, atr_pct_threshold=threshold, mode="absolute"
        )
        am = compute_metrics(adaptive_eq)
        print(f"  Threshold={threshold:.3f}: Return={am['return_pct']:+.4f}%, "
              f"Sharpe={am['sharpe']:.3f}, Sortino={am['sortino']:.3f}, MaxDD={am['max_dd_pct']:.3f}%")
        if am["sharpe"] > best_adaptive["sharpe"]:
            best_adaptive = {**am, "threshold": threshold, "mode": "absolute"}

    # ATR Adaptive allocation (rolling median)
    print("\n[ATR Adaptive - Rolling 20d Median]:")
    adaptive_eq_roll = combine_equity_adaptive(
        orb_result.equity_curve, vwap_result.equity_curve, df,
        atr_period=14, mode="rolling"
    )
    am_roll = compute_metrics(adaptive_eq_roll)
    print(f"  Rolling: Return={am_roll['return_pct']:+.4f}%, "
          f"Sharpe={am_roll['sharpe']:.3f}, Sortino={am_roll['sortino']:.3f}, MaxDD={am_roll['max_dd_pct']:.3f}%")
    if am_roll["sharpe"] > best_adaptive["sharpe"]:
        best_adaptive = {**am_roll, "mode": "rolling"}

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
        combined_oos_50 = combine_equity(orb_oos.equity_curve, vwap_oos.equity_curve, 0.5, 0.5)
        cm_oos_50 = compute_metrics(combined_oos_50)
        combined_oos_60 = combine_equity(orb_oos.equity_curve, vwap_oos.equity_curve, 0.6, 0.4)
        cm_oos_60 = compute_metrics(combined_oos_60)
        combined_oos_active = combine_results_active_reuse(orb_oos, vwap_oos)
        cm_oos_active = compute_metrics(combined_oos_active)
        print(f"\n{'Combined 50/50 OOS':<20} {cm_oos_50['return_pct']:>+9.4f}% {cm_oos_50['sharpe']:>10.3f}")
        print(f"{'Combined 60/40 OOS':<20} {cm_oos_60['return_pct']:>+9.4f}% {cm_oos_60['sharpe']:>10.3f}")
        print(f"{'Active Reuse OOS':<20} {cm_oos_active['return_pct']:>+9.4f}% {cm_oos_active['sharpe']:>10.3f}")

        # ATR Adaptive OOS (absolute)
        print("\n[ATR Adaptive OOS - Absolute]:")
        adaptive_oos_results: dict[str, dict] = {}
        for threshold in [0.013, 0.015, 0.017]:
            adaptive_oos = combine_equity_adaptive(
                orb_oos.equity_curve, vwap_oos.equity_curve, df_oos,
                atr_period=14, atr_pct_threshold=threshold, mode="absolute"
            )
            am_oos = compute_metrics(adaptive_oos)
            adaptive_oos_results[f"absolute_{threshold:.3f}"] = am_oos
            print(f"  Threshold={threshold:.3f}: Return={am_oos['return_pct']:+.4f}%, "
                  f"Sharpe={am_oos['sharpe']:.3f}, MaxDD={am_oos['max_dd_pct']:.3f}%")

        # ATR Adaptive OOS (rolling)
        print("\n[ATR Adaptive OOS - Rolling 20d Median]:")
        adaptive_oos_roll = combine_equity_adaptive(
            orb_oos.equity_curve, vwap_oos.equity_curve, df_oos,
            atr_period=14, mode="rolling"
        )
        am_oos_roll = compute_metrics(adaptive_oos_roll)
        print(f"  Rolling: Return={am_oos_roll['return_pct']:+.4f}%, "
              f"Sharpe={am_oos_roll['sharpe']:.3f}, MaxDD={am_oos_roll['max_dd_pct']:.3f}%")
    else:
        cm_oos_50 = {}
        cm_oos_60 = {}
        cm_oos_active = {}
        adaptive_oos_results = {}
        am_oos_roll = {}

    # Save results
    results = {
        "full": {
            "orb": orb_result.metrics,
            "vwap": vwap_result.metrics,
            "combined": {
                **allocation_metrics,
                "active_reuse_80": active_reuse_m,
                "atr_adaptive_best": best_adaptive,
                "atr_adaptive_rolling": am_roll,
            },
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
            "combined_50_50": cm_oos_50,
            "combined_60_40": cm_oos_60,
            "combined_active_reuse_80": cm_oos_active,
            "adaptive_absolute": adaptive_oos_results,
            "adaptive_rolling": am_oos_roll,
        },
        "split_date": str(split_date),
        "is_days": split_idx,
        "oos_days": n_days - split_idx,
    }

    with open(PROJECT_ROOT / "combined_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to combined_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
