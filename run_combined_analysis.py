#!/usr/bin/env python3
"""
Combined Portfolio Analysis + Out-of-Sample Validation
======================================================
1. Run ORB on the full dataset
2. Optionally run VWAP and combined overlays with --include-vwap
3. Split into in-sample / out-of-sample windows
4. Save metrics to combined_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_nq_data
from src.portfolio_overlay import (
    DEFAULT_ACTIVE_REUSE_WEIGHT,
    combine_equity_static,
    combine_results_active_reuse,
    combine_results_active_reuse_kelly,
)
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined Portfolio Analysis + Out-of-Sample Validation"
    )
    parser.add_argument(
        "--include-vwap",
        action="store_true",
        help="Enable the optional VWAP module and combined overlays",
    )
    return parser.parse_args()


def _trade_pnl(trade: dict) -> float:
    if not trade:
        return 0.0
    if "pnl" in trade:
        return float(trade.get("pnl", 0) or 0)
    if "PnL" in trade:
        return float(trade.get("PnL", 0) or 0)
    if "Return" in trade and "Size" in trade and "Avg Entry Price" in trade:
        return float(trade["Return"]) * float(trade["Size"]) * float(trade["Avg Entry Price"])
    return 0.0


def compute_metrics(equity: np.ndarray, trades: list | None = None) -> dict:
    """Compute standard metrics from an equity curve."""
    arr = np.array(equity, dtype=float)
    if len(arr) < 2:
        return {
            "return_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "profit_factor": 0.0,
        }

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

    total_trades = len(trades) if trades else 0
    wins = [t for t in (trades or []) if _trade_pnl(t) > 0]
    losses = [t for t in (trades or []) if _trade_pnl(t) <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = sum(_trade_pnl(t) for t in wins) if wins else 0
    gross_loss = abs(sum(_trade_pnl(t) for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    return {
        "return_pct": round(ret_pct, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd, 4),
        "win_rate": round(win_rate, 1),
        "trades": total_trades,
        "profit_factor": round(profit_factor, 3),
    }


def run_strategy_on_data(strategy_cls, params, df, engine):
    """Run a strategy and return BacktestResult."""
    strategy = strategy_cls(params=params)
    return engine.run(strategy, df)


def _load_daily_atr(atr_period: int = 14):
    """Load daily ATR% from qqq_1d.csv."""
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

    close = dfd["Close"]
    high = dfd["High"]
    low = dfd["Low"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    atr_pct = atr / close
    return {d.date() if hasattr(d, "date") else d: v for d, v in atr_pct.dropna().items()}


def combine_equity_adaptive(
    eq1,
    eq2,
    df,
    atr_period: int = 14,
    atr_pct_threshold: float = 0.015,
    mode: str = "absolute",
):
    """
    Adapt allocation by ATR regime.
    High ATR: ORB 70% / VWAP 30%
    Low ATR: ORB 30% / VWAP 70%
    """
    arr1 = np.array(eq1)
    arr2 = np.array(eq2)
    n = min(len(arr1), len(arr2))
    if n < 2:
        return arr1[:n] if len(arr1) > 0 else arr2[:n]

    daily_atr_map = _load_daily_atr(atr_period)
    if daily_atr_map is None:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_period * 78, min_periods=78).mean()
        atr_pct = atr / close
        daily_atr_map = dict(atr_pct.groupby(atr_pct.index.date).mean())

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
            w1, w2 = 0.7, 0.3
        else:
            w1, w2 = 0.3, 0.7

        combined_ret[i] = combined_ret[i - 1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * 100000


def main():
    args = parse_args()
    vwap_enabled = args.include_vwap

    df = fetch_nq_data()
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    orb_result = run_strategy_on_data(ORBStrategy, None, df, engine)
    vwap_result = run_strategy_on_data(VWAPReversionStrategy, None, df, engine) if vwap_enabled else None

    print("\n" + "=" * 70)
    print("FULL DATASET RESULTS")
    print("=" * 70)

    orb_m = orb_result.metrics
    print(
        f"ORB:  Return={orb_m['total_return_pct']:+.4f}%, Sharpe={orb_m['sharpe_ratio']:.3f}, "
        f"WR={orb_m['win_rate_pct']:.1f}%, Trades={orb_m['total_trades']}, DD={orb_m['max_drawdown_pct']:.3f}%"
    )
    if vwap_result is not None:
        vwap_m = vwap_result.metrics
        print(
            f"VWAP: Return={vwap_m['total_return_pct']:+.4f}%, Sharpe={vwap_m['sharpe_ratio']:.3f}, "
            f"WR={vwap_m['win_rate_pct']:.1f}%, Trades={vwap_m['total_trades']}, DD={vwap_m['max_drawdown_pct']:.3f}%"
        )
    else:
        print("VWAP: disabled in default flow. Use --include-vwap to enable optional module.")

    allocation_metrics: dict[str, dict] = {}
    active_reuse_m: dict = {}
    kelly_reuse_m: dict = {}
    best_adaptive: dict = {}
    am_roll: dict = {}
    kelly_reuse_eq = None

    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO ANALYSIS")
    print("=" * 70)

    if vwap_result is not None:
        allocations = [(0.6, 0.4), (0.5, 0.5), (0.7, 0.3)]
        for w_orb, w_vwap in allocations:
            combined_eq = combine_equity_static(
                orb_result.equity_curve, vwap_result.equity_curve, w_orb, w_vwap
            )
            cm = compute_metrics(combined_eq)
            allocation_metrics[f"static_{int(w_orb * 100)}_{int(w_vwap * 100)}"] = cm
            print(
                f"ORB {int(w_orb * 100)}% / VWAP {int(w_vwap * 100)}%: "
                f"Return={cm['return_pct']:+.4f}%, Sharpe={cm['sharpe']:.3f}, "
                f"Sortino={cm['sortino']:.3f}, MaxDD={cm['max_dd_pct']:.3f}%"
            )

        active_reuse_eq = combine_results_active_reuse(
            orb_result, vwap_result, active_weight=DEFAULT_ACTIVE_REUSE_WEIGHT
        )
        active_reuse_m = compute_metrics(active_reuse_eq)
        print(
            f"Active Reuse 80%: Return={active_reuse_m['return_pct']:+.4f}%, "
            f"Sharpe={active_reuse_m['sharpe']:.3f}, Sortino={active_reuse_m['sortino']:.3f}, "
            f"MaxDD={active_reuse_m['max_dd_pct']:.3f}%"
        )

        kelly_reuse_eq = combine_results_active_reuse_kelly(
            orb_result,
            vwap_result,
            active_weight=DEFAULT_ACTIVE_REUSE_WEIGHT,
            kelly_mult=0.5,
            lookback_trades=10,
        )
        kelly_reuse_m = compute_metrics(kelly_reuse_eq)
        print(
            f"Active Reuse 80% + Kelly: Return={kelly_reuse_m['return_pct']:+.4f}%, "
            f"Sharpe={kelly_reuse_m['sharpe']:.3f}, Sortino={kelly_reuse_m['sortino']:.3f}, "
            f"MaxDD={kelly_reuse_m['max_dd_pct']:.3f}%"
        )

        print("\n[ATR Adaptive - Absolute] 高 ATR→ORB 70%/VWAP 30%, 低 ATR→ORB 30%/VWAP 70%:")
        best_adaptive = {"sharpe": -999}
        for threshold in [0.013, 0.015, 0.017]:
            adaptive_eq = combine_equity_adaptive(
                orb_result.equity_curve,
                vwap_result.equity_curve,
                df,
                atr_period=14,
                atr_pct_threshold=threshold,
                mode="absolute",
            )
            am = compute_metrics(adaptive_eq)
            print(
                f"  Threshold={threshold:.3f}: Return={am['return_pct']:+.4f}%, "
                f"Sharpe={am['sharpe']:.3f}, Sortino={am['sortino']:.3f}, MaxDD={am['max_dd_pct']:.3f}%"
            )
            if am["sharpe"] > best_adaptive["sharpe"]:
                best_adaptive = {**am, "threshold": threshold, "mode": "absolute"}

        print("\n[ATR Adaptive - Rolling 20d Median]:")
        adaptive_eq_roll = combine_equity_adaptive(
            orb_result.equity_curve,
            vwap_result.equity_curve,
            df,
            atr_period=14,
            mode="rolling",
        )
        am_roll = compute_metrics(adaptive_eq_roll)
        print(
            f"  Rolling: Return={am_roll['return_pct']:+.4f}%, "
            f"Sharpe={am_roll['sharpe']:.3f}, Sortino={am_roll['sortino']:.3f}, MaxDD={am_roll['max_dd_pct']:.3f}%"
        )
        if am_roll["sharpe"] > best_adaptive["sharpe"]:
            best_adaptive = {**am_roll, "mode": "rolling"}
    else:
        print("VWAP optional module is disabled in default flow, so combined overlays are skipped.")

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    dates = sorted(set(df.index.date))
    n_days = len(dates)
    split_idx = int(n_days * 2 / 3)
    split_date = dates[split_idx]

    df_is = df[df.index.date < split_date]
    df_oos = df[df.index.date >= split_date]

    print(f"\nIn-Sample:  {dates[0]} ~ {dates[split_idx - 1]} ({split_idx} days, {len(df_is)} bars)")
    print(f"Out-of-Sample: {split_date} ~ {dates[-1]} ({n_days - split_idx} days, {len(df_oos)} bars)")

    orb_is = run_strategy_on_data(ORBStrategy, None, df_is, engine)
    orb_oos = run_strategy_on_data(ORBStrategy, None, df_oos, engine)
    vwap_is = run_strategy_on_data(VWAPReversionStrategy, None, df_is, engine) if vwap_enabled else None
    vwap_oos = run_strategy_on_data(VWAPReversionStrategy, None, df_oos, engine) if vwap_enabled else None

    bh_is = (df_is["Close"].iloc[-1] / df_is["Close"].iloc[0] - 1) * 100
    bh_oos = (df_oos["Close"].iloc[-1] / df_oos["Close"].iloc[0] - 1) * 100

    def print_split_row(label: str, metrics: dict) -> None:
        print(
            f"{label:<20} {metrics['total_return_pct']:>+9.4f}% "
            f"{metrics['sharpe_ratio']:>10.3f} {metrics['win_rate_pct']:>7.1f}% {metrics['total_trades']:>10}"
        )

    print(f"\n{'Strategy':<20} {'IS Return':>10} {'IS Sharpe':>10} {'IS WR':>8} {'IS Trades':>10}")
    print("-" * 60)
    print_split_row("ORB", orb_is.metrics)
    if vwap_is is not None:
        print_split_row("VWAP", vwap_is.metrics)
    print(f"{'Buy & Hold':<20} {bh_is:>+9.4f}%")

    print(f"\n{'Strategy':<20} {'OOS Return':>10} {'OOS Sharpe':>10} {'OOS WR':>8} {'OOS Trades':>10}")
    print("-" * 60)
    print_split_row("ORB", orb_oos.metrics)
    if vwap_oos is not None:
        print_split_row("VWAP", vwap_oos.metrics)
    print(f"{'Buy & Hold':<20} {bh_oos:>+9.4f}%")

    cm_oos_50: dict = {}
    cm_oos_60: dict = {}
    cm_oos_active: dict = {}
    cm_oos_kelly: dict = {}
    adaptive_oos_results: dict[str, dict] = {}
    am_oos_roll: dict = {}

    if vwap_oos is not None and len(orb_oos.equity_curve) > 0 and len(vwap_oos.equity_curve) > 0:
        combined_oos_50 = combine_equity_static(orb_oos.equity_curve, vwap_oos.equity_curve, 0.5, 0.5)
        cm_oos_50 = compute_metrics(combined_oos_50)
        combined_oos_60 = combine_equity_static(orb_oos.equity_curve, vwap_oos.equity_curve, 0.6, 0.4)
        cm_oos_60 = compute_metrics(combined_oos_60)
        combined_oos_active = combine_results_active_reuse(
            orb_oos, vwap_oos, active_weight=DEFAULT_ACTIVE_REUSE_WEIGHT
        )
        cm_oos_active = compute_metrics(combined_oos_active)
        combined_oos_kelly = combine_results_active_reuse_kelly(
            orb_oos,
            vwap_oos,
            active_weight=DEFAULT_ACTIVE_REUSE_WEIGHT,
            kelly_mult=0.5,
            lookback_trades=10,
        )
        cm_oos_kelly = compute_metrics(combined_oos_kelly)
        print(f"\n{'Combined 50/50 OOS':<20} {cm_oos_50['return_pct']:>+9.4f}% {cm_oos_50['sharpe']:>10.3f}")
        print(f"{'Combined 60/40 OOS':<20} {cm_oos_60['return_pct']:>+9.4f}% {cm_oos_60['sharpe']:>10.3f}")
        print(f"{'Active Reuse OOS':<20} {cm_oos_active['return_pct']:>+9.4f}% {cm_oos_active['sharpe']:>10.3f}")
        print(f"{'Kelly Reuse OOS':<20} {cm_oos_kelly['return_pct']:>+9.4f}% {cm_oos_kelly['sharpe']:>10.3f}")

        print("\n[ATR Adaptive OOS - Absolute]:")
        for threshold in [0.013, 0.015, 0.017]:
            adaptive_oos = combine_equity_adaptive(
                orb_oos.equity_curve,
                vwap_oos.equity_curve,
                df_oos,
                atr_period=14,
                atr_pct_threshold=threshold,
                mode="absolute",
            )
            am_oos = compute_metrics(adaptive_oos)
            adaptive_oos_results[f"absolute_{threshold:.3f}"] = am_oos
            print(
                f"  Threshold={threshold:.3f}: Return={am_oos['return_pct']:+.4f}%, "
                f"Sharpe={am_oos['sharpe']:.3f}, MaxDD={am_oos['max_dd_pct']:.3f}%"
            )

        print("\n[ATR Adaptive OOS - Rolling 20d Median]:")
        adaptive_oos_roll = combine_equity_adaptive(
            orb_oos.equity_curve,
            vwap_oos.equity_curve,
            df_oos,
            atr_period=14,
            mode="rolling",
        )
        am_oos_roll = compute_metrics(adaptive_oos_roll)
        print(
            f"  Rolling: Return={am_oos_roll['return_pct']:+.4f}%, "
            f"Sharpe={am_oos_roll['sharpe']:.3f}, MaxDD={am_oos_roll['max_dd_pct']:.3f}%"
        )
    elif not vwap_enabled:
        print("\nCombined OOS overlays skipped in ORB-only default flow.")

    print("\n" + "=" * 70)
    print("MONTE CARLO BOOTSTRAP CONFIDENCE INTERVALS (10,000 iterations)")
    print("=" * 70)

    def bootstrap_sharpe(equity: np.ndarray, n_iter: int = 10000, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return {"mean": 0, "std": 0, "ci_5": 0, "ci_95": 0, "p_positive": 0}
        ann = (252 * 78) ** 0.5
        boot_sharpes = np.empty(n_iter)
        n = len(returns)
        for i in range(n_iter):
            sample = rng.choice(returns, size=n, replace=True)
            s_std = sample.std()
            boot_sharpes[i] = (sample.mean() / s_std * ann) if s_std > 0 else 0
        return {
            "mean": round(float(np.mean(boot_sharpes)), 3),
            "std": round(float(np.std(boot_sharpes)), 3),
            "ci_5": round(float(np.percentile(boot_sharpes, 5)), 3),
            "ci_95": round(float(np.percentile(boot_sharpes, 95)), 3),
            "p_positive": round(float(np.mean(boot_sharpes > 0) * 100), 1),
        }

    def bootstrap_trades(trades: list, n_iter: int = 10000, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        pnls = [_trade_pnl(t) for t in trades]
        if len(pnls) < 3:
            return {"wr_mean": 0, "wr_ci_5": 0, "wr_ci_95": 0, "pf_mean": 0, "n_trades": len(pnls)}
        pnls = np.array(pnls)
        n = len(pnls)
        wr_boots = np.empty(n_iter)
        pf_boots = np.empty(n_iter)
        for i in range(n_iter):
            sample = rng.choice(pnls, size=n, replace=True)
            wr_boots[i] = np.mean(sample > 0) * 100
            wins = sample[sample > 0].sum()
            losses = abs(sample[sample <= 0].sum())
            pf_boots[i] = wins / losses if losses > 0 else 10.0
        return {
            "wr_mean": round(float(np.mean(wr_boots)), 1),
            "wr_ci_5": round(float(np.percentile(wr_boots, 5)), 1),
            "wr_ci_95": round(float(np.percentile(wr_boots, 95)), 1),
            "pf_mean": round(float(np.mean(pf_boots)), 3),
            "pf_ci_5": round(float(np.percentile(pf_boots, 5)), 3),
            "pf_ci_95": round(float(np.percentile(pf_boots, 95)), 3),
            "n_trades": n,
        }

    mc_results = {}
    strategy_results = [("ORB", orb_result)]
    if vwap_result is not None:
        strategy_results.append(("VWAP", vwap_result))

    for name, result in strategy_results:
        eq = np.array(result.equity_curve)
        bs = bootstrap_sharpe(eq)
        bt = bootstrap_trades(result.trades)
        mc_results[name] = {"sharpe_bootstrap": bs, "trade_bootstrap": bt}
        print(f"\n{name}:")
        print(
            f"  Sharpe: {bs['mean']:.3f} (95% CI: [{bs['ci_5']:.3f}, {bs['ci_95']:.3f}]), "
            f"P(Sharpe>0) = {bs['p_positive']:.1f}%"
        )
        print(f"  Win Rate: {bt['wr_mean']:.1f}% (95% CI: [{bt['wr_ci_5']:.1f}%, {bt['wr_ci_95']:.1f}%])")
        print(
            f"  Profit Factor: {bt['pf_mean']:.3f} "
            f"(95% CI: [{bt.get('pf_ci_5', 0):.3f}, {bt.get('pf_ci_95', 0):.3f}])"
        )

    if vwap_result is not None and kelly_reuse_eq is not None:
        bs_kelly = bootstrap_sharpe(np.array(kelly_reuse_eq))
        mc_results["Kelly_Reuse"] = {"sharpe_bootstrap": bs_kelly}
        print("\nKelly Reuse Combined:")
        print(
            f"  Sharpe: {bs_kelly['mean']:.3f} (95% CI: [{bs_kelly['ci_5']:.3f}, {bs_kelly['ci_95']:.3f}]), "
            f"P(Sharpe>0) = {bs_kelly['p_positive']:.1f}%"
        )

        orb_eq = np.array(orb_result.equity_curve)
        vwap_eq = np.array(vwap_result.equity_curve)
        n_min = min(len(orb_eq), len(vwap_eq))
        if n_min >= 2:
            orb_rets = np.diff(orb_eq[:n_min]) / orb_eq[: n_min - 1]
            vwap_rets = np.diff(vwap_eq[:n_min]) / vwap_eq[: n_min - 1]
            mask = np.isfinite(orb_rets) & np.isfinite(vwap_rets)
            corr = float(np.corrcoef(orb_rets[mask], vwap_rets[mask])[0, 1]) if mask.sum() > 10 else 0
        else:
            corr = 0
        mc_results["correlation"] = round(corr, 4)
        print(f"\nORB-VWAP Return Correlation: {corr:.4f}")
        if abs(corr) < 0.3:
            diversification_note = "低相關，保有分散效果"
        elif abs(corr) < 0.6:
            diversification_note = "中度相關，分散效果有限"
        else:
            diversification_note = "高度相關，組合優勢偏弱"
        print(f"  -> {diversification_note}")

    results = {
        "mode": "orb_plus_vwap" if vwap_enabled else "orb_only_default",
        "vwap_enabled": vwap_enabled,
        "full": {
            "orb": orb_result.metrics,
            "vwap": vwap_result.metrics if vwap_result is not None else None,
            "combined": (
                {
                    **allocation_metrics,
                    "active_reuse_80": active_reuse_m,
                    "active_reuse_kelly": kelly_reuse_m,
                    "atr_adaptive_best": best_adaptive,
                    "atr_adaptive_rolling": am_roll,
                }
                if vwap_enabled
                else {}
            ),
        },
        "in_sample": {
            "orb": orb_is.metrics,
            "vwap": vwap_is.metrics if vwap_is is not None else None,
            "buy_hold": round(bh_is, 4),
        },
        "out_of_sample": {
            "orb": orb_oos.metrics,
            "vwap": vwap_oos.metrics if vwap_oos is not None else None,
            "buy_hold": round(bh_oos, 4),
            "combined_50_50": cm_oos_50 if vwap_enabled else {},
            "combined_60_40": cm_oos_60 if vwap_enabled else {},
            "combined_active_reuse_80": cm_oos_active if vwap_enabled else {},
            "combined_kelly_reuse": cm_oos_kelly if vwap_enabled else {},
            "adaptive_absolute": adaptive_oos_results if vwap_enabled else {},
            "adaptive_rolling": am_oos_roll if vwap_enabled else {},
        },
        "split_date": str(split_date),
        "is_days": split_idx,
        "oos_days": n_days - split_idx,
        "monte_carlo": mc_results,
    }

    with open(PROJECT_ROOT / "combined_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to combined_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
      