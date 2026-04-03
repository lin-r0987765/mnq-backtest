#!/usr/bin/env python3
"""
迭代 #6 優化腳本
目標：
1. VWAP 增加交易量：k 降至 1.0-1.2，入場時間提前至 09:45，放寬/關閉 EMA 過濾
2. ORB 微調：vol_mult 低門檻 (0.5-0.7)，trailing 微調
3. 雙策略組合模擬（70/30 ORB/VWAP）
"""
import sys, json, time, os
import pandas as pd
sys.path.insert(0, '.')

# Patch yfinance to avoid import error in offline environment
import types
yf_mock = types.ModuleType("yfinance")
yf_mock.Ticker = None
sys.modules["yfinance"] = yf_mock

from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine
from src.optimizer.grid_search import grid_search

def main():
    print("=" * 60)
    print("迭代 #6 — VWAP 交易量提升 + 雙策略組合")
    print("=" * 60)

    # Load data from CSV (offline mode)
    print("\n[Step 1] Loading 5-min data from CSV...")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qqq_5m.csv")
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    # Flatten multi-level columns
    df.columns = df.columns.get_level_values(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("America/New_York")
    print(f"  Bars: {len(df)}  Range: {df.index[0]} ~ {df.index[-1]}")

    bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"  Buy & Hold: {bh_ret:+.2f}%")

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=1.0)

    # ── VWAP Grid Search（重點：增加交易量）──────────────────────────
    print("\n[Step 2] VWAP v5 grid search (focus: more trades)...")
    vwap_ranges = {
        "k": [1.0, 1.2, 1.5],
        "sl_k_add": [0.3, 0.5, 0.8],
        "std_window": [20, 30],
        "rsi_os": [35, 40, 45],
        "rsi_ob": [55, 60, 65],
        "max_trades_per_day": [2, 3, 4],
        "ema_trend_filter": [True, False],
        "ema_mode": ["price_vs_ema"],
        "entry_start_time": ["09:45"],
        "entry_end_time": ["15:30"],
        "dynamic_tp": [True],
        "tp_bonus_pct": [0.2, 0.3],
        "bb_width_min": [0.0005, 0.001],
    }

    t0 = time.time()
    vwap_top = grid_search(
        VWAPReversionStrategy, vwap_ranges, df, engine,
        optimize_metric="sharpe_ratio", top_n=15
    )
    print(f"  VWAP grid done in {time.time()-t0:.1f}s")

    print("\n  === VWAP Top 15 ===")
    vwap_results = []
    for i, r in enumerate(vwap_top):
        m = r.metrics
        p = r.params
        vwap_results.append({"rank": i+1, "params": p, "metrics": m})
        print(f"  #{i+1}: Ret={m['total_return_pct']:+.4f}% "
              f"Win={m['win_rate_pct']:.1f}% "
              f"Sharpe={m['sharpe_ratio']:.3f} "
              f"MaxDD={m['max_drawdown_pct']:.2f}% "
              f"Trades={m['total_trades']} "
              f"PF={m['profit_factor']:.2f} "
              f"| k={p.get('k')} ema={p.get('ema_trend_filter')} "
              f"rsi={p.get('rsi_os')}/{p.get('rsi_ob')} "
              f"maxT={p.get('max_trades_per_day')} "
              f"bb={p.get('bb_width_min')}")

    # ── ORB Grid Search（微調）──────────────────────────────────────
    print("\n[Step 3] ORB v6 grid search (fine-tune)...")
    orb_ranges = {
        "orb_bars": [3, 4, 6],
        "profit_ratio": [3.0, 3.5, 4.0, 5.0],
        "breakout_confirm_pct": [0.0002, 0.0003, 0.0005],
        "trailing_stop": [True],
        "trailing_pct": [0.01, 0.015, 0.02],
        "vol_confirm": [False, True],
        "vol_mult": [0.5, 0.7],
        "atr_trailing": [False, True],
        "atr_trail_mult": [1.5, 2.0],
    }

    t0 = time.time()
    orb_top = grid_search(
        ORBStrategy, orb_ranges, df, engine,
        optimize_metric="sharpe_ratio", top_n=15
    )
    print(f"  ORB grid done in {time.time()-t0:.1f}s")

    print("\n  === ORB Top 15 ===")
    orb_results = []
    for i, r in enumerate(orb_top):
        m = r.metrics
        p = r.params
        orb_results.append({"rank": i+1, "params": p, "metrics": m})
        print(f"  #{i+1}: Ret={m['total_return_pct']:+.4f}% "
              f"Win={m['win_rate_pct']:.1f}% "
              f"Sharpe={m['sharpe_ratio']:.3f} "
              f"MaxDD={m['max_drawdown_pct']:.2f}% "
              f"Trades={m['total_trades']} "
              f"PF={m['profit_factor']:.2f} "
              f"| bars={p.get('orb_bars')} pr={p.get('profit_ratio')} "
              f"bcp={p.get('breakout_confirm_pct')} tp={p.get('trailing_pct')} "
              f"vol={p.get('vol_confirm')} atr_t={p.get('atr_trailing')}")

    # ── 雙策略組合模擬 ─────────────────────────────────────────────
    print("\n[Step 4] Portfolio combination simulation...")
    if orb_top and vwap_top:
        best_orb = orb_top[0]
        best_vwap = vwap_top[0]

        # 模擬 70/30 組合
        orb_ret = best_orb.metrics.get("total_return_pct", 0)
        vwap_ret = best_vwap.metrics.get("total_return_pct", 0)
        orb_dd = best_orb.metrics.get("max_drawdown_pct", 0)
        vwap_dd = best_vwap.metrics.get("max_drawdown_pct", 0)
        orb_sharpe = best_orb.metrics.get("sharpe_ratio", 0)
        vwap_sharpe = best_vwap.metrics.get("sharpe_ratio", 0)

        for orb_w, vwap_w in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]:
            combo_ret = orb_ret * orb_w + vwap_ret * vwap_w
            combo_dd = orb_dd * orb_w + vwap_dd * vwap_w  # 近似（保守估計）
            combo_sharpe = orb_sharpe * orb_w + vwap_sharpe * vwap_w
            print(f"  ORB {orb_w:.0%} / VWAP {vwap_w:.0%}: "
                  f"Ret≈{combo_ret:+.4f}% Sharpe≈{combo_sharpe:.3f} DD≈{combo_dd:.2f}%")

    # Save all results
    all_results = {
        "iteration": 6,
        "buy_hold_return_pct": round(float(bh_ret), 4),
        "data_bars": len(df),
        "orb_top15": orb_results,
        "vwap_top15": vwap_results,
    }
    with open("optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Update metrics.json with best results
    if orb_top and vwap_top:
        best_orb_m = orb_top[0].metrics
        best_vwap_m = vwap_top[0].metrics
        metrics = {
            "total_return_pct": round(best_orb_m.get("total_return_pct", 0), 4),
            "final_equity": best_orb_m.get("final_equity", 100000),
            "initial_capital": 100000.0,
            "total_trades": best_orb_m.get("total_trades", 0),
            "winners": best_orb_m.get("winners", 0),
            "losers": best_orb_m.get("losers", 0),
            "win_rate": round(best_orb_m.get("win_rate_pct", 0), 4),
            "profit_factor": round(best_orb_m.get("profit_factor", 0), 4),
            "max_drawdown_pct": round(best_orb_m.get("max_drawdown_pct", 0), 4),
            "sharpe_ratio": round(best_orb_m.get("sharpe_ratio", 0), 4),
            "sortino_ratio": round(best_orb_m.get("sortino_ratio", 0), 4),
            "calmar_ratio": round(best_orb_m.get("calmar_ratio", 0), 4),
            "buy_hold_return_pct": round(float(bh_ret), 2),
            "alpha_vs_buy_hold": round(best_orb_m.get("total_return_pct", 0) - float(bh_ret), 2),
            "strategy": "ORB_v6_Optimised",
            "data_interval": "5m",
            "best_orb_params": orb_top[0].params,
            "vwap_v5_return_pct": round(best_vwap_m.get("total_return_pct", 0), 4),
            "vwap_v5_win_rate": round(best_vwap_m.get("win_rate_pct", 0), 4),
            "vwap_v5_trades": best_vwap_m.get("total_trades", 0),
            "vwap_v5_sharpe": round(best_vwap_m.get("sharpe_ratio", 0), 4),
            "vwap_v5_params": vwap_top[0].params,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    print("\n✅ 迭代 #6 優化完成")

if __name__ == "__main__":
    main()
