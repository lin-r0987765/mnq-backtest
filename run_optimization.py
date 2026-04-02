#!/usr/bin/env python3
"""
迭代 #4 優化腳本
- ORB 擴展 grid search（加入更多 orb_bars 和 profit_ratio 組合）
- VWAP 加入 EMA 趨勢過濾 + 擴展 grid search
"""
import sys, json, time
sys.path.insert(0, '.')

from src.data.fetcher import fetch_nq_data
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine
from src.optimizer.grid_search import grid_search

def main():
    print("=" * 60)
    print("迭代 #4 — 策略優化")
    print("=" * 60)

    # Download data
    print("\n[Step 1] Downloading NQ=F 5-min data...")
    df = fetch_nq_data(symbol="NQ=F", period="60d")
    print(f"  Bars: {len(df)}  Range: {df.index[0]} ~ {df.index[-1]}")

    bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"  Buy & Hold: {bh_ret:+.2f}%")

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=1.0)

    # ── ORB Grid Search ──────────────────────────────────────────────
    print("\n[Step 2] ORB expanded grid search...")
    orb_ranges = {
        "orb_bars": [3, 4, 6, 8, 10, 12],
        "profit_ratio": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        "breakout_confirm_pct": [0.0003, 0.0005, 0.001, 0.0015],
        "trailing_stop": [True],
        "trailing_pct": [0.003, 0.005, 0.008, 0.01, 0.015],
    }

    t0 = time.time()
    orb_top = grid_search(
        ORBStrategy, orb_ranges, df, engine,
        optimize_metric="total_return_pct", top_n=10
    )
    print(f"  ORB grid done in {time.time()-t0:.1f}s")

    print("\n  === ORB Top 10 ===")
    orb_results = []
    for i, r in enumerate(orb_top):
        m = r.metrics
        p = r.params
        orb_results.append({"rank": i+1, "params": p, "metrics": m})
        print(f"  #{i+1}: Ret={m['total_return_pct']:+.2f}% "
              f"Win={m['win_rate_pct']:.1f}% "
              f"Sharpe={m['sharpe_ratio']:.3f} "
              f"MaxDD={m['max_drawdown_pct']:.2f}% "
              f"Trades={m['total_trades']} "
              f"PF={m['profit_factor']:.2f} "
              f"| bars={p.get('orb_bars')} pr={p.get('profit_ratio')} "
              f"bcp={p.get('breakout_confirm_pct')} tp={p.get('trailing_pct')}")

    # ── VWAP Grid Search ─────────────────────────────────────────────
    print("\n[Step 3] VWAP expanded grid search...")
    vwap_ranges = {
        "k": [1.5, 2.0, 2.5, 3.0, 3.5],
        "sl_k_add": [0.3, 0.5, 0.8, 1.0],
        "std_window": [15, 20, 30],
        "rsi_os": [25, 30, 35],
        "rsi_ob": [65, 70, 75, 80],
        "max_trades_per_day": [1, 2, 3],
    }

    t0 = time.time()
    vwap_top = grid_search(
        VWAPReversionStrategy, vwap_ranges, df, engine,
        optimize_metric="total_return_pct", top_n=10
    )
    print(f"  VWAP grid done in {time.time()-t0:.1f}s")

    print("\n  === VWAP Top 10 ===")
    vwap_results = []
    for i, r in enumerate(vwap_top):
        m = r.metrics
        p = r.params
        vwap_results.append({"rank": i+1, "params": p, "metrics": m})
        print(f"  #{i+1}: Ret={m['total_return_pct']:+.2f}% "
              f"Win={m['win_rate_pct']:.1f}% "
              f"Sharpe={m['sharpe_ratio']:.3f} "
              f"MaxDD={m['max_drawdown_pct']:.2f}% "
              f"Trades={m['total_trades']} "
              f"PF={m['profit_factor']:.2f} "
              f"| k={p.get('k')} sl_k={p.get('sl_k_add')} "
              f"rsi={p.get('rsi_os')}/{p.get('rsi_ob')} "
              f"maxT={p.get('max_trades_per_day')}")

    # Save all results
    all_results = {
        "buy_hold_return_pct": round(float(bh_ret), 4),
        "data_bars": len(df),
        "orb_top10": orb_results,
        "vwap_top10": vwap_results,
    }
    with open("optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n✅ Optimization complete. Results saved to optimization_results.json")

if __name__ == "__main__":
    main()
