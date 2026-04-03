#!/usr/bin/env python3
"""
迭代 #4 驗證腳本
測試新的 ORB v4 和 VWAP v3 預設參數，以及 VWAP v3 grid search
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
    print("迭代 #4 — 驗證回測")
    print("=" * 60)

    df = fetch_nq_data(symbol="NQ=F", period="60d")
    bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"Bars: {len(df)}  B&H: {bh_ret:+.2f}%")

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=1.0)

    # ── ORB v4 default ───────────────────────────────────────────
    print("\n[1] ORB v4 (new default params):")
    orb = ORBStrategy()
    orb_res = engine.run(orb, df)
    m = orb_res.metrics
    print(f"  Return: {m['total_return_pct']:+.4f}%")
    print(f"  Win rate: {m['win_rate_pct']:.1f}%")
    print(f"  Sharpe: {m['sharpe_ratio']:.4f}")
    print(f"  MaxDD: {m['max_drawdown_pct']:.4f}%")
    print(f"  Trades: {m['total_trades']}")
    print(f"  PF: {m['profit_factor']:.4f}")
    print(f"  Alpha: {m['total_return_pct'] - bh_ret:+.4f}%")

    # ── VWAP v3 default ──────────────────────────────────────────
    print("\n[2] VWAP v3 (default with EMA trend filter):")
    vwap = VWAPReversionStrategy()
    vwap_res = engine.run(vwap, df)
    m = vwap_res.metrics
    print(f"  Return: {m['total_return_pct']:+.4f}%")
    print(f"  Win rate: {m['win_rate_pct']:.1f}%")
    print(f"  Sharpe: {m['sharpe_ratio']:.4f}")
    print(f"  MaxDD: {m['max_drawdown_pct']:.4f}%")
    print(f"  Trades: {m['total_trades']}")
    print(f"  PF: {m['profit_factor']:.4f}")
    print(f"  Alpha: {m['total_return_pct'] - bh_ret:+.4f}%")

    # ── VWAP v3 grid search for optimal params ────────────────────
    print("\n[3] VWAP v3 grid search (with EMA filter variations):")
    vwap_ranges = {
        "k": [2.0, 2.5, 3.0, 3.5],
        "sl_k_add": [0.3, 0.5, 0.8],
        "std_window": [20, 30],
        "rsi_os": [25, 30, 35],
        "rsi_ob": [75, 80],
        "max_trades_per_day": [1, 2],
        "ema_trend_filter": [True, False],
        "dynamic_tp": [True, False],
        "tp_bonus_pct": [0.2, 0.3, 0.5],
        "bb_width_min": [0.001, 0.002, 0.003],
    }

    t0 = time.time()
    vwap_top = grid_search(
        VWAPReversionStrategy, vwap_ranges, df, engine,
        optimize_metric="total_return_pct", top_n=10
    )
    print(f"  Grid done in {time.time()-t0:.1f}s")

    print("\n  === VWAP v3 Top 10 ===")
    vwap_results = []
    for i, r in enumerate(vwap_top):
        m = r.metrics
        p = r.params
        vwap_results.append({"rank": i+1, "params": p, "metrics": m})
        ema_on = p.get("ema_trend_filter", False)
        dtp = p.get("dynamic_tp", False)
        print(f"  #{i+1}: Ret={m['total_return_pct']:+.4f}% "
              f"Win={m['win_rate_pct']:.1f}% "
              f"Sharpe={m['sharpe_ratio']:.3f} "
              f"DD={m['max_drawdown_pct']:.2f}% "
              f"PF={m['profit_factor']:.2f} "
              f"T={m['total_trades']} "
              f"EMA={ema_on} DTP={dtp} "
              f"k={p.get('k')} bb={p.get('bb_width_min')}")

    # Save final validation results
    final = {
        "buy_hold_return_pct": round(float(bh_ret), 4),
        "orb_v4": {
            "metrics": orb_res.metrics,
            "params": orb_res.params
        },
        "vwap_v3_default": {
            "metrics": vwap_res.metrics,
            "params": vwap_res.params
        },
        "vwap_v3_grid_top10": vwap_results,
    }
    with open("validation_results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n✅ Validation complete. Saved to validation_results.json")

if __name__ == "__main__":
    main()
