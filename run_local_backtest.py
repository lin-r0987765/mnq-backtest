#!/usr/bin/env python3
"""
Local backtest runner — uses existing CSV files (no download).
Handles yfinance multi-level header format.
"""
import sys, os, json, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def load_csv_5m(path="qqq_5m.csv"):
    """Load 5m CSV with yfinance multi-header format."""
    df = pd.read_csv(path, header=[0,1], index_col=0)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("America/New_York")
    df = df[["Open","High","Low","Close","Volume"]].astype(float)
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    df.sort_index(inplace=True)
    df.index.name = None
    return df

def main():
    os.chdir(PROJECT_ROOT)
    
    from src.strategies.orb import ORBStrategy
    from src.strategies.vwap_reversion import VWAPReversionStrategy
    from src.backtest.engine import BacktestEngine, BacktestResult
    from src.optimizer.grid_search import grid_search, ORB_QUICK_RANGES, VWAP_QUICK_RANGES
    
    # Load data
    print("Loading 5m data from CSV...")
    df = load_csv_5m("qqq_5m.csv")
    print(f"  Bars: {len(df)} | {df.index[0]} ~ {df.index[-1]}")
    
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=1.0)
    
    # 1. Default runs
    print("\n=== ORB v5 (default) ===")
    orb_def = engine.run(ORBStrategy(), df)
    m = orb_def.metrics
    print(f"  Return: {m['total_return_pct']:+.2f}% | Win: {m['win_rate_pct']:.1f}% | "
          f"Sharpe: {m['sharpe_ratio']:.3f} | Trades: {m['total_trades']} | "
          f"MaxDD: {m['max_drawdown_pct']:.2f}%")
    
    print("\n=== VWAP v4 (default) ===")
    vwap_def = engine.run(VWAPReversionStrategy(), df)
    m2 = vwap_def.metrics
    print(f"  Return: {m2['total_return_pct']:+.2f}% | Win: {m2['win_rate_pct']:.1f}% | "
          f"Sharpe: {m2['sharpe_ratio']:.3f} | Trades: {m2['total_trades']} | "
          f"MaxDD: {m2['max_drawdown_pct']:.2f}%")
    
    # 2. Grid search
    print("\n=== ORB Grid Search ===")
    t0 = time.time()
    orb_top = grid_search(ORBStrategy, ORB_QUICK_RANGES, df, engine,
                          optimize_metric="sharpe_ratio", top_n=3)
    print(f"  ({time.time()-t0:.1f}s)")
    
    print("\n=== VWAP Grid Search ===")
    t0 = time.time()
    vwap_top = grid_search(VWAPReversionStrategy, VWAP_QUICK_RANGES, df, engine,
                           optimize_metric="sharpe_ratio", top_n=3)
    print(f"  ({time.time()-t0:.1f}s)")
    
    # 3. Best results
    best_orb = orb_top[0] if orb_top else orb_def
    best_vwap = vwap_top[0] if vwap_top else vwap_def
    
    print("\n=== BEST ORB ===")
    mo = best_orb.metrics
    print(f"  Params: {best_orb.params}")
    print(f"  Return: {mo['total_return_pct']:+.2f}% | Win: {mo['win_rate_pct']:.1f}% | "
          f"Sharpe: {mo['sharpe_ratio']:.3f} | Trades: {mo['total_trades']} | "
          f"MaxDD: {mo['max_drawdown_pct']:.2f}% | PF: {mo['profit_factor']:.3f}")
    
    print("\n=== BEST VWAP ===")
    mv = best_vwap.metrics
    print(f"  Params: {best_vwap.params}")
    print(f"  Return: {mv['total_return_pct']:+.2f}% | Win: {mv['win_rate_pct']:.1f}% | "
          f"Sharpe: {mv['sharpe_ratio']:.3f} | Trades: {mv['total_trades']} | "
          f"MaxDD: {mv['max_drawdown_pct']:.2f}% | PF: {mv['profit_factor']:.3f}")
    
    # Buy & Hold
    bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"\n  Buy & Hold: {bh_ret:+.2f}%")
    print(f"  ORB Alpha: {mo['total_return_pct'] - bh_ret:+.2f}%")
    print(f"  VWAP Alpha: {mv['total_return_pct'] - bh_ret:+.2f}%")
    
    # Save metrics
    metrics = {
        "total_return_pct": mo["total_return_pct"],
        "final_equity": 100000 * (1 + mo["total_return_pct"]/100),
        "initial_capital": 100000.0,
        "total_trades": mo["total_trades"],
        "winners": int(mo["total_trades"] * mo["win_rate_pct"] / 100),
        "losers": mo["total_trades"] - int(mo["total_trades"] * mo["win_rate_pct"] / 100),
        "win_rate": mo["win_rate_pct"],
        "profit_factor": mo["profit_factor"],
        "max_drawdown_pct": mo["max_drawdown_pct"],
        "sharpe_ratio": mo["sharpe_ratio"],
        "sortino_ratio": mo.get("sortino_ratio", 0),
        "calmar_ratio": mo.get("calmar_ratio", 0),
        "buy_hold_return_pct": round(bh_ret, 2),
        "alpha_vs_buy_hold": round(mo["total_return_pct"] - bh_ret, 2),
        "strategy": "ORB_v5_Optimised",
        "data_interval": "5m",
        "best_orb_params": best_orb.params,
        "vwap_v4_return_pct": mv["total_return_pct"],
        "vwap_v4_win_rate": mv["win_rate_pct"],
        "vwap_v4_trades": mv["total_trades"],
        "vwap_v4_sharpe": mv["sharpe_ratio"],
        "vwap_v4_params": best_vwap.params,
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print("\n✅ metrics.json saved")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
