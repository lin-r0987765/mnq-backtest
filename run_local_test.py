#!/usr/bin/env python3
"""Local backtest using CSV files - no network required."""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.strategies.orb import ORBStrategy
from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine

def load_csv(path="qqq_5m.csv"):
    df = pd.read_csv(path, header=[0,1], index_col=0)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("America/New_York")
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df[["Open","High","Low","Close","Volume"]].astype(float)
    df.sort_index(inplace=True)
    return df

def run_strategy(cls, params, df, engine, label=""):
    try:
        s = cls(params=params)
        r = engine.run(s, df)
        m = r.metrics
        print(f"  {label}: Return={m.get('total_return_pct',0):+.4f}% WR={m.get('win_rate_pct',0):.1f}% "
              f"Sharpe={m.get('sharpe_ratio',0):.3f} DD={m.get('max_drawdown_pct',0):.2f}% "
              f"Trades={m.get('total_trades',0)} PF={m.get('profit_factor',0):.3f}")
        return r
    except Exception as e:
        print(f"  {label}: ERROR - {e}")
        return None

if __name__ == "__main__":
    df = load_csv()
    print(f"Data: {len(df)} bars, {df.index[0].date()} ~ {df.index[-1].date()}")
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=1.0)
    
    # Baseline ORB (current best params)
    orb_params = {
        "orb_bars": 4, "profit_ratio": 3.5, "breakout_confirm_pct": 0.0003,
        "trailing_stop": True, "trailing_pct": 0.015,
        "vol_confirm": False, "vol_mult": 0.6,
        "atr_trailing": False, "atr_trail_mult": 1.5
    }
    print("\n=== ORB Baseline ===")
    orb_base = run_strategy(ORBStrategy, orb_params, df, engine, "ORB_v5")
    
    # Baseline VWAP (current best params from handoff)
    vwap_params = {
        "k": 1.5, "sl_k_add": 0.5, "std_window": 30,
        "rsi_os": 35, "rsi_ob": 65, "max_trades_per_day": 2,
        "ema_trend_filter": True, "ema_mode": "ema_cross",
        "dynamic_tp": True, "tp_bonus_pct": 0.2, "bb_width_min": 0.001
    }
    print("\n=== VWAP Baseline ===")
    vwap_base = run_strategy(VWAPReversionStrategy, vwap_params, df, engine, "VWAP_v4")
    
    # VWAP experiments - increase trade count
    print("\n=== VWAP Experiments ===")
    experiments = [
        ("k=1.2,ema_off", {**vwap_params, "k": 1.2, "ema_trend_filter": False}),
        ("k=1.2,price_vs_ema", {**vwap_params, "k": 1.2, "ema_mode": "price_vs_ema"}),
        ("k=1.2,ema_cross,start=09:45", {**vwap_params, "k": 1.2, "entry_start_time": "09:45"}),
        ("k=1.2,rsi40/60,bb0.0005", {**vwap_params, "k": 1.2, "rsi_os": 40, "rsi_ob": 60, "bb_width_min": 0.0005}),
        ("k=1.3,ema_off,trades=3", {**vwap_params, "k": 1.3, "ema_trend_filter": False, "max_trades_per_day": 3}),
        ("k=1.5,ema_off,rsi40/60", {**vwap_params, "k": 1.5, "ema_trend_filter": False, "rsi_os": 40, "rsi_ob": 60}),
        ("k=1.2,ema_cross,bb0.0005,trades=3", {**vwap_params, "k": 1.2, "bb_width_min": 0.0005, "max_trades_per_day": 3}),
        ("k=1.0,ema_cross,rsi40/60", {**vwap_params, "k": 1.0, "rsi_os": 40, "rsi_ob": 60}),
    ]
    
    best_vwap = None
    best_vwap_sharpe = -999
    for label, params in experiments:
        r = run_strategy(VWAPReversionStrategy, params, df, engine, label)
        if r and r.metrics.get("sharpe_ratio", 0) > best_vwap_sharpe and r.metrics.get("total_trades", 0) >= 10:
            best_vwap_sharpe = r.metrics["sharpe_ratio"]
            best_vwap = (label, params, r)
    
    # ORB experiments
    print("\n=== ORB Experiments ===")
    orb_experiments = [
        ("vol_confirm,mult=0.5", {**orb_params, "vol_confirm": True, "vol_mult": 0.5}),
        ("vol_confirm,mult=0.6", {**orb_params, "vol_confirm": True, "vol_mult": 0.6}),
        ("atr_trailing,mult=1.5", {**orb_params, "atr_trailing": True, "atr_trail_mult": 1.5}),
        ("atr_trailing,mult=2.0", {**orb_params, "atr_trailing": True, "atr_trail_mult": 2.0}),
        ("profit_ratio=4.0", {**orb_params, "profit_ratio": 4.0}),
        ("orb_bars=3,pr=3.5", {**orb_params, "orb_bars": 3, "profit_ratio": 3.5}),
        ("vol0.5+atr1.5", {**orb_params, "vol_confirm": True, "vol_mult": 0.5, "atr_trailing": True, "atr_trail_mult": 1.5}),
    ]
    
    best_orb = None
    best_orb_sharpe = -999
    for label, params in orb_experiments:
        r = run_strategy(ORBStrategy, params, df, engine, label)
        if r and r.metrics.get("sharpe_ratio", 0) > best_orb_sharpe:
            best_orb_sharpe = r.metrics["sharpe_ratio"]
            best_orb = (label, params, r)
    
    print("\n=== BEST RESULTS ===")
    if best_vwap:
        print(f"Best VWAP: {best_vwap[0]}")
        m = best_vwap[2].metrics
        print(f"  Return={m.get('total_return_pct',0):+.4f}% WR={m.get('win_rate_pct',0):.1f}% "
              f"Sharpe={m.get('sharpe_ratio',0):.3f} Trades={m.get('total_trades',0)}")
    if best_orb:
        print(f"Best ORB: {best_orb[0]}")
        m = best_orb[2].metrics
        print(f"  Return={m.get('total_return_pct',0):+.4f}% WR={m.get('win_rate_pct',0):.1f}% "
              f"Sharpe={m.get('sharpe_ratio',0):.3f} Trades={m.get('total_trades',0)}")
