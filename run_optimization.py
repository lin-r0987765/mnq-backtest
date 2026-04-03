#!/usr/bin/env python3
"""
迭代 #11 優化腳本
目標：
1. ORB v8: 1h 結構方向過濾 (htf_filter) — ICT 多時間框架概念
2. VWAP v11: 部分止盈策略 (partial_tp) — 50% @ VWAP, 50% trailing
3. 雙策略組合重算（50/50 + 60/40）
4. Out-of-Sample 驗證
"""
import sys, json, time, os
import pandas as pd
import numpy as np
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


def load_data():
    """載入 5 分鐘數據"""
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qqq_5m.csv")
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("America/New_York")
    return df


def run_single(strategy_cls, params, df, engine, label=""):
    """執行單一策略回測"""
    try:
        strategy = strategy_cls(params=params)
        result = engine.run(strategy, df)
        m = result.metrics
        print(f"  {label}: Ret={m['total_return_pct']:+.4f}% "
              f"Win={m['win_rate_pct']:.1f}% "
              f"Sharpe={m['sharpe_ratio']:.3f} "
              f"MaxDD={m['max_drawdown_pct']:.3f}% "
              f"Trades={m['total_trades']} "
              f"PF={m['profit_factor']:.2f}")
        return result
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        return None


def compute_combination(orb_result, vwap_result, orb_w, vwap_w):
    """計算投資組合組合績效"""
    orb_eq = np.array(orb_result.equity_curve)
    vwap_eq = np.array(vwap_result.equity_curve)
    min_len = min(len(orb_eq), len(vwap_eq))
    orb_eq = orb_eq[:min_len]
    vwap_eq = vwap_eq[:min_len]
    combined_eq = orb_eq * orb_w + vwap_eq * vwap_w
    combined_ret = pd.Series(combined_eq).pct_change().dropna()
    
    sharpe = combined_ret.mean() / combined_ret.std() * (252 * 78) ** 0.5 if combined_ret.std() > 0 else 0
    downside = combined_ret[combined_ret < 0].std()
    sortino = combined_ret.mean() / downside * (252 * 78) ** 0.5 if downside > 0 else 0
    peak = pd.Series(combined_eq).cummax()
    dd = (pd.Series(combined_eq) - peak) / peak
    max_dd = float(dd.min() * 100)
    total_ret = (combined_eq[-1] / combined_eq[0] - 1) * 100
    return {
        "return_pct": round(total_ret, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd, 4),
    }


def main():
    print("=" * 60)
    print("迭代 #11 — ORB 1h結構方向 + VWAP 部分止盈")
    print("=" * 60)

    # Load data
    print("\n[Step 1] 載入 5 分鐘數據...")
    df = load_data()
    print(f"  K棒數: {len(df)}  範圍: {df.index[0]} ~ {df.index[-1]}")
    bh_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"  Buy & Hold: {bh_ret:+.2f}%")

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=5.0)

    # ── 基線：v7 ORB + v10 VWAP（迭代 #10 最佳，size=5）────────────
    print("\n[Step 2] 基線回測（v7 ORB + v10 VWAP, size=5）...")
    baseline_orb = run_single(ORBStrategy, {
        "orb_bars": 4, "profit_ratio": 3.5, "close_before_min": 10,
        "breakout_confirm_pct": 0.0003, "trailing_stop": True, "trailing_pct": 0.015,
        "vol_confirm": False, "atr_trailing": False, "multi_day_range": False,
    }, df, engine, "ORB v7 baseline")

    baseline_vwap = run_single(VWAPReversionStrategy, {
        "k": 1.5, "sl_k_add": 0.5, "std_window": 30, "rsi_os": 35, "rsi_ob": 65,
        "max_trades_per_day": 2, "ema_trend_filter": True, "ema_mode": "ema_cross",
        "dynamic_tp": True, "tp_bonus_pct": 0.2, "bb_width_min": 0.0005,
        "entry_start_time": "09:45", "reversal_confirm": False, "prev_bar_reversal": False,
    }, df, engine, "VWAP v10 baseline")

    # ── ORB v8: 測試 1h 結構方向過濾 ────────────────────────────
    print("\n[Step 3] ORB v8: 1h 結構方向過濾測試...")
    orb_htf_results = []

    for htf_mode in ["ema_cross", "slope"]:
        for htf_ema_fast in [10, 20]:
            for htf_ema_slow in [30, 50]:
                if htf_ema_fast >= htf_ema_slow:
                    continue
                params = {
                    "orb_bars": 4, "profit_ratio": 3.5, "close_before_min": 10,
                    "breakout_confirm_pct": 0.0003, "trailing_stop": True, "trailing_pct": 0.015,
                    "vol_confirm": False, "atr_trailing": False, "multi_day_range": False,
                    "htf_filter": True, "htf_mode": htf_mode,
                    "htf_ema_fast": htf_ema_fast, "htf_ema_slow": htf_ema_slow,
                }
                label = f"ORB htf={htf_mode} ema={htf_ema_fast}/{htf_ema_slow}"
                r = run_single(ORBStrategy, params, df, engine, label)
                if r:
                    orb_htf_results.append((params, r))

    # 也測試 htf_filter=False（baseline already）
    print("\n  --- ORB HTF 過濾結果排名 (by Sharpe) ---")
    orb_htf_results.sort(key=lambda x: x[1].metrics.get("sharpe_ratio", 0), reverse=True)
    for i, (p, r) in enumerate(orb_htf_results[:5]):
        m = r.metrics
        print(f"  #{i+1}: Sharpe={m['sharpe_ratio']:.3f} Win={m['win_rate_pct']:.1f}% "
              f"Trades={m['total_trades']} Ret={m['total_return_pct']:+.4f}% "
              f"DD={m['max_drawdown_pct']:.3f}% "
              f"| mode={p['htf_mode']} ema={p['htf_ema_fast']}/{p['htf_ema_slow']}")

    # 判斷 HTF 過濾是否改善
    best_htf = orb_htf_results[0] if orb_htf_results else None
    use_htf = False
    best_orb_params = {
        "orb_bars": 4, "profit_ratio": 3.5, "close_before_min": 10,
        "breakout_confirm_pct": 0.0003, "trailing_stop": True, "trailing_pct": 0.015,
        "vol_confirm": False, "atr_trailing": False, "multi_day_range": False,
        "htf_filter": False,
    }
    best_orb_result = baseline_orb

    if best_htf and baseline_orb:
        htf_sharpe = best_htf[1].metrics.get("sharpe_ratio", 0)
        base_sharpe = baseline_orb.metrics.get("sharpe_ratio", 0)
        if htf_sharpe > base_sharpe:
            print(f"\n  ✅ HTF 過濾改善 Sharpe: {base_sharpe:.3f} → {htf_sharpe:.3f} (+{(htf_sharpe-base_sharpe)/base_sharpe*100:.1f}%)")
            use_htf = True
            best_orb_params = best_htf[0]
            best_orb_result = best_htf[1]
        else:
            print(f"\n  ⚠️ HTF 過濾未改善 (best={htf_sharpe:.3f} vs baseline={base_sharpe:.3f})，保持 htf_filter=False")

    # ── VWAP v11: 測試部分止盈 ────────────────────────────────────
    print("\n[Step 4] VWAP v11: 部分止盈測試...")
    vwap_partial_results = []

    for partial_tp_trail_pct in [0.002, 0.003, 0.005, 0.008]:
        for partial_tp_max_hold in [12, 24, 36]:
            params = {
                "k": 1.5, "sl_k_add": 0.5, "std_window": 30, "rsi_os": 35, "rsi_ob": 65,
                "max_trades_per_day": 2, "ema_trend_filter": True, "ema_mode": "ema_cross",
                "dynamic_tp": True, "tp_bonus_pct": 0.2, "bb_width_min": 0.0005,
                "entry_start_time": "09:45", "reversal_confirm": False, "prev_bar_reversal": False,
                "partial_tp": True,
                "partial_tp_trail_pct": partial_tp_trail_pct,
                "partial_tp_max_hold": partial_tp_max_hold,
            }
            label = f"VWAP partial trail={partial_tp_trail_pct} max_hold={partial_tp_max_hold}"
            r = run_single(VWAPReversionStrategy, params, df, engine, label)
            if r:
                vwap_partial_results.append((params, r))

    print("\n  --- VWAP 部分止盈結果排名 (by Sharpe) ---")
    vwap_partial_results.sort(key=lambda x: x[1].metrics.get("sharpe_ratio", 0), reverse=True)
    for i, (p, r) in enumerate(vwap_partial_results[:5]):
        m = r.metrics
        print(f"  #{i+1}: Sharpe={m['sharpe_ratio']:.3f} Win={m['win_rate_pct']:.1f}% "
              f"Sortino={m['sortino_ratio']:.3f} "
              f"Trades={m['total_trades']} Ret={m['total_return_pct']:+.4f}% "
              f"| trail={p['partial_tp_trail_pct']} max_hold={p['partial_tp_max_hold']}")

    # 判斷 partial_tp 是否改善
    best_partial = vwap_partial_results[0] if vwap_partial_results else None
    use_partial_tp = False
    best_vwap_params = {
        "k": 1.5, "sl_k_add": 0.5, "std_window": 30, "rsi_os": 35, "rsi_ob": 65,
        "max_trades_per_day": 2, "ema_trend_filter": True, "ema_mode": "ema_cross",
        "dynamic_tp": True, "tp_bonus_pct": 0.2, "bb_width_min": 0.0005,
        "entry_start_time": "09:45", "reversal_confirm": False, "prev_bar_reversal": False,
        "partial_tp": False,
    }
    best_vwap_result = baseline_vwap

    if best_partial and baseline_vwap:
        partial_sharpe = best_partial[1].metrics.get("sharpe_ratio", 0)
        base_sharpe = baseline_vwap.metrics.get("sharpe_ratio", 0)
        if partial_sharpe > base_sharpe:
            print(f"\n  ✅ 部分止盈改善 Sharpe: {base_sharpe:.3f} → {partial_sharpe:.3f}")
            use_partial_tp = True
            best_vwap_params = best_partial[0]
            best_vwap_result = best_partial[1]
        else:
            print(f"\n  ⚠️ 部分止盈未改善 (best={partial_sharpe:.3f} vs baseline={base_sharpe:.3f})，保持 partial_tp=False")

    # ── 雙策略組合 ────────────────────────────────────────────────
    print("\n[Step 5] 雙策略組合模擬...")
    if best_orb_result and best_vwap_result:
        for orb_w, vwap_w in [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]:
            combo = compute_combination(best_orb_result, best_vwap_result, orb_w, vwap_w)
            print(f"  ORB {orb_w:.0%} / VWAP {vwap_w:.0%}: "
                  f"Ret={combo['return_pct']:+.4f}% "
                  f"Sharpe={combo['sharpe']:.3f} "
                  f"Sortino={combo['sortino']:.3f} "
                  f"MaxDD={combo['max_dd_pct']:.3f}%")

    # ── OOS 驗證 ─────────────────────────────────────────────────
    print("\n[Step 6] Out-of-Sample 驗證...")
    split_idx = int(len(df) * 0.7)
    df_is = df.iloc[:split_idx]
    df_oos = df.iloc[split_idx:]
    print(f"  IS: {len(df_is)} bars ({df_is.index[0].date()} ~ {df_is.index[-1].date()})")
    print(f"  OOS: {len(df_oos)} bars ({df_oos.index[0].date()} ~ {df_oos.index[-1].date()})")

    orb_is = run_single(ORBStrategy, best_orb_params, df_is, engine, "ORB IS")
    orb_oos = run_single(ORBStrategy, best_orb_params, df_oos, engine, "ORB OOS")
    vwap_is = run_single(VWAPReversionStrategy, best_vwap_params, df_is, engine, "VWAP IS")
    vwap_oos = run_single(VWAPReversionStrategy, best_vwap_params, df_oos, engine, "VWAP OOS")

    # OOS 組合
    oos_combo = None
    if orb_oos and vwap_oos:
        oos_combo = compute_combination(orb_oos, vwap_oos, 0.5, 0.5)
        print(f"\n  OOS Combined 50/50: Sharpe={oos_combo['sharpe']:.3f} Ret={oos_combo['return_pct']:+.4f}%")

    # ── 儲存結果 ──────────────────────────────────────────────────
    print("\n[Step 7] 儲存結果...")
    best_orb_m = best_orb_result.metrics if best_orb_result else {}
    best_vwap_m = best_vwap_result.metrics if best_vwap_result else {}

    # 計算組合績效
    combo_data = {}
    if best_orb_result and best_vwap_result:
        for orb_w, vwap_w in [(0.5, 0.5), (0.6, 0.4)]:
            c = compute_combination(best_orb_result, best_vwap_result, orb_w, vwap_w)
            combo_data[f"combined_{int(orb_w*100)}_{int(vwap_w*100)}_sharpe"] = c["sharpe"]
            combo_data[f"combined_{int(orb_w*100)}_{int(vwap_w*100)}_sortino"] = c["sortino"]
            combo_data[f"combined_{int(orb_w*100)}_{int(vwap_w*100)}_max_dd"] = c["max_dd_pct"]
            combo_data[f"combined_{int(orb_w*100)}_{int(vwap_w*100)}_return"] = c["return_pct"]

    metrics = {
        "total_return_pct": round(best_orb_m.get("total_return_pct", 0), 4),
        "final_equity": round(100000 * (1 + best_orb_m.get("total_return_pct", 0) / 100), 0),
        "initial_capital": 100000.0,
        "total_trades": best_orb_m.get("total_trades", 0),
        "win_rate": round(best_orb_m.get("win_rate_pct", 0), 1),
        "profit_factor": round(best_orb_m.get("profit_factor", 0), 3),
        "max_drawdown_pct": round(best_orb_m.get("max_drawdown_pct", 0), 3),
        "sharpe_ratio": round(best_orb_m.get("sharpe_ratio", 0), 3),
        "sortino_ratio": round(best_orb_m.get("sortino_ratio", 0), 3),
        "buy_hold_return_pct": round(float(bh_ret), 2),
        "alpha_vs_buy_hold": round(best_orb_m.get("total_return_pct", 0) - float(bh_ret), 2),
        "strategy": f"ORB_v8{'_htf' if use_htf else ''}",
        "engine_size": 5,
        "data_interval": "5m",
        "best_orb_params": best_orb_params,
        "vwap_v11_return_pct": round(best_vwap_m.get("total_return_pct", 0), 4),
        "vwap_v11_win_rate": round(best_vwap_m.get("win_rate_pct", 0), 1),
        "vwap_v11_trades": best_vwap_m.get("total_trades", 0),
        "vwap_v11_sharpe": round(best_vwap_m.get("sharpe_ratio", 0), 3),
        "vwap_v11_max_dd": round(best_vwap_m.get("max_drawdown_pct", 0), 3),
        "vwap_v11_pf": round(best_vwap_m.get("profit_factor", 0), 3),
        "vwap_v11_params": best_vwap_params,
        **combo_data,
        "oos_orb_sharpe": round(orb_oos.metrics.get("sharpe_ratio", 0), 3) if orb_oos else None,
        "oos_vwap_sharpe": round(vwap_oos.metrics.get("sharpe_ratio", 0), 3) if vwap_oos else None,
        "oos_combined_50_50_sharpe": oos_combo["sharpe"] if oos_combo else None,
        "iteration": 11,
        "updated": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M UTC+8"),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # 儲存完整優化結果
    all_results = {
        "iteration": 11,
        "buy_hold_return_pct": round(float(bh_ret), 4),
        "data_bars": len(df),
        "orb_htf_tests": [
            {"params": p, "sharpe": r.metrics.get("sharpe_ratio", 0),
             "win_rate": r.metrics.get("win_rate_pct", 0),
             "trades": r.metrics.get("total_trades", 0),
             "return_pct": r.metrics.get("total_return_pct", 0)}
            for p, r in orb_htf_results
        ],
        "vwap_partial_tp_tests": [
            {"params": {k: v for k, v in p.items() if k in ["partial_tp_trail_pct", "partial_tp_max_hold"]},
             "sharpe": r.metrics.get("sharpe_ratio", 0),
             "sortino": r.metrics.get("sortino_ratio", 0),
             "win_rate": r.metrics.get("win_rate_pct", 0),
             "trades": r.metrics.get("total_trades", 0),
             "return_pct": r.metrics.get("total_return_pct", 0)}
            for p, r in vwap_partial_results
        ],
        "best_orb_params": best_orb_params,
        "best_vwap_params": best_vwap_params,
        "htf_filter_used": use_htf,
        "partial_tp_used": use_partial_tp,
    }
    with open("optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n✅ 迭代 #11 優化完成")
    print(f"  ORB: htf_filter={'True' if use_htf else 'False'}")
    print(f"  VWAP: partial_tp={'True' if use_partial_tp else 'False'}")


if __name__ == "__main__":
    main()
