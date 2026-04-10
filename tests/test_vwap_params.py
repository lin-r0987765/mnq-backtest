#!/usr/bin/env python3
"""
VWAP 參數測試 - 測試增加交易機會的三個方向
v19 迭代優化
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# 添加項目根目錄到 path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.vwap_reversion import VWAPReversionStrategy
from src.backtest.engine import BacktestEngine

# ─────────────────────────────────────────────────────────────
# 加載數據
# ─────────────────────────────────────────────────────────────
print("加載 5 分鐘數據...")
df = pd.read_csv("qqq_5m.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df.set_index("DateTime", inplace=True)
df = df.sort_index()
print(f"Data shape: {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")

# ─────────────────────────────────────────────────────────────
# 基準參數（v13）
# ─────────────────────────────────────────────────────────────
baseline_params = {
    "k": 1.5,
    "sl_k_add": 0.5,
    "std_window": 30,
    "rsi_os": 32,
    "rsi_ob": 68,
    "max_trades_per_day": 2,
    "ema_trend_filter": True,
    "ema_mode": "ema_cross",
    "dynamic_tp": True,
    "tp_bonus_pct": 0.2,
    "bb_width_min": 0.0005,
    "entry_start_time": "09:45",
    "reversal_confirm": False,
    "prev_bar_reversal": False,
    "partial_tp": True,
    "partial_tp_trail_pct": 0.002,
    "partial_tp_max_hold": 32,
}

# ─────────────────────────────────────────────────────────────
# 測試參數組合（三個方向）
# ─────────────────────────────────────────────────────────────
test_cases = {
    "baseline_v13": baseline_params.copy(),
    "bb_width_0.0003": {**baseline_params, "bb_width_min": 0.0003},
    "entry_0930": {**baseline_params, "entry_start_time": "09:30"},
    "max_trades_3": {**baseline_params, "max_trades_per_day": 3},
    # 聯合測試（若單個改善良好）
    "bb_0.0003_entry_0930": {**baseline_params, "bb_width_min": 0.0003, "entry_start_time": "09:30"},
    "bb_0.0003_trades_3": {**baseline_params, "bb_width_min": 0.0003, "max_trades_per_day": 3},
}

results = {}

for case_name, params in test_cases.items():
    print(f"\n{'='*70}")
    print(f"測試: {case_name}")
    print(f"{'='*70}")

    # 創建策略並運行回測
    strategy = VWAPReversionStrategy(params)
    engine = BacktestEngine(
        df,
        strategy,
        initial_capital=100000,
        position_size_pct=0.02,
        max_positions=3,
        commission_per_trade=1.0,
    )

    result = engine.run()

    # 提取關鍵指標
    metrics = {
        "trades": len(result.trades),
        "win_rate_pct": result.metrics.get("win_rate_pct", np.nan),
        "sharpe": result.metrics.get("sharpe_ratio", np.nan),
        "max_dd_pct": result.metrics.get("max_drawdown_pct", np.nan),
        "return_pct": result.metrics.get("total_return_pct", np.nan),
        "profit_factor": result.metrics.get("profit_factor", np.nan),
    }

    results[case_name] = metrics

    print(f"交易次數: {metrics['trades']}")
    print(f"勝率: {metrics['win_rate_pct']:.1f}%")
    print(f"Sharpe: {metrics['sharpe']:.3f}")
    print(f"MaxDD: {metrics['max_dd_pct']:.3f}%")
    print(f"Return: {metrics['return_pct']:.3f}%")
    print(f"PF: {metrics['profit_factor']:.3f}")

# ─────────────────────────────────────────────────────────────
# 比較結果
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("結果比較")
print(f"{'='*70}\n")

# 基準值
baseline = results["baseline_v13"]
print(f"基準 (v13):")
print(f"  Trades: {baseline['trades']}, Sharpe: {baseline['sharpe']:.3f}, WR: {baseline['win_rate_pct']:.1f}%, MaxDD: {baseline['max_dd_pct']:.3f}%")

print(f"\n改進方向:")
for case_name in ["bb_width_0.0003", "entry_0930", "max_trades_3"]:
    if case_name in results:
        m = results[case_name]
        trades_delta = m['trades'] - baseline['trades']
        sharpe_delta = m['sharpe'] - baseline['sharpe']
        wr_delta = m['win_rate_pct'] - baseline['win_rate_pct']
        dd_delta = m['max_dd_pct'] - baseline['max_dd_pct']

        print(f"\n{case_name}:")
        print(f"  Trades: {m['trades']} ({trades_delta:+d}), Sharpe: {m['sharpe']:.3f} ({sharpe_delta:+.3f}), WR: {m['win_rate_pct']:.1f}% ({wr_delta:+.1f}pp), MaxDD: {m['max_dd_pct']:.3f}% ({dd_delta:+.3f}pp)")

print(f"\n聯合測試:")
for case_name in ["bb_0.0003_entry_0930", "bb_0.0003_trades_3"]:
    if case_name in results:
        m = results[case_name]
        trades_delta = m['trades'] - baseline['trades']
        sharpe_delta = m['sharpe'] - baseline['sharpe']
        wr_delta = m['win_rate_pct'] - baseline['win_rate_pct']
        dd_delta = m['max_dd_pct'] - baseline['max_dd_pct']

        print(f"\n{case_name}:")
        print(f"  Trades: {m['trades']} ({trades_delta:+d}), Sharpe: {m['sharpe']:.3f} ({sharpe_delta:+.3f}), WR: {m['win_rate_pct']:.1f}% ({wr_delta:+.1f}pp), MaxDD: {m['max_dd_pct']:.3f}% ({dd_delta:+.3f}pp)")

# 保存結果
output_file = "vwap_param_test_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n結果已保存到: {output_file}")
