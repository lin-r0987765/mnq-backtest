#!/usr/bin/env python3
"""
簡化的回測運行器 - v19 參數測試
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加項目根目錄到 path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.strategies.vwap_reversion import VWAPReversionStrategy
    from src.backtest.engine import BacktestEngine

    # 加載數據
    print("加載 5 分鐘數據...")
    df = pd.read_csv("qqq_5m.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    df = df.sort_index()
    print(f"Data shape: {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")

    # 運行基準回測
    print("\n運行 VWAP v19 回測...")
    strategy = VWAPReversionStrategy()  # 使用新的預設參數
    engine = BacktestEngine(
        df,
        strategy,
        initial_capital=100000,
        position_size_pct=0.02,
        max_positions=3,
        commission_per_trade=1.0,
    )

    result = engine.run()
    m = result.metrics

    print(f"\n結果:")
    print(f"  交易次數: {m.get('total_trades', 0)}")
    print(f"  勝率: {m.get('win_rate_pct', 0):.1f}%")
    print(f"  Sharpe: {m.get('sharpe_ratio', 0):.3f}")
    print(f"  MaxDD: {m.get('max_drawdown_pct', 0):.3f}%")
    print(f"  Return: {m.get('total_return_pct', 0):.3f}%")
    print(f"  PF: {m.get('profit_factor', 0):.3f}")

except Exception as e:
    print(f"錯誤: {e}")
    import traceback
    traceback.print_exc()
