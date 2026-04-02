#!/usr/bin/env python3
"""取得 QQQ 多時間框架市場數據"""
import yfinance as yf
import pandas as pd

# === 5 分鐘線（近 60 天，Yahoo 免費上限）===
print("下載 QQQ 5 分鐘線...")
qqq_5m = yf.download("QQQ", period="60d", interval="5m")
qqq_5m.to_csv("qqq_5m.csv")
print(f"5 分鐘線: {len(qqq_5m)} 根 K 棒, 範圍 {qqq_5m.index[0]} ~ {qqq_5m.index[-1]}")

# === 1 小時線（多時間框架結構判斷用）===
print("下載 QQQ 1 小時線...")
qqq_1h = yf.download("QQQ", period="2y", interval="1h")
qqq_1h.to_csv("qqq_1h.csv")
print(f"1 小時線: {len(qqq_1h)} 根 K 棒, 範圍 {qqq_1h.index[0]} ~ {qqq_1h.index[-1]}")

# === 日線（趨勢過濾用）===
print("下載 QQQ 日線...")
qqq_1d = yf.download("QQQ", period="3y", interval="1d")
qqq_1d.to_csv("qqq_1d.csv")
print(f"日線: {len(qqq_1d)} 根 K 棒")

# 驗證數據完整性
for name, df in [("5m", qqq_5m), ("1h", qqq_1h), ("1d", qqq_1d)]:
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"⚠️ {name} 有 {nulls} 個空值，已前向填充")
        df.ffill(inplace=True)
        df.to_csv(f"qqq_{name}.csv")

print("✅ 數據取得完成")
