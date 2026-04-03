"""更新指標和交接文檔"""
import json, datetime, random, string

# === 更新 metrics.json ===
latest = json.load(open("results/backtest_20260402_043756_ORB_Optimised.json"))
m = latest["metrics"]

# 計算 buy & hold return
import pandas as pd
df_raw = pd.read_csv("qqq_5m.csv", header=[0, 1], index_col=0, parse_dates=True)
df_raw.columns = df_raw.columns.get_level_values(0)
bh_return = (df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] - 1) * 100

new_metrics = {
    "total_return_pct": round(m["total_return_pct"], 2),
    "final_equity": round(100000 * (1 + m["total_return_pct"] / 100), 2),
    "initial_capital": 100000.0,
    "total_trades": m["total_trades"],
    "winners": int(m["total_trades"] * m["win_rate_pct"] / 100),
    "losers": m["total_trades"] - int(m["total_trades"] * m["win_rate_pct"] / 100),
    "win_rate": round(m["win_rate_pct"], 2),
    "profit_factor": round(m["profit_factor"], 2),
    "max_drawdown_pct": round(m["max_drawdown_pct"], 2),
    "sharpe_ratio": round(m["sharpe_ratio"], 2),
    "sortino_ratio": round(m["sortino_ratio"], 2),
    "calmar_ratio": round(m["calmar_ratio"], 2),
    "buy_hold_return_pct": round(float(bh_return), 2),
    "alpha_vs_buy_hold": round(m["total_return_pct"] - float(bh_return), 2),
    "strategy": "ORB_Optimised_v3",
    "data_interval": "5m",
    "data_period": "60d",
    "best_params": {
        "orb_bars": 4,
        "profit_ratio": 4.0,
        "breakout_confirm_pct": 0.0005,
        "trailing_stop": True,
        "trailing_pct": 0.008
    }
}

with open("metrics.json", "w") as f:
    json.dump(new_metrics, f, indent=2)
print(f"✅ metrics.json 已更新: Return={new_metrics['total_return_pct']}%, Win={new_metrics['win_rate']}%, Sharpe={new_metrics['sharpe_ratio']}")

# === 更新 experience.json ===
exp = {
    "lessons": [
        "ORB v3 在 5 分鐘線上實現正回報 (+0.27%)",
        "orb_bars=4 (20分鐘) + profit_ratio=4.0 為最佳組合",
        "移動止損 (trailing_pct=0.8%) 有效保護利潤",
        "VWAP Reversion 策略在近期市場仍偏虧損，需進一步調整",
        "突破確認 (0.05%) 有效過濾假突破",
        "趨勢過濾器 (EMA) 在 5 分鐘線上過於嚴格，已移除"
    ],
    "suggestions": [
        "嘗試組合 ORB + VWAP 雙策略資金分配",
        "VWAP 策略可能需要更寬的 band (k >= 2.5) 或不同的 RSI 門檻",
        "考慮加入市場波動率 regime 判斷",
        "嘗試更長的 opening range (8-12 bars = 40-60 分鐘)",
        "下一步: 優化 VWAP 策略的虧損控制"
    ],
    "timestamp": datetime.datetime.now().isoformat(),
    "metrics_summary": {
        "win_rate": new_metrics["win_rate"],
        "sharpe": new_metrics["sharpe_ratio"],
        "max_drawdown": abs(new_metrics["max_drawdown_pct"]),
        "profit_factor": new_metrics["profit_factor"],
        "alpha": new_metrics["alpha_vs_buy_hold"],
        "total_trades": new_metrics["total_trades"]
    }
}

with open("experience.json", "w") as f:
    json.dump(exp, f, indent=2, ensure_ascii=False)
print("✅ experience.json 已更新")

# === 更新迭代日誌 ===
log_entry = {
    "iteration": 3,
    "timestamp": datetime.datetime.now().isoformat(),
    "version": "1.1.0",
    "parameters": {
        "orb_bars": 4,
        "profit_ratio": 4.0,
        "breakout_confirm_pct": 0.0005,
        "trailing_stop": True,
        "trailing_pct": 0.008
    },
    "metrics": new_metrics
}

with open("iteration_logs/optimization_history.jsonl", "a") as f:
    f.write(json.dumps(log_entry, default=str) + "\n")
print("✅ 迭代日誌已更新")

# === 建立 AGENT_HANDOFF.md ===
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
now_str = now.strftime("%Y-%m-%d %H:%M UTC+8")

handoff = f"""# AGENT_HANDOFF.md — Agent 協作交接文檔

> 此文檔由自動優化代理維護，用於多 Agent 協作時的狀態同步。
> 每次迭代開始前必須讀取，結束後必須更新。

---

## 🔒 鎖定狀態

| 欄位 | 值 |
|------|-----|
| 狀態 | `IDLE` |
| 佔用者 | — |
| 佔用時間 | — |
| 預計完成 | — |

> 狀態值：`IDLE`（空閒可接手）/ `RUNNING`（執行中勿動）/ `FAILED`（上次失敗需檢查）

---

## 📋 最近 5 次迭代摘要（最新在上，超過 5 行刪除最舊的）

| # | 時間 (UTC+8) | Agent | 修改摘要 | 勝率 | Alpha | 回撤 | Sharpe | 結果 |
|---|-------------|-------|---------|------|-------|------|--------|------|
| 3 | {now_str} | antigravity | ORB v3: 移除 EMA 過濾, 加突破確認 + 移動止損%, profit_ratio=4.0 | 56% | {new_metrics['alpha_vs_buy_hold']}% | {new_metrics['max_drawdown_pct']}% | {new_metrics['sharpe_ratio']} | ✅ 正回報 |
| 2 | 2026-04-01 | agent_prev | ORB+VWAP 基準, STRUCTURE_LOOKBACK=20, OB_LOOKBACK=15 | 51.1% | -0.85% | -14.28% | 1.29 | ⚠️ 負 Alpha |
| 1 | 2026-04-01 | agent_init | 初始參數, 日線回測 | 50.5% | +186.67% | -11.07% | 1.82 | ✅ |
| 0 | — | — | 初始狀態 | — | — | — | — | — |

---

## 🔄 當前策略狀態快照

**ORB 策略 (v3) 最佳參數:**
```
orb_bars: 4 (20 分鐘 opening range)
profit_ratio: 4.0 (TP = 4x range width)
breakout_confirm_pct: 0.0005 (突破確認 0.05%)
trailing_stop: True
trailing_pct: 0.008 (移動止損 0.8%)
close_before_min: 15
```

**VWAP Reversion 策略 (v2) 最佳參數:**
```
k: 1.5
sl_k_add: 0.5
std_window: 20
rsi_os: 30
rsi_ob: 75
max_trades_per_day: 1
```

**config.py 關鍵參數:**
```
ATR_PERIOD: 14 (整數)
STOP_LOSS_ATR_MULT: 2.0
TAKE_PROFIT_RR: 4.0
MIN_SCORE_TO_TRADE: 6.0
STRUCTURE_LOOKBACK: 20 (整數)
OB_LOOKBACK: 15 (整數)
```

---

## ⚠️ 待辦 / 已知問題（下一個 Agent 優先處理）

1. **VWAP 策略仍虧損**: 最佳表現仍為 -0.47%, 需要根本性改進。考慮:
   - 加入 EMA 趨勢過濾（只做順勢均值回歸）
   - 使用更寬的 band (k=3.0+) 減少交易次數但提高勝率
   - 或者完全替換為其他策略（如 Mean Reversion with Bollinger）
2. **ORB 交易次數偏低**: 60 筆交易 / 60 天 = ~1 筆/天，統計顯著性有限
3. **嘗試更長的 opening range**: orb_bars=8 或 12 可能在大趨勢日表現更好
4. **加入雙策略協同機制**: ORB 和 VWAP 可協同分配資金

---

## 🚫 已嘗試但失敗的方向（避免重複踩坑）

| 迭代 | 嘗試內容 | 失敗原因 |
|------|---------|---------|
| 3 | ORB v2: EMA50 趨勢過濾 + 成交量確認 | 過濾太嚴格，勝率僅 29%，回報 -3.15% |
| 3 | VWAP: vol_confirm + atr_min_pct 過濾 | 沒有明顯改善，勝率仍 ~30% |
| 2 | 日線回測參數直接套用 5 分鐘線 | 時間框架差異太大，Alpha 為負 |

---

## 📝 備註

- 本次使用 NQ=F 5 分鐘線（60 天, ~12,870 根 K 棒）
- 回測引擎使用 manual fallback（無 vectorbt）
- ORB 策略 v3 是目前唯一正回報的配置
- 所有 lookback/period 參數已確認為整數型
- 下次建議用 QQQ 數據做交叉驗證（避免過擬合 NQ=F）
"""

with open("AGENT_HANDOFF.md", "w", encoding="utf-8") as f:
    f.write(handoff)
print("✅ AGENT_HANDOFF.md 已建立")

# === 更新 backtest_report.txt ===
report = f"""======================================================================
    ICT 交易系統回測報告 - NQ=F (5 分鐘線)
    版本 v1.1.0 | 迭代 #3
======================================================================

【帳戶總覽】
  初始資金:     $100,000.00
  最終權益:     ${new_metrics['final_equity']:,.2f}
  總回報率:     {new_metrics['total_return_pct']}%
  Buy & Hold:   {new_metrics['buy_hold_return_pct']}%
  Alpha:        {new_metrics['alpha_vs_buy_hold']}%

【交易統計】
  策略:         ORB_Optimised_v3
  總交易次數:   {new_metrics['total_trades']}
  勝率:         {new_metrics['win_rate']}%
  獲利筆數:     {new_metrics['winners']}
  虧損筆數:     {new_metrics['losers']}
  利潤因子:     {new_metrics['profit_factor']}

【風險指標】
  Sharpe Ratio:  {new_metrics['sharpe_ratio']}
  Sortino Ratio: {new_metrics['sortino_ratio']}
  Calmar Ratio:  {new_metrics['calmar_ratio']}
  最大回撤:      {new_metrics['max_drawdown_pct']}%

【最佳參數】
  orb_bars: 4 (20 分鐘 opening range)
  profit_ratio: 4.0 (TP = 4x range width)
  breakout_confirm_pct: 0.0005
  trailing_stop: True (trailing_pct=0.8%)

【本次改進】
  1. ORB v3: 移除過度嚴格的 EMA/成交量過濾
  2. 加入突破確認閾值 (0.05%)
  3. 使用百分比移動止損替代固定止損
  4. 提高止盈倍數至 4.0x

【下次改進建議】
  1. 修復 VWAP 策略（仍虧損）
  2. 嘗試更長 opening range (40-60 分鐘)
  3. 加入市場波動率 regime 判斷
  4. 雙策略資金分配機制
======================================================================
"""

with open("backtest_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✅ backtest_report.txt 已更新")
print("\n🎯 所有文件更新完成！")
