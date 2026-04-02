# AGENT_HANDOFF.md — Agent 協作交接文檔

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
| 3 | 2026-04-02 12:41 UTC+8 | antigravity | ORB v3: 移除 EMA 過濾, 加突破確認 + 移動止損%, profit_ratio=4.0 | 56% | 5.95% | -1.23% | 0.25 | ✅ 正回報 |
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
