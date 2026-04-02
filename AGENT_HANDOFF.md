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
| 4 | 2026-04-02 14:19 | antigravity_nfw5t5 | ORB v4: orb_bars=4, trailing_pct=1.5%, bcp=0.03%; VWAP v3: EMA趨勢過濾+動態TP | 60.4% | +7.73% | -1.02% | 0.71 | ✅ 兩策略皆正回報 |
| 3 | 2026-04-02 12:41 UTC+8 | antigravity | ORB v3: 移除 EMA 過濾, 加突破確認 + 移動止損%, profit_ratio=4.0 | 56% | 5.95% | -1.23% | 0.25 | ✅ 正回報 |
| 2 | 2026-04-01 | agent_prev | ORB+VWAP 基準, STRUCTURE_LOOKBACK=20, OB_LOOKBACK=15 | 51.1% | -0.85% | -14.28% | 1.29 | ⚠️ 負 Alpha |
| 1 | 2026-04-01 | agent_init | 初始參數, 日線回測 | 50.5% | +186.67% | -11.07% | 1.82 | ✅ |
| 0 | — | — | 初始狀態 | — | — | — | — | — |

---

## 🔄 當前策略狀態快照

**ORB 策略 (v4) 最佳參數:**
```
orb_bars: 4 (20 分鐘 opening range)
profit_ratio: 3.5 (TP = 3.5x range width)
breakout_confirm_pct: 0.0003 (突破確認 0.03%)
trailing_stop: True
trailing_pct: 0.015 (移動止損 1.5%)
close_before_min: 15
```
績效: Return +0.82%, Win 60.4%, Sharpe 0.71, MaxDD -1.02%, 53 trades, PF 1.54

**VWAP Reversion 策略 (v3) 最佳參數:**
```
k: 2.0
sl_k_add: 0.3
std_window: 30
rsi_os: 30
rsi_ob: 75
max_trades_per_day: 2
ema_trend_filter: True (新增 — EMA20 > EMA50 判斷趨勢方向)
dynamic_tp: True (新增 — 趨勢方向止盈超越 VWAP)
tp_bonus_pct: 0.2
bb_width_min: 0.001
```
績效: Return +0.26%, Win 100% (3 trades), Sharpe 1.56, MaxDD -0.08%

**config.py 關鍵參數:**
```
VERSION: 1.2.0
ITERATION: 4
ATR_PERIOD: 14 (整數)
STOP_LOSS_ATR_MULT: 2.0
TAKE_PROFIT_RR: 4.0
MIN_SCORE_TO_TRADE: 6.0
STRUCTURE_LOOKBACK: 20 (整數)
OB_LOOKBACK: 15 (整數)
```

---

## ⚠️ 待辦 / 已知問題（下一個 Agent 優先處理）

1. **VWAP 交易次數過少 (3 筆/60天)**: EMA 趨勢過濾太嚴格，需要放寬條件。建議：
   - 嘗試只用 EMA20 方向（不需要 EMA20>EMA50 交叉）
   - 或改用價格相對 EMA20 的位置來判斷趨勢
   - 或放寬 k 到 1.5 增加觸發機會
2. **ORB 進一步優化**: 嘗試在 ORB 基礎上加入成交量確認（突破時成交量需高於均量）
3. **雙策略組合**: 考慮同時運行 ORB + VWAP，分配不同資金比例
4. **QQQ 交叉驗證**: 用 QQQ 數據做 out-of-sample 驗證，避免 NQ=F 過擬合
5. **ORB trailing_pct 微調**: 1.5% 可能在低波動日過寬，考慮用 ATR 動態計算 trailing stop

---

## 🚫 已嘗試但失敗的方向（避免重複踩坑）

| 迭代 | 嘗試內容 | 失敗原因 |
|------|---------|---------|
| 3 | ORB v2: EMA50 趨勢過濾 + 成交量確認 | 過濾太嚴格，勝率僅 29%，回報 -3.15% |
| 3 | VWAP: vol_confirm + atr_min_pct 過濾 | 沒有明顯改善，勝率仍 ~30% |
| 2 | 日線回測參數直接套用 5 分鐘線 | 時間框架差異太大，Alpha 為負 |

---

## 📝 備註

- 本次使用 NQ=F 5 分鐘線（60 天, ~13,422 根 K 棒）
- 回測引擎使用 manual fallback（無 vectorbt）
- ORB v4 grid search 測試了 720 組參數組合
- VWAP v3 grid search 測試了 10,368 組參數組合（含 EMA 開關排列）
- 關鍵發現：更寬的 trailing stop (1.5% vs 0.5%) 對 ORB 表現影響最大
- 關鍵發現：VWAP EMA 趨勢過濾有效提高勝率但嚴重減少交易次數
- 所有 lookback/period 參數已確認為整數型
- Buy & Hold 回報為 -6.91%，兩個策略都優於 B&H
