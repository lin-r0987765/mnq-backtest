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
| 9 | 2026-04-03 15:10 UTC+8 | agent_fasv07 | VWAP v9: k=1.5,sl_k=0.5; 反轉放寬3模式(仍關閉); ORB pr=4.0測試(退步); 組合size=5重算+OOS | 59.4%/71.4% | +5.82%/+5.72% | -0.12%/-0.015% | 1.52/2.14 | ✅ VWAP Sharpe+4.7% |
| 8 | 2026-04-03 10:10 UTC+8 | agent_o282l3 | size 1→5; VWAP reversal_confirm(失敗); ORB grid orb_bars=[3,4,5] | 59.4%/75.0% | +5.82%/+5.71% | -0.12%/-0.015% | 1.52/2.04 | ✅ 回報×5 |
| 7 | 2026-04-03 04:19 UTC+8 | agent_lssyn4 | VWAP v7: k=1.3,sl_k=0.7; 雙策略組合+OOS驗證; fetcher修復 | 59.4%/75.0% | +5.71%/+5.69% | -0.02%/-0.00% | 1.52/2.04 | ✅ VWAP +14%交易 +3.6pp WR |
| 6 | 2026-04-03 02:20 UTC+8 | agent_39tcbf | VWAP v5: start=09:45,bb=0.0005,ema_cross默認; ORB v6: 默認修正vol/atr=False | 59.4%/71.4% | +5.71%/+5.69% | -0.02%/-0.00% | 1.52/2.14 | ✅ VWAP Sharpe+23% |
| 5 | 2026-04-02 20:24 UTC+8 | agent_4cm45s | VWAP v4: k=1.5,rsi放寬,ema_mode; ORB v5: vol_confirm+atr_trailing | 59.4%/69.2% | +5.71%/+5.69% | -0.02%/-0.00% | 1.52/1.74 | ✅ 雙策略改善 |
| 4 | 2026-04-02 14:19 | antigravity_nfw5t5 | ORB v4: orb_bars=4, trailing_pct=1.5%, bcp=0.03%; VWAP v3: EMA趨勢過濾+動態TP | 60.4% | +7.73% | -1.02% | 0.71 | ✅ 兩策略皆正回報 |

---

## 🔄 當前策略狀態快照

**Engine Size**: 5

**ORB 策略 (v6) 最佳參數:**
```
orb_bars: 4
profit_ratio: 3.5
breakout_confirm_pct: 0.0003
trailing_stop: True
trailing_pct: 0.015
vol_confirm: False
atr_trailing: False
close_before_min: 15
```
績效 (size=5): Return +0.133%, Win 59.4%, Sharpe 1.523, MaxDD -0.119%, 64 trades, PF 1.63

**VWAP Reversion 策略 (v9) 最佳參數:**
```
k: 1.5
sl_k_add: 0.5
std_window: 30
rsi_os: 35
rsi_ob: 65
max_trades_per_day: 2
ema_trend_filter: True
ema_mode: ema_cross
dynamic_tp: True
tp_bonus_pct: 0.2
bb_width_min: 0.0005
entry_start_time: 09:45
reversal_confirm: False
reversal_mode: relaxed (新增但預設關閉)
```
績效 (size=5): Return +0.035%, Win 71.4% (14 trades), Sharpe 2.14, MaxDD -0.015%, PF 3.89

**雙策略組合 (size=5):**
- 50/50: Sharpe 1.884, Sortino 2.468, MaxDD -0.062%
- 60/40: Sharpe 1.772, Sortino 2.341, MaxDD -0.073%
- OOS Combined 60/40 Sharpe: 3.311

**config.py 關鍵參數:**
```
VERSION: 1.4.0
ITERATION: 9
ATR_PERIOD: 14
STOP_LOSS_ATR_MULT: 2.0
TAKE_PROFIT_RR: 4.0
MIN_SCORE_TO_TRADE: 6.0
STRUCTURE_LOOKBACK: 20
OB_LOOKBACK: 15
```

## ⚠️ 待辦 / 已知問題（下一個 Agent 優先處理）

1. **VWAP 交易數偏少 (14筆/60天)**: 嘗試 max_trades_per_day=3 或放寬 entry 時間至 09:35
2. **VWAP 前一根K棒反轉確認**: 改檢查 entry bar 前一根（而非當根）的方向
3. **ORB 多日聚合 range**: 嘗試前 2-3 天的 H/L 作為 range（捕捉更大突破）
4. **size=10 風險評估**: 回報翻倍但 MaxDD 可能接近 1%
5. **策略切換機制**: VIX 高（或 ATR 高）用 ORB，低用 VWAP

## 🚫 已嘗試但失敗的方向（避免重複踩坑）

| 迭代 | 嘗試內容 | 失敗原因 |
|------|---------|---------|
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
- v7: OOS 驗證通過 — ORB OOS Sharpe 2.78, VWAP OOS Sharpe 3.57
- v7: 50/50 組合 Sharpe 1.88 為最佳配置
- v7: data fetcher 已修復支援無 yfinance 環境（本地 CSV fallback）
- Buy & Hold 回報為 -6.91%，兩個策略都優於 B&H
