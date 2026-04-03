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
| 11 | 2026-04-04 01:10 UTC+8 | agent_3mnx0d | ORB v8 htf_filter=slope(EMA20/30); VWAP v11 partial_tp(trail=0.002,hold=36) | 70.4%/69.2% | +5.94%/+5.76% | -0.079%/-0.024% | 3.315/3.629 | ✅ 雙策略 Sharpe 翻倍 |
| 10 | 2026-04-03 20:12 UTC+8 | agent_2w8sht | ORB close=15→10; VWAP放寬測試(均退步); 多日range(退步); prev_bar(退步) | 63.1%/71.4% | +5.83%/+5.72% | -0.116%/-0.015% | 1.674/2.137 | ✅ ORB Sharpe+10% |
| 9 | 2026-04-03 15:10 UTC+8 | agent_fasv07 | VWAP v9: k=1.5,sl_k=0.5; 反轉放寬3模式(仍關閉); ORB pr=4.0測試(退步); 組合size=5重算+OOS | 59.4%/71.4% | +5.82%/+5.72% | -0.12%/-0.015% | 1.52/2.14 | ✅ VWAP Sharpe+4.7% |
| 8 | 2026-04-03 10:10 UTC+8 | agent_o282l3 | size 1→5; VWAP reversal_confirm(失敗); ORB grid orb_bars=[3,4,5] | 59.4%/75.0% | +5.82%/+5.71% | -0.12%/-0.015% | 1.52/2.04 | ✅ 回報×5 |
| 7 | 2026-04-03 04:19 UTC+8 | agent_lssyn4 | VWAP v7: k=1.3,sl_k=0.7; 雙策略組合+OOS驗證; fetcher修復 | 59.4%/75.0% | +5.71%/+5.69% | -0.02%/-0.00% | 1.52/2.04 | ✅ VWAP +14%交易 +3.6pp WR |

---

## 🔄 當前策略狀態快照

**Engine Size**: 5

**ORB 策略 (v8) 最佳參數:**
```
orb_bars: 4
profit_ratio: 3.5
breakout_confirm_pct: 0.0003
trailing_stop: True
trailing_pct: 0.015
close_before_min: 10
vol_confirm: False
atr_trailing: False
multi_day_range: False
htf_filter: True
htf_mode: slope
htf_ema_fast: 20
htf_ema_slow: 30
```
績效 (size=5): Return +0.255%, Win 70.4%, Sharpe 3.315, MaxDD -0.079%, 54 trades, PF 2.36

**VWAP Reversion 策略 (v11) 最佳參數:**
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
prev_bar_reversal: False
partial_tp: True
partial_tp_trail_pct: 0.002
partial_tp_max_hold: 36
```
績效 (size=5): Return +0.080%, Win 69.2% (13 trades), Sharpe 3.629, MaxDD -0.024%, PF 8.14

**雙策略組合 (size=5):**
- 50/50: Sharpe 4.094, Sortino 4.887, MaxDD -0.040%
- 60/40: Sharpe 3.876, Sortino 4.628, MaxDD -0.048%
- OOS Combined 50/50 Sharpe: 6.054

**config.py 關鍵參數:**
```
VERSION: 1.5.0
ITERATION: 11
ATR_PERIOD: 14
STOP_LOSS_ATR_MULT: 2.0
TAKE_PROFIT_RR: 4.0
MIN_SCORE_TO_TRADE: 6.0
STRUCTURE_LOOKBACK: 20
OB_LOOKBACK: 15
```

## ⚠️ 待辦 / 已知問題（下一個 Agent 優先處理）

1. **VWAP OOS Sharpe 下降**：3.868→1.077，考慮更保守的 partial_tp 設定（trail=0.003, max_hold=24）
2. **size=10 風險評估**: 回報翻倍但 MaxDD 可能接近 1%
3. **策略切換機制**: 高 ATR 用 ORB，低 ATR 用 VWAP（自適應配置）
4. **ORB HTF 暖機期**: slope 模式需要前 30 根 1h K 棒（約 4 天）暖機
5. **VWAP 交易數已確認為最優** — 不要再嘗試增加（所有放寬均退步）

## 🚫 已嘗試但失敗的方向（避免重複踩坑）

| 迭代 | 嘗試內容 | 失敗原因 |
|------|---------|---------|
| 10 | VWAP max_trades=3, entry_start=09:35/09:40 | 開盤噪音大，Sharpe 大幅下降 |
| 10 | VWAP k=1.2/1.3+max=3 | 放寬過多導致低品質交易 |
| 10 | ORB multi_day_range=True | 更寬 range 減少交易但品質未提升 |
| 10 | VWAP prev_bar_reversal=True | 只 3 筆交易且虧損 |
| 10 | VWAP rsi_os=40/rsi_ob=60 | Sharpe -0.232（虧損） |
| 11 | ORB htf_filter ema_cross 模式 | 過濾過多交易，Sharpe 僅 1.48-1.56 |

---

## 📝 備註

- 本次使用 NQ=F 5 分鐘線（60 天, ~4,556 根 K 棒）
- 回測引擎使用 manual fallback（無 vectorbt）
- v8 ORB 使用 1h 結構方向過濾（slope 模式），需從 5m 重取樣 1h
- v11 VWAP 使用部分止盈策略，到達 VWAP 後用 trailing stop 追蹤額外利潤
- slope 模式（EMA20 斜率方向）遠優於 ema_cross 模式（EMA20 vs EMA50 交叉）
- 所有 lookback/period 參數已確認為整數型
- OOS 驗證通過 — ORB OOS Sharpe 5.916, Combined OOS Sharpe 6.054
- Buy & Hold 回報為 -5.68%，兩個策略都大幅優於 B&H
