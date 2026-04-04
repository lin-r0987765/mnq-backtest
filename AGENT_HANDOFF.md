# AGENT_HANDOFF.md — Agent 協作交接文檔

> 此文檔由自動優化代理維護，用於多 Agent 協作時的狀態同步。
> 每次迭代開始前必須讀取，結束後必須更新。

---

## 鎖定狀態

| 欄位 | 值 |
|------|-----|
| 狀態 | `IDLE` |
| 佔用者 | — |
| 佔用時間 | — |
| 預計完成 | — |

> 狀態值：`IDLE`（空閒可接手）/ `RUNNING`（執行中勿動）/ `FAILED`（上次失敗需檢查）

---

## 最近 5 次迭代摘要（最新在上）

| # | 時間 (UTC+8) | Agent | 修改摘要 | 勝率 | Alpha | 回撤 | Sharpe | 結果 |
|---|-------------|-------|---------|------|-------|------|--------|------|
| 24 | 2026-04-05 00:27 UTC+8 | antigravity_opus_20260404 | 深度 fold 2 根因分析；ORB `breakout_confirm=0.0004` 測試但因 WF 降級回滾；VWAP 所有 filter 放寬方向全面惡化 | 69.8%/60.0% | +5.62%/+5.23% | -0.142%/-0.035% | 9.606/11.347 | ⚠️ 分析型迭代，無參數變更 |
| 23 | 2026-04-04 16:22 UTC+8 | codex_auto_20260404_f | 抽出共用 `portfolio_overlay.py`；統一 active reuse 在 combined / walk-forward 的邏輯 | 69.8%/60.0% | +5.62%/+5.23% | -0.142%/-0.035% | 9.606/11.347 | ✅ 零回歸重構完成 |
| 22 | 2026-04-04 16:00 UTC+8 | codex_auto_20260404_e | VWAP fold 2 分析；新增 `sideways_ema_gap` + `vol_norm_mode` hooks | 69.8%/60.0% | +5.62%/+5.23% | -0.142%/-0.035% | 9.606/11.347 | ✅ hooks 就位，待後續測試 |
| 21 | 2026-04-04 12:16 UTC+8 | codex_auto_20260404_d | 組合層新增 `Active Reuse 80%`；`run_combined_analysis.py` / `run_walk_forward.py` 同步套用 | 69.8%/60.0% | +5.62%/+5.23% | -0.142%/-0.035% | 9.606/11.347 | ✅ 組合全樣本 / OOS / WF 同步改善 |
| 20 | 2026-04-04 11:50 UTC+8 | codex_auto_20260404_c | VWAP `rsi_ob 68→66`；修復新版 yfinance CSV / daily ATR 讀取 | 69.8%/60.0% | +5.62%/+5.23% | -0.142%/-0.035% | 9.605/11.346 | ✅ VWAP OOS 與組合同步改善 |

---

## 當前策略狀態快照

**Engine Size**: 10

**ORB 策略 (v18)**
```text
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
skip_short_after_up_days: 2
skip_long_after_up_days: 3
```
績效 (size=10): Return `+0.586%`, Win `69.8%`, Sharpe `9.606`, MaxDD `-0.142%`, Trades `43`, PF `2.586`

**VWAP Reversion 策略 (v22)**
```text
k: 1.5
sl_k_add: 0.5
std_window: 30
rsi_os: 32
rsi_ob: 66
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
partial_tp_max_hold: 32
sideways_ema_gap: 0.0
vol_norm_mode: rolling
```
績效 (size=10): Return `+0.199%`, Win `60.0%`, Sharpe `11.347`, MaxDD `-0.035%`, Trades `10`, PF `6.929`

**雙策略組合 (size=10)**
- 50/50: Sharpe `5.292`, Sortino `5.792`, MaxDD `-0.071%`
- Active Reuse 80%: Sharpe `5.603`, Sortino `6.034`, MaxDD `-0.112%`
- ATR Adaptive abs `t=0.017`: Sharpe `5.221`, Sortino `5.353`, MaxDD `-0.100%`
- OOS Combined 50/50 Sharpe: `7.358`
- OOS Active Reuse 80% Sharpe: `7.845`
- OOS ATR Adaptive Sharpe: `6.816`

**Walk-Forward 驗證**
- ORB: 平均 Sharpe `4.083` (±`4.478`)，正 Sharpe 比率 `100%`
- VWAP: 平均 Sharpe `1.188` (±`2.926`)，正 Sharpe 比率 `50%`
- Combined Active Reuse 80%: 平均 Sharpe `5.123` (±`5.069`)，正 Sharpe 比率 `100%`
- **穩健性評級: A（優秀）**

**config.py 關鍵參數**
```text
VERSION: 2.6.0
ITERATION: 23
ATR_PERIOD: 14
STOP_LOSS_ATR_MULT: 2.0
TAKE_PROFIT_RR: 4.0
MIN_SCORE_TO_TRADE: 6.0
STRUCTURE_LOOKBACK: 20
OB_LOOKBACK: 15
```

## 待辦 / 已知問題（下一個 Agent 優先處理）

1. **VWAP fold 2 已確認為「無機會」期間（降優先級）**: 深度 bar-by-bar 分析確認，02/20~03/05 的所有 VWAP 信號被 EMA trend filter 或 vol filter 阻擋，且放開這些 filter 全面惡化品質（WR 10-15%，Sharpe 強負）。這是正確的保守行為，不應強行修補。
2. **VWAP fold 3 的單筆虧損 long**: 2026-03-10 14:15 長單仍是弱點，但 `entry_end_time=14:00` 會使整個 fold 3 變零交易，不可取。
3. **ORB `breakout_confirm=0.0004` 邊界情況**: 全樣本和 OOS 都改善，但 WF fold 1 Sharpe 從 +0.132 → -0.065，使 WF 降級到 B。如果未來嘗試，需搭配其他改進讓 fold 1 更穩定。
4. **策略接近最佳化極限**: 所有常規參數調整（RSI/vol/EMA/k/std_window）都已測試到邊界，未來應考慮：
   - 新增完全不同的 regime filter（如 VIX/realized vol 自適應）
   - 擴大數據範圍（更多歷史數據，或切換到更多標的如 SPY）
   - 新增第三個互補策略（如 momentum/trend following）
5. **Active reuse 已抽成共用模組**: [src/portfolio_overlay.py](C:\Users\LIN\Desktop\progamming\python\claude\mnq-backtest\src\portfolio_overlay.py)

## 已嘗試但失敗的方向（不要重試相同組合）

| 迭代 | 嘗試方向 | 原因 |
|------|---------|------|
| 24 | ORB `breakout_confirm_pct=0.0004` | 全樣本 Sh+0.7%/WR+1.6pp，但 WF fold 1 Sharpe 轉負（0.132→-0.065），WF 評級 A→B |
| 24 | VWAP `vol_norm_mode=tod` | 全樣本 Sh 11.3→-2.0，WR 60→39%，全面崩壞 |
| 24 | VWAP `vol_min_mult=0.3/0.2` | 全樣本 Sh→-2.4/-3.5，WR→36%，嚴重惡化 |
| 24 | VWAP `ema_trend_filter=False` + 各種 RSI 收緊 | 即使 rsi_os=25/rsi_ob=75，全樣本 Sh→-6~-9，60 筆 WR 27% |
| 24 | VWAP `vol_filter=False` | 等同 vol_min_mult=0.2，全面崩壞 |
| 23 | VWAP `sideways_ema_gap=0.0015~0.0025` + `vol_norm_mode=rolling/tod` | 全樣本 / OOS / fold 2 / fold 3 Sharpe 全面崩壞 |
| 23 | VWAP `reversal_confirm=True` 各種 `reversal_mode / prev_bar_reversal` | 交易數大幅塌陷，OOS 直接變 0 trades |
| 23 | VWAP `entry_end_time=14:00` | 雖然全樣本 Sharpe 上升，但交易數掉到 4，OOS Sharpe 低於基準，統計性不足 |
| 20 | VWAP `rsi_os=33/34` | 交易數增加，但 WR 與 OOS Sharpe 明顯下降 |
| 20 | VWAP `no_vol_filter` | 全樣本 Sharpe 崩到負值 |
| 20 | VWAP `ema_trend_filter=False` | 過度交易且全樣本 / OOS 都明顯惡化 |
| 20 | VWAP `partial_tp_max_hold=36` | Sharpe 與 OOS 一起退步 |
| 20 | VWAP `ema_mode=price_vs_ema` | 幾乎無交易 |
| 18 | `skip_short_after_down_days=3` | ORB / OOS 顯著惡化 |
| 18 | `skip_long_after_up_days=2` | fold 改善，但 OOS 降幅過大 |

## 第 24 輪 Fold 2 深度分析摘要（供後續 agent 參考）

**阻擋原因分佈**（10 個交易日，共 ~100 個潛在信號）:
- `ema_uptrend` 阻擋 SHORT: 02/20, 02/24, 02/25, 03/04 → EMA20>EMA50 上升趨勢中禁做空
- `ema_downtrend` 阻擋 LONG: 02/23, 02/26, 03/03, 03/05 → 下跌日禁做多
- `vol` 過濾: 02/23~02/26 → rolling 20-bar vol MA 被開盤高量拉高，午盤量不足
- `atr_low`: 02/25 下午 → 低波動期，ATR/price < 0.0005
- `time`: 幾個 09:40 信號被 entry_start=09:45 阻擋

**結論**: EMA filter 是 VWAP 成功的核心（關閉後全樣本 WR 降到 10-15%），fold 2 歸根結底是上升趨勢+橫盤交替的 regime，不適合均值回歸。

## 備註

- 本輪為分析型迭代，驗證了 fold 2 零交易是策略正確的保守行為
- 測試了 ORB breakout_confirm 微調，因 WF 降級回滾
- 數據已更新至 2026-04-02
- 驗證已完成：`run_backtest.py`、`run_combined_analysis.py`、`run_walk_forward.py`
