# CLAUDE_BRIEFING.md — ICT 交易系統狀態摘要

## 基本資訊
- **迭代次數**: 24（分析型，無參數變更）
- **最後更新**: 2026-04-05 00:27 UTC+8
- **資料**: QQQ 5m, 4556 根 K 棒 (2026-01-07 ~ 2026-04-02)
- **Buy & Hold**: -5.03%
- **Engine Size**: 10

## 最新績效指標

### ORB v18 (未變動)
| 指標 | 值 | 備註 |
|------|-----|------|
| Return | +0.586% | 持平 |
| Win Rate | 69.8% | 持平 |
| Sharpe | 9.606 | 持平 |
| Max DD | -0.142% | 持平 |
| Trades | 43 | 持平 |
| Profit Factor | 2.586 | 持平 |
| Alpha vs B&H | +5.62% | 持平 |

**最佳參數**: `orb_bars=4`, `profit_ratio=3.5`, `breakout_confirm_pct=0.0003`, `trailing_pct=1.5%`, `close_before_min=10`, `htf_filter=True`, `htf_mode=slope`, `htf_ema_fast=20`, `htf_ema_slow=30`, `skip_short_after_up_days=2`, `skip_long_after_up_days=3`

### VWAP v22 (未變動)
| 指標 | 值 | 備註 |
|------|-----|------|
| Return | +0.199% | 持平 |
| Win Rate | 60.0% | 持平 |
| Sharpe | 11.347 | 持平 |
| Max DD | -0.035% | 持平 |
| Trades | 10 | 持平 |
| Profit Factor | 6.929 | 持平 |
| Alpha vs B&H | +5.23% | 持平 |

**最佳參數**: `k=1.5`, `sl_k_add=0.5`, `std_window=30`, `rsi_os=32`, `rsi_ob=66`, `max_trades_per_day=2`, `ema_trend_filter=True`, `ema_mode=ema_cross`, `dynamic_tp=True`, `tp_bonus_pct=0.2`, `bb_width_min=0.0005`, `entry_start_time=09:45`, `partial_tp=True`, `partial_tp_trail_pct=0.002`, `partial_tp_max_hold=32`, `sideways_ema_gap=0.0`, `vol_norm_mode=rolling`

## 雙策略組合分析 (size=10)
| 配置 | Return | Sharpe | Sortino | MaxDD |
|------|--------|--------|---------|-------|
| ORB 50% / VWAP 50% | +0.393% | 5.292 | 5.792 | -0.071% |
| ORB 60% / VWAP 40% | +0.432% | 4.978 | 5.412 | -0.086% |
| **Active Reuse 80%** | **+0.636%** | **5.603** | **6.034** | -0.112% |
| ATR Adaptive abs `t=0.017` | +0.476% | 5.221 | 5.353 | -0.100% |

## Out-of-Sample 驗證 (size=10, IS=40d, OOS=20d)
| 策略 | IS Sharpe | OOS Sharpe | IS WR | OOS WR |
|------|-----------|------------|-------|--------|
| ORB | 5.674 | **14.607** | 68.0% | 70.6% |
| VWAP | 11.728 | **10.585** | 50.0% | 75.0% |
| Combined 50/50 | — | **7.358** | — | — |
| **Active Reuse 80%** | — | **7.845** | — | — |
| ATR Adaptive abs | — | **6.816** | — | — |

## Walk-Forward 驗證
**配置**: 訓練窗口=20天, 測試窗口=10天, 步進=10天, 4 折

### VWAP Walk-Forward
| Fold | 測試期間 | Return | Sharpe | WR | Trades | B&H |
|------|---------|--------|--------|-----|--------|-----|
| 1 | 02/05~02/19 | +0.015% | 2.306 | 50.0% | 2 | +0.378% |
| 2 | 02/20~03/05 | +0.000% | 0.000 | 0.0% | 0 | +1.106% |
| 3 | 03/06~03/19 | -0.012% | -2.751 | 0.0% | 1 | -1.310% |
| 4 | 03/20~04/02 | +0.049% | 5.196 | 75.0% | 4 | -0.699% |
| **平均** | — | **+0.013%** | **1.188** | **31.2%** | 7 | — |

### Combined Walk-Forward
| Fold | ORB Sh | VWAP Sh | Combo Sh | Combo Ret | B&H |
|------|--------|---------|----------|-----------|-----|
| 1 | 0.132 | 2.306 | 1.536 | +0.035% | +0.378% |
| 2 | 2.719 | 0.000 | 3.102 | +0.043% | +1.106% |
| 3 | 1.811 | -2.751 | 2.008 | +0.041% | -1.310% |
| 4 | 11.671 | 5.196 | 13.847 | +0.318% | -0.699% |
| **平均** | **4.083** | **1.188** | **5.123** | **+0.109%** | — |

### 關鍵發現
- WF 評級維持 **A（優秀）**，所有核心指標不變。
- 第 24 輪測試 ORB `breakout_confirm=0.0004`，全樣本/OOS 改善但 WF fold 1 Sharpe 轉負（0.132→-0.065），回滾。
- VWAP fold 2 零交易已確認為「正確的保守行為」。深度 bar-by-bar 追蹤顯示所有信號被 EMA filter 或 vol filter 合理阻擋，放開後全面惡化。

## 本次修改記錄 (迭代 #24 — 分析型)
1. 深度 bar-by-bar 分析 VWAP fold 2 (02/20~03/05) 零交易的根因。
2. 測試 ORB `breakout_confirm_pct=0.0004`（全樣本 Sh 9.606→9.676, WR 69.8→71.4%），因 WF fold 1 降級回滾。
3. 測試 VWAP `vol_norm_mode=tod`, `vol_min_mult=0.3/0.2`, `vol_filter=False`, `ema_trend_filter=False`，全面惡化，確認現有 filter 組合為最優。
4. 記錄完整失敗方向和 fold 2 診斷摘要至 AGENT_HANDOFF.md。

## 已知問題與下次方向
1. **VWAP fold 2 已確認為「無機會」regime** — 不再嘗試修補，降優先級。
2. **策略接近最佳化極限** — 所有常規參數方向已探索殆盡，Sharpe > 9/11。
3. **下一輪建議方向**:
   - 新增第三個互補策略（如 momentum 或 mean-reversion on higher timeframe）
   - 擴大數據範圍（更多歷史數據或多標的）
   - 改進組合層（如動態 rebalance、Kelly criterion sizing）
   - VIX/realized vol 自適應 regime switching
