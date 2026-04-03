# CLAUDE_BRIEFING.md — ICT 交易系統狀態摘要

## 基本資訊
- **迭代次數**: 11
- **最後更新**: 2026-04-04 01:10 UTC+8
- **資料**: QQQ 5m, 4556 根 K 棒 (2026-01-06 ~ 2026-04-01)
- **Buy & Hold**: -5.68%
- **Engine Size**: 5

## 最新績效指標

### ORB v8 (htf_filter=slope, EMA20/30, size=5)
| 指標 | 值 | vs v7 |
|------|-----|-------|
| Return | +0.255% | +74% |
| Win Rate | 70.4% | +7.3pp |
| Sharpe | 3.315 | +98% |
| Sortino | 3.896 | +74% |
| Max DD | -0.079% | 改善 |
| Trades | 54 | -11 |
| Profit Factor | 2.356 | +41% |
| Alpha vs B&H | +5.94% | +0.11pp |

**最佳參數**: orb_bars=4, profit_ratio=3.5, breakout_confirm_pct=0.0003, trailing_pct=1.5%, close_before_min=10, htf_filter=True, htf_mode=slope, htf_ema_fast=20, htf_ema_slow=30

### VWAP v11 (partial_tp, trail=0.002, max_hold=36, size=5)
| 指標 | 值 | vs v10 |
|------|-----|--------|
| Return | +0.080% | +127% |
| Win Rate | 69.2% | -2.2pp |
| Sharpe | 3.629 | +70% |
| Max DD | -0.024% | 略增 |
| Trades | 13 | -1 |
| Profit Factor | 8.138 | +109% |
| Alpha vs B&H | +5.76% | +0.04pp |

**最佳參數**: k=1.5, sl_k_add=0.5, std_window=30, rsi_os=35, rsi_ob=65, ema_mode=ema_cross, entry_start_time=09:45, partial_tp=True, partial_tp_trail_pct=0.002, partial_tp_max_hold=36

## 雙策略組合分析 (size=5)
| 配置 | Return | Sharpe | Sortino | MaxDD |
|------|--------|--------|---------|-------|
| ORB 50%/VWAP 50% | +0.167% | 4.094 | 4.887 | -0.040% |
| ORB 60%/VWAP 40% | +0.185% | 3.876 | 4.628 | -0.048% |
| ORB 70%/VWAP 30% | +0.202% | 3.694 | 4.397 | -0.056% |

**結論**: 50/50 仍為最佳風險調整配置。組合 Sharpe 從 2.033 大幅提升至 4.094 (+101%)。

## Out-of-Sample 驗證 (size=5)
| 策略 | IS Sharpe | OOS Sharpe | IS WR | OOS WR |
|------|-----------|------------|-------|--------|
| ORB | 1.409 | **5.916** | 67.6% | 68.4% |
| VWAP | 3.868 | **1.077** | 71.4% | 57.1% |
| Combined 50/50 | — | **6.054** | — | — |

**結論**: OOS Sharpe 大幅優於 IS（ORB），無過擬合跡象。VWAP OOS 下降但仍為正 Sharpe。

## 本次修改記錄 (迭代 #11)
1. **ORB v8: 1h 結構方向過濾（htf_filter=slope, EMA20/30）**:
   - ICT 多時間框架概念：用 1h EMA20 斜率判斷結構方向
   - 只在結構方向交易（斜率>0=看多, <0=看空）
   - Sharpe +98% (1.674→3.315), Win Rate +7.3pp
   - slope 模式遠優於 ema_cross 模式
   - ema_cross 模式過濾過多交易（只 37 筆），Sharpe 僅 1.5
2. **VWAP v11: 部分止盈策略（partial_tp=True）**:
   - 到達 VWAP 後不立即平倉，改用 trailing stop 追蹤
   - trail=0.002 (0.2%), max_hold=36 bars (3小時) 為最佳
   - Sharpe +70% (2.137→3.629), Return +127%
   - Profit Factor 翻倍 (3.89→8.14)
3. **雙策略組合 Sharpe 突破 4.0**: 50/50 = 4.094（歷史最佳）

## 已知問題與下次方向
1. VWAP OOS Sharpe 下降（3.868→1.077），可能需要更保守的 partial_tp 參數
2. size=10 風險評估待做（MaxDD 可能接近 1%）
3. 策略切換機制（高 ATR 用 ORB，低 ATR 用 VWAP）— 自適應配置
4. ORB htf_filter slope 模式需要足夠的 1h 數據暖機（前幾天可能錯過信號）
5. 考慮 VWAP partial_tp 的更保守設定（trail=0.003, max_hold=24）以改善 OOS
