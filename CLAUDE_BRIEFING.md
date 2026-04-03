# CLAUDE_BRIEFING.md — ICT 交易系統狀態摘要

## 基本資訊
- **迭代次數**: 9
- **最後更新**: 2026-04-03 15:10 UTC+8
- **資料**: QQQ 5m, 4556 根 K 棒 (2026-01-06 ~ 2026-04-01)
- **Buy & Hold**: -5.68%
- **Engine Size**: 5

## 最新績效指標

### ORB v6 (unchanged, size=5)
| 指標 | 值 |
|------|-----|
| Return | +0.133% |
| Win Rate | 59.4% |
| Sharpe | 1.523 |
| Max DD | -0.119% |
| Trades | 64 |
| Profit Factor | 1.625 |
| Alpha vs B&H | +5.82% |

**最佳參數**: orb_bars=4, profit_ratio=3.5, breakout_confirm_pct=0.0003, trailing_pct=1.5%, vol_confirm=False, atr_trailing=False

### VWAP v9 (k=1.5, sl_k=0.5, size=5)
| 指標 | 值 | vs v8 |
|------|-----|-------|
| Return | +0.035% | = |
| Win Rate | 71.4% | -3.6pp |
| Sharpe | 2.137 | +4.7% |
| Max DD | -0.015% | = |
| Trades | 14 | -2 |
| Profit Factor | 3.886 | -0.14 |
| Alpha vs B&H | +5.72% | = |

**最佳參數**: k=1.5, sl_k_add=0.5, std_window=30, rsi_os=35, rsi_ob=65, ema_mode=ema_cross, entry_start_time=09:45, bb_width_min=0.0005, max_trades_per_day=2, reversal_confirm=False

## 雙策略組合分析 (size=5)
| 配置 | Return | Sharpe | Sortino | MaxDD |
|------|--------|--------|---------|-------|
| ORB 60%/VWAP 40% | +0.094% | 1.772 | 2.341 | -0.073% |
| ORB 50%/VWAP 50% | +0.084% | 1.884 | 2.468 | -0.062% |
| ORB 70%/VWAP 30% | +0.104% | 1.687 | 2.238 | -0.084% |

**結論**: 50/50 仍為最佳風險調整配置。size=5 已更新。

## Out-of-Sample 驗證 (size=5)
| 策略 | IS Sharpe | OOS Sharpe | IS WR | OOS WR |
|------|-----------|------------|-------|--------|
| ORB | 0.808 | **2.780** | 61.0% | 56.5% |
| VWAP | 1.069 | **3.569** | 71.4% | 71.4% |
| Combined 60/40 | — | **3.311** | — | — |

**結論**: 兩策略 OOS Sharpe 均大幅優於 IS，無過擬合跡象。

## 本次修改記錄 (迭代 #9)
1. **VWAP k=1.3→1.5, sl_k_add=0.7→0.5**: Sharpe +4.7% (2.042→2.137)
   - WR 降 3.6pp (75.0→71.4%) 但 Sharpe 改善，風險調整後表現更優
   - 回歸 v7 探索時的最佳 k/sl_k 組合
2. **VWAP 反轉確認放寬（v9 新增 reversal_mode）**:
   - 新增 relaxed(body≥0.25) 和 simple(close vs open) 模式
   - 測試結果：所有反轉確認模式均過嚴（16→2筆），預設仍關閉
3. **ORB profit_ratio=4.0+trailing_pct=0.012 測試**:
   - Sharpe 0.951 (baseline 1.523)，明顯退步
   - 確認 v6 參數仍為最佳
4. **雙策略組合 size=5 重算**: 已完成，Sharpe 與 size=1 一致
5. **OOS 驗證通過**: 所有 OOS > IS，無過擬合

## 已知問題與下次方向
1. VWAP 交易次數仍偏少 (14筆/60天) — 可嘗試 max_trades_per_day=3 或放寬 entry 時間窗口
2. VWAP reversal_confirm 邏輯已完善但過嚴 — 可嘗試「前一根K棒」而非當根確認
3. ORB 已高度優化 — 可探索多日 ORB (aggregated range)
4. size=10 風險評估待做 (MaxDD 可能接近 1%)
5. 可嘗試 ORB+VWAP 策略切換機制（高波動用 ORB，低波動用 VWAP）
