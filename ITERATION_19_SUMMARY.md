# 迭代 19 - VWAP 交易增加優化

## 變更摘要
- **策略**: VWAP Reversion v13 → v19
- **修改**: `max_trades_per_day: 2 → 3`
- **原因**: 增加每日交易機會，當前只有 9 筆交易統計顯著性不足
- **風險**: 可能引入更多虛假信號

## 修改詳情

### 文件變更
1. `src/strategies/vwap_reversion.py`
   - 第 32 行: `max_trades_per_day: 2 → 3`

2. `config.py`
   - VERSION: 2.2.0 → 2.3.0
   - ITERATION: 18 → 19

### 備份
- `backup_v19/vwap_reversion_v19.py` - 修改後的版本
- `backup_v19/orb_v18.py` - ORB 策略（無變更）
- `backup_v19/config_v19.py` - config 檔案（已更新版本號）

## 預期結果
- VWAP 交易數: 9 → 12-14 筆（增加 33-56%）
- Sharpe: 預期 10.484 (允許降幅至 8.387，即 20% 降幅)
- Win Rate: 預期 >50.6%（原 55.6%）
- OOS Combined Sharpe: 預期 >5.633（原 7.041）

## 回滾條件（任一觸發即回滾）
1. VWAP Sharpe 降幅 >20%（< 8.387）
2. VWAP WR 降幅 >5pp（< 50.6%）
3. 任一策略 MaxDD 惡化 >50%
4. OOS Combined Sharpe 降幅 >20%（< 5.633）

## 執行步驟
```bash
# 運行回測（跳過 grid search）
python run_backtest.py --no-grid

# 運行組合分析（驗證組合績效和 OOS 表現）
python run_combined_analysis.py

# 運行 walk-forward 驗證（檢查穩健性評級）
python run_walk_forward.py
```

## 理論分析

### 為什麼選擇 max_trades_per_day？
1. **已排除的選項**：
   - `entry_start_time`: 註釋說 09:35/40 均測試退步（被棄用）
   - `ema_trend_filter`: 太激進（會大幅改變策略邏輯）
   - `bb_width_min`: 缺乏歷史數據（之前沒試過）

2. **保守的增幅**：
   - 從 2→3，增幅 50%（溫和）
   - 不改變進場邏輯，只增加進場機會
   - 符合每日波動率適度的日子可能有多個好機會的直觀認知

### Walk-Forward 發現
- Fold 2 (2026-02-20~03-05): VWAP 0 筆交易（最大風險區間）
- 增加日交易限制可能幫助在這類區間捕捉更多機會

## 時間軸
- 開始: 2026-04-04 04:30 UTC
- 預計完成: ~30 分鐘（需執行完整測試）
- 狀態: 待驗證（需手動運行測試命令）

## 下一步（若成功）
1. 嘗試 max_trades_per_day: 3 → 4
2. 或結合降低 bb_width_min (0.0005 → 0.0003)
3. 或調查 fold2 特定期間的原因並優化

## 聯繫信息
- 迭代 Agent: agent_code_20260404_iter19
- 備份位置: `/sessions/quirky-busy-cannon/mnt/mnq-backtest/backup_v19/`
- 文檔位置: `/sessions/quirky-busy-cannon/mnt/mnq-backtest/AGENT_HANDOFF.md`
