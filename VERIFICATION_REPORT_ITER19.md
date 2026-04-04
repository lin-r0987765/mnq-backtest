# ICT 整合交易系統 - 迭代 19 驗證報告

**日期**: 2026-04-04
**迭代**: v19 (VWAP max_trades_per_day 修改)
**狀態**: ⚠️ 待驗證 - 修改確認但性能測試異常

---

## 一、修改確認

### 1.1 參數修改
- **文件**: `src/strategies/vwap_reversion.py`
- **修改**: Line 32, `max_trades_per_day: 2 → 3`
- **目的**: 增加每日交易機會，目標從 9 筆增至 12-14 筆

```python
# 修改前 (v18)
"max_trades_per_day": 2,

# 修改後 (v19)
"max_trades_per_day": 3,       # v19 從 2 增至 3 增加交易機會
```

### 1.2 配置更新
- **文件**: `config.py`
- **修改**:
  - VERSION: `2.2.0 → 2.3.0`
  - ITERATION: `18 → 19`
- **狀態**: ✅ 已確認

---

## 二、Bug 修復

### 2.1 數據類型轉換問題
- **問題**: CSV 載入後的 OHLCV 欄位為字符串型態 (str)，導致算術運算失敗
- **症狀**: `TypeError: unsupported operand type(s) for -: 'str' and 'str'`
- **位置**: `src/data/fetcher.py`, `_clean_df()` 函數
- **修復**: 添加 `pd.to_numeric()` 轉換

```python
# 修復前
# 無轉換邏輯

# 修復後
for col in ["Open", "High", "Low", "Close"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "Volume" in df.columns:
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
```

- **狀態**: ✅ 已確認修復

---

## 三、測試執行

### 3.1 測試腳本
- **腳本**: `test_iter19.py` (新建)
- **用途**: 快速驗證 VWAP 策略在 v19 參數下的性能
- **數據**: QQQ 5分鐘線 4557 根 (2026-01-07 ~ 2026-04-02)

### 3.2 測試結果

| 指標 | v18 基準 | v19 測試結果 | 狀態 |
|------|---------|-----------|------|
| Sharpe | 10.484 | 4.534 | ❌ 下跌 56.7% |
| Win Rate | 55.6% | 66.7% | ✅ 改善 |
| Max DD | -0.035% | -0.04% | ✅ 在容差內 |
| Trades | 9 | 9 | ⚠️ 無增加 |
| Return | +0.180% | +0.18% | ✅ 相同 |
| PF | 6.345 | 10.283 | ✅ 改善 |

### 3.3 異常分析

**重大發現**: 交易數仍為 9 筆，未見預期的 12-14 筆

**可能原因**:
1. 策略信號生成本身受限於多個條件（RSI、BB、EMA、時間窗口等），不單由 max_trades_per_day 控制
2. 在實際數據中，同一日期符合全部進場條件的機會確實只有 9 次
3. max_trades_per_day=3 的限制實際上未被觸發（因信號本身就只有 9 個）

**Sharpe 值異常**:
- Sharpe 從 10.484 掉至 4.534，降幅 56.7%，遠超過 20% 容差
- 但 Return (+0.18%) 與基準完全相同
- WR (66.7%) 高於基準 (55.6%)
- 暗示：可能是測試腳本使用了不同的 dataset split 或 backtest engine 配置

---

## 四、驗證狀態

| 項目 | 狀態 | 備註 |
|------|------|------|
| 代碼修改確認 | ✅ 完成 | max_trades_per_day 正確設置為 3 |
| 數據類型修復 | ✅ 完成 | OHLCV 列已轉換為 float64 |
| 性能測試 | ⚠️ 異常 | Sharpe 掉幅過大，需重新驗證 |
| 組合分析 | ⏸️ 待定 | 等待性能測試澄清 |
| Walk-Forward | ⏸️ 待定 | 等待性能測試澄清 |
| 回滾判斷 | ⏸️ 待定 | 取決於完整測試結果 |

---

## 五、下一步行動清單 (CRITICAL)

### 優先級 P0 - 立即執行
```bash
# 完整驗證 (使用官方回測框架)
python3 run_backtest.py --no-grid 2>&1 | tee iter19_official_backtest.log

# 對比結果
# 若 VWAP Sharpe >= 8.387 (v18 10.484 的 80%)
#   → 繼續組合分析和 walk-forward 驗證
# 若 VWAP Sharpe < 8.387 OR WR < 50.6% OR 其他條件不符
#   → 執行回滾到 v18
```

### 優先級 P1 - 條件完成後
```bash
# 若第一步通過
python3 run_combined_analysis.py 2>&1 | tee iter19_combined.log
python3 run_walk_forward.py 2>&1 | tee iter19_walkforward.log
```

### 優先級 P2 - 最終確認
```bash
# 文件備份
mkdir -p backup_v19
cp src/strategies/vwap_reversion.py backup_v19/
cp src/data/fetcher.py backup_v19/
cp config.py backup_v19/

# 日誌備份
cp iter19_*.log iteration_logs/
```

---

## 六、回滾條件（任一觸發即回滾）

1. **VWAP Sharpe < 8.387** (v18 基準 10.484 的 80%)
2. **VWAP Win Rate < 50.6%** (v18 基準 55.6% 減 5pp)
3. **VWAP Max DD 惡化超過 50%** (-0.035% * 1.5 = -0.0525%)
4. **OOS 組合 Sharpe < 5.633** (v18 OOS 50/50 7.041 的 80%)

---

## 七、環境注記

**問題**:
- Bash 環境存在持續不穩定性 (多數命令執行失敗)
- 導致無法直接執行官方回測框架進行驗證

**現有數據**:
- test_iter19.py 輸出已保存: `test_iter19_output.log`
- 數據類型修復已確認: float64 轉換成功
- 代碼修改已確認: max_trades_per_day=3 正確設置

**下一 Agent 建議**:
- 優先使用官方回測框架 `run_backtest.py --no-grid`
- 若 bash 環境仍有問題，考慮使用替代方案或環境重置
- 對比 iteration_18_backtest.log 的輸出格式確保結果可比較性

---

## 附錄：性能閾值表

| 指標 | v18 基準 | 容差 | v19 閾值 | v19 測試值 | 是否通過 |
|------|---------|------|---------|-----------|---------|
| Sharpe | 10.484 | -20% | 8.387 | 4.534 | ❌ |
| WR | 55.6% | -5pp | 50.6% | 66.7% | ✅ |
| MaxDD | -0.035% | ±50% | -0.0525% | -0.04% | ✅ |
| OOS Combo | 7.041 | -20% | 5.633 | TBD | ⏳ |

