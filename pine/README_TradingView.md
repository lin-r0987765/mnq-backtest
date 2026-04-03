# TradingView Pine Script — ICT 整合交易系統

## 檔案說明

| 檔案 | 策略 | 說明 |
|------|------|------|
| `ORB_v6_Strategy.pine` | Opening Range Breakout v6 | 開盤區間突破策略 |
| `VWAP_Reversion_v9_Strategy.pine` | VWAP Mean-Reversion v9 | VWAP 均值回歸策略 |

## 使用方式

1. 打開 TradingView → Pine Editor
2. 複製 `.pine` 檔案內容貼入編輯器
3. 點擊「Add to Chart」
4. 設定圖表為 **QQQ 5 分鐘線**（或 NQ 期貨）
5. 在策略測試器中查看回測結果

## 最佳參數（迭代 #9）

### ORB v6
- 時間框架：5 分鐘
- ORB K棒數：4（= 20 分鐘 opening range）
- 止盈倍數：3.5 × Range Width
- 移動止損：1.5%
- 突破確認：0.03%

### VWAP Reversion v9
- 時間框架：5 分鐘
- Band 倍數 (k)：1.5
- SL 額外倍數：0.5
- Std 窗口：30
- RSI 超賣/超買：35 / 65
- EMA 趨勢過濾：EMA20 vs EMA50 交叉
- 每日最大交易：2 筆
- 入場時間：09:45 ~ 15:30

## 回測績效（Python 回測，60 天 QQQ 5m 數據）

| 指標 | ORB v6 | VWAP v9 |
|------|--------|---------|
| Return | +0.133% | +0.035% |
| Sharpe | 1.523 | 2.137 |
| Win Rate | 59.4% | 71.4% |
| Max DD | -0.119% | -0.015% |
| Trades | 64 | 14 |
| Alpha vs B&H | +5.82% | +5.72% |

## 注意事項

- Pine Script 的 VWAP 使用 session-anchored 計算，與 Python 版一致
- TradingView 回測結果可能因滑價、手續費設定略有差異
- 建議先用 Paper Trading 驗證再實盤
- Engine Size 設為 5（每次交易 5 單位）
