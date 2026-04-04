# TradingView Pine 使用說明

## 最新腳本

- `ORB_v18_Strategy.pine`
- `VWAP_Reversion_v20_Strategy.pine`

舊版 `ORB_v6_Strategy.pine`、`VWAP_Reversion_v9_Strategy.pine` 先保留在資料夾內做對照，不建議再拿來測最新結果。

## 建議測試方式

1. 在 TradingView 開 `QQQ` 的 `5m` 圖。
2. 圖表請使用美股 regular session。
3. 打開 Pine Editor，把對應 `.pine` 檔內容貼上。
4. `Add to chart` 後打開 Strategy Tester。

## 與 Python 對齊的核心設定

- 初始資金：`100000`
- 部位大小：`10`
- 手續費：`0.05%`
- 以 bar close 處理訂單：`process_orders_on_close=true`

## 目前對應版本

### ORB

- 對應 Python：`ORB v18`
- 重要特徵：
  - `orb_bars=4`
  - `profit_ratio=3.5`
  - `breakout_confirm_pct=0.0003`
  - `trailing_pct=0.015`
  - `1h slope filter`
  - `skip_short_after_up_days=2`
  - `skip_long_after_up_days=3`

### VWAP Reversion

- 對應 Python：`VWAP v20`
- 重要特徵：
  - `k=1.5`
  - `sl_k_add=0.5`
  - `std_window=30`
  - `rsi_os=32`
  - `rsi_ob=66`
  - `ema_mode=ema_cross`
  - `entry_start_time=09:45`
  - `partial_tp_trail_pct=0.002`
  - `partial_tp_max_hold=32`

## 重要限制

- Python 版本的 `50/50` 與 `ATR adaptive` 組合，是把兩條獨立 equity curve 做權重合成。
- TradingView 的單一 `strategy()` 很難 1:1 重現這種「兩個獨立策略再做組合權重」的回測方式。
- 所以這裡提供的是：
  - ORB 單獨測試腳本
  - VWAP 單獨測試腳本

如果你要驗證「組合結果」，建議：

1. 先在 TradingView 分別測 ORB 與 VWAP。
2. 再把兩邊的報表結果和 Python 的 `combined_analysis.json` 對照。

## 實務提醒

- Pine 與 Python 即使邏輯一致，成交細節仍可能有小差異。
- 差異來源通常是：
  - TradingView 的撮合規則
  - session 定義
  - bar close / intrabar fill 行為
  - request.security 的高週期同步方式

如果你要，我下一步可以直接幫你再做一個：

- `ICT_Combined_Helper.pine`

用來在圖上同時顯示 ORB / VWAP 訊號與 regime 狀態，方便你在 TradingView 人工比對。 
