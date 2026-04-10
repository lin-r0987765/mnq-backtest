# QQQ ICT 多時間框架重構計畫

## 摘要

這次改造採用 `1D / 4H / 1H / 15m / 5m / 1m` 的自上而下架構，目標是把目前「5m 單框架、低頻、低延展性」的 ICT 研究線，改成一條 `獲利優先` 的多時間框架策略。

本計畫的核心決策如下：

- `1m` 作為唯一執行基準時間框架
- `1D + 4H` 決定主方向
- `1H` 作為方向狀態過濾
- `15m` 負責建立 ICT setup 與區間
- `5m` 負責回踩與確認
- `1m` 負責實際進場觸發
- 第一版只做 `順大方向的 continuation / pullback lane`
- 原本的 `5m reversal lane` 保留為 benchmark，不直接混進第一版主線

這樣做的理由是：你現在最缺的不是再把 5m reversal 調得更花，而是先建立一條「方向更乾淨、進場更細、樣本數更可擴張」的主幹。

## 實作步驟

### Step 0：資料層先統一成 1m canonical source

- 將你已補上的 `QQQ_1Min` 與 `SPY_1Min` parquet 正規化為研究主線可直接使用的檔案，目標輸出到 `alpaca/normalized`，命名與現有 `qqq_5m_alpaca.csv` 風格一致。
- 更新研究 manifest，讓 `1m` 和現有 `1D/4H/1H/15m/5m` 一起成為正式資料來源。
- 執行期只吃 `1m` QQQ 與 `1m` SPY，所有較大時間框架都從 `1m` 內部重採樣產生，不在 runtime 混用多份不同粒度 CSV。
- 保留現有 `1D/4H/1H/15m/5m` normalized CSV，僅作為對時與驗證基準。

完成標準：
- `1m` QQQ / SPY 可穩定讀取
- 用 `1m` 重採樣出的 `5m` 與現有 `qqq_5m_alpaca.csv`、`spy_5m_alpaca.csv` 在時間戳與 OHLC 上誤差可控
- 沒有 look-ahead 對時問題

### Step 1：建立 MTF feature layer，不先動交易邏輯

- 在資料或策略內新增 MTF aggregation helper，固定由 `1m` 產出 `5m / 15m / 1H / 4H / 1D`。
- 所有 higher timeframe 指標都必須以 `closed="right"`、只在該 bar 收線後才對後續 `1m` bar 生效。
- QQQ 與 SPY 的 higher timeframe 也要用同一套 resample 規則，避免 SMT 在不同粒度上出現時間偏移。
- 先只產出這些 feature，不先下交易規則：
- `1D` 結構方向
- `4H` 結構方向
- `1H` 結構狀態與流動性位置
- `15m` sweep / MSS / FVG zone
- `5m` retest / rejection / micro confirmation
- `1m` micro MSS / rejection candle / entry trigger

完成標準：
- 可以在單一 `1m` DataFrame 上拿到所有時間框架的對齊欄位
- 單元測試能證明 higher timeframe 資訊不提前泄漏

### Step 2：先做一條 profit-first 的 top-down 主線

第一版主線固定採用以下判定，不做模糊化：

- `1D + 4H` 為主方向層
- 如果 `1D` 與 `4H` 同向，只允許做該方向
- 如果 `1D` 與 `4H` 互相衝突，當天該段直接不交易
- 如果 `1D` 中性、`4H` 有方向，只有在 `1H` 同方向時才允許該方向
- 如果 `4H` 中性，第一版不交易

`1H` 規則固定為方向狀態過濾器：

- `1H` 與主方向同向：允許進入 setup 搜尋
- `1H` 反向：直接拒絕該 setup
- `1H` 中性：可保留，但只允許高品質 setup

這裡不採「全部軟加權」，也不採「全部硬過濾」，而是混合式：
- `1D/4H` 是硬方向門檻
- `1H` 是 profit-first 的準硬過濾

完成標準：
- 策略能穩定輸出 `allowed_long / allowed_short / blocked_by_mtf_conflict`
- metadata 明確記錄每層被過濾掉多少 setup

### Step 3：把 ICT setup 重心從 5m reversal 改成 15m setup + 5m/1m entry

第一版主線不再以舊的「5m 單框架 reversal chain」為核心，而改成：

- `15m` 建立主 setup
- 條件固定為：主方向內的 liquidity sweep、MSS、displacement、FVG / OB zone
- `5m` 只做 setup 確認，不重新定義主方向
- `1m` 才做進場觸發

第一版 entry 規則固定如下：

- 只有當 `15m` 已建立有效 zone 時，才允許往下找 entry
- `5m` 必須出現回踩 zone 後的 rejection 或 micro shift
- `1m` 必須出現明確 micro MSS / rejection close 才進場
- 進場必須維持你現有標準：
- `100,000 USD`
- 可用 `100%` capital
- `min 40 shares QQQ`
- `Reward:Risk >= 1.5:1`

第一版不做的東西：

- 不把舊 5m reversal lane 直接與新主線混單
- 不先做 countertrend HTF reversal
- 不先做多 lane 融合
- 不先改 scaling / pyramiding / partials

完成標準：
- 可以產出單獨的 `mtf_topdown_continuation_baseline`
- 可以清楚分辨 `15m setup 數量`、`5m confirm 數量`、`1m actual entries`

### Step 4：研究腳本與校準順序固定，避免亂試

研究順序固定如下，不能跳步：

1. `MTF baseline replay`
- 先建立第一版 top-down baseline
- 只確認能不能在 profit-first 條件下打平或超過現有 MTF 原型

2. `Bias layer calibration`
- 只測 `1D/4H/1H` 的方向組合規則
- 不碰 entry 細節
- 目標是確認方向門檻太嚴還是剛好

3. `15m setup calibration`
- 只測 `15m` 的 MSS / FVG / sweep 幾何
- 不碰 `1m` trigger

4. `5m confirmation calibration`
- 只測 `5m` retest / rejection / confirmation 條件
- 不碰 `1D/4H/1H`

5. `1m execution calibration`
- 只測 `1m` micro shift / rejection entry strictness
- 不碰上層方向與 setup

第一版不優先校準：

- `RR`
- `SL ATR`
- `score`
- `OTE`
- `kill zone`
- continuation/reversal 混 lane

因為這些不是你目前最大的結構瓶頸。

### Step 5：驗證與 promotion gate

這條新 MTF 主線必須同時對兩個基準負責：

- 基準 A：你現在測到的 MTF 原型 `40 trades / +13.4177% / PF 3.1879`
- 基準 B：現有 5m active lite frontier 的 standardized economic replay `18 trades / +12.6548%`

第一版 promotion gate 固定為：

- `total_trades >= 40`
- `total_return_pct > 13.4177%`
- `profit_factor >= 3.1879`
- walk-forward holdout 總和為正
- 不得在 OOS 完全靠少數月份撐起來

第二版目標才是：

- `total_trades >= 60`
- `total_return_pct >= 15%`
- `profit_factor >= 3.0`

如果第一版達不到上述條件，下一步不是回頭調 5m 單框架，而是：
- 先檢查哪一層卡住最多 setup
- 再決定是否加入第二條 lane

## 介面與實作變更

- `src/strategies/ict_entry_model.py`
- 新增一組明確的 MTF 參數，不沿用模糊的單一 `higher_timeframe_alignment`
- 固定增加：
- canonical execution timeframe = `1m`
- bias timeframes = `1D`, `4H`, `1H`
- setup timeframe = `15m`
- confirmation timeframe = `5m`
- trigger timeframe = `1m`
- MTF conflict policy = 本計畫定義的混合式規則
- metadata 必須新增：
- `mtf_daily_blocked`
- `mtf_4h_blocked`
- `mtf_1h_blocked`
- `mtf_direction_long_allowed`
- `mtf_direction_short_allowed`
- `mtf_15m_setups`
- `mtf_5m_confirms`
- `mtf_1m_triggers`

- `src/data/fetcher.py` 或獨立 MTF helper
- 增加 `1m` canonical load 與 resample helper
- 明確支援從 `1m` 產出 `5m/15m/1H/4H/1D`

- `research/ict/...`
- 新增一組 MTF baseline、bias、setup、trigger、walk-forward 分析腳本
- 這組腳本全部都要以 `1m canonical source` 為前提

## 測試與驗收

必做測試：

- `1m -> 5m/15m/1H/4H/1D` 重採樣對時測試
- higher timeframe no-lookahead 測試
- `1D/4H` 衝突時必須阻止交易
- `1D neutral + 4H directional + 1H aligned` 時允許 setup
- 沒有 `15m zone` 時，不得直接由 `5m/1m` 開倉
- `5m` 有 confirm、`1m` 沒 trigger 時，不得開倉
- `RR < 1.5` 時必須拒絕進場
- standardized economic replay 必須能重現 baseline
- walk-forward 必須輸出 train / validation / holdout 三段

## 假設與預設

- 採用 `獲利優先`
- 採用「哪個好用哪個」的實作解讀：`1D/4H` 硬方向、`1H` profit-first 過濾、`15m/5m/1m` 負責 setup 到 entry
- `1m` 已存在 raw parquet，但還不是研究主線的 normalized input，因此 Phase 0 必做正規化
- 第一版主線只做 `top-down continuation / pullback`
- 原本 `5m` reversal lane 保留為 benchmark，不與新主線直接融合
- ORB production baseline 不變，這條仍是獨立 ICT research lane
