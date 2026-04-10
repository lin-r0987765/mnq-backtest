# ICT 專案深度診斷報告

**當前卡關狀態**: `63 trades / +7.4918% / PF 2.3102`（近六年）
**日期**: 2026-04-09
**分析範圍**: 整個 ICT 專案架構、策略邏輯、評分系統、bug、參數空間

---

## 一、你目前卡在哪裡的定位

`63 trades / +7.4918% / PF 2.3102` 在這個專案的 frontier 地圖上，是一個「中密度、中品質」的位置——介於：

- **strict**: `8 trades / +7.7249% / PF ∞`（極稀疏、幾乎無虧損）
- **lite quality**: `18 trades / +0.4353% / PF 5.0247`（小樣本高品質）
- **reversal balance**: `40 trades / +13.4177% / PF 3.1879`（中密度的反轉路徑）
- **continuation density**: `93 trades / +11.9309% / PF 1.8204`（目前最密）

你這條 63 筆的線，**問題不是策略不會賺錢**（PF 2.31 是健康的），而是它**沒有比 40 筆那條 reversal balance 線更好**——density 提高了一點，但報酬被吃掉、PF 也低於它。

這代表你新增進來的那 23 筆是「負的 expectancy 或低度正向的 noise trades」，稀釋了品質而沒有換到實質的 edge。

---

## 二、根因分析（從整個專案的脈絡看）

### 1. 評分系統沒有真正的篩選效果（DESIGN-1）

`ict_entry_model.py` 裡的打分是：

```
score = liquidity_sweep(3) + bos(2) + choch(3) = 8
       + 任一 delivery array(2)
```

任何通過結構轉換的 setup 起跳就 10 分，而預設 `min_score_to_trade = 6`。**所有通過基本 sequence 的 setup 都無條件達標**。

結果：當你為了從 18 → 40 → 63 放寬幾何條件（`fvg_min_gap_pct`、`revisit_depth_ratio`、`swing_threshold`、`liq_sweep_threshold`），新加進來的 setup 沒有任何「品質維度」可以擋住差的 sweep、差的 displacement、差的 FVG。這就是為什麼你每密度化一次，PF 就往下掉。

`SCORE_SWEEP_DEPTH_QUALITY / DISPLACEMENT_QUALITY / FVG_GAP_QUALITY` 這些權重雖然存在，但預設都是 0，實際上沒在發揮作用。

---

### 2. 兩個真實的 bug 正在讓「品質 gate」空轉

**BUG-1（P0 嚴重）**：`fvg_origin_body_min_pct` 原本要檢查 FVG 的**位移 K 線**（三根裡中間那根），但現行程式碼取的是 `zone[0]`，也就是**確認 K 線**。

這條品質 gate 本意是「過濾掉位移身體薄弱的 FVG」，實際卻在量錯誤的對象。你以為關上了一道閘門，但閘門其實裝在錯的位置。

位置：`src/strategies/ict_entry_model.py` 第 1396-1405 行（bullish）、第 1508-1517 行（bearish）。

**BUG-2（P0 嚴重）**：當 FVG 被 `origin_lag` 或 `origin_body` 拒絕時，程式用 `continue` 直接跳過整根 K 線。這造成兩件事：

- **OB / Breaker / IFVG 的降級路徑完全被跳過**——你每次拒絕一個 FVG，就同時把該 K 線所有備援 delivery array 一起丟掉。這會讓真正的品質 gate 產生「過度懲罰」副作用，逼你回頭去放寬 FVG 條件，進一步稀釋整體品質。
- **同一根 K 線的 `pending_short` 處理也被跳過**（BUG-3），讓雙向 setup 互相阻斷。

這三個 bug 加在一起的效果是：**你的高品質設定被 bug 噬掉收益，於是你憑直覺放寬參數補密度，結果又把 noise 納進來**。63 trades 這條線正是這個循環的典型產物。

---

### 3. EOD 邊界用的是 UTC 而不是 ET（BUG-4）

`next_is_new_day` 用 `.date()` 判斷跨日，但資料是 UTC，實際交易日是 ET。在 UTC 午夜（約 ET 晚上 7~8 點）附近會觸發錯誤的日內平倉。

這會隨機砍掉原本可以持有到隔日的 trade，在 6 年樣本上造成難以診斷的漏失。對於 63 筆的小樣本尤其致命——可能只要 2~3 筆被錯誤平倉，就讓 PF 從 2.8 掉到 2.3。

位置：`src/strategies/ict_entry_model.py` 第 1176-1179 行。

---

### 4. Session Array 視窗與 NY-Only profile 不相容（DESIGN-5）

預設的 `imbalance window (3:00-4:30 AM ET)` 落在 NY-Only profile 的交易時段 (9:00 AM - 3:00 PM ET) **之外**，永遠不會觸發；`structural window (10:30-11:30 AM ET)` 卻會阻擋該時段的 FVG 進場。

你等於關掉了半邊 refinement、只留下會扣分的那一邊。

---

### 5. 你是用 QQQ 5m 單框架做 ICT——但 ICT 的 edge 其實在 top-down

`ict/PLAN.md` 已經診斷到這點：

> 你現在最缺的不是再把 5m reversal 調得更花，而是先建立一條「方向更乾淨、進場更細、樣本數更可擴張」的主幹。

目前所有的 9 個 profile builder 都是在同一個 5m 框架內微調，本質上是在同一片已經被榨乾的地毯上挪家具。真正能同時提升 density **和** quality 的是 `1D/4H` 給方向 → `15m` 建 setup → `5m/1m` 進場的 MTF 架構。

你已經有 `build_ict_mtf_topdown_continuation_profile_params` 這條線，但它還是 `enable_smt=False` 且沒被當主戰場。

---

### 6. 統計顯著性本身也是問題

6 年 / 63 筆 ≈ **每年 10.5 筆**。用 PF 2.31 去推斷這條線是否「真的有 edge」是非常薄的樣本：

- 95% CI 的 PF 大概是 `[1.3, 4.1]`，意思是它可能只是 PF 1.3 的策略
- `+7.49% / 6 年 ≈ 每年約 1.2% 報酬`，扣掉滑價和手續費後幾乎沒有意義
- 這已經是 `ICT_PROMOTION_MEMO.md` 明確拒絕升級 ICT 的原因（Gate A: `>= 100 trades` FAIL）

---

## 三、你為什麼會卡住（策略層面的結構性 trade-off）

```
想提高 density → 必須放寬 geometry gates
    ↓
放寬 geometry → 評分系統無力過濾、bug 讓真正的品質 gate 失效
    ↓
品質下降 → 必須加回別的 filter 補 edge
    ↓
filter 又回來壓低 density
    ↓
回到起點
```

**這不是參數不對，是架構沒辦法讓 density 和 quality 正交地提升。**

---

## 四、該怎麼做（按優先順序）

### P0：先修三個 bug，再談任何參數調整

在 bug 修掉以前，你的參數掃描結果都帶著 noise，現在的 63 trades 可能本來就該是 55 或 75，無從判斷邊際動作是否真有效。

順序：

1. **BUG-1（FVG origin body 用錯 K 線）**——改 index 就能修。修完後重跑你現在的 profile，**把新的 baseline 記下來**。這個 baseline 才是後續比較的參考點。
2. **BUG-2（FVG 拒絕跳過 OB/Breaker 降級）**——把 `continue` 改成 `zone = None`，讓降級路徑生效。預期效果：density 會上升（多了 OB/Breaker 入場），但因為是有品質過 FVG 失敗才降級的，PF 不應該崩。
3. **BUG-4（EOD 用 ET 時區）**——加 `tz_convert("America/New_York")`。小改動，但避免隨機漏失。

完成 P0 後：**重跑 63 trades 的 profile**，你會得到一個新的真實 baseline，那才是你真正要面對的分數。

---

### P1：重建評分系統，讓品質軸真正可用

目前 `score_sweep_depth_quality`、`score_displacement_quality`、`score_fvg_gap_quality` 都是 0——**這是最低成本的改進槓桿**。

具體做法（概念層級）：

- 讓 **BOS 和 CHOCH 互斥**，不要雙算。CHOCH 是反轉，BOS 是延續，一個 setup 本質上只可能是其中之一。
- 引入「分級」的品質維度：sweep 深度（wick 多深）、displacement body vs ATR、FVG gap vs ATR、sweep-to-shift 的速度。給每個維度 0~2 分，真正有等級之分。
- 把 `min_score_to_trade` 從 6 調高到 10~12，讓評分變成真正的門檻，而不是擺設。

這樣做之後，**你就可以用評分取代參數放寬**：想提高 density，不是放寬 `fvg_min_gap_pct`，而是降 `min_score_to_trade`；想提高品質，反之。這才是一個可以線性掃描的單一控制軸。

---

### P2：把 63 trades profile 本身做 ablation 分析

針對目前 63 trades 的 profile，逐一關閉以下 filters 並重跑，看哪個 filter 實際上是負貢獻：

- `use_smt_filter`
- `use_premium_discount_filter`
- `use_external_liquidity_filter`
- `use_amd_filter`
- `use_macro_timing_windows`
- `use_prev_session_anchor_filter`
- `use_session_array_refinement`（特別要測，因為 DESIGN-5 的問題）
- `use_kill_zones`

很有可能有 1~2 個 filter 是「看起來合理但實際扣分」的。這不需要改邏輯，只需要跑 N 次 backtest。

---

### P3：做 time-of-day / day-of-week 診斷（這是免費的 alpha）

`EXECUTIVE_SUMMARY.txt` 和 `RESEARCH_HYPOTHESES.txt` 已經在 404-trade 的 QC baseline 上找出明確的 pattern：

- **10am-1pm ET** 的 PF 遠高於其他時段
- **Monday / Friday** win rate 比 Tuesday 高 15pp
- **週二** 是明確的負向偏差
- **Jul / May** 是 peak，**Feb / Mar** 是負貢獻
- 89% 的虧損單曾經處於獲利狀態（exit 過早）

你**還沒把這些 insight 套到 ICT lane 上**。請針對你現在的 63 筆 trades，輸出一份 time/day/month 分佈表：

- 如果它和 QC baseline 一致 → 你就有非常高把握可以加一組 session filter 直接提升 PF（不需要改策略邏輯）
- 如果不一致 → 代表 ICT lane 的 edge 來源不同，那個 insight 也很有價值

---

### P4：Exit 品質分析（最大槓桿的潛在改進）

QC baseline 分析已經明確指出：**89% 的虧損單都曾經是獲利的**，39% 的虧損單 MFE > $50。這是整個專案裡估值 +$15,000 的單一最大機會。

你目前的 `stop_loss_atr_mult = 2.0`、`take_profit_rr = 3.0` 是固定的 bracket exit。請做以下離線分析（不改程式碼，只做數據分析）：

- 對現有 63 筆 trades，計算每筆的 MFE（最大有利偏移）和 MAE（最大不利偏移）
- 畫 MFE / final P&L 的散點圖
- 問：**如果在 MFE 達到某個閾值後開始 trailing stop，會發生什麼？**
- 問：**如果虧損單只要 MFE > 1R 就平倉鎖定，會發生什麼？**

這會告訴你 exit 是在漏錢還是在保護錢。

---

### P5：考慮把主戰場挪到 MTF top-down 路線

這是中期的架構決策，不是立即該做的，但 `ict/PLAN.md` 已經寫得很清楚：在 5m 單框架繼續調參的邊際報酬正在遞減。

當你把 P0~P4 做完、還是卡在 PF 2.3 附近時，就該認真把 `build_ict_mtf_topdown_continuation_*` 當主線而不是候選實驗。多時間框架的 edge 來自「方向乾淨 → setup 才有意義」，而不是 5m 內部再堆一層 filter。

---

## 五、建議的下一步（最小行動方案）

如果你只想先做一件事看到改善，建議：

> **先不碰參數、先修 BUG-1 和 BUG-4，重跑同一個 63-trades profile，然後把 MFE/MAE 輸出出來做 exit 診斷。**

理由：

- BUG-1 和 BUG-4 是單點修正，風險極低
- 重跑給你一個乾淨的 baseline，後續任何參數動作才有比較基準
- MFE/MAE 診斷會告訴你「錢是進場漏掉還是出場漏掉」——這會決定接下來 2~3 週的工作方向

### 請不要再做的事（基於你現在卡關的模式）

- ❌ 不要再掃 `fvg_min_gap_pct` / `revisit_depth_ratio` / `swing_threshold` 的邊際值——你已經在這條等高線上走了好幾輪（18 → 22 → 40 → 63 → 93），繼續走不會有突破
- ❌ 不要再新增 profile builder——你已經有 9 個
- ❌ 不要把 continuation 和 reversal fusion 成一條 lane——`ict_combined_lanes.json` 已是 `1464 trades / -6.04%`，已被明確拒絕

---

## 六、一句話總結

> 你卡住不是因為參數還沒調對，而是因為 **(a) 有 bug 讓你的品質 gate 空轉、(b) 評分系統無效所以 density 和 quality 無法正交提升、(c) 你在 5m 單框架的空間已經榨乾**。先修 bug，再重建評分，再考慮 MTF top-down——這是能把 63 trades 推到有統計意義且經濟可行的水平的順序。

---

## 附錄：資料來源

- `ICT_DEEP_ANALYSIS_REPORT.md` — bug 清單與設計層級問題
- `ICT_PROMOTION_MEMO.md` — 各 lane benchmark 與 promotion gate
- `EXECUTIVE_SUMMARY.txt` — QC baseline 404-trade 分析
- `RESEARCH_HYPOTHESES.txt` — time/day/month seasonality 假設
- `ict/PLAN.md` — MTF top-down 重構計畫
- `AGENT_HANDOFF.md` — 專案當前狀態與 Round 320 delta
- `src/strategies/ict_entry_model.py` — 策略實作主體（2092 行）
