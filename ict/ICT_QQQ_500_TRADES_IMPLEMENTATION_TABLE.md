# 審查報告：為何六年交易只有 18 trades / +0.4353% / PF 5.0247

這是一個典型的**「過度擬合（Over-filtering）與條件漏斗（Decision Funnel）」效應**。

雖然這個策略的 Profit Factor 高達 5.02，表現出極高的「勝率與盈虧比」品質，但 6 年僅 18 次交易意味著系統因為**過度嚴格地追求「教科書級別的完美 ICT 形態」**，扼殺了 99.9% 潛在的不完美行情。

我仔細審查了您 `src\strategies\ict_entry_model.py` 的原始碼，問題根源在於策略中一系列**嚴格把關的幾何、時間與指標條件形成了多層「AND（且）」過濾邏輯**。只要上述任何一環稍微不完美，整個 setup 就會被無情拋棄。這從您程式碼內註解明確提到的保留 `"18-trade" branch` 就可以獲得證實（如 `build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params` 函式註解所言）。

以下為具體讓訊號一路被「篩到死」的五個核心漏斗關卡：

### 1. Sweep 後的反轉時間過短 (Time-to-Shift Restriction)

當價格成功獵殺流動性（Sweep）後，您的程式碼設定了極其嚴格的有效期限：

```python
primary_expiry_index = idx + liq_sweep_recovery_bars  # 通常設定為 3 或 4
```

這代表市場在掃過前低/前高之後，**必須在短短的 3-4 根 K 線內立刻V型反轉並突破對側的市場結構點（Structure Shift）**。一旦價格在底部盤整稍微久一點，這個訊號就會直接被歸類為 `sweep_expired_before_shift` 並遭到刪除。

### 2. 要求過激的 FVG 動能缺口 (Massive FVG Requirement)

在發生結構轉變（Shift）時，程式要求必須留下極為明顯的動能失衡（Fair Value Gap）：

```python
"fvg_min_gap_pct": 0.0006 到 0.0010 之間
```

這個條件極其苛刻，對於 NQ（那斯達克）等指數而言，0.1% (`0.0010`) 意味著在單一或連貫 K 線中必須硬生生拉出十幾甚至二十點的「無重疊缺口」。多數正常合理大小的 FVG 會過不了這個門檻，導致失去進場區。

### 3. 太過完美的 SMT 分歧（SMT Divergence Filter）

如果啟用了 `use_smt_filter`，程式會嚴格比對主商品與對標商品（PeerHigh/PeerLow，通常是 ES 與 NQ）是否發生背離：

```python
smt_confirmed = bool(peer_data_ready and not peer_swept_high)
```

這要求當主商品突破結構點時，另一商品「絕對不能」突破它的對應結構。這種極為乾淨的宏觀分歧本來就不是每天都有，因此大多數普通的 Sweep 會在這層被標記為 `smt_filtered_sweeps` 而剔除。

### 4. 強迫要精準回踩 50% 深度 (Consequent Encroachment)

好不容易有了一個好的 Sweep、迅速的 Shift 加上巨大的 FVG，在等待進場時，策略又加了一道極嚴格的關卡：

```python
required_touch = pending_long.zone_upper - gap_height * fvg_revisit_depth_ratio
if low > required_touch:
    metadata["fvg_depth_filtered_retests"] += 1
    long_entry_resolved = True  # 直接放棄進場
```

`fvg_revisit_depth_ratio` 常被設為 `0.5`，也就是說價格回踩時，**不可以只摸到 FVG 邊緣就走，必須精準刺入 FVG 的 50% 深度（ICT 的 CE 概念）**。只要走勢太強只碰到 10% 深度就噴發，您就完全接不到單。加上後續的 `fvg_revisit_min_delay_bars` 要求不能立刻回踩，導致訊號再次銳減。

### 5. 交易時間與輔助指標閘門 (Session & Context Gates)

從 `_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES` 這類長名稱設定可以看出，此策略可能限制了：

- **NY Only**：一天只做紐約特定的幾個鐘頭。
- **Reward/Risk Ratio**：潛在獲利風險比（RR）太小不進場。
- 其他可能開啟的條件：PD Array 匹配（Discount / Premium）、AMD 模型把關、或是 Previous Session Anchor 限制。

---

### 💡 程式碼除錯與優化建議

因為策略內有非常好的追蹤機制，您在 `metadata` 字典裡寫滿了 Counters（例如 `metadata["fvg_depth_filtered_retests"] += 1`）。

**第一步：觀察哪裡過度殺生**  
強烈建議您在回測結束後，印出這個 `metadata`，排名一下哪些原因斃掉了最多的 Setup，例如：是 `score_filtered_shifts` 太多？還是 `displacement_filtered_shifts` 太兇？

**第二步：將「硬條件 (Hard Filter)」轉化為「軟評分 (Soft / Score Filter)」**  
雖然程式已經引進了 Score 機制 (`score >= min_score_to_trade`)，但很多條件依然使用布林值 (`if not XXX : continue`) 硬核阻擋。若想增加交易量（例如增加到每年 50~100 次的合理水準），您需要：

1. **放寬 FVG 缺口要求**：將 `fvg_min_gap_pct` 從 `0.0010` 下調至 `0.0003`。
2. **放寬進場回踩深度**：將 `fvg_revisit_depth_ratio` 改為 `0.0` 或 `0.1`，只要價格碰到 FVG 邊沿就建倉。
3. **增加反轉容忍期**：將 `liq_sweep_recovery_bars` 開放至 `8` 或 `10`，允許 Sweep 取籌碼後具備稍長的盤整震盪再發動結構突破。

---
