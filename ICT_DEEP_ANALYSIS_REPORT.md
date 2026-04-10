# ICT 專案深度解析報告

**日期**: 2026-04-08
**版本**: v3.50.139 / Iteration 226
**分析範圍**: `src/strategies/ict_entry_model.py` (1723 行) 及相關測試/分析工具鏈

---

## 一、架構概覽

此專案實作了一套 **機械式 ICT 進場模型**，核心流程為：

**Liquidity Sweep → Structure Shift (MSS/CHOCH) → Delivery Array (FVG/OB/Breaker/IFVG) → Entry**

並疊加了多層上下文過濾器：Premium/Discount、SMT Divergence、AMD Path、Previous Session Anchor、Session Array Refinement、Kill Zone、Macro Timing、Daily Bias。

目前共有 9 個逐層堆疊的 profile builder 函式，代表從最基礎到最前沿的參數組合。

---

## 二、已確認的 Bug

### BUG-1 (嚴重): FVG Origin Body 檢查使用了錯誤的 K 線

**位置**: 第 1396-1405 行（bullish）、第 1508-1517 行（bearish）

**問題**: `fvg_origin_body_min_pct` 檢查應該評估 FVG 的**位移 K 線**（3 根 K 線模式中的中間那根），但程式碼使用 `zone[0]`（第 3 根 K 線 / 確認 K 線）。

```python
# 目前的程式碼（錯誤）
zone_index = int(zone[0])  # 這是第 3 根 K 線
origin_open = float(df["Open"].iat[zone_index])
# ...

# 應該改為
zone_index = int(zone[0])
origin_idx = zone_index - 1  # 位移 K 線是中間那根
origin_open = float(df["Open"].iat[origin_idx])
```

**影響**: Origin body quality 過濾器評估的是錯誤的 K 線，可能誤判 FVG 品質。

**修復方案**:
```python
# 在 bullish 和 bearish 兩個區塊中：
zone_index = int(zone[0])
origin_idx = max(zone_index - 1, 0)  # 中間 K 線 = 位移 K 線
origin_open = float(df["Open"].iat[origin_idx])
origin_high = float(df["High"].iat[origin_idx])
origin_low = float(df["Low"].iat[origin_idx])
origin_close = float(df["Close"].iat[origin_idx])
```

---

### BUG-2 (嚴重): FVG 被拒絕時 `continue` 跳過了整根 K 線的所有處理

**位置**: 第 1395、1405、1475、1517 行

**問題**: 當 FVG 通過了 `_detect_fvg_zone` 但被 `fvg_origin_max_lag_bars` 或 `fvg_origin_body_min_pct` 拒絕時，程式碼使用 `continue` 跳過整根 K 線。這造成兩個後果：

1. **OB/Breaker/IFVG 後備路徑被跳過**：FVG 品質不夠好時，應該降級到 OB → Breaker → IFVG 後備路徑，而不是放棄這根 K 線。
2. **`pending_short` 處理被跳過**：在 `pending_long` 的 swept 狀態處理中使用 `continue`，會讓同一根 K 線的 `pending_short` 轉態邏輯完全被跳過。

**修復方案**: 將 `continue` 改為 `zone = None`，讓流程自然降級到後備路徑：

```python
# 目前：
if idx - zone_index > fvg_origin_max_lag_bars:
    metadata["fvg_origin_lag_filtered_shifts"] += 1
    continue  # <-- 問題

# 修復：
if idx - zone_index > fvg_origin_max_lag_bars:
    metadata["fvg_origin_lag_filtered_shifts"] += 1
    zone = None
    zone_kind = "ob"
    # 接下來自然進入 OB 偵測...
```

同樣的修復需套用於：displacement body、displacement range、structure buffer 等所有 `continue` 點。不過 displacement 類的拒絕確實應該跳過整個結構轉換（因為如果位移品質不夠，結構轉換本身不成立），所以那些 `continue` 可能是有意為之。但 FVG origin 的拒絕不應連帶跳過 OB 後備路徑。

---

### BUG-3 (中等): `pending_long` 處理中的 `continue` 跳過 `pending_short` 處理

**位置**: 第 1356-1466 行中所有 `continue` 語句

**問題**: 主迴圈的結構是：
```
for idx in range(len(df)):
    ... exit logic ...
    ... pending_long sweep -> shift logic ...  # continue 在這裡跳出
    ... pending_short sweep -> shift logic ...  # 被跳過
    ... entry trigger logic ...                 # 被跳過
```

當 `pending_long` 的 swept 狀態處理遇到 displacement 拒絕並 `continue` 時，`pending_short` 的結構轉換檢查在這根 K 線上完全不會執行。

**影響**: 在雙向同時有 pending setup 的情況下，一方的拒絕會阻斷另一方的進展。

**修復方案**: 將 `continue` 替換為流程控制標記（flag），或重構為兩個獨立的處理區塊。

---

### BUG-4 (中等): `next_is_new_day` 未做時區轉換

**位置**: 第 1176-1179 行

**問題**:
```python
next_is_new_day = (
    idx == len(df.index) - 1
    or pd.Timestamp(df.index[idx + 1]).date() != pd.Timestamp(ts).date()
)
```

`.date()` 使用 UTC 日期邊界判斷日內交易結束。但美股交易所的實際交易日邊界是 ET 時區。如果 K 線資料是 UTC 時間戳，午夜 UTC 前後的 K 線會被錯誤地判定為跨日。

**影響**: 在 UTC 午夜附近（ET 晚上 7-8 點），可能觸發不正確的 EOD 平倉。

**修復方案**:
```python
def _same_trading_day(ts1, ts2, tz="America/New_York"):
    d1 = _ensure_utc(ts1).tz_convert(tz).date()
    d2 = _ensure_utc(ts2).tz_convert(tz).date()
    return d1 == d2
```

---

## 三、設計層級問題

### DESIGN-1: 評分系統形同虛設

**位置**: 第 1439 行

```python
score = score_liquidity_sweep + score_bos + score_choch  # 3 + 2 + 3 = 8
```

每次結構轉換都自動獲得 Liquidity Sweep (3) + BOS (2) + CHOCH (3) = **8 分**，再加上任何 delivery array 的分數（FVG/OB/Breaker/IFVG 都是 2 分），總分至少 **10 分**。預設 `min_score_to_trade = 6`。

**結果**: 所有通過結構轉換的 setup 都會達到最低分數門檻。評分系統完全沒有篩選效果。

**改善建議**:
- 區分 BOS 和 CHOCH（CHOCH 是反轉，BOS 是延續），不要同時計分
- 引入真正可變的分數維度（例如 sweep 深度、displacement 強度、FVG 品質）
- 或直接移除評分系統，改用硬性的 pass/fail 過濾器

---

### DESIGN-2: 巨型單函式 `generate_signals` (770+ 行)

主要邏輯全部塞在一個超長函式裡，包含：
- 預計算（ATR、Daily Bias、Premium/Discount、AMD、Previous Session Anchor）
- 主迴圈中的退出邏輯、掃描邏輯、結構轉換邏輯、進場觸發邏輯
- 大量重複的 bullish/bearish 對稱程式碼

**改善建議**: 拆分為：
1. `_precompute_context(df, params)` → 所有預計算指標
2. `_process_sweep(idx, direction, ...)` → 掃描偵測與過濾
3. `_process_shift(pending, idx, ...)` → 結構轉換與 delivery array 選擇
4. `_process_entry(pending, idx, ...)` → 回測觸發邏輯
5. `_process_exit(position, idx, ...)` → 退出邏輯

---

### DESIGN-3: Profile Builder 大量重複

9 個 profile builder 函式結構完全相同：
```python
def build_xxx_params(*, enable_smt=True, overrides=None):
    params = {**_DEFAULT_PARAMS, **_XXX_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides: params.update(overrides)
    return params
```

**改善建議**: 用一個通用工廠函式 + 註冊表：
```python
_PROFILES = {
    "research": _ICT_RESEARCH_PROFILE_OVERRIDES,
    "paired_survivor": _ICT_PAIRED_SURVIVOR_PROFILE_OVERRIDES,
    ...
}

def build_profile(name: str, *, enable_smt=True, overrides=None):
    base = _PROFILES[name]
    params = {**_DEFAULT_PARAMS, **base}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides: params.update(overrides)
    return params
```

---

### DESIGN-4: 效能瓶頸 — 純 Python 逐 K 線迴圈

`generate_signals` 對每根 K 線做 Python-level 迭代，包含大量物件存取和條件判斷。對於 5 分鐘資料橫跨數年的資料集，這會非常慢。

**改善建議**:
- 短期：用 NumPy 陣列預提取 OHLC，避免每根 K 線的 `df.iloc[idx]` 存取
- 中期：將 sweep detection 和 rolling 計算向量化
- 長期：考慮 Numba JIT 或 Cython 加速核心迴圈

---

### DESIGN-5: Session Array 預設視窗與 NY-Only Profile 不匹配

預設的 imbalance window (3:00-4:30 AM ET) 在 NY-Only profile 的交易時段 (9:00 AM - 3:00 PM ET) 之外，永遠不會被觸發。structural window (10:30-11:30 AM ET) 在交易時段內，會阻擋該時段的 FVG 進場。

**改善建議**: 為 NY-Only profile 定義專屬的 session array 視窗，例如：
- Imbalance: 9:30-10:30 AM ET（開盤後第一個小時）
- Structural: 11:00-12:00 PM ET（午間盤整期）

---

## 四、程式碼品質改善建議

### 4.1 型別安全

- 許多 `self.params["xxx"]` 存取沒有型別檢查，容易因拼字錯誤產生 KeyError
- 建議用 `@dataclass` 或 Pydantic 模型取代 `dict[str, Any]`

### 4.2 測試覆蓋率

- `test_ict_strategy.py` 有 60k 字元，覆蓋了基本功能
- **缺少**: FVG origin body 方向邏輯的邊界測試、`continue` 對 pending_short 影響的測試、跨 UTC 午夜的 EOD 退出測試

### 4.3 分析器檔案爆炸

- 目前有 **60+ 個** `analyze_ict_*.py` 和對應的 `test_ict_*.py` 檔案
- 建議合併為參數化分析框架：一個分析器 + 一個參數設定檔

---

## 五、Bug 修復優先順序

| 優先順序 | Bug | 影響 | 修復難度 |
|---------|-----|------|---------|
| P0 | BUG-1: FVG origin body 檢查錯誤 K 線 | 過濾器評估錯誤目標 | 低（改 index） |
| P0 | BUG-2: FVG 拒絕時 continue 跳過後備路徑 | 遺失 OB/Breaker/IFVG 入場機會 | 中（重構流程） |
| P1 | BUG-3: continue 跳過 pending_short | 雙向 setup 互相阻斷 | 中（重構迴圈） |
| P1 | BUG-4: next_is_new_day 時區問題 | UTC 午夜附近錯誤平倉 | 低（加時區轉換） |
| P2 | DESIGN-1: 評分系統無效 | 所有 setup 都通過 | 中（重新設計評分） |

---

## 六、建議的下一步迭代方向

1. **修復 P0 Bug** — 特別是 FVG origin body 的錯誤 K 線問題，這直接影響最新的校準結果
2. **重構 `continue` 為降級路徑** — 讓 FVG 品質拒絕自然降級到 OB/Breaker 後備
3. **拆分 `generate_signals`** — 先抽取預計算和 sweep detection 為獨立函式
4. **Profile builder 工廠化** — 減少維護負擔
5. **新增邊界測試** — 覆蓋上述所有 bug 的回歸測試
