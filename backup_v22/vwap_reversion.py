"""
VWAP Mean-Reversion Strategy — v13 RSI 收窄 + 日級別 ATR 自適應版

改進項目（迭代 #14）：
1. RSI 閾值收窄 35/65→32/68，WR 46.2%→55.6%（+9.4pp），Sharpe 10.1→10.5
2. 保留 v12 partial_tp 設定（trail=0.002, max_hold=32）
3. 日級別 ATR 自適應（搭配 run_combined_analysis.py 使用）
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

_DEFAULT_PARAMS: dict[str, Any] = {
    "k": 1.5,                # Band multiplier — v9: 1.3→1.5 Sharpe +4.7%
    "sl_k_add": 0.5,         # Extra k for stop-loss — v9: 0.7→0.5 配合更寬 band
    "std_window": 30,        # Rolling window for std (grid confirmed 30 optimal)
    "rsi_period": 14,        # RSI period
    "rsi_os": 32,            # RSI oversold — v14 收窄提升 WR 55.6%（原 35）
    "rsi_ob": 66,            # v20: 小幅放寬 overbought 門檻，增加優質 short 機會
    "close_before_min": 15,
    "atr_period": 14,        # ATR 計算週期
    "atr_min_pct": 0.0005,   # 最小 ATR/價格比（過濾低波動期）
    "vol_filter": True,      # 成交量過濾開關
    "vol_min_mult": 0.5,     # 成交量至少要均量的倍數
    "entry_start_time": "09:45",   # 最早入場時間（09:35/09:40 測試均退步）
    "entry_end_time": "15:30",     # 最晚入場時間
    "max_trades_per_day": 2,       # 每日最大交易次數（3筆測試退步，維持2）
    # v4 趨勢過濾（改為 price vs EMA20）
    "ema_trend_filter": True,      # EMA 趨勢過濾開關
    "ema_fast": 20,                # EMA 週期（用於趨勢判斷）
    "ema_slow": 50,                # 慢速 EMA（保留但不用於過濾）
    "ema_mode": "ema_cross",       # v5: ema_cross 為最佳模式（EMA20>EMA50）
    "dynamic_tp": True,            # 動態止盈（趨勢方向多賺）
    "tp_bonus_pct": 0.2,           # TP 超越 VWAP 的比例
    "bb_width_min": 0.0005,        # BB 寬度最小值 — v5 放寬增加交易機會
    # v9 反轉 K 棒確認（放寬版）
    "reversal_confirm": False,      # 反轉 K 棒確認開關
    "reversal_mode": "relaxed",     # strict(v8,body>0.4)/relaxed(body>0.25)/simple(close vs open)
    # v10 前一根K棒反轉確認
    "prev_bar_reversal": False,     # 檢查前一根K棒的反轉方向（而非當根）
    # v11 部分止盈
    "partial_tp": True,             # v12: 預設開啟（v11 驗證有效）
    "partial_tp_trail_pct": 0.002,  # v12: trail=0.2% — grid最佳（0.002 > 0.003）
    "partial_tp_max_hold": 32,      # v12: max_hold=32 bars (2h40m) — Sharpe 最佳
}


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_vwap_and_std(
    session: pd.DataFrame, std_window: int
) -> tuple[pd.Series, pd.Series]:
    """Compute session-anchored VWAP and rolling price std."""
    typical = (session["High"] + session["Low"] + session["Close"]) / 3
    cum_tp_vol = (typical * session["Volume"]).cumsum()
    cum_vol = session["Volume"].cumsum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol
    std = typical.rolling(std_window, min_periods=2).std()
    return vwap, std


def _is_bullish_reversal(row: pd.Series, mode: str = "relaxed") -> bool:
    """Check for bullish reversal candle.

    Modes:
    - strict (v8): body > 40% of range
    - relaxed (v9): body > 25% of range
    - simple: just close > open
    """
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    if mode == "simple":
        return c > o
    body = abs(c - o)
    full_range = h - l
    if full_range <= 0:
        return False
    body_threshold = 0.4 if mode == "strict" else 0.25
    # Hammer: small body at top, long lower wick
    lower_wick = min(o, c) - l
    if lower_wick > 2 * body and c >= o:
        return True
    # Bullish close (close > open with body > threshold of range)
    if c > o and body / full_range > body_threshold:
        return True
    return False


def _is_bearish_reversal(row: pd.Series, mode: str = "relaxed") -> bool:
    """Check for bearish reversal candle.

    Modes:
    - strict (v8): body > 40% of range
    - relaxed (v9): body > 25% of range
    - simple: just close < open
    """
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    if mode == "simple":
        return c < o
    body = abs(c - o)
    full_range = h - l
    if full_range <= 0:
        return False
    body_threshold = 0.4 if mode == "strict" else 0.25
    # Shooting star: small body at bottom, long upper wick
    upper_wick = h - max(o, c)
    if upper_wick > 2 * body and c <= o:
        return True
    # Bearish close (close < open with body > threshold of range)
    if c < o and body / full_range > body_threshold:
        return True
    return False


class VWAPReversionStrategy(BaseStrategy):
    """VWAP mean-reversion strategy — v12 保守 partial_tp + exit 修復版"""

    name = "VWAP_Reversion"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**_DEFAULT_PARAMS, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        k: float = float(self.params["k"])
        sl_k_add: float = float(self.params["sl_k_add"])
        std_window: int = int(self.params["std_window"])
        rsi_period: int = int(self.params["rsi_period"])
        rsi_os: float = float(self.params["rsi_os"])
        rsi_ob: float = float(self.params["rsi_ob"])
        close_before_min: int = int(self.params["close_before_min"])
        atr_period: int = int(self.params.get("atr_period", 14))
        atr_min_pct: float = float(self.params.get("atr_min_pct", 0.0005))
        vol_filter: bool = bool(self.params.get("vol_filter", True))
        vol_min_mult: float = float(self.params.get("vol_min_mult", 0.5))
        entry_start = self.params.get("entry_start_time", "10:00")
        entry_end = self.params.get("entry_end_time", "15:30")
        max_trades = int(self.params.get("max_trades_per_day", 1))

        # v4 趨勢過濾參數
        ema_trend_filter: bool = bool(self.params.get("ema_trend_filter", True))
        ema_fast: int = int(self.params.get("ema_fast", 20))
        ema_slow: int = int(self.params.get("ema_slow", 50))
        ema_mode: str = str(self.params.get("ema_mode", "price_vs_ema"))
        dynamic_tp: bool = bool(self.params.get("dynamic_tp", True))
        tp_bonus_pct: float = float(self.params.get("tp_bonus_pct", 0.2))
        bb_width_min: float = float(self.params.get("bb_width_min", 0.001))
        reversal_confirm: bool = bool(self.params.get("reversal_confirm", True))
        reversal_mode: str = str(self.params.get("reversal_mode", "relaxed"))
        prev_bar_reversal: bool = bool(self.params.get("prev_bar_reversal", False))
        partial_tp: bool = bool(self.params.get("partial_tp", False))
        partial_tp_trail_pct: float = float(self.params.get("partial_tp_trail_pct", 0.003))
        partial_tp_max_hold: int = int(self.params.get("partial_tp_max_hold", 24))

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)
        sl_stop = pd.Series(np.nan, index=df.index)
        tp_stop = pd.Series(np.nan, index=df.index)

        # 預計算 RSI
        rsi_full = _compute_rsi(df["Close"], rsi_period)

        # 預計算 ATR
        high = df["High"]
        low = df["Low"]
        close_col = df["Close"]
        tr = pd.concat([
            high - low,
            (high - close_col.shift(1)).abs(),
            (low - close_col.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_full = tr.rolling(atr_period, min_periods=1).mean()

        # 預計算成交量均線
        vol_ma = df["Volume"].rolling(20, min_periods=1).mean()

        # v3: 預計算 EMA 趨勢
        ema_fast_series = df["Close"].ewm(span=ema_fast, adjust=False).mean()
        ema_slow_series = df["Close"].ewm(span=ema_slow, adjust=False).mean()

        for date, session in df.groupby(df.index.date):
            sess = session.between_time("09:30", "16:00")
            if len(sess) < std_window + 5:
                continue

            vwap, std = _compute_vwap_and_std(sess, std_window)
            rsi = rsi_full.reindex(sess.index)
            atr = atr_full.reindex(sess.index)

            force_close_ts = sess.index[-1] - pd.Timedelta(minutes=close_before_min)

            in_long = False
            in_short = False
            daily_trades = 0
            entry_vwap = 0.0
            entry_std = 0.0
            prev_row = None  # v10: 追蹤前一根K棒
            # v11 部分止盈狀態
            partial_phase = 0  # 0=未入場, 1=全持倉, 2=部分止盈後半倉trailing
            partial_best_price = 0.0
            partial_bars_held = 0

            for ts in sess.index:
                row = sess.loc[ts]
                close = row["Close"]
                v = vwap.loc[ts]
                s = std.loc[ts]
                r = rsi.loc[ts]
                current_atr = atr.loc[ts]

                if pd.isna(v) or pd.isna(s) or s == 0 or pd.isna(r):
                    continue

                lower_band = v - k * s
                upper_band = v + k * s
                sl_lower = v - (k + sl_k_add) * s
                sl_upper = v + (k + sl_k_add) * s

                # v3: Bollinger Band 寬度檢查
                bb_width = (upper_band - lower_band) / v if v > 0 else 0
                if bb_width < bb_width_min:
                    continue

                # 強制平倉
                if ts >= force_close_ts:
                    if in_long:
                        exits_long[ts] = True
                        in_long = False
                    if in_short:
                        exits_short[ts] = True
                        in_short = False
                    continue

                if not in_long and not in_short:
                    # 每日交易次數限制
                    if daily_trades >= max_trades:
                        continue

                    # 時間窗口限制
                    time_str = ts.strftime("%H:%M")
                    if time_str < entry_start or time_str > entry_end:
                        continue

                    # ATR 過濾
                    if not pd.isna(current_atr):
                        avg_price = close
                        if avg_price > 0 and current_atr / avg_price < atr_min_pct:
                            continue

                    # 成交量過濾
                    if vol_filter:
                        current_vol = row.get("Volume", 0)
                        avg_vol = vol_ma.loc[ts] if ts in vol_ma.index else 0
                        if avg_vol > 0 and current_vol < avg_vol * vol_min_mult:
                            continue

                    # v4: EMA 趨勢方向（支援兩種模式）
                    ema_f = ema_fast_series.loc[ts] if ts in ema_fast_series.index else np.nan
                    ema_s = ema_slow_series.loc[ts] if ts in ema_slow_series.index else np.nan
                    if ema_mode == "price_vs_ema":
                        # v4 新模式：價格 vs EMA20（更寬鬆）
                        is_uptrend = (not pd.isna(ema_f) and close > ema_f)
                        is_downtrend = (not pd.isna(ema_f) and close < ema_f)
                    else:
                        # v3 舊模式：EMA20 vs EMA50 交叉
                        is_uptrend = (not pd.isna(ema_f) and not pd.isna(ema_s) and ema_f > ema_s)
                        is_downtrend = (not pd.isna(ema_f) and not pd.isna(ema_s) and ema_f < ema_s)

                    # 做多入場：價格低於下軌 + RSI 超賣
                    if close < lower_band and r < rsi_os:
                        # v3: 如果開啟 EMA 過濾，只在上升趨勢或無趨勢時做多
                        if ema_trend_filter and is_downtrend:
                            prev_row = row  # v10
                            continue
                        # v8/v10: 反轉 K 棒確認（支援前一根模式）
                        if reversal_confirm:
                            if prev_bar_reversal and prev_row is not None:
                                if not _is_bullish_reversal(prev_row, reversal_mode):
                                    prev_row = row
                                    continue
                            elif not prev_bar_reversal:
                                if not _is_bullish_reversal(row, reversal_mode):
                                    prev_row = row
                                    continue
                        entries_long[ts] = True
                        sl_stop[ts] = sl_lower
                        # v3: 動態 TP
                        if dynamic_tp and is_uptrend:
                            tp_stop[ts] = v + tp_bonus_pct * k * s  # 趨勢方向多賺
                        else:
                            tp_stop[ts] = v
                        in_long = True
                        entry_vwap = v
                        entry_std = s
                        daily_trades += 1
                        if partial_tp:
                            partial_phase = 1
                            partial_bars_held = 0
                    # 做空入場：價格高於上軌 + RSI 超買
                    elif close > upper_band and r > rsi_ob:
                        # v3: 如果開啟 EMA 過濾，只在下降趨勢或無趨勢時做空
                        if ema_trend_filter and is_uptrend:
                            prev_row = row  # v10
                            continue
                        # v8/v10: 反轉 K 棒確認（支援前一根模式）
                        if reversal_confirm:
                            if prev_bar_reversal and prev_row is not None:
                                if not _is_bearish_reversal(prev_row, reversal_mode):
                                    prev_row = row
                                    continue
                            elif not prev_bar_reversal:
                                if not _is_bearish_reversal(row, reversal_mode):
                                    prev_row = row
                                    continue
                        entries_short[ts] = True
                        sl_stop[ts] = sl_upper
                        if dynamic_tp and is_downtrend:
                            tp_stop[ts] = v - tp_bonus_pct * k * s
                        else:
                            tp_stop[ts] = v
                        in_short = True
                        entry_vwap = v
                        entry_std = s
                        daily_trades += 1
                        if partial_tp:
                            partial_phase = 1
                            partial_bars_held = 0
                else:
                    # ── Exit 邏輯 ──
                    # 使用當前 VWAP 和 std（隨 session 變動）
                    sl_lower_now = v - (k + sl_k_add) * s
                    sl_upper_now = v + (k + sl_k_add) * s

                    if partial_tp and partial_phase > 0:
                        partial_bars_held += 1

                    if in_long:
                        # 停損：跌破當前下軌 SL
                        if close <= sl_lower_now:
                            exits_long[ts] = True
                            in_long = False
                            partial_phase = 0
                        elif partial_tp and partial_phase == 1 and close >= v:
                            # Phase 1→2: 到達當前 VWAP，開始 trailing
                            partial_phase = 2
                            partial_best_price = close
                        elif partial_tp and partial_phase == 2:
                            partial_best_price = max(partial_best_price, close)
                            trail_sl = partial_best_price * (1 - partial_tp_trail_pct)
                            if close <= trail_sl or partial_bars_held >= partial_tp_max_hold:
                                exits_long[ts] = True
                                in_long = False
                                partial_phase = 0
                        elif not partial_tp and close >= v:
                            # 無 partial_tp：到達 VWAP 即平倉
                            exits_long[ts] = True
                            in_long = False

                    if in_short:
                        if close >= sl_upper_now:
                            exits_short[ts] = True
                            in_short = False
                            partial_phase = 0
                        elif partial_tp and partial_phase == 1 and close <= v:
                            partial_phase = 2
                            partial_best_price = close
                        elif partial_tp and partial_phase == 2:
                            partial_best_price = min(partial_best_price, close)
                            trail_sl = partial_best_price * (1 + partial_tp_trail_pct)
                            if close >= trail_sl or partial_bars_held >= partial_tp_max_hold:
                                exits_short[ts] = True
                                in_short = False
                                partial_phase = 0
                        elif not partial_tp and close <= v:
                            exits_short[ts] = True
                            in_short = False

                prev_row = row  # v10

        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            metadata={
                "strategy": self.name,
                "params": self.get_params(),
            },
        )
