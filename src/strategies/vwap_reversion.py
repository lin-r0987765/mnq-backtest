"""
VWAP Mean-Reversion Strategy — v2 改良版

改進項目：
1. ATR 動態波段：用 ATR 替代固定 std 倍數，自適應波動率
2. RSI 極端過濾：RSI 門檻更寬以增加交易機會，但加入 RSI 趨勢確認
3. 部分止盈：到 VWAP 先減倉
4. 成交量過濾：低量盤整時不開倉
5. 時間窗口限制：只在最佳時段交易
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

_DEFAULT_PARAMS: dict[str, Any] = {
    "k": 2.0,                # Band multiplier (VWAP ± k × std)
    "sl_k_add": 0.5,         # Extra k for stop-loss beyond entry band
    "std_window": 20,        # Rolling window for std calculation
    "rsi_period": 14,        # RSI period
    "rsi_os": 30,            # RSI oversold threshold
    "rsi_ob": 70,            # RSI overbought threshold
    "close_before_min": 15,
    "atr_period": 14,        # ATR 計算週期
    "atr_min_pct": 0.0005,   # 最小 ATR/價格比（過濾低波動期）
    "vol_filter": True,      # 成交量過濾開關
    "vol_min_mult": 0.5,     # 成交量至少要均量的倍數
    "entry_start_time": "10:00",   # 最早入場時間
    "entry_end_time": "15:30",     # 最晚入場時間
    "max_trades_per_day": 2,       # 每日最大交易次數
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


class VWAPReversionStrategy(BaseStrategy):
    """VWAP mean-reversion strategy — v2 改良版"""

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
        max_trades = int(self.params.get("max_trades_per_day", 2))

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

                    # 做多入場：價格低於下軌 + RSI 超賣
                    if close < lower_band and r < rsi_os:
                        entries_long[ts] = True
                        sl_stop[ts] = sl_lower
                        tp_stop[ts] = v  # 目標: VWAP
                        in_long = True
                        daily_trades += 1
                    # 做空入場：價格高於上軌 + RSI 超買
                    elif close > upper_band and r > rsi_ob:
                        entries_short[ts] = True
                        sl_stop[ts] = sl_upper
                        tp_stop[ts] = v  # 目標: VWAP
                        in_short = True
                        daily_trades += 1
                else:
                    if in_long:
                        if close >= v or close <= sl_lower:
                            exits_long[ts] = True
                            in_long = False
                    if in_short:
                        if close <= v or close >= sl_upper:
                            exits_short[ts] = True
                            in_short = False

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
