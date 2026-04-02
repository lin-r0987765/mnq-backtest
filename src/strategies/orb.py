"""
Opening Range Breakout (ORB) Strategy — v3 保守版

核心改進：
1. 不使用 EMA 趨勢過濾（在 5 分鐘線上會過濾太多好交易）
2. 使用更寬的 opening range（orb_bars=6，30 分鐘）來減少假突破
3. 要求突破幅度超過 ATR 的一定比例（確認突破）
4. 提高止盈倍數，降低交易頻率
5. 使用回測確認的移動止損
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

_DEFAULT_PARAMS: dict[str, Any] = {
    "orb_bars": 6,              # Opening range: 6 × 5min = 30min
    "profit_ratio": 3.0,        # TP = range_width × profit_ratio
    "close_before_min": 15,     # 收盤前強制平倉
    "breakout_confirm_pct": 0.001,  # 突破確認：收盤價需超過 range 邊界的百分比
    "entry_delay_bars": 0,      # 突破後等待幾根 K 棒確認
    "trailing_stop": True,      # 移動止損
    "trailing_pct": 0.005,      # 移動止損百分比（0.5%）
}


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy — v3 保守版"""

    name = "ORB"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**_DEFAULT_PARAMS, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        orb_bars: int = int(self.params["orb_bars"])
        profit_ratio: float = float(self.params["profit_ratio"])
        close_before_min: int = int(self.params["close_before_min"])
        breakout_pct: float = float(self.params.get("breakout_confirm_pct", 0.001))
        trailing_stop: bool = bool(self.params.get("trailing_stop", True))
        trailing_pct: float = float(self.params.get("trailing_pct", 0.005))

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)
        sl_stop = pd.Series(np.nan, index=df.index)
        tp_stop = pd.Series(np.nan, index=df.index)

        for date, session in df.groupby(df.index.date):
            sess = session.between_time("09:30", "16:00")
            if len(sess) < orb_bars + 5:
                continue

            # Opening range
            orb = sess.iloc[:orb_bars]
            orb_high = orb["High"].max()
            orb_low = orb["Low"].min()
            range_width = orb_high - orb_low

            if range_width <= 0:
                continue

            # 過濾太窄的 range（< 0.1% 的價格）
            mid_price = (orb_high + orb_low) / 2
            if range_width / mid_price < 0.001:
                continue

            # 突破確認閾值
            long_entry_level = orb_high * (1 + breakout_pct)
            short_entry_level = orb_low * (1 - breakout_pct)

            tp_long = orb_high + profit_ratio * range_width
            tp_short = orb_low - profit_ratio * range_width

            post_orb = sess.iloc[orb_bars:]
            last_ts = sess.index[-1]
            force_close_ts = last_ts - pd.Timedelta(minutes=close_before_min)

            in_long = False
            in_short = False
            best_price_long = 0.0
            best_price_short = float("inf")
            entry_price = 0.0

            for ts, row in post_orb.iterrows():
                close = row["Close"]

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
                    # 做多：收盤價突破 opening range 高點（帶確認）
                    if close > long_entry_level:
                        entries_long[ts] = True
                        sl_stop[ts] = orb_low
                        tp_stop[ts] = tp_long
                        in_long = True
                        best_price_long = close
                        entry_price = close
                    # 做空：收盤價跌破 opening range 低點（帶確認）
                    elif close < short_entry_level:
                        entries_short[ts] = True
                        sl_stop[ts] = orb_high
                        tp_stop[ts] = tp_short
                        in_short = True
                        best_price_short = close
                        entry_price = close
                else:
                    if in_long:
                        best_price_long = max(best_price_long, close)
                        # 移動止損
                        if trailing_stop:
                            trail_sl = best_price_long * (1 - trailing_pct)
                            effective_sl = max(orb_low, trail_sl)
                        else:
                            effective_sl = orb_low

                        if close <= effective_sl or close >= tp_long:
                            exits_long[ts] = True
                            in_long = False

                    if in_short:
                        best_price_short = min(best_price_short, close)
                        if trailing_stop:
                            trail_sl = best_price_short * (1 + trailing_pct)
                            effective_sl = min(orb_high, trail_sl)
                        else:
                            effective_sl = orb_high

                        if close >= effective_sl or close <= tp_short:
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
