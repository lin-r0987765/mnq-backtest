"""
Opening Range Breakout (ORB) Strategy.

Rules
-----
* Opening range = first `orb_bars` × 5-minute bars after 09:30 US/Eastern.
* Long entry  : close crosses above range high  → stop-loss at range low,
                take-profit at high + profit_ratio × range_width.
* Short entry : close crosses below range low   → stop-loss at range high,
                take-profit at low − profit_ratio × range_width.
* Force-close : any open position in the last `close_before_min` minutes
                of the regular session (default 15 min → 15:45).
* Only one direction trade per day.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

_DEFAULT_PARAMS: dict[str, Any] = {
    "orb_bars": 3,          # Number of 5-min bars for the opening range (3 → 15 min)
    "profit_ratio": 2.0,    # Take-profit = range_width × profit_ratio
    "close_before_min": 15, # Minutes before close to force-flatten
}


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy."""

    name = "ORB"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**_DEFAULT_PARAMS, **(params or {})})

    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """
        Accepts a multi-day OHLCV DataFrame (5-min bars, US/Eastern).
        Generates entry / exit signals across all sessions.
        """
        orb_bars: int = int(self.params["orb_bars"])
        profit_ratio: float = float(self.params["profit_ratio"])
        close_before_min: int = int(self.params["close_before_min"])

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)
        sl_stop = pd.Series(np.nan, index=df.index)
        tp_stop = pd.Series(np.nan, index=df.index)

        # Process each trading day independently
        for date, session in df.groupby(df.index.date):
            # Regular session only
            sess = session.between_time("09:30", "16:00")
            if len(sess) < orb_bars + 2:
                continue

            # Opening range
            orb = sess.iloc[:orb_bars]
            orb_high = orb["High"].max()
            orb_low = orb["Low"].min()
            range_width = orb_high - orb_low

            if range_width <= 0:
                continue

            tp_long = orb_high + profit_ratio * range_width
            tp_short = orb_low - profit_ratio * range_width

            # Bars after the opening range
            post_orb = sess.iloc[orb_bars:]

            # Force-close timestamp
            last_ts = sess.index[-1]
            force_close_ts = last_ts - pd.Timedelta(minutes=close_before_min)

            in_long = False
            in_short = False

            for i, (ts, row) in enumerate(post_orb.iterrows()):
                close = row["Close"]

                # Force-close before end of day
                if ts >= force_close_ts:
                    if in_long:
                        exits_long[ts] = True
                        in_long = False
                    if in_short:
                        exits_short[ts] = True
                        in_short = False
                    continue

                if not in_long and not in_short:
                    # Entry signals (one trade per day)
                    if close > orb_high:
                        entries_long[ts] = True
                        sl_stop[ts] = orb_low
                        tp_stop[ts] = tp_long
                        in_long = True
                    elif close < orb_low:
                        entries_short[ts] = True
                        sl_stop[ts] = orb_high
                        tp_stop[ts] = tp_short
                        in_short = True
                else:
                    # Exit management
                    if in_long:
                        if close <= orb_low or close >= tp_long:
                            exits_long[ts] = True
                            in_long = False
                    if in_short:
                        if close >= orb_high or close <= tp_short:
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
