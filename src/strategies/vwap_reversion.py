"""
VWAP Mean-Reversion Strategy.

Rules
-----
* VWAP is reset each trading day at 09:30.
* Rolling std window controls the "band width".
* Long entry  : close < VWAP − k × std  AND  RSI(rsi_period) < rsi_os
  Stop-loss    : VWAP − (k + sl_k_add) × std
  Target       : VWAP (mid-band)
* Short entry : close > VWAP + k × std  AND  RSI(rsi_period) > rsi_ob
  Stop-loss    : VWAP + (k + sl_k_add) × std
  Target       : VWAP (mid-band)
* Force-close : last `close_before_min` minutes of regular session.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

_DEFAULT_PARAMS: dict[str, Any] = {
    "k": 1.5,             # Band multiplier (VWAP ± k × std)
    "sl_k_add": 0.5,      # Extra k for stop-loss beyond entry band
    "std_window": 20,     # Rolling window for std calculation
    "rsi_period": 14,     # RSI period
    "rsi_os": 35,         # RSI oversold threshold (long entry)
    "rsi_ob": 65,         # RSI overbought threshold (short entry)
    "close_before_min": 15,
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

    # Rolling std of typical price (not VWAP-normalised) for band width
    std = typical.rolling(std_window, min_periods=2).std()
    return vwap, std


class VWAPReversionStrategy(BaseStrategy):
    """VWAP mean-reversion strategy."""

    name = "VWAP_Reversion"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**_DEFAULT_PARAMS, **(params or {})})

    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        k: float = float(self.params["k"])
        sl_k_add: float = float(self.params["sl_k_add"])
        std_window: int = int(self.params["std_window"])
        rsi_period: int = int(self.params["rsi_period"])
        rsi_os: float = float(self.params["rsi_os"])
        rsi_ob: float = float(self.params["rsi_ob"])
        close_before_min: int = int(self.params["close_before_min"])

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)
        sl_stop = pd.Series(np.nan, index=df.index)
        tp_stop = pd.Series(np.nan, index=df.index)

        # Compute RSI across entire dataset (so each session has warmup)
        rsi_full = _compute_rsi(df["Close"], rsi_period)

        for date, session in df.groupby(df.index.date):
            sess = session.between_time("09:30", "16:00")
            if len(sess) < std_window + 2:
                continue

            vwap, std = _compute_vwap_and_std(sess, std_window)
            rsi = rsi_full.reindex(sess.index)

            force_close_ts = sess.index[-1] - pd.Timedelta(minutes=close_before_min)

            in_long = False
            in_short = False
            entry_vwap = np.nan

            for ts in sess.index:
                row = sess.loc[ts]
                close = row["Close"]
                v = vwap.loc[ts]
                s = std.loc[ts]
                r = rsi.loc[ts]

                if pd.isna(v) or pd.isna(s) or s == 0 or pd.isna(r):
                    continue

                lower_band = v - k * s
                upper_band = v + k * s
                sl_lower = v - (k + sl_k_add) * s
                sl_upper = v + (k + sl_k_add) * s

                # Force close
                if ts >= force_close_ts:
                    if in_long:
                        exits_long[ts] = True
                        in_long = False
                    if in_short:
                        exits_short[ts] = True
                        in_short = False
                    continue

                if not in_long and not in_short:
                    if close < lower_band and r < rsi_os:
                        entries_long[ts] = True
                        sl_stop[ts] = sl_lower
                        tp_stop[ts] = v  # target = VWAP
                        entry_vwap = v
                        in_long = True
                    elif close > upper_band and r > rsi_ob:
                        entries_short[ts] = True
                        sl_stop[ts] = sl_upper
                        tp_stop[ts] = v  # target = VWAP
                        entry_vwap = v
                        in_short = True
                else:
                    if in_long:
                        # Exit: price reaches VWAP or stop-loss hit
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
