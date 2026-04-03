"""
Opening Range Breakout (ORB) Strategy — v5 成交量+ATR版

核心改進（迭代 #5）：
1. 成交量確認：突破 K 棒成交量需 > 20 期均量（減少假突破）
2. ATR 動態 trailing stop：取代固定 1.5%，根據市場波動自適應
3. 保留 v4 最佳參數基底：orb_bars=4, profit_ratio=3.5, breakout_confirm_pct=0.0003
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

_DEFAULT_PARAMS: dict[str, Any] = {
    "orb_bars": 4,              # Opening range: 4 × 5min = 20min
    "profit_ratio": 3.5,        # TP = range_width × profit_ratio
    "close_before_min": 15,     # 收盤前強制平倉
    "breakout_confirm_pct": 0.0003,  # 突破確認 0.03%
    "entry_delay_bars": 0,      # 突破後等待幾根 K 棒確認
    "trailing_stop": True,      # 移動止損
    "trailing_pct": 0.015,      # 固定移動止損百分比（fallback）
    # v5 新增（v6: 默認關閉 — grid search 確認不啟用時更優）
    "vol_confirm": False,       # 成交量確認開關 — grid最佳為False
    "vol_ma_period": 20,        # 成交量均線週期
    "vol_mult": 0.6,            # 突破時成交量需 >= 均量 × vol_mult
    "atr_trailing": False,      # ATR 動態 trailing stop — grid最佳為False
    "atr_period": 14,           # ATR 計算週期
    "atr_trail_mult": 1.5,      # trailing stop = best_price ± ATR × mult
}


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy — v6 默認參數修正版"""

    name = "ORB"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**_DEFAULT_PARAMS, **(params or {})})

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        orb_bars: int = int(self.params["orb_bars"])
        profit_ratio: float = float(self.params["profit_ratio"])
        close_before_min: int = int(self.params["close_before_min"])
        breakout_pct: float = float(self.params.get("breakout_confirm_pct", 0.001))
        trailing_stop: bool = bool(self.params.get("trailing_stop", True))
        trailing_pct: float = float(self.params.get("trailing_pct", 0.015))
        # v5 新增
        vol_confirm: bool = bool(self.params.get("vol_confirm", True))
        vol_ma_period: int = int(self.params.get("vol_ma_period", 20))
        vol_mult: float = float(self.params.get("vol_mult", 1.0))
        atr_trailing: bool = bool(self.params.get("atr_trailing", True))
        atr_period: int = int(self.params.get("atr_period", 14))
        atr_trail_mult: float = float(self.params.get("atr_trail_mult", 2.0))

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)
        sl_stop = pd.Series(np.nan, index=df.index)
        tp_stop = pd.Series(np.nan, index=df.index)

        # v5: 預計算成交量均線
        vol_ma = df["Volume"].rolling(vol_ma_period, min_periods=1).mean()

        # v5: 預計算 ATR
        high_s = df["High"]
        low_s = df["Low"]
        close_s = df["Close"]
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s - close_s.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_full = tr.rolling(atr_period, min_periods=1).mean()

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
                    # v5: 成交量確認
                    if vol_confirm:
                        cur_vol = row.get("Volume", 0)
                        avg_vol = vol_ma.loc[ts] if ts in vol_ma.index else 0
                        if avg_vol > 0 and cur_vol < avg_vol * vol_mult:
                            continue  # 成交量不足，跳過

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
                    # v5: ATR 動態 trailing stop
                    current_atr = atr_full.loc[ts] if ts in atr_full.index else 0

                    if in_long:
                        best_price_long = max(best_price_long, close)
                        if trailing_stop:
                            if atr_trailing and current_atr > 0:
                                trail_sl = best_price_long - atr_trail_mult * current_atr
                            else:
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
                            if atr_trailing and current_atr > 0:
                                trail_sl = best_price_short + atr_trail_mult * current_atr
                            else:
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
