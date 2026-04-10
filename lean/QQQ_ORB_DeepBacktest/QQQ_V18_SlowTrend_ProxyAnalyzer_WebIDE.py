from AlgorithmImports import *

from collections import deque
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd


class QQQV18SlowTrendProxyAnalyzerWebIDE(QCAlgorithm):
    SYMBOL = "QQQ"
    SCRIPT_VERSION = "v18-slowtrend-proxy-analyzer"
    INITIAL_CASH = 100000
    START = (2017, 4, 3)
    END = (2026, 4, 2)
    ANALYSIS_INITIAL_CAPITAL = 100000.0

    POSITION_SIZE_PCT = 0.25
    MIN_TRADE_QUANTITY = 1
    MAX_TRADE_QUANTITY = 500
    ORB_BARS = 4
    PROFIT_RATIO = 3.5
    BREAKOUT_CONFIRM_PCT = 0.0003
    ENTRY_DELAY_BARS = 0
    TRAILING_PCT = 0.013
    EARLY_TIGHT_TRAIL_PCT = 0.013
    EARLY_TIGHT_TRAIL_MINUTES = 0
    CLOSE_BEFORE_MIN = 10
    MAX_ENTRIES_PER_SESSION = 1
    ENTRY_START_HOUR_UTC = 0
    ENTRY_END_HOUR_UTC = 17
    MIN_RANGE_PCT = 0.001
    HTF_FILTER = True
    HTF_MODE = "slope"
    HTF_EMA_FAST = 20
    HTF_EMA_SLOW = 30
    SKIP_SHORT_AFTER_UP_DAYS = 2
    SKIP_LONG_AFTER_UP_DAYS = 3
    MULTI_DAY_RANGE = False
    MULTI_DAY_LOOKBACK = 2
    REGIME_FILTER = True
    REGIME_MODE = "prev_day_up_and_mom3_positive"
    REGIME_ALLOW_SHORTS = False
    REGIME_MIN_HISTORY_DAYS = 4
    CANDIDATE_NAMES = ("mom5_positive", "mom7_positive", "mom10_positive", "close_above_sma5", "close_above_sma8", "close_above_sma10", "close_above_sma15", "close_above_sma20")

    def initialize(self) -> None:
        self.set_time_zone(TimeZones.NEW_YORK)
        self.set_start_date(*self.START)
        self.set_end_date(*self.END)
        self.set_cash(float(self.INITIAL_CASH))
        self.settings.seed_initial_prices = True

        equity = self.add_equity(
            self.SYMBOL,
            Resolution.MINUTE,
            fill_forward=False,
            extended_market_hours=False,
            data_normalization_mode=DataNormalizationMode.ADJUSTED,
        )
        self.symbol = equity.symbol
        self.set_benchmark(self.symbol)
        self.daily_feature_frame = pd.DataFrame()
        self.analysis_calendar_dates = []
        self.daily_feature_map = {}
        self.analysis_trades = []
        self.open_trade = None

        self.orb_history = deque(maxlen=max(self.MULTI_DAY_LOOKBACK, 5))
        self.current_session_date = None
        self.current_session_open = None
        self.current_session_close = None
        self.raw_orb_high = None
        self.raw_orb_low = None
        self.orb_high = None
        self.orb_low = None
        self.range_width = None
        self.tp_long = None
        self.tp_short = None
        self.long_entry_level = None
        self.short_entry_level = None
        self.bars_from_open = 0
        self.hourly_bias = 0
        self.last_fast_ema = None
        self.up_day_streak = 0
        self.best_price_long = None
        self.best_price_short = None
        self.position_entry_time = None
        self.entry_count = 0
        self.exit_count = 0
        self.session_entry_count = 0
        self.reentry_blocked_count = 0
        self.total_shares_traded = 0
        self.current_session_market_close = None
        self.half_day_session_count = 0
        self.regime_allow_long_today = True
        self.regime_allow_short_today = True
        self.regime_label = "disabled"
        self.regime_pass_days = 0
        self.regime_block_days = 0
        self.regime_history_shortage_days = 0

        self.fast_ema = ExponentialMovingAverage(self.HTF_EMA_FAST)
        self.slow_ema = ExponentialMovingAverage(self.HTF_EMA_SLOW)

        self.five_minute_consolidator = TradeBarConsolidator(timedelta(minutes=5))
        self.five_minute_consolidator.data_consolidated += self.on_five_minute_bar
        self.subscription_manager.add_consolidator(self.symbol, self.five_minute_consolidator)

        self.hour_consolidator = TradeBarConsolidator(timedelta(hours=1))
        self.hour_consolidator.data_consolidated += self.on_hour_bar
        self.subscription_manager.add_consolidator(self.symbol, self.hour_consolidator)
        self.register_indicator(self.symbol, self.fast_ema, self.hour_consolidator)
        self.register_indicator(self.symbol, self.slow_ema, self.hour_consolidator)

        self.schedule.on(
            self.date_rules.every_day(self.symbol),
            self.time_rules.before_market_close(self.symbol, self.CLOSE_BEFORE_MIN),
            self.force_flatten,
        )

        self.set_warm_up(timedelta(days=20), Resolution.MINUTE)
        self.debug(
            "QQQ slow-trend proxy analyzer init | "
            f"version={self.SCRIPT_VERSION} | symbol={self.SYMBOL} | "
            f"regime_mode={self.REGIME_MODE} | candidates={len(self.CANDIDATE_NAMES)}"
        )

    @property
    def TRADE_QUANTITY(self) -> int:
        """Dynamically compute position size based on portfolio value and current price."""
        price = self.securities[self.symbol].price
        if price <= 0:
            return self.MIN_TRADE_QUANTITY
        qty = int(self.portfolio.total_portfolio_value * self.POSITION_SIZE_PCT / price)
        return max(self.MIN_TRADE_QUANTITY, min(qty, self.MAX_TRADE_QUANTITY))

    def on_hour_bar(self, sender, bar: TradeBar) -> None:
        if not self.fast_ema.is_ready:
            return

        fast_now = float(self.fast_ema.current.value)
        slow_now = float(self.slow_ema.current.value) if self.slow_ema.is_ready else fast_now

        if self.HTF_MODE == "slope":
            if self.last_fast_ema is None:
                self.hourly_bias = 0
            elif fast_now > self.last_fast_ema:
                self.hourly_bias = 1
            elif fast_now < self.last_fast_ema:
                self.hourly_bias = -1
            else:
                self.hourly_bias = 0
        else:
            if fast_now > slow_now:
                self.hourly_bias = 1
            elif fast_now < slow_now:
                self.hourly_bias = -1
            else:
                self.hourly_bias = 0

        self.last_fast_ema = fast_now

    def on_five_minute_bar(self, sender, bar: TradeBar) -> None:
        if self.is_warming_up:
            return

        if not self.securities[self.symbol].exchange.hours.is_open(bar.end_time, False):
            return

        bar_date = bar.end_time.date()
        if self.current_session_date != bar_date:
            self._start_new_session(bar)
        else:
            self.current_session_close = bar.close
            self.bars_from_open += 1

        self._update_orb(bar)
        self._manage_open_position(bar)

        if self._entry_window_closed(bar.end_time):
            return
        if self.portfolio[self.symbol].invested:
            return
        if self.range_width is None or self.range_width <= 0:
            return
        if self.bars_from_open <= self.ORB_BARS + self.ENTRY_DELAY_BARS:
            return
        if not self._entry_time_filter_passed():
            return

        allow_short_today = not (
            self.SKIP_SHORT_AFTER_UP_DAYS > 0 and self.up_day_streak >= self.SKIP_SHORT_AFTER_UP_DAYS
        )
        allow_long_today = not (
            self.SKIP_LONG_AFTER_UP_DAYS > 0 and self.up_day_streak >= self.SKIP_LONG_AFTER_UP_DAYS
        )
        allow_long_today = allow_long_today and self.regime_allow_long_today
        allow_short_today = allow_short_today and self.regime_allow_short_today

        if bar.close > self.long_entry_level:
            if not allow_long_today:
                return
            if self.HTF_FILTER and self.hourly_bias == -1:
                return
            if not self._can_take_session_entry():
                return
            qty = self.TRADE_QUANTITY
            self.market_order(self.symbol, qty, tag=f"ORB Long qty={qty}")
            self.best_price_long = bar.close
            self.best_price_short = None
            self.position_entry_time = bar.end_time
            self.entry_count += 1
            self.session_entry_count += 1
            self.total_shares_traded += qty
            return

        if bar.close < self.short_entry_level:
            if not allow_short_today:
                return
            if self.HTF_FILTER and self.hourly_bias == 1:
                return
            if not self._can_take_session_entry():
                return
            qty = self.TRADE_QUANTITY
            self.market_order(self.symbol, -qty, tag=f"ORB Short qty={qty}")
            self.best_price_short = bar.close
            self.best_price_long = None
            self.position_entry_time = bar.end_time
            self.entry_count += 1
            self.session_entry_count += 1
            self.total_shares_traded += qty

    def on_order_event(self, order_event: OrderEvent) -> None:
        if order_event.symbol != self.symbol:
            return
        if order_event.status != OrderStatus.FILLED:
            return
        if order_event.fill_quantity == 0:
            return

        timestamp = pd.Timestamp(order_event.utc_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        quantity = int(order_event.fill_quantity)
        fill_price = float(order_event.fill_price)
        fee_amount = self._extract_order_fee(order_event)

        if quantity > 0:
            if self.open_trade is None:
                self.open_trade = {
                    "side": "long",
                    "entry_time": timestamp,
                    "entry_price": fill_price,
                    "quantity": abs(quantity),
                    "entry_fee": fee_amount,
                }
            elif self.open_trade["side"] == "short":
                trade = self.open_trade
                pnl = (float(trade["entry_price"]) - fill_price) * float(trade["quantity"]) - float(
                    trade["entry_fee"]
                ) - fee_amount
                self.analysis_trades.append(
                    {
                        "entry_time": trade["entry_time"],
                        "exit_time": timestamp,
                        "entry_date": pd.Timestamp(trade["entry_time"]).tz_convert("America/New_York").date(),
                        "exit_date": timestamp.tz_convert("America/New_York").date(),
                        "entry_year": int(pd.Timestamp(trade["entry_time"]).tz_convert("America/New_York").year),
                        "net_pnl": float(pnl),
                        "is_win_net": int(pnl > 0.0),
                        "is_long": 0,
                        "is_short": 1,
                    }
                )
                self.open_trade = None
        else:
            if self.open_trade is not None and self.open_trade["side"] == "long":
                trade = self.open_trade
                pnl = (fill_price - float(trade["entry_price"])) * float(trade["quantity"]) - float(
                    trade["entry_fee"]
                ) - fee_amount
                self.analysis_trades.append(
                    {
                        "entry_time": trade["entry_time"],
                        "exit_time": timestamp,
                        "entry_date": pd.Timestamp(trade["entry_time"]).tz_convert("America/New_York").date(),
                        "exit_date": timestamp.tz_convert("America/New_York").date(),
                        "entry_year": int(pd.Timestamp(trade["entry_time"]).tz_convert("America/New_York").year),
                        "net_pnl": float(pnl),
                        "is_win_net": int(pnl > 0.0),
                        "is_long": 1,
                        "is_short": 0,
                    }
                )
                self.open_trade = None
            elif self.open_trade is None:
                self.open_trade = {
                    "side": "short",
                    "entry_time": timestamp,
                    "entry_price": fill_price,
                    "quantity": abs(quantity),
                    "entry_fee": fee_amount,
                }

    def force_flatten(self) -> None:
        if self.portfolio[self.symbol].invested:
            self.liquidate(self.symbol, "ORB EOD Flatten")
            self.exit_count += 1
            self.position_entry_time = None

    def on_end_of_algorithm(self) -> None:
        avg_qty = round(self.total_shares_traded / self.entry_count, 1) if self.entry_count > 0 else 0
        self.debug(
            "QQQ slow-trend proxy analyzer summary | "
            f"version={self.SCRIPT_VERSION} | entries={self.entry_count} | exits={self.exit_count} | "
            f"avg_qty={avg_qty} | half_days={self.half_day_session_count} | "
            f"regime_pass_days={self.regime_pass_days} | regime_block_days={self.regime_block_days} | "
            f"reentry_blocked={self.reentry_blocked_count}"
        )
        self._emit_proxy_analysis()

    def _start_new_session(self, bar: TradeBar) -> None:
        if self.current_session_open is not None and self.current_session_close is not None:
            if self.current_session_close > self.current_session_open:
                self.up_day_streak += 1
            else:
                self.up_day_streak = 0

        if self.raw_orb_high is not None and self.raw_orb_low is not None:
            self.orb_history.append((self.raw_orb_high, self.raw_orb_low))

        self.current_session_date = bar.end_time.date()
        self.current_session_open = bar.open
        self.current_session_close = bar.close
        self.session_entry_count = 0
        self.current_session_market_close = self.securities[self.symbol].exchange.hours.get_next_market_close(
            bar.end_time,
            False,
        )
        if self.current_session_market_close.hour != 16 or self.current_session_market_close.minute != 0:
            self.half_day_session_count += 1
        self._update_regime_state()
        self.raw_orb_high = bar.high
        self.raw_orb_low = bar.low
        self.orb_high = None
        self.orb_low = None
        self.range_width = None
        self.tp_long = None
        self.tp_short = None
        self.long_entry_level = None
        self.short_entry_level = None
        self.bars_from_open = 1
        self.best_price_long = None
        self.best_price_short = None
        self.position_entry_time = None

    def _update_regime_state(self) -> None:
        if not self.REGIME_FILTER:
            self.regime_allow_long_today = True
            self.regime_allow_short_today = True
            self.regime_label = "disabled"
            return

        lookback = max(self.REGIME_MIN_HISTORY_DAYS + 3, 5)
        history = self.history(self.symbol, lookback, Resolution.DAILY)
        if history.empty:
            self.regime_allow_long_today = False
            self.regime_allow_short_today = False
            self.regime_label = "history_empty"
            self.regime_history_shortage_days += 1
            return

        if isinstance(history.index, pd.MultiIndex):
            time_level_name = history.index.names[-1] or "time"
            history = history.reset_index().sort_values(time_level_name).reset_index(drop=True)
            time_index = history[time_level_name]
        else:
            history = history.sort_index().reset_index()
            time_level_name = history.columns[0]
            time_index = history[time_level_name]

        if len(history) > 0 and pd.Timestamp(time_index.iloc[-1]).date() >= self.current_session_date:
            history = history.iloc[:-1]

        close_s = history["close"].astype(float).reset_index(drop=True)
        min_ready = self.REGIME_MIN_HISTORY_DAYS
        if len(close_s) < min_ready:
            self.regime_allow_long_today = False
            self.regime_allow_short_today = False
            self.regime_label = "history_short"
            self.regime_history_shortage_days += 1
            return

        daily_ret = close_s.pct_change()
        prev_day_return = float(daily_ret.iloc[-1]) if pd.notna(daily_ret.iloc[-1]) else np.nan
        mom3_return_s = close_s.pct_change(3)
        mom3_return = float(mom3_return_s.iloc[-1]) if pd.notna(mom3_return_s.iloc[-1]) else np.nan
        prev_day_up = bool(pd.notna(prev_day_return) and prev_day_return > 0.0)
        mom3_positive = bool(pd.notna(mom3_return) and mom3_return > 0.0)
        allow_long = prev_day_up and mom3_positive

        self.regime_allow_long_today = allow_long
        self.regime_allow_short_today = bool(self.REGIME_ALLOW_SHORTS and (not allow_long))
        if pd.notna(prev_day_return) and pd.notna(mom3_return):
            self.regime_label = (
                f"prev_day_up={int(prev_day_up)}|mom3_positive={int(mom3_positive)}|"
                f"prev_day_return_pct={prev_day_return * 100.0:.3f}|mom3_return_pct={mom3_return * 100.0:.3f}"
            )
        else:
            self.regime_label = "prev_day_up=0|mom3_positive=0|prev_day_return_pct=nan|mom3_return_pct=nan"
        if self.regime_allow_long_today or self.regime_allow_short_today:
            self.regime_pass_days += 1
        else:
            self.regime_block_days += 1

    def _update_orb(self, bar: TradeBar) -> None:
        if self.bars_from_open <= self.ORB_BARS:
            self.raw_orb_high = max(self.raw_orb_high, bar.high) if self.raw_orb_high is not None else bar.high
            self.raw_orb_low = min(self.raw_orb_low, bar.low) if self.raw_orb_low is not None else bar.low

        if self.raw_orb_high is None or self.raw_orb_low is None:
            return

        orb_high = self.raw_orb_high
        orb_low = self.raw_orb_low

        if self.MULTI_DAY_RANGE and len(self.orb_history) > 0:
            lookback = min(self.MULTI_DAY_LOOKBACK, len(self.orb_history))
            recent = list(self.orb_history)[-lookback:]
            orb_high = max([high for high, _ in recent] + [orb_high])
            orb_low = min([low for _, low in recent] + [orb_low])

        range_width = orb_high - orb_low
        mid_price = (orb_high + orb_low) / 2.0 if (orb_high + orb_low) != 0 else 0
        if range_width <= 0 or mid_price <= 0:
            return
        if (range_width / mid_price) < self.MIN_RANGE_PCT:
            return

        self.orb_high = orb_high
        self.orb_low = orb_low
        self.range_width = range_width
        self.long_entry_level = orb_high * (1.0 + self.BREAKOUT_CONFIRM_PCT)
        self.short_entry_level = orb_low * (1.0 - self.BREAKOUT_CONFIRM_PCT)
        self.tp_long = orb_high + self.PROFIT_RATIO * range_width
        self.tp_short = orb_low - self.PROFIT_RATIO * range_width

    def _manage_open_position(self, bar: TradeBar) -> None:
        holding = self.portfolio[self.symbol]
        if not holding.invested:
            self.position_entry_time = None
            return
        if self.range_width is None:
            return

        trailing_pct = self._active_trailing_pct(bar.end_time)
        if holding.is_long:
            self.best_price_long = max(self.best_price_long or bar.close, bar.close)
            trail_stop = self.best_price_long * (1.0 - trailing_pct)
            effective_stop = max(self.orb_low, trail_stop) if self.orb_low is not None else trail_stop
            if bar.close <= effective_stop:
                self.liquidate(self.symbol, "ORB Long Stop")
                self.exit_count += 1
                self.position_entry_time = None
            elif self.tp_long is not None and bar.close >= self.tp_long:
                self.liquidate(self.symbol, "ORB Long TP")
                self.exit_count += 1
                self.position_entry_time = None
        elif holding.is_short:
            self.best_price_short = min(self.best_price_short or bar.close, bar.close)
            trail_stop = self.best_price_short * (1.0 + trailing_pct)
            effective_stop = min(self.orb_high, trail_stop) if self.orb_high is not None else trail_stop
            if bar.close >= effective_stop:
                self.liquidate(self.symbol, "ORB Short Stop")
                self.exit_count += 1
                self.position_entry_time = None
            elif self.tp_short is not None and bar.close <= self.tp_short:
                self.liquidate(self.symbol, "ORB Short TP")
                self.exit_count += 1
                self.position_entry_time = None

    def _active_trailing_pct(self, timestamp) -> float:
        if (
            self.position_entry_time is not None
            and self.EARLY_TIGHT_TRAIL_PCT < self.TRAILING_PCT
            and self.EARLY_TIGHT_TRAIL_MINUTES > 0
            and timestamp - self.position_entry_time <= timedelta(minutes=self.EARLY_TIGHT_TRAIL_MINUTES)
        ):
            return self.EARLY_TIGHT_TRAIL_PCT
        return self.TRAILING_PCT

    def _can_take_session_entry(self) -> bool:
        if self.session_entry_count >= self.MAX_ENTRIES_PER_SESSION:
            self.reentry_blocked_count += 1
            return False
        return True

    def _entry_time_filter_passed(self) -> bool:
        h = self.utc_time.hour
        return self.ENTRY_START_HOUR_UTC <= h < self.ENTRY_END_HOUR_UTC

    def _entry_window_closed(self, timestamp) -> bool:
        if self.current_session_market_close is None:
            return False
        return timestamp >= self.current_session_market_close - timedelta(minutes=self.CLOSE_BEFORE_MIN)

    def _extract_order_fee(self, order_event: OrderEvent) -> float:
        try:
            return abs(float(order_event.order_fee.value.amount))
        except Exception:
            return 0.0

    def _build_daily_feature_frame(self) -> pd.DataFrame:
        start = pd.Timestamp(datetime(self.START[0], self.START[1], self.START[2]), tz="UTC")
        end_exclusive = pd.Timestamp(self.time)
        if end_exclusive.tzinfo is None:
            end_exclusive = end_exclusive.tz_localize("UTC")
        else:
            end_exclusive = end_exclusive.tz_convert("UTC")
        end_exclusive = end_exclusive + pd.Timedelta(days=1)
        history = self.history(self.symbol, start, end_exclusive, Resolution.DAILY)
        if history.empty:
            return pd.DataFrame(columns=["date", "prev_day_up", "mom3_positive", *self.CANDIDATE_NAMES])

        if isinstance(history.index, pd.MultiIndex):
            time_level_name = history.index.names[-1] or "time"
            frame = history.reset_index().sort_values(time_level_name).reset_index(drop=True)
        else:
            frame = history.sort_index().reset_index()
            time_level_name = frame.columns[0]

        frame[time_level_name] = pd.to_datetime(frame[time_level_name], utc=True, errors="coerce")
        frame = frame[frame[time_level_name].notna()].copy()
        frame["market_date"] = frame[time_level_name].dt.tz_convert("America/New_York").dt.date
        close = frame["close"].astype(float)
        frame["prev_day_up"] = close.pct_change() > 0.0
        frame["mom3_positive"] = close.pct_change(3) > 0.0
        frame["mom5_positive"] = close.pct_change(5) > 0.0
        frame["mom7_positive"] = close.pct_change(7) > 0.0
        frame["mom10_positive"] = close.pct_change(10) > 0.0
        frame["close_above_sma5"] = close > close.rolling(5).mean()
        frame["close_above_sma8"] = close > close.rolling(8).mean()
        frame["close_above_sma10"] = close > close.rolling(10).mean()
        frame["close_above_sma15"] = close > close.rolling(15).mean()
        frame["close_above_sma20"] = close > close.rolling(20).mean()
        frame["date"] = frame["market_date"].shift(-1)
        frame = frame[frame["date"].notna()].copy()
        return frame[
            [
                "date",
                "prev_day_up",
                "mom3_positive",
                "mom5_positive",
                "mom7_positive",
                "mom10_positive",
                "close_above_sma5",
                "close_above_sma8",
                "close_above_sma10",
                "close_above_sma15",
                "close_above_sma20",
            ]
        ].copy()

    def _compute_profit_factor(self, net_pnl: pd.Series) -> float:
        gross_profit = float(net_pnl[net_pnl > 0].sum())
        gross_loss = float(-net_pnl[net_pnl < 0].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _rolling_summary(self, calendar_dates: list[object], subset: pd.DataFrame, months: int) -> dict:
        daily = pd.DataFrame({"date": pd.to_datetime(calendar_dates)})
        if len(subset):
            pnl_by_day = subset.groupby("exit_date")["net_pnl"].sum().rename("net_pnl")
            daily = daily.merge(
                pnl_by_day,
                left_on=daily["date"].dt.date,
                right_index=True,
                how="left",
            ).fillna({"net_pnl": 0.0})
        else:
            daily["net_pnl"] = 0.0
        daily["equity"] = self.ANALYSIS_INITIAL_CAPITAL + daily["net_pnl"].cumsum()
        daily["ret"] = daily["equity"].pct_change().fillna(0.0)
        month_ends = daily.set_index("date")["equity"].resample("ME").last().dropna().index
        rows = []
        for end in month_ends:
            start = (pd.Timestamp(end) - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
            win = daily[(daily["date"] >= start) & (daily["date"] <= end)]
            if len(win) < 40:
                continue
            trades_win = subset[
                (pd.to_datetime(subset["exit_date"]) >= start) & (pd.to_datetime(subset["exit_date"]) <= end)
            ]
            pf = self._compute_profit_factor(trades_win["net_pnl"]) if len(trades_win) else 0.0
            std = float(win["ret"].std(ddof=0))
            sharpe = 0.0 if std == 0 else float(np.sqrt(252.0) * win["ret"].mean() / std)
            net = float((win["equity"].iloc[-1] / win["equity"].iloc[0] - 1.0) * 100.0)
            rows.append({"sharpe": sharpe, "net": net, "pf": pf})

        out = pd.DataFrame(rows)
        finite_pf = out["pf"].replace([np.inf, -np.inf], np.nan) if len(out) else pd.Series(dtype=float)
        return {
            "window_count": int(len(out)),
            "positive_sharpe_pct": round(float((out["sharpe"] > 0).mean() * 100.0), 1) if len(out) else 0.0,
            "median_sharpe": round(float(out["sharpe"].median()), 3) if len(out) else 0.0,
            "median_net_profit_pct": round(float(out["net"].median()), 3) if len(out) else 0.0,
            "median_profit_factor": round(float(finite_pf.median()), 3) if finite_pf.notna().any() else 0.0,
        }

    def _stats_for_subset(self, subset: pd.DataFrame) -> dict:
        if subset.empty:
            return {
                "trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "positive_years": 0,
                "negative_years": 0,
                "rolling_6m_positive_pct": 0.0,
                "rolling_12m_positive_pct": 0.0,
            }
        by_year = subset.groupby("entry_year")["net_pnl"].sum()
        pf = self._compute_profit_factor(subset["net_pnl"])
        return {
            "trades": int(len(subset)),
            "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2),
            "profit_factor": round(float(pf), 3) if np.isfinite(pf) else float("inf"),
            "net_pnl": round(float(subset["net_pnl"].sum()), 2),
            "positive_years": int((by_year > 0).sum()),
            "negative_years": int((by_year < 0).sum()),
            "rolling_6m_positive_pct": self._rolling_summary(
                self.analysis_calendar_dates, subset, 6
            )["positive_sharpe_pct"],
            "rolling_12m_positive_pct": self._rolling_summary(
                self.analysis_calendar_dates, subset, 12
            )["positive_sharpe_pct"],
        }

    def _emit_proxy_analysis(self) -> None:
        self.daily_feature_frame = self._build_daily_feature_frame()
        self.analysis_calendar_dates = list(self.daily_feature_frame["date"]) if len(self.daily_feature_frame) else []
        self.daily_feature_map = {
            row["date"]: row for row in self.daily_feature_frame.to_dict(orient="records")
        }
        trades = pd.DataFrame(self.analysis_trades)
        if trades.empty:
            self.debug("QQQ slow-trend proxy analysis | status=no_trades_recorded")
            return
        if len(self.daily_feature_frame) == 0:
            self.debug("QQQ slow-trend proxy analysis | status=no_daily_features")
            return

        merged = trades.copy()
        for column in ["prev_day_up", "mom3_positive", *self.CANDIDATE_NAMES]:
            merged[column] = merged["entry_date"].map(
                lambda d, c=column: self.daily_feature_map.get(d, {}).get(c, False)
            )

        baseline_mask = merged["prev_day_up"].fillna(False) & merged["mom3_positive"].fillna(False)
        baseline = merged[baseline_mask].copy()
        baseline_stats = self._stats_for_subset(baseline)
        self.debug(
            "QQQ slow-trend proxy baseline | "
            f"trades={baseline_stats['trades']} | net_pnl={baseline_stats['net_pnl']} | "
            f"win_rate_pct={baseline_stats['win_rate_pct']} | profit_factor={baseline_stats['profit_factor']} | "
            f"rolling6={baseline_stats['rolling_6m_positive_pct']} | "
            f"rolling12={baseline_stats['rolling_12m_positive_pct']}"
        )

        summary_rows = []
        for candidate in self.CANDIDATE_NAMES:
            kept = baseline[baseline[candidate].fillna(False)].copy()
            excluded = baseline[~baseline[candidate].fillna(False)].copy()
            kept_stats = self._stats_for_subset(kept)
            excluded_stats = self._stats_for_subset(excluded)
            verdict = (
                "QC_STRONG"
                if (
                    kept_stats["net_pnl"] > baseline_stats["net_pnl"]
                    and excluded_stats["net_pnl"] < 0
                    and kept_stats["rolling_6m_positive_pct"] >= baseline_stats["rolling_6m_positive_pct"]
                    and kept_stats["rolling_12m_positive_pct"] >= baseline_stats["rolling_12m_positive_pct"]
                )
                else "QC_WEAK_OR_MIXED"
            )
            row = {
                "label": candidate,
                "trades": kept_stats["trades"],
                "net_pnl": kept_stats["net_pnl"],
                "excluded_net_pnl": excluded_stats["net_pnl"],
                "win_rate_pct": kept_stats["win_rate_pct"],
                "profit_factor": kept_stats["profit_factor"],
                "rolling_6m_positive_pct": kept_stats["rolling_6m_positive_pct"],
                "rolling_12m_positive_pct": kept_stats["rolling_12m_positive_pct"],
                "delta_vs_baseline": round(float(kept_stats["net_pnl"] - baseline_stats["net_pnl"]), 2),
                "verdict": verdict,
            }
            summary_rows.append(row)
            self.debug(
                "QQQ slow-trend proxy candidate | "
                f"label={row['label']} | trades={row['trades']} | net_pnl={row['net_pnl']} | "
                f"excluded_net_pnl={row['excluded_net_pnl']} | win_rate_pct={row['win_rate_pct']} | "
                f"profit_factor={row['profit_factor']} | rolling6={row['rolling_6m_positive_pct']} | "
                f"rolling12={row['rolling_12m_positive_pct']} | delta_vs_baseline={row['delta_vs_baseline']} | "
                f"verdict={row['verdict']}"
            )

        ranked = sorted(
            summary_rows,
            key=lambda item: (
                item["verdict"] == "QC_STRONG",
                item["net_pnl"],
                item["rolling_6m_positive_pct"],
                item["rolling_12m_positive_pct"],
            ),
            reverse=True,
        )
        compact_summary = {
            "analysis_scope": "v18_slow_trend_proxy_family",
            "baseline": baseline_stats,
            "top_candidates": ranked[:5],
        }
        self.debug(
            "QQQ slow-trend proxy compact-json | " + json.dumps(compact_summary, separators=(",", ":"))
        )
