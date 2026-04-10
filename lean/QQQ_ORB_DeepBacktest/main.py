from AlgorithmImports import *

from collections import deque
from datetime import timedelta
import numpy as np
import pandas as pd


class QQQOrbDeepBacktest(QCAlgorithm):
    """QQQ ORB deep backtest scaffold for QuantConnect / LEAN."""

    SCRIPT_VERSION = "v26-profit-lock"
    SUPPORTED_SYMBOL = "QQQ"

    def initialize(self) -> None:
        self.set_time_zone(TimeZones.NEW_YORK)

        start_year = int(self.get_parameter("start_year") or 2017)
        start_month = int(self.get_parameter("start_month") or 4)
        start_day = int(self.get_parameter("start_day") or 3)
        end_year = int(self.get_parameter("end_year") or 2026)
        end_month = int(self.get_parameter("end_month") or 4)
        end_day = int(self.get_parameter("end_day") or 2)

        self.set_start_date(start_year, start_month, start_day)
        self.set_end_date(end_year, end_month, end_day)
        self.set_cash(float(self.get_parameter("cash") or 100000))
        self.backtest_start = f"{start_year:04d}-{start_month:02d}-{start_day:02d}"
        self.backtest_end = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"

        self.ticker = (self.get_parameter("symbol") or self.SUPPORTED_SYMBOL).upper()
        if self.ticker != self.SUPPORTED_SYMBOL:
            raise ValueError(f"This deep backtest is configured for {self.SUPPORTED_SYMBOL} only.")

        self.position_size_pct = float(self.get_parameter("position_size_pct") or 0.25)
        self.min_trade_quantity = int(float(self.get_parameter("min_trade_quantity") or 1))
        self.max_trade_quantity = int(float(self.get_parameter("max_trade_quantity") or 500))
        self.orb_bars = int(self.get_parameter("orb_bars") or 4)
        self.profit_ratio = float(self.get_parameter("profit_ratio") or 3.5)
        self.breakout_confirm_pct = float(self.get_parameter("breakout_confirm_pct") or 0.0003)
        self.entry_delay_bars = int(self.get_parameter("entry_delay_bars") or 0)
        self.trailing_pct = float(self.get_parameter("trailing_pct") or 0.013)
        self.breakeven_trigger_mult = float(self.get_parameter("breakeven_trigger_mult") or 1.25)
        self.breakeven_active_minutes = int(self.get_parameter("breakeven_active_minutes") or 180)
        self.profit_lock_trigger_mult = float(self.get_parameter("profit_lock_trigger_mult") or 1.50)
        self.profit_lock_level_mult = float(self.get_parameter("profit_lock_level_mult") or 0.25)
        self.early_tight_trail_pct = float(self.get_parameter("early_tight_trail_pct") or 0.013)
        self.early_tight_trail_minutes = int(self.get_parameter("early_tight_trail_minutes") or 0)
        self.close_before_min = int(self.get_parameter("close_before_min") or 10)
        self.max_entries_per_session = int(self.get_parameter("max_entries_per_session") or 1)
        self.entry_start_hour_utc = int(self.get_parameter("entry_start_hour_utc") or 0)
        self.entry_end_hour_utc = int(self.get_parameter("entry_end_hour_utc") or 17)
        self.min_range_pct = float(self.get_parameter("min_range_pct") or 0.001)
        self.htf_filter = str(self.get_parameter("htf_filter") or "true").lower() == "true"
        self.htf_mode = self.get_parameter("htf_mode") or "slope"
        self.htf_ema_fast = int(self.get_parameter("htf_ema_fast") or 20)
        self.htf_ema_slow = int(self.get_parameter("htf_ema_slow") or 30)
        self.skip_short_after_up_days = int(self.get_parameter("skip_short_after_up_days") or 2)
        self.skip_long_after_up_days = int(self.get_parameter("skip_long_after_up_days") or 3)
        self.multi_day_range = str(self.get_parameter("multi_day_range") or "false").lower() == "true"
        self.multi_day_lookback = int(self.get_parameter("multi_day_lookback") or 2)
        self.regime_filter = str(self.get_parameter("regime_filter") or "true").lower() == "true"
        self.regime_mode = self.get_parameter("regime_mode") or "prev_day_up_and_mom3_positive"
        self.regime_allow_shorts = str(self.get_parameter("regime_allow_shorts") or "false").lower() == "true"
        self.regime_min_history_days = int(self.get_parameter("regime_min_history_days") or 4)
        self.settings.seed_initial_prices = True

        equity = self.add_equity(
            self.ticker,
            Resolution.MINUTE,
            fill_forward=False,
            extended_market_hours=False,
            data_normalization_mode=DataNormalizationMode.ADJUSTED,
        )
        self.symbol = equity.symbol
        self.set_benchmark(self.symbol)

        self.orb_history = deque(maxlen=max(self.multi_day_lookback, 5))
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
        self.entry_price = None
        self.breakeven_activated = False
        self.breakeven_gate_expired = False
        self.profit_lock_activated = False
        self.profit_lock_price = None
        self.entry_count = 0
        self.exit_count = 0
        self.session_entry_count = 0
        self.reentry_blocked_count = 0
        self.total_shares_traded = 0
        self.breakeven_activation_count = 0
        self.breakeven_stop_exit_count = 0
        self.breakeven_gate_expired_count = 0
        self.profit_lock_activation_count = 0
        self.profit_lock_stop_exit_count = 0
        self.current_session_market_close = None
        self.half_day_session_count = 0
        self.regime_allow_long_today = True
        self.regime_allow_short_today = True
        self.regime_label = "disabled"
        self.regime_pass_days = 0
        self.regime_block_days = 0
        self.regime_history_shortage_days = 0

        self.fast_ema = ExponentialMovingAverage(self.htf_ema_fast)
        self.slow_ema = ExponentialMovingAverage(self.htf_ema_slow)

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
            self.time_rules.before_market_close(self.symbol, self.close_before_min),
            self.force_flatten,
        )

        self.set_warm_up(timedelta(days=20), Resolution.MINUTE)
        self.debug(
            "QQQ ORB DeepBacktest init | "
            f"version={self.SCRIPT_VERSION} | cutoff=exchange-hours-aware | security_type=equity | "
            f"regime_filter={self.regime_filter} | regime_mode={self.regime_mode} | "
            f"regime_shorts={self.regime_allow_shorts} | max_entries_per_session={self.max_entries_per_session} | "
            f"entry_start_hour_utc={self.entry_start_hour_utc} | entry_end_hour_utc={self.entry_end_hour_utc} | "
            f"sizing=dynamic({self.position_size_pct:.0%}) | symbol={self.ticker} | orb_bars={self.orb_bars} | "
            f"profit_ratio={self.profit_ratio} | trailing_pct={self.trailing_pct} | "
            f"breakeven_trigger_mult={self.breakeven_trigger_mult} | "
            f"breakeven_active_minutes={self.breakeven_active_minutes} | "
            f"profit_lock_trigger_mult={self.profit_lock_trigger_mult} | "
            f"profit_lock_level_mult={self.profit_lock_level_mult} | "
            f"early_tight_trail_pct={self.early_tight_trail_pct} | "
            f"early_tight_trail_minutes={self.early_tight_trail_minutes} | "
            f"htf_mode={self.htf_mode} | period={self.backtest_start}->{self.backtest_end}"
        )

    @property
    def trade_quantity(self) -> int:
        """Dynamically compute position size based on portfolio value and current price."""
        price = self.securities[self.symbol].price
        if price <= 0:
            return self.min_trade_quantity
        qty = int(self.portfolio.total_portfolio_value * self.position_size_pct / price)
        return max(self.min_trade_quantity, min(qty, self.max_trade_quantity))

    def on_hour_bar(self, sender, bar: TradeBar) -> None:
        if not self.fast_ema.is_ready:
            return

        fast_now = float(self.fast_ema.current.value)
        slow_now = float(self.slow_ema.current.value) if self.slow_ema.is_ready else fast_now

        if self.htf_mode == "slope":
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
        if self.bars_from_open <= self.orb_bars + self.entry_delay_bars:
            return
        if not self._entry_time_filter_passed():
            return

        allow_short_today = not (
            self.skip_short_after_up_days > 0 and self.up_day_streak >= self.skip_short_after_up_days
        )
        allow_long_today = not (
            self.skip_long_after_up_days > 0 and self.up_day_streak >= self.skip_long_after_up_days
        )
        allow_long_today = allow_long_today and self.regime_allow_long_today
        allow_short_today = allow_short_today and self.regime_allow_short_today

        if bar.close > self.long_entry_level:
            if not allow_long_today:
                return
            if self.htf_filter and self.hourly_bias == -1:
                return
            if not self._can_take_session_entry():
                return
            qty = self.trade_quantity
            self.market_order(self.symbol, qty, tag=f"ORB Long qty={qty}")
            self.best_price_long = bar.close
            self.best_price_short = None
            self.position_entry_time = bar.end_time
            self.entry_price = bar.close
            self.breakeven_activated = False
            self.breakeven_gate_expired = False
            self.profit_lock_activated = False
            self.profit_lock_price = None
            self.entry_count += 1
            self.session_entry_count += 1
            self.total_shares_traded += qty
            return

        if bar.close < self.short_entry_level:
            if not allow_short_today:
                return
            if self.htf_filter and self.hourly_bias == 1:
                return
            if not self._can_take_session_entry():
                return
            qty = self.trade_quantity
            self.market_order(self.symbol, -qty, tag=f"ORB Short qty={qty}")
            self.best_price_short = bar.close
            self.best_price_long = None
            self.position_entry_time = bar.end_time
            self.entry_price = bar.close
            self.breakeven_activated = False
            self.breakeven_gate_expired = False
            self.profit_lock_activated = False
            self.profit_lock_price = None
            self.entry_count += 1
            self.session_entry_count += 1
            self.total_shares_traded += qty

    def force_flatten(self) -> None:
        if self.portfolio[self.symbol].invested:
            self.liquidate(self.symbol, "ORB EOD Flatten")
            self.exit_count += 1
            self._reset_position_state()

    def on_end_of_algorithm(self) -> None:
        avg_qty = round(self.total_shares_traded / self.entry_count, 1) if self.entry_count > 0 else 0
        self.debug(
            "QQQ ORB DeepBacktest summary | "
            f"version={self.SCRIPT_VERSION} | entries={self.entry_count} | exits={self.exit_count} | "
            f"total_shares={self.total_shares_traded} | avg_qty={avg_qty} | "
            f"up_day_streak={self.up_day_streak} | half_days={self.half_day_session_count} | "
            f"regime_pass_days={self.regime_pass_days} | regime_block_days={self.regime_block_days} | "
            f"regime_history_shortage_days={self.regime_history_shortage_days} | "
            f"reentry_blocked={self.reentry_blocked_count} | max_entries_per_session={self.max_entries_per_session} | "
            f"breakeven_activations={self.breakeven_activation_count} | "
            f"breakeven_stop_exits={self.breakeven_stop_exit_count} | "
            f"breakeven_gate_expired={self.breakeven_gate_expired_count} | "
            f"profit_lock_trigger_mult={self.profit_lock_trigger_mult} | "
            f"profit_lock_level_mult={self.profit_lock_level_mult} | "
            f"profit_lock_activations={self.profit_lock_activation_count} | "
            f"profit_lock_stop_exits={self.profit_lock_stop_exit_count}"
        )

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
            self.debug(
                "QQQ ORB DeepBacktest half-day detected | "
                f"version={self.SCRIPT_VERSION} | session_date={self.current_session_date} | "
                f"market_close={self.current_session_market_close} | "
                f"entry_cutoff={self.current_session_market_close - timedelta(minutes=self.close_before_min)}"
            )
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
        self._reset_position_state()

    def _update_regime_state(self) -> None:
        if not self.regime_filter:
            self.regime_allow_long_today = True
            self.regime_allow_short_today = True
            self.regime_label = "disabled"
            return

        lookback = max(self.regime_min_history_days + 3, 5)
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
        min_ready = self.regime_min_history_days
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
        self.regime_allow_short_today = bool(self.regime_allow_shorts and (not allow_long))
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
        if self.bars_from_open <= self.orb_bars:
            self.raw_orb_high = max(self.raw_orb_high, bar.high) if self.raw_orb_high is not None else bar.high
            self.raw_orb_low = min(self.raw_orb_low, bar.low) if self.raw_orb_low is not None else bar.low

        if self.raw_orb_high is None or self.raw_orb_low is None:
            return

        orb_high = self.raw_orb_high
        orb_low = self.raw_orb_low

        if self.multi_day_range and len(self.orb_history) > 0:
            lookback = min(self.multi_day_lookback, len(self.orb_history))
            recent = list(self.orb_history)[-lookback:]
            orb_high = max([high for high, _ in recent] + [orb_high])
            orb_low = min([low for _, low in recent] + [orb_low])

        range_width = orb_high - orb_low
        mid_price = (orb_high + orb_low) / 2.0 if (orb_high + orb_low) != 0 else 0
        if range_width <= 0 or mid_price <= 0:
            return
        if (range_width / mid_price) < self.min_range_pct:
            return

        self.orb_high = orb_high
        self.orb_low = orb_low
        self.range_width = range_width
        self.long_entry_level = orb_high * (1.0 + self.breakout_confirm_pct)
        self.short_entry_level = orb_low * (1.0 - self.breakout_confirm_pct)
        self.tp_long = orb_high + self.profit_ratio * range_width
        self.tp_short = orb_low - self.profit_ratio * range_width

    def _manage_open_position(self, bar: TradeBar) -> None:
        holding = self.portfolio[self.symbol]
        if not holding.invested:
            self._reset_position_state()
            return
        if self.range_width is None or self.entry_price is None:
            return

        trailing_pct = self._active_trailing_pct(bar.end_time)
        gate_active = self._breakeven_gate_active(bar.end_time)
        be_trigger_points = self._breakeven_trigger_points()
        profit_lock_trigger_points = self._profit_lock_trigger_points()
        profit_lock_points = self._profit_lock_points()
        if not gate_active and not self.breakeven_gate_expired:
            self.breakeven_gate_expired = True
            self.breakeven_gate_expired_count += 1
        if holding.is_long:
            self.best_price_long = max(self.best_price_long or bar.close, bar.close)
            if (
                be_trigger_points > 0.0
                and gate_active
                and not self.breakeven_activated
                and bar.close - self.entry_price >= be_trigger_points
            ):
                self.breakeven_activated = True
                self.breakeven_activation_count += 1
            if (
                gate_active
                and profit_lock_trigger_points > 0.0
                and profit_lock_points > 0.0
                and not self.profit_lock_activated
                and bar.close - self.entry_price >= profit_lock_trigger_points
            ):
                self.profit_lock_activated = True
                self.profit_lock_price = self.entry_price + profit_lock_points
                self.profit_lock_activation_count += 1
            trail_stop = self.best_price_long * (1.0 - trailing_pct)
            if self.breakeven_activated and gate_active:
                base_stop = max(self.entry_price, trail_stop)
            else:
                base_stop = max(self.orb_low, trail_stop) if self.orb_low is not None else trail_stop
            effective_stop = base_stop
            if self.profit_lock_activated and self.profit_lock_price is not None:
                effective_stop = max(effective_stop, self.profit_lock_price)
            if bar.close <= effective_stop:
                is_profit_lock_stop = (
                    self.profit_lock_activated
                    and self.profit_lock_price is not None
                    and self.profit_lock_price >= base_stop
                    and bar.close <= self.profit_lock_price * 1.001
                )
                is_be_stop = (
                    (not is_profit_lock_stop)
                    and self.breakeven_activated
                    and gate_active
                    and bar.close <= self.entry_price * 1.001
                )
                exit_tag = "ORB Long Stop"
                if is_profit_lock_stop:
                    exit_tag = "ORB Long Profit Lock Stop"
                elif is_be_stop:
                    exit_tag = "ORB Long BE Stop"
                self.liquidate(self.symbol, exit_tag)
                self.exit_count += 1
                if is_profit_lock_stop:
                    self.profit_lock_stop_exit_count += 1
                elif is_be_stop:
                    self.breakeven_stop_exit_count += 1
                self._reset_position_state()
            elif self.tp_long is not None and bar.close >= self.tp_long:
                self.liquidate(self.symbol, "ORB Long TP")
                self.exit_count += 1
                self._reset_position_state()
        elif holding.is_short:
            self.best_price_short = min(self.best_price_short or bar.close, bar.close)
            if (
                be_trigger_points > 0.0
                and gate_active
                and not self.breakeven_activated
                and self.entry_price - bar.close >= be_trigger_points
            ):
                self.breakeven_activated = True
                self.breakeven_activation_count += 1
            if (
                gate_active
                and profit_lock_trigger_points > 0.0
                and profit_lock_points > 0.0
                and not self.profit_lock_activated
                and self.entry_price - bar.close >= profit_lock_trigger_points
            ):
                self.profit_lock_activated = True
                self.profit_lock_price = self.entry_price - profit_lock_points
                self.profit_lock_activation_count += 1
            trail_stop = self.best_price_short * (1.0 + trailing_pct)
            if self.breakeven_activated and gate_active:
                base_stop = min(self.entry_price, trail_stop)
            else:
                base_stop = min(self.orb_high, trail_stop) if self.orb_high is not None else trail_stop
            effective_stop = base_stop
            if self.profit_lock_activated and self.profit_lock_price is not None:
                effective_stop = min(effective_stop, self.profit_lock_price)
            if bar.close >= effective_stop:
                is_profit_lock_stop = (
                    self.profit_lock_activated
                    and self.profit_lock_price is not None
                    and self.profit_lock_price <= base_stop
                    and bar.close >= self.profit_lock_price * 0.999
                )
                is_be_stop = (
                    (not is_profit_lock_stop)
                    and self.breakeven_activated
                    and gate_active
                    and bar.close >= self.entry_price * 0.999
                )
                exit_tag = "ORB Short Stop"
                if is_profit_lock_stop:
                    exit_tag = "ORB Short Profit Lock Stop"
                elif is_be_stop:
                    exit_tag = "ORB Short BE Stop"
                self.liquidate(self.symbol, exit_tag)
                self.exit_count += 1
                if is_profit_lock_stop:
                    self.profit_lock_stop_exit_count += 1
                elif is_be_stop:
                    self.breakeven_stop_exit_count += 1
                self._reset_position_state()
            elif self.tp_short is not None and bar.close <= self.tp_short:
                self.liquidate(self.symbol, "ORB Short TP")
                self.exit_count += 1
                self._reset_position_state()

    def _active_trailing_pct(self, timestamp) -> float:
        if (
            self.position_entry_time is not None
            and self.early_tight_trail_pct < self.trailing_pct
            and self.early_tight_trail_minutes > 0
            and timestamp - self.position_entry_time <= timedelta(minutes=self.early_tight_trail_minutes)
        ):
            return self.early_tight_trail_pct
        return self.trailing_pct

    def _breakeven_gate_active(self, timestamp) -> bool:
        if self.position_entry_time is None or self.breakeven_active_minutes <= 0:
            return False
        return timestamp - self.position_entry_time <= timedelta(minutes=self.breakeven_active_minutes)

    def _breakeven_trigger_points(self) -> float:
        if self.range_width is None or self.breakeven_trigger_mult <= 0:
            return 0.0
        return self.breakeven_trigger_mult * self.range_width

    def _profit_lock_trigger_points(self) -> float:
        if self.range_width is None or self.profit_lock_trigger_mult <= 0:
            return 0.0
        return self.profit_lock_trigger_mult * self.range_width

    def _profit_lock_points(self) -> float:
        if self.range_width is None or self.profit_lock_level_mult <= 0:
            return 0.0
        return self.profit_lock_level_mult * self.range_width

    def _reset_position_state(self) -> None:
        self.best_price_long = None
        self.best_price_short = None
        self.position_entry_time = None
        self.entry_price = None
        self.breakeven_activated = False
        self.breakeven_gate_expired = False
        self.profit_lock_activated = False
        self.profit_lock_price = None

    def _can_take_session_entry(self) -> bool:
        if self.session_entry_count >= self.max_entries_per_session:
            self.reentry_blocked_count += 1
            return False
        return True

    def _entry_time_filter_passed(self) -> bool:
        h = self.utc_time.hour
        return self.entry_start_hour_utc <= h < self.entry_end_hour_utc

    def _entry_window_closed(self, timestamp) -> bool:
        if self.current_session_market_close is None:
            return False
        return timestamp >= self.current_session_market_close - timedelta(minutes=self.close_before_min)
