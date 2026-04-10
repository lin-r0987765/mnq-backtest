"""Shared Blueshift runtime for the v26 ORB research lane."""

from __future__ import annotations

from datetime import time
from typing import Dict, Optional

import pandas as pd

try:  # pragma: no cover - available only inside Blueshift
    from blueshift.api import (
        date_rules,
        get_datetime,
        order_target_percent,
        record,
        schedule_function,
        symbol,
        time_rules,
    )
except ImportError:  # pragma: no cover - local syntax/test fallback
    date_rules = None
    time_rules = None
    record = None

    def get_datetime():
        raise RuntimeError("get_datetime is only available inside Blueshift")

    def order_target_percent(*_args, **_kwargs):
        raise RuntimeError("order_target_percent is only available inside Blueshift")

    def schedule_function(*_args, **_kwargs):
        raise RuntimeError("schedule_function is only available inside Blueshift")

    def symbol(*_args, **_kwargs):
        raise RuntimeError("symbol is only available inside Blueshift")


BASELINE_DEFAULTS: Dict[str, object] = {
    "script_version": "v26-profit-lock-blueshift",
    "baseline_reference": "v26-profit-lock",
    "research_only": False,
    "symbol": "QQQ",
    "position_size_pct": 0.25,
    "orb_bars": 4,
    "profit_ratio": 3.5,
    "breakout_confirm_pct": 0.0003,
    "trailing_pct": 0.013,
    "breakeven_trigger_mult": 1.25,
    "breakeven_active_minutes": 180,
    "profit_lock_trigger_mult": 1.50,
    "profit_lock_level_mult": 0.25,
    "max_entries_per_session": 1,
    "entry_start_hour_utc": 0,
    "entry_end_hour_utc": 17,
    "close_before_min": 10,
    "regime_filter": True,
    "regime_mode": "prev_day_up_and_mom3_positive",
    "regime_min_history_days": 4,
    "orb_reentry_enabled": False,
    "orb_reentry_arm_progress_mult": 1.0,
    "orb_reentry_depth_mult": 0.25,
    "orb_reentry_confirm_bars": 1,
}


def build_config(overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    cfg = dict(BASELINE_DEFAULTS)
    if overrides:
        cfg.update(overrides)
    return cfg


def initialize_strategy(context, overrides: Optional[Dict[str, object]] = None) -> None:
    cfg = build_config(overrides)
    context.asset = symbol(cfg["symbol"])
    context.cfg = cfg
    context.current_session = None
    context.current_orb = None
    context.orb_history = []
    context.regime_allow_long = True
    context.regime_label = "pending"
    context.entry_count_today = 0
    context.position_side = 0
    context.entry_price = None
    context.entry_time = None
    context.best_price = None
    context.breakeven_activated = False
    context.profit_lock_activated = False
    context.profit_lock_price = None
    context.orb_reentry_armed = False
    context.orb_reentry_inside_count = 0
    context.summary = {
        "entries": 0,
        "exits": 0,
        "breakeven_activations": 0,
        "profit_lock_activations": 0,
        "profit_lock_stop_exits": 0,
        "orb_reentry_armed_count": 0,
        "orb_reentry_exit_count": 0,
    }

    schedule_function(_start_session, date_rules.every_day(), time_rules.market_open(minutes=0))
    schedule_function(_process_five_minute, date_rules.every_day(), time_rules.every_nth_minute(5))
    schedule_function(
        _force_flatten,
        date_rules.every_day(),
        time_rules.market_close(minutes=int(cfg["close_before_min"])),
    )
    schedule_function(_record_session_snapshot, date_rules.every_day(), time_rules.market_close(minutes=1))

    print(
        "Blueshift ORB init | "
        f"version={cfg['script_version']} | baseline_reference={cfg['baseline_reference']} | "
        f"research_only={int(bool(cfg['research_only']))} | symbol={cfg['symbol']} | "
        f"regime_mode={cfg['regime_mode']} | trailing_pct={cfg['trailing_pct']} | "
        f"breakeven_trigger_mult={cfg['breakeven_trigger_mult']} | "
        f"breakeven_active_minutes={cfg['breakeven_active_minutes']} | "
        f"profit_lock_trigger_mult={cfg['profit_lock_trigger_mult']} | "
        f"profit_lock_level_mult={cfg['profit_lock_level_mult']} | "
        f"orb_reentry_enabled={int(bool(cfg['orb_reentry_enabled']))} | "
        f"orb_reentry_arm_progress_mult={cfg['orb_reentry_arm_progress_mult']} | "
        f"orb_reentry_depth_mult={cfg['orb_reentry_depth_mult']} | "
        f"orb_reentry_confirm_bars={cfg['orb_reentry_confirm_bars']}"
    )


def handle_data(context, data) -> None:
    """Main loop intentionally kept empty; scheduled callbacks run the strategy."""


def _start_session(context, data) -> None:
    now = pd.Timestamp(get_datetime())
    context.current_session = now.tz_convert("America/New_York").date() if now.tzinfo else now.date()
    context.current_orb = None
    context.entry_count_today = 0
    _reset_position_state(context)
    _update_regime_state(context, data)


def _update_regime_state(context, data) -> None:
    cfg = context.cfg
    if not cfg["regime_filter"]:
        context.regime_allow_long = True
        context.regime_label = "disabled"
        return

    lookback = max(int(cfg["regime_min_history_days"]) + 3, 5)
    daily = data.history(context.asset, ["close"], lookback, "1d")
    daily = _normalize_history(daily)
    if daily.empty or "close" not in daily.columns or len(daily) < lookback:
        context.regime_allow_long = False
        context.regime_label = "history_shortage"
        return

    closes = daily["close"].astype(float)
    prev_day_return = closes.iloc[-1] / closes.iloc[-2] - 1.0
    mom3_return = closes.iloc[-1] / closes.iloc[-4] - 1.0
    prev_day_up = bool(prev_day_return > 0.0)
    mom3_positive = bool(mom3_return > 0.0)
    context.regime_allow_long = prev_day_up and mom3_positive
    context.regime_label = (
        f"prev_day_up={int(prev_day_up)}|mom3_positive={int(mom3_positive)}|"
        f"prev_day_return_pct={prev_day_return * 100.0:.3f}|mom3_return_pct={mom3_return * 100.0:.3f}"
    )


def _process_five_minute(context, data) -> None:
    now = pd.Timestamp(get_datetime())
    now_utc = now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
    if not _entry_time_filter_passed(context, now_utc):
        _manage_open_position(context, data, now)
        return

    session_5m = _fetch_session_bars(context, data, now)
    if session_5m.empty:
        return

    latest = session_5m.iloc[-1]
    _update_orb_state(context, session_5m)
    _manage_open_position(context, data, now, latest_bar=latest)

    if context.position_side != 0:
        return
    if context.entry_count_today >= int(context.cfg["max_entries_per_session"]):
        return
    if not context.regime_allow_long:
        return
    if context.current_orb is None:
        return

    close = float(latest["close"])
    if close > context.current_orb["long_entry_level"]:
        order_target_percent(context.asset, float(context.cfg["position_size_pct"]))
        context.position_side = 1
        context.entry_price = close
        context.entry_time = now
        context.best_price = close
        context.breakeven_activated = False
        context.profit_lock_activated = False
        context.profit_lock_price = None
        context.orb_reentry_armed = False
        context.orb_reentry_inside_count = 0
        context.entry_count_today += 1
        context.summary["entries"] += 1
        print(
            "Blueshift ORB long entry | "
            f"version={context.cfg['script_version']} | close={close:.4f} | "
            f"orb_high={context.current_orb['orb_high']:.4f} | range_width={context.current_orb['range_width']:.4f}"
        )


def _manage_open_position(context, data, now, latest_bar=None) -> None:
    if context.position_side != 1 or context.current_orb is None or context.entry_price is None:
        return
    if latest_bar is None:
        session_5m = _fetch_session_bars(context, data, now)
        if session_5m.empty:
            return
        latest_bar = session_5m.iloc[-1]

    close = float(latest_bar["close"])
    range_width = float(context.current_orb["range_width"])
    orb_low = float(context.current_orb["orb_low"])
    context.best_price = max(context.best_price or close, close)

    be_trigger = float(context.cfg["breakeven_trigger_mult"]) * range_width
    if (
        be_trigger > 0.0
        and _breakeven_gate_active(context, now)
        and not context.breakeven_activated
        and close - context.entry_price >= be_trigger
    ):
        context.breakeven_activated = True
        context.summary["breakeven_activations"] += 1

    profit_lock_trigger = float(context.cfg["profit_lock_trigger_mult"]) * range_width
    profit_lock_points = float(context.cfg["profit_lock_level_mult"]) * range_width
    if (
        profit_lock_trigger > 0.0
        and not context.profit_lock_activated
        and close - context.entry_price >= profit_lock_trigger
    ):
        context.profit_lock_activated = True
        context.profit_lock_price = context.entry_price + profit_lock_points
        context.summary["profit_lock_activations"] += 1

    trailing_pct = float(context.cfg["trailing_pct"])
    trail_stop = context.best_price * (1.0 - trailing_pct)
    if context.breakeven_activated and _breakeven_gate_active(context, now):
        base_stop = max(context.entry_price, trail_stop)
    else:
        base_stop = max(orb_low, trail_stop)
    effective_stop = max(base_stop, context.profit_lock_price or float("-inf"))

    if bool(context.cfg["orb_reentry_enabled"]):
        _update_orb_reentry_state(context, close)
        if context.orb_reentry_inside_count >= int(context.cfg["orb_reentry_confirm_bars"]):
            _exit_position(context, "ORB Reentry Exit")
            context.summary["orb_reentry_exit_count"] += 1
            return

    if close <= effective_stop:
        if context.profit_lock_activated and context.profit_lock_price is not None and close <= context.profit_lock_price * 1.001:
            context.summary["profit_lock_stop_exits"] += 1
            _exit_position(context, "Profit Lock Stop")
        elif context.breakeven_activated and _breakeven_gate_active(context, now) and close <= context.entry_price * 1.001:
            _exit_position(context, "BE Stop")
        else:
            _exit_position(context, "Long Stop")


def _update_orb_reentry_state(context, close: float) -> None:
    if context.current_orb is None or context.entry_price is None or context.best_price is None:
        return
    range_width = float(context.current_orb["range_width"])
    arm_progress = float(context.cfg["orb_reentry_arm_progress_mult"]) * range_width
    if (not context.orb_reentry_armed) and (context.best_price - context.entry_price >= arm_progress):
        context.orb_reentry_armed = True
        context.summary["orb_reentry_armed_count"] += 1
    if not context.orb_reentry_armed:
        return
    reentry_level = float(context.current_orb["orb_high"]) - float(context.cfg["orb_reentry_depth_mult"]) * range_width
    if close <= reentry_level:
        context.orb_reentry_inside_count += 1
    else:
        context.orb_reentry_inside_count = 0


def _exit_position(context, reason: str) -> None:
    order_target_percent(context.asset, 0.0)
    context.summary["exits"] += 1
    print(
        "Blueshift ORB exit | "
        f"version={context.cfg['script_version']} | reason={reason} | "
        f"entry_price={context.entry_price:.4f} | best_price={float(context.best_price or 0.0):.4f}"
    )
    _reset_position_state(context)


def _force_flatten(context, data) -> None:
    if context.position_side != 0:
        _exit_position(context, "EOD Flatten")


def _record_session_snapshot(context, data) -> None:
    if record is None:
        return
    record(
        entries=context.summary["entries"],
        exits=context.summary["exits"],
        breakeven_activations=context.summary["breakeven_activations"],
        profit_lock_activations=context.summary["profit_lock_activations"],
        orb_reentry_armed_count=context.summary["orb_reentry_armed_count"],
        orb_reentry_exit_count=context.summary["orb_reentry_exit_count"],
    )


def _entry_time_filter_passed(context, timestamp_utc: pd.Timestamp) -> bool:
    hour = timestamp_utc.hour
    start = int(context.cfg["entry_start_hour_utc"])
    end = int(context.cfg["entry_end_hour_utc"])
    return start <= hour < end


def _breakeven_gate_active(context, now: pd.Timestamp) -> bool:
    if context.entry_time is None:
        return False
    active_minutes = int(context.cfg["breakeven_active_minutes"])
    return (now - context.entry_time).total_seconds() <= active_minutes * 60


def _update_orb_state(context, session_5m: pd.DataFrame) -> None:
    cfg = context.cfg
    orb_bars = int(cfg["orb_bars"])
    if len(session_5m) < orb_bars:
        return
    orb_window = session_5m.iloc[:orb_bars]
    raw_orb_high = float(orb_window["high"].max())
    raw_orb_low = float(orb_window["low"].min())
    orb_high = raw_orb_high
    orb_low = raw_orb_low
    if bool(cfg.get("multi_day_range")) and context.orb_history:
        lookback = min(int(cfg.get("multi_day_lookback", 2)), len(context.orb_history))
        recent = context.orb_history[-lookback:]
        orb_high = max([item["raw_orb_high"] for item in recent] + [orb_high])
        orb_low = min([item["raw_orb_low"] for item in recent] + [orb_low])
    range_width = orb_high - orb_low
    if range_width <= 0:
        context.current_orb = None
        return
    midpoint = (orb_high + orb_low) / 2.0
    if midpoint <= 0 or range_width / midpoint < 0.001:
        context.current_orb = None
        return
    breakout_pct = float(cfg["breakout_confirm_pct"])
    context.current_orb = {
        "raw_orb_high": raw_orb_high,
        "raw_orb_low": raw_orb_low,
        "orb_high": orb_high,
        "orb_low": orb_low,
        "range_width": range_width,
        "long_entry_level": orb_high * (1.0 + breakout_pct),
    }


def _fetch_session_bars(context, data, now: pd.Timestamp) -> pd.DataFrame:
    minute_history = data.history(context.asset, ["open", "high", "low", "close", "volume"], 390, "1m")
    frame = _normalize_history(minute_history)
    if frame.empty:
        return frame
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    ny_index = frame.index.tz_convert("America/New_York")
    session_date = (now.tz_convert("America/New_York") if now.tzinfo else now).date()
    session_mask = ny_index.date == session_date
    session_frame = frame.loc[session_mask]
    if session_frame.empty:
        return session_frame
    session_5m = (
        session_frame.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    return session_5m


def _normalize_history(frame) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    normalized = frame.copy()
    if isinstance(normalized.index, pd.MultiIndex):
        normalized = normalized.reset_index()
        time_col = normalized.columns[0]
        normalized = normalized.drop(columns=[col for col in normalized.columns if col == "asset"], errors="ignore")
        normalized = normalized.set_index(time_col)
    normalized.columns = [str(col).lower() for col in normalized.columns]
    normalized.index = pd.DatetimeIndex(normalized.index)
    return normalized.sort_index()


def _reset_position_state(context) -> None:
    context.position_side = 0
    context.entry_price = None
    context.entry_time = None
    context.best_price = None
    context.breakeven_activated = False
    context.profit_lock_activated = False
    context.profit_lock_price = None
    context.orb_reentry_armed = False
    context.orb_reentry_inside_count = 0

