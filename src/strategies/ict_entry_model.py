"""
Mechanical ICT-style entry model.

This module implements a systematic approximation of several ICT concepts:

- liquidity sweep / stop hunt of recent highs or lows
- market structure shift after the sweep
- fair value gap (FVG) as the preferred entry zone
- optional consequent-encroachment and rejection-quality gating on FVG revisits
- order block (OB) as a fallback entry zone
- breaker block as a second fallback delivery array
- inversion fair value gap (IFVG) as a later-stage fallback delivery array
- optional external-liquidity gating so only higher-order sweeps arm setups
- optional SMT divergence gating using peer-symbol highs/lows
- optional macro timing window gating inside the broader kill-zone schedule
- optional previous-session anchor gating using prior session range and open liquidity
- optional session-specific dealing-array refinement for imbalance vs structural windows

The implementation is intentionally deterministic and avoids look-ahead by
using only bars that are available at the current decision point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.fetcher import resample_ohlcv
from src.strategies.base import BaseStrategy, StrategyResult

try:  # pragma: no cover - configuration import is optional for portability
    import config as ict_config
except Exception:  # pragma: no cover
    ict_config = None


def _cfg(name: str, fallback: Any) -> Any:
    return getattr(ict_config, name, fallback) if ict_config is not None else fallback


_DEFAULT_PARAMS: dict[str, Any] = {
    "structure_lookback": int(_cfg("STRUCTURE_LOOKBACK", 20)),
    "swing_threshold": int(_cfg("SWING_THRESHOLD", 3)),
    "structure_reference_mode": str(_cfg("STRUCTURE_REFERENCE_MODE", "rolling")),
    "ob_lookback": int(_cfg("OB_LOOKBACK", 15)),
    "breaker_lookback": int(_cfg("BREAKER_LOOKBACK", 15)),
    "ifvg_lookback": int(_cfg("IFVG_LOOKBACK", 20)),
    "ob_body_min_pct": float(_cfg("OB_BODY_MIN_PCT", 0.3)),
    "ob_mitigation_touch": bool(_cfg("OB_MITIGATION_TOUCH", True)),
    "fvg_min_gap_pct": float(_cfg("FVG_MIN_GAP_PCT", 0.001)),
    "fvg_max_age": int(_cfg("FVG_MAX_AGE", 20)),
    "fvg_revisit_min_delay_bars": int(_cfg("FVG_REVISIT_MIN_DELAY_BARS", 1)),
    "max_pending_setups_per_direction": int(_cfg("MAX_PENDING_SETUPS_PER_DIRECTION", 1)),
    "max_reentries_per_setup": int(_cfg("MAX_REENTRIES_PER_SETUP", 0)),
    "fvg_origin_max_lag_bars": int(_cfg("FVG_ORIGIN_MAX_LAG_BARS", 0)),
    "fvg_origin_body_min_pct": float(_cfg("FVG_ORIGIN_BODY_MIN_PCT", 0.0)),
    "fvg_origin_body_atr_mult": float(_cfg("FVG_ORIGIN_BODY_ATR_MULT", 0.0)),
    "fvg_origin_close_position_min_pct": float(_cfg("FVG_ORIGIN_CLOSE_POSITION_MIN_PCT", 0.0)),
    "fvg_origin_opposite_wick_max_pct": float(_cfg("FVG_ORIGIN_OPPOSITE_WICK_MAX_PCT", 0.0)),
    "fvg_origin_range_atr_mult": float(_cfg("FVG_ORIGIN_RANGE_ATR_MULT", 0.0)),
    "fvg_max_retest_touches": int(_cfg("FVG_MAX_RETEST_TOUCHES", 0)),
    "displacement_body_min_pct": float(_cfg("DISPLACEMENT_BODY_MIN_PCT", 0.0)),
    "displacement_range_atr_mult": float(_cfg("DISPLACEMENT_RANGE_ATR_MULT", 0.0)),
    "structure_shift_close_buffer_ratio": float(_cfg("STRUCTURE_SHIFT_CLOSE_BUFFER_RATIO", 0.0)),
    "structure_shift_intrabar_tolerance_ratio": float(
        _cfg("STRUCTURE_SHIFT_INTRABAR_TOLERANCE_RATIO", 0.0)
    ),
    "structure_shift_intrabar_close_position_min_pct": float(
        _cfg("STRUCTURE_SHIFT_INTRABAR_CLOSE_POSITION_MIN_PCT", 0.0)
    ),
    "fvg_revisit_depth_ratio": float(_cfg("FVG_REVISIT_DEPTH_RATIO", 0.0)),
    "fvg_rejection_close_ratio": float(_cfg("FVG_REJECTION_CLOSE_RATIO", 0.0)),
    "fvg_rejection_wick_ratio": float(_cfg("FVG_REJECTION_WICK_RATIO", 0.0)),
    "fvg_rejection_body_min_pct": float(_cfg("FVG_REJECTION_BODY_MIN_PCT", 0.0)),
    "enable_continuation_entry": bool(_cfg("ENABLE_CONTINUATION_ENTRY", False)),
    "allow_long_entries": bool(_cfg("ALLOW_LONG_ENTRIES", True)),
    "allow_short_entries": bool(_cfg("ALLOW_SHORT_ENTRIES", True)),
    "liq_sweep_lookback": int(_cfg("LIQ_SWEEP_LOOKBACK", 50)),
    "liq_sweep_threshold": float(_cfg("LIQ_SWEEP_THRESHOLD", 0.001)),
    "liq_sweep_recovery_bars": int(_cfg("LIQ_SWEEP_RECOVERY_BARS", 3)),
    "slow_recovery_enabled": bool(_cfg("SLOW_RECOVERY_ENABLED", False)),
    "slow_recovery_bars": int(_cfg("SLOW_RECOVERY_BARS", 0)),
    "liq_sweep_reclaim_ratio": float(_cfg("LIQ_SWEEP_RECLAIM_RATIO", 0.0)),
    "ote_fib_low": float(_cfg("OTE_FIB_LOW", 0.618)),
    "ote_fib_high": float(_cfg("OTE_FIB_HIGH", 0.786)),
    "atr_period": int(_cfg("ATR_PERIOD", 14)),
    "stop_loss_atr_mult": float(_cfg("STOP_LOSS_ATR_MULT", 2.0)),
    "take_profit_rr": float(_cfg("TAKE_PROFIT_RR", 4.0)),
    "min_reward_risk_ratio": float(_cfg("MIN_REWARD_RISK_RATIO", 1.5)),
    "require_fvg_delivery": bool(_cfg("REQUIRE_FVG_DELIVERY", False)),
    "require_ote_zone": bool(_cfg("REQUIRE_OTE_ZONE", False)),
    "score_order_block": float(_cfg("SCORE_ORDER_BLOCK", 2)),
    "score_breaker_block": float(_cfg("SCORE_BREAKER_BLOCK", 2)),
    "score_ifvg": float(_cfg("SCORE_IFVG", 2)),
    "score_fvg": float(_cfg("SCORE_FVG", 2)),
    "score_liquidity_sweep": float(_cfg("SCORE_LIQUIDITY_SWEEP", 3)),
    "score_ote_zone": float(_cfg("SCORE_OTE_ZONE", 2)),
    "score_bos": float(_cfg("SCORE_BOS", 2)),
    "score_choch": float(_cfg("SCORE_CHOCH", 3)),
    "score_sweep_depth_quality": float(_cfg("SCORE_SWEEP_DEPTH_QUALITY", 0.0)),
    "score_displacement_quality": float(_cfg("SCORE_DISPLACEMENT_QUALITY", 0.0)),
    "score_fvg_gap_quality": float(_cfg("SCORE_FVG_GAP_QUALITY", 0.0)),
    "min_score_to_trade": float(_cfg("MIN_SCORE_TO_TRADE", 6)),
    "trade_sessions": bool(_cfg("TRADE_SESSIONS", True)),
    "london_open": int(_cfg("LONDON_OPEN", 8)),
    "london_close": int(_cfg("LONDON_CLOSE", 16)),
    "ny_open": int(_cfg("NY_OPEN", 13)),
    "ny_close": int(_cfg("NY_CLOSE", 21)),
    "use_kill_zones": bool(_cfg("USE_KILL_ZONES", False)),
    "kill_zone_timezone": str(_cfg("KILL_ZONE_TIMEZONE", "America/New_York")),
    "london_kill_start": int(_cfg("LONDON_KILL_START", 3)),
    "london_kill_end": int(_cfg("LONDON_KILL_END", 4)),
    "ny_am_kill_start": int(_cfg("NY_AM_KILL_START", 10)),
    "ny_am_kill_end": int(_cfg("NY_AM_KILL_END", 11)),
    "ny_pm_kill_start": int(_cfg("NY_PM_KILL_START", 14)),
    "ny_pm_kill_end": int(_cfg("NY_PM_KILL_END", 15)),
    "use_daily_bias_filter": bool(_cfg("USE_DAILY_BIAS_FILTER", False)),
    "daily_bias_mode": str(_cfg("DAILY_BIAS_MODE", "statistical")),
    "daily_bias_lookback": int(_cfg("DAILY_BIAS_LOOKBACK", 3)),
    "daily_bias_swing_threshold": int(_cfg("DAILY_BIAS_SWING_THRESHOLD", 1)),
    "daily_bias_bull_threshold": float(_cfg("DAILY_BIAS_BULL_THRESHOLD", 0.6)),
    "daily_bias_bear_threshold": float(_cfg("DAILY_BIAS_BEAR_THRESHOLD", 0.4)),
    "use_regime_adaptation": bool(_cfg("USE_REGIME_ADAPTATION", False)),
    "regime_atr_period": int(_cfg("REGIME_ATR_PERIOD", 14)),
    "regime_atr_pct_window": int(_cfg("REGIME_ATR_PCT_WINDOW", 48)),
    "regime_atr_high_mult": float(_cfg("REGIME_ATR_HIGH_MULT", 1.15)),
    "regime_adx_period": int(_cfg("REGIME_ADX_PERIOD", 14)),
    "regime_adx_trend_threshold": float(_cfg("REGIME_ADX_TREND_THRESHOLD", 22.0)),
    "regime_high_liq_sweep_threshold": float(_cfg("REGIME_HIGH_LIQ_SWEEP_THRESHOLD", 0.0)),
    "regime_high_fvg_min_gap_pct": float(_cfg("REGIME_HIGH_FVG_MIN_GAP_PCT", 0.0)),
    "regime_high_fvg_revisit_depth_ratio": float(_cfg("REGIME_HIGH_FVG_REVISIT_DEPTH_RATIO", -1.0)),
    "regime_high_fvg_revisit_min_delay_bars": int(_cfg("REGIME_HIGH_FVG_REVISIT_MIN_DELAY_BARS", -1)),
    "regime_high_smt_threshold": float(_cfg("REGIME_HIGH_SMT_THRESHOLD", 0.0)),
    "regime_high_min_reward_risk_ratio": float(_cfg("REGIME_HIGH_MIN_REWARD_RISK_RATIO", 0.0)),
    "use_higher_timeframe_alignment": bool(_cfg("USE_HIGHER_TIMEFRAME_ALIGNMENT", False)),
    "higher_timeframe_alignment_mode": str(_cfg("HIGHER_TIMEFRAME_ALIGNMENT_MODE", "hard")),
    "higher_timeframe_bias_timeframe": str(_cfg("HIGHER_TIMEFRAME_BIAS_TIMEFRAME", "1H")),
    "higher_timeframe_bias_lookback": int(_cfg("HIGHER_TIMEFRAME_BIAS_LOOKBACK", 12)),
    "higher_timeframe_bias_swing_threshold": int(_cfg("HIGHER_TIMEFRAME_BIAS_SWING_THRESHOLD", 1)),
    "higher_timeframe_mismatch_score_penalty": float(
        _cfg("HIGHER_TIMEFRAME_MISMATCH_SCORE_PENALTY", 0.0)
    ),
    "use_mtf_topdown_continuation": bool(_cfg("USE_MTF_TOPDOWN_CONTINUATION", False)),
    "mtf_execution_timeframe": str(_cfg("MTF_EXECUTION_TIMEFRAME", "1m")),
    "mtf_bias_daily_timeframe": str(_cfg("MTF_BIAS_DAILY_TIMEFRAME", "1D")),
    "mtf_bias_4h_timeframe": str(_cfg("MTF_BIAS_4H_TIMEFRAME", "4H")),
    "mtf_bias_1h_timeframe": str(_cfg("MTF_BIAS_1H_TIMEFRAME", "1H")),
    "mtf_setup_timeframe": str(_cfg("MTF_SETUP_TIMEFRAME", "15m")),
    "mtf_confirmation_timeframe": str(_cfg("MTF_CONFIRMATION_TIMEFRAME", "5m")),
    "mtf_trigger_timeframe": str(_cfg("MTF_TRIGGER_TIMEFRAME", "1m")),
    "mtf_bias_lookback": int(_cfg("MTF_BIAS_LOOKBACK", 12)),
    "mtf_bias_swing_threshold": int(_cfg("MTF_BIAS_SWING_THRESHOLD", 1)),
    "mtf_setup_structure_lookback": int(_cfg("MTF_SETUP_STRUCTURE_LOOKBACK", 8)),
    "mtf_setup_swing_threshold": int(_cfg("MTF_SETUP_SWING_THRESHOLD", 1)),
    "mtf_setup_fvg_min_gap_pct": float(_cfg("MTF_SETUP_FVG_MIN_GAP_PCT", 0.0006)),
    "mtf_setup_displacement_body_min_pct": float(
        _cfg("MTF_SETUP_DISPLACEMENT_BODY_MIN_PCT", 0.35)
    ),
    "mtf_setup_zone_expiry_bars": int(_cfg("MTF_SETUP_ZONE_EXPIRY_BARS", 16)),
    "mtf_confirmation_structure_lookback": int(_cfg("MTF_CONFIRMATION_STRUCTURE_LOOKBACK", 3)),
    "mtf_confirmation_close_ratio": float(_cfg("MTF_CONFIRMATION_CLOSE_RATIO", 0.6)),
    "mtf_confirmation_body_min_pct": float(_cfg("MTF_CONFIRMATION_BODY_MIN_PCT", 0.35)),
    "mtf_confirmation_rejection_wick_ratio": float(
        _cfg("MTF_CONFIRMATION_REJECTION_WICK_RATIO", 0.0)
    ),
    "mtf_confirmation_rejection_volume_ratio": float(
        _cfg("MTF_CONFIRMATION_REJECTION_VOLUME_RATIO", 0.0)
    ),
    "mtf_trigger_structure_lookback": int(_cfg("MTF_TRIGGER_STRUCTURE_LOOKBACK", 5)),
    "mtf_trigger_close_ratio": float(_cfg("MTF_TRIGGER_CLOSE_RATIO", 0.6)),
    "mtf_trigger_body_min_pct": float(_cfg("MTF_TRIGGER_BODY_MIN_PCT", 0.3)),
    "mtf_trigger_rejection_wick_ratio": float(_cfg("MTF_TRIGGER_REJECTION_WICK_RATIO", 0.0)),
    "mtf_trigger_rejection_volume_ratio": float(_cfg("MTF_TRIGGER_REJECTION_VOLUME_RATIO", 0.0)),
    "mtf_trigger_expiry_bars": int(_cfg("MTF_TRIGGER_EXPIRY_BARS", 30)),
    "mtf_touch_confirm_bars": int(_cfg("MTF_TOUCH_CONFIRM_BARS", 0)),
    "mtf_touch_trigger_bars": int(_cfg("MTF_TOUCH_TRIGGER_BARS", 0)),
    "mtf_volume_lookback": int(_cfg("MTF_VOLUME_LOOKBACK", 20)),
    "mtf_fast_retest_entry_enabled": bool(_cfg("MTF_FAST_RETEST_ENTRY_ENABLED", False)),
    "mtf_fast_retest_displacement_body_min_pct": float(
        _cfg("MTF_FAST_RETEST_DISPLACEMENT_BODY_MIN_PCT", 0.0)
    ),
    "mtf_fast_retest_displacement_close_ratio": float(
        _cfg("MTF_FAST_RETEST_DISPLACEMENT_CLOSE_RATIO", 0.0)
    ),
    "mtf_fast_retest_min_close_ratio": float(_cfg("MTF_FAST_RETEST_MIN_CLOSE_RATIO", 0.0)),
    "mtf_max_stop_distance_atr_mult": float(_cfg("MTF_MAX_STOP_DISTANCE_ATR_MULT", 0.0)),
    "mtf_neutral_hourly_allows_high_quality": bool(
        _cfg("MTF_NEUTRAL_HOURLY_ALLOWS_HIGH_QUALITY", True)
    ),
    "mtf_neutral_hourly_min_score": float(_cfg("MTF_NEUTRAL_HOURLY_MIN_SCORE", 7.0)),
    "mtf_timing_timezone": str(_cfg("MTF_TIMING_TIMEZONE", "America/New_York")),
    "mtf_allowed_entry_weekdays": _cfg("MTF_ALLOWED_ENTRY_WEEKDAYS", ""),
    "mtf_allowed_entry_hours": _cfg("MTF_ALLOWED_ENTRY_HOURS", ""),
    "use_premium_discount_filter": bool(_cfg("USE_PREMIUM_DISCOUNT_FILTER", False)),
    "premium_discount_lookback": int(_cfg("PREMIUM_DISCOUNT_LOOKBACK", 5)),
    "premium_discount_neutral_band": float(_cfg("PREMIUM_DISCOUNT_NEUTRAL_BAND", 0.05)),
    "premium_discount_filter_mode": str(_cfg("PREMIUM_DISCOUNT_FILTER_MODE", "hard")),
    "premium_discount_mismatch_score_penalty": float(
        _cfg("PREMIUM_DISCOUNT_MISMATCH_SCORE_PENALTY", 0.0)
    ),
    "use_external_liquidity_filter": bool(_cfg("USE_EXTERNAL_LIQUIDITY_FILTER", False)),
    "external_liquidity_lookback": int(_cfg("EXTERNAL_LIQUIDITY_LOOKBACK", 80)),
    "external_liquidity_tolerance": float(_cfg("EXTERNAL_LIQUIDITY_TOLERANCE", 0.001)),
    "use_smt_filter": bool(_cfg("USE_SMT_FILTER", False)),
    "smt_lookback": int(_cfg("SMT_LOOKBACK", 20)),
    "smt_threshold": float(_cfg("SMT_THRESHOLD", 0.001)),
    "smt_mode": str(_cfg("SMT_MODE", "bar_extreme")),
    "smt_swing_threshold": int(_cfg("SMT_SWING_THRESHOLD", 1)),
    "use_amd_filter": bool(_cfg("USE_AMD_FILTER", False)),
    "amd_accumulation_bars": int(_cfg("AMD_ACCUMULATION_BARS", 3)),
    "amd_manipulation_threshold": float(_cfg("AMD_MANIPULATION_THRESHOLD", 0.001)),
    "amd_require_midpoint_reclaim": bool(_cfg("AMD_REQUIRE_MIDPOINT_RECLAIM", True)),
    "use_macro_timing_windows": bool(_cfg("USE_MACRO_TIMING_WINDOWS", False)),
    "macro_timezone": str(_cfg("MACRO_TIMEZONE", "America/New_York")),
    "macro_windows": tuple(_cfg("MACRO_WINDOWS", ((9, 50, 10, 10), (10, 50, 11, 10)))),
    "use_prev_session_anchor_filter": bool(_cfg("USE_PREV_SESSION_ANCHOR_FILTER", False)),
    "prev_session_anchor_tolerance": float(_cfg("PREV_SESSION_ANCHOR_TOLERANCE", 0.05)),
    "prev_session_anchor_filter_mode": str(_cfg("PREV_SESSION_ANCHOR_FILTER_MODE", "hard")),
    "prev_session_anchor_mismatch_score_penalty": float(
        _cfg("PREV_SESSION_ANCHOR_MISMATCH_SCORE_PENALTY", 0.0)
    ),
    "use_session_array_refinement": bool(_cfg("USE_SESSION_ARRAY_REFINEMENT", False)),
    "session_array_filter_mode": str(_cfg("SESSION_ARRAY_FILTER_MODE", "hard")),
    "session_array_mismatch_score_penalty": float(_cfg("SESSION_ARRAY_MISMATCH_SCORE_PENALTY", 0.0)),
    "dealing_array_timezone": str(_cfg("DEALING_ARRAY_TIMEZONE", "America/New_York")),
    "imbalance_array_windows": tuple(_cfg("IMBALANCE_ARRAY_WINDOWS", ((3, 0, 4, 30), (9, 30, 10, 30)))),
    "structural_array_windows": tuple(_cfg("STRUCTURAL_ARRAY_WINDOWS", ((10, 30, 11, 30), (13, 0, 14, 30)))),
}

_ICT_RESEARCH_PROFILE_OVERRIDES: dict[str, Any] = {
    "use_kill_zones": True,
    "use_daily_bias_filter": True,
    "use_premium_discount_filter": True,
    "use_external_liquidity_filter": True,
    "use_amd_filter": True,
    "use_macro_timing_windows": True,
    "use_prev_session_anchor_filter": True,
    "use_session_array_refinement": True,
}

_ICT_PAIRED_SURVIVOR_PROFILE_OVERRIDES: dict[str, Any] = {
    "use_kill_zones": False,
    "use_daily_bias_filter": False,
    "use_premium_discount_filter": False,
    "use_external_liquidity_filter": True,
    "use_amd_filter": False,
    "use_macro_timing_windows": False,
    "use_prev_session_anchor_filter": True,
    "use_session_array_refinement": False,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PROFILE_OVERRIDES,
    "use_session_array_refinement": True,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_OVERRIDES,
    "liq_sweep_threshold": 0.0008,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_OVERRIDES,
    "smt_lookback": 10,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_OVERRIDES,
    "use_premium_discount_filter": True,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_OVERRIDES,
    "trade_sessions": True,
    "london_open": 0,
    "london_close": 0,
    "ny_open": 14,
    "ny_close": 20,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_OVERRIDES,
    "liq_sweep_recovery_bars": 4,
}

_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_OVERRIDES,
    "structure_lookback": 12,
    "fvg_min_gap_pct": 0.0006,
    "fvg_revisit_depth_ratio": 0.5,
    "fvg_revisit_min_delay_bars": 3,
    "displacement_body_min_pct": 0.10,
}

_ICT_LITE_REVERSAL_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES,
    "use_premium_discount_filter": False,
    "use_external_liquidity_filter": False,
    "use_amd_filter": False,
    "use_macro_timing_windows": False,
    "use_prev_session_anchor_filter": False,
    "use_session_array_refinement": False,
    "use_kill_zones": False,
}

_ICT_LITE_REVERSAL_QUICK_DENSITY_REPAIR_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_PROFILE_OVERRIDES,
    "structure_lookback": 5,
    "liq_sweep_recovery_bars": 15,
    "fvg_min_gap_pct": 0.0003,
    "fvg_revisit_depth_ratio": 0.0,
}

_ICT_LITE_REVERSAL_QUICK_SWING_STRUCTURE_REPAIR_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUICK_DENSITY_REPAIR_OVERRIDES,
    "structure_reference_mode": "swing",
    "swing_threshold": 2,
}

_ICT_FVG_FIB_RETRACEMENT_RESEARCH_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUICK_SWING_STRUCTURE_REPAIR_OVERRIDES,
    "trade_sessions": False,
    "use_smt_filter": False,
    "require_fvg_delivery": True,
    "require_ote_zone": True,
    "ote_fib_low": 0.50,
    "ote_fib_high": 0.79,
    "score_ote_zone": 0.0,
    "min_score_to_trade": 5.0,
}

_ICT_STRICT_SOFT_PREMIUM_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES,
    "premium_discount_filter_mode": "soft",
    "premium_discount_mismatch_score_penalty": 2.0,
}

_ICT_COMPLETE_SOFT_PREMIUM_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_RESEARCH_PROFILE_OVERRIDES,
    "premium_discount_filter_mode": "soft",
    "premium_discount_mismatch_score_penalty": 2.0,
}

_ICT_STRICT_SOFT_SESSION_ARRAY_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES,
    "session_array_filter_mode": "soft",
    "session_array_mismatch_score_penalty": 2.0,
}

_ICT_COMPLETE_SOFT_SESSION_ARRAY_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_RESEARCH_PROFILE_OVERRIDES,
    "session_array_filter_mode": "soft",
    "session_array_mismatch_score_penalty": 2.0,
}

_ICT_STRICT_SOFT_PREV_SESSION_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES,
    "prev_session_anchor_filter_mode": "soft",
    "prev_session_anchor_mismatch_score_penalty": 2.0,
}

_ICT_COMPLETE_SOFT_PREV_SESSION_PROFILE_OVERRIDES: dict[str, Any] = {
    **_ICT_RESEARCH_PROFILE_OVERRIDES,
    "prev_session_anchor_filter_mode": "soft",
    "prev_session_anchor_mismatch_score_penalty": 2.0,
}

_ICT_LITE_REVERSAL_RELAXED_SMT_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_PROFILE_OVERRIDES,
    "smt_threshold": 0.0015,
}

_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_RELAXED_SMT_OVERRIDES,
    "liq_sweep_threshold": 0.0006,
}

_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_FASTER_RETEST_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_OVERRIDES,
    "fvg_revisit_min_delay_bars": 2,
    "fvg_min_gap_pct": 0.0010,
}

_ICT_LITE_REVERSAL_DUAL_SPEED_RECOVERY_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_FASTER_RETEST_OVERRIDES,
    "slow_recovery_enabled": True,
    "slow_recovery_bars": 8,
}

_ICT_LITE_REVERSAL_QUALIFIED_CONTINUATION_DENSITY_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_DUAL_SPEED_RECOVERY_OVERRIDES,
    "enable_continuation_entry": True,
    "slow_recovery_bars": 12,
    "structure_lookback": 8,
    "smt_threshold": 0.0013,
    "fvg_min_gap_pct": 0.0003,
    "fvg_revisit_depth_ratio": 0.35,
    "fvg_revisit_min_delay_bars": 4,
    "take_profit_rr": 3.0,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_FASTER_RETEST_OVERRIDES,
    "enable_continuation_entry": False,
    "slow_recovery_enabled": True,
    "slow_recovery_bars": 12,
    "structure_lookback": 12,
    "smt_threshold": 0.0010,
    "fvg_min_gap_pct": 0.0003,
    "fvg_revisit_depth_ratio": 0.5,
    "fvg_revisit_min_delay_bars": 4,
    "take_profit_rr": 3.0,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_OVERRIDES,
    "allow_short_entries": False,
    "liq_sweep_threshold": 0.0004,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_TIMING_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_CANDIDATE_OVERRIDES,
    "fvg_revisit_min_delay_bars": 3,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_TIMING_CANDIDATE_OVERRIDES,
    "liq_sweep_threshold": 0.00035,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_CAPACITY_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_CANDIDATE_OVERRIDES,
    "max_pending_setups_per_direction": 2,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE11_PENDING3_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_CAPACITY_CANDIDATE_OVERRIDES,
    "structure_lookback": 11,
    "max_pending_setups_per_direction": 3,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_PENDING3_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE11_PENDING3_CANDIDATE_OVERRIDES,
    "structure_lookback": 10,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_DEPTH04_PENDING3_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_PENDING3_CANDIDATE_OVERRIDES,
    "fvg_revisit_depth_ratio": 0.4,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_PENDING3_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_DEPTH04_PENDING3_CANDIDATE_OVERRIDES,
    "fvg_min_gap_pct": 0.0002,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_INTRABAR020_CP070_PENDING3_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_PENDING3_CANDIDATE_OVERRIDES,
    "structure_shift_intrabar_tolerance_ratio": 0.20,
    "structure_shift_intrabar_close_position_min_pct": 0.70,
}

_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_INTRABAR020_CP070_TP45_PENDING3_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_INTRABAR020_CP070_PENDING3_CANDIDATE_OVERRIDES,
    "take_profit_rr": 4.5,
}

_ICT_CORE_400_BASELINE_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_QUALIFIED_CONTINUATION_DENSITY_OVERRIDES,
    "trade_sessions": False,
    "structure_lookback": 5,
    "liq_sweep_threshold": 0.0003,
    "liq_sweep_recovery_bars": 15,
    "slow_recovery_enabled": True,
    "slow_recovery_bars": 20,
    "fvg_min_gap_pct": 0.0002,
    "fvg_revisit_depth_ratio": 0.0,
    "fvg_revisit_min_delay_bars": 1,
    "take_profit_rr": 2.0,
    "max_pending_setups_per_direction": 2,
}

_ICT_CORE_400_SHORT_ONLY_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_BASELINE_OVERRIDES,
    "allow_long_entries": False,
}

_ICT_CORE_400_SHORT_STAT_BIAS_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_ONLY_OVERRIDES,
    "use_daily_bias_filter": True,
    "stop_loss_atr_mult": 1.5,
    "take_profit_rr": 2.5,
}

_ICT_CORE_400_SHORT_STRUCTURE_BIAS_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_ONLY_OVERRIDES,
    "use_daily_bias_filter": True,
    "daily_bias_mode": "structure",
    "daily_bias_lookback": 5,
    "stop_loss_atr_mult": 1.5,
}

_ICT_CORE_400_SHORT_STRUCTURE_BIAS_LB6_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_BIAS_CANDIDATE_OVERRIDES,
    "daily_bias_lookback": 6,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_BIAS_LB6_CANDIDATE_OVERRIDES,
    "structure_lookback": 6,
    "liq_sweep_threshold": 0.000275,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_DENSITY_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_BIAS_LB6_CANDIDATE_OVERRIDES,
    "structure_lookback": 6,
    "liq_sweep_threshold": 0.00025,
    "liq_sweep_recovery_bars": 18,
    "slow_recovery_bars": 24,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_DENSITY_CANDIDATE_OVERRIDES,
    "liq_sweep_threshold": 0.000275,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_SL135_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_CANDIDATE_OVERRIDES,
    "stop_loss_atr_mult": 1.35,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_SL135_DAILY_BIAS_LB8_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_SL135_CANDIDATE_OVERRIDES,
    "daily_bias_lookback": 8,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_CANDIDATE_OVERRIDES,
    "max_pending_setups_per_direction": 3,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_PENDING4_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_CANDIDATE_OVERRIDES,
    "max_pending_setups_per_direction": 4,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_PENDING4_CANDIDATE_OVERRIDES,
    "liq_sweep_threshold": 0.000265,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_SL135_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_CANDIDATE_OVERRIDES,
    "stop_loss_atr_mult": 1.35,
}

_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_SL135_DAILY_BIAS_LB8_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_SL135_CANDIDATE_OVERRIDES,
    "daily_bias_lookback": 8,
}

_ICT_LITE_REVERSAL_REGIME_MTF_ALIGNMENT_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_FASTER_RETEST_OVERRIDES,
    "enable_continuation_entry": False,
    "use_regime_adaptation": True,
    "regime_atr_period": 14,
    "regime_atr_pct_window": 48,
    "regime_atr_high_mult": 1.15,
    "regime_adx_period": 14,
    "regime_adx_trend_threshold": 22.0,
    "regime_high_smt_threshold": 0.0010,
    "regime_high_fvg_min_gap_pct": 0.0003,
    "regime_high_fvg_revisit_depth_ratio": 0.35,
    "regime_high_fvg_revisit_min_delay_bars": 2,
    "regime_high_min_reward_risk_ratio": 1.25,
    "use_higher_timeframe_alignment": True,
    "higher_timeframe_alignment_mode": "hard",
    "higher_timeframe_bias_timeframe": "1H",
    "higher_timeframe_bias_lookback": 12,
    "higher_timeframe_bias_swing_threshold": 1,
}

_ICT_MTF_TOPDOWN_CONTINUATION_OVERRIDES: dict[str, Any] = {
    **_ICT_LITE_REVERSAL_PROFILE_OVERRIDES,
    "use_mtf_topdown_continuation": True,
    "trade_sessions": True,
    "london_open": 0,
    "london_close": 0,
    "ny_open": 14,
    "ny_close": 20,
    "use_higher_timeframe_alignment": False,
    "use_daily_bias_filter": False,
    "use_premium_discount_filter": False,
    "use_external_liquidity_filter": False,
    "use_amd_filter": False,
    "use_macro_timing_windows": False,
    "use_prev_session_anchor_filter": False,
    "use_session_array_refinement": False,
    "use_kill_zones": False,
    "enable_continuation_entry": False,
    "take_profit_rr": 2.5,
    "min_reward_risk_ratio": 1.5,
    "score_sweep_depth_quality": 1.0,
    "score_displacement_quality": 1.0,
    "score_fvg_gap_quality": 1.0,
    "mtf_setup_structure_lookback": 8,
    "mtf_setup_fvg_min_gap_pct": 0.0006,
    "mtf_setup_displacement_body_min_pct": 0.35,
    "mtf_setup_zone_expiry_bars": 16,
    "mtf_confirmation_structure_lookback": 3,
    "mtf_confirmation_close_ratio": 0.6,
    "mtf_confirmation_body_min_pct": 0.35,
    "mtf_confirmation_rejection_wick_ratio": 0.0,
    "mtf_confirmation_rejection_volume_ratio": 0.0,
    "mtf_trigger_structure_lookback": 5,
    "mtf_trigger_close_ratio": 0.6,
    "mtf_trigger_body_min_pct": 0.3,
    "mtf_trigger_rejection_wick_ratio": 0.0,
    "mtf_trigger_rejection_volume_ratio": 0.0,
    "mtf_trigger_expiry_bars": 30,
    "mtf_touch_confirm_bars": 0,
    "mtf_touch_trigger_bars": 0,
    "mtf_volume_lookback": 20,
    "mtf_fast_retest_entry_enabled": False,
    "mtf_fast_retest_displacement_body_min_pct": 0.0,
    "mtf_fast_retest_displacement_close_ratio": 0.0,
    "mtf_fast_retest_min_close_ratio": 0.0,
    "mtf_max_stop_distance_atr_mult": 0.0,
}

_ICT_MTF_TOPDOWN_CONTINUATION_QUALITY_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_MTF_TOPDOWN_CONTINUATION_OVERRIDES,
    "mtf_setup_structure_lookback": 12,
    "mtf_setup_fvg_min_gap_pct": 0.0010,
    "mtf_confirmation_structure_lookback": 4,
    "mtf_confirmation_body_min_pct": 0.30,
    "mtf_trigger_structure_lookback": 7,
    "mtf_trigger_body_min_pct": 0.40,
}

_ICT_MTF_TOPDOWN_CONTINUATION_EXECUTION_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_MTF_TOPDOWN_CONTINUATION_OVERRIDES,
    "mtf_confirmation_structure_lookback": 4,
    "mtf_trigger_expiry_bars": 15,
}

_ICT_MTF_TOPDOWN_CONTINUATION_SETUP_EXECUTION_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_MTF_TOPDOWN_CONTINUATION_EXECUTION_CANDIDATE_OVERRIDES,
    "mtf_setup_fvg_min_gap_pct": 0.0008,
    "mtf_setup_displacement_body_min_pct": 0.30,
}

_ICT_MTF_TOPDOWN_CONTINUATION_TIMING_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_MTF_TOPDOWN_CONTINUATION_SETUP_EXECUTION_CANDIDATE_OVERRIDES,
    "mtf_allowed_entry_hours": (9, 10, 11, 12, 14, 15),
}

_ICT_MTF_TOPDOWN_CONTINUATION_REGULARIZED_LONG_ONLY_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_MTF_TOPDOWN_CONTINUATION_QUALITY_CANDIDATE_OVERRIDES,
    "allow_short_entries": False,
    "mtf_allowed_entry_hours": (10, 11, 12, 14, 15),
}

_ICT_MTF_TOPDOWN_CONTINUATION_REGULARIZED_LONG_ONLY_AM_CANDIDATE_OVERRIDES: dict[str, Any] = {
    **_ICT_MTF_TOPDOWN_CONTINUATION_QUALITY_CANDIDATE_OVERRIDES,
    "allow_short_entries": False,
    "mtf_allowed_entry_hours": (9, 10, 11, 12),
}


def build_ict_research_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current repo-approved ICT research profile.

    This profile turns on the implemented deterministic ICT filters while keeping
    SMT explicitly opt-in so the strategy cannot silently run a fake SMT test
    without peer-symbol data.
    """
    params = {**_DEFAULT_PARAMS, **_ICT_RESEARCH_PROFILE_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the broader paired-data ICT profile anchored to the current robust survivors.

    This profile keeps the relaxed paired-data lane active while preserving the two
    context filters that survived reintroduction on the broader QQQ+SPY Alpaca lane:
    previous-session anchors and external-liquidity gating.
    """
    params = {**_DEFAULT_PARAMS, **_ICT_PAIRED_SURVIVOR_PROFILE_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust pairwise extension on top of the paired-data survivor base.

    The current broader paired-data calibration shows that adding session-array
    refinement on top of the robust survivor base preserves positive activity
    better than the other robust pairwise additions.
    """
    params = {**_DEFAULT_PARAMS, **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_loose_sweep_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust geometry extension on top of the survivor-plus-session-array profile.

    The current broader paired-data geometry calibration shows that loosening
    the sweep threshold to 0.0008 materially improves activity and quality while
    preserving the survivor-plus-session-array stack.
    """
    params = {**_DEFAULT_PARAMS, **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust SMT extension on top of the loose-sweep paired ICT profile.

    The current broader paired-data SMT recalibration shows that shortening the
    SMT lookback to 10 preserves most of the loose-sweep profile's activity while
    materially improving return and drawdown characteristics.
    """
    params = {**_DEFAULT_PARAMS, **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust context reintroduction on top of the short-SMT paired ICT profile.

    The current broader paired-data reintroduction pass shows that premium/discount
    context can be added back on top of the short-SMT base without reducing trades
    or degrading quality.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust session-gating extension on top of the short-SMT premium paired ICT profile.

    The current broader paired-data session-gating calibration shows that an
    NY-only core window improves the short-SMT premium-plus-session-array base
    while preserving activity, making it the first robust extension on that base.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust sweep-geometry extension on top of the NY-only paired ICT frontier.

    The current broader paired-data sweep-geometry calibration shows that
    extending the liquidity-sweep recovery window to 4 bars improves the
    NY-only short-SMT premium-plus-session-array base while preserving and
    slightly increasing activity.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current structure-aware paired ICT frontier.

    The current broader paired-data structure calibration shows that shortening
    the structure lookback to 12 preserves the slow-recovery NY-only stack while
    slightly increasing both activity and total return. The latest FVG min-gap
    calibration then improves that stronger frontier again at `0.0006`, and the
    subsequent consequent-encroachment revisit-depth calibration improves it once
    more at `fvg_revisit_depth_ratio = 0.5`, and the latest revisit-delay
    calibration improves that CE-extended frontier again at
    `fvg_revisit_min_delay_bars = 3`. The most recent post-repair upstream
    recalibration then confirms that a light displacement-body filter at `0.10`
    is the first robust extension that materially improves the corrected lane,
    so this helper now represents the current repo-approved paired ICT frontier.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_PAIRED_SURVIVOR_PLUS_SESSION_ARRAY_LOOSE_SWEEP_SHORT_SMT_PREMIUM_NY_ONLY_SLOW_RECOVERY_SHORT_STRUCTURE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_strict_soft_premium_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the roadmap-aligned strict ICT lane with premium/discount softened into score pressure.

    This keeps the PDF-aligned strict context stack intact while converting the
    premium/discount mismatch from a hard sweep reject into a downstream score
    penalty so density can be studied without silently disabling the filter.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_STRICT_SOFT_PREMIUM_PROFILE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_complete_soft_premium_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the full ICT stack with premium/discount softened into score pressure.

    This keeps the complete research profile intact while converting
    premium/discount mismatches from a hard rejection into downstream score
    pressure so the full stack can be iterated from a less brittle baseline.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_COMPLETE_SOFT_PREMIUM_PROFILE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_strict_soft_session_array_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the roadmap-aligned strict ICT lane with session-array mismatches softened into score pressure.

    This keeps the strict PDF-aligned context stack intact while converting a
    disallowed delivery-array timing mismatch into a downstream score penalty
    instead of a hard shift reject.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_STRICT_SOFT_SESSION_ARRAY_PROFILE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_complete_soft_session_array_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the full ICT stack with session-array refinement softened into score pressure.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_COMPLETE_SOFT_SESSION_ARRAY_PROFILE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_strict_soft_prev_session_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the roadmap-aligned strict no-SMT profile with prev-session-anchor softened into score pressure.

    This helper preserves the current strict context stack while converting
    prev-session anchor mismatches from sweep-stage hard rejects into
    shift-stage score pressure so the roadmap can measure density vs quality.
    """
    params = {**_DEFAULT_PARAMS, **_ICT_STRICT_SOFT_PREV_SESSION_PROFILE_OVERRIDES}
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_complete_soft_prev_session_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the full ICT stack with previous-session anchors softened into score pressure.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_COMPLETE_SOFT_PREV_SESSION_PROFILE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first 500-trades roadmap lite reversal baseline.

    This lighter lane keeps the reversal core aligned to the ICT PDF path
    (sweep -> structure shift -> displacement -> FVG revisit) while removing
    the heaviest pre-arm context filters identified by the strict-frontier
    funnel instrumentation. SMT remains enabled by default as the only paired
    guardrail while the lane works toward materially higher trade density.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_PROFILE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_quick_density_repair_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the quick density-repair branch from the 500-trades roadmap.

    This profile applies the lowest-risk four-parameter repair bundle:
    shorter structure lookback, longer sweep recovery, smaller FVG minimum gap,
    and no revisit-depth requirement.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUICK_DENSITY_REPAIR_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_quick_swing_structure_repair_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first structure-architecture repair branch for the 500-trades roadmap.

    This profile keeps the quick density-repair bundle but swaps rolling
    structure references for confirmed swing highs/lows so the sweep->shift
    handoff can be tested on a more ICT-like structure definition.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUICK_SWING_STRUCTURE_REPAIR_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_fvg_fib_retracement_research_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the simplified FVG + Fibonacci retracement research lane.

    This branch keeps only the core sweep -> shift -> FVG sequence, then
    hard-requires the delivery FVG to sit inside the requested retracement
    window before a setup can arm. Other delivery arrays are disabled so the
    replay answers the narrower research question directly.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_FVG_FIB_RETRACEMENT_RESEARCH_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_relaxed_smt_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust SMT extension on top of the lite ICT reversal baseline.

    The initial 500-trades SMT-density pass shows that keeping SMT enabled while
    relaxing the divergence threshold to `0.0015` improves both trade count and
    return relative to the first lite reversal baseline.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_RELAXED_SMT_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_relaxed_smt_looser_sweep_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust geometry extension on top of the relaxed-SMT lite baseline.

    The first lite-geometry round shows that loosening the sweep threshold to
    `0.0006` improves both trade count and total return relative to the relaxed-SMT
    lite baseline while keeping the rest of the lighter reversal stack unchanged.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first robust retest extension on top of the lite geometry frontier.

    The active lite frontier now combines the faster retest promotion with the
    stronger FVG-gap setting `0.0010`, which improves total return while
    preserving the `18-trade` branch.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_RELAXED_SMT_LOOSER_SWEEP_FASTER_RETEST_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_dual_speed_recovery_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build an exploratory dual-speed reversal branch on top of the active lite frontier.

    The fast branch preserves the promoted frontier's `liq_sweep_recovery_bars = 4`,
    while the slow branch keeps qualifying sweeps alive until `slow_recovery_bars`
    so slower structure shifts can still arm without replacing the fast setup path.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_DUAL_SPEED_RECOVERY_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_continuation_density_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first positive high-density ICT branch beyond the 18-trade lite frontier.

    This branch intentionally prioritizes the 500-trades roadmap goal of
    materially higher activity while still keeping the lane positive. It layers
    continuation refreshes onto the dual-speed recovery branch and reopens a
    small amount of structure / FVG geometry to admit more valid first-touch
    retests without reverting to the previously rejected fully loose density
    repair path.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_CONTINUATION_DENSITY_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a pure reversal-FVG branch that balances the sparse quality lane and
    the much denser continuation lane.

    This branch keeps the core ICT sequence intact:
    sweep -> reversal shift -> FVG -> FVG retest entry,
    while using a slower recovery window and slightly lighter FVG geometry so
    the reversal path can produce materially more trades without inheriting the
    continuation lane's much lower-quality tail.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the stronger long-only reversal lane found in the latest mixed-lane sweep.

    Relative to the balanced reversal baseline, this candidate keeps the same
    reversal-only architecture but loosens the liquidity-sweep threshold just
    enough to add materially more long entries without degrading the combined
    asymmetric lane's return profile.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_timing_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the stronger long-only reversal timing candidate from the latest density sweep.

    Relative to the first refined long-only branch, this candidate enters one
    bar earlier on FVG revisits. That small timing change lifted both trade
    count and return in the latest asymmetric mixed-lane replay and improved the
    quick walk-forward readback.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_TIMING_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current best long-only reversal refinement from the post-timing sweep.

    This candidate keeps the earlier FVG revisit timing from the delay-3 branch
    and slightly loosens the sweep threshold to `0.00035`, which lifted both
    full-sample trade count and return while preserving the positive
    walk-forward readback in the latest mixed-lane replay.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_capacity_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the denser sibling to the current best long-only sweep035 refinement.

    This branch keeps the improved sweep035 geometry intact and adds a second
    pending slot so the long leg can keep more overlapping setups alive. In the
    latest density sweep it produced the strongest trade-count gain that still
    preserved a positive validation readback.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_CAPACITY_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure11_pending3_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the stronger structure-11 / pending-3 sibling to the sweep035 capacity branch.

    This variant keeps the sweep035 long timing intact, relaxes the structure
    window by one bar, and adds a third pending slot. In the latest asymmetric
    replay it improved both density and walk-forward quality versus the prior
    sweep035-capacity long candidate.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE11_PENDING3_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_pending3_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the structure-10 sibling to the structure-11/pending-3 long repair.

    This branch narrows the reversal structure window by one more bar while
    keeping the broader pending capacity and sweep035 timing intact. In the
    latest asymmetric lane replay it added long-side activity with nearly flat
    full-sample return and still-positive walk-forward holdouts.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_PENDING3_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_depth04_pending3_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the depth-0.4 sibling to the structure-10/pending-3 long repair.

    This branch keeps the stronger structure-10 shift geometry intact while
    allowing a shallower 40% FVG revisit. In the latest asymmetric replay it
    improved both balanced and density mixed-lane returns without giving back
    the positive walk-forward holdout readback.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_DEPTH04_PENDING3_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_pending3_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the gap-0.0002 sibling to the structure-10 / depth-0.4 long repair.

    This branch keeps the promoted structure-10 and 40%-revisit geometry intact
    while slightly lowering the minimum FVG gap. In the latest asymmetric mixed-
    lane replay that small zone-width relaxation improved both full-sample
    return and holdout quality without materially inflating drawdown.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_PENDING3_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_pending3_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the intrabar-shift sibling to the promoted gap-0.0002 / depth-0.4 long repair.

    This branch keeps the current long geometry intact and only reopens a very
    small class of displacement bars that wick through structure, close back
    near the break, and finish high in the candle. In the latest asymmetric
    replay that rescued a handful of otherwise-expired long shifts and improved
    both balanced and density full-sample returns without hurting holdout
    stability.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_INTRABAR020_CP070_PENDING3_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_qualified_reversal_balance_long_refined_sweep035_structure10_gap020_depth04_intrabar020_cp070_tp45_pending3_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the TP-4.5 sibling to the promoted intrabar long repair.

    This branch keeps the repaired long shift conversion unchanged and only lets
    the long lane hold for a 4.5R target. In the latest asymmetric replay that
    produced the best combined full-sample and holdout results without changing
    trade count materially.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_QUALIFIED_REVERSAL_BALANCE_LONG_REFINED_SWEEP035_STRUCTURE10_GAP020_DEPTH04_INTRABAR020_CP070_TP45_PENDING3_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_baseline_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first formal ICT-core density baseline for the 400-trades target.

    This branch keeps the mechanical ICT sequence intact while stripping the
    full-context stack back to the bare core and reopening the geometry enough
    to test whether the desired trade density is structurally reachable on the
    full regularized dataset.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_BASELINE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_only_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the ICT-core density baseline with the long side disabled.

    This helper exists as a diagnostic branch so direction asymmetry can be
    measured directly without silently changing the rest of the high-density
    core lane.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_ONLY_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_stat_bias_candidate_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first short-only ICT-core repair that restores positive edge.

    This branch keeps the dense ICT-core entry architecture but restricts it to
    short-side setups, then restores quality with daily-bias gating plus a
    tighter stop / modestly higher target profile.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STAT_BIAS_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_bias_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the higher-density short-only ICT-core repair with structure bias.

    This branch trades only the short side, restores a broader structure-based
    daily-bias filter, and tightens the stop so the density lane can recover a
    positive edge without collapsing back to the very low-frequency benchmark.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_BIAS_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_bias_lb6_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the stronger structure-bias short repair discovered in the mixed-lane follow-up sweep.

    Relative to the first structure-bias short repair, extending the structure
    daily-bias lookback to 6 preserved almost all of the higher-density short
    activity while materially improving the asymmetric mixed-lane return.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_BIAS_LB6_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the stronger return-first short repair found in the post-lb6 local sweep.

    This version keeps the structure daily-bias repair from the lb6 candidate,
    then slightly relaxes sweep geometry and structure timing so the asymmetric
    mixed lane can lift return and profit factor without materially collapsing
    trade count.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_density_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the higher-density refinement adjacent to the stronger return-first short repair.

    Relative to the return-first refinement, this version extends recovery
    timing and keeps the slightly looser sweep threshold so the short leg can
    add more activity while staying meaningfully positive inside the mixed lane.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_DENSITY_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_recovery_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current best return-first short repair on the complete history.

    This version keeps the structure-bias refinement architecture but pairs the
    slightly looser sweep threshold with the extended recovery window, which is
    currently the strongest asymmetric mixed-lane short leg by return and PF.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_recovery_sl135_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the tighter-stop sibling to the refined recovery short repair.

    This branch keeps the refined short recovery geometry intact and only pulls
    the stop from 1.5 ATR to 1.35 ATR. In the latest asymmetric replay that
    improved return, PF, and holdout quality without changing trade count.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_SL135_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the robustness-first sibling to the tighter-stop refined recovery short repair.

    This branch keeps the 1.35 ATR stop but extends the structure daily-bias
    lookback from 6 to 8. On the complete-history asymmetric replay that gave
    up a few trades while materially improving PF and walk-forward holdout.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_RECOVERY_SL135_DAILY_BIAS_LB8_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_capacity_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the denser adjacent repair to the return-first refined recovery candidate.

    This version adds one extra pending slot on top of the refined recovery
    branch so the short leg can reclaim more overlapping sweeps while keeping
    the improved structure and recovery geometry intact.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the denser pending-4 sibling to the refined capacity short repair.

    This branch keeps the current refined-capacity short geometry intact and
    adds one more pending slot. In the latest asymmetric density sweep it
    improved full-sample trade count and return without degrading the quick
    walk-forward holdout readback.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_PENDING4_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the threshold-0.000265 sibling to the pending-4 short repair.

    This branch keeps the pending-4 capacity expansion while softening the
    short-side sweep threshold one notch. In the latest density frontier it
    added activity and improved the walk-forward holdout readback versus the
    current pending-4 candidate.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the tighter-stop sibling to the threshold-0.000265 / pending-4 short repair.

    This branch preserves the current density geometry and only tightens the
    short stop to 1.35 ATR. In the latest density replay it improved both
    full-sample return and holdout performance without reducing trade count.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_SL135_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the robustness-first sibling to the denser tighter-stop short repair.

    This version preserves the threshold-0.000265 / pending-4 density geometry
    and 1.35 ATR stop, but extends the structure daily-bias lookback to 8. The
    latest asymmetric replay showed better PF and markedly stronger holdout
    performance, albeit at slightly lower trade count.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_CORE_400_SHORT_STRUCTURE_REFINED_CAPACITY_THRESHOLD0265_PENDING4_SL135_DAILY_BIAS_LB8_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_lite_reversal_regime_mtf_alignment_profile_params(
    *,
    enable_smt: bool = True,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a regime-aware and 1H-aligned reversal profile.

    The low-volatility baseline stays close to the sparse quality lane, while
    high-volatility / high-trend bars can loosen selected FVG and SMT gates.
    Higher-timeframe alignment then acts as the guardrail so only 5m setups that
    agree with the 1H structure receive the more permissive treatment.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_LITE_REVERSAL_REGIME_MTF_ALIGNMENT_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first 1m-canonical top-down ICT continuation lane.

    This profile keeps the execution risk standards from the local ICT research
    lane while moving direction selection to `1D/4H/1H` and moving execution to
    `15m -> 5m -> 1m`.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_quality_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the first quality-tightened candidate on top of the 1m MTF continuation baseline.

    This candidate keeps the top-down continuation lane but tightens the 15m
    setup gap and the 5m confirmation structure/body rules so the research lane
    can trade materially less often while improving gross signal quality.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_QUALITY_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_execution_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the repaired execution candidate for the 1m MTF continuation lane.

    This candidate keeps the repaired `15m sweep -> shift -> FVG/OB` setup
    logic intact while tightening the execution stack where the latest
    calibration found the cleanest gains: a slightly broader 5m confirmation
    structure window and a shorter 1m trigger-expiry window.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_EXECUTION_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current setup-plus-execution candidate for the 1m MTF continuation lane.

    This candidate keeps the repaired execution stack while applying the best
    surviving 15m setup refinements from the latest fee-corrected calibration:
    a slightly smaller displacement-body requirement and a modestly wider FVG
    gap floor.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_SETUP_EXECUTION_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_timing_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current timing-filtered candidate for the 1m MTF continuation lane.

    This candidate keeps the stronger setup-plus-execution stack intact and then
    removes the weakest execution hour from the latest timing study by excluding
    `13:00 ET` entries.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_TIMING_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the current full-data long-only candidate for the 1m MTF continuation lane.

    This candidate is anchored to the complete regularized 1m history. It keeps
    the tighter quality setup stack, disables shorts entirely, and narrows entry
    timing to the stronger `10-12 ET` and `14-15 ET` windows.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_REGULARIZED_LONG_ONLY_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


def build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params(
    *,
    enable_smt: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the AM-only full-data long-only candidate for the 1m MTF continuation lane.

    This version keeps the same complete-data quality stack but restricts
    entries to the morning `9-12 ET` window when the regularized replay favors
    higher trade quality over raw signal count.
    """
    params = {
        **_DEFAULT_PARAMS,
        **_ICT_MTF_TOPDOWN_CONTINUATION_REGULARIZED_LONG_ONLY_AM_CANDIDATE_OVERRIDES,
    }
    params["use_smt_filter"] = bool(enable_smt)
    if overrides:
        params.update(overrides)
    return params


@dataclass
class _PendingSetup:
    direction: int
    sweep_index: int
    sweep_level: float
    sweep_reference_level: float
    sweep_depth: float
    structure_level: float
    shift_extreme: float
    expiry_index: int
    primary_expiry_index: int
    state: str = "swept"
    armed_index: int | None = None
    zone_kind: str | None = None
    zone_lower: float | None = None
    zone_upper: float | None = None
    score: float = 0.0
    retest_touches: int = 0
    retest_seen: bool = False
    recovery_phase: str = "fast"
    premium_discount_mismatch: bool = False
    prev_session_anchor_mismatch: bool = False
    entry_attempts: int = 0
    zone_index: int | None = None
    continuation_refreshes: int = 0


@dataclass
class _MTFSetup:
    direction: int
    setup_time: pd.Timestamp
    source_index: int
    zone_kind: str
    zone_lower: float
    zone_upper: float
    sweep_level: float
    shift_extreme: float
    expiry_time: pd.Timestamp
    score: float
    confirm_time: pd.Timestamp | None = None
    trigger_expiry_time: pd.Timestamp | None = None
    entry_time: pd.Timestamp | None = None
    timing_blocked: bool = False
    zone_touch_time: pd.Timestamp | None = None
    confirm_touch_expiry_time: pd.Timestamp | None = None
    trigger_touch_expiry_time: pd.Timestamp | None = None
    fast_retest_armed: bool = False


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _same_trading_day(
    ts1: pd.Timestamp,
    ts2: pd.Timestamp,
    *,
    timezone: str = "America/New_York",
) -> bool:
    day1 = _ensure_utc(ts1).tz_convert(timezone).date()
    day2 = _ensure_utc(ts2).tz_convert(timezone).date()
    return day1 == day2


def _normalize_int_filter_values(
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None

    raw_items: list[Any]
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        cleaned = cleaned.replace("[", "").replace("]", "").replace(" ", "")
        raw_items = [item for item in cleaned.split(",") if item != ""]
    elif isinstance(value, (int, np.integer)):
        raw_items = [int(value)]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        return None

    resolved: list[int] = []
    for item in raw_items:
        if isinstance(item, str) and "-" in item:
            start_text, end_text = item.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            resolved.extend(range(start, end + 1))
            continue

        try:
            resolved.append(int(item))
        except (TypeError, ValueError):
            continue

    filtered = sorted({item for item in resolved if minimum <= item <= maximum})
    if not filtered:
        return None
    return tuple(filtered)


def _entry_timing_gate(
    ts: pd.Timestamp,
    *,
    timezone: str,
    allowed_weekdays: tuple[int, ...] | None,
    allowed_hours: tuple[int, ...] | None,
) -> tuple[bool, str | None]:
    if allowed_weekdays is None and allowed_hours is None:
        return True, None

    local_stamp = _ensure_utc(ts).tz_convert(timezone)
    if allowed_weekdays is not None and local_stamp.weekday() not in allowed_weekdays:
        return False, "weekday"
    if allowed_hours is not None and local_stamp.hour not in allowed_hours:
        return False, "hour"
    return True, None


def _in_trade_session(
    ts: pd.Timestamp,
    enabled: bool,
    london_open: int,
    london_close: int,
    ny_open: int,
    ny_close: int,
) -> bool:
    if not enabled:
        return True
    hour = _ensure_utc(ts).hour
    in_london = london_open <= hour < london_close
    in_new_york = ny_open <= hour < ny_close
    return in_london or in_new_york


def _in_kill_zone(
    ts: pd.Timestamp,
    enabled: bool,
    timezone_name: str,
    windows: list[tuple[int, int]],
) -> bool:
    if not enabled:
        return True

    stamp = _ensure_utc(ts)
    try:
        local_stamp = stamp.tz_convert(timezone_name)
    except Exception:
        local_stamp = stamp

    minute_of_day = local_stamp.hour * 60 + local_stamp.minute
    for start_hour, end_hour in windows:
        start_minute = start_hour * 60
        end_minute = end_hour * 60
        if start_minute <= minute_of_day < end_minute:
            return True
    return False


def _in_macro_window(
    ts: pd.Timestamp,
    enabled: bool,
    timezone_name: str,
    windows: list[tuple[int, int, int, int]],
) -> bool:
    if not enabled:
        return True

    stamp = _ensure_utc(ts)
    try:
        local_stamp = stamp.tz_convert(timezone_name)
    except Exception:
        local_stamp = stamp

    minute_of_day = local_stamp.hour * 60 + local_stamp.minute
    for start_hour, start_minute, end_hour, end_minute in windows:
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        if start_total <= minute_of_day < end_total:
            return True
    return False


def _classify_dealing_array_window(
    ts: pd.Timestamp,
    enabled: bool,
    timezone_name: str,
    imbalance_windows: list[tuple[int, int, int, int]],
    structural_windows: list[tuple[int, int, int, int]],
) -> str:
    if not enabled:
        return "neutral"

    stamp = _ensure_utc(ts)
    try:
        local_stamp = stamp.tz_convert(timezone_name)
    except Exception:
        local_stamp = stamp

    minute_of_day = local_stamp.hour * 60 + local_stamp.minute
    for start_hour, start_minute, end_hour, end_minute in imbalance_windows:
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        if start_total <= minute_of_day < end_total:
            return "imbalance"

    for start_hour, start_minute, end_hour, end_minute in structural_windows:
        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute
        if start_total <= minute_of_day < end_total:
            return "structural"

    return "neutral"


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def _compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(period, min_periods=1).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.rolling(period, min_periods=1).mean() / atr
    minus_di = 100.0 * minus_dm.rolling(period, min_periods=1).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.rolling(period, min_periods=1).mean().fillna(0.0)


def _compute_intraday_regime(
    df: pd.DataFrame,
    *,
    enabled: bool,
    atr_period: int,
    atr_pct_window: int,
    atr_high_mult: float,
    adx_period: int,
    adx_trend_threshold: float,
) -> tuple[pd.Series, dict[str, int]]:
    if not enabled:
        return pd.Series(0, index=df.index, dtype=int), {
            "high_regime_bars": 0,
            "volatility_high_regime_bars": 0,
            "trend_high_regime_bars": 0,
        }

    if df.empty:
        return pd.Series(dtype=int, index=df.index), {
            "high_regime_bars": 0,
            "volatility_high_regime_bars": 0,
            "trend_high_regime_bars": 0,
        }

    atr = _compute_atr(df, max(1, atr_period))
    close = df["Close"].astype(float).abs().replace(0.0, np.nan)
    atr_pct = (atr / close).fillna(0.0)
    atr_window = max(1, atr_pct_window)
    atr_baseline = atr_pct.rolling(atr_window, min_periods=min(5, atr_window)).median()
    high_volatility = (atr_baseline > 0.0) & (atr_pct >= atr_baseline * max(1.0, atr_high_mult))
    adx = _compute_adx(df, max(1, adx_period))
    high_trend = adx >= adx_trend_threshold
    high_regime = (high_volatility | high_trend).astype(int)
    return high_regime, {
        "high_regime_bars": int(high_regime.sum()),
        "volatility_high_regime_bars": int(high_volatility.fillna(False).sum()),
        "trend_high_regime_bars": int(high_trend.fillna(False).sum()),
    }


def _compute_higher_timeframe_structure_bias(
    df: pd.DataFrame,
    *,
    enabled: bool,
    timeframe: str,
    lookback: int,
    swing_threshold: int,
) -> pd.Series:
    if not enabled:
        return pd.Series(0, index=df.index, dtype=int)

    if df.empty:
        return pd.Series(dtype=int, index=df.index)

    higher = (
        df[["Open", "High", "Low", "Close"]]
        .resample(str(timeframe).lower(), label="right", closed="right")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
        )
        .dropna()
    )
    if higher.empty:
        return pd.Series(0, index=df.index, dtype=int)

    carried_bias = 0
    bias_values: list[int] = []
    for pos in range(len(higher)):
        if pos == 0:
            bias_values.append(0)
            continue
        prev_close = float(higher["Close"].iat[pos - 1])
        swing_high = _latest_confirmed_swing_level(
            higher["High"],
            pos,
            lookback=max(1, lookback),
            threshold=max(1, swing_threshold),
            swing_type="high",
        )
        swing_low = _latest_confirmed_swing_level(
            higher["Low"],
            pos,
            lookback=max(1, lookback),
            threshold=max(1, swing_threshold),
            swing_type="low",
        )
        if pd.notna(swing_high) and prev_close > float(swing_high):
            carried_bias = 1
        elif pd.notna(swing_low) and prev_close < float(swing_low):
            carried_bias = -1
        bias_values.append(carried_bias)

    higher_bias = pd.Series(bias_values, index=higher.index, dtype=int)
    aligned = higher_bias.reindex(df.index, method="ffill").fillna(0)
    return aligned.astype(int)


def _compute_daily_bias(
    df: pd.DataFrame,
    *,
    enabled: bool,
    mode: str,
    lookback: int,
    swing_threshold: int,
    bull_threshold: float,
    bear_threshold: float,
    trading_timezone: str = "America/New_York",
) -> pd.Series:
    if not enabled:
        return pd.Series(0, index=df.index, dtype=int)

    if df.empty:
        return pd.Series(dtype=int, index=df.index)

    session_dates = pd.Index(
        [_ensure_utc(ts).tz_convert(trading_timezone).date() for ts in df.index],
        name="session_date",
    )
    daily = (
        df.assign(session_date=session_dates)
        .groupby("session_date")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
        )
    )

    bias_mode = str(mode).strip().lower()
    if bias_mode not in {"statistical", "structure"}:
        bias_mode = "statistical"

    bias_by_session: dict[Any, int] = {}
    sessions = list(daily.index)
    carried_bias = 0
    for pos, session in enumerate(sessions):
        if pos == 0:
            bias_by_session[session] = 0
            continue

        prev_day = daily.iloc[pos - 1]
        if bias_mode == "structure":
            prev_close = float(prev_day["Close"])
            swing_high = _latest_confirmed_swing_level(
                daily["High"],
                pos,
                lookback=lookback,
                threshold=max(1, swing_threshold),
                swing_type="high",
            )
            swing_low = _latest_confirmed_swing_level(
                daily["Low"],
                pos,
                lookback=lookback,
                threshold=max(1, swing_threshold),
                swing_type="low",
            )

            if pd.notna(swing_high) and prev_close > float(swing_high):
                carried_bias = 1
            elif pd.notna(swing_low) and prev_close < float(swing_low):
                carried_bias = -1
            bias_by_session[session] = carried_bias
            continue

        prior_end = pos
        prior_start = max(0, prior_end - lookback)
        window = daily.iloc[prior_start:prior_end]

        if window.empty:
            bias_by_session[session] = 0
            continue

        range_high = float(window["High"].max())
        range_low = float(window["Low"].min())
        range_width = range_high - range_low
        if range_width <= 0:
            bias_by_session[session] = 0
            continue

        close_pos = (float(prev_day["Close"]) - range_low) / range_width
        bullish = float(prev_day["Close"]) > float(prev_day["Open"]) and close_pos >= bull_threshold
        bearish = float(prev_day["Close"]) < float(prev_day["Open"]) and close_pos <= bear_threshold

        if bullish and not bearish:
            bias_by_session[session] = 1
        elif bearish and not bullish:
            bias_by_session[session] = -1
        else:
            bias_by_session[session] = 0

    return pd.Series([bias_by_session.get(day, 0) for day in session_dates], index=df.index, dtype=int)


def _compute_premium_discount_context(
    df: pd.DataFrame,
    *,
    enabled: bool,
    lookback: int,
    neutral_band: float,
    trading_timezone: str = "America/New_York",
) -> pd.Series:
    if not enabled:
        return pd.Series(0, index=df.index, dtype=int)

    if df.empty:
        return pd.Series(dtype=int, index=df.index)

    session_dates = pd.Index(
        [_ensure_utc(ts).tz_convert(trading_timezone).date() for ts in df.index],
        name="session_date",
    )
    daily = (
        df.assign(session_date=session_dates)
        .groupby("session_date")
        .agg(
            High=("High", "max"),
            Low=("Low", "min"),
        )
    )

    side_by_session: dict[Any, tuple[float | None, float | None]] = {}
    sessions = list(daily.index)
    for pos, session in enumerate(sessions):
        prior_end = pos
        prior_start = max(0, prior_end - lookback)
        window = daily.iloc[prior_start:prior_end]
        if window.empty:
            side_by_session[session] = (None, None)
            continue

        range_high = float(window["High"].max())
        range_low = float(window["Low"].min())
        range_width = range_high - range_low
        if range_width <= 0:
            side_by_session[session] = (None, None)
            continue
        side_by_session[session] = (range_low, range_high)

    contexts: list[int] = []
    for ts, close in zip(df.index, df["Close"], strict=False):
        session = _ensure_utc(ts).tz_convert(trading_timezone).date()
        range_low, range_high = side_by_session.get(session, (None, None))
        if range_low is None or range_high is None:
            contexts.append(0)
            continue

        position = (float(close) - range_low) / (range_high - range_low)
        if position <= 0.5 - neutral_band:
            contexts.append(1)   # discount
        elif position >= 0.5 + neutral_band:
            contexts.append(-1)  # premium
        else:
            contexts.append(0)

    return pd.Series(contexts, index=df.index, dtype=int)


def _compute_amd_path_bias(
    df: pd.DataFrame,
    *,
    enabled: bool,
    accumulation_bars: int,
    manipulation_threshold: float,
    require_midpoint_reclaim: bool,
    trading_timezone: str = "America/New_York",
) -> tuple[pd.Series, dict[str, int]]:
    if not enabled:
        return pd.Series(0, index=df.index, dtype=int), {
            "bullish_sessions": 0,
            "bearish_sessions": 0,
        }

    if df.empty:
        return pd.Series(dtype=int, index=df.index), {
            "bullish_sessions": 0,
            "bearish_sessions": 0,
        }

    session_dates = pd.Index(
        [_ensure_utc(ts).tz_convert(trading_timezone).date() for ts in df.index],
        name="session_date",
    )
    session_series = pd.Series(session_dates, index=df.index)
    bias = pd.Series(0, index=df.index, dtype=int)
    bullish_sessions = 0
    bearish_sessions = 0

    for session in pd.unique(session_series):
        session_positions = np.flatnonzero(session_series.to_numpy() == session)
        if len(session_positions) <= accumulation_bars:
            continue

        accumulation_positions = session_positions[:accumulation_bars]
        accumulation_high = float(df["High"].iloc[accumulation_positions].max())
        accumulation_low = float(df["Low"].iloc[accumulation_positions].min())
        midpoint = (accumulation_high + accumulation_low) / 2.0

        bullish_manip_seen = False
        bearish_manip_seen = False
        session_bias = 0

        for pos in session_positions[accumulation_bars:]:
            high = float(df["High"].iat[pos])
            low = float(df["Low"].iat[pos])
            close = float(df["Close"].iat[pos])

            if session_bias == 0 and not bullish_manip_seen:
                bullish_manip_seen = bool(
                    low < accumulation_low * (1 - manipulation_threshold)
                    and close >= accumulation_low
                )
                if bullish_manip_seen and (not require_midpoint_reclaim or close >= midpoint):
                    session_bias = 1
                    bullish_sessions += 1

            if session_bias == 0 and not bearish_manip_seen:
                bearish_manip_seen = bool(
                    high > accumulation_high * (1 + manipulation_threshold)
                    and close <= accumulation_high
                )
                if bearish_manip_seen and (not require_midpoint_reclaim or close <= midpoint):
                    session_bias = -1
                    bearish_sessions += 1

            if session_bias == 0 and require_midpoint_reclaim:
                if bullish_manip_seen and close >= midpoint:
                    session_bias = 1
                    bullish_sessions += 1
                elif bearish_manip_seen and close <= midpoint:
                    session_bias = -1
                    bearish_sessions += 1

            bias.iat[pos] = session_bias

    return bias, {
        "bullish_sessions": bullish_sessions,
        "bearish_sessions": bearish_sessions,
    }


def _compute_prev_session_anchor_bias(
    df: pd.DataFrame,
    *,
    enabled: bool,
    tolerance: float,
    trading_timezone: str = "America/New_York",
) -> tuple[pd.Series, dict[str, int]]:
    if not enabled:
        return pd.Series(0, index=df.index, dtype=int), {
            "long_bars": 0,
            "short_bars": 0,
        }

    if df.empty:
        return pd.Series(dtype=int, index=df.index), {
            "long_bars": 0,
            "short_bars": 0,
        }

    session_dates = pd.Index(
        [_ensure_utc(ts).tz_convert(trading_timezone).date() for ts in df.index],
        name="session_date",
    )
    session_series = pd.Series(session_dates, index=df.index)
    daily = (
        df.assign(session_date=session_dates)
        .groupby("session_date")
        .agg(
            High=("High", "max"),
            Low=("Low", "min"),
        )
    )
    sessions = list(daily.index)
    prev_anchor_map: dict[Any, tuple[float | None, float | None]] = {}
    for pos, session in enumerate(sessions):
        if pos == 0:
            prev_anchor_map[session] = (None, None)
        else:
            prev_day = daily.iloc[pos - 1]
            prev_anchor_map[session] = (float(prev_day["High"]), float(prev_day["Low"]))

    running_high = df.groupby(session_series)["High"].cummax()
    running_low = df.groupby(session_series)["Low"].cummin()
    bias: list[int] = []
    long_bars = 0
    short_bars = 0

    for ts, close, run_high, run_low in zip(
        df.index,
        df["Close"],
        running_high,
        running_low,
        strict=False,
    ):
        session = _ensure_utc(ts).tz_convert(trading_timezone).date()
        prev_high, prev_low = prev_anchor_map.get(session, (None, None))
        if prev_high is None or prev_low is None:
            bias.append(0)
            continue

        prev_range = prev_high - prev_low
        if prev_range <= 0:
            bias.append(0)
            continue

        midpoint = (prev_high + prev_low) / 2.0
        buffer = prev_range * tolerance
        long_ok = float(close) <= midpoint + buffer and float(run_high) <= prev_high - buffer
        short_ok = float(close) >= midpoint - buffer and float(run_low) >= prev_low + buffer

        if long_ok and not short_ok:
            bias.append(1)
            long_bars += 1
        elif short_ok and not long_ok:
            bias.append(-1)
            short_bars += 1
        else:
            bias.append(0)

    return pd.Series(bias, index=df.index, dtype=int), {
        "long_bars": long_bars,
        "short_bars": short_bars,
    }


def _body_ratio(open_: float, high: float, low: float, close: float) -> float:
    full_range = high - low
    if full_range <= 0:
        return 0.0
    return abs(close - open_) / full_range


def _close_position_ratio(high: float, low: float, close: float) -> float:
    full_range = high - low
    if full_range <= 0:
        return 0.0
    return (close - low) / full_range


def _opposite_wick_ratio(open_: float, high: float, low: float, close: float, *, bullish: bool) -> float:
    full_range = high - low
    if full_range <= 0:
        return 0.0
    if bullish:
        return max(min(open_, close) - low, 0.0) / full_range
    return max(high - max(open_, close), 0.0) / full_range


def _detect_fvg_zone(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    bullish: bool,
    min_gap_pct: float,
) -> tuple[int, float, float] | None:
    start = max(start_index + 2, 2)
    # Align with the other delivery-array detectors by preferring the most
    # recent valid zone instead of the oldest one in the shift window.
    for idx in range(end_index, start - 1, -1):
        high_two_back = float(df["High"].iat[idx - 2])
        low_two_back = float(df["Low"].iat[idx - 2])
        current_low = float(df["Low"].iat[idx])
        current_high = float(df["High"].iat[idx])
        price = float(df["Close"].iat[idx])
        if price <= 0:
            continue

        if bullish and current_low > high_two_back:
            gap_size = current_low - high_two_back
            if gap_size / price >= min_gap_pct:
                return idx, high_two_back, current_low

        if not bullish and current_high < low_two_back:
            gap_size = low_two_back - current_high
            if gap_size / price >= min_gap_pct:
                return idx, current_high, low_two_back

    return None


def _project_reward_risk_ratio(entry_price: float, stop_price: float, target_price: float) -> float:
    reward = abs(float(target_price) - float(entry_price))
    risk = abs(float(entry_price) - float(stop_price))
    if risk <= 0:
        return 0.0
    return reward / risk


def _structure_confirmation_score(score_bos: float, score_choch: float) -> float:
    # A post-sweep reversal should not receive BOS and CHOCH credit at the same
    # time. Treat the structure-confirmation contribution as a single event.
    return max(score_bos, score_choch)


def _quality_score_bonus(
    *,
    sweep_depth: float,
    sweep_reference_level: float,
    liq_sweep_threshold: float,
    displacement_body_ratio: float,
    displacement_body_min_pct: float,
    fvg_gap_size: float,
    price: float,
    fvg_min_gap_pct: float,
    score_sweep_depth_quality: float,
    score_displacement_quality: float,
    score_fvg_gap_quality: float,
) -> float:
    def _bounded_bonus(value: float, reference: float, weight: float) -> float:
        if weight <= 0.0 or value <= 0.0 or reference <= 0.0:
            return 0.0
        normalized = min(value / reference, 2.0)
        return max(normalized - 1.0, 0.0) * weight

    safe_price = max(abs(float(price)), 1e-12)
    sweep_reference = max(abs(float(sweep_reference_level)) * liq_sweep_threshold, 1e-12)
    displacement_reference = max(
        displacement_body_min_pct if displacement_body_min_pct > 0.0 else 0.25,
        1e-12,
    )
    fvg_gap_reference = max(
        safe_price * (fvg_min_gap_pct if fvg_min_gap_pct > 0.0 else 0.0005),
        1e-12,
    )
    return (
        _bounded_bonus(sweep_depth, sweep_reference, score_sweep_depth_quality)
        + _bounded_bonus(displacement_body_ratio, displacement_reference, score_displacement_quality)
        + _bounded_bonus(fvg_gap_size, fvg_gap_reference, score_fvg_gap_quality)
    )


def _detect_order_block_zone(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    bullish: bool,
    body_min_pct: float,
    lookback: int,
) -> tuple[int, float, float] | None:
    lower_bound = max(0, max(start_index, end_index - lookback))
    for idx in range(end_index - 1, lower_bound - 1, -1):
        open_ = float(df["Open"].iat[idx])
        high = float(df["High"].iat[idx])
        low = float(df["Low"].iat[idx])
        close = float(df["Close"].iat[idx])
        body = _body_ratio(open_, high, low, close)

        if bullish and close < open_ and body >= body_min_pct:
            return idx, low, high
        if not bullish and close > open_ and body >= body_min_pct:
            return idx, low, high

    return None


def _detect_breaker_block_zone(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    bullish: bool,
    lookback: int,
) -> tuple[int, float, float] | None:
    lower_bound = max(0, max(start_index, end_index - lookback))
    close_slice = df["Close"]

    for idx in range(end_index - 1, lower_bound - 1, -1):
        open_ = float(df["Open"].iat[idx])
        high = float(df["High"].iat[idx])
        low = float(df["Low"].iat[idx])
        close = float(df["Close"].iat[idx])

        if bullish and close < open_:
            future_closes = close_slice.iloc[idx + 1 : end_index + 1]
            if not future_closes.empty and bool((future_closes > high).any()):
                return idx, low, high

        if not bullish and close > open_:
            future_closes = close_slice.iloc[idx + 1 : end_index + 1]
            if not future_closes.empty and bool((future_closes < low).any()):
                return idx, low, high

    return None


def _detect_ifvg_zone(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    *,
    bullish: bool,
    lookback: int,
) -> tuple[int, float, float] | None:
    lower_bound = max(start_index + 2, max(2, end_index - lookback))

    for idx in range(end_index - 1, lower_bound - 1, -1):
        high_two_back = float(df["High"].iat[idx - 2])
        low_two_back = float(df["Low"].iat[idx - 2])
        current_low = float(df["Low"].iat[idx])
        current_high = float(df["High"].iat[idx])

        if bullish:
            if current_high < low_two_back:
                zone_lower = current_high
                zone_upper = low_two_back
                future_closes = df["Close"].iloc[idx + 1 : end_index + 1]
                if not future_closes.empty and bool((future_closes > zone_upper).any()):
                    return idx, zone_lower, zone_upper
        else:
            if current_low > high_two_back:
                zone_lower = high_two_back
                zone_upper = current_low
                future_closes = df["Close"].iloc[idx + 1 : end_index + 1]
                if not future_closes.empty and bool((future_closes < zone_lower).any()):
                    return idx, zone_lower, zone_upper

    return None


def _zone_in_ote(
    direction: int,
    *,
    zone_lower: float,
    zone_upper: float,
    sweep_level: float,
    shift_extreme: float,
    fib_low: float,
    fib_high: float,
) -> bool:
    zone_mid = (zone_lower + zone_upper) / 2

    if direction > 0:
        swing_range = shift_extreme - sweep_level
        if swing_range <= 0:
            return False
        lower = shift_extreme - swing_range * fib_high
        upper = shift_extreme - swing_range * fib_low
        return lower <= zone_mid <= upper

    swing_range = sweep_level - shift_extreme
    if swing_range <= 0:
        return False
    lower = shift_extreme + swing_range * fib_low
    upper = shift_extreme + swing_range * fib_high
    return lower <= zone_mid <= upper


def _latest_confirmed_swing_level(
    series: pd.Series,
    idx: int,
    *,
    lookback: int,
    threshold: int,
    swing_type: str,
) -> float:
    if idx <= 0 or lookback <= 0 or threshold <= 0:
        return float("nan")

    confirmed_end = idx - threshold - 1
    if confirmed_end < threshold:
        return float("nan")

    search_start = max(threshold, idx - lookback)
    values = series.astype(float)
    mode = swing_type.lower()

    for center in range(confirmed_end, search_start - 1, -1):
        left = values.iloc[center - threshold : center]
        right = values.iloc[center + 1 : center + threshold + 1]
        if len(left) < threshold or len(right) < threshold:
            continue
        candidate = float(values.iat[center])
        if mode == "high":
            if candidate > float(left.max()) and candidate > float(right.max()):
                return candidate
        elif mode == "low":
            if candidate < float(left.min()) and candidate < float(right.min()):
                return candidate
        else:
            raise ValueError(f"Unsupported swing_type: {swing_type}")

    return float("nan")


class ICTEntryModelStrategy(BaseStrategy):
    """Mechanical ICT entry model with sweep -> shift -> FVG/OB/breaker/IFVG retest."""

    name = "ICT_Entry_Model"

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__({**_DEFAULT_PARAMS, **(params or {})})

    def _generate_mtf_topdown_continuation_signals(self, df: pd.DataFrame) -> StrategyResult:
        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)

        atr_period = int(self.params["atr_period"])
        take_profit_rr = float(self.params["take_profit_rr"])
        stop_loss_atr_mult = float(self.params["stop_loss_atr_mult"])
        min_reward_risk_ratio = float(self.params["min_reward_risk_ratio"])
        allow_long_entries = bool(self.params.get("allow_long_entries", True))
        allow_short_entries = bool(self.params.get("allow_short_entries", True))
        score_liquidity_sweep = float(self.params["score_liquidity_sweep"])
        score_order_block = float(self.params["score_order_block"])
        score_fvg = float(self.params["score_fvg"])
        score_bos = float(self.params["score_bos"])
        score_choch = float(self.params["score_choch"])
        score_sweep_depth_quality = float(self.params["score_sweep_depth_quality"])
        score_displacement_quality = float(self.params["score_displacement_quality"])
        score_fvg_gap_quality = float(self.params["score_fvg_gap_quality"])
        min_score_to_trade = float(self.params["min_score_to_trade"])
        liq_sweep_threshold = float(self.params["liq_sweep_threshold"])
        liq_sweep_reclaim_ratio = float(self.params["liq_sweep_reclaim_ratio"])
        ob_lookback = int(self.params["ob_lookback"])
        ob_body_min_pct = float(self.params["ob_body_min_pct"])
        mtf_bias_lookback = int(self.params["mtf_bias_lookback"])
        mtf_bias_swing_threshold = int(self.params["mtf_bias_swing_threshold"])
        mtf_setup_structure_lookback = int(self.params["mtf_setup_structure_lookback"])
        mtf_setup_swing_threshold = int(self.params["mtf_setup_swing_threshold"])
        mtf_setup_fvg_min_gap_pct = float(self.params["mtf_setup_fvg_min_gap_pct"])
        mtf_setup_displacement_body_min_pct = float(
            self.params["mtf_setup_displacement_body_min_pct"]
        )
        mtf_setup_zone_expiry_bars = max(1, int(self.params["mtf_setup_zone_expiry_bars"]))
        mtf_confirmation_structure_lookback = int(
            self.params["mtf_confirmation_structure_lookback"]
        )
        mtf_confirmation_close_ratio = float(self.params["mtf_confirmation_close_ratio"])
        mtf_confirmation_body_min_pct = float(self.params["mtf_confirmation_body_min_pct"])
        mtf_confirmation_rejection_wick_ratio = float(
            self.params["mtf_confirmation_rejection_wick_ratio"]
        )
        mtf_confirmation_rejection_volume_ratio = float(
            self.params["mtf_confirmation_rejection_volume_ratio"]
        )
        mtf_trigger_structure_lookback = int(self.params["mtf_trigger_structure_lookback"])
        mtf_trigger_close_ratio = float(self.params["mtf_trigger_close_ratio"])
        mtf_trigger_body_min_pct = float(self.params["mtf_trigger_body_min_pct"])
        mtf_trigger_rejection_wick_ratio = float(self.params["mtf_trigger_rejection_wick_ratio"])
        mtf_trigger_rejection_volume_ratio = float(self.params["mtf_trigger_rejection_volume_ratio"])
        mtf_trigger_expiry_bars = max(1, int(self.params["mtf_trigger_expiry_bars"]))
        mtf_touch_confirm_bars = max(0, int(self.params["mtf_touch_confirm_bars"]))
        mtf_touch_trigger_bars = max(0, int(self.params["mtf_touch_trigger_bars"]))
        mtf_volume_lookback = max(1, int(self.params["mtf_volume_lookback"]))
        mtf_fast_retest_entry_enabled = bool(self.params["mtf_fast_retest_entry_enabled"])
        mtf_fast_retest_displacement_body_min_pct = float(
            self.params["mtf_fast_retest_displacement_body_min_pct"]
        )
        mtf_fast_retest_displacement_close_ratio = float(
            self.params["mtf_fast_retest_displacement_close_ratio"]
        )
        mtf_fast_retest_min_close_ratio = float(self.params["mtf_fast_retest_min_close_ratio"])
        mtf_max_stop_distance_atr_mult = float(self.params["mtf_max_stop_distance_atr_mult"])
        mtf_neutral_hourly_allows_high_quality = bool(
            self.params["mtf_neutral_hourly_allows_high_quality"]
        )
        mtf_neutral_hourly_min_score = float(self.params["mtf_neutral_hourly_min_score"])
        mtf_timing_timezone = str(self.params["mtf_timing_timezone"])
        mtf_allowed_entry_weekdays = _normalize_int_filter_values(
            self.params.get("mtf_allowed_entry_weekdays"),
            minimum=0,
            maximum=6,
        )
        mtf_allowed_entry_hours = _normalize_int_filter_values(
            self.params.get("mtf_allowed_entry_hours"),
            minimum=0,
            maximum=23,
        )

        atr = _compute_atr(df, atr_period)
        daily_bias = _compute_higher_timeframe_structure_bias(
            df,
            enabled=True,
            timeframe=str(self.params["mtf_bias_daily_timeframe"]),
            lookback=mtf_bias_lookback,
            swing_threshold=mtf_bias_swing_threshold,
        )
        four_hour_bias = _compute_higher_timeframe_structure_bias(
            df,
            enabled=True,
            timeframe=str(self.params["mtf_bias_4h_timeframe"]),
            lookback=mtf_bias_lookback,
            swing_threshold=mtf_bias_swing_threshold,
        )
        hourly_bias = _compute_higher_timeframe_structure_bias(
            df,
            enabled=True,
            timeframe=str(self.params["mtf_bias_1h_timeframe"]),
            lookback=mtf_bias_lookback,
            swing_threshold=mtf_bias_swing_threshold,
        )
        frame_15m = resample_ohlcv(
            df,
            str(self.params["mtf_setup_timeframe"]),
            label="right",
            closed="right",
        )
        frame_5m = resample_ohlcv(
            df,
            str(self.params["mtf_confirmation_timeframe"]),
            label="right",
            closed="right",
        )
        if frame_15m.empty or frame_5m.empty:
            return StrategyResult(
                entries_long=entries_long,
                exits_long=exits_long,
                entries_short=entries_short,
                exits_short=exits_short,
                metadata={"warning": "mtf resample produced empty frame"},
            )

        def _frame_step(frame: pd.DataFrame, fallback: pd.Timedelta) -> pd.Timedelta:
            if len(frame.index) >= 2:
                diffs = frame.index.to_series().diff().dropna()
                if not diffs.empty:
                    return pd.Timedelta(diffs.median())
            return fallback

        setup_step = _frame_step(frame_15m, pd.Timedelta(minutes=15))
        execution_step = _frame_step(df, pd.Timedelta(minutes=1))
        frame_15m_high_ref = (
            frame_15m["High"]
            .rolling(mtf_setup_structure_lookback, min_periods=mtf_setup_structure_lookback)
            .max()
            .shift(1)
        )
        frame_15m_low_ref = (
            frame_15m["Low"]
            .rolling(mtf_setup_structure_lookback, min_periods=mtf_setup_structure_lookback)
            .min()
            .shift(1)
        )
        frame_5m_high_ref = (
            frame_5m["High"]
            .rolling(
                mtf_confirmation_structure_lookback,
                min_periods=mtf_confirmation_structure_lookback,
            )
            .max()
            .shift(1)
        )
        frame_5m_low_ref = (
            frame_5m["Low"]
            .rolling(
                mtf_confirmation_structure_lookback,
                min_periods=mtf_confirmation_structure_lookback,
            )
            .min()
            .shift(1)
        )
        trigger_high_ref = (
            df["High"]
            .rolling(mtf_trigger_structure_lookback, min_periods=mtf_trigger_structure_lookback)
            .max()
            .shift(1)
        )
        trigger_low_ref = (
            df["Low"]
            .rolling(mtf_trigger_structure_lookback, min_periods=mtf_trigger_structure_lookback)
            .min()
            .shift(1)
        )
        frame_5m_volume_ref = (
            frame_5m["Volume"].rolling(mtf_volume_lookback, min_periods=1).median().shift(1)
            if "Volume" in frame_5m.columns
            else pd.Series(np.nan, index=frame_5m.index, dtype=float)
        )
        trigger_volume_ref = (
            df["Volume"].rolling(mtf_volume_lookback, min_periods=1).median().shift(1)
            if "Volume" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )

        daily_bias_15m = daily_bias.reindex(frame_15m.index, method="ffill").fillna(0).astype(int)
        four_hour_bias_15m = (
            four_hour_bias.reindex(frame_15m.index, method="ffill").fillna(0).astype(int)
        )
        hourly_bias_15m = hourly_bias.reindex(frame_15m.index, method="ffill").fillna(0).astype(int)

        metadata: dict[str, Any] = {
            "mtf_enabled": True,
            "mtf_execution_timeframe": str(self.params["mtf_execution_timeframe"]),
            "mtf_bias_daily_timeframe": str(self.params["mtf_bias_daily_timeframe"]),
            "mtf_bias_4h_timeframe": str(self.params["mtf_bias_4h_timeframe"]),
            "mtf_bias_1h_timeframe": str(self.params["mtf_bias_1h_timeframe"]),
            "mtf_setup_timeframe": str(self.params["mtf_setup_timeframe"]),
            "mtf_confirmation_timeframe": str(self.params["mtf_confirmation_timeframe"]),
            "mtf_trigger_timeframe": str(self.params["mtf_trigger_timeframe"]),
            "mtf_daily_blocked": 0,
            "mtf_4h_blocked": 0,
            "mtf_1h_blocked": 0,
            "mtf_direction_long_allowed": 0,
            "mtf_direction_short_allowed": 0,
            "mtf_15m_setups": 0,
            "mtf_5m_confirms": 0,
            "mtf_1m_triggers": 0,
            "mtf_setup_fvg_zones": 0,
            "mtf_setup_ob_zones": 0,
            "mtf_setup_missing_zone": 0,
            "mtf_setup_missing_context": 0,
            "mtf_setup_score_filtered": 0,
            "mtf_setup_expired": 0,
            "mtf_confirm_expired": 0,
            "mtf_zone_touches": 0,
            "mtf_touch_activated_confirms": 0,
            "mtf_touch_activated_triggers": 0,
            "mtf_fast_retest_confirms": 0,
            "mtf_fast_retest_entries": 0,
            "mtf_neutral_1h_high_quality_setups": 0,
            "mtf_weekday_blocked": 0,
            "mtf_hour_blocked": 0,
            "rr_gate_enabled": min_reward_risk_ratio > 0.0,
            "rr_filtered_entries": 0,
            "mtf_stop_distance_filtered_entries": 0,
            "long_entries": 0,
            "short_entries": 0,
            "fvg_entries": 0,
        }

        def _count_direction_block(reason: str) -> None:
            if reason == "daily":
                metadata["mtf_daily_blocked"] += 1
            elif reason == "4h":
                metadata["mtf_4h_blocked"] += 1
            elif reason == "1h":
                metadata["mtf_1h_blocked"] += 1

        def _count_timing_block(setups: list[_MTFSetup], *, reason: str | None) -> None:
            if reason == "weekday":
                key = "mtf_weekday_blocked"
            elif reason == "hour":
                key = "mtf_hour_blocked"
            else:
                return
            for setup in setups:
                if setup.timing_blocked:
                    continue
                setup.timing_blocked = True
                metadata[key] += 1

        def _direction_gate(
            *,
            direction: int,
            daily_value: int,
            four_hour_value: int,
            hourly_value: int,
        ) -> tuple[bool, str | None, bool]:
            if four_hour_value == 0:
                return False, "4h", False
            if daily_value != 0 and daily_value != four_hour_value:
                return False, "daily", False
            if direction != four_hour_value:
                return False, "4h", False
            if hourly_value == -direction:
                return False, "1h", False
            if daily_value == 0:
                if hourly_value != direction:
                    return False, "1h", False
                return True, None, False
            if hourly_value == 0:
                if not mtf_neutral_hourly_allows_high_quality:
                    return False, "1h", False
                return True, None, True
            return True, None, False

        def _setup_score(
            *,
            direction: int,
            zone_kind: str,
            sweep_depth: float,
            sweep_reference_level: float,
            body_ratio: float,
            gap_size: float,
            price: float,
        ) -> float:
            del direction
            zone_score = score_fvg if zone_kind == "fvg" else score_order_block
            base_score = (
                zone_score
                + score_liquidity_sweep * 0.5
                + _structure_confirmation_score(score_bos, score_choch) * 0.5
            )
            quality_bonus = _quality_score_bonus(
                sweep_depth=sweep_depth,
                sweep_reference_level=sweep_reference_level,
                liq_sweep_threshold=liq_sweep_threshold,
                displacement_body_ratio=body_ratio,
                displacement_body_min_pct=mtf_setup_displacement_body_min_pct,
                fvg_gap_size=gap_size if zone_kind == "fvg" else 0.0,
                price=price,
                fvg_min_gap_pct=mtf_setup_fvg_min_gap_pct,
                score_sweep_depth_quality=score_sweep_depth_quality,
                score_displacement_quality=score_displacement_quality,
                score_fvg_gap_quality=score_fvg_gap_quality,
            )
            return base_score + quality_bonus

        def _find_pullback_setup_context(
            *,
            direction: int,
            current_pos: int,
            current_close: float,
        ) -> tuple[int, float, float, float] | None:
            earliest_sweep_pos = max(0, current_pos - max(mtf_setup_structure_lookback, 2))
            latest_sweep_pos = current_pos - max(mtf_setup_swing_threshold, 1)
            if latest_sweep_pos < earliest_sweep_pos:
                return None

            for sweep_idx in range(latest_sweep_pos, earliest_sweep_pos - 1, -1):
                if direction > 0:
                    prior_sweep_level = frame_15m_low_ref.iat[sweep_idx]
                    if pd.isna(prior_sweep_level):
                        continue
                    sweep_level = float(frame_15m["Low"].iat[sweep_idx])
                    sweep_close = float(frame_15m["Close"].iat[sweep_idx])
                    sweep_depth = float(prior_sweep_level) - sweep_level
                    min_depth = abs(float(prior_sweep_level)) * max(liq_sweep_threshold, 0.0)
                    if sweep_depth <= max(min_depth, 0.0):
                        continue
                    reclaim_level = float(prior_sweep_level) + sweep_depth * max(
                        liq_sweep_reclaim_ratio,
                        0.0,
                    )
                    if sweep_close <= reclaim_level:
                        continue
                    structure_slice = frame_15m["High"].iloc[sweep_idx:current_pos]
                    if structure_slice.empty:
                        continue
                    structure_level = float(structure_slice.max())
                    if current_close <= structure_level:
                        continue
                    return sweep_idx, sweep_level, structure_level, sweep_depth

                prior_sweep_level = frame_15m_high_ref.iat[sweep_idx]
                if pd.isna(prior_sweep_level):
                    continue
                sweep_level = float(frame_15m["High"].iat[sweep_idx])
                sweep_close = float(frame_15m["Close"].iat[sweep_idx])
                sweep_depth = sweep_level - float(prior_sweep_level)
                min_depth = abs(float(prior_sweep_level)) * max(liq_sweep_threshold, 0.0)
                if sweep_depth <= max(min_depth, 0.0):
                    continue
                reclaim_level = float(prior_sweep_level) - sweep_depth * max(
                    liq_sweep_reclaim_ratio,
                    0.0,
                )
                if sweep_close >= reclaim_level:
                    continue
                structure_slice = frame_15m["Low"].iloc[sweep_idx:current_pos]
                if structure_slice.empty:
                    continue
                structure_level = float(structure_slice.min())
                if current_close >= structure_level:
                    continue
                return sweep_idx, sweep_level, structure_level, sweep_depth

            return None

        def _detect_setup_zone(
            *,
            sweep_idx: int,
            current_pos: int,
            bullish: bool,
        ) -> tuple[str, tuple[int, float, float]] | None:
            zone = _detect_fvg_zone(
                frame_15m,
                sweep_idx,
                current_pos,
                bullish=bullish,
                min_gap_pct=mtf_setup_fvg_min_gap_pct,
            )
            if zone is not None:
                metadata["mtf_setup_fvg_zones"] += 1
                return "fvg", zone

            zone = _detect_order_block_zone(
                frame_15m,
                sweep_idx,
                current_pos,
                bullish=bullish,
                body_min_pct=ob_body_min_pct,
                lookback=ob_lookback,
            )
            if zone is not None:
                metadata["mtf_setup_ob_zones"] += 1
                return "ob", zone

            return None

        setup_events: dict[pd.Timestamp, list[_MTFSetup]] = {}
        for pos, ts in enumerate(frame_15m.index):
            row_15m = frame_15m.iloc[pos]
            open_15m = float(row_15m["Open"])
            high_15m = float(row_15m["High"])
            low_15m = float(row_15m["Low"])
            close_15m = float(row_15m["Close"])
            candle_range = max(high_15m - low_15m, 1e-12)
            body_ratio = abs(close_15m - open_15m) / candle_range
            daily_value = int(daily_bias_15m.iat[pos]) if pos < len(daily_bias_15m) else 0
            four_hour_value = int(four_hour_bias_15m.iat[pos]) if pos < len(four_hour_bias_15m) else 0
            hourly_value = int(hourly_bias_15m.iat[pos]) if pos < len(hourly_bias_15m) else 0

            bullish_shift = close_15m > open_15m and body_ratio >= mtf_setup_displacement_body_min_pct
            bearish_shift = close_15m < open_15m and body_ratio >= mtf_setup_displacement_body_min_pct
            if not bullish_shift and not bearish_shift:
                continue

            for direction in (1, -1):
                if direction == 1 and not bullish_shift:
                    continue
                if direction == -1 and not bearish_shift:
                    continue
                if direction == 1 and not allow_long_entries:
                    continue
                if direction == -1 and not allow_short_entries:
                    continue

                context = _find_pullback_setup_context(
                    direction=direction,
                    current_pos=pos,
                    current_close=close_15m,
                )
                if context is None:
                    metadata["mtf_setup_missing_context"] += 1
                    continue

                allowed, blocked_reason, high_quality_only = _direction_gate(
                    direction=direction,
                    daily_value=daily_value,
                    four_hour_value=four_hour_value,
                    hourly_value=hourly_value,
                )
                if not allowed:
                    _count_direction_block(blocked_reason or "4h")
                    continue

                sweep_idx, sweep_level, _, sweep_depth = context
                zone_context = _detect_setup_zone(
                    sweep_idx=sweep_idx,
                    current_pos=pos,
                    bullish=direction == 1,
                )
                if zone_context is None:
                    metadata["mtf_setup_missing_zone"] += 1
                    continue

                zone_kind, zone = zone_context
                _, zone_lower, zone_upper = zone
                score = _setup_score(
                    direction=direction,
                    zone_kind=zone_kind,
                    sweep_depth=sweep_depth,
                    sweep_reference_level=sweep_level,
                    body_ratio=body_ratio,
                    gap_size=max(zone_upper - zone_lower, 0.0),
                    price=close_15m,
                )
                if score < min_score_to_trade:
                    metadata["mtf_setup_score_filtered"] += 1
                    continue
                if high_quality_only and score < mtf_neutral_hourly_min_score:
                    metadata["mtf_1h_blocked"] += 1
                    continue

                if high_quality_only:
                    metadata["mtf_neutral_1h_high_quality_setups"] += 1
                if direction == 1:
                    metadata["mtf_direction_long_allowed"] += 1
                else:
                    metadata["mtf_direction_short_allowed"] += 1
                metadata["mtf_15m_setups"] += 1
                setup_events.setdefault(ts, []).append(
                    _MTFSetup(
                        direction=direction,
                        setup_time=pd.Timestamp(ts),
                        source_index=pos,
                        zone_kind=zone_kind,
                        zone_lower=float(zone_lower),
                        zone_upper=float(zone_upper),
                        sweep_level=float(sweep_level),
                        shift_extreme=float(high_15m if direction == 1 else low_15m),
                        expiry_time=pd.Timestamp(ts) + setup_step * mtf_setup_zone_expiry_bars,
                        score=float(score),
                    )
                )

        def _volume_ratio_ok(
            *,
            current_volume: float,
            reference_volume: float,
            minimum_ratio: float,
        ) -> bool:
            if minimum_ratio <= 0.0:
                return True
            if pd.isna(current_volume) or pd.isna(reference_volume) or reference_volume <= 0.0:
                return False
            return float(current_volume) >= float(reference_volume) * minimum_ratio

        def _activate_setup_touch(setup: _MTFSetup, *, current_ts: pd.Timestamp) -> None:
            if setup.zone_touch_time is None:
                setup.zone_touch_time = current_ts
                metadata["mtf_zone_touches"] += 1
            if mtf_touch_confirm_bars > 0:
                confirm_expiry = current_ts + setup_step * mtf_touch_confirm_bars
                if (
                    setup.confirm_touch_expiry_time is None
                    or confirm_expiry > setup.confirm_touch_expiry_time
                ):
                    setup.confirm_touch_expiry_time = confirm_expiry
            if mtf_touch_trigger_bars > 0:
                trigger_expiry = current_ts + execution_step * mtf_touch_trigger_bars
                if (
                    setup.trigger_touch_expiry_time is None
                    or trigger_expiry > setup.trigger_touch_expiry_time
                ):
                    setup.trigger_touch_expiry_time = trigger_expiry

        def _confirm_setup(
            setup: _MTFSetup,
            *,
            current_ts: pd.Timestamp,
            bar_open: float,
            bar_high: float,
            bar_low: float,
            bar_close: float,
            current_volume: float,
            reference_volume: float,
            prior_high_level: float,
            prior_low_level: float,
        ) -> tuple[bool, bool, bool]:
            overlap = bar_low <= setup.zone_upper and bar_high >= setup.zone_lower
            if overlap:
                _activate_setup_touch(setup, current_ts=current_ts)
            touch_active = bool(
                overlap
                or (
                    setup.confirm_touch_expiry_time is not None
                    and current_ts <= setup.confirm_touch_expiry_time
                )
            )
            if not touch_active:
                return False, False, False
            candle_range = max(bar_high - bar_low, 1e-12)
            body_ratio = abs(bar_close - bar_open) / candle_range
            close_ratio = _close_position_ratio(bar_high, bar_low, bar_close)
            touch_window_only = bool(touch_active and not overlap)
            if setup.direction == 1:
                rejection_wick = _opposite_wick_ratio(
                    bar_open,
                    bar_high,
                    bar_low,
                    bar_close,
                    bullish=True,
                )
                rejection = (
                    bar_close > bar_open
                    and close_ratio >= mtf_confirmation_close_ratio
                    and body_ratio >= mtf_confirmation_body_min_pct
                    and rejection_wick >= mtf_confirmation_rejection_wick_ratio
                    and _volume_ratio_ok(
                        current_volume=current_volume,
                        reference_volume=reference_volume,
                        minimum_ratio=mtf_confirmation_rejection_volume_ratio,
                    )
                )
                micro_shift = pd.notna(prior_high_level) and bar_close > float(prior_high_level)
                strong_displacement = bool(
                    micro_shift
                    and body_ratio >= mtf_fast_retest_displacement_body_min_pct
                    and close_ratio >= mtf_fast_retest_displacement_close_ratio
                )
                return rejection or micro_shift, strong_displacement, touch_window_only
            rejection_wick = _opposite_wick_ratio(
                bar_open,
                bar_high,
                bar_low,
                bar_close,
                bullish=False,
            )
            rejection = (
                bar_close < bar_open
                and (1.0 - close_ratio) >= mtf_confirmation_close_ratio
                and body_ratio >= mtf_confirmation_body_min_pct
                and rejection_wick >= mtf_confirmation_rejection_wick_ratio
                and _volume_ratio_ok(
                    current_volume=current_volume,
                    reference_volume=reference_volume,
                    minimum_ratio=mtf_confirmation_rejection_volume_ratio,
                )
            )
            micro_shift = pd.notna(prior_low_level) and bar_close < float(prior_low_level)
            strong_displacement = bool(
                micro_shift
                and body_ratio >= mtf_fast_retest_displacement_body_min_pct
                and (1.0 - close_ratio) >= mtf_fast_retest_displacement_close_ratio
            )
            return rejection or micro_shift, strong_displacement, touch_window_only

        def _trigger_ready(
            setup: _MTFSetup,
            *,
            current_ts: pd.Timestamp,
            idx: int,
            bar_open: float,
            bar_high: float,
            bar_low: float,
            bar_close: float,
            current_volume: float,
            reference_volume: float,
        ) -> tuple[bool, bool, bool]:
            overlap = bar_low <= setup.zone_upper and bar_high >= setup.zone_lower
            if overlap:
                _activate_setup_touch(setup, current_ts=current_ts)
            touch_active = bool(
                overlap
                or (
                    setup.trigger_touch_expiry_time is not None
                    and current_ts <= setup.trigger_touch_expiry_time
                )
            )
            if not touch_active:
                return False, False, False
            candle_range = max(bar_high - bar_low, 1e-12)
            body_ratio = abs(bar_close - bar_open) / candle_range
            close_ratio = _close_position_ratio(bar_high, bar_low, bar_close)
            touch_window_only = bool(touch_active and not overlap)
            if setup.direction == 1:
                micro_shift = pd.notna(trigger_high_ref.iat[idx]) and bar_close > float(trigger_high_ref.iat[idx])
                rejection_wick = _opposite_wick_ratio(
                    bar_open,
                    bar_high,
                    bar_low,
                    bar_close,
                    bullish=True,
                )
                rejection = (
                    bar_close > bar_open
                    and close_ratio >= mtf_trigger_close_ratio
                    and body_ratio >= mtf_trigger_body_min_pct
                    and rejection_wick >= mtf_trigger_rejection_wick_ratio
                    and _volume_ratio_ok(
                        current_volume=current_volume,
                        reference_volume=reference_volume,
                        minimum_ratio=mtf_trigger_rejection_volume_ratio,
                    )
                )
                fast_retest = bool(
                    mtf_fast_retest_entry_enabled
                    and setup.fast_retest_armed
                    and overlap
                    and bar_close > bar_open
                    and close_ratio >= mtf_fast_retest_min_close_ratio
                    and bar_close >= setup.zone_lower
                )
                return rejection or micro_shift or fast_retest, fast_retest, touch_window_only
            micro_shift = pd.notna(trigger_low_ref.iat[idx]) and bar_close < float(trigger_low_ref.iat[idx])
            rejection_wick = _opposite_wick_ratio(
                bar_open,
                bar_high,
                bar_low,
                bar_close,
                bullish=False,
            )
            rejection = (
                bar_close < bar_open
                and (1.0 - close_ratio) >= mtf_trigger_close_ratio
                and body_ratio >= mtf_trigger_body_min_pct
                and rejection_wick >= mtf_trigger_rejection_wick_ratio
                and _volume_ratio_ok(
                    current_volume=current_volume,
                    reference_volume=reference_volume,
                    minimum_ratio=mtf_trigger_rejection_volume_ratio,
                )
            )
            fast_retest = bool(
                mtf_fast_retest_entry_enabled
                and setup.fast_retest_armed
                and overlap
                and bar_close < bar_open
                and (1.0 - close_ratio) >= mtf_fast_retest_min_close_ratio
                and bar_close <= setup.zone_upper
            )
            return rejection or micro_shift or fast_retest, fast_retest, touch_window_only

        def _prune_active_setups(setups: list[_MTFSetup], *, current_ts: pd.Timestamp) -> list[_MTFSetup]:
            retained: list[_MTFSetup] = []
            for setup in setups:
                if current_ts > setup.expiry_time:
                    metadata["mtf_setup_expired"] += 1
                    continue
                if setup.trigger_expiry_time is not None and current_ts > setup.trigger_expiry_time:
                    metadata["mtf_confirm_expired"] += 1
                    continue
                retained.append(setup)
            return retained

        def _eligible_trigger_candidates(
            setups: list[_MTFSetup],
            *,
            current_ts: pd.Timestamp,
        ) -> list[_MTFSetup]:
            return [
                setup
                for setup in setups
                if setup.confirm_time is not None
                and current_ts > setup.confirm_time
                and (setup.trigger_expiry_time is None or current_ts <= setup.trigger_expiry_time)
            ]

        active_longs: list[_MTFSetup] = []
        active_shorts: list[_MTFSetup] = []
        position = 0
        active_stop = np.nan
        active_target = np.nan
        five_min_event_times = set(frame_5m.index)

        for idx, ts in enumerate(df.index):
            row = df.iloc[idx]
            open_ = float(row["Open"])
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            current_volume = float(row["Volume"]) if "Volume" in row.index and pd.notna(row["Volume"]) else np.nan
            current_atr = float(atr.iat[idx]) if pd.notna(atr.iat[idx]) else 0.0
            next_is_new_day = idx == len(df.index) - 1 or not _same_trading_day(df.index[idx + 1], ts)
            in_session = _in_trade_session(
                pd.Timestamp(ts),
                bool(self.params["trade_sessions"]),
                int(self.params["london_open"]),
                int(self.params["london_close"]),
                int(self.params["ny_open"]),
                int(self.params["ny_close"]),
            )

            if position > 0:
                if low <= active_stop or close <= active_stop or high >= active_target or close >= active_target or next_is_new_day:
                    exits_long.iat[idx] = True
                    position = 0
                    active_stop = np.nan
                    active_target = np.nan
                    active_longs = []
                    active_shorts = []
                    continue
            elif position < 0:
                if high >= active_stop or close >= active_stop or low <= active_target or close <= active_target or next_is_new_day:
                    exits_short.iat[idx] = True
                    position = 0
                    active_stop = np.nan
                    active_target = np.nan
                    active_longs = []
                    active_shorts = []
                    continue

            current_ts = pd.Timestamp(ts)
            active_longs = _prune_active_setups(active_longs, current_ts=current_ts)
            active_shorts = _prune_active_setups(active_shorts, current_ts=current_ts)

            for setup in setup_events.get(current_ts, []):
                if setup.direction == 1:
                    active_longs.append(setup)
                else:
                    active_shorts.append(setup)

            if current_ts in five_min_event_times:
                five_bar = frame_5m.loc[current_ts]
                five_open = float(five_bar["Open"])
                five_high = float(five_bar["High"])
                five_low = float(five_bar["Low"])
                five_close = float(five_bar["Close"])
                five_volume = (
                    float(five_bar["Volume"])
                    if "Volume" in five_bar.index and pd.notna(five_bar["Volume"])
                    else np.nan
                )
                five_volume_baseline = (
                    float(frame_5m_volume_ref.loc[current_ts])
                    if current_ts in frame_5m_volume_ref.index and pd.notna(frame_5m_volume_ref.loc[current_ts])
                    else np.nan
                )
                prior_high_level = frame_5m_high_ref.loc[current_ts]
                prior_low_level = frame_5m_low_ref.loc[current_ts]
                for setup in active_longs:
                    if setup.confirm_time is not None or current_ts <= setup.setup_time:
                        continue
                    confirmed, strong_displacement, touch_window_only = _confirm_setup(
                        setup,
                        current_ts=current_ts,
                        bar_open=five_open,
                        bar_high=five_high,
                        bar_low=five_low,
                        bar_close=five_close,
                        current_volume=five_volume,
                        reference_volume=five_volume_baseline,
                        prior_high_level=prior_high_level,
                        prior_low_level=prior_low_level,
                    )
                    if confirmed:
                        setup.confirm_time = current_ts
                        setup.trigger_expiry_time = current_ts + execution_step * mtf_trigger_expiry_bars
                        setup.fast_retest_armed = strong_displacement
                        metadata["mtf_5m_confirms"] += 1
                        if touch_window_only:
                            metadata["mtf_touch_activated_confirms"] += 1
                        if strong_displacement:
                            metadata["mtf_fast_retest_confirms"] += 1
                for setup in active_shorts:
                    if setup.confirm_time is not None or current_ts <= setup.setup_time:
                        continue
                    confirmed, strong_displacement, touch_window_only = _confirm_setup(
                        setup,
                        current_ts=current_ts,
                        bar_open=five_open,
                        bar_high=five_high,
                        bar_low=five_low,
                        bar_close=five_close,
                        current_volume=five_volume,
                        reference_volume=five_volume_baseline,
                        prior_high_level=prior_high_level,
                        prior_low_level=prior_low_level,
                    )
                    if confirmed:
                        setup.confirm_time = current_ts
                        setup.trigger_expiry_time = current_ts + execution_step * mtf_trigger_expiry_bars
                        setup.fast_retest_armed = strong_displacement
                        metadata["mtf_5m_confirms"] += 1
                        if touch_window_only:
                            metadata["mtf_touch_activated_confirms"] += 1
                        if strong_displacement:
                            metadata["mtf_fast_retest_confirms"] += 1

            if position == 0 and in_session:
                long_candidates = _eligible_trigger_candidates(
                    active_longs,
                    current_ts=current_ts,
                )
                short_candidates = _eligible_trigger_candidates(
                    active_shorts,
                    current_ts=current_ts,
                )
                timing_allowed, timing_block_reason = _entry_timing_gate(
                    current_ts,
                    timezone=mtf_timing_timezone,
                    allowed_weekdays=mtf_allowed_entry_weekdays,
                    allowed_hours=mtf_allowed_entry_hours,
                )
                if not timing_allowed:
                    _count_timing_block(long_candidates + short_candidates, reason=timing_block_reason)
                    continue

                long_candidates.sort(key=lambda setup: (setup.confirm_time, setup.score), reverse=True)
                for setup in long_candidates:
                    trigger_ready, fast_retest_entry, touch_window_only = _trigger_ready(
                        setup,
                        current_ts=current_ts,
                        idx=idx,
                        bar_open=open_,
                        bar_high=high,
                        bar_low=low,
                        bar_close=close,
                        current_volume=current_volume,
                        reference_volume=(
                            float(trigger_volume_ref.iat[idx])
                            if pd.notna(trigger_volume_ref.iat[idx])
                            else np.nan
                        ),
                    )
                    if not trigger_ready:
                        continue
                    risk_anchor = min(setup.zone_lower, low, close)
                    atr_buffer = max(current_atr * stop_loss_atr_mult, close * 0.002)
                    candidate_stop = min(risk_anchor - current_atr * 0.1, close - atr_buffer)
                    if candidate_stop >= close:
                        candidate_stop = close - atr_buffer
                    stop_distance = abs(close - candidate_stop)
                    if (
                        mtf_max_stop_distance_atr_mult > 0.0
                        and current_atr > 0.0
                        and stop_distance > current_atr * mtf_max_stop_distance_atr_mult
                    ):
                        metadata["mtf_stop_distance_filtered_entries"] += 1
                        continue
                    candidate_target = close + (close - candidate_stop) * take_profit_rr
                    projected_rr = _project_reward_risk_ratio(close, candidate_stop, candidate_target)
                    if projected_rr < min_reward_risk_ratio:
                        metadata["rr_filtered_entries"] += 1
                        continue
                    entries_long.iat[idx] = True
                    position = 1
                    active_stop = candidate_stop
                    active_target = candidate_target
                    setup.entry_time = current_ts
                    metadata["mtf_1m_triggers"] += 1
                    metadata["long_entries"] += 1
                    metadata["fvg_entries"] += 1
                    if touch_window_only:
                        metadata["mtf_touch_activated_triggers"] += 1
                    if fast_retest_entry:
                        metadata["mtf_fast_retest_entries"] += 1
                    active_longs = []
                    active_shorts = []
                    break

            if position == 0 and in_session and not bool(entries_long.iat[idx]):
                short_candidates.sort(key=lambda setup: (setup.confirm_time, setup.score), reverse=True)
                for setup in short_candidates:
                    trigger_ready, fast_retest_entry, touch_window_only = _trigger_ready(
                        setup,
                        current_ts=current_ts,
                        idx=idx,
                        bar_open=open_,
                        bar_high=high,
                        bar_low=low,
                        bar_close=close,
                        current_volume=current_volume,
                        reference_volume=(
                            float(trigger_volume_ref.iat[idx])
                            if pd.notna(trigger_volume_ref.iat[idx])
                            else np.nan
                        ),
                    )
                    if not trigger_ready:
                        continue
                    risk_anchor = max(setup.zone_upper, high, close)
                    atr_buffer = max(current_atr * stop_loss_atr_mult, close * 0.002)
                    candidate_stop = max(risk_anchor + current_atr * 0.1, close + atr_buffer)
                    if candidate_stop <= close:
                        candidate_stop = close + atr_buffer
                    stop_distance = abs(candidate_stop - close)
                    if (
                        mtf_max_stop_distance_atr_mult > 0.0
                        and current_atr > 0.0
                        and stop_distance > current_atr * mtf_max_stop_distance_atr_mult
                    ):
                        metadata["mtf_stop_distance_filtered_entries"] += 1
                        continue
                    candidate_target = close - (candidate_stop - close) * take_profit_rr
                    projected_rr = _project_reward_risk_ratio(close, candidate_stop, candidate_target)
                    if projected_rr < min_reward_risk_ratio:
                        metadata["rr_filtered_entries"] += 1
                        continue
                    entries_short.iat[idx] = True
                    position = -1
                    active_stop = candidate_stop
                    active_target = candidate_target
                    setup.entry_time = current_ts
                    metadata["mtf_1m_triggers"] += 1
                    metadata["short_entries"] += 1
                    metadata["fvg_entries"] += 1
                    if touch_window_only:
                        metadata["mtf_touch_activated_triggers"] += 1
                    if fast_retest_entry:
                        metadata["mtf_fast_retest_entries"] += 1
                    active_longs = []
                    active_shorts = []
                    break

        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            metadata=metadata,
        )

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        if df.empty:
            empty = pd.Series(dtype=bool, index=df.index)
            return StrategyResult(
                entries_long=empty,
                exits_long=empty,
                entries_short=empty,
                exits_short=empty,
                metadata={"warning": "empty input"},
            )
        if bool(self.params.get("use_mtf_topdown_continuation", False)):
            return self._generate_mtf_topdown_continuation_signals(df)

        structure_lookback = int(self.params["structure_lookback"])
        swing_threshold = int(self.params["swing_threshold"])
        structure_reference_mode = str(self.params.get("structure_reference_mode", "rolling")).lower()
        ob_lookback = int(self.params["ob_lookback"])
        breaker_lookback = int(self.params["breaker_lookback"])
        ifvg_lookback = int(self.params["ifvg_lookback"])
        ob_body_min_pct = float(self.params["ob_body_min_pct"])
        fvg_min_gap_pct = float(self.params["fvg_min_gap_pct"])
        fvg_max_age = int(self.params["fvg_max_age"])
        fvg_revisit_min_delay_bars = int(self.params["fvg_revisit_min_delay_bars"])
        max_pending_setups_per_direction = max(1, int(self.params["max_pending_setups_per_direction"]))
        max_reentries_per_setup = max(0, int(self.params.get("max_reentries_per_setup", 0)))
        allow_long_entries = bool(self.params.get("allow_long_entries", True))
        allow_short_entries = bool(self.params.get("allow_short_entries", True))
        fvg_origin_max_lag_bars = int(self.params["fvg_origin_max_lag_bars"])
        fvg_origin_body_min_pct = float(self.params["fvg_origin_body_min_pct"])
        fvg_origin_body_atr_mult = float(self.params["fvg_origin_body_atr_mult"])
        fvg_origin_close_position_min_pct = float(self.params["fvg_origin_close_position_min_pct"])
        fvg_origin_opposite_wick_max_pct = float(self.params["fvg_origin_opposite_wick_max_pct"])
        fvg_origin_range_atr_mult = float(self.params["fvg_origin_range_atr_mult"])
        fvg_max_retest_touches = int(self.params["fvg_max_retest_touches"])
        displacement_body_min_pct = float(self.params["displacement_body_min_pct"])
        displacement_range_atr_mult = float(self.params["displacement_range_atr_mult"])
        structure_shift_close_buffer_ratio = float(self.params["structure_shift_close_buffer_ratio"])
        structure_shift_intrabar_tolerance_ratio = float(
            self.params.get("structure_shift_intrabar_tolerance_ratio", 0.0)
        )
        structure_shift_intrabar_close_position_min_pct = float(
            self.params.get("structure_shift_intrabar_close_position_min_pct", 0.0)
        )
        fvg_revisit_depth_ratio = float(self.params["fvg_revisit_depth_ratio"])
        fvg_rejection_close_ratio = float(self.params["fvg_rejection_close_ratio"])
        fvg_rejection_wick_ratio = float(self.params["fvg_rejection_wick_ratio"])
        fvg_rejection_body_min_pct = float(self.params["fvg_rejection_body_min_pct"])
        enable_continuation_entry = bool(self.params.get("enable_continuation_entry", False))
        liq_sweep_lookback = int(self.params["liq_sweep_lookback"])
        liq_sweep_threshold = float(self.params["liq_sweep_threshold"])
        liq_sweep_recovery_bars = int(self.params["liq_sweep_recovery_bars"])
        slow_recovery_enabled = bool(self.params["slow_recovery_enabled"])
        slow_recovery_bars = max(int(self.params["slow_recovery_bars"]), 0)
        if not slow_recovery_enabled or slow_recovery_bars <= liq_sweep_recovery_bars:
            slow_recovery_enabled = False
            slow_recovery_bars = liq_sweep_recovery_bars
        liq_sweep_reclaim_ratio = float(self.params["liq_sweep_reclaim_ratio"])
        ote_fib_low = float(self.params["ote_fib_low"])
        ote_fib_high = float(self.params["ote_fib_high"])
        atr_period = int(self.params["atr_period"])
        stop_loss_atr_mult = float(self.params["stop_loss_atr_mult"])
        take_profit_rr = float(self.params["take_profit_rr"])
        min_reward_risk_ratio = float(self.params["min_reward_risk_ratio"])
        require_fvg_delivery = bool(self.params.get("require_fvg_delivery", False))
        require_ote_zone = bool(self.params.get("require_ote_zone", False))
        score_order_block = float(self.params["score_order_block"])
        score_breaker_block = float(self.params["score_breaker_block"])
        score_ifvg = float(self.params["score_ifvg"])
        score_fvg = float(self.params["score_fvg"])
        score_liquidity_sweep = float(self.params["score_liquidity_sweep"])
        score_ote_zone = float(self.params["score_ote_zone"])
        score_bos = float(self.params["score_bos"])
        score_choch = float(self.params["score_choch"])
        score_sweep_depth_quality = float(self.params["score_sweep_depth_quality"])
        score_displacement_quality = float(self.params["score_displacement_quality"])
        score_fvg_gap_quality = float(self.params["score_fvg_gap_quality"])
        min_score_to_trade = float(self.params["min_score_to_trade"])

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)

        atr = _compute_atr(df, atr_period)
        trading_timezone = "America/New_York"
        daily_bias = _compute_daily_bias(
            df,
            enabled=bool(self.params["use_daily_bias_filter"]),
            mode=str(self.params.get("daily_bias_mode", "statistical")),
            lookback=int(self.params["daily_bias_lookback"]),
            swing_threshold=int(self.params.get("daily_bias_swing_threshold", 1)),
            bull_threshold=float(self.params["daily_bias_bull_threshold"]),
            bear_threshold=float(self.params["daily_bias_bear_threshold"]),
            trading_timezone=trading_timezone,
        )
        regime_series, regime_stats = _compute_intraday_regime(
            df,
            enabled=bool(self.params.get("use_regime_adaptation", False)),
            atr_period=int(self.params.get("regime_atr_period", atr_period)),
            atr_pct_window=int(self.params.get("regime_atr_pct_window", 48)),
            atr_high_mult=float(self.params.get("regime_atr_high_mult", 1.15)),
            adx_period=int(self.params.get("regime_adx_period", 14)),
            adx_trend_threshold=float(self.params.get("regime_adx_trend_threshold", 22.0)),
        )
        higher_timeframe_bias = _compute_higher_timeframe_structure_bias(
            df,
            enabled=bool(self.params.get("use_higher_timeframe_alignment", False)),
            timeframe=str(self.params.get("higher_timeframe_bias_timeframe", "1H")),
            lookback=int(self.params.get("higher_timeframe_bias_lookback", 12)),
            swing_threshold=int(self.params.get("higher_timeframe_bias_swing_threshold", 1)),
        )
        premium_discount_context = _compute_premium_discount_context(
            df,
            enabled=bool(self.params["use_premium_discount_filter"]),
            lookback=int(self.params["premium_discount_lookback"]),
            neutral_band=float(self.params["premium_discount_neutral_band"]),
            trading_timezone=trading_timezone,
        )
        premium_discount_filter_mode = str(
            self.params.get("premium_discount_filter_mode", "hard")
        ).strip().lower()
        if premium_discount_filter_mode not in {"hard", "soft"}:
            premium_discount_filter_mode = "hard"
        premium_discount_mismatch_score_penalty = max(
            0.0,
            float(self.params.get("premium_discount_mismatch_score_penalty", 0.0)),
        )
        use_amd_filter = bool(self.params["use_amd_filter"])
        amd_bias, amd_stats = _compute_amd_path_bias(
            df,
            enabled=use_amd_filter,
            accumulation_bars=int(self.params["amd_accumulation_bars"]),
            manipulation_threshold=float(self.params["amd_manipulation_threshold"]),
            require_midpoint_reclaim=bool(self.params["amd_require_midpoint_reclaim"]),
            trading_timezone=trading_timezone,
        )
        use_prev_session_anchor_filter = bool(self.params["use_prev_session_anchor_filter"])
        prev_session_anchor_filter_mode = str(
            self.params.get("prev_session_anchor_filter_mode", "hard")
        ).strip().lower()
        if prev_session_anchor_filter_mode not in {"hard", "soft"}:
            prev_session_anchor_filter_mode = "hard"
        prev_session_anchor_mismatch_score_penalty = max(
            0.0,
            float(self.params.get("prev_session_anchor_mismatch_score_penalty", 0.0)),
        )
        prev_session_anchor_bias, prev_session_anchor_stats = _compute_prev_session_anchor_bias(
            df,
            enabled=use_prev_session_anchor_filter,
            tolerance=float(self.params["prev_session_anchor_tolerance"]),
            trading_timezone=trading_timezone,
        )
        use_external_liquidity_filter = bool(self.params["use_external_liquidity_filter"])
        external_liquidity_lookback = int(self.params["external_liquidity_lookback"])
        external_liquidity_tolerance = float(self.params["external_liquidity_tolerance"])
        use_smt_filter = bool(self.params["use_smt_filter"])
        smt_lookback = int(self.params["smt_lookback"])
        smt_threshold = float(self.params["smt_threshold"])
        smt_mode = str(self.params.get("smt_mode", "bar_extreme")).strip().lower()
        if smt_mode not in {"bar_extreme", "confirmed_swing"}:
            smt_mode = "bar_extreme"
        smt_swing_threshold = max(
            1,
            int(self.params.get("smt_swing_threshold", max(1, swing_threshold))),
        )
        use_macro_timing_windows = bool(self.params["use_macro_timing_windows"])
        macro_timezone = str(self.params["macro_timezone"])
        macro_windows = [tuple(window) for window in self.params["macro_windows"]]
        use_session_array_refinement = bool(self.params["use_session_array_refinement"])
        session_array_filter_mode = str(
            self.params.get("session_array_filter_mode", "hard")
        ).strip().lower()
        if session_array_filter_mode not in {"hard", "soft"}:
            session_array_filter_mode = "hard"
        session_array_mismatch_score_penalty = max(
            0.0,
            float(self.params.get("session_array_mismatch_score_penalty", 0.0)),
        )
        higher_timeframe_alignment_mode = str(
            self.params.get("higher_timeframe_alignment_mode", "hard")
        ).strip().lower()
        if higher_timeframe_alignment_mode not in {"hard", "soft"}:
            higher_timeframe_alignment_mode = "hard"
        higher_timeframe_mismatch_score_penalty = max(
            0.0,
            float(self.params.get("higher_timeframe_mismatch_score_penalty", 0.0)),
        )
        dealing_array_timezone = str(self.params["dealing_array_timezone"])
        imbalance_array_windows = [tuple(window) for window in self.params["imbalance_array_windows"]]
        structural_array_windows = [tuple(window) for window in self.params["structural_array_windows"]]
        rolling_high = df["High"].rolling(liq_sweep_lookback, min_periods=liq_sweep_lookback).max().shift(1)
        rolling_low = df["Low"].rolling(liq_sweep_lookback, min_periods=liq_sweep_lookback).min().shift(1)
        external_rolling_high = (
            df["High"].rolling(external_liquidity_lookback, min_periods=external_liquidity_lookback).max().shift(1)
        )
        external_rolling_low = (
            df["Low"].rolling(external_liquidity_lookback, min_periods=external_liquidity_lookback).min().shift(1)
        )
        peer_available = {"PeerHigh", "PeerLow"}.issubset(df.columns)
        if use_smt_filter and not peer_available:
            raise ValueError(
                "SMT filter requires PeerHigh and PeerLow columns. "
                "Merge peer-symbol data before enabling use_smt_filter."
            )
        if peer_available:
            peer_rolling_high = (
                df["PeerHigh"].rolling(smt_lookback, min_periods=smt_lookback).max().shift(1)
            )
            peer_rolling_low = (
                df["PeerLow"].rolling(smt_lookback, min_periods=smt_lookback).min().shift(1)
            )
        else:
            peer_rolling_high = pd.Series(np.nan, index=df.index, dtype=float)
            peer_rolling_low = pd.Series(np.nan, index=df.index, dtype=float)

        metadata: dict[str, Any] = {
            "bullish_sweep_candidates": 0,
            "bearish_sweep_candidates": 0,
            "bullish_sweeps": 0,
            "bearish_sweeps": 0,
            "external_targeted_sweeps": 0,
            "smt_confirmed_sweeps": 0,
            "bullish_shift_candidates": 0,
            "bearish_shift_candidates": 0,
            "bullish_shifts": 0,
            "bearish_shifts": 0,
            "bullish_retest_candidates": 0,
            "bearish_retest_candidates": 0,
            "long_entries": 0,
            "short_entries": 0,
            "fvg_entries": 0,
            "ob_entries": 0,
            "breaker_entries": 0,
            "ifvg_entries": 0,
            "delivery_missing_shifts": 0,
            "score_filtered_shifts": 0,
            "score_quality_boosted_shifts": 0,
            "score_quality_bonus_total": 0.0,
            "regime_adaptation_enabled": bool(self.params.get("use_regime_adaptation", False)),
            "high_regime_bars": int(regime_stats.get("high_regime_bars", 0)),
            "volatility_high_regime_bars": int(regime_stats.get("volatility_high_regime_bars", 0)),
            "trend_high_regime_bars": int(regime_stats.get("trend_high_regime_bars", 0)),
            "higher_timeframe_alignment_enabled": bool(self.params.get("use_higher_timeframe_alignment", False)),
            "higher_timeframe_filtered_setups": 0,
            "higher_timeframe_softened_setups": 0,
            "higher_timeframe_score_penalty_shifts": 0,
            "kill_zone_enabled": bool(self.params["use_kill_zones"]),
            "kill_zone_filtered_sweeps": 0,
            "fvg_depth_filtered_retests": 0,
            "fvg_delay_filtered_retests": 0,
            "fvg_origin_lag_filtered_shifts": 0,
            "fvg_origin_body_filtered_shifts": 0,
            "fvg_origin_body_atr_filtered_shifts": 0,
            "fvg_origin_close_filtered_shifts": 0,
            "fvg_origin_wick_filtered_shifts": 0,
            "fvg_origin_range_filtered_shifts": 0,
            "fvg_touch_filtered_retests": 0,
            "fvg_close_filtered_retests": 0,
            "fvg_wick_filtered_retests": 0,
            "fvg_body_filtered_retests": 0,
            "daily_bias_enabled": bool(self.params["use_daily_bias_filter"]),
            "daily_bias_mode": str(self.params.get("daily_bias_mode", "statistical")).strip().lower(),
            "daily_bias_filtered_setups": 0,
            "premium_discount_enabled": bool(self.params["use_premium_discount_filter"]),
            "premium_discount_filter_mode": premium_discount_filter_mode,
            "premium_discount_filtered_setups": 0,
            "premium_discount_mismatch_setups": 0,
            "premium_discount_softened_setups": 0,
            "premium_discount_score_penalty_shifts": 0,
            "amd_enabled": use_amd_filter,
            "amd_filtered_setups": 0,
            "amd_bullish_sessions": int(amd_stats["bullish_sessions"]),
            "amd_bearish_sessions": int(amd_stats["bearish_sessions"]),
            "prev_session_anchor_enabled": use_prev_session_anchor_filter,
            "prev_session_anchor_filter_mode": prev_session_anchor_filter_mode,
            "prev_session_anchor_filtered_setups": 0,
            "prev_session_anchor_mismatch_setups": 0,
            "prev_session_anchor_softened_setups": 0,
            "prev_session_anchor_score_penalty_shifts": 0,
            "prev_session_anchor_long_bars": int(prev_session_anchor_stats["long_bars"]),
            "prev_session_anchor_short_bars": int(prev_session_anchor_stats["short_bars"]),
            "session_array_refinement_enabled": use_session_array_refinement,
            "session_array_filter_mode": session_array_filter_mode,
            "session_array_filtered_shifts": 0,
            "session_array_softened_shifts": 0,
            "session_array_score_penalty_shifts": 0,
            "macro_timing_enabled": use_macro_timing_windows,
            "macro_timing_filtered_sweeps": 0,
            "rr_gate_enabled": min_reward_risk_ratio > 0.0,
            "rr_filtered_entries": 0,
            "require_fvg_delivery": require_fvg_delivery,
            "require_ote_zone": require_ote_zone,
            "fvg_required_filtered_shifts": 0,
            "ote_required_filtered_shifts": 0,
            "reentry_enabled": max_reentries_per_setup > 0,
            "max_reentries_per_setup": max_reentries_per_setup,
            "reentry_entries": 0,
            "reentry_stop_rearms": 0,
            "reentry_exhausted_setups": 0,
            "continuation_entry_enabled": enable_continuation_entry,
            "continuation_zone_refreshes": 0,
            "continuation_entries": 0,
            "slow_recovery_enabled": slow_recovery_enabled,
            "slow_recovery_bars": slow_recovery_bars,
            "sweep_blocked_by_existing_pending": 0,
            "sweep_expired_before_shift": 0,
            "armed_setup_expired_before_retest": 0,
            "fast_recovery_shifts": 0,
            "slow_recovery_shifts": 0,
            "fast_recovery_entries": 0,
            "slow_recovery_entries": 0,
            "displacement_filtered_shifts": 0,
            "displacement_range_filtered_shifts": 0,
            "structure_buffer_filtered_shifts": 0,
            "intrabar_shift_candidates": 0,
            "intrabar_shift_confirmed": 0,
            "external_liquidity_filter_enabled": use_external_liquidity_filter,
            "external_liquidity_filtered_sweeps": 0,
            "smt_enabled": use_smt_filter,
            "smt_peer_available": peer_available,
            "smt_mode": smt_mode,
            "smt_swing_threshold": smt_swing_threshold,
            "smt_filtered_sweeps": 0,
            "smt_missing_peer_bars": 0,
            "smt_missing_peer_structure": 0,
            "structure_reference_mode": structure_reference_mode,
            "swing_structure_missing_reference": 0,
        }

        def _evaluate_smt_confirmation(
            *,
            idx: int,
            direction: int,
            effective_threshold: float,
        ) -> tuple[bool, bool, bool]:
            if not peer_available:
                return False, False, False

            peer_column = "PeerLow" if direction > 0 else "PeerHigh"
            current_peer_value = df[peer_column].iat[idx]
            if pd.isna(current_peer_value):
                return False, False, False

            if smt_mode == "confirmed_swing":
                peer_reference = _latest_confirmed_swing_level(
                    df[peer_column],
                    idx,
                    lookback=smt_lookback,
                    threshold=smt_swing_threshold,
                    swing_type="low" if direction > 0 else "high",
                )
                if pd.isna(peer_reference):
                    return False, True, False
            else:
                peer_reference = (
                    peer_rolling_low.iat[idx] if direction > 0 else peer_rolling_high.iat[idx]
                )
                if pd.isna(peer_reference):
                    return False, False, False

            peer_value = float(current_peer_value)
            if direction > 0:
                peer_swept = peer_value < float(peer_reference) * (1 - effective_threshold)
            else:
                peer_swept = peer_value > float(peer_reference) * (1 + effective_threshold)
            return True, False, not peer_swept

        def _record_pending_expiry(setup: _PendingSetup) -> None:
            if setup.state == "swept":
                metadata["sweep_expired_before_shift"] += 1
            elif setup.state == "armed":
                metadata["armed_setup_expired_before_retest"] += 1

        def _recovery_phase(setup: _PendingSetup, current_idx: int) -> str:
            if slow_recovery_enabled and current_idx > setup.primary_expiry_index:
                return "slow"
            return "fast"

        def _expire_pending_setups(
            setups: list[_PendingSetup],
            *,
            current_idx: int,
        ) -> list[_PendingSetup]:
            active_setups: list[_PendingSetup] = []
            for setup in setups:
                if current_idx > setup.expiry_index:
                    _record_pending_expiry(setup)
                else:
                    active_setups.append(setup)
            return active_setups

        def _eligible_armed_setups(
            setups: list[_PendingSetup],
            *,
            current_idx: int,
            bar_low: float,
            bar_high: float,
        ) -> list[_PendingSetup]:
            armed = [
                setup
                for setup in setups
                if setup.state == "armed"
                and setup.armed_index is not None
                and current_idx > setup.armed_index
                and setup.zone_lower is not None
                and setup.zone_upper is not None
                and bar_low <= setup.zone_upper
                and bar_high >= setup.zone_lower
            ]
            armed.sort(
                key=lambda setup: (
                    setup.armed_index if setup.armed_index is not None else -1,
                    setup.score,
                    setup.sweep_index,
                ),
                reverse=True,
            )
            return armed

        def _retain_reentry_candidates(
            setups: list[_PendingSetup],
            *,
            direction: int,
        ) -> list[_PendingSetup]:
            if max_reentries_per_setup <= 0:
                return []
            retained: list[_PendingSetup] = []
            for setup in setups:
                if setup.direction != direction:
                    continue
                if setup.state != "armed":
                    continue
                if setup.entry_attempts <= max_reentries_per_setup:
                    retained.append(setup)
                    metadata["reentry_stop_rearms"] += 1
                else:
                    metadata["reentry_exhausted_setups"] += 1
            return retained

        def _refresh_continuation_zone(
            setup: _PendingSetup,
            *,
            current_idx: int,
            bullish: bool,
        ) -> None:
            if not enable_continuation_entry:
                return
            if setup.state != "armed":
                return
            if setup.zone_kind not in {"fvg", "ifvg"}:
                return
            refreshed_zone = _detect_fvg_zone(
                df,
                setup.sweep_index,
                current_idx,
                bullish=bullish,
                min_gap_pct=fvg_min_gap_pct,
            )
            if refreshed_zone is None:
                return
            refreshed_zone_index, refreshed_lower, refreshed_upper = refreshed_zone
            if setup.zone_index is not None and refreshed_zone_index <= setup.zone_index:
                return
            setup.zone_kind = "fvg"
            setup.zone_index = int(refreshed_zone_index)
            setup.zone_lower = float(refreshed_lower)
            setup.zone_upper = float(refreshed_upper)
            setup.armed_index = current_idx
            setup.expiry_index = current_idx + fvg_max_age
            setup.retest_seen = False
            setup.retest_touches = 0
            setup.continuation_refreshes += 1
            metadata["continuation_zone_refreshes"] += 1

        pending_longs: list[_PendingSetup] = []
        pending_shorts: list[_PendingSetup] = []

        position = 0
        active_stop = np.nan
        active_target = np.nan

        for idx, ts in enumerate(df.index):
            row = df.iloc[idx]
            open_ = float(row["Open"])
            close = float(row["Close"])
            high = float(row["High"])
            low = float(row["Low"])
            current_atr = float(atr.iat[idx]) if pd.notna(atr.iat[idx]) else 0.0
            bias = int(daily_bias.iat[idx]) if idx < len(daily_bias) else 0
            high_regime = bool(regime_series.iat[idx]) if idx < len(regime_series) else False
            higher_bias = int(higher_timeframe_bias.iat[idx]) if idx < len(higher_timeframe_bias) else 0
            pd_context = int(premium_discount_context.iat[idx]) if idx < len(premium_discount_context) else 0
            amd = int(amd_bias.iat[idx]) if idx < len(amd_bias) else 0
            prev_anchor = int(prev_session_anchor_bias.iat[idx]) if idx < len(prev_session_anchor_bias) else 0
            effective_liq_sweep_threshold = liq_sweep_threshold
            effective_fvg_min_gap_pct = fvg_min_gap_pct
            effective_fvg_revisit_depth_ratio = fvg_revisit_depth_ratio
            effective_fvg_revisit_min_delay_bars = fvg_revisit_min_delay_bars
            effective_smt_threshold = smt_threshold
            effective_min_reward_risk_ratio = min_reward_risk_ratio
            if high_regime and bool(self.params.get("use_regime_adaptation", False)):
                high_liq_sweep_threshold = float(self.params.get("regime_high_liq_sweep_threshold", 0.0))
                if high_liq_sweep_threshold > 0.0:
                    effective_liq_sweep_threshold = high_liq_sweep_threshold
                high_fvg_min_gap_pct = float(self.params.get("regime_high_fvg_min_gap_pct", 0.0))
                if high_fvg_min_gap_pct > 0.0:
                    effective_fvg_min_gap_pct = high_fvg_min_gap_pct
                high_fvg_revisit_depth_ratio = float(
                    self.params.get("regime_high_fvg_revisit_depth_ratio", -1.0)
                )
                if high_fvg_revisit_depth_ratio >= 0.0:
                    effective_fvg_revisit_depth_ratio = high_fvg_revisit_depth_ratio
                high_fvg_revisit_min_delay_bars = int(
                    self.params.get("regime_high_fvg_revisit_min_delay_bars", -1)
                )
                if high_fvg_revisit_min_delay_bars >= 0:
                    effective_fvg_revisit_min_delay_bars = high_fvg_revisit_min_delay_bars
                high_smt_threshold = float(self.params.get("regime_high_smt_threshold", 0.0))
                if high_smt_threshold > 0.0:
                    effective_smt_threshold = high_smt_threshold
                high_min_reward_risk_ratio = float(self.params.get("regime_high_min_reward_risk_ratio", 0.0))
                if high_min_reward_risk_ratio > 0.0:
                    effective_min_reward_risk_ratio = high_min_reward_risk_ratio
            in_session = _in_trade_session(
                pd.Timestamp(ts),
                bool(self.params["trade_sessions"]),
                int(self.params["london_open"]),
                int(self.params["london_close"]),
                int(self.params["ny_open"]),
                int(self.params["ny_close"]),
            )
            in_kill_zone = _in_kill_zone(
                pd.Timestamp(ts),
                bool(self.params["use_kill_zones"]),
                str(self.params["kill_zone_timezone"]),
                [
                    (int(self.params["london_kill_start"]), int(self.params["london_kill_end"])),
                    (int(self.params["ny_am_kill_start"]), int(self.params["ny_am_kill_end"])),
                    (int(self.params["ny_pm_kill_start"]), int(self.params["ny_pm_kill_end"])),
                ],
            )
            in_macro_window = _in_macro_window(
                pd.Timestamp(ts),
                use_macro_timing_windows,
                macro_timezone,
                macro_windows,
            )
            dealing_array_window = _classify_dealing_array_window(
                pd.Timestamp(ts),
                use_session_array_refinement,
                dealing_array_timezone,
                imbalance_array_windows,
                structural_array_windows,
            )
            next_is_new_day = (
                idx == len(df.index) - 1
                or not _same_trading_day(df.index[idx + 1], ts)
            )

            if position > 0:
                exit_reason: str | None = None
                if low <= active_stop or close <= active_stop:
                    exits_long.iat[idx] = True
                    position = 0
                    exit_reason = "stop"
                elif high >= active_target or close >= active_target:
                    exits_long.iat[idx] = True
                    position = 0
                    exit_reason = "target"
                elif next_is_new_day:
                    exits_long.iat[idx] = True
                    position = 0
                    exit_reason = "eod"

                if exits_long.iat[idx]:
                    active_stop = np.nan
                    active_target = np.nan
                    if exit_reason == "stop":
                        pending_longs = _retain_reentry_candidates(pending_longs, direction=1)
                    else:
                        pending_longs = []
                    pending_shorts = []
                    continue

            elif position < 0:
                exit_reason: str | None = None
                if high >= active_stop or close >= active_stop:
                    exits_short.iat[idx] = True
                    position = 0
                    exit_reason = "stop"
                elif low <= active_target or close <= active_target:
                    exits_short.iat[idx] = True
                    position = 0
                    exit_reason = "target"
                elif next_is_new_day:
                    exits_short.iat[idx] = True
                    position = 0
                    exit_reason = "eod"

                if exits_short.iat[idx]:
                    active_stop = np.nan
                    active_target = np.nan
                    pending_longs = []
                    if exit_reason == "stop":
                        pending_shorts = _retain_reentry_candidates(pending_shorts, direction=-1)
                    else:
                        pending_shorts = []
                    continue

            if next_is_new_day and position == 0:
                for setup in pending_longs:
                    _record_pending_expiry(setup)
                for setup in pending_shorts:
                    _record_pending_expiry(setup)
                pending_longs = []
                pending_shorts = []

            pending_longs = _expire_pending_setups(pending_longs, current_idx=idx)
            pending_shorts = _expire_pending_setups(pending_shorts, current_idx=idx)

            if allow_long_entries and len(pending_longs) < max_pending_setups_per_direction:
                prior_low = rolling_low.iat[idx]
                sweep_depth = (float(prior_low) - float(low)) if pd.notna(prior_low) else np.nan
                reclaim_level = (
                    float(prior_low) + sweep_depth * liq_sweep_reclaim_ratio
                    if pd.notna(prior_low) and pd.notna(sweep_depth)
                    else np.nan
                )
                if (
                    pd.notna(prior_low)
                    and low < prior_low * (1 - effective_liq_sweep_threshold)
                    and pd.notna(reclaim_level)
                    and close > reclaim_level
                ):
                    metadata["bullish_sweep_candidates"] += 1
                    external_prior_low = external_rolling_low.iat[idx]
                    is_external_sweep = bool(
                        pd.notna(external_prior_low)
                        and prior_low <= float(external_prior_low) * (1 + external_liquidity_tolerance)
                    )
                    peer_data_ready, smt_missing_structure, smt_confirmed = _evaluate_smt_confirmation(
                        idx=idx,
                        direction=1,
                        effective_threshold=effective_smt_threshold,
                    )
                    rejected = False
                    if bool(self.params["use_daily_bias_filter"]) and bias != 1:
                        metadata["daily_bias_filtered_setups"] += 1
                        rejected = True
                    premium_discount_mismatch = bool(
                        self.params["use_premium_discount_filter"]
                    ) and pd_context != 1
                    if premium_discount_mismatch:
                        metadata["premium_discount_mismatch_setups"] += 1
                        if premium_discount_filter_mode == "hard":
                            metadata["premium_discount_filtered_setups"] += 1
                            rejected = True
                        else:
                            metadata["premium_discount_softened_setups"] += 1
                    prev_session_anchor_mismatch = bool(use_prev_session_anchor_filter) and prev_anchor != 1
                    if prev_session_anchor_mismatch:
                        metadata["prev_session_anchor_mismatch_setups"] += 1
                        if prev_session_anchor_filter_mode == "hard":
                            metadata["prev_session_anchor_filtered_setups"] += 1
                            rejected = True
                        else:
                            metadata["prev_session_anchor_softened_setups"] += 1
                    higher_timeframe_mismatch = bool(
                        self.params.get("use_higher_timeframe_alignment", False)
                    ) and higher_bias != 1
                    if higher_timeframe_mismatch:
                        if higher_timeframe_alignment_mode == "hard":
                            metadata["higher_timeframe_filtered_setups"] += 1
                            rejected = True
                        else:
                            metadata["higher_timeframe_softened_setups"] += 1
                    if use_amd_filter and amd != 1:
                        metadata["amd_filtered_setups"] += 1
                        rejected = True
                    if use_external_liquidity_filter and not is_external_sweep:
                        metadata["external_liquidity_filtered_sweeps"] += 1
                        rejected = True
                    if use_smt_filter and smt_missing_structure:
                        metadata["smt_missing_peer_structure"] += 1
                        rejected = True
                    if use_smt_filter and not smt_missing_structure and not peer_data_ready:
                        metadata["smt_missing_peer_bars"] += 1
                        rejected = True
                    if use_smt_filter and peer_data_ready and not smt_confirmed:
                        metadata["smt_filtered_sweeps"] += 1
                        rejected = True
                    if use_macro_timing_windows and not in_macro_window:
                        metadata["macro_timing_filtered_sweeps"] += 1
                        rejected = True
                    if bool(self.params["use_kill_zones"]) and not in_kill_zone:
                        metadata["kill_zone_filtered_sweeps"] += 1
                        rejected = True
                    if not rejected:
                        if structure_reference_mode == "swing":
                            reference_high = _latest_confirmed_swing_level(
                                df["High"],
                                idx,
                                lookback=structure_lookback,
                                threshold=swing_threshold,
                                swing_type="high",
                            )
                        else:
                            start = max(0, idx - structure_lookback)
                            reference_high = float(df["High"].iloc[start:idx].max()) if idx > start else np.nan
                        if pd.notna(reference_high):
                            primary_expiry_index = idx + liq_sweep_recovery_bars
                            final_expiry_index = idx + slow_recovery_bars
                            pending_longs.append(
                                _PendingSetup(
                                    direction=1,
                                    sweep_index=idx,
                                    sweep_level=low,
                                    sweep_reference_level=float(prior_low),
                                    sweep_depth=float(sweep_depth),
                                    structure_level=reference_high,
                                    shift_extreme=high,
                                    expiry_index=final_expiry_index,
                                    primary_expiry_index=primary_expiry_index,
                                    premium_discount_mismatch=premium_discount_mismatch,
                                    prev_session_anchor_mismatch=prev_session_anchor_mismatch,
                                )
                            )
                            metadata["bullish_sweeps"] += 1
                        elif structure_reference_mode == "swing":
                            metadata["swing_structure_missing_reference"] += 1
                        if is_external_sweep:
                            metadata["external_targeted_sweeps"] += 1
                        if use_smt_filter and smt_confirmed:
                            metadata["smt_confirmed_sweeps"] += 1
            elif allow_long_entries:
                prior_low = rolling_low.iat[idx]
                sweep_depth = (float(prior_low) - float(low)) if pd.notna(prior_low) else np.nan
                reclaim_level = (
                    float(prior_low) + sweep_depth * liq_sweep_reclaim_ratio
                    if pd.notna(prior_low) and pd.notna(sweep_depth)
                    else np.nan
                )
                if (
                    pd.notna(prior_low)
                    and low < prior_low * (1 - effective_liq_sweep_threshold)
                    and pd.notna(reclaim_level)
                    and close > reclaim_level
                ):
                    metadata["bullish_sweep_candidates"] += 1
                    metadata["sweep_blocked_by_existing_pending"] += 1

            if allow_short_entries and len(pending_shorts) < max_pending_setups_per_direction:
                prior_high = rolling_high.iat[idx]
                sweep_depth = (float(high) - float(prior_high)) if pd.notna(prior_high) else np.nan
                reclaim_level = (
                    float(prior_high) - sweep_depth * liq_sweep_reclaim_ratio
                    if pd.notna(prior_high) and pd.notna(sweep_depth)
                    else np.nan
                )
                if (
                    pd.notna(prior_high)
                    and high > prior_high * (1 + effective_liq_sweep_threshold)
                    and pd.notna(reclaim_level)
                    and close < reclaim_level
                ):
                    metadata["bearish_sweep_candidates"] += 1
                    external_prior_high = external_rolling_high.iat[idx]
                    is_external_sweep = bool(
                        pd.notna(external_prior_high)
                        and prior_high >= float(external_prior_high) * (1 - external_liquidity_tolerance)
                    )
                    peer_data_ready, smt_missing_structure, smt_confirmed = _evaluate_smt_confirmation(
                        idx=idx,
                        direction=-1,
                        effective_threshold=effective_smt_threshold,
                    )
                    rejected = False
                    if bool(self.params["use_daily_bias_filter"]) and bias != -1:
                        metadata["daily_bias_filtered_setups"] += 1
                        rejected = True
                    premium_discount_mismatch = bool(
                        self.params["use_premium_discount_filter"]
                    ) and pd_context != -1
                    if premium_discount_mismatch:
                        metadata["premium_discount_mismatch_setups"] += 1
                        if premium_discount_filter_mode == "hard":
                            metadata["premium_discount_filtered_setups"] += 1
                            rejected = True
                        else:
                            metadata["premium_discount_softened_setups"] += 1
                    prev_session_anchor_mismatch = bool(use_prev_session_anchor_filter) and prev_anchor != -1
                    if prev_session_anchor_mismatch:
                        metadata["prev_session_anchor_mismatch_setups"] += 1
                        if prev_session_anchor_filter_mode == "hard":
                            metadata["prev_session_anchor_filtered_setups"] += 1
                            rejected = True
                        else:
                            metadata["prev_session_anchor_softened_setups"] += 1
                    higher_timeframe_mismatch = bool(
                        self.params.get("use_higher_timeframe_alignment", False)
                    ) and higher_bias != -1
                    if higher_timeframe_mismatch:
                        if higher_timeframe_alignment_mode == "hard":
                            metadata["higher_timeframe_filtered_setups"] += 1
                            rejected = True
                        else:
                            metadata["higher_timeframe_softened_setups"] += 1
                    if use_amd_filter and amd != -1:
                        metadata["amd_filtered_setups"] += 1
                        rejected = True
                    if use_external_liquidity_filter and not is_external_sweep:
                        metadata["external_liquidity_filtered_sweeps"] += 1
                        rejected = True
                    if use_smt_filter and smt_missing_structure:
                        metadata["smt_missing_peer_structure"] += 1
                        rejected = True
                    if use_smt_filter and not smt_missing_structure and not peer_data_ready:
                        metadata["smt_missing_peer_bars"] += 1
                        rejected = True
                    if use_smt_filter and peer_data_ready and not smt_confirmed:
                        metadata["smt_filtered_sweeps"] += 1
                        rejected = True
                    if use_macro_timing_windows and not in_macro_window:
                        metadata["macro_timing_filtered_sweeps"] += 1
                        rejected = True
                    if bool(self.params["use_kill_zones"]) and not in_kill_zone:
                        metadata["kill_zone_filtered_sweeps"] += 1
                        rejected = True
                    if not rejected:
                        if structure_reference_mode == "swing":
                            reference_low = _latest_confirmed_swing_level(
                                df["Low"],
                                idx,
                                lookback=structure_lookback,
                                threshold=swing_threshold,
                                swing_type="low",
                            )
                        else:
                            start = max(0, idx - structure_lookback)
                            reference_low = float(df["Low"].iloc[start:idx].min()) if idx > start else np.nan
                        if pd.notna(reference_low):
                            primary_expiry_index = idx + liq_sweep_recovery_bars
                            final_expiry_index = idx + slow_recovery_bars
                            pending_shorts.append(
                                _PendingSetup(
                                    direction=-1,
                                    sweep_index=idx,
                                    sweep_level=high,
                                    sweep_reference_level=float(prior_high),
                                    sweep_depth=float(sweep_depth),
                                    structure_level=reference_low,
                                    shift_extreme=low,
                                    expiry_index=final_expiry_index,
                                    primary_expiry_index=primary_expiry_index,
                                    premium_discount_mismatch=premium_discount_mismatch,
                                    prev_session_anchor_mismatch=prev_session_anchor_mismatch,
                                )
                            )
                            metadata["bearish_sweeps"] += 1
                        elif structure_reference_mode == "swing":
                            metadata["swing_structure_missing_reference"] += 1
                        if is_external_sweep:
                            metadata["external_targeted_sweeps"] += 1
                        if use_smt_filter and smt_confirmed:
                            metadata["smt_confirmed_sweeps"] += 1
            elif allow_short_entries:
                prior_high = rolling_high.iat[idx]
                sweep_depth = (float(high) - float(prior_high)) if pd.notna(prior_high) else np.nan
                reclaim_level = (
                    float(prior_high) - sweep_depth * liq_sweep_reclaim_ratio
                    if pd.notna(prior_high) and pd.notna(sweep_depth)
                    else np.nan
                )
                if (
                    pd.notna(prior_high)
                    and high > prior_high * (1 + effective_liq_sweep_threshold)
                    and pd.notna(reclaim_level)
                    and close < reclaim_level
                ):
                    metadata["bearish_sweep_candidates"] += 1
                    metadata["sweep_blocked_by_existing_pending"] += 1

            for pending_long in list(pending_longs):
                if pending_long.state != "swept":
                    continue
                pending_long.shift_extreme = max(pending_long.shift_extreme, high)
                candle_range = max(high - low, 1e-12)
                intrabar_shift_confirmed = False
                shift_triggered = close > pending_long.structure_level
                if (
                    not shift_triggered
                    and structure_shift_intrabar_tolerance_ratio > 0.0
                    and high > pending_long.structure_level
                    and close >= open_
                ):
                    metadata["intrabar_shift_candidates"] += 1
                    close_floor = (
                        pending_long.structure_level - candle_range * structure_shift_intrabar_tolerance_ratio
                    )
                    close_position = _close_position_ratio(high, low, close)
                    if (
                        close >= close_floor
                        and close_position >= structure_shift_intrabar_close_position_min_pct
                    ):
                        shift_triggered = True
                        intrabar_shift_confirmed = True
                if shift_triggered and in_kill_zone:
                    metadata["bullish_shift_candidates"] += 1
                    shift_rejected = False
                    required_close_buffer = candle_range * structure_shift_close_buffer_ratio
                    if close - pending_long.structure_level < required_close_buffer:
                        metadata["structure_buffer_filtered_shifts"] += 1
                        shift_rejected = True
                    body_ratio = abs(close - open_) / candle_range
                    if not shift_rejected and body_ratio < displacement_body_min_pct:
                        metadata["displacement_filtered_shifts"] += 1
                        shift_rejected = True
                    if (
                        not shift_rejected
                        and displacement_range_atr_mult > 0.0
                        and current_atr > 0.0
                    ):
                        required_range = current_atr * displacement_range_atr_mult
                        if candle_range < required_range:
                            metadata["displacement_range_filtered_shifts"] += 1
                            shift_rejected = True
                    if not shift_rejected:
                        if intrabar_shift_confirmed:
                            metadata["intrabar_shift_confirmed"] += 1
                        zone = _detect_fvg_zone(
                            df,
                            pending_long.sweep_index,
                            idx,
                            bullish=True,
                            min_gap_pct=effective_fvg_min_gap_pct,
                        )
                        zone_kind = "fvg"
                        if zone is not None and fvg_origin_max_lag_bars > 0:
                            zone_index = int(zone[0])
                            if idx - zone_index > fvg_origin_max_lag_bars:
                                metadata["fvg_origin_lag_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_body_min_pct > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_open = float(df["Open"].iat[origin_idx])
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            origin_body_ratio = _body_ratio(origin_open, origin_high, origin_low, origin_close)
                            if origin_close <= origin_open or origin_body_ratio < fvg_origin_body_min_pct:
                                metadata["fvg_origin_body_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_body_atr_mult > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_open = float(df["Open"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            origin_atr = float(atr.iat[origin_idx]) if origin_idx < len(atr) else 0.0
                            origin_body = abs(origin_close - origin_open)
                            if origin_atr <= 0.0 or origin_body < origin_atr * fvg_origin_body_atr_mult:
                                metadata["fvg_origin_body_atr_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_close_position_min_pct > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            origin_close_position = _close_position_ratio(origin_high, origin_low, origin_close)
                            if origin_close_position < fvg_origin_close_position_min_pct:
                                metadata["fvg_origin_close_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_opposite_wick_max_pct > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_open = float(df["Open"].iat[origin_idx])
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            opposite_wick_ratio = _opposite_wick_ratio(
                                origin_open,
                                origin_high,
                                origin_low,
                                origin_close,
                                bullish=True,
                            )
                            if opposite_wick_ratio > fvg_origin_opposite_wick_max_pct:
                                metadata["fvg_origin_wick_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_range_atr_mult > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_atr = float(atr.iat[origin_idx]) if origin_idx < len(atr) else 0.0
                            if origin_atr <= 0.0 or (origin_high - origin_low) < origin_atr * fvg_origin_range_atr_mult:
                                metadata["fvg_origin_range_filtered_shifts"] += 1
                                zone = None
                        if zone is None and require_fvg_delivery:
                            metadata["fvg_required_filtered_shifts"] += 1
                        if zone is None and not require_fvg_delivery:
                            zone = _detect_order_block_zone(
                                df,
                                pending_long.sweep_index,
                                idx,
                                bullish=True,
                                body_min_pct=ob_body_min_pct,
                                lookback=ob_lookback,
                            )
                            zone_kind = "ob"
                        if zone is None and not require_fvg_delivery:
                            zone = _detect_breaker_block_zone(
                                df,
                                pending_long.sweep_index,
                                idx,
                                bullish=True,
                                lookback=breaker_lookback,
                            )
                            zone_kind = "breaker"
                        if zone is None and not require_fvg_delivery:
                            zone = _detect_ifvg_zone(
                                df,
                                pending_long.sweep_index,
                                idx,
                                bullish=True,
                                lookback=ifvg_lookback,
                            )
                            zone_kind = "ifvg"

                        if zone is not None:
                            zone_allowed = True
                            if dealing_array_window == "imbalance":
                                zone_allowed = zone_kind in {"fvg", "ifvg"}
                            elif dealing_array_window == "structural":
                                zone_allowed = zone_kind in {"ob", "breaker"}
                            if use_session_array_refinement and not zone_allowed:
                                if session_array_filter_mode == "hard":
                                    metadata["session_array_filtered_shifts"] += 1
                                    zone = None
                                else:
                                    metadata["session_array_softened_shifts"] += 1

                        if zone is not None:
                            _, zone_lower, zone_upper = zone
                            score = score_liquidity_sweep + _structure_confirmation_score(
                                score_bos,
                                score_choch,
                            )
                            if zone_kind == "fvg":
                                score += score_fvg
                            elif zone_kind == "ifvg":
                                score += score_ifvg
                            elif zone_kind == "breaker":
                                score += score_breaker_block
                            else:
                                score += score_order_block
                            quality_bonus = _quality_score_bonus(
                                sweep_depth=pending_long.sweep_depth,
                                sweep_reference_level=pending_long.sweep_reference_level,
                                liq_sweep_threshold=effective_liq_sweep_threshold,
                                displacement_body_ratio=body_ratio,
                                displacement_body_min_pct=displacement_body_min_pct,
                                fvg_gap_size=max(zone_upper - zone_lower, 0.0),
                                price=close,
                                fvg_min_gap_pct=effective_fvg_min_gap_pct,
                                score_sweep_depth_quality=score_sweep_depth_quality,
                                score_displacement_quality=score_displacement_quality,
                                score_fvg_gap_quality=score_fvg_gap_quality,
                            )
                            score += quality_bonus
                            if quality_bonus > 0.0:
                                metadata["score_quality_boosted_shifts"] += 1
                                metadata["score_quality_bonus_total"] += quality_bonus
                            if (
                                use_session_array_refinement
                                and not zone_allowed
                                and session_array_filter_mode == "soft"
                            ):
                                score = max(0.0, score - session_array_mismatch_score_penalty)
                                if session_array_mismatch_score_penalty > 0.0:
                                    metadata["session_array_score_penalty_shifts"] += 1
                            if pending_long.premium_discount_mismatch and premium_discount_filter_mode == "soft":
                                score = max(0.0, score - premium_discount_mismatch_score_penalty)
                                if premium_discount_mismatch_score_penalty > 0.0:
                                    metadata["premium_discount_score_penalty_shifts"] += 1
                            if (
                                pending_long.prev_session_anchor_mismatch
                                and prev_session_anchor_filter_mode == "soft"
                            ):
                                score = max(0.0, score - prev_session_anchor_mismatch_score_penalty)
                                if prev_session_anchor_mismatch_score_penalty > 0.0:
                                    metadata["prev_session_anchor_score_penalty_shifts"] += 1
                            if (
                                bool(self.params.get("use_higher_timeframe_alignment", False))
                                and higher_bias != 1
                                and higher_timeframe_alignment_mode == "soft"
                            ):
                                score = max(0.0, score - higher_timeframe_mismatch_score_penalty)
                                if higher_timeframe_mismatch_score_penalty > 0.0:
                                    metadata["higher_timeframe_score_penalty_shifts"] += 1
                            zone_in_ote = _zone_in_ote(
                                1,
                                zone_lower=zone_lower,
                                zone_upper=zone_upper,
                                sweep_level=pending_long.sweep_level,
                                shift_extreme=pending_long.shift_extreme,
                                fib_low=ote_fib_low,
                                fib_high=ote_fib_high,
                            )
                            if require_ote_zone and not zone_in_ote:
                                metadata["ote_required_filtered_shifts"] += 1
                                zone = None
                            elif zone_in_ote:
                                score += score_ote_zone
                            if zone is not None and score >= min_score_to_trade:
                                recovery_phase = _recovery_phase(pending_long, idx)
                                pending_long.state = "armed"
                                pending_long.armed_index = idx
                                pending_long.zone_kind = zone_kind
                                pending_long.zone_lower = zone_lower
                                pending_long.zone_upper = zone_upper
                                pending_long.zone_index = int(zone[0])
                                pending_long.score = score
                                pending_long.expiry_index = idx + fvg_max_age
                                pending_long.recovery_phase = recovery_phase
                                metadata["bullish_shifts"] += 1
                                metadata[f"{recovery_phase}_recovery_shifts"] += 1
                            elif zone is not None:
                                metadata["score_filtered_shifts"] += 1
                        else:
                            metadata["delivery_missing_shifts"] += 1

            for pending_short in list(pending_shorts):
                if pending_short.state != "swept":
                    continue
                pending_short.shift_extreme = min(pending_short.shift_extreme, low)
                candle_range = max(high - low, 1e-12)
                intrabar_shift_confirmed = False
                shift_triggered = close < pending_short.structure_level
                if (
                    not shift_triggered
                    and structure_shift_intrabar_tolerance_ratio > 0.0
                    and low < pending_short.structure_level
                    and close <= open_
                ):
                    metadata["intrabar_shift_candidates"] += 1
                    close_ceiling = (
                        pending_short.structure_level + candle_range * structure_shift_intrabar_tolerance_ratio
                    )
                    close_position = _close_position_ratio(high, low, close)
                    if (
                        close <= close_ceiling
                        and close_position <= (1.0 - structure_shift_intrabar_close_position_min_pct)
                    ):
                        shift_triggered = True
                        intrabar_shift_confirmed = True
                if shift_triggered and in_kill_zone:
                    metadata["bearish_shift_candidates"] += 1
                    shift_rejected = False
                    required_close_buffer = candle_range * structure_shift_close_buffer_ratio
                    if pending_short.structure_level - close < required_close_buffer:
                        metadata["structure_buffer_filtered_shifts"] += 1
                        shift_rejected = True
                    body_ratio = abs(close - open_) / candle_range
                    if not shift_rejected and body_ratio < displacement_body_min_pct:
                        metadata["displacement_filtered_shifts"] += 1
                        shift_rejected = True
                    if (
                        not shift_rejected
                        and displacement_range_atr_mult > 0.0
                        and current_atr > 0.0
                    ):
                        required_range = current_atr * displacement_range_atr_mult
                        if candle_range < required_range:
                            metadata["displacement_range_filtered_shifts"] += 1
                            shift_rejected = True
                    if not shift_rejected:
                        if intrabar_shift_confirmed:
                            metadata["intrabar_shift_confirmed"] += 1
                        zone = _detect_fvg_zone(
                            df,
                            pending_short.sweep_index,
                            idx,
                            bullish=False,
                            min_gap_pct=effective_fvg_min_gap_pct,
                        )
                        zone_kind = "fvg"
                        if zone is not None and fvg_origin_max_lag_bars > 0:
                            zone_index = int(zone[0])
                            if idx - zone_index > fvg_origin_max_lag_bars:
                                metadata["fvg_origin_lag_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_body_min_pct > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_open = float(df["Open"].iat[origin_idx])
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            origin_body_ratio = _body_ratio(origin_open, origin_high, origin_low, origin_close)
                            if origin_close >= origin_open or origin_body_ratio < fvg_origin_body_min_pct:
                                metadata["fvg_origin_body_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_body_atr_mult > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_open = float(df["Open"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            origin_atr = float(atr.iat[origin_idx]) if origin_idx < len(atr) else 0.0
                            origin_body = abs(origin_close - origin_open)
                            if origin_atr <= 0.0 or origin_body < origin_atr * fvg_origin_body_atr_mult:
                                metadata["fvg_origin_body_atr_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_close_position_min_pct > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            origin_close_position = 1.0 - _close_position_ratio(origin_high, origin_low, origin_close)
                            if origin_close_position < fvg_origin_close_position_min_pct:
                                metadata["fvg_origin_close_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_opposite_wick_max_pct > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_open = float(df["Open"].iat[origin_idx])
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_close = float(df["Close"].iat[origin_idx])
                            opposite_wick_ratio = _opposite_wick_ratio(
                                origin_open,
                                origin_high,
                                origin_low,
                                origin_close,
                                bullish=False,
                            )
                            if opposite_wick_ratio > fvg_origin_opposite_wick_max_pct:
                                metadata["fvg_origin_wick_filtered_shifts"] += 1
                                zone = None
                        if zone is not None and fvg_origin_range_atr_mult > 0.0:
                            zone_index = int(zone[0])
                            origin_idx = max(zone_index - 1, 0)
                            origin_high = float(df["High"].iat[origin_idx])
                            origin_low = float(df["Low"].iat[origin_idx])
                            origin_atr = float(atr.iat[origin_idx]) if origin_idx < len(atr) else 0.0
                            if origin_atr <= 0.0 or (origin_high - origin_low) < origin_atr * fvg_origin_range_atr_mult:
                                metadata["fvg_origin_range_filtered_shifts"] += 1
                                zone = None
                        if zone is None and require_fvg_delivery:
                            metadata["fvg_required_filtered_shifts"] += 1
                        if zone is None and not require_fvg_delivery:
                            zone = _detect_order_block_zone(
                                df,
                                pending_short.sweep_index,
                                idx,
                                bullish=False,
                                body_min_pct=ob_body_min_pct,
                                lookback=ob_lookback,
                            )
                            zone_kind = "ob"
                        if zone is None and not require_fvg_delivery:
                            zone = _detect_breaker_block_zone(
                                df,
                                pending_short.sweep_index,
                                idx,
                                bullish=False,
                                lookback=breaker_lookback,
                            )
                            zone_kind = "breaker"
                        if zone is None and not require_fvg_delivery:
                            zone = _detect_ifvg_zone(
                                df,
                                pending_short.sweep_index,
                                idx,
                                bullish=False,
                                lookback=ifvg_lookback,
                            )
                            zone_kind = "ifvg"

                        if zone is not None:
                            zone_allowed = True
                            if dealing_array_window == "imbalance":
                                zone_allowed = zone_kind in {"fvg", "ifvg"}
                            elif dealing_array_window == "structural":
                                zone_allowed = zone_kind in {"ob", "breaker"}
                            if use_session_array_refinement and not zone_allowed:
                                if session_array_filter_mode == "hard":
                                    metadata["session_array_filtered_shifts"] += 1
                                    zone = None
                                else:
                                    metadata["session_array_softened_shifts"] += 1

                        if zone is not None:
                            _, zone_lower, zone_upper = zone
                            score = score_liquidity_sweep + _structure_confirmation_score(
                                score_bos,
                                score_choch,
                            )
                            if zone_kind == "fvg":
                                score += score_fvg
                            elif zone_kind == "ifvg":
                                score += score_ifvg
                            elif zone_kind == "breaker":
                                score += score_breaker_block
                            else:
                                score += score_order_block
                            quality_bonus = _quality_score_bonus(
                                sweep_depth=pending_short.sweep_depth,
                                sweep_reference_level=pending_short.sweep_reference_level,
                                liq_sweep_threshold=effective_liq_sweep_threshold,
                                displacement_body_ratio=body_ratio,
                                displacement_body_min_pct=displacement_body_min_pct,
                                fvg_gap_size=max(zone_upper - zone_lower, 0.0),
                                price=close,
                                fvg_min_gap_pct=effective_fvg_min_gap_pct,
                                score_sweep_depth_quality=score_sweep_depth_quality,
                                score_displacement_quality=score_displacement_quality,
                                score_fvg_gap_quality=score_fvg_gap_quality,
                            )
                            score += quality_bonus
                            if quality_bonus > 0.0:
                                metadata["score_quality_boosted_shifts"] += 1
                                metadata["score_quality_bonus_total"] += quality_bonus
                            if (
                                use_session_array_refinement
                                and not zone_allowed
                                and session_array_filter_mode == "soft"
                            ):
                                score = max(0.0, score - session_array_mismatch_score_penalty)
                                if session_array_mismatch_score_penalty > 0.0:
                                    metadata["session_array_score_penalty_shifts"] += 1
                            if pending_short.premium_discount_mismatch and premium_discount_filter_mode == "soft":
                                score = max(0.0, score - premium_discount_mismatch_score_penalty)
                                if premium_discount_mismatch_score_penalty > 0.0:
                                    metadata["premium_discount_score_penalty_shifts"] += 1
                            if (
                                pending_short.prev_session_anchor_mismatch
                                and prev_session_anchor_filter_mode == "soft"
                            ):
                                score = max(0.0, score - prev_session_anchor_mismatch_score_penalty)
                                if prev_session_anchor_mismatch_score_penalty > 0.0:
                                    metadata["prev_session_anchor_score_penalty_shifts"] += 1
                            if (
                                bool(self.params.get("use_higher_timeframe_alignment", False))
                                and higher_bias != -1
                                and higher_timeframe_alignment_mode == "soft"
                            ):
                                score = max(0.0, score - higher_timeframe_mismatch_score_penalty)
                                if higher_timeframe_mismatch_score_penalty > 0.0:
                                    metadata["higher_timeframe_score_penalty_shifts"] += 1
                            zone_in_ote = _zone_in_ote(
                                -1,
                                zone_lower=zone_lower,
                                zone_upper=zone_upper,
                                sweep_level=pending_short.sweep_level,
                                shift_extreme=pending_short.shift_extreme,
                                fib_low=ote_fib_low,
                                fib_high=ote_fib_high,
                            )
                            if require_ote_zone and not zone_in_ote:
                                metadata["ote_required_filtered_shifts"] += 1
                                zone = None
                            elif zone_in_ote:
                                score += score_ote_zone
                            if zone is not None and score >= min_score_to_trade:
                                recovery_phase = _recovery_phase(pending_short, idx)
                                pending_short.state = "armed"
                                pending_short.armed_index = idx
                                pending_short.zone_kind = zone_kind
                                pending_short.zone_lower = zone_lower
                                pending_short.zone_upper = zone_upper
                                pending_short.zone_index = int(zone[0])
                                pending_short.score = score
                                pending_short.expiry_index = idx + fvg_max_age
                                pending_short.recovery_phase = recovery_phase
                                metadata["bearish_shifts"] += 1
                                metadata[f"{recovery_phase}_recovery_shifts"] += 1
                            elif zone is not None:
                                metadata["score_filtered_shifts"] += 1
                        else:
                            metadata["delivery_missing_shifts"] += 1

            long_entry_executed = False
            if position == 0 and in_session and in_kill_zone:
                for pending_long in pending_longs:
                    _refresh_continuation_zone(
                        pending_long,
                        current_idx=idx,
                        bullish=True,
                    )
                for pending_long in _eligible_armed_setups(
                    pending_longs,
                    current_idx=idx,
                    bar_low=low,
                    bar_high=high,
                ):
                    if not pending_long.retest_seen:
                        metadata["bullish_retest_candidates"] += 1
                        pending_long.retest_seen = True
                    long_entry_resolved = False
                    if pending_long.zone_kind in {"fvg", "ifvg"}:
                        pending_long.retest_touches += 1
                        if (
                            fvg_max_retest_touches > 0
                            and pending_long.retest_touches > fvg_max_retest_touches
                        ):
                            metadata["fvg_touch_filtered_retests"] += 1
                            pending_longs.remove(pending_long)
                            long_entry_resolved = True
                    if (
                        not long_entry_resolved
                        and pending_long.armed_index is not None
                        and idx - pending_long.armed_index < effective_fvg_revisit_min_delay_bars
                    ):
                        metadata["fvg_delay_filtered_retests"] += 1
                        long_entry_resolved = True
                    if not long_entry_resolved and pending_long.zone_kind in {"fvg", "ifvg"}:
                        gap_height = max(pending_long.zone_upper - pending_long.zone_lower, 1e-12)
                        required_touch = pending_long.zone_upper - gap_height * effective_fvg_revisit_depth_ratio
                        if low > required_touch:
                            metadata["fvg_depth_filtered_retests"] += 1
                            long_entry_resolved = True
                        if not long_entry_resolved and fvg_rejection_close_ratio > 0.0:
                            required_close = pending_long.zone_lower + gap_height * fvg_rejection_close_ratio
                            if close < required_close:
                                metadata["fvg_close_filtered_retests"] += 1
                                long_entry_resolved = True
                        if not long_entry_resolved and fvg_rejection_wick_ratio > 0.0:
                            body_low = min(open_, close)
                            required_wick = gap_height * fvg_rejection_wick_ratio
                            if body_low - low < required_wick:
                                metadata["fvg_wick_filtered_retests"] += 1
                                long_entry_resolved = True
                        if not long_entry_resolved and fvg_rejection_body_min_pct > 0.0:
                            candle_range = max(high - low, 1e-12)
                            body_ratio = abs(close - open_) / candle_range
                            if close <= open_ or body_ratio < fvg_rejection_body_min_pct:
                                metadata["fvg_body_filtered_retests"] += 1
                                long_entry_resolved = True
                    if long_entry_resolved:
                        continue
                    risk_anchor = min(pending_long.sweep_level, pending_long.zone_lower)
                    atr_buffer = max(current_atr * stop_loss_atr_mult, close * 0.002)
                    candidate_stop = min(risk_anchor - current_atr * 0.1, close - atr_buffer)
                    if candidate_stop >= close:
                        candidate_stop = close - atr_buffer
                    candidate_target = close + (close - candidate_stop) * take_profit_rr
                    projected_rr = _project_reward_risk_ratio(close, candidate_stop, candidate_target)
                    if projected_rr < effective_min_reward_risk_ratio:
                        metadata["rr_filtered_entries"] += 1
                        continue
                    pending_long.entry_attempts += 1
                    entries_long.iat[idx] = True
                    position = 1
                    active_stop = candidate_stop
                    active_target = candidate_target
                    metadata["long_entries"] += 1
                    if pending_long.entry_attempts > 1:
                        metadata["reentry_entries"] += 1
                    if pending_long.continuation_refreshes > 0:
                        metadata["continuation_entries"] += 1
                    metadata[f"{pending_long.recovery_phase}_recovery_entries"] += 1
                    if pending_long.zone_kind == "fvg":
                        metadata["fvg_entries"] += 1
                    elif pending_long.zone_kind == "ifvg":
                        metadata["ifvg_entries"] += 1
                    elif pending_long.zone_kind == "breaker":
                        metadata["breaker_entries"] += 1
                    else:
                        metadata["ob_entries"] += 1
                    if max_reentries_per_setup > 0:
                        pending_longs = [pending_long]
                    else:
                        pending_longs = []
                    pending_shorts = []
                    long_entry_executed = True
                    break

            if not long_entry_executed and position == 0 and in_session and in_kill_zone:
                for pending_short in pending_shorts:
                    _refresh_continuation_zone(
                        pending_short,
                        current_idx=idx,
                        bullish=False,
                    )
                for pending_short in _eligible_armed_setups(
                    pending_shorts,
                    current_idx=idx,
                    bar_low=low,
                    bar_high=high,
                ):
                    if not pending_short.retest_seen:
                        metadata["bearish_retest_candidates"] += 1
                        pending_short.retest_seen = True
                    short_entry_resolved = False
                    if pending_short.zone_kind in {"fvg", "ifvg"}:
                        pending_short.retest_touches += 1
                        if (
                            fvg_max_retest_touches > 0
                            and pending_short.retest_touches > fvg_max_retest_touches
                        ):
                            metadata["fvg_touch_filtered_retests"] += 1
                            pending_shorts.remove(pending_short)
                            short_entry_resolved = True
                    if (
                        not short_entry_resolved
                        and pending_short.armed_index is not None
                        and idx - pending_short.armed_index < effective_fvg_revisit_min_delay_bars
                    ):
                        metadata["fvg_delay_filtered_retests"] += 1
                        short_entry_resolved = True
                    if not short_entry_resolved and pending_short.zone_kind in {"fvg", "ifvg"}:
                        gap_height = max(pending_short.zone_upper - pending_short.zone_lower, 1e-12)
                        required_touch = pending_short.zone_lower + gap_height * effective_fvg_revisit_depth_ratio
                        if high < required_touch:
                            metadata["fvg_depth_filtered_retests"] += 1
                            short_entry_resolved = True
                        if not short_entry_resolved and fvg_rejection_close_ratio > 0.0:
                            required_close = pending_short.zone_upper - gap_height * fvg_rejection_close_ratio
                            if close > required_close:
                                metadata["fvg_close_filtered_retests"] += 1
                                short_entry_resolved = True
                        if not short_entry_resolved and fvg_rejection_wick_ratio > 0.0:
                            body_high = max(open_, close)
                            required_wick = gap_height * fvg_rejection_wick_ratio
                            if high - body_high < required_wick:
                                metadata["fvg_wick_filtered_retests"] += 1
                                short_entry_resolved = True
                        if not short_entry_resolved and fvg_rejection_body_min_pct > 0.0:
                            candle_range = max(high - low, 1e-12)
                            body_ratio = abs(close - open_) / candle_range
                            if close >= open_ or body_ratio < fvg_rejection_body_min_pct:
                                metadata["fvg_body_filtered_retests"] += 1
                                short_entry_resolved = True
                    if short_entry_resolved:
                        continue
                    risk_anchor = max(pending_short.sweep_level, pending_short.zone_upper)
                    atr_buffer = max(current_atr * stop_loss_atr_mult, close * 0.002)
                    candidate_stop = max(risk_anchor + current_atr * 0.1, close + atr_buffer)
                    if candidate_stop <= close:
                        candidate_stop = close + atr_buffer
                    candidate_target = close - (candidate_stop - close) * take_profit_rr
                    projected_rr = _project_reward_risk_ratio(close, candidate_stop, candidate_target)
                    if projected_rr < effective_min_reward_risk_ratio:
                        metadata["rr_filtered_entries"] += 1
                        continue
                    pending_short.entry_attempts += 1
                    entries_short.iat[idx] = True
                    position = -1
                    active_stop = candidate_stop
                    active_target = candidate_target
                    metadata["short_entries"] += 1
                    if pending_short.entry_attempts > 1:
                        metadata["reentry_entries"] += 1
                    if pending_short.continuation_refreshes > 0:
                        metadata["continuation_entries"] += 1
                    metadata[f"{pending_short.recovery_phase}_recovery_entries"] += 1
                    if pending_short.zone_kind == "fvg":
                        metadata["fvg_entries"] += 1
                    elif pending_short.zone_kind == "ifvg":
                        metadata["ifvg_entries"] += 1
                    elif pending_short.zone_kind == "breaker":
                        metadata["breaker_entries"] += 1
                    else:
                        metadata["ob_entries"] += 1
                    pending_longs = []
                    if max_reentries_per_setup > 0:
                        pending_shorts = [pending_short]
                    else:
                        pending_shorts = []
                    break

        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            metadata=metadata,
        )
