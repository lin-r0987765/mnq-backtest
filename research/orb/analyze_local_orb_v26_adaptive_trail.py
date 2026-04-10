#!/usr/bin/env python3
"""
Alpaca-backed local research for adaptive trailing tightening after profit-lock.

Why this exists:
- v26-profit-lock is the official baseline.
- The weakness map shows the `late_washout` bucket (38 trades, -$2792,
  opportunity gap $4936) as the largest unrealised upside.
- Prior late-session exit branches (stagnation timeout, stall-plus-giveback,
  low-progress timeout, mid-trade ratchet, ORB re-entry) are all closed.
- This branch is structurally different: instead of adding a fixed floor or
  a time-based exit, it dynamically narrows the 1.3% trailing stop after
  profit-lock confirms a strong trade, progressively locking more profit
  as the trade ages.

Mechanism tested here:
- Keep the full v26 baseline unchanged:
  - BE trigger = 1.25 x ORB range during first 180 minutes
  - persistent profit lock = +0.25 x ORB range after 1.50 x ORB range MFE
- Add optional adaptive trail tightening after profit-lock:
  - once profit-lock activates, wait `tighten_delay_bars` additional bars
  - then linearly interpolate trailing_pct from 1.3% toward `tighten_target_pct`
    over `tighten_step_bars` bars
  - effective_trail_pct = trailing_pct - progress * (trailing_pct - target_pct)
    where progress = min(1.0, bars_since_delay / tighten_step_bars)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from research.orb.analyze_alpaca_v26_reference import build_time_folds, compute_fold_metrics
from research.orb.analyze_local_orb_v25_profit_lock import (
    BASE_BE_GATE_MIN,
    BASE_BE_TRIGGER,
    BASE_PARAMS,
    MAX_ENTRIES_PER_SESSION,
    PROJECT_ROOT,
    analyze_path_impact,
    compute_htf_bias,
    compute_metrics,
    load_csv_5m,
    simulate_orb_v25_profit_lock,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
DEFAULT_INTRADAY_PATH = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
OUTPUT_PATH = RESULTS_DIR / "local_orb_v26_adaptive_trail_alpaca.json"
WEAKNESS_MAP_PATH = RESULTS_DIR / "v26_weakness_map.json"
ANALYSIS_VERSION = "v1_alpaca_v26_adaptive_trail"

BASE_V26_PROFIT_LOCK_TRIGGER = 1.50
BASE_V26_PROFIT_LOCK_LEVEL = 0.25

# --- Grid ---
TIGHTEN_DELAY_BARS = [0, 2, 4, 6]
TIGHTEN_TARGET_PCTS = [0.006, 0.008, 0.010]
TIGHTEN_STEP_BARS = [4, 8, 12, 20]


@dataclass
class AdaptiveTrailTradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    duration_min: float
    exit_reason: str
    mfe: float
    mae: float
    be_activated: bool
    be_gate_expired: bool
    profit_lock_activated: bool
    bars_in_trade: int
    adaptive_trail_active: bool
    final_effective_trail_pct: float


def load_weakness_reference() -> dict:
    if not WEAKNESS_MAP_PATH.exists():
        return {"status": "weakness_map_missing", "path": str(WEAKNESS_MAP_PATH)}

    payload = json.loads(WEAKNESS_MAP_PATH.read_text(encoding="utf-8"))
    late_washout = next(
        (row for row in payload.get("category_stats", []) if row.get("category") == "late_washout"),
        None,
    )
    return {
        "status": "loaded",
        "path": str(WEAKNESS_MAP_PATH),
        "late_washout": late_washout,
    }


def compute_metrics_extended(trades: list[AdaptiveTrailTradeRecord]) -> dict:
    metrics = compute_metrics(trades)
    metrics["adaptive_trail_active_count"] = sum(1 for t in trades if t.adaptive_trail_active)
    return metrics


def compute_annual_metrics(trades: list[AdaptiveTrailTradeRecord]) -> list[dict]:
    rows = []
    for year in sorted({t.entry_time.year for t in trades}):
        subset = [t for t in trades if t.entry_time.year == year]
        rows.append({"year": year, **compute_metrics_extended(subset)})
    return rows


def _make_trade_record_dict(
    *,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    side: str,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    mfe: float,
    mae: float,
    be_activated: bool,
    be_gate_expired: bool,
    profit_lock_activated: bool,
    bars_in_trade: int,
) -> dict:
    fee_pct = 0.0005
    gross = exit_price - entry_price if side == "long" else entry_price - exit_price
    fees = (entry_price + exit_price) * fee_pct
    pnl = gross - fees
    return {
        "entry_time": entry_ts,
        "exit_time": exit_ts,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "duration_min": float((exit_ts - entry_ts).total_seconds() / 60.0),
        "exit_reason": exit_reason,
        "mfe": mfe,
        "mae": mae,
        "be_activated": be_activated,
        "be_gate_expired": be_gate_expired,
        "profit_lock_activated": profit_lock_activated,
        "bars_in_trade": bars_in_trade,
    }


def simulate_orb_v26_adaptive_trail(
    df: pd.DataFrame,
    params: dict,
    *,
    tighten_delay_bars: int | None,
    tighten_target_pct: float | None,
    tighten_step_bars: int | None,
) -> list[AdaptiveTrailTradeRecord]:
    """Full v26 simulation with optional adaptive trailing tightening."""
    orb_bars = int(params["orb_bars"])
    profit_ratio = float(params["profit_ratio"])
    close_before_min = int(params["close_before_min"])
    breakout_pct = float(params["breakout_confirm_pct"])
    entry_delay_bars = int(params["entry_delay_bars"])
    trailing_pct = float(params["trailing_pct"])
    htf_filter = bool(params["htf_filter"])
    htf_mode = str(params["htf_mode"])
    htf_ema_fast = int(params["htf_ema_fast"])
    htf_ema_slow = int(params["htf_ema_slow"])
    skip_short_after_up_days = int(params["skip_short_after_up_days"])
    skip_long_after_up_days = int(params["skip_long_after_up_days"])
    initial_stop_mult = float(params["initial_stop_mult"])

    use_adaptive_trail = (
        tighten_delay_bars is not None
        and tighten_target_pct is not None
        and tighten_step_bars is not None
        and tighten_target_pct < trailing_pct
        and tighten_step_bars > 0
    )

    htf_bias = compute_htf_bias(df, htf_ema_fast, htf_ema_slow, htf_mode) if htf_filter else None

    trades: list[AdaptiveTrailTradeRecord] = []
    up_day_streak = 0

    for _, session in df.groupby(df.index.date):
        sess = session.between_time("09:30", "16:00")
        if len(sess) < orb_bars + 5:
            continue

        allow_short_today = not (
            skip_short_after_up_days > 0 and up_day_streak >= skip_short_after_up_days
        )
        allow_long_today = not (
            skip_long_after_up_days > 0 and up_day_streak >= skip_long_after_up_days
        )

        orb = sess.iloc[:orb_bars]
        orb_high = float(orb["High"].max())
        orb_low = float(orb["Low"].min())
        range_width = orb_high - orb_low
        if range_width <= 0:
            day_open = float(sess["Open"].iloc[0])
            day_close = float(sess["Close"].iloc[-1])
            up_day_streak = up_day_streak + 1 if day_close > day_open else 0
            continue

        mid_price = (orb_high + orb_low) / 2.0
        if mid_price <= 0 or range_width / mid_price < 0.001:
            day_open = float(sess["Open"].iloc[0])
            day_close = float(sess["Close"].iloc[-1])
            up_day_streak = up_day_streak + 1 if day_close > day_open else 0
            continue

        long_entry_level = orb_high * (1.0 + breakout_pct)
        short_entry_level = orb_low * (1.0 - breakout_pct)
        tp_long = orb_high + profit_ratio * range_width
        tp_short = orb_low - profit_ratio * range_width
        long_initial_stop = orb_low - (initial_stop_mult - 1.0) * range_width
        short_initial_stop = orb_high + (initial_stop_mult - 1.0) * range_width

        be_trigger_pts = BASE_BE_TRIGGER * range_width
        profit_lock_trigger_pts = BASE_V26_PROFIT_LOCK_TRIGGER * range_width
        profit_lock_pts = BASE_V26_PROFIT_LOCK_LEVEL * range_width

        post_orb = sess.iloc[orb_bars + entry_delay_bars:]
        if post_orb.empty:
            day_open = float(sess["Open"].iloc[0])
            day_close = float(sess["Close"].iloc[-1])
            up_day_streak = up_day_streak + 1 if day_close > day_open else 0
            continue

        last_ts = sess.index[-1]
        force_close_ts = last_ts - pd.Timedelta(minutes=close_before_min)

        in_long = False
        in_short = False
        entry_price = 0.0
        entry_ts: pd.Timestamp | None = None
        best_price_long = 0.0
        best_price_short = float("inf")
        worst_price_long = float("inf")
        worst_price_short = 0.0
        bars_in_trade = 0
        be_activated = False
        be_gate_expired = False
        profit_lock_activated = False
        profit_lock_price: float | None = None
        profit_lock_bar: int | None = None  # bar count when profit-lock activated
        adaptive_trail_active = False
        effective_trail_pct = trailing_pct
        session_entry_count = 0

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            # --- EOD flatten ---
            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    trades.append(
                        AdaptiveTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="long",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="eod",
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            adaptive_trail_active=adaptive_trail_active,
                            final_effective_trail_pct=effective_trail_pct,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    trades.append(
                        AdaptiveTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="short",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="eod",
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            adaptive_trail_active=adaptive_trail_active,
                            final_effective_trail_pct=effective_trail_pct,
                        )
                    )
                    in_short = False
                continue

            # --- Entry logic ---
            if not in_long and not in_short:
                if session_entry_count >= MAX_ENTRIES_PER_SESSION:
                    continue

                bias = int(htf_bias.loc[ts]) if htf_bias is not None and ts in htf_bias.index else 0
                if close > long_entry_level:
                    if not allow_long_today:
                        continue
                    if htf_filter and bias == -1:
                        continue
                    in_long = True
                    entry_price = close
                    entry_ts = ts
                    best_price_long = close
                    worst_price_long = close
                    bars_in_trade = 1
                    be_activated = False
                    be_gate_expired = False
                    profit_lock_activated = False
                    profit_lock_price = None
                    profit_lock_bar = None
                    adaptive_trail_active = False
                    effective_trail_pct = trailing_pct
                    session_entry_count += 1
                    continue

                if close < short_entry_level:
                    if not allow_short_today:
                        continue
                    if htf_filter and bias == 1:
                        continue
                    in_short = True
                    entry_price = close
                    entry_ts = ts
                    best_price_short = close
                    worst_price_short = close
                    bars_in_trade = 1
                    be_activated = False
                    be_gate_expired = False
                    profit_lock_activated = False
                    profit_lock_price = None
                    profit_lock_bar = None
                    adaptive_trail_active = False
                    effective_trail_pct = trailing_pct
                    session_entry_count += 1
                    continue

            if entry_ts is None:
                continue

            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            gate_active = elapsed_min <= BASE_BE_GATE_MIN
            if not gate_active and not be_gate_expired:
                be_gate_expired = True

            bars_in_trade += 1

            # --- Long management ---
            if in_long:
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)
                unrealised = close - entry_price

                # BE activation
                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True

                # Profit-lock activation
                if (
                    not profit_lock_activated
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price + profit_lock_pts
                    profit_lock_bar = bars_in_trade

                # Adaptive trail computation
                if use_adaptive_trail and profit_lock_activated and profit_lock_bar is not None:
                    bars_since_lock = bars_in_trade - profit_lock_bar
                    if bars_since_lock >= tighten_delay_bars:
                        bars_in_tighten = bars_since_lock - tighten_delay_bars
                        progress = min(1.0, bars_in_tighten / tighten_step_bars)
                        effective_trail_pct = trailing_pct - progress * (trailing_pct - tighten_target_pct)
                        adaptive_trail_active = True
                    else:
                        effective_trail_pct = trailing_pct
                else:
                    effective_trail_pct = trailing_pct

                # Stop computation
                trail_sl = best_price_long * (1.0 - effective_trail_pct)
                effective_sl = max(long_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = max(effective_sl, entry_price)
                if profit_lock_activated and profit_lock_price is not None:
                    effective_sl = max(effective_sl, profit_lock_price)

                if close <= effective_sl:
                    exit_reason = "stop"
                    if profit_lock_activated and profit_lock_price is not None and close <= profit_lock_price * 1.001:
                        exit_reason = "profit_lock_stop"
                    elif be_activated and gate_active and close <= entry_price * 1.001:
                        exit_reason = "be_stop"
                    elif adaptive_trail_active:
                        exit_reason = "adaptive_trail_stop"
                    trades.append(
                        AdaptiveTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="long",
                                entry_price=entry_price, exit_price=close,
                                exit_reason=exit_reason,
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            adaptive_trail_active=adaptive_trail_active,
                            final_effective_trail_pct=effective_trail_pct,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

                if close >= tp_long:
                    trades.append(
                        AdaptiveTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="long",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="target",
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            adaptive_trail_active=adaptive_trail_active,
                            final_effective_trail_pct=effective_trail_pct,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            # --- Short management ---
            if in_short:
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)
                unrealised = entry_price - close

                # BE activation
                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True

                # Profit-lock activation
                if (
                    not profit_lock_activated
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price - profit_lock_pts
                    profit_lock_bar = bars_in_trade

                # Adaptive trail computation
                if use_adaptive_trail and profit_lock_activated and profit_lock_bar is not None:
                    bars_since_lock = bars_in_trade - profit_lock_bar
                    if bars_since_lock >= tighten_delay_bars:
                        bars_in_tighten = bars_since_lock - tighten_delay_bars
                        progress = min(1.0, bars_in_tighten / tighten_step_bars)
                        effective_trail_pct = trailing_pct - progress * (trailing_pct - tighten_target_pct)
                        adaptive_trail_active = True
                    else:
                        effective_trail_pct = trailing_pct
                else:
                    effective_trail_pct = trailing_pct

                # Stop computation
                trail_sl = best_price_short * (1.0 + effective_trail_pct)
                effective_sl = min(short_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = min(effective_sl, entry_price)
                if profit_lock_activated and profit_lock_price is not None:
                    effective_sl = min(effective_sl, profit_lock_price)

                if close >= effective_sl:
                    exit_reason = "stop"
                    if profit_lock_activated and profit_lock_price is not None and close >= profit_lock_price * 0.999:
                        exit_reason = "profit_lock_stop"
                    elif be_activated and gate_active and close >= entry_price * 0.999:
                        exit_reason = "be_stop"
                    elif adaptive_trail_active:
                        exit_reason = "adaptive_trail_stop"
                    trades.append(
                        AdaptiveTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="short",
                                entry_price=entry_price, exit_price=close,
                                exit_reason=exit_reason,
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            adaptive_trail_active=adaptive_trail_active,
                            final_effective_trail_pct=effective_trail_pct,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

                if close <= tp_short:
                    trades.append(
                        AdaptiveTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="short",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="target",
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            adaptive_trail_active=adaptive_trail_active,
                            final_effective_trail_pct=effective_trail_pct,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


# ---------- candidate selection ----------

def _variant_sort_key(row: dict) -> tuple:
    metrics = row["metrics"]
    path = row["path_impact"]
    return (
        row["improved_time_folds"],
        row["improved_years"],
        row["pnl_delta"],
        -path["clipped_winners"],
        metrics["profit_factor"],
    )


def summarize_candidate(row: dict) -> dict:
    path = row["path_impact"]
    metrics = row["metrics"]
    return {
        "label": row["label"],
        "tighten_delay_bars": row["tighten_delay_bars"],
        "tighten_target_pct": row["tighten_target_pct"],
        "tighten_step_bars": row["tighten_step_bars"],
        "pnl": metrics["total_pnl"],
        "pnl_delta": row["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_time_folds": row["improved_time_folds"],
        "positive_time_folds": row["positive_time_folds"],
        "improved_years": row["improved_years"],
        "positive_years": row["positive_years"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "adaptive_trail_active_count": metrics["adaptive_trail_active_count"],
        "adaptive_trail_stop_exits": sum(
            1 for t_label in ["adaptive_trail_stop"]
            if metrics.get(t_label, 0) > 0
        ),
    }


def choose_candidate_summary(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["pnl_delta"] > 0]
    zero_clip = [row for row in positive if row["path_impact"]["clipped_winners"] == 0]
    balanced = [
        row
        for row in positive
        if row["improved_time_folds"] >= 2 and row["improved_years"] >= 3
    ]

    best_positive = max(positive, key=_variant_sort_key, default=None)
    best_zero_clip = max(zero_clip, key=_variant_sort_key, default=None)
    best_balanced = max(balanced, key=_variant_sort_key, default=None)

    strongest = best_balanced or best_zero_clip or best_positive
    if strongest is None:
        verdict = "ALPACA_LOCAL_REJECTED"
    elif strongest["improved_time_folds"] >= 3 and strongest["improved_years"] >= 4:
        verdict = "READY_FOR_QC_PROXY"
    elif strongest["improved_time_folds"] >= 2 and strongest["improved_years"] >= 3:
        verdict = "ALPACA_LOCAL_STRONG_NEAR_MISS"
    else:
        verdict = "ALPACA_LOCAL_NEAR_MISS"

    return {
        "best_positive": summarize_candidate(best_positive) if best_positive else None,
        "best_zero_clip_positive": summarize_candidate(best_zero_clip) if best_zero_clip else None,
        "best_balanced": summarize_candidate(best_balanced) if best_balanced else None,
        "verdict": verdict,
    }


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data_path = Path(args.intraday_csv)
    output_path = Path(args.output_path)

    print("Loading local data for v26 adaptive trail tightening research...")
    intraday = load_csv_5m(data_path)
    weakness_reference = load_weakness_reference()

    # --- Baseline (v26 without adaptive trail) ---
    baseline_trades = simulate_orb_v25_profit_lock(
        intraday,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER,
        profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL,
    )
    baseline_extended = [
        AdaptiveTrailTradeRecord(
            **t.__dict__,
            adaptive_trail_active=False,
            final_effective_trail_pct=float(BASE_PARAMS["trailing_pct"]),
        )
        for t in baseline_trades
    ]
    baseline_metrics = compute_metrics_extended(baseline_extended)
    baseline_years = compute_annual_metrics(baseline_extended)
    baseline_year_map = {row["year"]: row for row in baseline_years}
    time_folds = build_time_folds(intraday, fold_count=4)
    baseline_folds = compute_fold_metrics(baseline_extended, time_folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}

    print(
        f"Alpaca baseline v26: trades={baseline_metrics['trades']}, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}, "
        f"positiveYears={sum(1 for row in baseline_years if row['total_pnl'] > 0)}, "
        f"positiveFolds={sum(1 for row in baseline_folds if row['total_pnl'] > 0)}"
    )

    # --- Grid sweep ---
    all_results = []
    for delay in TIGHTEN_DELAY_BARS:
        for target_pct in TIGHTEN_TARGET_PCTS:
            for step_bars in TIGHTEN_STEP_BARS:
                label = (
                    f"adaptive_trail_delay{delay}_target{target_pct:.3f}_step{step_bars}"
                )
                print(f"Running {label}...")
                variant_trades = simulate_orb_v26_adaptive_trail(
                    intraday,
                    BASE_PARAMS,
                    tighten_delay_bars=delay,
                    tighten_target_pct=target_pct,
                    tighten_step_bars=step_bars,
                )
                variant_metrics = compute_metrics_extended(variant_trades)
                variant_years = compute_annual_metrics(variant_trades)
                variant_year_map = {row["year"]: row for row in variant_years}
                variant_folds = compute_fold_metrics(variant_trades, time_folds)
                path_impact = analyze_path_impact(baseline_trades, variant_trades)

                improved_time_folds = sum(
                    1
                    for row in variant_folds
                    if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
                )
                positive_time_folds = sum(1 for row in variant_folds if row["total_pnl"] > 0)
                improved_years = sum(
                    1
                    for year, row in variant_year_map.items()
                    if row["total_pnl"] > baseline_year_map.get(year, {"total_pnl": float("-inf")})["total_pnl"]
                )
                positive_years = sum(1 for row in variant_years if row["total_pnl"] > 0)
                pnl_delta = round(variant_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)

                adaptive_trail_stop_exits = sum(
                    1 for t in variant_trades if t.exit_reason == "adaptive_trail_stop"
                )

                result = {
                    "label": label,
                    "tighten_delay_bars": delay,
                    "tighten_target_pct": target_pct,
                    "tighten_step_bars": step_bars,
                    "metrics": variant_metrics,
                    "annual_metrics": variant_years,
                    "time_folds": variant_folds,
                    "pnl_delta": pnl_delta,
                    "improved_time_folds": improved_time_folds,
                    "positive_time_folds": positive_time_folds,
                    "improved_years": improved_years,
                    "positive_years": positive_years,
                    "path_impact": path_impact,
                    "adaptive_trail_stop_exits": adaptive_trail_stop_exits,
                }
                all_results.append(result)
                print(
                    f"  delta={pnl_delta:+.2f}, "
                    f"folds={improved_time_folds}/{len(time_folds)}, "
                    f"years={improved_years}/{len(baseline_years)}, "
                    f"adaptiveStops={adaptive_trail_stop_exits}, "
                    f"adaptiveActive={variant_metrics['adaptive_trail_active_count']}, "
                    f"netPath={path_impact['net_impact']:+.2f}"
                )

    candidate_summary = choose_candidate_summary(all_results)

    # --- Sort results ---
    all_results.sort(key=_variant_sort_key, reverse=True)

    payload = {
        "research_scope": "local_orb_v26_adaptive_trail_alpaca",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v26-profit-lock",
        "local_data_policy": "alpaca_reference_lane_only_qc_promotion_unchanged",
        "data": {
            "source": data_path.name,
            "path": str(data_path),
            "bars": int(len(intraday)),
            "start": str(intraday.index.min()),
            "end": str(intraday.index.max()),
        },
        "weakness_reference": weakness_reference,
        "method": (
            "Alpaca-backed local feasibility test for adaptive trailing tightening after "
            "profit-lock activation on top of v26. Once profit-lock is confirmed active "
            "(MFE >= 1.50x ORB range), the 1.3% trailing stop is progressively narrowed "
            "toward a tighter target percentage over a configurable number of bars, after "
            "an optional delay."
        ),
        "mechanism": {
            "trigger": "profit-lock activation (MFE >= 1.50x ORB range)",
            "delay": "tighten_delay_bars after profit-lock",
            "transition": "linear interpolation from 1.3% -> tighten_target_pct over tighten_step_bars",
            "formula": "effective_trail = 0.013 - progress * (0.013 - target_pct), progress = min(1, bars_since_delay / step_bars)",
        },
        "baseline": {
            "label": "v26_BE=1.25x_gate=180min_profitLock=0.25x_after_1.50x",
            "metrics": baseline_metrics,
            "annual_metrics": baseline_years,
            "time_folds": baseline_folds,
        },
        "grid": {
            "tighten_delay_bars": TIGHTEN_DELAY_BARS,
            "tighten_target_pcts": TIGHTEN_TARGET_PCTS,
            "tighten_step_bars": TIGHTEN_STEP_BARS,
            "total_variants": len(all_results),
        },
        "all_variants": all_results,
        "candidate_summary": candidate_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(all_results)} variants to {output_path}")
    print(f"Verdict: {candidate_summary['verdict']}")
    if candidate_summary["best_positive"]:
        bp = candidate_summary["best_positive"]
        print(
            f"Best positive: {bp['label']} — "
            f"delta={bp['pnl_delta']:+.2f}, "
            f"folds={bp['improved_time_folds']}/{len(time_folds)}, "
            f"years={bp['improved_years']}/{len(baseline_years)}, "
            f"clipped={bp['clipped_winners']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
