#!/usr/bin/env python3
"""
Alpaca-backed local research for a structural ORB pullback-reclaim entry on top of v26 exits.

Why this exists:
- v26-profit-lock is the official baseline.
- Post-v26 mechanism layering is largely exhausted.
- The next valid direction must be structurally different, not another exit-side patch.
- This branch changes the entry architecture itself while staying ORB-only:
  breakout first, then pullback toward the ORB boundary, then reclaim entry.

Mechanism tested here:
- Replace immediate breakout entry with a delayed structural entry:
  - wait for initial breakout beyond the ORB boundary,
  - optionally require breakout progress of `arm_progress x ORB range`,
  - wait for a pullback back toward / into the ORB boundary by `retest_depth x ORB range`,
  - enter only after `confirm_bars` closes reclaim the breakout boundary again.
- Keep the full v26 baseline exits unchanged:
  - BE trigger = 1.25 x ORB range during first 180 minutes
  - persistent profit lock = +0.25 x ORB range after 1.50 x ORB range MFE
  - fixed trailing stop remains unchanged
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
OUTPUT_PATH = RESULTS_DIR / "local_orb_v26_pullback_reclaim_entry_alpaca.json"
ANALYSIS_VERSION = "v1_alpaca_v26_pullback_reclaim_entry"

BASE_V26_PROFIT_LOCK_TRIGGER = 1.50
BASE_V26_PROFIT_LOCK_LEVEL = 0.25

ARM_PROGRESS_MULTS = [0.00, 0.25, 0.50]
RETEST_DEPTH_MULTS = [0.00, 0.10, 0.25]
RECLAIM_CONFIRM_BARS = [1, 2]


@dataclass
class ExtendedTradeRecord:
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
    pullback_reclaim_entry: bool


def compute_metrics_extended(trades: list[ExtendedTradeRecord]) -> dict:
    metrics = compute_metrics(trades)
    metrics["pullback_reclaim_entry_count"] = sum(1 for trade in trades if trade.pullback_reclaim_entry)
    return metrics


def compute_annual_metrics(trades: list[ExtendedTradeRecord]) -> list[dict]:
    rows = []
    for year in sorted({trade.entry_time.year for trade in trades}):
        subset = [trade for trade in trades if trade.entry_time.year == year]
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


def simulate_orb_v26_pullback_reclaim_entry(
    df: pd.DataFrame,
    params: dict,
    *,
    arm_progress_mult: float,
    retest_depth_mult: float,
    confirm_bars: int,
) -> list[ExtendedTradeRecord]:
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

    htf_bias = compute_htf_bias(df, htf_ema_fast, htf_ema_slow, htf_mode) if htf_filter else None

    trades: list[ExtendedTradeRecord] = []
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

        long_breakout_level = orb_high * (1.0 + breakout_pct)
        short_breakout_level = orb_low * (1.0 - breakout_pct)
        long_retest_level = orb_high - retest_depth_mult * range_width
        short_retest_level = orb_low + retest_depth_mult * range_width

        tp_long = orb_high + profit_ratio * range_width
        tp_short = orb_low - profit_ratio * range_width
        long_initial_stop = orb_low - (initial_stop_mult - 1.0) * range_width
        short_initial_stop = orb_high + (initial_stop_mult - 1.0) * range_width

        be_trigger_pts = BASE_BE_TRIGGER * range_width
        profit_lock_trigger_pts = BASE_V26_PROFIT_LOCK_TRIGGER * range_width
        profit_lock_pts = BASE_V26_PROFIT_LOCK_LEVEL * range_width
        arm_progress_pts = arm_progress_mult * range_width

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
        session_entry_count = 0

        long_breakout_seen = False
        long_breakout_peak = 0.0
        long_retest_seen = False
        long_reclaim_count = 0

        short_breakout_seen = False
        short_breakout_trough = float("inf")
        short_retest_seen = False
        short_reclaim_count = 0

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    trades.append(
                        ExtendedTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts,
                                exit_ts=ts,
                                side="long",
                                entry_price=entry_price,
                                exit_price=close,
                                exit_reason="eod",
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            pullback_reclaim_entry=True,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    trades.append(
                        ExtendedTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts,
                                exit_ts=ts,
                                side="short",
                                entry_price=entry_price,
                                exit_price=close,
                                exit_reason="eod",
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            pullback_reclaim_entry=True,
                        )
                    )
                    in_short = False
                continue

            if not in_long and not in_short:
                if session_entry_count >= MAX_ENTRIES_PER_SESSION:
                    continue

                bias = int(htf_bias.loc[ts]) if htf_bias is not None and ts in htf_bias.index else 0

                if allow_long_today and not (htf_filter and bias == -1):
                    if close > long_breakout_level:
                        long_breakout_seen = True
                        long_breakout_peak = max(long_breakout_peak, close)
                    if long_breakout_seen and (long_breakout_peak - orb_high) >= arm_progress_pts:
                        if close <= long_retest_level:
                            long_retest_seen = True
                            long_reclaim_count = 0
                        elif long_retest_seen and close > long_breakout_level:
                            long_reclaim_count += 1
                        elif long_retest_seen:
                            long_reclaim_count = 0

                    if long_retest_seen and long_reclaim_count >= confirm_bars:
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
                        session_entry_count += 1
                        continue

                if allow_short_today and not (htf_filter and bias == 1):
                    if close < short_breakout_level:
                        short_breakout_seen = True
                        short_breakout_trough = min(short_breakout_trough, close)
                    if short_breakout_seen and (orb_low - short_breakout_trough) >= arm_progress_pts:
                        if close >= short_retest_level:
                            short_retest_seen = True
                            short_reclaim_count = 0
                        elif short_retest_seen and close < short_breakout_level:
                            short_reclaim_count += 1
                        elif short_retest_seen:
                            short_reclaim_count = 0

                    if short_retest_seen and short_reclaim_count >= confirm_bars:
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
                        session_entry_count += 1
                        continue

            if entry_ts is None:
                continue

            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            gate_active = elapsed_min <= BASE_BE_GATE_MIN
            if not gate_active and not be_gate_expired:
                be_gate_expired = True

            bars_in_trade += 1

            if in_long:
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)
                unrealised = close - entry_price

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if (
                    not profit_lock_activated
                    and gate_active
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price + profit_lock_pts

                trail_sl = best_price_long * (1.0 - trailing_pct)
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
                    trades.append(
                        ExtendedTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts,
                                exit_ts=ts,
                                side="long",
                                entry_price=entry_price,
                                exit_price=close,
                                exit_reason=exit_reason,
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            pullback_reclaim_entry=True,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

                if close >= tp_long:
                    trades.append(
                        ExtendedTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts,
                                exit_ts=ts,
                                side="long",
                                entry_price=entry_price,
                                exit_price=close,
                                exit_reason="target",
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            pullback_reclaim_entry=True,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            if in_short:
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)
                unrealised = entry_price - close

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if (
                    not profit_lock_activated
                    and gate_active
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price - profit_lock_pts

                trail_sl = best_price_short * (1.0 + trailing_pct)
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
                    trades.append(
                        ExtendedTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts,
                                exit_ts=ts,
                                side="short",
                                entry_price=entry_price,
                                exit_price=close,
                                exit_reason=exit_reason,
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            pullback_reclaim_entry=True,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

                if close <= tp_short:
                    trades.append(
                        ExtendedTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts,
                                exit_ts=ts,
                                side="short",
                                entry_price=entry_price,
                                exit_price=close,
                                exit_reason="target",
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            pullback_reclaim_entry=True,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def _variant_sort_key(row: dict) -> tuple:
    metrics = row["metrics"]
    return (
        row["improved_time_folds"],
        row["improved_years"],
        row["pnl_delta"],
        metrics["profit_factor"],
        -abs(row["trade_count_delta"]),
    )


def summarize_candidate(row: dict) -> dict:
    metrics = row["metrics"]
    path = row["path_impact"]
    return {
        "label": row["label"],
        "arm_progress_mult": row["arm_progress_mult"],
        "retest_depth_mult": row["retest_depth_mult"],
        "confirm_bars": row["confirm_bars"],
        "trades": metrics["trades"],
        "trade_count_delta": row["trade_count_delta"],
        "pnl": metrics["total_pnl"],
        "pnl_delta": row["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_time_folds": row["improved_time_folds"],
        "positive_time_folds": row["positive_time_folds"],
        "improved_years": row["improved_years"],
        "positive_years": row["positive_years"],
        "common_trades": path["common_trades"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "pullback_reclaim_entry_count": metrics["pullback_reclaim_entry_count"],
    }


def choose_candidate_summary(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["pnl_delta"] > 0]
    balanced = [
        row
        for row in positive
        if row["improved_time_folds"] >= 2 and row["improved_years"] >= 3
    ]
    strongest = max(balanced or positive, key=_variant_sort_key, default=None)
    best_positive = max(positive, key=_variant_sort_key, default=None)
    best_balanced = max(balanced, key=_variant_sort_key, default=None)

    if strongest is None:
        verdict = "ALPACA_LOCAL_REJECTED"
    elif strongest["improved_time_folds"] >= 3 and strongest["improved_years"] >= 4:
        verdict = "READY_FOR_BLUESHIFT_EVALUATOR"
    else:
        verdict = "ALPACA_LOCAL_NEAR_MISS"

    return {
        "best_positive": summarize_candidate(best_positive) if best_positive else None,
        "best_balanced": summarize_candidate(best_balanced) if best_balanced else None,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data_path = Path(args.intraday_csv)
    output_path = Path(args.output_path)

    print("Loading Alpaca data for v26 pullback-reclaim entry research...")
    intraday = load_csv_5m(data_path)

    baseline_trades = simulate_orb_v25_profit_lock(
        intraday,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER,
        profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL,
    )
    baseline_extended = [
        ExtendedTradeRecord(**trade.__dict__, pullback_reclaim_entry=False)
        for trade in baseline_trades
    ]
    baseline_metrics = compute_metrics_extended(baseline_extended)
    baseline_years = compute_annual_metrics(baseline_extended)
    baseline_year_map = {row["year"]: row for row in baseline_years}
    time_folds = build_time_folds(intraday, fold_count=4)
    baseline_folds = compute_fold_metrics(baseline_extended, time_folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}

    print(
        f"Alpaca baseline v26: trades={baseline_metrics['trades']}, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, PF={baseline_metrics['profit_factor']:.3f}, "
        f"positiveYears={sum(1 for row in baseline_years if row['total_pnl'] > 0)}, "
        f"positiveFolds={sum(1 for row in baseline_folds if row['total_pnl'] > 0)}"
    )

    all_results = []
    for arm_progress in ARM_PROGRESS_MULTS:
        for retest_depth in RETEST_DEPTH_MULTS:
            for confirm_bars in RECLAIM_CONFIRM_BARS:
                label = (
                    f"pullback_reclaim_after_{arm_progress:.3f}x_"
                    f"depth_{retest_depth:.2f}x_confirm_{confirm_bars}bar"
                )
                print(f"Running {label}...")
                variant_trades = simulate_orb_v26_pullback_reclaim_entry(
                    intraday,
                    BASE_PARAMS,
                    arm_progress_mult=arm_progress,
                    retest_depth_mult=retest_depth,
                    confirm_bars=confirm_bars,
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
                trade_count_delta = int(variant_metrics["trades"] - baseline_metrics["trades"])

                result = {
                    "label": label,
                    "arm_progress_mult": arm_progress,
                    "retest_depth_mult": retest_depth,
                    "confirm_bars": confirm_bars,
                    "metrics": variant_metrics,
                    "annual_metrics": variant_years,
                    "time_folds": variant_folds,
                    "pnl_delta": pnl_delta,
                    "trade_count_delta": trade_count_delta,
                    "improved_time_folds": improved_time_folds,
                    "positive_time_folds": positive_time_folds,
                    "improved_years": improved_years,
                    "positive_years": positive_years,
                    "path_impact": path_impact,
                }
                all_results.append(result)
                print(
                    f"  trades={variant_metrics['trades']}, delta={pnl_delta:+.2f}, "
                    f"folds={improved_time_folds}/{len(time_folds)}, "
                    f"years={improved_years}/{len(baseline_years)}, "
                    f"common={path_impact['common_trades']}"
                )

    candidate_summary = choose_candidate_summary(all_results)
    payload = {
        "research_scope": "local_orb_v26_pullback_reclaim_entry_alpaca",
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
        "method": (
            "Alpaca-backed local feasibility test for a structural ORB pullback-reclaim entry on top of v26 exits. "
            "Instead of entering on the first breakout, the branch waits for breakout, pullback toward the ORB boundary, "
            "and reclaim confirmation before entering."
        ),
        "baseline": {
            "label": "v26_immediate_breakout_entry_with_profit_lock",
            "metrics": baseline_metrics,
            "annual_metrics": baseline_years,
            "time_folds": baseline_folds,
        },
        "grid": {
            "arm_progress_mults": ARM_PROGRESS_MULTS,
            "retest_depth_mults": RETEST_DEPTH_MULTS,
            "confirm_bars_options": RECLAIM_CONFIRM_BARS,
        },
        "all_variants": all_results,
        "candidate_summary": candidate_summary,
        "structural_conclusion": {
            "research_interpretation": (
                "This branch changes the ORB trade architecture itself by delaying entry until a breakout proves itself, "
                "pulls back, and reclaims the ORB boundary. Because entry timestamps differ, path-impact metrics based on "
                "common entry times are only secondary diagnostics here."
            ),
            "next_step_rule": (
                "Only build a Blueshift evaluator if the strongest variant improves full-sample Alpaca PnL and also "
                "shows at least 2/4 improved time folds and 3+ improved years."
            ),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved results to {output_path}")
    print(json.dumps(candidate_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
