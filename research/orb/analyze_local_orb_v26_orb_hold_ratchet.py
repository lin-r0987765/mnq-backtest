#!/usr/bin/env python3
"""
Alpaca-backed local research for an ORB hold ratchet on top of v26.

Why this exists:
- v26-profit-lock is the official baseline.
- The most underexplored weakness after v26 remains the hard-stop-loss bucket:
  trades that made moderate progress but still reverted into a stop-based loss.
- ORB re-entry itself was locally strong but QC-mixed; the first refinement that
  gated it behind profit lock collapsed to zero triggers.

Mechanism tested here:
- Keep the full v26 baseline unchanged:
  - BE trigger = 1.25 x ORB range during first 180 minutes
  - persistent profit lock = +0.25 x ORB range after 1.50 x ORB range MFE
- Add a new optional structural ratchet:
  - once the trade has achieved `arm_progress x ORB range` MFE,
  - promote a structural stop floor near the ORB breakout hold zone,
  - the floor is `orb_high - floor_offset x ORB range` for longs
    and `orb_low + floor_offset x ORB range` for shorts,
  - once armed for `min_bars_after_arm`, a close through that floor exits immediately.
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
OUTPUT_PATH = RESULTS_DIR / "local_orb_v26_orb_hold_ratchet_alpaca.json"
WEAKNESS_MAP_PATH = RESULTS_DIR / "v26_weakness_map.json"
PRIOR_ORB_REENTRY_PATH = RESULTS_DIR / "local_orb_v26_orb_reentry_exit_alpaca.json"
ANALYSIS_VERSION = "v1_alpaca_v26_orb_hold_ratchet"

BASE_V26_PROFIT_LOCK_TRIGGER = 1.50
BASE_V26_PROFIT_LOCK_LEVEL = 0.25

ARM_PROGRESS_MULTS = [0.50, 0.625, 0.75, 1.00]
FLOOR_OFFSET_MULTS = [0.00, 0.10, 0.25]
MIN_BARS_AFTER_ARM_OPTIONS = [0, 1, 2]


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
    orb_hold_ratchet_armed: bool
    orb_hold_ratchet_exit: bool


def load_weakness_reference() -> dict:
    if not WEAKNESS_MAP_PATH.exists():
        return {
            "status": "weakness_map_missing",
            "path": str(WEAKNESS_MAP_PATH),
        }

    payload = json.loads(WEAKNESS_MAP_PATH.read_text(encoding="utf-8"))
    hard_stop = next(
        (row for row in payload.get("category_stats", []) if row.get("category") == "hard_stop_loss"),
        None,
    )
    return {
        "status": "loaded",
        "path": str(WEAKNESS_MAP_PATH),
        "hard_stop_loss": hard_stop,
    }


def load_prior_orb_reentry_reference() -> dict:
    if not PRIOR_ORB_REENTRY_PATH.exists():
        return {"status": "missing", "path": str(PRIOR_ORB_REENTRY_PATH)}
    data = json.loads(PRIOR_ORB_REENTRY_PATH.read_text(encoding="utf-8"))
    return {
        "status": "loaded",
        "path": str(PRIOR_ORB_REENTRY_PATH),
        "candidate_summary": data.get("candidate_summary", {}),
    }


def classify_equivalence(candidate_summary: dict, prior_reference: dict) -> dict:
    current = (candidate_summary or {}).get("best_positive")
    prior_candidate_summary = (prior_reference or {}).get("candidate_summary", {})
    prior_best = (prior_candidate_summary or {}).get("best_positive")

    if not current or not prior_best:
        return {
            "status": "insufficient_reference",
            "equivalent_to_prior_orb_reentry": False,
        }

    comparison_keys = [
        "improved_time_folds",
        "improved_years",
        "saved_losses",
        "clipped_winners",
    ]
    exact_match = all(current.get(key) == prior_best.get(key) for key in comparison_keys)
    pnl_match = abs(float(current.get("pnl_delta", 0.0)) - float(prior_best.get("pnl_delta", 0.0))) <= 1e-6
    equivalent = exact_match and pnl_match

    result = {
        "status": "compared",
        "equivalent_to_prior_orb_reentry": equivalent,
        "current_label": current.get("label"),
        "prior_label": prior_best.get("label"),
        "matched_fields": {
            "pnl_delta": pnl_match,
            **{key: current.get(key) == prior_best.get(key) for key in comparison_keys},
        },
    }
    if equivalent:
        result["equivalence_reason"] = (
            "The strongest ORB hold ratchet variant reproduces the prior ORB re-entry branch on the decision metrics "
            "(pnl delta, improved folds, improved years, saved losses, clipped winners), so this should not be treated "
            "as a genuinely new post-v26 mechanism."
        )
    return result


def compute_metrics_extended(trades: list[ExtendedTradeRecord]) -> dict:
    metrics = compute_metrics(trades)
    metrics["orb_hold_ratchet_exit_count"] = sum(1 for trade in trades if trade.exit_reason == "orb_hold_ratchet_exit")
    metrics["orb_hold_ratchet_armed_count"] = sum(1 for trade in trades if trade.orb_hold_ratchet_armed)
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


def simulate_orb_v26_orb_hold_ratchet(
    df: pd.DataFrame,
    params: dict,
    *,
    arm_progress_mult: float,
    floor_offset_mult: float,
    min_bars_after_arm: int,
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

        long_entry_level = orb_high * (1.0 + breakout_pct)
        short_entry_level = orb_low * (1.0 - breakout_pct)
        tp_long = orb_high + profit_ratio * range_width
        tp_short = orb_low - profit_ratio * range_width
        long_initial_stop = orb_low - (initial_stop_mult - 1.0) * range_width
        short_initial_stop = orb_high + (initial_stop_mult - 1.0) * range_width

        be_trigger_pts = BASE_BE_TRIGGER * range_width
        profit_lock_trigger_pts = BASE_V26_PROFIT_LOCK_TRIGGER * range_width
        profit_lock_pts = BASE_V26_PROFIT_LOCK_LEVEL * range_width
        arm_progress_pts = arm_progress_mult * range_width
        floor_offset_pts = floor_offset_mult * range_width

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
        orb_hold_ratchet_armed = False
        bars_since_hold_arm = 0
        session_entry_count = 0

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
                            orb_hold_ratchet_armed=orb_hold_ratchet_armed,
                            orb_hold_ratchet_exit=False,
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
                            orb_hold_ratchet_armed=orb_hold_ratchet_armed,
                            orb_hold_ratchet_exit=False,
                        )
                    )
                    in_short = False
                continue

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
                    orb_hold_ratchet_armed = False
                    bars_since_hold_arm = 0
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
                    orb_hold_ratchet_armed = False
                    bars_since_hold_arm = 0
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

                if (not orb_hold_ratchet_armed) and (best_price_long - entry_price) >= arm_progress_pts:
                    orb_hold_ratchet_armed = True
                    bars_since_hold_arm = 0
                elif orb_hold_ratchet_armed:
                    bars_since_hold_arm += 1

                trail_sl = best_price_long * (1.0 - trailing_pct)
                effective_sl = max(long_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = max(effective_sl, entry_price)
                if profit_lock_activated and profit_lock_price is not None:
                    effective_sl = max(effective_sl, profit_lock_price)
                if orb_hold_ratchet_armed and bars_since_hold_arm >= min_bars_after_arm:
                    hold_floor = orb_high - floor_offset_pts
                    effective_sl = max(effective_sl, hold_floor)

                if close <= effective_sl:
                    exit_reason = "stop"
                    if profit_lock_activated and profit_lock_price is not None and close <= profit_lock_price * 1.001:
                        exit_reason = "profit_lock_stop"
                    elif be_activated and gate_active and close <= entry_price * 1.001:
                        exit_reason = "be_stop"
                    elif orb_hold_ratchet_armed and bars_since_hold_arm >= min_bars_after_arm and close <= (orb_high - floor_offset_pts):
                        exit_reason = "orb_hold_ratchet_exit"
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
                            orb_hold_ratchet_armed=orb_hold_ratchet_armed,
                            orb_hold_ratchet_exit=exit_reason == "orb_hold_ratchet_exit",
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
                            orb_hold_ratchet_armed=orb_hold_ratchet_armed,
                            orb_hold_ratchet_exit=False,
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

                if (not orb_hold_ratchet_armed) and (entry_price - best_price_short) >= arm_progress_pts:
                    orb_hold_ratchet_armed = True
                    bars_since_hold_arm = 0
                elif orb_hold_ratchet_armed:
                    bars_since_hold_arm += 1

                trail_sl = best_price_short * (1.0 + trailing_pct)
                effective_sl = min(short_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = min(effective_sl, entry_price)
                if profit_lock_activated and profit_lock_price is not None:
                    effective_sl = min(effective_sl, profit_lock_price)
                if orb_hold_ratchet_armed and bars_since_hold_arm >= min_bars_after_arm:
                    hold_floor = orb_low + floor_offset_pts
                    effective_sl = min(effective_sl, hold_floor)

                if close >= effective_sl:
                    exit_reason = "stop"
                    if profit_lock_activated and profit_lock_price is not None and close >= profit_lock_price * 0.999:
                        exit_reason = "profit_lock_stop"
                    elif be_activated and gate_active and close >= entry_price * 0.999:
                        exit_reason = "be_stop"
                    elif orb_hold_ratchet_armed and bars_since_hold_arm >= min_bars_after_arm and close >= (orb_low + floor_offset_pts):
                        exit_reason = "orb_hold_ratchet_exit"
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
                            orb_hold_ratchet_armed=orb_hold_ratchet_armed,
                            orb_hold_ratchet_exit=exit_reason == "orb_hold_ratchet_exit",
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
                            orb_hold_ratchet_armed=orb_hold_ratchet_armed,
                            orb_hold_ratchet_exit=False,
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
        "arm_progress_mult": row["arm_progress_mult"],
        "floor_offset_mult": row["floor_offset_mult"],
        "min_bars_after_arm": row["min_bars_after_arm"],
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
        "orb_hold_ratchet_exit_count": metrics["orb_hold_ratchet_exit_count"],
        "orb_hold_ratchet_armed_count": metrics["orb_hold_ratchet_armed_count"],
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
        verdict = "READY_FOR_BLUESHIFT_EVALUATOR"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data_path = Path(args.intraday_csv)
    output_path = Path(args.output_path)

    print("Loading Alpaca data for v26 ORB hold ratchet research...")
    intraday = load_csv_5m(data_path)
    weakness_reference = load_weakness_reference()

    baseline_trades = simulate_orb_v25_profit_lock(
        intraday,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER,
        profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL,
    )
    baseline_extended = [
        ExtendedTradeRecord(**trade.__dict__, orb_hold_ratchet_armed=False, orb_hold_ratchet_exit=False)
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
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}, "
        f"positiveYears={sum(1 for row in baseline_years if row['total_pnl'] > 0)}, "
        f"positiveFolds={sum(1 for row in baseline_folds if row['total_pnl'] > 0)}"
    )

    all_results = []
    for arm_progress in ARM_PROGRESS_MULTS:
        for floor_offset in FLOOR_OFFSET_MULTS:
            for min_bars_after_arm in MIN_BARS_AFTER_ARM_OPTIONS:
                label = (
                    f"orb_hold_ratchet_after_{arm_progress:.3f}x_"
                    f"floorOffset_{floor_offset:.2f}x_after_{min_bars_after_arm}bar"
                )
                print(f"Running {label}...")
                variant_trades = simulate_orb_v26_orb_hold_ratchet(
                    intraday,
                    BASE_PARAMS,
                    arm_progress_mult=arm_progress,
                    floor_offset_mult=floor_offset,
                    min_bars_after_arm=min_bars_after_arm,
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

                result = {
                    "label": label,
                    "arm_progress_mult": arm_progress,
                    "floor_offset_mult": floor_offset,
                    "min_bars_after_arm": min_bars_after_arm,
                    "metrics": variant_metrics,
                    "annual_metrics": variant_years,
                    "time_folds": variant_folds,
                    "pnl_delta": pnl_delta,
                    "improved_time_folds": improved_time_folds,
                    "positive_time_folds": positive_time_folds,
                    "improved_years": improved_years,
                    "positive_years": positive_years,
                    "path_impact": path_impact,
                }
                all_results.append(result)
                print(
                    f"  delta={pnl_delta:+.2f}, "
                    f"folds={improved_time_folds}/{len(time_folds)}, "
                    f"years={improved_years}/{len(baseline_years)}, "
                    f"holdExits={variant_metrics['orb_hold_ratchet_exit_count']}, "
                    f"netPath={path_impact['net_impact']:+.2f}"
                )

    candidate_summary = choose_candidate_summary(all_results)
    prior_orb_reentry_reference = load_prior_orb_reentry_reference()
    equivalence = classify_equivalence(candidate_summary, prior_orb_reentry_reference)

    if equivalence.get("equivalent_to_prior_orb_reentry"):
        candidate_summary["verdict"] = "ALPACA_EQUIVALENT_TO_PRIOR_ORB_REENTRY"
        if candidate_summary.get("best_positive"):
            candidate_summary["best_positive"]["equivalent_to_prior_orb_reentry"] = True
        if candidate_summary.get("best_zero_clip_positive"):
            candidate_summary["best_zero_clip_positive"]["equivalent_to_prior_orb_reentry"] = True
        if candidate_summary.get("best_balanced"):
            candidate_summary["best_balanced"]["equivalent_to_prior_orb_reentry"] = True

    payload = {
        "research_scope": "local_orb_v26_orb_hold_ratchet_alpaca",
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
        "prior_orb_reentry_reference": prior_orb_reentry_reference,
        "equivalence_assessment": equivalence,
        "method": (
            "Alpaca-backed local feasibility test for an ORB hold ratchet on top of v26. "
            "Once a breakout has made enough favourable progress, the branch promotes a structural stop floor "
            "near the ORB breakout hold zone so moderate-progress reversals are cut earlier without relying on "
            "raw ORB re-entry exits."
        ),
        "baseline": {
            "label": "v26_BE=1.25x_gate=180min_profitLock=0.25x_after_1.50x",
            "metrics": baseline_metrics,
            "annual_metrics": baseline_years,
            "time_folds": baseline_folds,
        },
        "grid": {
            "arm_progress_mults": ARM_PROGRESS_MULTS,
            "floor_offset_mults": FLOOR_OFFSET_MULTS,
            "min_bars_after_arm_options": MIN_BARS_AFTER_ARM_OPTIONS,
        },
        "all_variants": all_results,
        "candidate_summary": candidate_summary,
        "structural_conclusion": {
            "research_interpretation": (
                "This branch promotes the ORB breakout hold zone into a persistent stop floor after enough progress, "
                "but the strongest Alpaca result now appears behaviorally equivalent to the earlier ORB re-entry branch."
            ),
            "next_step_rule": (
                "Do not build a Blueshift evaluator if the strongest ORB hold ratchet result is equivalent to the earlier "
                "ORB re-entry branch; only reopen it if later evidence shows meaningfully different QC or local behavior."
            ),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved results to {output_path}")
    print(json.dumps(candidate_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
