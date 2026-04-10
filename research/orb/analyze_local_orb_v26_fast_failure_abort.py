#!/usr/bin/env python3
"""
Hybrid v26 research for an early fast-failure abort exit.

Why this exists:
- The late-session washout family has now been consolidated and remains
  non-launchable from local-only evidence.
- Accepted QC v26 trades still show another distinct failure mode:
  a meaningful cluster of quick losers never makes enough early progress,
  yet still absorbs a large adverse move before the normal baseline exits.

Mechanism tested here:
- Keep the full v26 baseline unchanged:
  - BE trigger = 1.25 x ORB range during first 180 minutes
  - persistent profit lock = +0.25 x ORB range after 1.50 x ORB range MFE
- Add a new optional early abort:
  - after N minutes from entry,
  - if best favourable move so far is still below `min_progress x ORB range`,
  - and current unrealized PnL is below `-abort_loss x ORB range`,
  - exit immediately at the current close.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.orb.analyze_local_orb_v25_profit_lock import (
    BASE_BE_GATE_MIN,
    BASE_BE_TRIGGER,
    BASE_PARAMS,
    DATA_PATH,
    MAX_ENTRIES_PER_SESSION,
    PROJECT_ROOT,
    WALK_FORWARD_PATH,
    _make_trade_record,
    analyze_path_impact,
    compute_htf_bias,
    load_csv_5m,
    load_test_folds,
    simulate_orb_v25_profit_lock,
)
from research.orb.analyze_local_orb_v26_stagnation_exit import (
    BASE_V26_PROFIT_LOCK_LEVEL,
    BASE_V26_PROFIT_LOCK_TRIGGER,
    QC_TRADES_PATH,
    compute_fold_results_extended,
    compute_metrics_extended,
)


OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_v26_fast_failure_abort.json"
)
ANALYSIS_VERSION = "v1_v26_early_fast_failure_abort"

FAILURE_CHECK_MINS = [30, 45, 60, 75]
MIN_PROGRESS_MULTS = [0.25, 0.50, 0.75]
ABORT_LOSS_MULTS = [0.75, 1.00, 1.25]


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
    fast_failure_exit: bool


def build_qc_quick_failure_diagnostic() -> dict:
    if not QC_TRADES_PATH.exists():
        return {
            "bundle": None,
            "status": "qc_baseline_bundle_missing",
            "path_trades": str(QC_TRADES_PATH),
            "loser_count": 0,
            "quick_loser_definition": "P&L <= 0 and duration <= 120 minutes",
            "quick_loser_count": 0,
            "quick_loser_mean_pnl": 0.0,
            "quick_loser_mfe_lt_25_count": 0,
            "quick_loser_mfe_lt_50_count": 0,
            "large_mae_low_progress_count": 0,
            "large_mae_low_progress_mean_pnl": 0.0,
        }

    trades = pd.read_csv(QC_TRADES_PATH)
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
    trades["duration_min"] = (trades["Exit Time"] - trades["Entry Time"]).dt.total_seconds() / 60.0
    for col in ["P&L", "MAE", "MFE", "Drawdown"]:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    losers = trades[trades["P&L"] <= 0].copy()
    quick_losers = losers[losers["duration_min"] <= 120.0].copy()
    large_mae_low_progress = losers[(losers["MFE"] < 50.0) & (losers["MAE"] < -75.0)].copy()

    return {
        "bundle": "Square Blue Termite",
        "loser_count": int(len(losers)),
        "quick_loser_definition": "P&L <= 0 and duration <= 120 minutes",
        "quick_loser_count": int(len(quick_losers)),
        "quick_loser_mean_pnl": round(float(quick_losers["P&L"].mean()), 4)
        if not quick_losers.empty
        else 0.0,
        "quick_loser_mfe_lt_25_count": int((quick_losers["MFE"] < 25.0).sum()),
        "quick_loser_mfe_lt_50_count": int((quick_losers["MFE"] < 50.0).sum()),
        "large_mae_low_progress_count": int(len(large_mae_low_progress)),
        "large_mae_low_progress_mean_pnl": round(float(large_mae_low_progress["P&L"].mean()), 4)
        if not large_mae_low_progress.empty
        else 0.0,
    }


def simulate_orb_v26_fast_failure_abort(
    df: pd.DataFrame,
    params: dict,
    *,
    failure_check_min: int | None,
    min_progress_mult: float | None,
    abort_loss_mult: float | None,
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

    use_fast_failure = (
        failure_check_min is not None
        and failure_check_min > 0
        and min_progress_mult is not None
        and min_progress_mult > 0
        and abort_loss_mult is not None
        and abort_loss_mult > 0
    )

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
        min_progress_pts = min_progress_mult * range_width if use_fast_failure else None
        abort_loss_pts = abort_loss_mult * range_width if use_fast_failure else None

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

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    record = _make_trade_record(
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
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=False))
                    in_long = False
                if in_short and entry_ts is not None:
                    record = _make_trade_record(
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
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=False))
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
                peak_progress = best_price_long - entry_price

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if (
                    not profit_lock_activated
                    and gate_active
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price + profit_lock_pts

                if (
                    use_fast_failure
                    and elapsed_min >= failure_check_min
                    and peak_progress <= min_progress_pts
                    and unrealised <= -abort_loss_pts
                ):
                    record = _make_trade_record(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side="long",
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="fast_failure_exit",
                        mfe=best_price_long - entry_price,
                        mae=entry_price - worst_price_long,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=profit_lock_activated,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=True))
                    in_long = False
                    entry_ts = None
                    continue

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
                    record = _make_trade_record(
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
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=False))
                    in_long = False
                    entry_ts = None
                    continue

                if close >= tp_long:
                    record = _make_trade_record(
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
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=False))
                    in_long = False
                    entry_ts = None
                    continue

            if in_short:
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)
                unrealised = entry_price - close
                peak_progress = entry_price - best_price_short

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if (
                    not profit_lock_activated
                    and gate_active
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price - profit_lock_pts

                if (
                    use_fast_failure
                    and elapsed_min >= failure_check_min
                    and peak_progress <= min_progress_pts
                    and unrealised <= -abort_loss_pts
                ):
                    record = _make_trade_record(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side="short",
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="fast_failure_exit",
                        mfe=entry_price - best_price_short,
                        mae=worst_price_short - entry_price,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=profit_lock_activated,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=True))
                    in_short = False
                    entry_ts = None
                    continue

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
                    record = _make_trade_record(
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
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=False))
                    in_short = False
                    entry_ts = None
                    continue

                if close <= tp_short:
                    record = _make_trade_record(
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
                    )
                    trades.append(ExtendedTradeRecord(**record.__dict__, fast_failure_exit=False))
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def summarize_candidate(row: dict, path: dict) -> dict:
    return {
        "label": row["label"],
        "failure_check_min": row["failure_check_min"],
        "min_progress_mult": row["min_progress_mult"],
        "abort_loss_mult": row["abort_loss_mult"],
        "pnl": row["metrics"]["total_pnl"],
        "pnl_delta": row["pnl_delta"],
        "profit_factor": row["metrics"]["profit_factor"],
        "win_rate_pct": row["metrics"]["win_rate_pct"],
        "improved_folds": row["improved_folds"],
        "positive_test_folds": row["positive_test_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "fast_failure_exit_count": row["metrics"]["fast_failure_exit_count"],
    }


def choose_candidate_summary(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["pnl_delta"] > 0]
    zero_clip = [row for row in positive if row["path_impact"]["clipped_winners"] == 0]
    balanced = [row for row in positive if row["improved_folds"] >= 2]

    best_positive = max(positive, key=lambda row: row["pnl_delta"], default=None)
    best_zero_clip = max(zero_clip, key=lambda row: row["pnl_delta"], default=None)
    best_balanced = max(
        balanced,
        key=lambda row: (row["improved_folds"], row["pnl_delta"]),
        default=None,
    )

    strongest = best_balanced or best_zero_clip or best_positive
    if strongest is None:
        verdict = "LOCAL_REJECTED"
    elif strongest["improved_folds"] >= 3:
        verdict = "READY_FOR_QC_PROXY"
    elif strongest["improved_folds"] >= 2:
        verdict = "LOCAL_STRONG_NEAR_MISS"
    else:
        verdict = "LOCAL_NEAR_MISS"

    return {
        "best_positive": summarize_candidate(best_positive, best_positive["path_impact"])
        if best_positive
        else None,
        "best_zero_clip_positive": summarize_candidate(best_zero_clip, best_zero_clip["path_impact"])
        if best_zero_clip
        else None,
        "best_balanced": summarize_candidate(best_balanced, best_balanced["path_impact"])
        if best_balanced
        else None,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday-csv", default=str(DATA_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data_path = Path(args.intraday_csv)
    output_path = Path(args.output_path)

    print("Loading local data for v26 fast-failure abort research...")
    df = load_csv_5m(data_path)
    folds = load_test_folds(WALK_FORWARD_PATH)
    qc_diag = build_qc_quick_failure_diagnostic()

    baseline_trades = simulate_orb_v25_profit_lock(
        df,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER,
        profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL,
    )
    baseline_metrics = compute_metrics_extended(
        [ExtendedTradeRecord(**t.__dict__, fast_failure_exit=False) for t in baseline_trades]
    )
    baseline_metrics["fast_failure_exit_count"] = 0
    baseline_folds = compute_fold_results_extended(
        [ExtendedTradeRecord(**t.__dict__, fast_failure_exit=False) for t in baseline_trades],
        folds,
    )
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}

    print(
        f"Local baseline v26: trades={baseline_metrics['trades']}, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}, "
        f"EOD={baseline_metrics['eod_exits']}, "
        f"profitLockStops={baseline_metrics['profit_lock_stop_exits']}"
    )
    print(
        "QC quick-failure diagnostic: "
        f"quickLosers={qc_diag['quick_loser_count']}, "
        f"mfeLt25={qc_diag['quick_loser_mfe_lt_25_count']}, "
        f"mfeLt50={qc_diag['quick_loser_mfe_lt_50_count']}"
    )

    all_results = []
    for check_min in FAILURE_CHECK_MINS:
        for progress_mult in MIN_PROGRESS_MULTS:
            for abort_mult in ABORT_LOSS_MULTS:
                label = (
                    f"fast_failure_after_{check_min}m_progress_lt_{progress_mult:.2f}x_loss_gt_{abort_mult:.2f}x"
                )
                print(f"Running {label}...")
                variant_trades = simulate_orb_v26_fast_failure_abort(
                    df,
                    BASE_PARAMS,
                    failure_check_min=check_min,
                    min_progress_mult=progress_mult,
                    abort_loss_mult=abort_mult,
                )
                variant_metrics = compute_metrics_extended(variant_trades)
                variant_metrics["fast_failure_exit_count"] = sum(
                    1 for t in variant_trades if t.exit_reason == "fast_failure_exit"
                )
                variant_folds = compute_fold_results_extended(variant_trades, folds)
                improved_folds = sum(
                    1
                    for row in variant_folds
                    if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
                )
                positive_test_folds = sum(1 for row in variant_folds if row["total_pnl"] > 0)
                path_impact = analyze_path_impact(baseline_trades, variant_trades)
                result = {
                    "label": label,
                    "failure_check_min": check_min,
                    "min_progress_mult": progress_mult,
                    "abort_loss_mult": abort_mult,
                    "metrics": variant_metrics,
                    "fold_results": variant_folds,
                    "improved_folds": improved_folds,
                    "positive_test_folds": positive_test_folds,
                    "pnl_delta": round(
                        variant_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4
                    ),
                    "eod_delta": variant_metrics["eod_exits"] - baseline_metrics["eod_exits"],
                    "path_impact": path_impact,
                }
                all_results.append(result)
                print(
                    f"  delta={result['pnl_delta']:+.2f}, "
                    f"folds={improved_folds}/{len(folds)}, "
                    f"ffExits={variant_metrics['fast_failure_exit_count']}, "
                    f"EOD_delta={result['eod_delta']:+d}, "
                    f"netPath={path_impact['net_impact']:+.2f}"
                )

    candidate_summary = choose_candidate_summary(all_results)

    payload = {
        "research_scope": "local_orb_v26_fast_failure_abort",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v26-profit-lock",
        "data": {
            "source": data_path.name,
            "path": str(data_path),
            "bars": int(len(df)),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Hybrid v26 feasibility test for an early fast-failure abort exit. "
            "The branch exits only if a trade still shows too little best-so-far progress after a minimum age "
            "and is already underwater by a configured ORB-multiple loss."
        ),
        "qc_v26_quick_failure_diagnostic": qc_diag,
        "baseline": {
            "label": "v26_BE=1.25x_gate=180min_profitLock=0.25x_after_1.50x",
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "grid": {
            "failure_check_mins": FAILURE_CHECK_MINS,
            "min_progress_mults": MIN_PROGRESS_MULTS,
            "abort_loss_mults": ABORT_LOSS_MULTS,
        },
        "all_variants": all_results,
        "candidate_summary": candidate_summary,
        "structural_conclusion": {
            "qc_observation": (
                "Accepted v26 trades still contain a meaningful cluster of quick losers that never make much early progress "
                "before absorbing large adverse moves."
            ),
            "research_interpretation": (
                "That makes fast-failure aborts a materially different branch from the already-tested late-session washout exits."
            ),
            "next_step_rule": (
                "Do not launch a QC candidate unless this branch improves local full-sample PnL and at least 2/4 walk-forward folds."
            ),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved results to {output_path}")
    print(json.dumps(candidate_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
