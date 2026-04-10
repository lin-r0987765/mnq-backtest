#!/usr/bin/env python3
"""
Approximate the QC early-duration breakeven lane with fixed-price triggers locally.

QC trade analysis identifies the strongest conservative lane as:
    duration <= 180 minutes and MFE > $25

The local path simulator works in per-share price space rather than total trade
P&L dollars, so this script approximates the QC anchor with fixed absolute price
triggers, then applies the same time-gated breakeven logic on qqq_5m.csv.

This is still short-sample path evidence only. The purpose is to test whether
the QC-only conservative lane can survive a more direct local path approximation
than the earlier range-multiple trigger sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.orb.analyze_local_orb_timegated_breakeven import (
    BASE_PARAMS,
    DATA_PATH,
    MAX_ENTRIES_PER_SESSION,
    WALK_FORWARD_PATH,
    _make_trade_record,
    analyze_path_impact,
    compute_fold_results,
    compute_htf_bias,
    compute_metrics,
    load_csv_5m,
    load_test_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_absolute_be_mapping.json"
)
ANALYSIS_VERSION = "v1_absolute_price_trigger_mapping"

# Approximate $25 total MFE threshold from QC quantity distribution.
# For the accepted trades, $25 maps to roughly $0.15-$0.50 per share across the
# 10th-90th percentile quantity range, so sweep a compact set inside that band.
ABSOLUTE_PRICE_TRIGGERS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TIME_GATES = [120, 180, 240]


def simulate_orb_absolute_be(
    df,
    params: dict,
    be_trigger_points: float,
    time_gate_minutes: int,
):
    """
    Simulate ORB strategy with a fixed-price breakeven trigger and time gate.

    - be_trigger_points: unrealised per-share price move required to activate BE.
    - time_gate_minutes: BE only stays active within the first N minutes.
    """
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
    trades = []
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
        entry_ts = None
        best_price_long = 0.0
        best_price_short = float("inf")
        worst_price_long = float("inf")
        worst_price_short = 0.0
        bars_in_trade = 0
        be_activated = False
        be_gate_expired = False
        session_entry_count = 0

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    trades.append(
                        _make_trade_record(
                            entry_ts,
                            ts,
                            "long",
                            entry_price,
                            close,
                            "eod",
                            mfe,
                            mae,
                            be_activated,
                            be_gate_expired,
                            bars_in_trade,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    trades.append(
                        _make_trade_record(
                            entry_ts,
                            ts,
                            "short",
                            entry_price,
                            close,
                            "eod",
                            mfe,
                            mae,
                            be_activated,
                            be_gate_expired,
                            bars_in_trade,
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
                    session_entry_count += 1
                    continue

            if entry_ts is None:
                continue
            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            gate_active = elapsed_min <= time_gate_minutes
            if not gate_active and not be_gate_expired:
                be_gate_expired = True

            if in_long and entry_ts is not None:
                bars_in_trade += 1
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)

                if gate_active and not be_activated and (close - entry_price) >= be_trigger_points:
                    be_activated = True

                trail_sl = best_price_long * (1.0 - trailing_pct)
                if be_activated and gate_active:
                    effective_sl = max(entry_price, trail_sl)
                else:
                    effective_sl = max(long_initial_stop, trail_sl)

                if close <= effective_sl:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    exit_reason = "be_stop" if (be_activated and gate_active and close <= entry_price * 1.001) else "stop"
                    trades.append(
                        _make_trade_record(
                            entry_ts,
                            ts,
                            "long",
                            entry_price,
                            close,
                            exit_reason,
                            mfe,
                            mae,
                            be_activated,
                            be_gate_expired,
                            bars_in_trade,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue
                if close >= tp_long:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    trades.append(
                        _make_trade_record(
                            entry_ts,
                            ts,
                            "long",
                            entry_price,
                            close,
                            "target",
                            mfe,
                            mae,
                            be_activated,
                            be_gate_expired,
                            bars_in_trade,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            if in_short and entry_ts is not None:
                bars_in_trade += 1
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)

                if gate_active and not be_activated and (entry_price - close) >= be_trigger_points:
                    be_activated = True

                trail_sl = best_price_short * (1.0 + trailing_pct)
                if be_activated and gate_active:
                    effective_sl = min(entry_price, trail_sl)
                else:
                    effective_sl = min(short_initial_stop, trail_sl)

                if close >= effective_sl:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    exit_reason = "be_stop" if (be_activated and gate_active and close >= entry_price * 0.999) else "stop"
                    trades.append(
                        _make_trade_record(
                            entry_ts,
                            ts,
                            "short",
                            entry_price,
                            close,
                            exit_reason,
                            mfe,
                            mae,
                            be_activated,
                            be_gate_expired,
                            bars_in_trade,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue
                if close <= tp_short:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    trades.append(
                        _make_trade_record(
                            entry_ts,
                            ts,
                            "short",
                            entry_price,
                            close,
                            "target",
                            mfe,
                            mae,
                            be_activated,
                            be_gate_expired,
                            bars_in_trade,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def candidate_sort_key(result: dict) -> tuple:
    path = result["path_impact"]
    return (
        result["path_impact"]["clipped_winners"] == 0,
        result["improved_vs_baseline_folds"],
        result["pnl_delta"],
        -path["clipped_winners"],
        path["saved_losses"],
    )


def summarize_result(result: dict | None) -> dict | None:
    if result is None:
        return None
    metrics = result["metrics"]
    path = result["path_impact"]
    return {
        "label": result["label"],
        "trigger_points": result["trigger_points"],
        "time_gate_min": result["time_gate_min"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "be_stop_exits": metrics["be_stop_exits"],
    }


def main() -> int:
    print("Loading 5m data...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"Enforcing max_entries_per_session={MAX_ENTRIES_PER_SESSION}")
    print(f"Absolute BE triggers: {ABSOLUTE_PRICE_TRIGGERS}")
    print(f"Time gates: {TIME_GATES}")

    baseline_trades = simulate_orb_absolute_be(df, BASE_PARAMS, be_trigger_points=10_000.0, time_gate_minutes=9999)
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_folds = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}
    print(
        f"Baseline: {baseline_metrics['trades']} trades, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}"
    )

    results = []
    for trigger in ABSOLUTE_PRICE_TRIGGERS:
        for gate in TIME_GATES:
            label = f"absBE={trigger:.2f}_gate={gate}min"
            trades = simulate_orb_absolute_be(df, BASE_PARAMS, be_trigger_points=trigger, time_gate_minutes=gate)
            metrics = compute_metrics(trades)
            fold_results = compute_fold_results(trades, folds)
            improved_folds = sum(
                1 for row in fold_results if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
            )
            path_impact = analyze_path_impact(baseline_trades, trades)
            pnl_delta = round(metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)
            print(
                f"{label}: pnl_delta={pnl_delta:+.4f}, saved={path_impact['saved_losses']}, "
                f"clipped={path_impact['clipped_winners']}, folds={improved_folds}/4"
            )
            results.append(
                {
                    "label": label,
                    "trigger_points": trigger,
                    "time_gate_min": gate,
                    "metrics": metrics,
                    "fold_results": fold_results,
                    "improved_vs_baseline_folds": improved_folds,
                    "pnl_delta": pnl_delta,
                    "path_impact": path_impact,
                }
            )

    best_180 = max(
        [row for row in results if row["time_gate_min"] == 180],
        key=candidate_sort_key,
        default=None,
    )
    best_zero_clip = max(
        [row for row in results if row["path_impact"]["clipped_winners"] == 0 and row["pnl_delta"] > 0],
        key=candidate_sort_key,
        default=None,
    )
    best_overall = max(results, key=candidate_sort_key, default=None)

    payload = {
        "research_scope": "local_orb_absolute_price_breakeven_mapping",
        "analysis_version": ANALYSIS_VERSION,
        "data": {
            "source": DATA_PATH.name,
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Approximate the QC conservative lane with absolute per-share breakeven "
            "triggers and time gates on local 5m data."
        ),
        "baseline": {
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "absolute_price_triggers": ABSOLUTE_PRICE_TRIGGERS,
        "time_gates": TIME_GATES,
        "variants": results,
        "candidate_summary": {
            "best_overall": summarize_result(best_overall),
            "best_180min_candidate": summarize_result(best_180),
            "best_zero_clip_positive": summarize_result(best_zero_clip),
            "interpretation": (
                "If low absolute-price triggers remain harmful locally, then the QC-only "
                "MFE>$25 lane cannot be approximated by a naive low-trigger BE rule. "
                "That would mean the conservative QC lane still lacks a clean path-level mapping."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {OUTPUT_PATH}")

    if best_overall:
        print(f"Best overall: {best_overall['label']} delta={best_overall['pnl_delta']:+.4f}")
    if best_180:
        print(f"Best 180-minute candidate: {best_180['label']} delta={best_180['pnl_delta']:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
