#!/usr/bin/env python3
"""
Late-session time-decay trailing stop simulation on local 5-minute data.

This tests a genuinely new exit mechanism after v23/v24 were rejected:
- keep the standard 1.3% trail early in the trade
- only tighten the trail as holding time increases
- avoid the failure mode of "wider trail / no trail", which let too much PnL
  deteriorate into the close

Mechanism:
- Start with the baseline trail_pct = 1.3%
- After `decay_start_min`, tighten the trail by `decay_step_pct`
  every `decay_step_min` minutes
- Never go tighter than `min_trail_pct`

This is a local-only path-level study. Because accepted baseline QC raw trades
are currently missing from the workspace, this script does not attempt QC-proxy
promotion; it only identifies whether a local research-only leader exists.
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
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_time_decay_trail.json"
)
ANALYSIS_VERSION = "v1_late_session_time_decay_trail"

DECAY_STARTS = [120, 180, 240]
DECAY_STEP_MINS = [30, 60]
DECAY_STEP_PCTS = [0.001, 0.002]
MIN_TRAIL_PCTS = [0.010, 0.008, 0.006]


def simulate_orb_time_decay_trail(
    df: pd.DataFrame,
    params: dict,
    *,
    decay_start_min: int,
    decay_step_min: int,
    decay_step_pct: float,
    min_trail_pct: float,
):
    """
    Simulate ORB strategy with a time-decay trailing stop.

    Trail schedule:
    - baseline trail until `decay_start_min`
    - then every `decay_step_min`, subtract `decay_step_pct`
    - do not go below `min_trail_pct`
    """
    orb_bars = int(params["orb_bars"])
    profit_ratio = float(params["profit_ratio"])
    close_before_min = int(params["close_before_min"])
    breakout_pct = float(params["breakout_confirm_pct"])
    entry_delay_bars = int(params["entry_delay_bars"])
    normal_trailing_pct = float(params["trailing_pct"])
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
                            False,
                            False,
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
                            False,
                            False,
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
                    session_entry_count += 1
                    continue

            if entry_ts is None:
                continue

            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            if elapsed_min <= decay_start_min:
                active_trail_pct = normal_trailing_pct
            else:
                decay_steps = int((elapsed_min - decay_start_min) // decay_step_min) + 1
                tightened = normal_trailing_pct - decay_steps * decay_step_pct
                active_trail_pct = max(min_trail_pct, tightened)

            if in_long:
                bars_in_trade += 1
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)

                trail_sl = best_price_long * (1.0 - active_trail_pct)
                effective_sl = max(long_initial_stop, trail_sl)

                if close <= effective_sl:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    exit_reason = "decay_stop" if elapsed_min > decay_start_min else "stop"
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
                            False,
                            elapsed_min > decay_start_min,
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
                            False,
                            elapsed_min > decay_start_min,
                            bars_in_trade,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            if in_short:
                bars_in_trade += 1
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)

                trail_sl = best_price_short * (1.0 + active_trail_pct)
                effective_sl = min(short_initial_stop, trail_sl)

                if close >= effective_sl:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    exit_reason = "decay_stop" if elapsed_min > decay_start_min else "stop"
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
                            False,
                            elapsed_min > decay_start_min,
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
                            False,
                            elapsed_min > decay_start_min,
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


def variant_sort_key(result: dict) -> tuple:
    path = result["path_impact"]
    return (
        result["improved_vs_baseline_folds"],
        result["positive_test_folds"],
        result["pnl_delta"],
        -path["clipped_winners"],
        path["saved_losses"],
    )


def summarize_variant(result: dict | None) -> dict | None:
    if result is None:
        return None
    metrics = result["metrics"]
    path = result["path_impact"]
    return {
        "label": result["label"],
        "decay_start_min": result["decay_start_min"],
        "decay_step_min": result["decay_step_min"],
        "decay_step_pct": result["decay_step_pct"],
        "min_trail_pct": result["min_trail_pct"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "positive_test_folds": result["positive_test_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "decay_stop_exits": result["decay_stop_exits"],
    }


def main() -> int:
    print("Loading 5m data...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"Enforcing max_entries_per_session={MAX_ENTRIES_PER_SESSION}")

    baseline_trades = simulate_orb_time_decay_trail(
        df,
        BASE_PARAMS,
        decay_start_min=9999,
        decay_step_min=60,
        decay_step_pct=0.0,
        min_trail_pct=float(BASE_PARAMS["trailing_pct"]),
    )
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_folds = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}

    print(
        f"Baseline: trades={baseline_metrics['trades']}, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}"
    )

    results = []
    for decay_start in DECAY_STARTS:
        for step_min in DECAY_STEP_MINS:
            for step_pct in DECAY_STEP_PCTS:
                for min_trail in MIN_TRAIL_PCTS:
                    if min_trail >= float(BASE_PARAMS["trailing_pct"]):
                        continue

                    label = (
                        f"decayStart={decay_start}_step={step_min}_"
                        f"delta={step_pct:.3f}_floor={min_trail:.3f}"
                    )
                    trades = simulate_orb_time_decay_trail(
                        df,
                        BASE_PARAMS,
                        decay_start_min=decay_start,
                        decay_step_min=step_min,
                        decay_step_pct=step_pct,
                        min_trail_pct=min_trail,
                    )
                    metrics = compute_metrics(trades)
                    fold_results = compute_fold_results(trades, folds)
                    improved_folds = sum(
                        1
                        for row in fold_results
                        if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
                    )
                    positive_folds = sum(1 for row in fold_results if row["total_pnl"] > 0)
                    path_impact = analyze_path_impact(baseline_trades, trades)
                    pnl_delta = round(metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)
                    decay_stop_exits = sum(1 for t in trades if t.exit_reason == "decay_stop")

                    print(
                        f"{label}: delta={pnl_delta:+.4f}, folds={improved_folds}/{len(folds)}, "
                        f"saved={path_impact['saved_losses']}, clipped={path_impact['clipped_winners']}, "
                        f"decay_stops={decay_stop_exits}"
                    )

                    results.append(
                        {
                            "label": label,
                            "decay_start_min": decay_start,
                            "decay_step_min": step_min,
                            "decay_step_pct": step_pct,
                            "min_trail_pct": min_trail,
                            "metrics": metrics,
                            "fold_results": fold_results,
                            "improved_vs_baseline_folds": improved_folds,
                            "positive_test_folds": positive_folds,
                            "pnl_delta": pnl_delta,
                            "path_impact": path_impact,
                            "decay_stop_exits": decay_stop_exits,
                        }
                    )

    best_overall = max(results, key=variant_sort_key, default=None)
    best_positive = max([r for r in results if r["pnl_delta"] > 0], key=variant_sort_key, default=None)
    best_zero_clip_positive = max(
        [r for r in results if r["pnl_delta"] > 0 and r["path_impact"]["clipped_winners"] == 0],
        key=variant_sort_key,
        default=None,
    )

    payload = {
        "research_scope": "local_orb_time_decay_trail",
        "analysis_version": ANALYSIS_VERSION,
        "data": {
            "source": DATA_PATH.name,
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Late-session time-decay trailing stop. Keep the baseline 1.3% trail early, "
            "then progressively tighten it as the trade ages. This directly targets the "
            "EOD deterioration observed when widening or removing the trail."
        ),
        "baseline": {
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "grid": {
            "decay_starts": DECAY_STARTS,
            "decay_step_mins": DECAY_STEP_MINS,
            "decay_step_pcts": DECAY_STEP_PCTS,
            "min_trail_pcts": MIN_TRAIL_PCTS,
        },
        "variants": results,
        "candidate_summary": {
            "best_overall": summarize_variant(best_overall),
            "best_positive": summarize_variant(best_positive),
            "best_zero_clip_positive": summarize_variant(best_zero_clip_positive),
            "interpretation": (
                "A compelling local-only leader would improve both full-sample PnL and walk-forward "
                "folds without heavy clipping. Because accepted QC baseline raw trades are not currently "
                "present in the workspace, any positive result here remains research-only until a QC-proxy "
                "or real QC rerun becomes possible."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {OUTPUT_PATH}")

    if best_overall:
        print(f"Best overall: {best_overall['label']} delta={best_overall['pnl_delta']:+.4f}")
    if best_positive:
        print(f"Best positive: {best_positive['label']} delta={best_positive['pnl_delta']:+.4f}")
    else:
        print("No positive candidate found.")
    if best_zero_clip_positive:
        print(
            f"Best zero-clip positive: {best_zero_clip_positive['label']} "
            f"delta={best_zero_clip_positive['pnl_delta']:+.4f}"
        )
    else:
        print("No zero-clip positive candidate found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
