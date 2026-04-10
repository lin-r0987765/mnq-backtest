#!/usr/bin/env python3
"""
Tightened trailing stop within a time gate on local 5-minute data.

This is a fundamentally different approach from breakeven mapping.
Instead of trying to move the stop to entry price, we simply tighten
the trailing stop percentage during the early phase of the trade.

Rationale:
- The QC-only conservative lane shows that early-duration trades (<=180min)
  benefit from tighter risk management without clipping winners
- A tighter trail during the early phase captures the same intuition
  WITHOUT requiring an MFE threshold mapping
- After the time gate expires, the trailing reverts to the normal 1.3%

Sweep: tight_trail 0.003-0.010 with 120/180/240 minute gates.
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
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_tight_trail.json"
)
ANALYSIS_VERSION = "v1_tightened_trailing_stop"

# Tight trailing percentages (base is 0.013 = 1.3%)
TIGHT_TRAILS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010]
TIME_GATES = [120, 180, 240]


def simulate_orb_tight_trail(
    df: pd.DataFrame,
    params: dict,
    tight_trail_pct: float,
    time_gate_minutes: int,
):
    """
    Simulate ORB strategy with a tightened trailing stop within a time gate.

    - tight_trail_pct: trailing stop pct used within the first N minutes
    - time_gate_minutes: how long the tight trail is active
    - After the gate expires, revert to normal trailing_pct from params
    """
    orb_bars = int(params["orb_bars"])
    profit_ratio = float(params["profit_ratio"])
    close_before_min = int(params["close_before_min"])
    breakout_pct = float(params["breakout_confirm_pct"])
    entry_delay_bars = int(params["entry_delay_bars"])
    normal_trailing_pct = float(params["trailing_pct"])  # 0.013
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
        be_gate_expired = False
        session_entry_count = 0

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    trades.append(_make_trade_record(
                        entry_ts, ts, "long", entry_price, close, "eod",
                        mfe, mae, False, be_gate_expired, bars_in_trade,
                    ))
                    in_long = False
                if in_short and entry_ts is not None:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    trades.append(_make_trade_record(
                        entry_ts, ts, "short", entry_price, close, "eod",
                        mfe, mae, False, be_gate_expired, bars_in_trade,
                    ))
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
                    be_gate_expired = False
                    session_entry_count += 1
                    continue

            if entry_ts is None:
                continue
            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            gate_active = elapsed_min <= time_gate_minutes
            if not gate_active and not be_gate_expired:
                be_gate_expired = True

            # Select trailing pct based on time gate
            active_trail_pct = tight_trail_pct if gate_active else normal_trailing_pct

            if in_long and entry_ts is not None:
                bars_in_trade += 1
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)

                trail_sl = best_price_long * (1.0 - active_trail_pct)
                effective_sl = max(long_initial_stop, trail_sl)

                if close <= effective_sl:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    exit_reason = "tight_stop" if gate_active else "stop"
                    trades.append(_make_trade_record(
                        entry_ts, ts, "long", entry_price, close, exit_reason,
                        mfe, mae, False, be_gate_expired, bars_in_trade,
                    ))
                    in_long = False
                    entry_ts = None
                    continue
                if close >= tp_long:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    trades.append(_make_trade_record(
                        entry_ts, ts, "long", entry_price, close, "target",
                        mfe, mae, False, be_gate_expired, bars_in_trade,
                    ))
                    in_long = False
                    entry_ts = None
                    continue

            if in_short and entry_ts is not None:
                bars_in_trade += 1
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)

                trail_sl = best_price_short * (1.0 + active_trail_pct)
                effective_sl = min(short_initial_stop, trail_sl)

                if close >= effective_sl:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    exit_reason = "tight_stop" if gate_active else "stop"
                    trades.append(_make_trade_record(
                        entry_ts, ts, "short", entry_price, close, exit_reason,
                        mfe, mae, False, be_gate_expired, bars_in_trade,
                    ))
                    in_short = False
                    entry_ts = None
                    continue
                if close <= tp_short:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    trades.append(_make_trade_record(
                        entry_ts, ts, "short", entry_price, close, "target",
                        mfe, mae, False, be_gate_expired, bars_in_trade,
                    ))
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
        path["clipped_winners"] == 0,
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
        "tight_trail_pct": result["tight_trail_pct"],
        "time_gate_min": result["time_gate_min"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "tight_stop_exits": sum(1 for t in result.get("_trades_ref", []) if getattr(t, "exit_reason", "") == "tight_stop"),
    }


def main() -> int:
    print("Loading 5m data...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"Enforcing max_entries_per_session={MAX_ENTRIES_PER_SESSION}")
    print(f"Tight trailing pcts: {TIGHT_TRAILS}")
    print(f"Time gates: {TIME_GATES}")

    # Baseline: normal trailing_pct = 0.013 always (simulate by setting tight=normal)
    baseline_trades = simulate_orb_tight_trail(
        df, BASE_PARAMS,
        tight_trail_pct=float(BASE_PARAMS["trailing_pct"]),
        time_gate_minutes=9999,
    )
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_folds = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}
    print(
        f"Baseline: {baseline_metrics['trades']} trades, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}"
    )

    results = []
    for trail in TIGHT_TRAILS:
        for gate in TIME_GATES:
            label = f"tightTrail={trail*100:.1f}%_gate={gate}min"
            trades = simulate_orb_tight_trail(df, BASE_PARAMS, tight_trail_pct=trail, time_gate_minutes=gate)
            metrics = compute_metrics(trades)
            fold_results = compute_fold_results(trades, folds)
            improved_folds = sum(
                1 for row in fold_results if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
            )
            path_impact = analyze_path_impact(baseline_trades, trades)
            pnl_delta = round(metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)

            # Count tight_stop exits from the trades list directly
            tight_stop_count = sum(1 for t in trades if t.exit_reason == "tight_stop")

            print(
                f"{label}: pnl_delta={pnl_delta:+.4f}, saved={path_impact['saved_losses']}, "
                f"clipped={path_impact['clipped_winners']}, folds={improved_folds}/{len(folds)}, "
                f"tight_exits={tight_stop_count}"
            )
            results.append({
                "label": label,
                "tight_trail_pct": trail,
                "time_gate_min": gate,
                "metrics": metrics,
                "fold_results": fold_results,
                "improved_vs_baseline_folds": improved_folds,
                "pnl_delta": pnl_delta,
                "path_impact": path_impact,
                "tight_stop_exits": tight_stop_count,
            })

    # Find best candidates
    best_180 = max(
        [r for r in results if r["time_gate_min"] == 180],
        key=candidate_sort_key,
        default=None,
    )
    best_zero_clip_positive = max(
        [r for r in results if r["path_impact"]["clipped_winners"] == 0 and r["pnl_delta"] > 0],
        key=candidate_sort_key,
        default=None,
    )
    best_overall = max(results, key=candidate_sort_key, default=None)

    # Summarize without _trades_ref since we tracked tight_stop_exits directly
    def summarize(r):
        if r is None:
            return None
        metrics = r["metrics"]
        path = r["path_impact"]
        return {
            "label": r["label"],
            "tight_trail_pct": r["tight_trail_pct"],
            "time_gate_min": r["time_gate_min"],
            "pnl_delta": r["pnl_delta"],
            "profit_factor": metrics["profit_factor"],
            "win_rate_pct": metrics["win_rate_pct"],
            "improved_folds": r["improved_vs_baseline_folds"],
            "saved_losses": path["saved_losses"],
            "clipped_winners": path["clipped_winners"],
            "net_path_impact": path["net_impact"],
            "tight_stop_exits": r["tight_stop_exits"],
        }

    payload = {
        "research_scope": "local_orb_tightened_trailing_stop",
        "analysis_version": ANALYSIS_VERSION,
        "data": {
            "source": DATA_PATH.name,
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Tightened trailing stop within a time gate. Instead of a hard breakeven "
            "trigger, simply use a tighter trailing_pct during the early phase. "
            "This avoids the MFE threshold mapping problem entirely. "
            "After the time gate, reverts to normal trailing (1.3%)."
        ),
        "baseline": {
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "tight_trails": TIGHT_TRAILS,
        "normal_trail": float(BASE_PARAMS["trailing_pct"]),
        "time_gates": TIME_GATES,
        "variants": results,
        "candidate_summary": {
            "best_overall": summarize(best_overall),
            "best_180min_candidate": summarize(best_180),
            "best_zero_clip_positive": summarize(best_zero_clip_positive),
            "interpretation": (
                "If a tighter trail during early phase improves PnL locally without "
                "clipping, this provides a path-level confirmation that the QC-only "
                "early-duration benefit is real and implementable. The advantage is "
                "that this approach needs no MFE dollar mapping at all."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {OUTPUT_PATH}")

    if best_overall:
        print(f"Best overall: {best_overall['label']} delta={best_overall['pnl_delta']:+.4f}")
    if best_180:
        print(f"Best 180min: {best_180['label']} delta={best_180['pnl_delta']:+.4f}")
    if best_zero_clip_positive:
        print(f"Best zero-clip positive: {best_zero_clip_positive['label']} delta={best_zero_clip_positive['pnl_delta']:+.4f}")
    else:
        print("No zero-clip positive candidate found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
