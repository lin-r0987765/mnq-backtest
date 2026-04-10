#!/usr/bin/env python3
"""
Local path-level research for a late-stage peak-giveback guard on top of v25.

Mechanism:
- keep the official v25 baseline unchanged:
  - BE trigger = 1.25 x ORB range
  - BE active during first 180 minutes
- add a new late-stage protection:
  - after the trade has lasted at least N minutes
  - and after peak favourable excursion reaches T x ORB range
  - exit if giveback from the peak reaches G x ORB range

This is materially different from:
- fixed profit-lock thresholds anchored to entry
- wide / removed trail
- time-decay trail grids
- the already-promoted time-gated breakeven branch itself
"""
from __future__ import annotations

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
    OUTPUT_PATH as _UNUSED_OUTPUT_PATH,
    PROJECT_ROOT,
    WALK_FORWARD_PATH,
    _make_trade_record,
    analyze_path_impact,
    compute_fold_results,
    compute_htf_bias,
    compute_metrics,
    load_csv_5m,
    load_test_folds,
    simulate_orb_v25_profit_lock,
)


OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_v25_peak_giveback.json"
ANALYSIS_VERSION = "v1_v25_late_peak_giveback_guard"

GIVEBACK_START_MINS = [180, 240, 300]
PEAK_TRIGGER_MULTS = [1.50, 2.00, 2.50, 3.00]
GIVEBACK_CAP_MULTS = [0.50, 0.75, 1.00, 1.25]


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
    peak_guard_armed: bool
    bars_in_trade: int


def simulate_orb_v25_peak_giveback(
    df: pd.DataFrame,
    params: dict,
    *,
    giveback_start_min: int,
    peak_trigger_mult: float,
    giveback_cap_mult: float,
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
        peak_trigger_pts = peak_trigger_mult * range_width
        giveback_cap_pts = giveback_cap_mult * range_width

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
        peak_guard_armed = False
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
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
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
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
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
                    peak_guard_armed = False
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
                    peak_guard_armed = False
                    session_entry_count += 1
                    continue

            if entry_ts is None:
                continue

            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            gate_active = elapsed_min <= BASE_BE_GATE_MIN
            if not gate_active and not be_gate_expired:
                be_gate_expired = True

            if in_long:
                bars_in_trade += 1
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)
                unrealised = close - entry_price
                peak_mfe = best_price_long - entry_price
                giveback = best_price_long - close

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True

                if elapsed_min >= giveback_start_min and peak_mfe >= peak_trigger_pts:
                    peak_guard_armed = True

                trail_sl = best_price_long * (1.0 - trailing_pct)
                effective_sl = max(long_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = max(effective_sl, entry_price)

                if peak_guard_armed and giveback >= giveback_cap_pts:
                    record = _make_trade_record(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side="long",
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="peak_giveback_stop",
                        mfe=peak_mfe,
                        mae=entry_price - worst_price_long,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

                if close <= effective_sl:
                    exit_reason = "be_stop" if (be_activated and gate_active and close <= entry_price * 1.001) else "stop"
                    record = _make_trade_record(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side="long",
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason=exit_reason,
                        mfe=peak_mfe,
                        mae=entry_price - worst_price_long,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
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
                        mfe=peak_mfe,
                        mae=entry_price - worst_price_long,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            if in_short:
                bars_in_trade += 1
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)
                unrealised = entry_price - close
                peak_mfe = entry_price - best_price_short
                giveback = close - best_price_short

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True

                if elapsed_min >= giveback_start_min and peak_mfe >= peak_trigger_pts:
                    peak_guard_armed = True

                trail_sl = best_price_short * (1.0 + trailing_pct)
                effective_sl = min(short_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = min(effective_sl, entry_price)

                if peak_guard_armed and giveback >= giveback_cap_pts:
                    record = _make_trade_record(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side="short",
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason="peak_giveback_stop",
                        mfe=peak_mfe,
                        mae=worst_price_short - entry_price,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

                if close >= effective_sl:
                    exit_reason = "be_stop" if (be_activated and gate_active and close >= entry_price * 0.999) else "stop"
                    record = _make_trade_record(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side="short",
                        entry_price=entry_price,
                        exit_price=close,
                        exit_reason=exit_reason,
                        mfe=peak_mfe,
                        mae=worst_price_short - entry_price,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
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
                        mfe=peak_mfe,
                        mae=worst_price_short - entry_price,
                        be_activated=be_activated,
                        be_gate_expired=be_gate_expired,
                        profit_lock_activated=False,
                        bars_in_trade=bars_in_trade,
                    )
                    trades.append(
                        ExtendedTradeRecord(
                            **record.__dict__,
                            peak_guard_armed=peak_guard_armed,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def compute_extended_metrics(trades: list[ExtendedTradeRecord]) -> dict:
    base = compute_metrics(trades)  # type: ignore[arg-type]
    base["peak_giveback_stop_exits"] = sum(1 for t in trades if t.exit_reason == "peak_giveback_stop")
    base["peak_guard_armed_count"] = sum(1 for t in trades if t.peak_guard_armed)
    return base


def compute_extended_fold_results(trades: list[ExtendedTradeRecord], folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        subset = [t for t in trades if fold["start"] <= t.entry_time.date() <= fold["end"]]
        rows.append(
            {
                "fold": fold["fold"],
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                **compute_extended_metrics(subset),
            }
        )
    return rows


def summarize_variant(result: dict | None) -> dict | None:
    if result is None:
        return None
    metrics = result["metrics"]
    path = result["path_impact"]
    return {
        "label": result["label"],
        "giveback_start_min": result["giveback_start_min"],
        "peak_trigger_mult": result["peak_trigger_mult"],
        "giveback_cap_mult": result["giveback_cap_mult"],
        "pnl": metrics["total_pnl"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "positive_test_folds": result["positive_test_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "peak_giveback_stop_exits": metrics["peak_giveback_stop_exits"],
        "peak_guard_armed_count": metrics["peak_guard_armed_count"],
    }


def variant_sort_key(result: dict) -> tuple:
    path = result["path_impact"]
    metrics = result["metrics"]
    return (
        result["improved_vs_baseline_folds"],
        result["pnl_delta"],
        -path["clipped_winners"],
        path["saved_losses"],
        metrics["profit_factor"],
    )


def main() -> int:
    print("Loading local data for v25 late peak-giveback research...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(
        f"Baseline: v25 BE={BASE_BE_TRIGGER:.2f}x gate={BASE_BE_GATE_MIN}min | "
        f"max_entries_per_session={MAX_ENTRIES_PER_SESSION}"
    )

    baseline_trades = simulate_orb_v25_profit_lock(
        df,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=None,
        profit_lock_level_mult=None,
    )
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_folds = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}

    print(
        f"Baseline: trades={baseline_metrics['trades']}, "
        f"PnL={baseline_metrics['total_pnl']:+.2f}, "
        f"PF={baseline_metrics['profit_factor']:.3f}, "
        f"EOD={baseline_metrics['eod_exits']}"
    )

    all_results = []
    for start_min in GIVEBACK_START_MINS:
        for trigger in PEAK_TRIGGER_MULTS:
            for cap in GIVEBACK_CAP_MULTS:
                label = f"peak_cap={cap:.2f}x_after_{trigger:.2f}x_from_{start_min}m"
                print(f"\nRunning {label}...")
                variant_trades = simulate_orb_v25_peak_giveback(
                    df,
                    BASE_PARAMS,
                    giveback_start_min=start_min,
                    peak_trigger_mult=trigger,
                    giveback_cap_mult=cap,
                )
                variant_metrics = compute_extended_metrics(variant_trades)
                variant_folds = compute_extended_fold_results(variant_trades, folds)
                positive_folds = sum(1 for row in variant_folds if row["total_pnl"] > 0)
                improved_folds = sum(
                    1
                    for row in variant_folds
                    if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
                )
                path_impact = analyze_path_impact(baseline_trades, variant_trades)  # type: ignore[arg-type]
                pnl_delta = round(variant_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)
                eod_delta = variant_metrics["eod_exits"] - baseline_metrics["eod_exits"]

                print(
                    f"  PnL={variant_metrics['total_pnl']:+.2f} (delta={pnl_delta:+.2f}), "
                    f"PF={variant_metrics['profit_factor']:.3f}, "
                    f"peakStops={variant_metrics['peak_giveback_stop_exits']}, "
                    f"armed={variant_metrics['peak_guard_armed_count']}, "
                    f"folds={improved_folds}/{len(folds)}, "
                    f"EOD_delta={eod_delta:+d}"
                )
                print(
                    f"  Path: saved={path_impact['saved_losses']}, "
                    f"clipped={path_impact['clipped_winners']}, "
                    f"net={path_impact['net_impact']:+.2f}"
                )

                all_results.append(
                    {
                        "giveback_start_min": start_min,
                        "peak_trigger_mult": trigger,
                        "giveback_cap_mult": cap,
                        "label": label,
                        "metrics": variant_metrics,
                        "fold_results": variant_folds,
                        "positive_test_folds": positive_folds,
                        "improved_vs_baseline_folds": improved_folds,
                        "pnl_delta": pnl_delta,
                        "eod_exit_delta": eod_delta,
                        "path_impact": path_impact,
                    }
                )

    best_overall = max(all_results, key=variant_sort_key)
    best_positive = max([r for r in all_results if r["pnl_delta"] > 0], key=variant_sort_key, default=None)
    best_zero_clip_positive = max(
        [r for r in all_results if r["pnl_delta"] > 0 and r["path_impact"]["clipped_winners"] == 0],
        key=variant_sort_key,
        default=None,
    )
    best_balanced = max([r for r in all_results if r["improved_vs_baseline_folds"] >= 2], key=variant_sort_key, default=None)

    best_passes_local_bar = (
        best_overall["pnl_delta"] > 0
        and best_overall["improved_vs_baseline_folds"] >= 3
        and best_overall["metrics"]["profit_factor"] >= baseline_metrics["profit_factor"]
    )

    payload = {
        "research_scope": "local_orb_v25_peak_giveback",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v25-timegated-be",
        "data": {
            "source": DATA_PATH.name,
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Local path-level simulation of a late-stage peak-giveback guard layered on top of "
            "the v25 baseline. After a minimum trade age and minimum peak MFE threshold, exit if "
            "the giveback from the peak reaches a configured ORB-multiple cap."
        ),
        "baseline": {
            "label": f"v25_BE={BASE_BE_TRIGGER:.2f}x_gate={BASE_BE_GATE_MIN}min",
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "grid": {
            "giveback_start_mins": GIVEBACK_START_MINS,
            "peak_trigger_mults": PEAK_TRIGGER_MULTS,
            "giveback_cap_mults": GIVEBACK_CAP_MULTS,
        },
        "all_variants": all_results,
        "best_overall": {
            "label": best_overall["label"],
            "giveback_start_min": best_overall["giveback_start_min"],
            "peak_trigger_mult": best_overall["peak_trigger_mult"],
            "giveback_cap_mult": best_overall["giveback_cap_mult"],
            "pnl_delta": best_overall["pnl_delta"],
            "improved_folds": best_overall["improved_vs_baseline_folds"],
            "passes_local_bar": best_passes_local_bar,
        },
        "candidate_summary": {
            "best_positive": summarize_variant(best_positive),
            "best_zero_clip_positive": summarize_variant(best_zero_clip_positive),
            "best_balanced": summarize_variant(best_balanced),
            "interpretation": (
                "A late peak-giveback guard is only interesting if it improves full-sample local PnL "
                "while remaining reasonably stable across walk-forward folds. If not, do not launch it."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {OUTPUT_PATH}")
    print(
        f"Best overall: {best_overall['label']} "
        f"(delta={best_overall['pnl_delta']:+.2f}, passes={best_passes_local_bar})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
