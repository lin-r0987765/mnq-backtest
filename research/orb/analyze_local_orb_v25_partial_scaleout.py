#!/usr/bin/env python3
"""
Local path-level research for a partial scale-out mechanism on top of v25.

Mechanism:
- keep v25 baseline rules intact:
  - time-gated breakeven: 1.25x ORB range during first 180 minutes
  - trailing stop: 1.3%
- add a single partial scale-out:
  - when unrealized profit reaches `scaleout_trigger x ORB range`
  - close a fraction of the position
  - leave the remainder on the normal v25 regime

Why this branch exists:
- prior full-position profit protection ideas clipped winners
- partial scale-out is materially new because it monetizes some MFE without
  forcing the entire position out
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from research.orb.analyze_local_orb_v25_profit_lock import (
    BASE_BE_GATE_MIN,
    BASE_BE_TRIGGER,
    BASE_PARAMS,
    FEE_PCT,
    INITIAL_CASH,
    PROJECT_ROOT,
    WALK_FORWARD_PATH,
    DATA_PATH,
    MAX_ENTRIES_PER_SESSION,
    analyze_path_impact,
    compute_htf_bias,
    load_csv_5m,
    load_test_folds,
)


OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_v25_partial_scaleout.json"
ANALYSIS_VERSION = "v1_v25_single_scaleout"

SCALEOUT_TRIGGERS = [1.25, 1.50, 1.75, 2.00]
SCALEOUT_FRACTIONS = [0.25, 1.0 / 3.0, 0.50]


@dataclass
class TradeRecord:
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
    scaleout_activated: bool
    scaleout_fraction: float
    scaleout_price: float | None
    bars_in_trade: int


def leg_pnl(entry_price: float, exit_price: float, side: str, fraction: float) -> float:
    gross = exit_price - entry_price if side == "long" else entry_price - exit_price
    fees = (entry_price + exit_price) * FEE_PCT
    return fraction * (gross - fees)


def make_trade(
    *,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    exit_reason: str,
    mfe: float,
    mae: float,
    be_activated: bool,
    be_gate_expired: bool,
    scaleout_activated: bool,
    scaleout_fraction: float,
    scaleout_price: float | None,
    bars_in_trade: int,
) -> TradeRecord:
    return TradeRecord(
        entry_time=entry_ts,
        exit_time=exit_ts,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        duration_min=float((exit_ts - entry_ts).total_seconds() / 60.0),
        exit_reason=exit_reason,
        mfe=mfe,
        mae=mae,
        be_activated=be_activated,
        be_gate_expired=be_gate_expired,
        scaleout_activated=scaleout_activated,
        scaleout_fraction=scaleout_fraction,
        scaleout_price=scaleout_price,
        bars_in_trade=bars_in_trade,
    )


def simulate_orb_v25_partial_scaleout(
    df: pd.DataFrame,
    params: dict,
    *,
    scaleout_trigger_mult: float | None,
    scaleout_fraction: float,
) -> list[TradeRecord]:
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

    use_scaleout = scaleout_trigger_mult is not None and scaleout_fraction > 0.0

    htf_bias = compute_htf_bias(df, htf_ema_fast, htf_ema_slow, htf_mode) if htf_filter else None

    trades: list[TradeRecord] = []
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
        scaleout_trigger_pts = scaleout_trigger_mult * range_width if use_scaleout else None

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
        scaleout_activated = False
        scaleout_price: float | None = None
        remaining_fraction = 1.0
        realized_pnl = 0.0
        session_entry_count = 0

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    pnl = realized_pnl + leg_pnl(entry_price, close, "long", remaining_fraction)
                    trades.append(
                        make_trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side="long",
                            entry_price=entry_price,
                            exit_price=close,
                            pnl=pnl,
                            exit_reason="eod",
                            mfe=best_price_long - entry_price,
                            mae=entry_price - worst_price_long,
                            be_activated=be_activated,
                            be_gate_expired=be_gate_expired,
                            scaleout_activated=scaleout_activated,
                            scaleout_fraction=scaleout_fraction if scaleout_activated else 0.0,
                            scaleout_price=scaleout_price,
                            bars_in_trade=bars_in_trade,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    pnl = realized_pnl + leg_pnl(entry_price, close, "short", remaining_fraction)
                    trades.append(
                        make_trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side="short",
                            entry_price=entry_price,
                            exit_price=close,
                            pnl=pnl,
                            exit_reason="eod",
                            mfe=entry_price - best_price_short,
                            mae=worst_price_short - entry_price,
                            be_activated=be_activated,
                            be_gate_expired=be_gate_expired,
                            scaleout_activated=scaleout_activated,
                            scaleout_fraction=scaleout_fraction if scaleout_activated else 0.0,
                            scaleout_price=scaleout_price,
                            bars_in_trade=bars_in_trade,
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
                    scaleout_activated = False
                    scaleout_price = None
                    remaining_fraction = 1.0
                    realized_pnl = 0.0
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
                    scaleout_activated = False
                    scaleout_price = None
                    remaining_fraction = 1.0
                    realized_pnl = 0.0
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

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True

                if (
                    use_scaleout
                    and not scaleout_activated
                    and unrealised >= scaleout_trigger_pts
                ):
                    realized_pnl += leg_pnl(entry_price, close, "long", scaleout_fraction)
                    remaining_fraction -= scaleout_fraction
                    scaleout_activated = True
                    scaleout_price = close

                trail_sl = best_price_long * (1.0 - trailing_pct)
                effective_sl = max(long_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = max(effective_sl, entry_price)

                if close <= effective_sl:
                    pnl = realized_pnl + leg_pnl(entry_price, close, "long", remaining_fraction)
                    exit_reason = "be_stop" if (be_activated and gate_active and close <= entry_price * 1.001) else "stop"
                    if scaleout_activated:
                        exit_reason = f"scaleout_plus_{exit_reason}"
                    trades.append(
                        make_trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side="long",
                            entry_price=entry_price,
                            exit_price=close,
                            pnl=pnl,
                            exit_reason=exit_reason,
                            mfe=best_price_long - entry_price,
                            mae=entry_price - worst_price_long,
                            be_activated=be_activated,
                            be_gate_expired=be_gate_expired,
                            scaleout_activated=scaleout_activated,
                            scaleout_fraction=scaleout_fraction if scaleout_activated else 0.0,
                            scaleout_price=scaleout_price,
                            bars_in_trade=bars_in_trade,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

                if close >= tp_long:
                    pnl = realized_pnl + leg_pnl(entry_price, close, "long", remaining_fraction)
                    exit_reason = "target"
                    if scaleout_activated:
                        exit_reason = "scaleout_plus_target"
                    trades.append(
                        make_trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side="long",
                            entry_price=entry_price,
                            exit_price=close,
                            pnl=pnl,
                            exit_reason=exit_reason,
                            mfe=best_price_long - entry_price,
                            mae=entry_price - worst_price_long,
                            be_activated=be_activated,
                            be_gate_expired=be_gate_expired,
                            scaleout_activated=scaleout_activated,
                            scaleout_fraction=scaleout_fraction if scaleout_activated else 0.0,
                            scaleout_price=scaleout_price,
                            bars_in_trade=bars_in_trade,
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

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True

                if (
                    use_scaleout
                    and not scaleout_activated
                    and unrealised >= scaleout_trigger_pts
                ):
                    realized_pnl += leg_pnl(entry_price, close, "short", scaleout_fraction)
                    remaining_fraction -= scaleout_fraction
                    scaleout_activated = True
                    scaleout_price = close

                trail_sl = best_price_short * (1.0 + trailing_pct)
                effective_sl = min(short_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = min(effective_sl, entry_price)

                if close >= effective_sl:
                    pnl = realized_pnl + leg_pnl(entry_price, close, "short", remaining_fraction)
                    exit_reason = "be_stop" if (be_activated and gate_active and close >= entry_price * 0.999) else "stop"
                    if scaleout_activated:
                        exit_reason = f"scaleout_plus_{exit_reason}"
                    trades.append(
                        make_trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side="short",
                            entry_price=entry_price,
                            exit_price=close,
                            pnl=pnl,
                            exit_reason=exit_reason,
                            mfe=entry_price - best_price_short,
                            mae=worst_price_short - entry_price,
                            be_activated=be_activated,
                            be_gate_expired=be_gate_expired,
                            scaleout_activated=scaleout_activated,
                            scaleout_fraction=scaleout_fraction if scaleout_activated else 0.0,
                            scaleout_price=scaleout_price,
                            bars_in_trade=bars_in_trade,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

                if close <= tp_short:
                    pnl = realized_pnl + leg_pnl(entry_price, close, "short", remaining_fraction)
                    exit_reason = "target"
                    if scaleout_activated:
                        exit_reason = "scaleout_plus_target"
                    trades.append(
                        make_trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side="short",
                            entry_price=entry_price,
                            exit_price=close,
                            pnl=pnl,
                            exit_reason=exit_reason,
                            mfe=entry_price - best_price_short,
                            mae=worst_price_short - entry_price,
                            be_activated=be_activated,
                            be_gate_expired=be_gate_expired,
                            scaleout_activated=scaleout_activated,
                            scaleout_fraction=scaleout_fraction if scaleout_activated else 0.0,
                            scaleout_price=scaleout_price,
                            bars_in_trade=bars_in_trade,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def compute_metrics(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "total_pnl": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "trade_sharpe": 0.0,
            "avg_trade_pnl": 0.0,
            "max_drawdown_pct": 0.0,
            "eod_exits": 0,
            "scaleout_activated_count": 0,
            "scaleout_plus_eod_exits": 0,
            "scaleout_plus_stop_exits": 0,
            "scaleout_plus_be_stop_exits": 0,
            "scaleout_plus_target_exits": 0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
        }
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    equity = INITIAL_CASH
    peak = INITIAL_CASH
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = (equity - peak) / peak * 100.0
        max_dd = min(max_dd, dd)
    return {
        "trades": len(trades),
        "total_pnl": round(sum(pnls), 4),
        "win_rate_pct": round((len(wins) / len(pnls) * 100.0), 2),
        "profit_factor": round((sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0, 4),
        "trade_sharpe": round((mean(pnls) / stdev(pnls)) if len(pnls) > 1 and stdev(pnls) > 0 else 0.0, 4),
        "avg_trade_pnl": round(mean(pnls), 4),
        "max_drawdown_pct": round(max_dd, 4),
        "eod_exits": sum(1 for t in trades if t.exit_reason == "eod"),
        "scaleout_activated_count": sum(1 for t in trades if t.scaleout_activated),
        "scaleout_plus_eod_exits": sum(1 for t in trades if t.exit_reason == "scaleout_plus_eod"),
        "scaleout_plus_stop_exits": sum(1 for t in trades if t.exit_reason == "scaleout_plus_stop"),
        "scaleout_plus_be_stop_exits": sum(1 for t in trades if t.exit_reason == "scaleout_plus_be_stop"),
        "scaleout_plus_target_exits": sum(1 for t in trades if t.exit_reason == "scaleout_plus_target"),
        "avg_mfe": round(mean(t.mfe for t in trades), 4),
        "avg_mae": round(mean(t.mae for t in trades), 4),
    }


def compute_fold_results(trades: list[TradeRecord], folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        subset = [t for t in trades if fold["start"] <= t.entry_time.date() <= fold["end"]]
        rows.append({"fold": fold["fold"], "start": str(fold["start"]), "end": str(fold["end"]), **compute_metrics(subset)})
    return rows


def summarize_variant(result: dict | None) -> dict | None:
    if result is None:
        return None
    metrics = result["metrics"]
    path = result["path_impact"]
    return {
        "label": result["label"],
        "scaleout_trigger": result["scaleout_trigger"],
        "scaleout_fraction": result["scaleout_fraction"],
        "pnl": metrics["total_pnl"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "positive_test_folds": result["positive_test_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "scaleout_activated_count": metrics["scaleout_activated_count"],
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
    print("Loading local data for v25 partial scale-out research...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"Baseline: v25 BE={BASE_BE_TRIGGER:.2f}x gate={BASE_BE_GATE_MIN}min")

    baseline_trades = simulate_orb_v25_partial_scaleout(
        df,
        BASE_PARAMS,
        scaleout_trigger_mult=None,
        scaleout_fraction=0.0,
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
    for trigger in SCALEOUT_TRIGGERS:
        for fraction in SCALEOUT_FRACTIONS:
            label = f"scaleout_{fraction:.2f}_at_{trigger:.2f}x"
            print(f"\nRunning {label}...")
            variant_trades = simulate_orb_v25_partial_scaleout(
                df,
                BASE_PARAMS,
                scaleout_trigger_mult=trigger,
                scaleout_fraction=fraction,
            )
            variant_metrics = compute_metrics(variant_trades)
            variant_folds = compute_fold_results(variant_trades, folds)
            positive_folds = sum(1 for row in variant_folds if row["total_pnl"] > 0)
            improved_folds = sum(
                1
                for row in variant_folds
                if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
            )
            path_impact = analyze_path_impact(baseline_trades, variant_trades)
            pnl_delta = round(variant_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)
            eod_delta = variant_metrics["eod_exits"] - baseline_metrics["eod_exits"]

            print(
                f"  PnL={variant_metrics['total_pnl']:+.2f} (delta={pnl_delta:+.2f}), "
                f"PF={variant_metrics['profit_factor']:.3f}, "
                f"scaleouts={variant_metrics['scaleout_activated_count']}, "
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
                    "scaleout_trigger": trigger,
                    "scaleout_fraction": round(fraction, 4),
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
        "research_scope": "local_orb_v25_partial_scaleout",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v25-timegated-be",
        "data": {
            "source": DATA_PATH.name,
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Local path-level simulation of a single partial scale-out layered on top of the "
            "v25 baseline. When a profit threshold is reached, a fraction of the position is "
            "closed and the remainder continues under the normal v25 regime."
        ),
        "baseline": {
            "label": f"v25_BE={BASE_BE_TRIGGER:.2f}x_gate={BASE_BE_GATE_MIN}min",
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "grid": {
            "scaleout_triggers": SCALEOUT_TRIGGERS,
            "scaleout_fractions": [round(x, 4) for x in SCALEOUT_FRACTIONS],
        },
        "all_variants": all_results,
        "best_overall": {
            "label": best_overall["label"],
            "scaleout_trigger": best_overall["scaleout_trigger"],
            "scaleout_fraction": best_overall["scaleout_fraction"],
            "pnl_delta": best_overall["pnl_delta"],
            "improved_folds": best_overall["improved_vs_baseline_folds"],
            "passes_local_bar": best_passes_local_bar,
        },
        "candidate_summary": {
            "best_positive": summarize_variant(best_positive),
            "best_zero_clip_positive": summarize_variant(best_zero_clip_positive),
            "best_balanced": summarize_variant(best_balanced),
            "interpretation": (
                "A partial scale-out branch is interesting only if it improves full-sample local "
                "PnL and shows stronger walk-forward behavior than the prior full-position locks."
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
