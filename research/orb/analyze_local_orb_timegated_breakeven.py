#!/usr/bin/env python3
"""
Time-gated breakeven stop simulation on local 5-minute data.

This script implements the research priority from iteration 84:
Apply breakeven stop ONLY within the first N minutes of a trade,
then revert to normal trailing stop after the time gate expires.

Key insight from QC conditional analysis (iteration 84 addendum):
- Trades <=180 min with MFE>$50: 20 losses saved, 0 clipped, net +$3,076
- Trades >300 min: 24 saved, 32 clipped, net -$239
- The entire clipping risk is concentrated in long-duration trades

This script:
1. Runs the baseline ORB engine bar-by-bar on qqq_5m.csv
2. Adds a time-gated breakeven stop: BE only active within first T minutes
3. After time gate, BE reverts to normal trailing stop (even if previously activated)
4. Tests multiple time gate durations and BE trigger levels
5. Evaluates walk-forward stability

IMPORTANT: Enforces max_entries_per_session=1 matching QC production.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "qqq_5m.csv"
WALK_FORWARD_PATH = PROJECT_ROOT / "walk_forward_results.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_timegated_breakeven.json"
ANALYSIS_VERSION = "v2_with_300min_gate_and_candidate_summary"

INITIAL_CASH = 100_000.0
FEE_PCT = 0.0005
MAX_ENTRIES_PER_SESSION = 1

BASE_PARAMS = {
    "orb_bars": 4,
    "profit_ratio": 3.5,
    "close_before_min": 10,
    "breakout_confirm_pct": 0.0003,
    "entry_delay_bars": 0,
    "trailing_pct": 0.013,
    "htf_filter": True,
    "htf_mode": "slope",
    "htf_ema_fast": 20,
    "htf_ema_slow": 30,
    "skip_short_after_up_days": 2,
    "skip_long_after_up_days": 3,
    "initial_stop_mult": 1.0,
}

# Time gate durations in minutes (how long after entry BE stop is active)
TIME_GATES = [60, 90, 120, 150, 180, 240, 300, 9999]  # 9999 = always active (no gate)

# BE trigger thresholds as multiples of ORB range width
BE_TRIGGERS = [0.75, 1.00, 1.25]


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
    be_gate_expired: bool  # whether the time gate expired during this trade
    bars_in_trade: int


def load_csv_5m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    df.index.name = None
    return df


def compute_htf_bias(df_5m: pd.DataFrame, ema_fast: int, ema_slow: int, mode: str) -> pd.Series:
    df_1h = (
        df_5m.resample("1h")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
    ema_f = df_1h["Close"].ewm(span=ema_fast, adjust=False).mean()
    bias = pd.Series(0, index=df_1h.index, dtype=int)
    if mode == "slope":
        slope = ema_f.diff()
        bias[slope > 0] = 1
        bias[slope < 0] = -1
    else:
        ema_s = df_1h["Close"].ewm(span=ema_slow, adjust=False).mean()
        bias[ema_f > ema_s] = 1
        bias[ema_f < ema_s] = -1
    return bias.reindex(df_5m.index, method="ffill").fillna(0).astype(int)


def load_test_folds(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = []
    for fold in payload["orb"]["folds"]:
        start_s, end_s = [part.strip() for part in fold["test_period"].split("~")]
        folds.append(
            {
                "fold": int(fold["fold"]),
                "start": pd.Timestamp(start_s).date(),
                "end": pd.Timestamp(end_s).date(),
            }
        )
    return folds


def simulate_orb_timegated_be(
    df: pd.DataFrame, params: dict, be_trigger_mult: float, time_gate_minutes: int
) -> list[TradeRecord]:
    """
    Simulate ORB strategy with a TIME-GATED breakeven stop.

    - be_trigger_mult: once unrealised profit >= be_trigger_mult * orb_range,
                       move stop to breakeven (entry price).
    - time_gate_minutes: BE stop is only active for the first N minutes after entry.
                         After that, revert to normal trailing stop even if BE was activated.
    - If be_trigger_mult <= 0.0, this is the baseline (no breakeven).
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

    use_breakeven = be_trigger_mult > 0.0

    htf_bias = compute_htf_bias(df, htf_ema_fast, htf_ema_slow, htf_mode) if htf_filter else None

    trades: list[TradeRecord] = []
    up_day_streak = 0

    for session_date, session in df.groupby(df.index.date):
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

        be_trigger_pts = be_trigger_mult * range_width if use_breakeven else 0.0

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
                            entry_ts, ts, "long", entry_price, close, "eod",
                            mfe, mae, be_activated, be_gate_expired, bars_in_trade,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    trades.append(
                        _make_trade_record(
                            entry_ts, ts, "short", entry_price, close, "eod",
                            mfe, mae, be_activated, be_gate_expired, bars_in_trade,
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

            # Compute elapsed minutes since entry
            if entry_ts is None:
                continue
            elapsed_min = (ts - entry_ts).total_seconds() / 60.0
            gate_active = elapsed_min <= time_gate_minutes

            # If gate just expired, mark it
            if not gate_active and not be_gate_expired:
                be_gate_expired = True

            if in_long and entry_ts is not None:
                bars_in_trade += 1
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)

                # Check breakeven activation (only if gate is still active)
                if use_breakeven and gate_active and not be_activated:
                    unrealised = close - entry_price
                    if unrealised >= be_trigger_pts:
                        be_activated = True

                # Compute effective stop
                trail_sl = best_price_long * (1.0 - trailing_pct)
                if be_activated and gate_active:
                    # BE stop active: floor at entry price
                    effective_sl = max(entry_price, trail_sl)
                else:
                    # Normal mode (either BE never activated, or gate expired)
                    effective_sl = max(long_initial_stop, trail_sl)

                if close <= effective_sl:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    exit_reason = "be_stop" if (be_activated and gate_active and close <= entry_price * 1.001) else "stop"
                    trades.append(
                        _make_trade_record(
                            entry_ts, ts, "long", entry_price, close, exit_reason,
                            mfe, mae, be_activated, be_gate_expired, bars_in_trade,
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
                            entry_ts, ts, "long", entry_price, close, "target",
                            mfe, mae, be_activated, be_gate_expired, bars_in_trade,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            if in_short and entry_ts is not None:
                bars_in_trade += 1
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)

                if use_breakeven and gate_active and not be_activated:
                    unrealised = entry_price - close
                    if unrealised >= be_trigger_pts:
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
                            entry_ts, ts, "short", entry_price, close, exit_reason,
                            mfe, mae, be_activated, be_gate_expired, bars_in_trade,
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
                            entry_ts, ts, "short", entry_price, close, "target",
                            mfe, mae, be_activated, be_gate_expired, bars_in_trade,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def _make_trade_record(
    entry_ts, exit_ts, side, entry_price, exit_price, exit_reason,
    mfe, mae, be_activated, be_gate_expired, bars_in_trade,
) -> TradeRecord:
    gross = exit_price - entry_price if side == "long" else entry_price - exit_price
    fees = (entry_price + exit_price) * FEE_PCT
    pnl = gross - fees
    duration_min = float((exit_ts - entry_ts).total_seconds() / 60.0)
    return TradeRecord(
        entry_time=entry_ts,
        exit_time=exit_ts,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        duration_min=duration_min,
        exit_reason=exit_reason,
        mfe=mfe,
        mae=mae,
        be_activated=be_activated,
        be_gate_expired=be_gate_expired,
        bars_in_trade=bars_in_trade,
    )


def compute_metrics(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {
            "trades": 0, "total_pnl": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "trade_sharpe": 0.0, "avg_trade_pnl": 0.0, "max_drawdown_pct": 0.0,
            "stop_exits": 0, "be_stop_exits": 0, "target_exits": 0, "eod_exits": 0,
            "be_activated_count": 0, "be_gate_expired_count": 0,
            "avg_mfe": 0.0, "avg_mae": 0.0,
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
        "stop_exits": sum(1 for t in trades if t.exit_reason == "stop"),
        "be_stop_exits": sum(1 for t in trades if t.exit_reason == "be_stop"),
        "target_exits": sum(1 for t in trades if t.exit_reason == "target"),
        "eod_exits": sum(1 for t in trades if t.exit_reason == "eod"),
        "be_activated_count": sum(1 for t in trades if t.be_activated),
        "be_gate_expired_count": sum(1 for t in trades if t.be_gate_expired),
        "avg_mfe": round(mean(t.mfe for t in trades), 4),
        "avg_mae": round(mean(t.mae for t in trades), 4),
    }


def compute_fold_results(trades: list[TradeRecord], folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        subset = [
            t for t in trades if fold["start"] <= t.entry_time.date() <= fold["end"]
        ]
        metrics = compute_metrics(subset)
        rows.append({"fold": fold["fold"], "start": str(fold["start"]), "end": str(fold["end"]), **metrics})
    return rows


def analyze_path_impact(
    baseline_trades: list[TradeRecord],
    variant_trades: list[TradeRecord],
) -> dict:
    """Compare variant against baseline on a per-trade basis."""
    base_by_entry = {t.entry_time: t for t in baseline_trades}
    var_by_entry = {t.entry_time: t for t in variant_trades}
    common = set(base_by_entry.keys()) & set(var_by_entry.keys())

    saved_losses = 0
    clipped_winners = 0
    salvage_amount = 0.0
    harm_amount = 0.0
    unchanged = 0
    improved_other = 0

    for entry_ts in common:
        bt = base_by_entry[entry_ts]
        vt = var_by_entry[entry_ts]
        delta = vt.pnl - bt.pnl
        if abs(delta) < 0.01:
            unchanged += 1
        elif bt.pnl < 0 and delta > 0:
            saved_losses += 1
            salvage_amount += delta
        elif bt.pnl > 0 and delta < 0:
            clipped_winners += 1
            harm_amount += abs(delta)
        elif delta > 0:
            improved_other += 1
            salvage_amount += delta
        elif delta < 0:
            harm_amount += abs(delta)

    return {
        "common_trades": len(common),
        "only_baseline": len(set(base_by_entry.keys()) - set(var_by_entry.keys())),
        "only_variant": len(set(var_by_entry.keys()) - set(base_by_entry.keys())),
        "unchanged": unchanged,
        "saved_losses": saved_losses,
        "clipped_winners": clipped_winners,
        "improved_other": improved_other,
        "salvage": round(salvage_amount, 4),
        "harm": round(harm_amount, 4),
        "net_impact": round(salvage_amount - harm_amount, 4),
    }


def _variant_sort_key(result: dict) -> tuple:
    path = result["path_impact"]
    return (
        result["improved_vs_baseline_folds"],
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
        "be_trigger": result["be_trigger"],
        "time_gate_min": result["time_gate_min"],
        "pnl": metrics["total_pnl"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "positive_test_folds": result["positive_test_folds"],
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
    print(f"Time gates: {TIME_GATES}")
    print(f"BE triggers: {BE_TRIGGERS}")

    # Baseline (no breakeven)
    print("\nRunning baseline...")
    baseline_trades = simulate_orb_timegated_be(df, BASE_PARAMS, be_trigger_mult=0.0, time_gate_minutes=9999)
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_folds = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {r["fold"]: r for r in baseline_folds}

    print(f"Baseline: {baseline_metrics['trades']} trades, PnL={baseline_metrics['total_pnl']:+.2f}, "
          f"PF={baseline_metrics['profit_factor']:.3f}, WR={baseline_metrics['win_rate_pct']:.1f}%")

    # Grid search: BE trigger x time gate
    all_results = []
    for trigger in BE_TRIGGERS:
        for gate in TIME_GATES:
            label = f"BE={trigger:.2f}x_gate={gate}min"
            print(f"\nRunning {label}...")
            variant_trades = simulate_orb_timegated_be(df, BASE_PARAMS, be_trigger_mult=trigger, time_gate_minutes=gate)
            variant_metrics = compute_metrics(variant_trades)
            variant_folds = compute_fold_results(variant_trades, folds)

            positive_folds = sum(1 for r in variant_folds if r["total_pnl"] > 0)
            improved_folds = sum(
                1 for r in variant_folds
                if r["total_pnl"] > baseline_fold_map[r["fold"]]["total_pnl"]
            )

            path_impact = analyze_path_impact(baseline_trades, variant_trades)
            pnl_delta = round(variant_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)

            print(f"  PnL={variant_metrics['total_pnl']:+.2f} (delta={pnl_delta:+.2f}), "
                  f"PF={variant_metrics['profit_factor']:.3f}, "
                  f"BE_stops={variant_metrics['be_stop_exits']}, "
                  f"gate_expired={variant_metrics['be_gate_expired_count']}, "
                  f"folds={improved_folds}/{len(folds)}")
            print(f"  Path: saved={path_impact['saved_losses']}, "
                  f"clipped={path_impact['clipped_winners']}, "
                  f"net={path_impact['net_impact']:+.2f}")

            all_results.append({
                "be_trigger": trigger,
                "time_gate_min": gate,
                "label": label,
                "metrics": variant_metrics,
                "fold_results": variant_folds,
                "positive_test_folds": positive_folds,
                "improved_vs_baseline_folds": improved_folds,
                "pnl_delta": pnl_delta,
                "path_impact": path_impact,
            })

    # Find best variant
    best = max(
        all_results,
        key=lambda r: (
            r["improved_vs_baseline_folds"],
            r["positive_test_folds"],
            r["metrics"]["total_pnl"],
        ),
    )

    best_passes = (
        best["improved_vs_baseline_folds"] >= 3
        and best["metrics"]["total_pnl"] > baseline_metrics["total_pnl"]
        and best["metrics"]["profit_factor"] >= baseline_metrics["profit_factor"]
    )

    # Also find best time-gated-only (exclude 9999 = always-on)
    gated_results = [r for r in all_results if r["time_gate_min"] < 9999]
    best_gated = max(
        gated_results,
        key=lambda r: (
            r["improved_vs_baseline_folds"],
            r["positive_test_folds"],
            r["metrics"]["total_pnl"],
        ),
    ) if gated_results else None

    best_zero_clip_positive = max(
        [
            r
            for r in gated_results
            if r["pnl_delta"] > 0 and r["path_impact"]["clipped_winners"] == 0
        ],
        key=_variant_sort_key,
        default=None,
    )
    best_total_positive = max(
        [r for r in gated_results if r["pnl_delta"] > 0],
        key=_variant_sort_key,
        default=None,
    )
    best_180 = max(
        [r for r in gated_results if r["time_gate_min"] == 180],
        key=_variant_sort_key,
        default=None,
    )
    best_300 = max(
        [r for r in gated_results if r["time_gate_min"] == 300],
        key=_variant_sort_key,
        default=None,
    )

    # Compare gated vs always-on for same trigger
    comparison = {}
    for trigger in BE_TRIGGERS:
        always_on = [r for r in all_results if r["be_trigger"] == trigger and r["time_gate_min"] == 9999]
        gated = [r for r in all_results if r["be_trigger"] == trigger and r["time_gate_min"] < 9999]
        if always_on and gated:
            ao = always_on[0]
            bg = max(gated, key=lambda r: (r["improved_vs_baseline_folds"], r["metrics"]["total_pnl"]))
            comparison[f"BE={trigger:.2f}x"] = {
                "always_on": {
                    "pnl": ao["metrics"]["total_pnl"],
                    "pnl_delta": ao["pnl_delta"],
                    "clipped": ao["path_impact"]["clipped_winners"],
                    "saved": ao["path_impact"]["saved_losses"],
                    "improved_folds": ao["improved_vs_baseline_folds"],
                },
                "best_gated": {
                    "gate_min": bg["time_gate_min"],
                    "pnl": bg["metrics"]["total_pnl"],
                    "pnl_delta": bg["pnl_delta"],
                    "clipped": bg["path_impact"]["clipped_winners"],
                    "saved": bg["path_impact"]["saved_losses"],
                    "improved_folds": bg["improved_vs_baseline_folds"],
                },
                "gating_benefit": {
                    "reduced_clipping": ao["path_impact"]["clipped_winners"] - bg["path_impact"]["clipped_winners"],
                    "pnl_improvement": round(bg["pnl_delta"] - ao["pnl_delta"], 4),
                },
            }

    payload = {
        "research_scope": "local_orb_timegated_breakeven_simulation",
        "analysis_version": ANALYSIS_VERSION,
        "data": {
            "source": str(DATA_PATH.name),
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Time-gated breakeven stop simulation. BE stop is only active within "
            "the first N minutes after entry. After time gate expires, stop reverts "
            "to normal trailing stop. This targets the zero-clipping zone discovered "
            "in QC conditional analysis (trades <=180min with MFE>$50: 0 clipped winners)."
        ),
        "constraints": {
            "max_entries_per_session": MAX_ENTRIES_PER_SESSION,
        },
        "baseline": {
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "all_variants": all_results,
        "gated_vs_always_on_comparison": comparison,
        "best_overall": {
            "label": best["label"],
            "be_trigger": best["be_trigger"],
            "time_gate_min": best["time_gate_min"],
            "pnl_delta": best["pnl_delta"],
            "improved_folds": best["improved_vs_baseline_folds"],
            "passes_promotion_bar": best_passes,
        },
        "best_gated_only": {
            "label": best_gated["label"],
            "be_trigger": best_gated["be_trigger"],
            "time_gate_min": best_gated["time_gate_min"],
            "pnl_delta": best_gated["pnl_delta"],
            "improved_folds": best_gated["improved_vs_baseline_folds"],
            "clipped": best_gated["path_impact"]["clipped_winners"],
            "saved": best_gated["path_impact"]["saved_losses"],
        } if best_gated else None,
        "candidate_summary": {
            "best_zero_clip_positive": summarize_variant(best_zero_clip_positive),
            "best_total_positive": summarize_variant(best_total_positive),
            "best_180min_candidate": summarize_variant(best_180),
            "best_300min_candidate": summarize_variant(best_300),
            "interpretation": (
                "Use the 180-minute view as the conservative zero-clipping lane "
                "and the 300-minute view as the higher-uplift stress test. "
                "A candidate is only compelling if it stays positive locally "
                "and avoids heavy clipping on the short sample."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {OUTPUT_PATH}")

    print("\n=== Summary ===")
    print(f"Best overall: {best['label']} (delta={best['pnl_delta']:+.2f}, passes={best_passes})")
    if best_gated:
        print(f"Best gated: {best_gated['label']} (delta={best_gated['pnl_delta']:+.2f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())