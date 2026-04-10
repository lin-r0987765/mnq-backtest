#!/usr/bin/env python3
"""
Alpaca-backed local research for an ORB opening-regime classifier on top of v26.
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
OUTPUT_PATH = RESULTS_DIR / "local_orb_v26_opening_regime_classifier_alpaca.json"
ANALYSIS_VERSION = "v1_alpaca_v26_opening_regime_classifier"

BASE_V26_PROFIT_LOCK_TRIGGER = 1.50
BASE_V26_PROFIT_LOCK_LEVEL = 0.25

CONTINUATION_PROGRESS_MULTS = [0.25, 0.50, 0.75]
REVERSAL_DEPTH_MULTS = [0.10, 0.25, 0.50]
CONFIRM_BARS_OPTIONS = [1, 2]
DECISION_WINDOW_BARS = [3, 6]


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
    entry_mode: str


def compute_metrics_extended(trades: list[ExtendedTradeRecord]) -> dict:
    metrics = compute_metrics(trades)
    metrics["continuation_entry_count"] = sum(1 for trade in trades if trade.entry_mode == "continuation")
    metrics["reversal_entry_count"] = sum(1 for trade in trades if trade.entry_mode == "reversal")
    return metrics


def compute_annual_metrics(trades: list[ExtendedTradeRecord]) -> list[dict]:
    rows = []
    for year in sorted({trade.entry_time.year for trade in trades}):
        subset = [trade for trade in trades if trade.entry_time.year == year]
        rows.append({"year": year, **compute_metrics_extended(subset)})
    return rows


def _record(
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
    entry_mode: str,
) -> ExtendedTradeRecord:
    fee_pct = 0.0005
    gross = exit_price - entry_price if side == "long" else entry_price - exit_price
    fees = (entry_price + exit_price) * fee_pct
    return ExtendedTradeRecord(
        entry_time=entry_ts,
        exit_time=exit_ts,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=gross - fees,
        duration_min=float((exit_ts - entry_ts).total_seconds() / 60.0),
        exit_reason=exit_reason,
        mfe=mfe,
        mae=mae,
        be_activated=be_activated,
        be_gate_expired=be_gate_expired,
        profit_lock_activated=profit_lock_activated,
        bars_in_trade=bars_in_trade,
        entry_mode=entry_mode,
    )


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
        "continuation_progress_mult": row["continuation_progress_mult"],
        "reversal_depth_mult": row["reversal_depth_mult"],
        "confirm_bars": row["confirm_bars"],
        "decision_window_bars": row["decision_window_bars"],
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
        "continuation_entry_count": metrics["continuation_entry_count"],
        "reversal_entry_count": metrics["reversal_entry_count"],
    }


def choose_candidate_summary(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["pnl_delta"] > 0]
    balanced = [row for row in positive if row["improved_time_folds"] >= 2 and row["improved_years"] >= 3]
    strongest = max(balanced or positive, key=_variant_sort_key, default=None)
    if strongest is None:
        verdict = "ALPACA_LOCAL_REJECTED"
    elif strongest["improved_time_folds"] >= 3 and strongest["improved_years"] >= 4:
        verdict = "READY_FOR_BLUESHIFT_EVALUATOR"
    else:
        verdict = "ALPACA_LOCAL_NEAR_MISS"
    return {
        "best_positive": summarize_candidate(max(positive, key=_variant_sort_key, default=None)) if positive else None,
        "best_balanced": summarize_candidate(max(balanced, key=_variant_sort_key, default=None)) if balanced else None,
        "verdict": verdict,
    }


def simulate_orb_v26_opening_regime_classifier(
    df: pd.DataFrame,
    params: dict,
    *,
    continuation_progress_mult: float,
    reversal_depth_mult: float,
    confirm_bars: int,
    decision_window_bars: int,
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
        allow_short_today = not (skip_short_after_up_days > 0 and up_day_streak >= skip_short_after_up_days)
        allow_long_today = not (skip_long_after_up_days > 0 and up_day_streak >= skip_long_after_up_days)
        orb = sess.iloc[:orb_bars]
        orb_high = float(orb["High"].max())
        orb_low = float(orb["Low"].min())
        range_width = orb_high - orb_low
        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        if range_width <= 0:
            up_day_streak = up_day_streak + 1 if day_close > day_open else 0
            continue
        mid_price = (orb_high + orb_low) / 2.0
        if mid_price <= 0 or range_width / mid_price < 0.001:
            up_day_streak = up_day_streak + 1 if day_close > day_open else 0
            continue

        long_breakout_level = orb_high * (1.0 + breakout_pct)
        short_breakout_level = orb_low * (1.0 - breakout_pct)
        upside_cont_level = orb_high + continuation_progress_mult * range_width
        downside_cont_level = orb_low - continuation_progress_mult * range_width
        upside_fail_level = orb_high - reversal_depth_mult * range_width
        downside_fail_level = orb_low + reversal_depth_mult * range_width
        long_target = orb_high + profit_ratio * range_width
        short_target = orb_low - profit_ratio * range_width
        long_initial_stop = orb_low - (initial_stop_mult - 1.0) * range_width
        short_initial_stop = orb_high + (initial_stop_mult - 1.0) * range_width
        be_trigger_pts = BASE_BE_TRIGGER * range_width
        profit_lock_trigger_pts = BASE_V26_PROFIT_LOCK_TRIGGER * range_width
        profit_lock_pts = BASE_V26_PROFIT_LOCK_LEVEL * range_width

        post_orb = sess.iloc[orb_bars + entry_delay_bars:]
        if post_orb.empty:
            up_day_streak = up_day_streak + 1 if day_close > day_open else 0
            continue
        force_close_ts = sess.index[-1] - pd.Timedelta(minutes=close_before_min)

        in_long = in_short = False
        entry_price = 0.0
        entry_ts: pd.Timestamp | None = None
        best_price_long = 0.0
        best_price_short = float("inf")
        worst_price_long = float("inf")
        worst_price_short = 0.0
        bars_in_trade = 0
        be_activated = be_gate_expired = profit_lock_activated = False
        profit_lock_price: float | None = None
        entry_mode = ""
        session_entry_count = 0
        breakout_direction: str | None = None
        breakout_bar_count = cont_count = fail_count = 0

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])
            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    trades.append(_record(
                        entry_ts=entry_ts, exit_ts=ts, side="long", entry_price=entry_price, exit_price=close,
                        exit_reason="eod", mfe=best_price_long - entry_price, mae=entry_price - worst_price_long,
                        be_activated=be_activated, be_gate_expired=be_gate_expired,
                        profit_lock_activated=profit_lock_activated, bars_in_trade=bars_in_trade, entry_mode=entry_mode))
                    in_long = False
                if in_short and entry_ts is not None:
                    trades.append(_record(
                        entry_ts=entry_ts, exit_ts=ts, side="short", entry_price=entry_price, exit_price=close,
                        exit_reason="eod", mfe=entry_price - best_price_short, mae=worst_price_short - entry_price,
                        be_activated=be_activated, be_gate_expired=be_gate_expired,
                        profit_lock_activated=profit_lock_activated, bars_in_trade=bars_in_trade, entry_mode=entry_mode))
                    in_short = False
                continue

            if not in_long and not in_short:
                if session_entry_count >= MAX_ENTRIES_PER_SESSION:
                    continue
                bias = int(htf_bias.loc[ts]) if htf_bias is not None and ts in htf_bias.index else 0
                if breakout_direction is None:
                    if allow_long_today and not (htf_filter and bias == -1) and close > long_breakout_level:
                        breakout_direction = "up"
                        breakout_bar_count = cont_count = fail_count = 0
                    elif allow_short_today and not (htf_filter and bias == 1) and close < short_breakout_level:
                        breakout_direction = "down"
                        breakout_bar_count = cont_count = fail_count = 0

                if breakout_direction == "up":
                    breakout_bar_count += 1
                    cont_count = cont_count + 1 if allow_long_today and not (htf_filter and bias == -1) and close >= upside_cont_level else 0
                    fail_count = fail_count + 1 if allow_short_today and not (htf_filter and bias == 1) and close <= upside_fail_level else 0
                    if cont_count >= confirm_bars:
                        in_long, entry_mode = True, "continuation"
                    elif fail_count >= confirm_bars:
                        in_short, entry_mode = True, "reversal"
                    elif breakout_bar_count >= decision_window_bars:
                        breakout_direction = None
                        continue
                elif breakout_direction == "down":
                    breakout_bar_count += 1
                    cont_count = cont_count + 1 if allow_short_today and not (htf_filter and bias == 1) and close <= downside_cont_level else 0
                    fail_count = fail_count + 1 if allow_long_today and not (htf_filter and bias == -1) and close >= downside_fail_level else 0
                    if cont_count >= confirm_bars:
                        in_short, entry_mode = True, "continuation"
                    elif fail_count >= confirm_bars:
                        in_long, entry_mode = True, "reversal"
                    elif breakout_bar_count >= decision_window_bars:
                        breakout_direction = None
                        continue

                if in_long or in_short:
                    entry_price = close
                    entry_ts = ts
                    best_price_long = close
                    best_price_short = close
                    worst_price_long = close
                    worst_price_short = close
                    bars_in_trade = 1
                    be_activated = be_gate_expired = profit_lock_activated = False
                    profit_lock_price = None
                    session_entry_count += 1
                    breakout_direction = None
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
                if not profit_lock_activated and gate_active and unrealised >= profit_lock_trigger_pts:
                    profit_lock_activated, profit_lock_price = True, entry_price + profit_lock_pts
                trail_sl = best_price_long * (1.0 - trailing_pct)
                effective_sl = max(long_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = max(effective_sl, entry_price)
                if profit_lock_activated and profit_lock_price is not None:
                    effective_sl = max(effective_sl, profit_lock_price)
                if close <= effective_sl or close >= long_target:
                    exit_reason = "target" if close >= long_target else "stop"
                    if exit_reason == "stop":
                        if profit_lock_activated and profit_lock_price is not None and close <= profit_lock_price * 1.001:
                            exit_reason = "profit_lock_stop"
                        elif be_activated and gate_active and close <= entry_price * 1.001:
                            exit_reason = "be_stop"
                    trades.append(_record(
                        entry_ts=entry_ts, exit_ts=ts, side="long", entry_price=entry_price, exit_price=close,
                        exit_reason=exit_reason, mfe=best_price_long - entry_price, mae=entry_price - worst_price_long,
                        be_activated=be_activated, be_gate_expired=be_gate_expired,
                        profit_lock_activated=profit_lock_activated, bars_in_trade=bars_in_trade, entry_mode=entry_mode))
                    in_long = False
                    entry_ts = None
                    continue

            if in_short:
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)
                unrealised = entry_price - close
                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if not profit_lock_activated and gate_active and unrealised >= profit_lock_trigger_pts:
                    profit_lock_activated, profit_lock_price = True, entry_price - profit_lock_pts
                trail_sl = best_price_short * (1.0 + trailing_pct)
                effective_sl = min(short_initial_stop, trail_sl)
                if be_activated and gate_active:
                    effective_sl = min(effective_sl, entry_price)
                if profit_lock_activated and profit_lock_price is not None:
                    effective_sl = min(effective_sl, profit_lock_price)
                if close >= effective_sl or close <= short_target:
                    exit_reason = "target" if close <= short_target else "stop"
                    if exit_reason == "stop":
                        if profit_lock_activated and profit_lock_price is not None and close >= profit_lock_price * 0.999:
                            exit_reason = "profit_lock_stop"
                        elif be_activated and gate_active and close >= entry_price * 0.999:
                            exit_reason = "be_stop"
                    trades.append(_record(
                        entry_ts=entry_ts, exit_ts=ts, side="short", entry_price=entry_price, exit_price=close,
                        exit_reason=exit_reason, mfe=entry_price - best_price_short, mae=worst_price_short - entry_price,
                        be_activated=be_activated, be_gate_expired=be_gate_expired,
                        profit_lock_activated=profit_lock_activated, bars_in_trade=bars_in_trade, entry_mode=entry_mode))
                    in_short = False
                    entry_ts = None
                    continue

        up_day_streak = up_day_streak + 1 if day_close > day_open else 0
    return trades


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    data_path = Path(args.intraday_csv)
    output_path = Path(args.output_path)
    df = load_csv_5m(data_path)
    baseline_trades = simulate_orb_v25_profit_lock(
        df, BASE_PARAMS, be_trigger_mult=BASE_BE_TRIGGER, be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER, profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL)
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_annual = [{"year": year, **compute_metrics([t for t in baseline_trades if t.entry_time.year == year])}
                       for year in sorted({t.entry_time.year for t in baseline_trades})]
    folds = build_time_folds(df, fold_count=4)
    baseline_fold_metrics = compute_fold_metrics(baseline_trades, folds)

    variant_rows = []
    for continuation_progress_mult in CONTINUATION_PROGRESS_MULTS:
        for reversal_depth_mult in REVERSAL_DEPTH_MULTS:
            for confirm_bars in CONFIRM_BARS_OPTIONS:
                for decision_window_bars in DECISION_WINDOW_BARS:
                    label = (
                        f"opening_classifier_cont_{continuation_progress_mult:.3f}x"
                        f"_fail_{reversal_depth_mult:.2f}x_confirm_{confirm_bars}bar"
                        f"_window_{decision_window_bars}bar"
                    )
                    variant_trades = simulate_orb_v26_opening_regime_classifier(
                        df, BASE_PARAMS, continuation_progress_mult=continuation_progress_mult,
                        reversal_depth_mult=reversal_depth_mult, confirm_bars=confirm_bars,
                        decision_window_bars=decision_window_bars)
                    metrics = compute_metrics_extended(variant_trades)
                    annual_metrics = compute_annual_metrics(variant_trades)
                    fold_metrics = compute_fold_metrics(variant_trades, folds)
                    path_impact = analyze_path_impact(baseline_trades, variant_trades)
                    variant_rows.append({
                        "label": label,
                        "continuation_progress_mult": continuation_progress_mult,
                        "reversal_depth_mult": reversal_depth_mult,
                        "confirm_bars": confirm_bars,
                        "decision_window_bars": decision_window_bars,
                        "metrics": metrics,
                        "annual_metrics": annual_metrics,
                        "time_folds": fold_metrics,
                        "path_impact": path_impact,
                        "pnl_delta": round(metrics["total_pnl"] - baseline_metrics["total_pnl"], 4),
                        "trade_count_delta": metrics["trades"] - baseline_metrics["trades"],
                        "improved_time_folds": sum(1 for b, v in zip(baseline_fold_metrics, fold_metrics) if v["total_pnl"] > b["total_pnl"]),
                        "positive_time_folds": sum(1 for fold in fold_metrics if fold["total_pnl"] > 0.0),
                        "improved_years": sum(1 for b, v in zip(baseline_annual, annual_metrics) if v["total_pnl"] > b["total_pnl"]),
                        "positive_years": sum(1 for row in annual_metrics if row["total_pnl"] > 0.0),
                    })

    variant_rows.sort(key=_variant_sort_key, reverse=True)
    summary = choose_candidate_summary(variant_rows)
    payload = {
        "research_scope": "local_orb_v26_opening_regime_classifier_alpaca",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v26-profit-lock",
        "local_data_policy": "alpaca_reference_lane_only_qc_promotion_unchanged",
        "data": {"source": data_path.name, "path": str(data_path), "bars": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max())},
        "method": "Alpaca-backed local feasibility test for an ORB opening-regime classifier on top of v26 exits. Instead of entering on the first breakout, the branch waits to classify that breakout as continuation or failure, then enters accordingly.",
        "baseline": {"label": "v26_immediate_breakout_entry_with_profit_lock", "metrics": baseline_metrics, "annual_metrics": baseline_annual, "time_folds": baseline_fold_metrics},
        "variants": variant_rows,
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote opening regime classifier analysis to {output_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
