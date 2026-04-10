#!/usr/bin/env python3
"""
Alpaca-backed local research for ORB-range-denominated trailing stop.

Why this exists:
- v26-profit-lock is the official baseline.
- Six post-v26 exit-side mechanisms have been tested and rejected.
- In v26, every protection level is denominated in ORB range EXCEPT the
  trailing stop, which uses a fixed 1.3% of price.
- On volatile days (wide ORB), 1.3% ≈ 1.36x ORB range — tight.
- On calm days (narrow ORB), 1.3% ≈ 4.76x ORB range — nearly inert.
- This inconsistency means the trailing stop never adapts to the day's actual
  volatility structure.

Mechanism tested here:
- Keep the full v26 baseline unchanged (BE, profit-lock).
- Replace `trail_stop = best_price * (1 - 0.013)` with
  `trail_stop = best_price - K * orb_range`.
- K is the trail multiple, tested at: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0].
- Median neutral K is ~2.6 (equivalent to 1.3% on a median-ORB day).

Anti-overfitting design:
- Only 6 variants (vs 48+ in prior branches).
- Theoretically motivated: normalises the one inconsistent stop component.
- No new interaction parameters.
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
OUTPUT_PATH = RESULTS_DIR / "local_orb_v26_orb_trail_alpaca.json"
ANALYSIS_VERSION = "v1_alpaca_v26_orb_range_trail"

BASE_V26_PROFIT_LOCK_TRIGGER = 1.50
BASE_V26_PROFIT_LOCK_LEVEL = 0.25

# --- Grid: deliberately tiny ---
TRAIL_ORB_MULTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


@dataclass
class OrbTrailTradeRecord:
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
    day_orb_range: float
    pct_equiv_trail: float  # what % the ORB trail was on this day


def compute_metrics_extended(trades: list[OrbTrailTradeRecord]) -> dict:
    metrics = compute_metrics(trades)
    if trades:
        metrics["avg_pct_equiv_trail"] = round(
            sum(t.pct_equiv_trail for t in trades) / len(trades), 6
        )
    else:
        metrics["avg_pct_equiv_trail"] = 0.0
    return metrics


def compute_annual_metrics(trades: list[OrbTrailTradeRecord]) -> list[dict]:
    rows = []
    for year in sorted({t.entry_time.year for t in trades}):
        subset = [t for t in trades if t.entry_time.year == year]
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


def simulate_orb_v26_orb_trail(
    df: pd.DataFrame,
    params: dict,
    *,
    trail_orb_mult: float,
) -> list[OrbTrailTradeRecord]:
    """Full v26 simulation with ORB-range-denominated trailing stop."""
    orb_bars = int(params["orb_bars"])
    profit_ratio = float(params["profit_ratio"])
    close_before_min = int(params["close_before_min"])
    breakout_pct = float(params["breakout_confirm_pct"])
    entry_delay_bars = int(params["entry_delay_bars"])
    htf_filter = bool(params["htf_filter"])
    htf_mode = str(params["htf_mode"])
    htf_ema_fast = int(params["htf_ema_fast"])
    htf_ema_slow = int(params["htf_ema_slow"])
    skip_short_after_up_days = int(params["skip_short_after_up_days"])
    skip_long_after_up_days = int(params["skip_long_after_up_days"])
    initial_stop_mult = float(params["initial_stop_mult"])

    htf_bias = compute_htf_bias(df, htf_ema_fast, htf_ema_slow, htf_mode) if htf_filter else None

    trades: list[OrbTrailTradeRecord] = []
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

        # ORB-range trail distance in $ terms
        trail_distance = trail_orb_mult * range_width
        # What % of price this corresponds to (for reporting)
        pct_equiv = trail_distance / mid_price if mid_price > 0 else 0.013

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

            # --- EOD flatten ---
            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    trades.append(
                        OrbTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="long",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="eod",
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            day_orb_range=range_width,
                            pct_equiv_trail=pct_equiv,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    trades.append(
                        OrbTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="short",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="eod",
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            day_orb_range=range_width,
                            pct_equiv_trail=pct_equiv,
                        )
                    )
                    in_short = False
                continue

            # --- Entry logic ---
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

            # --- Long management ---
            if in_long:
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)
                unrealised = close - entry_price

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if not profit_lock_activated and unrealised >= profit_lock_trigger_pts:
                    profit_lock_activated = True
                    profit_lock_price = entry_price + profit_lock_pts

                # ORB-range-based trailing stop
                trail_sl = best_price_long - trail_distance
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
                        OrbTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="long",
                                entry_price=entry_price, exit_price=close,
                                exit_reason=exit_reason,
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            day_orb_range=range_width,
                            pct_equiv_trail=pct_equiv,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

                if close >= tp_long:
                    trades.append(
                        OrbTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="long",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="target",
                                mfe=best_price_long - entry_price,
                                mae=entry_price - worst_price_long,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            day_orb_range=range_width,
                            pct_equiv_trail=pct_equiv,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            # --- Short management ---
            if in_short:
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)
                unrealised = entry_price - close

                if not be_activated and gate_active and unrealised >= be_trigger_pts:
                    be_activated = True
                if not profit_lock_activated and unrealised >= profit_lock_trigger_pts:
                    profit_lock_activated = True
                    profit_lock_price = entry_price - profit_lock_pts

                # ORB-range-based trailing stop
                trail_sl = best_price_short + trail_distance
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
                        OrbTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="short",
                                entry_price=entry_price, exit_price=close,
                                exit_reason=exit_reason,
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            day_orb_range=range_width,
                            pct_equiv_trail=pct_equiv,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

                if close <= tp_short:
                    trades.append(
                        OrbTrailTradeRecord(
                            **_make_trade_record_dict(
                                entry_ts=entry_ts, exit_ts=ts, side="short",
                                entry_price=entry_price, exit_price=close,
                                exit_reason="target",
                                mfe=entry_price - best_price_short,
                                mae=worst_price_short - entry_price,
                                be_activated=be_activated,
                                be_gate_expired=be_gate_expired,
                                profit_lock_activated=profit_lock_activated,
                                bars_in_trade=bars_in_trade,
                            ),
                            day_orb_range=range_width,
                            pct_equiv_trail=pct_equiv,
                        )
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


# ---------- candidate selection ----------

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
        "trail_orb_mult": row["trail_orb_mult"],
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
        "avg_pct_equiv_trail": metrics.get("avg_pct_equiv_trail", 0),
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
        verdict = "READY_FOR_QC_PROXY"
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


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    data_path = Path(args.intraday_csv)
    output_path = Path(args.output_path)

    print("Loading local data for v26 ORB-range trail research...")
    intraday = load_csv_5m(data_path)

    # --- Baseline ---
    baseline_trades = simulate_orb_v25_profit_lock(
        intraday,
        BASE_PARAMS,
        be_trigger_mult=BASE_BE_TRIGGER,
        be_gate_minutes=BASE_BE_GATE_MIN,
        profit_lock_trigger_mult=BASE_V26_PROFIT_LOCK_TRIGGER,
        profit_lock_level_mult=BASE_V26_PROFIT_LOCK_LEVEL,
    )
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_years_raw = {}
    for year in sorted({t.entry_time.year for t in baseline_trades}):
        subset = [t for t in baseline_trades if t.entry_time.year == year]
        baseline_years_raw[year] = compute_metrics(subset)
    baseline_years = [{"year": y, **m} for y, m in baseline_years_raw.items()]

    time_folds = build_time_folds(intraday, fold_count=4)
    baseline_folds = compute_fold_metrics(baseline_trades, time_folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_folds}

    print(
        "Alpaca baseline v26 (pct trail): trades=%d, PnL=%+.2f, PF=%.3f, "
        "positiveYears=%d, positiveFolds=%d"
        % (
            baseline_metrics["trades"],
            baseline_metrics["total_pnl"],
            baseline_metrics["profit_factor"],
            sum(1 for row in baseline_years if row["total_pnl"] > 0),
            sum(1 for row in baseline_folds if row["total_pnl"] > 0),
        )
    )

    # --- Grid sweep ---
    all_results = []
    for trail_mult in TRAIL_ORB_MULTS:
        label = "orb_trail_%.1fx" % trail_mult
        print("Running %s..." % label)
        variant_trades = simulate_orb_v26_orb_trail(
            intraday,
            BASE_PARAMS,
            trail_orb_mult=trail_mult,
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
            if row["total_pnl"] > baseline_years_raw.get(year, {"total_pnl": float("-inf")})["total_pnl"]
        )
        positive_years = sum(1 for row in variant_years if row["total_pnl"] > 0)
        pnl_delta = round(variant_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)

        result = {
            "label": label,
            "trail_orb_mult": trail_mult,
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
            "  delta=%+.2f, PF=%.3f, folds=%d/%d, years=%d/%d, "
            "saved=%d, clipped=%d, stops=%d, eod=%d, avgPctEquiv=%.4f"
            % (
                pnl_delta,
                variant_metrics["profit_factor"],
                improved_time_folds, len(time_folds),
                improved_years, len(baseline_years),
                path_impact["saved_losses"], path_impact["clipped_winners"],
                variant_metrics["stop_exits"], variant_metrics["eod_exits"],
                variant_metrics.get("avg_pct_equiv_trail", 0),
            )
        )

    candidate_summary = choose_candidate_summary(all_results)
    all_results.sort(key=_variant_sort_key, reverse=True)

    payload = {
        "research_scope": "local_orb_v26_orb_trail_alpaca",
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
            "Alpaca-backed local feasibility test for ORB-range-denominated trailing stop. "
            "Replaces the fixed 1.3% trailing stop with trail_stop = best_price - K * orb_range. "
            "The median-neutral K is ~2.6 (equivalent to 1.3% on a median-ORB day). "
            "This normalises the trailing stop to each day's actual volatility structure."
        ),
        "structural_motivation": (
            "In v26, every protection level is ORB-range denominated except the trailing stop. "
            "The 1.3% trail translates to 1.36x ORB range on volatile days (p10) vs 4.76x on calm days (p90). "
            "ORB-range denomination removes this inconsistency."
        ),
        "baseline": {
            "label": "v26_trail=1.3pct_BE=1.25x_gate=180min_profitLock=0.25x_after_1.50x",
            "metrics": baseline_metrics,
            "annual_metrics": baseline_years,
            "time_folds": baseline_folds,
        },
        "grid": {
            "trail_orb_mults": TRAIL_ORB_MULTS,
            "total_variants": len(all_results),
        },
        "all_variants": all_results,
        "candidate_summary": candidate_summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nWrote %d variants to %s" % (len(all_results), output_path))
    print("Verdict: %s" % candidate_summary["verdict"])
    if candidate_summary["best_positive"]:
        bp = candidate_summary["best_positive"]
        print(
            "Best positive: %s — delta=%+.2f, folds=%d/%d, years=%d/%d, clipped=%d"
            % (bp["label"], bp["pnl_delta"], bp["improved_time_folds"],
               len(time_folds), bp["improved_years"], len(baseline_years),
               bp["clipped_winners"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
