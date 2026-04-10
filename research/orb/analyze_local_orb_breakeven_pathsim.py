#!/usr/bin/env python3
"""
Path-level breakeven stop simulation on local 5-minute data.

This script addresses the primary research gap identified in iteration 83:
the breakeven hypothesis has only been validated via trade-level proxies
(MFE/Drawdown columns from QC trades) but never via actual bar-path simulation.

This script:
1. Runs the baseline ORB engine bar-by-bar on qqq_5m.csv
2. Adds a breakeven stop mechanism at various trigger thresholds
3. Tracks per-trade bar-level MFE/MAE/exit paths
4. Evaluates on walk-forward test folds
5. Reports net PnL delta, walk-forward stability, and per-trade path statistics

The breakeven mechanism:
- Once unrealised profit (from entry_price) reaches `BE_trigger × ORB_range`,
  the stop-loss is raised to entry_price (breakeven)
- If price subsequently retraces to entry_price, the trade exits at ~0 PnL (- fees)
- If price continues, the normal trailing stop / target / EOD exits still apply

IMPORTANT: This version enforces max_entries_per_session=1, matching the
production QC code. Previous versions did not enforce this constraint, which
inflated the apparent benefit of breakeven stops by allowing re-entries.

Note: this is still limited to the local qqq_5m.csv data window (2026 Q1),
so results should be treated as short-sample path-level evidence.
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
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_breakeven_pathsim.json"

INITIAL_CASH = 100_000.0
FEE_PCT = 0.0005
MAX_ENTRIES_PER_SESSION = 1  # Must match QC production code

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

# Breakeven trigger thresholds as multiples of ORB range width
BE_TRIGGERS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]


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
    mfe: float  # maximum favourable excursion in price points
    mae: float  # maximum adverse excursion in price points (positive = bigger loss)
    be_activated: bool  # whether breakeven stop was activated during this trade
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


def simulate_orb_with_breakeven(
    df: pd.DataFrame, params: dict, be_trigger_mult: float
) -> list[TradeRecord]:
    """
    Simulate ORB strategy with a breakeven stop mechanism.

    be_trigger_mult: once unrealised profit >= be_trigger_mult * orb_range,
                     move the stop to breakeven (entry price).
                     If 0.0, this is the baseline (no breakeven stop).

    IMPORTANT: This enforces max_entries_per_session=1, matching QC production.
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

        # Breakeven trigger thresholds in price points
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
        entry_ts: pd.Timestamp | None = None
        best_price_long = 0.0
        best_price_short = float("inf")
        worst_price_long = float("inf")
        worst_price_short = 0.0
        bars_in_trade = 0
        be_activated = False
        session_entry_count = 0  # Track entries per session (QC constraint)

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    trades.append(
                        _make_trade_record(
                            entry_ts, ts, "long", entry_price, close, "eod",
                            mfe, mae, be_activated, bars_in_trade,
                        )
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    trades.append(
                        _make_trade_record(
                            entry_ts, ts, "short", entry_price, close, "eod",
                            mfe, mae, be_activated, bars_in_trade,
                        )
                    )
                    in_short = False
                continue

            if not in_long and not in_short:
                # Enforce max_entries_per_session=1
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
                    session_entry_count += 1
                    continue

            if in_long and entry_ts is not None:
                bars_in_trade += 1
                best_price_long = max(best_price_long, close)
                worst_price_long = min(worst_price_long, close)

                # Check breakeven activation
                if use_breakeven and not be_activated:
                    unrealised = close - entry_price
                    if unrealised >= be_trigger_pts:
                        be_activated = True

                # Compute effective stop
                trail_sl = best_price_long * (1.0 - trailing_pct)
                if be_activated:
                    # Breakeven stop = entry price (or trailing, whichever is higher)
                    effective_sl = max(entry_price, trail_sl)
                else:
                    effective_sl = max(long_initial_stop, trail_sl)

                if close <= effective_sl:
                    mfe = best_price_long - entry_price
                    mae = entry_price - worst_price_long
                    exit_reason = "be_stop" if be_activated and close <= entry_price * 1.001 else "stop"
                    trades.append(
                        _make_trade_record(
                            entry_ts, ts, "long", entry_price, close, exit_reason,
                            mfe, mae, be_activated, bars_in_trade,
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
                            mfe, mae, be_activated, bars_in_trade,
                        )
                    )
                    in_long = False
                    entry_ts = None
                    continue

            if in_short and entry_ts is not None:
                bars_in_trade += 1
                best_price_short = min(best_price_short, close)
                worst_price_short = max(worst_price_short, close)

                # Check breakeven activation
                if use_breakeven and not be_activated:
                    unrealised = entry_price - close
                    if unrealised >= be_trigger_pts:
                        be_activated = True

                # Compute effective stop
                trail_sl = best_price_short * (1.0 + trailing_pct)
                if be_activated:
                    effective_sl = min(entry_price, trail_sl)
                else:
                    effective_sl = min(short_initial_stop, trail_sl)

                if close >= effective_sl:
                    mfe = entry_price - best_price_short
                    mae = worst_price_short - entry_price
                    exit_reason = "be_stop" if be_activated and close >= entry_price * 0.999 else "stop"
                    trades.append(
                        _make_trade_record(
                            entry_ts, ts, "short", entry_price, close, exit_reason,
                            mfe, mae, be_activated, bars_in_trade,
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
                            mfe, mae, be_activated, bars_in_trade,
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
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    side: str,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    mfe: float,
    mae: float,
    be_activated: bool,
    bars_in_trade: int,
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
        bars_in_trade=bars_in_trade,
    )


def compute_metrics(trades: list[TradeRecord]) -> dict:
    if not trades:
        return {
            "trades": 0, "total_pnl": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "trade_sharpe": 0.0, "avg_trade_pnl": 0.0, "max_drawdown_pct": 0.0,
            "stop_exits": 0, "be_stop_exits": 0, "target_exits": 0, "eod_exits": 0,
            "be_activated_count": 0, "avg_mfe": 0.0, "avg_mae": 0.0,
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
        rows.append(
            {
                "fold": fold["fold"],
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                **metrics,
            }
        )
    return rows


def analyze_path_level_be_impact(
    baseline_trades: list[TradeRecord],
    be_trades: list[TradeRecord],
    be_trigger: float,
) -> dict:
    """Analyze per-trade changes between baseline and breakeven variant."""
    base_by_entry = {t.entry_time: t for t in baseline_trades}
    be_by_entry = {t.entry_time: t for t in be_trades}

    common_entries = set(base_by_entry.keys()) & set(be_by_entry.keys())
    only_base = set(base_by_entry.keys()) - set(be_by_entry.keys())
    only_be = set(be_by_entry.keys()) - set(base_by_entry.keys())

    saved_losses = 0
    clipped_winners = 0
    salvage_amount = 0.0
    harm_amount = 0.0
    unchanged = 0

    for entry_ts in common_entries:
        base_t = base_by_entry[entry_ts]
        be_t = be_by_entry[entry_ts]

        if abs(base_t.pnl - be_t.pnl) < 0.01:
            unchanged += 1
        elif base_t.pnl < 0 and be_t.pnl > base_t.pnl:
            saved_losses += 1
            salvage_amount += (be_t.pnl - base_t.pnl)
        elif base_t.pnl > 0 and be_t.pnl < base_t.pnl:
            clipped_winners += 1
            harm_amount += (base_t.pnl - be_t.pnl)

    return {
        "be_trigger": be_trigger,
        "common_trades": len(common_entries),
        "only_in_baseline": len(only_base),
        "only_in_be_variant": len(only_be),
        "unchanged": unchanged,
        "saved_losses": saved_losses,
        "salvage_amount": round(salvage_amount, 4),
        "clipped_winners": clipped_winners,
        "harm_amount": round(harm_amount, 4),
        "net_impact": round(salvage_amount - harm_amount, 4),
    }


def main() -> int:
    print("Loading 5m data...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"Enforcing max_entries_per_session={MAX_ENTRIES_PER_SESSION}")

    # Run baseline first
    print("\nRunning baseline (no breakeven)...")
    baseline_trades = simulate_orb_with_breakeven(df, BASE_PARAMS, be_trigger_mult=0.0)
    baseline_metrics = compute_metrics(baseline_trades)
    baseline_folds = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {r["fold"]: r for r in baseline_folds}

    print(f"Baseline: {baseline_metrics['trades']} trades, PnL={baseline_metrics['total_pnl']:+.2f}, "
          f"WR={baseline_metrics['win_rate_pct']:.1f}%, PF={baseline_metrics['profit_factor']:.3f}")

    # Run each breakeven trigger variant
    results = []
    for trigger in BE_TRIGGERS:
        if trigger == 0.0:
            positive_folds = sum(1 for r in baseline_folds if r["total_pnl"] > 0)
            results.append({
                "be_trigger_mult": trigger,
                "label": "baseline",
                "metrics": baseline_metrics,
                "fold_results": baseline_folds,
                "positive_test_folds": positive_folds,
                "improved_vs_baseline_folds": 0,
                "path_impact": None,
            })
            continue

        print(f"\nRunning BE={trigger:.2f}x...")
        be_trades = simulate_orb_with_breakeven(df, BASE_PARAMS, be_trigger_mult=trigger)
        be_metrics = compute_metrics(be_trades)
        be_folds = compute_fold_results(be_trades, folds)

        positive_folds = sum(1 for r in be_folds if r["total_pnl"] > 0)
        improved_folds = sum(
            1 for r in be_folds
            if r["total_pnl"] > baseline_fold_map[r["fold"]]["total_pnl"]
        )

        path_impact = analyze_path_level_be_impact(baseline_trades, be_trades, trigger)

        pnl_delta = round(be_metrics["total_pnl"] - baseline_metrics["total_pnl"], 4)
        print(f"  Trades={be_metrics['trades']}, PnL={be_metrics['total_pnl']:+.2f} "
              f"(delta={pnl_delta:+.2f}), WR={be_metrics['win_rate_pct']:.1f}%, "
              f"PF={be_metrics['profit_factor']:.3f}, "
              f"BE_stops={be_metrics['be_stop_exits']}, "
              f"folds={improved_folds}/{len(folds)}")
        print(f"  Path impact: saved={path_impact['saved_losses']}, "
              f"clipped={path_impact['clipped_winners']}, "
              f"net={path_impact['net_impact']:+.2f}")

        results.append({
            "be_trigger_mult": trigger,
            "label": f"BE={trigger:.2f}x",
            "metrics": be_metrics,
            "fold_results": be_folds,
            "positive_test_folds": positive_folds,
            "improved_vs_baseline_folds": improved_folds,
            "path_impact": path_impact,
        })

    # Find the best variant
    candidates = [r for r in results if r["be_trigger_mult"] > 0]
    best = max(
        candidates,
        key=lambda r: (
            r["improved_vs_baseline_folds"],
            r["positive_test_folds"],
            r["metrics"]["total_pnl"],
        ),
    )

    # Determine if any variant passes the promotion bar
    best_passes = (
        best["improved_vs_baseline_folds"] >= 3
        and best["metrics"]["total_pnl"] > baseline_metrics["total_pnl"]
        and best["metrics"]["profit_factor"] >= baseline_metrics["profit_factor"]
    )

    payload = {
        "research_scope": "local_orb_breakeven_path_simulation",
        "version": "v2_with_max_entries_enforcement",
        "data": {
            "source": str(DATA_PATH.name),
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Full bar-by-bar simulation of breakeven stop mechanism on local 5m data. "
            "Unlike trade-level proxies, this actually implements the BE stop in the "
            "simulation engine and tracks exact exit paths. "
            "This version enforces max_entries_per_session=1, matching QC production."
        ),
        "constraints": {
            "max_entries_per_session": MAX_ENTRIES_PER_SESSION,
            "note": "Matches QC production code. No re-entries after BE exit."
        },
        "baseline": {
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "variants": results,
        "best_variant": {
            "label": best["label"],
            "be_trigger_mult": best["be_trigger_mult"],
            "pnl_delta": round(best["metrics"]["total_pnl"] - baseline_metrics["total_pnl"], 4),
            "improved_folds": best["improved_vs_baseline_folds"],
            "positive_folds": best["positive_test_folds"],
            "passes_promotion_bar": best_passes,
        },
        "conclusion": {
            "path_simulation_completed": True,
            "any_variant_passes": best_passes,
            "limitation": (
                "Local data covers only 2026-01 to 2026-04. "
                "Even a strong path-level result here should be cross-checked "
                "against the QC trade-level tradeoff proxy over 2017-2026."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Saved path-level simulation to {OUTPUT_PATH}")
    print(f"\nSummary table:")
    print(f"{'Trigger':<12} {'PnL':>10} {'Delta':>8} {'PF':>7} {'WR%':>7} {'Folds':>6} {'BE Stops':>9} {'Net Impact':>11}")
    print("-" * 76)
    for r in results:
        m = r["metrics"]
        delta = m["total_pnl"] - baseline_metrics["total_pnl"]
        net_imp = r["path_impact"]["net_impact"] if r["path_impact"] else 0.0
        print(f"{r['label']:<12} {m['total_pnl']:>+10.2f} {delta:>+8.2f} {m['profit_factor']:>7.3f} "
              f"{m['win_rate_pct']:>6.1f}% {r['improved_vs_baseline_folds']:>3}/4   "
              f"{m['be_stop_exits']:>5}     {net_imp:>+10.2f}")

    print(f"\nBest variant: {best['label']}")
    print(f"  PnL delta: {best['metrics']['total_pnl'] - baseline_metrics['total_pnl']:+.4f}")
    print(f"  Improved folds: {best['improved_vs_baseline_folds']}/{len(folds)}")
    print(f"  Passes promotion bar: {best_passes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
