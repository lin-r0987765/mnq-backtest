#!/usr/bin/env python3
"""
Local path-level research for a post-breakeven profit-lock mechanism on top of v25.

Why this exists:
- v25-timegated-be is now the official baseline.
- Real QC still shows a large concentration of EOD flatten exits.
- The next genuinely new exit-side branch should target late-session giveback
  without reviving rejected wide-trail / no-trail / time-decay branches.

Mechanism tested here:
- Keep the v25 baseline time-gated breakeven:
  - breakeven trigger = 1.25 x ORB range
  - breakeven active only during first 180 minutes after entry
- Add a new optional profit-lock ratchet:
  - if unrealised profit reaches `profit_lock_trigger x ORB range`
    during the first 180 minutes,
  - raise the stop to `entry +/- profit_lock_level x ORB range`
  - unlike the BE gate, the profit-lock persists for the rest of the trade
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "qqq_5m.csv"
WALK_FORWARD_PATH = PROJECT_ROOT / "walk_forward_results.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_v25_profit_lock.json"
ANALYSIS_VERSION = "v1_v25_profit_lock_persistent_after_gate"

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

BASE_BE_TRIGGER = 1.25
BASE_BE_GATE_MIN = 180

PROFIT_LOCK_TRIGGERS = [1.50, 1.75, 2.00, 2.25]
PROFIT_LOCK_LEVELS = [0.25, 0.50, 0.75, 1.00]


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
    profit_lock_activated: bool
    bars_in_trade: int


def load_csv_5m(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().lower()
    if first_line.startswith("datetime,"):
        df = pd.read_csv(path, index_col=0)
    else:
        try:
            df = pd.read_csv(path, header=[0, 1], index_col=0)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            else:
                raise ValueError("single-level header")
        except Exception:
            df = pd.read_csv(path, index_col=0)

    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in {"open", "high", "low", "close", "volume"}:
            rename_map[column] = normalized.title()
    df = df.rename(columns=rename_map)

    df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert("America/New_York")
    df = df[df.index.notna()].copy()
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
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


def simulate_orb_v25_profit_lock(
    df: pd.DataFrame,
    params: dict,
    *,
    be_trigger_mult: float,
    be_gate_minutes: int,
    profit_lock_trigger_mult: float | None,
    profit_lock_level_mult: float | None,
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

    use_profit_lock = (
        profit_lock_trigger_mult is not None
        and profit_lock_level_mult is not None
        and profit_lock_trigger_mult > 0.0
        and profit_lock_level_mult > 0.0
    )

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

        be_trigger_pts = be_trigger_mult * range_width
        profit_lock_trigger_pts = (
            profit_lock_trigger_mult * range_width if use_profit_lock else None
        )
        profit_lock_pts = profit_lock_level_mult * range_width if use_profit_lock else None

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
                    trades.append(
                        _make_trade_record(
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
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    trades.append(
                        _make_trade_record(
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
            gate_active = elapsed_min <= be_gate_minutes
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
                    use_profit_lock
                    and not profit_lock_activated
                    and gate_active
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price + float(profit_lock_pts)

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
                    trades.append(
                        _make_trade_record(
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
                    )
                    in_long = False
                    entry_ts = None
                    continue

                if close >= tp_long:
                    trades.append(
                        _make_trade_record(
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
                    use_profit_lock
                    and not profit_lock_activated
                    and gate_active
                    and unrealised >= profit_lock_trigger_pts
                ):
                    profit_lock_activated = True
                    profit_lock_price = entry_price - float(profit_lock_pts)

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
                    trades.append(
                        _make_trade_record(
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
                    )
                    in_short = False
                    entry_ts = None
                    continue

                if close <= tp_short:
                    trades.append(
                        _make_trade_record(
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
                    )
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def _make_trade_record(
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
        profit_lock_activated=profit_lock_activated,
        bars_in_trade=bars_in_trade,
    )


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
            "stop_exits": 0,
            "be_stop_exits": 0,
            "profit_lock_stop_exits": 0,
            "target_exits": 0,
            "eod_exits": 0,
            "be_activated_count": 0,
            "be_gate_expired_count": 0,
            "profit_lock_activated_count": 0,
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
        "stop_exits": sum(1 for t in trades if t.exit_reason == "stop"),
        "be_stop_exits": sum(1 for t in trades if t.exit_reason == "be_stop"),
        "profit_lock_stop_exits": sum(1 for t in trades if t.exit_reason == "profit_lock_stop"),
        "target_exits": sum(1 for t in trades if t.exit_reason == "target"),
        "eod_exits": sum(1 for t in trades if t.exit_reason == "eod"),
        "be_activated_count": sum(1 for t in trades if t.be_activated),
        "be_gate_expired_count": sum(1 for t in trades if t.be_gate_expired),
        "profit_lock_activated_count": sum(1 for t in trades if t.profit_lock_activated),
        "avg_mfe": round(mean(t.mfe for t in trades), 4),
        "avg_mae": round(mean(t.mae for t in trades), 4),
    }


def compute_fold_results(trades: list[TradeRecord], folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        subset = [t for t in trades if fold["start"] <= t.entry_time.date() <= fold["end"]]
        rows.append(
            {
                "fold": fold["fold"],
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                **compute_metrics(subset),
            }
        )
    return rows


def analyze_path_impact(baseline_trades: list[TradeRecord], variant_trades: list[TradeRecord]) -> dict:
    base_by_entry = {t.entry_time: t for t in baseline_trades}
    var_by_entry = {t.entry_time: t for t in variant_trades}
    common = set(base_by_entry) & set(var_by_entry)

    saved_losses = 0
    clipped_winners = 0
    improved_other = 0
    unchanged = 0
    salvage_amount = 0.0
    harm_amount = 0.0

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
        else:
            harm_amount += abs(delta)

    return {
        "common_trades": len(common),
        "unchanged": unchanged,
        "saved_losses": saved_losses,
        "clipped_winners": clipped_winners,
        "improved_other": improved_other,
        "salvage": round(salvage_amount, 4),
        "harm": round(harm_amount, 4),
        "net_impact": round(salvage_amount - harm_amount, 4),
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


def summarize_variant(result: dict | None) -> dict | None:
    if result is None:
        return None
    metrics = result["metrics"]
    path = result["path_impact"]
    return {
        "label": result["label"],
        "profit_lock_trigger": result["profit_lock_trigger"],
        "profit_lock_level": result["profit_lock_level"],
        "pnl": metrics["total_pnl"],
        "pnl_delta": result["pnl_delta"],
        "profit_factor": metrics["profit_factor"],
        "win_rate_pct": metrics["win_rate_pct"],
        "improved_folds": result["improved_vs_baseline_folds"],
        "positive_test_folds": result["positive_test_folds"],
        "saved_losses": path["saved_losses"],
        "clipped_winners": path["clipped_winners"],
        "net_path_impact": path["net_impact"],
        "profit_lock_stop_exits": metrics["profit_lock_stop_exits"],
        "profit_lock_activated_count": metrics["profit_lock_activated_count"],
    }


def main() -> int:
    print("Loading local data for v25 profit-lock research...")
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}")
    print(f"Baseline: v25 BE={BASE_BE_TRIGGER:.2f}x gate={BASE_BE_GATE_MIN}min")

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
        f"EOD={baseline_metrics['eod_exits']}, "
        f"BE_activated={baseline_metrics['be_activated_count']}"
    )

    all_results = []
    for trigger in PROFIT_LOCK_TRIGGERS:
        for level in PROFIT_LOCK_LEVELS:
            label = f"profit_lock={level:.2f}x_after_{trigger:.2f}x"
            print(f"\nRunning {label}...")
            variant_trades = simulate_orb_v25_profit_lock(
                df,
                BASE_PARAMS,
                be_trigger_mult=BASE_BE_TRIGGER,
                be_gate_minutes=BASE_BE_GATE_MIN,
                profit_lock_trigger_mult=trigger,
                profit_lock_level_mult=level,
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
                f"lockStops={variant_metrics['profit_lock_stop_exits']}, "
                f"lockActs={variant_metrics['profit_lock_activated_count']}, "
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
                    "profit_lock_trigger": trigger,
                    "profit_lock_level": level,
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
    best_local_balanced = max(
        [r for r in all_results if r["improved_vs_baseline_folds"] >= 2],
        key=variant_sort_key,
        default=None,
    )

    best_passes_local_bar = (
        best_overall["pnl_delta"] > 0
        and best_overall["improved_vs_baseline_folds"] >= 3
        and best_overall["metrics"]["profit_factor"] >= baseline_metrics["profit_factor"]
    )

    payload = {
        "research_scope": "local_orb_v25_profit_lock",
        "analysis_version": ANALYSIS_VERSION,
        "baseline_version": "v25-timegated-be",
        "data": {
            "source": DATA_PATH.name,
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "method": (
            "Local path-level simulation of a persistent profit-lock ratchet layered on top "
            "of the v25 time-gated breakeven baseline. Profit lock can only activate during "
            "the first 180 minutes, but once activated its stop persists for the rest of the trade."
        ),
        "baseline": {
            "label": f"v25_BE={BASE_BE_TRIGGER:.2f}x_gate={BASE_BE_GATE_MIN}min",
            "metrics": baseline_metrics,
            "fold_results": baseline_folds,
        },
        "grid": {
            "profit_lock_triggers": PROFIT_LOCK_TRIGGERS,
            "profit_lock_levels": PROFIT_LOCK_LEVELS,
        },
        "all_variants": all_results,
        "best_overall": {
            "label": best_overall["label"],
            "profit_lock_trigger": best_overall["profit_lock_trigger"],
            "profit_lock_level": best_overall["profit_lock_level"],
            "pnl_delta": best_overall["pnl_delta"],
            "improved_folds": best_overall["improved_vs_baseline_folds"],
            "passes_local_bar": best_passes_local_bar,
        },
        "candidate_summary": {
            "best_positive": summarize_variant(best_positive),
            "best_zero_clip_positive": summarize_variant(best_zero_clip_positive),
            "best_balanced": summarize_variant(best_local_balanced),
            "interpretation": (
                "A profit-lock branch is only interesting if it improves full-sample local PnL "
                "without creating large winner clipping. If even the best persistent ratchet "
                "cannot clear that bar, the next exit-side branch should move away from naive "
                "profit-lock thresholds."
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
