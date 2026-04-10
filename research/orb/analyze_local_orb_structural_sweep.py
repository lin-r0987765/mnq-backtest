#!/usr/bin/env python3
"""
Structural research sweep for the local ORB proxy.

This script intentionally does not modify the QuantConnect production files.
It uses local 5-minute data to compare wider initial stops and delayed entries
against the current local ORB baseline, then scores candidates on walk-forward
test folds defined in walk_forward_results.json.
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
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "local_orb_structural_sweep.json"

INITIAL_CASH = 100_000.0
FEE_PCT = 0.0005

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


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    duration_min: float
    exit_reason: str


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
    ema_s = df_1h["Close"].ewm(span=ema_slow, adjust=False).mean()

    bias = pd.Series(0, index=df_1h.index, dtype=int)
    if mode == "slope":
        slope = ema_f.diff()
        bias[slope > 0] = 1
        bias[slope < 0] = -1
    else:
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


def simulate_orb(df: pd.DataFrame, params: dict) -> list[Trade]:
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

    trades: list[Trade] = []
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

        post_orb = sess.iloc[orb_bars + entry_delay_bars :]
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

        for ts, row in post_orb.iterrows():
            close = float(row["Close"])

            if ts >= force_close_ts:
                if in_long and entry_ts is not None:
                    trades.append(
                        make_trade(entry_ts, ts, "long", entry_price, close, "eod")
                    )
                    in_long = False
                if in_short and entry_ts is not None:
                    trades.append(
                        make_trade(entry_ts, ts, "short", entry_price, close, "eod")
                    )
                    in_short = False
                continue

            if not in_long and not in_short:
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
                    continue

            if in_long and entry_ts is not None:
                best_price_long = max(best_price_long, close)
                trail_sl = best_price_long * (1.0 - trailing_pct)
                effective_sl = max(long_initial_stop, trail_sl)
                if close <= effective_sl:
                    trades.append(make_trade(entry_ts, ts, "long", entry_price, close, "stop"))
                    in_long = False
                    entry_ts = None
                    continue
                if close >= tp_long:
                    trades.append(make_trade(entry_ts, ts, "long", entry_price, close, "target"))
                    in_long = False
                    entry_ts = None
                    continue

            if in_short and entry_ts is not None:
                best_price_short = min(best_price_short, close)
                trail_sl = best_price_short * (1.0 + trailing_pct)
                effective_sl = min(short_initial_stop, trail_sl)
                if close >= effective_sl:
                    trades.append(make_trade(entry_ts, ts, "short", entry_price, close, "stop"))
                    in_short = False
                    entry_ts = None
                    continue
                if close <= tp_short:
                    trades.append(make_trade(entry_ts, ts, "short", entry_price, close, "target"))
                    in_short = False
                    entry_ts = None
                    continue

        day_open = float(sess["Open"].iloc[0])
        day_close = float(sess["Close"].iloc[-1])
        up_day_streak = up_day_streak + 1 if day_close > day_open else 0

    return trades


def make_trade(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    side: str,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
) -> Trade:
    gross = exit_price - entry_price if side == "long" else entry_price - exit_price
    fees = (entry_price + exit_price) * FEE_PCT
    pnl = gross - fees
    duration_min = float((exit_ts - entry_ts).total_seconds() / 60.0)
    return Trade(
        entry_time=entry_ts,
        exit_time=exit_ts,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        duration_min=duration_min,
        exit_reason=exit_reason,
    )


def compute_trade_metrics(trades: list[Trade]) -> dict:
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

    stop_trades = [t for t in trades if t.exit_reason == "stop"]
    target_trades = [t for t in trades if t.exit_reason == "target"]
    eod_trades = [t for t in trades if t.exit_reason == "eod"]
    under_2h = [t for t in trades if t.duration_min <= 120.0]
    over_2h = [t for t in trades if t.duration_min > 120.0]

    return {
        "trades": len(trades),
        "total_pnl": round(sum(pnls), 4),
        "return_pct": round(sum(pnls) / INITIAL_CASH * 100.0, 4),
        "win_rate_pct": round((len(wins) / len(pnls) * 100.0) if pnls else 0.0, 4),
        "profit_factor": round((sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0, 4),
        "trade_sharpe": round((mean(pnls) / stdev(pnls)) if len(pnls) > 1 and stdev(pnls) > 0 else 0.0, 4),
        "avg_trade_pnl": round(mean(pnls), 4) if pnls else 0.0,
        "max_drawdown_pct": round(max_dd, 4),
        "stop_exits": len(stop_trades),
        "target_exits": len(target_trades),
        "eod_exits": len(eod_trades),
        "under_2h_trades": len(under_2h),
        "under_2h_pnl": round(sum(t.pnl for t in under_2h), 4),
        "over_2h_trades": len(over_2h),
        "over_2h_pnl": round(sum(t.pnl for t in over_2h), 4),
    }


def compute_fold_results(trades: list[Trade], folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        subset = [
            t for t in trades if fold["start"] <= t.entry_time.date() <= fold["end"]
        ]
        metrics = compute_trade_metrics(subset) if subset else {
            "trades": 0,
            "total_pnl": 0.0,
            "win_rate_pct": 0.0,
            "trade_sharpe": 0.0,
            "profit_factor": 0.0,
            "under_2h_pnl": 0.0,
        }
        rows.append(
            {
                "fold": fold["fold"],
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                "trades": metrics["trades"],
                "total_pnl": round(metrics["total_pnl"], 4),
                "win_rate_pct": round(metrics["win_rate_pct"], 4),
                "trade_sharpe": round(metrics["trade_sharpe"], 4),
                "profit_factor": round(metrics["profit_factor"], 4),
                "under_2h_pnl": round(metrics["under_2h_pnl"], 4),
            }
        )
    return rows


def summarize_candidate(
    name: str,
    params: dict,
    trades: list[Trade],
    folds: list[dict],
    baseline_fold_map: dict[int, dict],
) -> dict:
    metrics = compute_trade_metrics(trades)
    fold_results = compute_fold_results(trades, folds)
    positive_folds = sum(1 for row in fold_results if row["total_pnl"] > 0)
    improved_vs_baseline = sum(
        1
        for row in fold_results
        if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
    )

    return {
        "name": name,
        "params": params,
        "metrics": metrics,
        "fold_results": fold_results,
        "positive_test_folds": positive_folds,
        "improved_vs_baseline_folds": improved_vs_baseline,
    }


def main() -> int:
    df = load_csv_5m(DATA_PATH)
    folds = load_test_folds(WALK_FORWARD_PATH)

    candidates: list[tuple[str, dict]] = []
    for entry_delay in [0, 1, 2, 3]:
        for stop_mult in [1.0, 1.25, 1.5, 1.75, 2.0]:
            params = {**BASE_PARAMS, "entry_delay_bars": entry_delay, "initial_stop_mult": stop_mult}
            name = f"delay_{entry_delay}_stop_{stop_mult:.2f}x"
            candidates.append((name, params))

    baseline_name = "delay_0_stop_1.00x"
    baseline_params = dict(candidates[0][1])
    baseline_trades = simulate_orb(df, baseline_params)
    baseline_metrics = compute_trade_metrics(baseline_trades)
    baseline_fold_results = compute_fold_results(baseline_trades, folds)
    baseline_fold_map = {row["fold"]: row for row in baseline_fold_results}

    summaries = []
    for name, params in candidates:
        trades = simulate_orb(df, params)
        summaries.append(summarize_candidate(name, params, trades, folds, baseline_fold_map))

    summaries.sort(
        key=lambda row: (
            row["improved_vs_baseline_folds"],
            row["positive_test_folds"],
            row["metrics"]["total_pnl"],
            row["metrics"]["profit_factor"],
        ),
        reverse=True,
    )

    baseline_summary = next(row for row in summaries if row["name"] == baseline_name)
    leader = summaries[0]
    leader_gap = round(leader["metrics"]["total_pnl"] - baseline_metrics["total_pnl"], 4)
    clear_leader = (
        leader["name"] != baseline_name
        and leader["improved_vs_baseline_folds"] >= 3
        and leader["metrics"]["total_pnl"] > baseline_metrics["total_pnl"]
        and leader["metrics"]["profit_factor"] >= baseline_metrics["profit_factor"]
    )

    refined_stop_scan = []
    for stop_mult in [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
        params = {**BASE_PARAMS, "initial_stop_mult": stop_mult}
        trades = simulate_orb(df, params)
        metrics = compute_trade_metrics(trades)
        fold_results = compute_fold_results(trades, folds)
        refined_stop_scan.append(
            {
                "initial_stop_mult": stop_mult,
                "total_pnl": metrics["total_pnl"],
                "profit_factor": metrics["profit_factor"],
                "stop_exits": metrics["stop_exits"],
                "under_2h_pnl": metrics["under_2h_pnl"],
                "improved_vs_baseline_folds": sum(
                    1
                    for row in fold_results
                    if row["total_pnl"] > baseline_fold_map[row["fold"]]["total_pnl"]
                ),
            }
        )

    payload = {
        "data": {
            "source": str(DATA_PATH.name),
            "bars": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "research_scope": "local_orb_structural_proxy",
        "baseline": baseline_summary,
        "top_candidates": summaries[:5],
        "all_candidates": summaries,
        "refined_stop_scan_delay_0": refined_stop_scan,
        "walk_forward_test_folds": [
            {"fold": fold["fold"], "start": str(fold["start"]), "end": str(fold["end"])}
            for fold in folds
        ],
        "research_leader": leader if clear_leader else None,
        "conclusion": {
            "clear_leader_found": clear_leader,
            "top_candidate": leader["name"],
            "top_candidate_pnl_delta_vs_baseline": leader_gap,
            "note": (
                "Local structural proxy only. Even if a leader exists, it must be validated further "
                "before any QC_STRATEGY change."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved structural sweep to {OUTPUT_PATH}")
    print("Top candidates:")
    for row in summaries[:5]:
        m = row["metrics"]
        print(
            f"{row['name']}: pnl={m['total_pnl']:+.2f}, trades={m['trades']}, "
            f"pf={m['profit_factor']:.3f}, under2h={m['under_2h_pnl']:+.2f}, "
            f"folds={row['improved_vs_baseline_folds']}/4"
        )
    if clear_leader:
        print(f"Research leader: {leader['name']}")
    else:
        print("Research leader: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
