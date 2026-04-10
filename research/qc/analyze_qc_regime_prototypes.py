from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.qc.analyze_qc_webide_result import resolve_bundle
from daily_session_alignment import align_features_to_next_session, load_daily_market_frame

INITIAL_CAPITAL = 100000.0


def load_trades(result_dir: Path) -> pd.DataFrame:
    bundle = resolve_bundle(result_dir)
    trades = pd.read_csv(bundle.trades_path)
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
    trades["entry_date"] = trades["Entry Time"].dt.tz_convert("America/New_York").dt.date
    trades["exit_date"] = trades["Exit Time"].dt.tz_convert("America/New_York").dt.date
    trades["net_pnl"] = pd.to_numeric(trades["P&L"], errors="coerce").fillna(0.0) - pd.to_numeric(
        trades["Fees"], errors="coerce"
    ).fillna(0.0)
    trades["is_win_net"] = (trades["net_pnl"] > 0).astype(int)
    trades["is_long"] = trades["Direction"].str.lower().eq("buy")
    trades["is_short"] = trades["Direction"].str.lower().eq("sell")
    return trades


def build_daily_regime_frame(daily_path: Path) -> pd.DataFrame:
    qqq = load_daily_market_frame(daily_path)
    qqq["close"] = qqq["Close"]
    qqq["ema50"] = qqq["close"].ewm(span=50, adjust=False).mean()
    qqq["ema200"] = qqq["close"].ewm(span=200, adjust=False).mean()
    qqq["trend_up"] = (qqq["close"] > qqq["ema200"]) & (qqq["ema50"] > qqq["ema200"])

    ret = qqq["close"].pct_change()
    trend_strength_raw = qqq["close"].pct_change(20).abs() / ret.abs().rolling(20).sum().replace(0.0, np.nan)
    trend_strength_threshold = trend_strength_raw.rolling(252, min_periods=60).median()
    qqq["trend_strength"] = trend_strength_raw > trend_strength_threshold

    gap_abs = (qqq["Open"] / qqq["Close"].shift(1) - 1.0).abs()
    qqq["low_gap"] = gap_abs < gap_abs.rolling(60, min_periods=20).median()
    qqq["up_5d"] = qqq["close"] > qqq["close"].rolling(5).mean()
    aligned, _ = align_features_to_next_session(qqq, ["trend_up", "trend_strength", "low_gap", "up_5d"])
    return aligned


def compute_profit_factor(net_pnl: pd.Series) -> float:
    gross_profit = float(net_pnl[net_pnl > 0].sum())
    gross_loss = float(-net_pnl[net_pnl < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def rolling_summary(calendar_dates: list[object], subset: pd.DataFrame, months: int) -> dict[str, float]:
    daily = pd.DataFrame({"date": pd.to_datetime(calendar_dates)})
    pnl_by_day = subset.groupby("exit_date")["net_pnl"].sum().rename("net_pnl")
    daily = daily.merge(pnl_by_day, left_on=daily["date"].dt.date, right_index=True, how="left").fillna({"net_pnl": 0.0})
    daily["equity"] = INITIAL_CAPITAL + daily["net_pnl"].cumsum()
    daily["ret"] = daily["equity"].pct_change().fillna(0.0)
    month_ends = daily.set_index("date")["equity"].resample("ME").last().dropna().index
    rows = []
    for end in month_ends:
        start = (pd.Timestamp(end) - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
        win = daily[(daily["date"] >= start) & (daily["date"] <= end)]
        if len(win) < 40:
            continue
        trades_win = subset[(pd.to_datetime(subset["exit_date"]) >= start) & (pd.to_datetime(subset["exit_date"]) <= end)]
        pf = compute_profit_factor(trades_win["net_pnl"]) if len(trades_win) else 0.0
        std = float(win["ret"].std(ddof=0))
        sharpe = 0.0 if std == 0 else float(np.sqrt(252.0) * win["ret"].mean() / std)
        net = float((win["equity"].iloc[-1] / win["equity"].iloc[0] - 1.0) * 100.0)
        rows.append({"sharpe": sharpe, "net": net, "pf": pf})
    out = pd.DataFrame(rows)
    finite_pf = out["pf"].replace([np.inf, -np.inf], np.nan)
    return {
        "window_count": int(len(out)),
        "positive_sharpe_pct": round(float((out["sharpe"] > 0).mean() * 100.0), 1) if len(out) else 0.0,
        "median_sharpe": round(float(out["sharpe"].median()), 3) if len(out) else 0.0,
        "median_net_profit_pct": round(float(out["net"].median()), 3) if len(out) else 0.0,
        "median_profit_factor": round(float(finite_pf.median()), 3) if finite_pf.notna().any() else 0.0,
    }


def evaluate_candidates(result_dir: Path, daily_path: Path) -> dict[str, object]:
    trades = load_trades(result_dir)
    regime = build_daily_regime_frame(daily_path)
    merged = trades.merge(regime, left_on="entry_date", right_on="date", how="left")

    candidates = {
        "baseline_all": pd.Series(True, index=merged.index),
        "long_up_trendstrength_lowgap": merged["is_long"] & merged["trend_up"] & merged["trend_strength"] & merged["low_gap"],
        "long_up_trendstrength_up5d": merged["is_long"] & merged["trend_up"] & merged["trend_strength"] & merged["up_5d"],
    }

    results: dict[str, object] = {}
    calendar_dates = sorted(set(regime["date"]))
    for name, mask in candidates.items():
        subset = merged[mask.fillna(False)].copy()
        pf = compute_profit_factor(subset["net_pnl"]) if len(subset) else 0.0
        results[name] = {
            "trades": int(len(subset)),
            "win_rate_pct": round(float(subset["is_win_net"].mean() * 100.0), 2) if len(subset) else 0.0,
            "profit_factor": round(float(pf), 3) if np.isfinite(pf) else float("inf"),
            "net_pnl": round(float(subset["net_pnl"].sum()), 2),
            "avg_trade_pnl": round(float(subset["net_pnl"].mean()), 3) if len(subset) else 0.0,
            "rolling_6m": rolling_summary(calendar_dates, subset, 6),
            "rolling_12m": rolling_summary(calendar_dates, subset, 12),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate regime-gated ORB prototypes from QuantConnect exports.")
    parser.add_argument("result_dir", nargs="?", default="QuantConnect results/2017-2026")
    parser.add_argument("--daily-csv", default="qqq_1d.csv")
    args = parser.parse_args()

    results = evaluate_candidates(Path(args.result_dir), Path(args.daily_csv))
    out_dir = Path("results") / "qc_regime_prototypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.result_dir).name.replace(" ", "_")
    (out_dir / f"{stem}_prototype_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
