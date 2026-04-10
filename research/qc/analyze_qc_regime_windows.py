from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.qc.analyze_qc_webide_result import build_report, resolve_bundle


@dataclass
class WindowSummary:
    window_months: int
    window_count: int
    profitable_window_pct: float
    positive_sharpe_window_pct: float
    median_sharpe: float
    median_net_profit_pct: float
    median_win_rate_pct: float
    median_profit_factor: float
    best_sharpe: float
    best_period: str
    worst_sharpe: float
    worst_period: str


def load_daily_equity(json_path: Path) -> pd.DataFrame:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    chart = obj["charts"]["Strategy Equity"]
    series_key = next(iter(chart["series"]))
    values = chart["series"][series_key]["values"]
    equity = pd.DataFrame(values, columns=["timestamp", "open", "high", "low", "close"])
    equity["datetime"] = pd.to_datetime(equity["timestamp"], unit="s", utc=True)
    equity["date"] = equity["datetime"].dt.normalize().dt.tz_localize(None)
    daily = equity.groupby("date", as_index=False)["close"].last().rename(columns={"close": "equity_close"})
    daily["daily_return"] = daily["equity_close"].pct_change().fillna(0.0)
    return daily


def load_trades(trades_path: Path) -> pd.DataFrame:
    trades = pd.read_csv(trades_path)
    trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True).dt.tz_localize(None)
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True).dt.tz_localize(None)
    trades["P&L"] = pd.to_numeric(trades["P&L"], errors="coerce").fillna(0.0)
    trades["Fees"] = pd.to_numeric(trades["Fees"], errors="coerce").fillna(0.0)
    trades["Net P&L"] = trades["P&L"] - trades["Fees"]
    trades["IsWinNet"] = (trades["Net P&L"] > 0).astype(int)
    return trades


def compute_sharpe(returns: pd.Series) -> float:
    std = float(returns.std(ddof=0))
    if std == 0:
        return 0.0
    return float(np.sqrt(252.0) * returns.mean() / std)


def compute_max_drawdown_pct(returns: pd.Series) -> float:
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min() * 100.0)


def compute_profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def rolling_windows(daily: pd.DataFrame, trades: pd.DataFrame, months: int) -> pd.DataFrame:
    month_ends = (
        daily.set_index("date")["equity_close"]
        .resample("ME")
        .last()
        .dropna()
        .index
    )
    rows: list[dict[str, object]] = []
    daily_indexed = daily.set_index("date")
    min_date = daily_indexed.index.min()
    for end_date in month_ends:
        raw_start_date = (pd.Timestamp(end_date) - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
        if raw_start_date < min_date:
            continue
        start_date = raw_start_date
        window_daily = daily_indexed.loc[(daily_indexed.index >= start_date) & (daily_indexed.index <= end_date)]
        if len(window_daily) < 40:
            continue
        window_trades = trades.loc[(trades["Exit Time"] >= start_date) & (trades["Exit Time"] <= end_date)]
        pnl = window_trades["Net P&L"] if len(window_trades) else pd.Series(dtype=float)
        trades_count = int(len(window_trades))
        win_rate = float(window_trades["IsWinNet"].mean() * 100.0) if trades_count else 0.0
        profit_factor = compute_profit_factor(pnl) if trades_count else 0.0
        returns = window_daily["daily_return"]
        rows.append(
            {
                "window_months": months,
                "start_date": start_date.date().isoformat(),
                "end_date": pd.Timestamp(end_date).date().isoformat(),
                "trades": trades_count,
                "win_rate_pct": round(win_rate, 2),
                "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else float("inf"),
                "net_profit_pct": round(float(((1.0 + returns).prod() - 1.0) * 100.0), 3),
                "sharpe": round(compute_sharpe(returns), 3),
                "max_drawdown_pct": round(compute_max_drawdown_pct(returns), 3),
            }
        )
    return pd.DataFrame(rows)


def summarize_windows(df: pd.DataFrame, months: int) -> WindowSummary:
    best = df.sort_values("sharpe", ascending=False).iloc[0]
    worst = df.sort_values("sharpe", ascending=True).iloc[0]
    finite_pf = df["profit_factor"].replace([np.inf, -np.inf], np.nan)
    return WindowSummary(
        window_months=months,
        window_count=int(len(df)),
        profitable_window_pct=round(float((df["net_profit_pct"] > 0).mean() * 100.0), 1),
        positive_sharpe_window_pct=round(float((df["sharpe"] > 0).mean() * 100.0), 1),
        median_sharpe=round(float(df["sharpe"].median()), 3),
        median_net_profit_pct=round(float(df["net_profit_pct"].median()), 3),
        median_win_rate_pct=round(float(df["win_rate_pct"].median()), 2),
        median_profit_factor=round(float(finite_pf.median()), 3) if finite_pf.notna().any() else 0.0,
        best_sharpe=round(float(best["sharpe"]), 3),
        best_period=f"{best['start_date']} -> {best['end_date']}",
        worst_sharpe=round(float(worst["sharpe"]), 3),
        worst_period=f"{worst['start_date']} -> {worst['end_date']}",
    )


def write_outputs(result_dir: Path, outputs: dict[int, pd.DataFrame], summaries: list[WindowSummary], qc_report: dict[str, object]) -> None:
    out_dir = Path("results") / "qc_regime_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result_dir.name.replace(" ", "_")
    for months, df in outputs.items():
        df.to_csv(out_dir / f"{safe_name}_{months}m_windows.csv", index=False)
    summary_payload = {
        "source_dir": str(result_dir),
        "qc_report": qc_report,
        "window_summaries": [asdict(item) for item in summaries],
    }
    (out_dir / f"{safe_name}_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"來源: {result_dir}",
        f"版本標記: {qc_report['contains_version_marker']}",
        f"half-day 標記: {qc_report['contains_halfday_marker']}",
        f"same_bar_eod_reentry_count: {qc_report['same_bar_eod_reentry_count']}",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"{item.window_months}m 視窗:",
                f"- 視窗數: {item.window_count}",
                f"- 正報酬比例: {item.profitable_window_pct}%",
                f"- 正 Sharpe 比例: {item.positive_sharpe_window_pct}%",
                f"- 中位 Sharpe: {item.median_sharpe}",
                f"- 中位 Net Profit: {item.median_net_profit_pct}%",
                f"- 中位 Win Rate: {item.median_win_rate_pct}%",
                f"- 中位 Profit Factor: {item.median_profit_factor}",
                f"- 最佳區間: {item.best_period} | Sharpe {item.best_sharpe}",
                f"- 最差區間: {item.worst_period} | Sharpe {item.worst_sharpe}",
                "",
            ]
        )
    (out_dir / f"{safe_name}_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze QuantConnect result stability across rolling windows.")
    parser.add_argument("result_dir", nargs="?", default="QuantConnect results/2017-2026")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    bundle = resolve_bundle(result_dir)
    qc_report = build_report(result_dir)
    daily = load_daily_equity(bundle.json_path)
    trades = load_trades(bundle.trades_path)

    outputs: dict[int, pd.DataFrame] = {}
    summaries: list[WindowSummary] = []
    for months in (6, 12):
        df = rolling_windows(daily, trades, months)
        outputs[months] = df
        summaries.append(summarize_windows(df, months))

    write_outputs(result_dir, outputs, summaries, qc_report)
    payload = {
        "source_dir": str(result_dir),
        "window_summaries": [asdict(item) for item in summaries],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
