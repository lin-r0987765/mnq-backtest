from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestEngine
from src.data.fetcher import load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_mtf_topdown_continuation_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_profile_params,
    build_ict_mtf_topdown_continuation_quality_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params,
    build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_timing_candidate_profile_params,
)

DEFAULT_INTRADAY = PROJECT_ROOT / "regularized" / "QQQ" / "qqq_1m_regular_et.csv"
if not DEFAULT_INTRADAY.exists():
    DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_1m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "regularized" / "SPY" / "spy_1m_regular_et.csv"
if not DEFAULT_PEER_CSV.exists():
    DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_1m_alpaca.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "qc_regime_prototypes"
RESEARCH_STANDARD = {
    "initial_cash": 10_000.0,
    "fees_pct": 0.0005,
    "position_size_mode": "capital_pct",
    "capital_usage_pct": 1.0,
    "min_shares": 0,
}
ET_TIMEZONE = "America/New_York"


def _build_profile(profile: str) -> tuple[str, dict[str, Any]]:
    key = str(profile).strip().lower()
    if key == "execution_candidate":
        return (
            "mtf_topdown_continuation_execution_candidate",
            build_ict_mtf_topdown_continuation_execution_candidate_profile_params(enable_smt=False),
        )
    if key == "setup_execution_candidate":
        return (
            "mtf_topdown_continuation_setup_execution_candidate",
            build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params(enable_smt=False),
        )
    if key == "timing_candidate":
        return (
            "mtf_topdown_continuation_timing_candidate",
            build_ict_mtf_topdown_continuation_timing_candidate_profile_params(enable_smt=False),
        )
    if key == "quality_candidate":
        return (
            "mtf_topdown_continuation_quality_candidate",
            build_ict_mtf_topdown_continuation_quality_candidate_profile_params(enable_smt=False),
        )
    if key == "regularized_long_only_candidate":
        return (
            "mtf_topdown_continuation_regularized_long_only_candidate",
            build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params(
                enable_smt=False
            ),
        )
    if key == "regularized_long_only_am_candidate":
        return (
            "mtf_topdown_continuation_regularized_long_only_am_candidate",
            build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params(
                enable_smt=False
            ),
        )
    return (
        "mtf_topdown_continuation_baseline",
        build_ict_mtf_topdown_continuation_profile_params(enable_smt=False),
    )


def _resolve_output_path(requested_output: str | None, *, profile_name: str, year: int) -> Path:
    if requested_output:
        return Path(requested_output)
    return DEFAULT_OUTPUT_DIR / f"ict_mtf_topdown_{profile_name}_holdout_{year}_breakdown.json"


def _slice_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01", tz=ET_TIMEZONE)
    end = pd.Timestamp(f"{year}-12-31 23:59:59.999999", tz=ET_TIMEZONE)
    return df.loc[start:end].copy()


def _data_window(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "trading_days": 0}
    return {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "trading_days": int(len(set(df.index.date))),
    }


def _json_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _trades_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    frame = pd.DataFrame(trades).copy()
    if "entry_time" in frame.columns:
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True).dt.tz_convert(ET_TIMEZONE)
        frame["entry_hour_et"] = frame["entry_time"].dt.hour
        frame["entry_weekday_et"] = frame["entry_time"].dt.day_name()
        frame["entry_month"] = frame["entry_time"].dt.strftime("%Y-%m")
    if "exit_time" in frame.columns:
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True).dt.tz_convert(ET_TIMEZONE)
    frame["pnl"] = frame.get("pnl", 0.0).apply(_json_number)
    frame["bars_held"] = frame.get("bars_held", 0.0).apply(_json_number)
    frame["win"] = frame["pnl"] > 0.0
    return frame


def _group_summary(df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if df.empty or column not in df.columns:
        return []
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            trades=("pnl", "size"),
            pnl_sum=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            win_rate=("win", "mean"),
            avg_bars_held=("bars_held", "mean"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                column: str(row[column]),
                "trades": int(row["trades"]),
                "pnl_sum": round(float(row["pnl_sum"]), 4),
                "avg_pnl": round(float(row["avg_pnl"]), 4),
                "win_rate_pct": round(float(row["win_rate"]) * 100.0, 4),
                "avg_bars_held": round(float(row["avg_bars_held"]), 4),
            }
        )
    rows.sort(key=lambda item: (item["pnl_sum"], item["trades"]), reverse=True)
    return rows


def _top_trades(df: pd.DataFrame, *, best: bool, limit: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    ordered = df.sort_values("pnl", ascending=not best).head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        rows.append(
            {
                "side": str(row.get("side", "")),
                "entry_time": row["entry_time"].isoformat() if "entry_time" in row else None,
                "exit_time": row["exit_time"].isoformat() if "exit_time" in row else None,
                "entry_hour_et": int(row.get("entry_hour_et", -1)),
                "entry_weekday_et": str(row.get("entry_weekday_et", "")),
                "pnl": round(float(row["pnl"]), 4),
                "bars_held": round(float(row.get("bars_held", 0.0)), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break down a 1m ICT MTF profile on a holdout year by hour, weekday, and month."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV))
    parser.add_argument(
        "--profile",
        choices=[
            "baseline",
            "quality_candidate",
            "execution_candidate",
            "setup_execution_candidate",
            "timing_candidate",
            "regularized_long_only_candidate",
            "regularized_long_only_am_candidate",
        ],
        default="regularized_long_only_candidate",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = load_ohlcv_csv(args.peer_csv)
    merged = merge_peer_columns(intraday_df, peer_df)
    holdout_df = _slice_year(merged, args.year)

    profile_name, params = _build_profile(args.profile)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    result = engine.run(ICTEntryModelStrategy(params=params), holdout_df)
    trades = _trades_frame(result.trades)

    output = {
        "analysis": "ict_mtf_topdown_holdout_breakdown",
        "profile": profile_name,
        "holdout_year": int(args.year),
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_csv": str(Path(args.peer_csv)),
        "data_window": _data_window(holdout_df),
        "rows_primary": int(len(holdout_df)),
        "rows_peer_matched": int(holdout_df["PeerHigh"].notna().sum()) if "PeerHigh" in holdout_df.columns else 0,
        "metrics": result.metrics,
        "trade_count": int(len(trades)),
        "by_entry_hour_et": _group_summary(trades, "entry_hour_et"),
        "by_entry_weekday_et": _group_summary(trades, "entry_weekday_et"),
        "by_entry_month": _group_summary(trades, "entry_month"),
        "top_winners": _top_trades(trades, best=True),
        "top_losers": _top_trades(trades, best=False),
    }

    output_path = _resolve_output_path(args.output, profile_name=profile_name, year=args.year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
