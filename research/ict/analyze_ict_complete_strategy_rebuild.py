from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ict.analyze_ict_frontier_funnel import _build_funnel_summary
from research.ict.analyze_ict_lite_reversal_baseline import RESEARCH_STANDARD, _engine_config_payload
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_complete_soft_premium_profile_params,
    build_ict_complete_soft_prev_session_profile_params,
    build_ict_complete_soft_session_array_profile_params,
    build_ict_research_profile_params,
)

_REGULARIZED_ROOT = PROJECT_ROOT / "regularized"
DEFAULT_INTRADAY = (
    _REGULARIZED_ROOT / "QQQ" / "qqq_5m_regular_et.csv"
    if (_REGULARIZED_ROOT / "QQQ" / "qqq_5m_regular_et.csv").exists()
    else PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
)
DEFAULT_PEER_CSV = (
    _REGULARIZED_ROOT / "SPY" / "spy_5m_regular_et.csv"
    if (_REGULARIZED_ROOT / "SPY" / "spy_5m_regular_et.csv").exists()
    else PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_complete_strategy_rebuild.json"
)


def _data_window(df) -> dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "trading_days": 0}
    return {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "trading_days": int(len(set(df.index.date))),
    }


def _period_label(df) -> str:
    window = _data_window(df)
    if window["start"] is None or window["end"] is None:
        return "empty"
    return f'{window["start"]} -> {window["end"]}'


def _run_profile(
    merged,
    *,
    label: str,
    params: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    metadata = signals.metadata
    return {
        "label": label,
        "metrics": result.metrics,
        "params": {
            "use_smt_filter": bool(params["use_smt_filter"]),
            "use_kill_zones": bool(params["use_kill_zones"]),
            "use_daily_bias_filter": bool(params["use_daily_bias_filter"]),
            "use_premium_discount_filter": bool(params["use_premium_discount_filter"]),
            "premium_discount_filter_mode": str(params["premium_discount_filter_mode"]),
            "use_external_liquidity_filter": bool(params["use_external_liquidity_filter"]),
            "use_amd_filter": bool(params["use_amd_filter"]),
            "use_macro_timing_windows": bool(params["use_macro_timing_windows"]),
            "use_prev_session_anchor_filter": bool(params["use_prev_session_anchor_filter"]),
            "prev_session_anchor_filter_mode": str(params["prev_session_anchor_filter_mode"]),
            "use_session_array_refinement": bool(params["use_session_array_refinement"]),
            "session_array_filter_mode": str(params["session_array_filter_mode"]),
            "trade_sessions": bool(params["trade_sessions"]),
            "take_profit_rr": float(params["take_profit_rr"]),
            "min_reward_risk_ratio": float(params["min_reward_risk_ratio"]),
        },
        "delivery_breakdown": {
            "fvg_entries": int(metadata.get("fvg_entries", 0)),
            "ob_entries": int(metadata.get("ob_entries", 0)),
            "breaker_entries": int(metadata.get("breaker_entries", 0)),
            "ifvg_entries": int(metadata.get("ifvg_entries", 0)),
        },
        "funnel": _build_funnel_summary(metadata),
    }


def _ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [
        {
            "label": row["label"],
            "total_trades": int(row["metrics"]["total_trades"]),
            "total_return_pct": float(row["metrics"]["total_return_pct"]),
            "profit_factor": float(row["metrics"]["profit_factor"]),
            "win_rate_pct": float(row["metrics"]["win_rate_pct"]),
        }
        for row in rows
    ]
    ranked.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the complete ICT baseline on the current full-history dataset before new iterations."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="full_local")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(**RESEARCH_STANDARD)

    payloads = [
        _run_profile(
            merged,
            label="complete_ict_no_smt",
            params=build_ict_research_profile_params(enable_smt=False),
            engine=engine,
        ),
        _run_profile(
            merged,
            label="complete_ict_with_smt",
            params=build_ict_research_profile_params(enable_smt=True),
            engine=engine,
        ),
        _run_profile(
            merged,
            label="complete_soft_premium_with_smt",
            params=build_ict_complete_soft_premium_profile_params(enable_smt=True),
            engine=engine,
        ),
        _run_profile(
            merged,
            label="complete_soft_session_array_with_smt",
            params=build_ict_complete_soft_session_array_profile_params(enable_smt=True),
            engine=engine,
        ),
        _run_profile(
            merged,
            label="complete_soft_prev_session_with_smt",
            params=build_ict_complete_soft_prev_session_profile_params(enable_smt=True),
            engine=engine,
        ),
    ]

    output = {
        "analysis": "ict_complete_strategy_rebuild",
        "risk_standard": {
            "reward_risk_gate": ">= 1.5:1",
            "engine": _engine_config_payload(engine),
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": _period_label(intraday_df),
        "peer_fetch_period": args.period if not args.peer_csv else None,
        "data_window": _data_window(intraday_df),
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "profiles": payloads,
        "ranking": _ranking(payloads),
        "interpretation": (
            "This rebuild resets the ICT iteration root to the complete ruleset on the full regularized history before any new density or MTF branches are promoted."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
