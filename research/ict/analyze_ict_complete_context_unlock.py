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
from src.strategies.ict_entry_model import ICTEntryModelStrategy, build_ict_research_profile_params


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
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_complete_context_unlock.json"
)

_BASE_UNLOCK_OVERRIDES = {
    "use_amd_filter": False,
    "use_macro_timing_windows": False,
    "use_kill_zones": False,
}


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


def _variant_overrides() -> dict[str, dict[str, Any]]:
    return {
        "base_unlocked": {},
        "plus_no_daily_bias": {"use_daily_bias_filter": False},
        "plus_no_premium": {"use_premium_discount_filter": False},
        "plus_no_prev_session": {"use_prev_session_anchor_filter": False},
        "plus_no_external_liquidity": {"use_external_liquidity_filter": False},
        "plus_no_session_array": {"use_session_array_refinement": False},
        "plus_no_daily_premium_prev": {
            "use_daily_bias_filter": False,
            "use_premium_discount_filter": False,
            "use_prev_session_anchor_filter": False,
        },
    }


def _run_variant(
    merged,
    *,
    label: str,
    overrides: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    params_overrides = dict(_BASE_UNLOCK_OVERRIDES)
    params_overrides.update(overrides)
    params = build_ict_research_profile_params(enable_smt=True, overrides=params_overrides)
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    funnel = _build_funnel_summary(signals.metadata)
    return {
        "label": label,
        "overrides": params_overrides,
        "metrics": result.metrics,
        "funnel": funnel,
    }


def _ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        funnel = row["funnel"]
        filtered = funnel["filtered_breakdown"]
        ranked.append(
            {
                "label": row["label"],
                "total_trades": int(row["metrics"]["total_trades"]),
                "total_return_pct": float(row["metrics"]["total_return_pct"]),
                "profit_factor": float(row["metrics"]["profit_factor"]),
                "accepted_sweeps": int(funnel["stages"]["accepted_sweeps"]),
                "shift_candidates": int(funnel["stages"]["shift_candidates"]),
                "armed_setups": int(funnel["stages"]["armed_setups"]),
                "entries": int(funnel["stages"]["entries"]),
                "daily_bias_filtered_setups": int(filtered.get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(filtered.get("premium_discount_filtered_setups", 0)),
                "prev_session_anchor_filtered_setups": int(filtered.get("prev_session_anchor_filtered_setups", 0)),
                "external_liquidity_filtered_sweeps": int(filtered.get("external_liquidity_filtered_sweeps", 0)),
                "session_array_filtered_shifts": int(filtered.get("session_array_filtered_shifts", 0)),
                "delivery_missing_shifts": int(filtered.get("delivery_missing_shifts", 0)),
            }
        )
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
        description="Diagnose the first viable context unlock path from the complete ICT strategy on the full dataset."
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
        _run_variant(
            merged,
            label=label,
            overrides=overrides,
            engine=engine,
        )
        for label, overrides in _variant_overrides().items()
    ]

    output = {
        "analysis": "ict_complete_context_unlock",
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
        "base_unlock_overrides": _BASE_UNLOCK_OVERRIDES,
        "profiles": payloads,
        "ranking": _ranking(payloads),
        "interpretation": (
            "These variants keep the complete ICT stack conceptually intact while identifying which hard context filters must be relaxed first to restore trade flow on the full-history dataset."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
