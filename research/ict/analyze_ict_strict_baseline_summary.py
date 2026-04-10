from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_ict_frontier_funnel import _build_funnel_summary
from analyze_ict_lite_reversal_baseline import RESEARCH_STANDARD, _engine_config_payload
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_strict_baseline_summary.json"


def _strict_summary_payload(
    merged,
    *,
    params: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    funnel = _build_funnel_summary(signals.metadata)
    return _assemble_strict_summary(
        metrics=result.metrics,
        metadata=signals.metadata,
        params=params,
        engine=engine,
    )


def _assemble_strict_summary(
    *,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    params: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    funnel = _build_funnel_summary(metadata)
    return {
        "label": "strict_ict_frontier_standardized",
        "metrics": metrics,
        "delivery_breakdown": {
            "fvg_entries": int(metadata.get("fvg_entries", 0)),
            "ob_entries": int(metadata.get("ob_entries", 0)),
            "breaker_entries": int(metadata.get("breaker_entries", 0)),
            "ifvg_entries": int(metadata.get("ifvg_entries", 0)),
        },
        "sweep_breakdown": {
            "bullish_sweeps": int(metadata.get("bullish_sweeps", 0)),
            "bearish_sweeps": int(metadata.get("bearish_sweeps", 0)),
            "raw_sweep_candidates": int(funnel["stages"]["raw_sweep_candidates"]),
            "accepted_sweeps": int(funnel["stages"]["accepted_sweeps"]),
        },
        "risk_block": {
            "take_profit_rr": float(params["take_profit_rr"]),
            "min_reward_risk_ratio": float(params["min_reward_risk_ratio"]),
            "position_size_mode": str(engine.position_size_mode),
            "capital_usage_pct": float(engine.capital_usage_pct),
            "min_shares": int(engine.min_shares),
        },
        "funnel": funnel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the standardized strict ICT baseline summary used by the 500-trades roadmap."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    params = (
        build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
            enable_smt=True
        )
    )

    payload = _strict_summary_payload(merged, params=params, engine=engine)
    output = {
        "analysis": "ict_strict_baseline_summary",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "risk_standard": {
            "reward_risk_gate": ">= 1.5:1",
            "engine": _engine_config_payload(engine),
        },
        "frontier_params": {
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "displacement_body_min_pct": float(params["displacement_body_min_pct"]),
        },
        "summary": payload,
        "interpretation": (
            "This standardized strict baseline summary is the fixed benchmark block for the 500-trades roadmap and should be reused instead of re-deriving strict baseline metrics ad hoc."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
