from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_frontier_funnel.json"


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _build_funnel_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    bullish_sweep_candidates = int(metadata.get("bullish_sweep_candidates", 0))
    bearish_sweep_candidates = int(metadata.get("bearish_sweep_candidates", 0))
    raw_sweep_candidates = bullish_sweep_candidates + bearish_sweep_candidates

    bullish_sweeps = int(metadata.get("bullish_sweeps", 0))
    bearish_sweeps = int(metadata.get("bearish_sweeps", 0))
    accepted_sweeps = bullish_sweeps + bearish_sweeps

    bullish_shift_candidates = int(metadata.get("bullish_shift_candidates", 0))
    bearish_shift_candidates = int(metadata.get("bearish_shift_candidates", 0))
    shift_candidates = bullish_shift_candidates + bearish_shift_candidates

    bullish_shifts = int(metadata.get("bullish_shifts", 0))
    bearish_shifts = int(metadata.get("bearish_shifts", 0))
    armed_setups = bullish_shifts + bearish_shifts

    bullish_retests = int(metadata.get("bullish_retest_candidates", 0))
    bearish_retests = int(metadata.get("bearish_retest_candidates", 0))
    retest_candidates = bullish_retests + bearish_retests

    long_entries = int(metadata.get("long_entries", 0))
    short_entries = int(metadata.get("short_entries", 0))
    entries = long_entries + short_entries

    filtered_breakdown = {
        "daily_bias_filtered_setups": int(metadata.get("daily_bias_filtered_setups", 0)),
        "premium_discount_filtered_setups": int(metadata.get("premium_discount_filtered_setups", 0)),
        "amd_filtered_setups": int(metadata.get("amd_filtered_setups", 0)),
        "prev_session_anchor_filtered_setups": int(metadata.get("prev_session_anchor_filtered_setups", 0)),
        "external_liquidity_filtered_sweeps": int(metadata.get("external_liquidity_filtered_sweeps", 0)),
        "smt_filtered_sweeps": int(metadata.get("smt_filtered_sweeps", 0)),
        "smt_missing_peer_bars": int(metadata.get("smt_missing_peer_bars", 0)),
        "macro_timing_filtered_sweeps": int(metadata.get("macro_timing_filtered_sweeps", 0)),
        "kill_zone_filtered_sweeps": int(metadata.get("kill_zone_filtered_sweeps", 0)),
        "displacement_filtered_shifts": int(metadata.get("displacement_filtered_shifts", 0)),
        "displacement_range_filtered_shifts": int(metadata.get("displacement_range_filtered_shifts", 0)),
        "structure_buffer_filtered_shifts": int(metadata.get("structure_buffer_filtered_shifts", 0)),
        "session_array_filtered_shifts": int(metadata.get("session_array_filtered_shifts", 0)),
        "delivery_missing_shifts": int(metadata.get("delivery_missing_shifts", 0)),
        "score_filtered_shifts": int(metadata.get("score_filtered_shifts", 0)),
        "fvg_delay_filtered_retests": int(metadata.get("fvg_delay_filtered_retests", 0)),
        "fvg_depth_filtered_retests": int(metadata.get("fvg_depth_filtered_retests", 0)),
        "fvg_touch_filtered_retests": int(metadata.get("fvg_touch_filtered_retests", 0)),
        "fvg_close_filtered_retests": int(metadata.get("fvg_close_filtered_retests", 0)),
        "fvg_wick_filtered_retests": int(metadata.get("fvg_wick_filtered_retests", 0)),
        "fvg_body_filtered_retests": int(metadata.get("fvg_body_filtered_retests", 0)),
        "sweep_blocked_by_existing_pending": int(metadata.get("sweep_blocked_by_existing_pending", 0)),
        "sweep_expired_before_shift": int(metadata.get("sweep_expired_before_shift", 0)),
        "armed_setup_expired_before_retest": int(metadata.get("armed_setup_expired_before_retest", 0)),
    }
    top_filters = [
        {"label": key, "count": value}
        for key, value in sorted(filtered_breakdown.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:8]

    return {
        "stages": {
            "raw_sweep_candidates": raw_sweep_candidates,
            "accepted_sweeps": accepted_sweeps,
            "shift_candidates": shift_candidates,
            "armed_setups": armed_setups,
            "retest_candidates": retest_candidates,
            "entries": entries,
        },
        "direction_breakdown": {
            "bullish_sweep_candidates": bullish_sweep_candidates,
            "bearish_sweep_candidates": bearish_sweep_candidates,
            "bullish_sweeps": bullish_sweeps,
            "bearish_sweeps": bearish_sweeps,
            "bullish_shift_candidates": bullish_shift_candidates,
            "bearish_shift_candidates": bearish_shift_candidates,
            "bullish_armed_setups": bullish_shifts,
            "bearish_armed_setups": bearish_shifts,
            "bullish_retest_candidates": bullish_retests,
            "bearish_retest_candidates": bearish_retests,
            "long_entries": long_entries,
            "short_entries": short_entries,
        },
        "conversion_rates_pct": {
            "sweep_acceptance_rate": _pct(accepted_sweeps, raw_sweep_candidates),
            "accepted_sweep_to_shift_rate": _pct(shift_candidates, accepted_sweeps),
            "shift_to_armed_rate": _pct(armed_setups, shift_candidates),
            "armed_to_retest_rate": _pct(retest_candidates, armed_setups),
            "retest_to_entry_rate": _pct(entries, retest_candidates),
            "raw_sweep_to_entry_rate": _pct(entries, raw_sweep_candidates),
        },
        "filtered_breakdown": filtered_breakdown,
        "top_filters": top_filters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instrument the active strict ICT frontier funnel to locate the main density bottlenecks."
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

    params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
        enable_smt=True
    )
    strategy = ICTEntryModelStrategy(params=params)
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    funnel = _build_funnel_summary(signals.metadata)

    output = {
        "analysis": "ict_frontier_funnel",
        "profile": "strict_ict_frontier",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "frontier_params": {
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "displacement_body_min_pct": float(params["displacement_body_min_pct"]),
        },
        "metrics": result.metrics,
        "delivery_breakdown": {
            "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
            "ob_entries": int(signals.metadata.get("ob_entries", 0)),
            "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
            "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
        },
        "funnel": funnel,
        "interpretation": (
            "Use the stage counts and top filter ranking to decide whether the next 500-trades optimization should relax sweep density, revisit timing, session gates, or downstream entry purity."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
