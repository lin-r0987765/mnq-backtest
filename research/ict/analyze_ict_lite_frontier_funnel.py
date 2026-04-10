from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_ict_frontier_funnel import _build_funnel_summary
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_frontier_funnel.json"


def _build_stage_drop_summary(funnel: dict[str, Any]) -> list[dict[str, Any]]:
    stages = funnel["stages"]
    transitions = [
        ("raw_sweep_candidates", "accepted_sweeps"),
        ("accepted_sweeps", "shift_candidates"),
        ("shift_candidates", "armed_setups"),
        ("armed_setups", "retest_candidates"),
        ("retest_candidates", "entries"),
    ]
    drops: list[dict[str, Any]] = []
    for source, target in transitions:
        source_count = int(stages[source])
        target_count = int(stages[target])
        drops.append(
            {
                "transition": f"{source}->{target}",
                "source_count": source_count,
                "target_count": target_count,
                "drop_count": max(source_count - target_count, 0),
            }
        )
    drops.sort(key=lambda row: (row["drop_count"], row["source_count"]), reverse=True)
    return drops


def _verdict(entries: int, top_filter: dict[str, Any] | None, trade_gap_to_100: int) -> tuple[str, str]:
    if entries >= 100:
        return (
            "LITE_FRONTIER_FUNNEL_CLEARS_100_TRADE_GATE",
            "The active lite frontier already clears the first roadmap gate, so the next step should shift from density diagnosis to stability and OOS validation.",
        )
    if top_filter is None:
        return (
            "LITE_FRONTIER_FUNNEL_IDENTIFIED_NO_FILTER_COUNTS",
            "The active lite frontier stays below the first roadmap gate, but the current metadata does not show a positive filter choke point to target next.",
        )
    return (
        "LITE_FRONTIER_FUNNEL_IDENTIFIED_NEXT_CHOKE_POINT",
        f"The active lite frontier is still {trade_gap_to_100} trades short of the 100-trade gate, and the largest recorded filter choke point is {top_filter['label']} = {top_filter['count']}.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instrument the active lite ICT frontier funnel to locate the next density bottleneck on the 500-trades roadmap."
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

    params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(enable_smt=True)
    strategy = ICTEntryModelStrategy(params=params)
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    funnel = _build_funnel_summary(signals.metadata)
    stage_drops = _build_stage_drop_summary(funnel)
    entries = int(funnel["stages"]["entries"])
    trade_gap_to_100 = max(100 - entries, 0)
    top_filter = funnel["top_filters"][0] if funnel["top_filters"] else None
    verdict, interpretation = _verdict(entries, top_filter, trade_gap_to_100)

    output = {
        "analysis": "ict_lite_frontier_funnel",
        "profile": "lite_ict_reversal_relaxed_smt_looser_sweep_faster_retest_frontier",
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
            "smt_threshold": float(params["smt_threshold"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "displacement_body_min_pct": float(params["displacement_body_min_pct"]),
        },
        "metrics": result.metrics,
        "trade_gap_to_100": trade_gap_to_100,
        "delivery_breakdown": {
            "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
            "ob_entries": int(signals.metadata.get("ob_entries", 0)),
            "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
            "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
        },
        "funnel": funnel,
        "stage_drop_summary": stage_drops,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
