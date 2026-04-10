from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_ict_continuation_lane_compare import ICTContinuationProxyStrategy
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.base import BaseStrategy, StrategyResult
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_combined_lanes.json"


class _PrecomputedSignalsStrategy(BaseStrategy):
    name = "ICTCombinedLaneStrategy"

    def __init__(self, params: dict[str, Any], signals: StrategyResult):
        super().__init__(params=params)
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        return self._signals


def _merge_lane_signals(
    primary: StrategyResult,
    secondary: StrategyResult,
) -> StrategyResult:
    index = primary.entries_long.index
    entries_long = pd.Series(False, index=index)
    exits_long = pd.Series(False, index=index)
    entries_short = pd.Series(False, index=index)
    exits_short = pd.Series(False, index=index)

    active_lane: str | None = None
    active_side = 0
    metadata: dict[str, Any] = {
        "reversal_entries": 0,
        "continuation_entries": 0,
        "same_bar_lane_conflicts": 0,
        "suppressed_continuation_entries": 0,
        "combined_entries": 0,
    }

    for idx in range(len(index)):
        if active_side > 0:
            if active_lane == "reversal" and bool(primary.exits_long.iat[idx]):
                exits_long.iat[idx] = True
                active_lane = None
                active_side = 0
            elif active_lane == "continuation" and bool(secondary.exits_long.iat[idx]):
                exits_long.iat[idx] = True
                active_lane = None
                active_side = 0
        elif active_side < 0:
            if active_lane == "reversal" and bool(primary.exits_short.iat[idx]):
                exits_short.iat[idx] = True
                active_lane = None
                active_side = 0
            elif active_lane == "continuation" and bool(secondary.exits_short.iat[idx]):
                exits_short.iat[idx] = True
                active_lane = None
                active_side = 0

        primary_long = bool(primary.entries_long.iat[idx])
        primary_short = bool(primary.entries_short.iat[idx])
        secondary_long = bool(secondary.entries_long.iat[idx])
        secondary_short = bool(secondary.entries_short.iat[idx])
        continuation_requested = secondary_long or secondary_short

        if active_side != 0:
            if continuation_requested:
                metadata["suppressed_continuation_entries"] += 1
            continue

        if (primary_long or primary_short) and continuation_requested:
            metadata["same_bar_lane_conflicts"] += 1

        if primary_long:
            entries_long.iat[idx] = True
            active_lane = "reversal"
            active_side = 1
            metadata["reversal_entries"] += 1
            metadata["combined_entries"] += 1
            continue
        if primary_short:
            entries_short.iat[idx] = True
            active_lane = "reversal"
            active_side = -1
            metadata["reversal_entries"] += 1
            metadata["combined_entries"] += 1
            continue
        if secondary_long:
            entries_long.iat[idx] = True
            active_lane = "continuation"
            active_side = 1
            metadata["continuation_entries"] += 1
            metadata["combined_entries"] += 1
            continue
        if secondary_short:
            entries_short.iat[idx] = True
            active_lane = "continuation"
            active_side = -1
            metadata["continuation_entries"] += 1
            metadata["combined_entries"] += 1

    metadata["fvg_entries"] = int(primary.metadata.get("fvg_entries", 0)) + int(
        secondary.metadata.get("fvg_entries", 0)
    )
    return StrategyResult(
        entries_long=entries_long,
        exits_long=exits_long,
        entries_short=entries_short,
        exits_short=exits_short,
        metadata=metadata,
    )


def _verdict(
    reversal_metrics: dict[str, float],
    combined_metrics: dict[str, float],
) -> tuple[str, str]:
    reversal_trades = int(reversal_metrics["total_trades"])
    combined_trades = int(combined_metrics["total_trades"])
    reversal_return = float(reversal_metrics["total_return_pct"])
    combined_return = float(combined_metrics["total_return_pct"])
    if combined_trades > reversal_trades and combined_return >= reversal_return:
        return (
            "ROBUST_COMBINED_LANE_EXTENSION_IDENTIFIED",
            "The combined reversal-plus-continuation lane improves both density and total return versus the active lite reversal frontier.",
        )
    if combined_trades > reversal_trades and combined_return > 0:
        return (
            "COMBINED_LANE_DENSITY_EXTENSION_ONLY",
            "The combined lane adds trade density and stays positive, but it does not beat the active lite reversal frontier on return.",
        )
    if combined_return > 0:
        return (
            "COMBINED_LANE_SURVIVES_BUT_NOT_DENSER",
            "The combined lane stays positive, but it does not improve trade density versus the active lite reversal frontier.",
        )
    return (
        "COMBINED_LANE_REJECTED",
        "The combined lane does not currently justify promotion over the active lite reversal frontier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the active lite reversal frontier against a combined reversal + continuation lane."
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
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    reversal_strategy = ICTEntryModelStrategy(params=params)
    reversal_result = engine.run(reversal_strategy, merged)
    reversal_signals = reversal_strategy.generate_signals(merged)

    continuation_strategy = ICTContinuationProxyStrategy(params=params)
    continuation_result = engine.run(continuation_strategy, merged)
    continuation_signals = continuation_strategy.generate_signals(merged)

    combined_signals = _merge_lane_signals(reversal_signals, continuation_signals)
    combined_strategy = _PrecomputedSignalsStrategy(params=params, signals=combined_signals)
    combined_result = engine.run(combined_strategy, merged)

    verdict, interpretation = _verdict(reversal_result.metrics, combined_result.metrics)
    output = {
        "analysis": "ict_combined_lanes",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "active_lite_reversal_frontier": {
            "metrics": reversal_result.metrics,
        },
        "continuation_proxy_lane": {
            "metrics": continuation_result.metrics,
        },
        "combined_lane": {
            "metrics": combined_result.metrics,
            "metadata": combined_signals.metadata,
        },
        "trade_delta_vs_reversal": int(combined_result.metrics["total_trades"]) - int(reversal_result.metrics["total_trades"]),
        "return_delta_vs_reversal_pct": round(
            float(combined_result.metrics["total_return_pct"]) - float(reversal_result.metrics["total_return_pct"]),
            4,
        ),
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
