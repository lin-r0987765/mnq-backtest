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
    build_ict_lite_reversal_quick_density_repair_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_quick_density_repair.json"


def _run_profile(
    merged,
    *,
    profile_name: str,
    params: dict[str, Any],
    engine: BacktestEngine,
) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "profile": profile_name,
        "params": {
            "use_smt_filter": bool(params["use_smt_filter"]),
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "use_premium_discount_filter": bool(params["use_premium_discount_filter"]),
            "use_prev_session_anchor_filter": bool(params["use_prev_session_anchor_filter"]),
            "use_external_liquidity_filter": bool(params["use_external_liquidity_filter"]),
            "use_session_array_refinement": bool(params["use_session_array_refinement"]),
            "trade_sessions": bool(params["trade_sessions"]),
        },
        "metrics": result.metrics,
        "delivery_breakdown": {
            "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
            "ob_entries": int(signals.metadata.get("ob_entries", 0)),
            "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
            "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
        },
        "funnel": _build_funnel_summary(signals.metadata),
    }


def _density_summary(
    strict_payload: dict[str, Any],
    active_lite_payload: dict[str, Any],
    quick_payload: dict[str, Any],
) -> dict[str, Any]:
    strict_trades = int(strict_payload["metrics"]["total_trades"])
    lite_trades = int(active_lite_payload["metrics"]["total_trades"])
    quick_trades = int(quick_payload["metrics"]["total_trades"])
    return {
        "strict_trades": strict_trades,
        "active_lite_trades": lite_trades,
        "quick_repair_trades": quick_trades,
        "gain_vs_strict": quick_trades - strict_trades,
        "gain_vs_active_lite": quick_trades - lite_trades,
        "multiple_vs_strict": round((quick_trades / strict_trades), 2) if strict_trades > 0 else 0.0,
        "multiple_vs_active_lite": round((quick_trades / lite_trades), 2) if lite_trades > 0 else 0.0,
        "strict_total_return_pct": float(strict_payload["metrics"]["total_return_pct"]),
        "active_lite_total_return_pct": float(active_lite_payload["metrics"]["total_return_pct"]),
        "quick_repair_total_return_pct": float(quick_payload["metrics"]["total_return_pct"]),
        "toward_100_trade_gate": max(0, 100 - quick_trades),
        "toward_500_trade_gate": max(0, 500 - quick_trades),
    }


def _verdict(
    density: dict[str, Any],
    active_lite_payload: dict[str, Any],
    quick_payload: dict[str, Any],
) -> tuple[str, str]:
    quick_trades = int(density["quick_repair_trades"])
    lite_trades = int(density["active_lite_trades"])
    quick_return = float(quick_payload["metrics"]["total_return_pct"])
    lite_return = float(active_lite_payload["metrics"]["total_return_pct"])

    if quick_trades >= 100 and quick_return > 0:
        return (
            "QUICK_DENSITY_REPAIR_CLEARS_FIRST_DENSITY_GATE",
            "The four-change repair branch clears the first 100-trade gate while staying profitable.",
        )
    if quick_trades > lite_trades and quick_return >= lite_return:
        return (
            "QUICK_DENSITY_REPAIR_ROBUST_EXTENSION_IDENTIFIED",
            "The four-change repair branch improves both density and return versus the active lite frontier.",
        )
    if quick_trades > lite_trades and quick_return > 0:
        return (
            "QUICK_DENSITY_REPAIR_DENSITY_EXTENSION_ONLY",
            "The four-change repair branch improves density materially, but it remains a density-first branch rather than a clean promotion over the active lite frontier.",
        )
    if quick_return > 0:
        return (
            "QUICK_DENSITY_REPAIR_POSITIVE_BUT_NOT_DENSER",
            "The four-change repair branch stays positive, but it does not improve density enough over the active lite frontier.",
        )
    return (
        "QUICK_DENSITY_REPAIR_REJECTED",
        "The four-change repair branch does not improve the active lite frontier enough to justify promotion.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the four-change quick density-repair ICT branch against the strict and active lite frontiers."
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
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    strict_payload = _run_profile(
        merged,
        profile_name="strict_ict_frontier",
        params=build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
            enable_smt=True
        ),
        engine=engine,
    )
    active_lite_payload = _run_profile(
        merged,
        profile_name="active_lite_frontier",
        params=build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(enable_smt=True),
        engine=engine,
    )
    quick_payload = _run_profile(
        merged,
        profile_name="quick_density_repair",
        params=build_ict_lite_reversal_quick_density_repair_profile_params(enable_smt=False),
        engine=engine,
    )

    density = _density_summary(strict_payload, active_lite_payload, quick_payload)
    verdict, interpretation = _verdict(density, active_lite_payload, quick_payload)

    output = {
        "analysis": "ict_lite_quick_density_repair",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "strict_frontier": strict_payload,
        "active_lite_frontier": active_lite_payload,
        "quick_density_repair": quick_payload,
        "density_comparison": density,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
