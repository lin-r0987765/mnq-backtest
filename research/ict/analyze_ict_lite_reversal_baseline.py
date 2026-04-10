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
    build_ict_lite_reversal_profile_params,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_reversal_baseline.json"
RESEARCH_STANDARD = {
    "initial_cash": 10_000.0,
    "fees_pct": 0.0005,
    "position_size_mode": "capital_pct",
    "capital_usage_pct": 1.0,
    "min_shares": 0,
}


def _engine_config_payload(engine: BacktestEngine) -> dict[str, Any]:
    return {
        "initial_cash": float(engine.initial_cash),
        "fees_pct": float(engine.fees_pct),
        "position_size_mode": str(engine.position_size_mode),
        "fixed_shares": None if engine.fixed_shares is None else int(engine.fixed_shares),
        "capital_usage_pct": float(engine.capital_usage_pct),
        "min_shares": int(engine.min_shares),
    }


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
            "use_premium_discount_filter": bool(params["use_premium_discount_filter"]),
            "use_external_liquidity_filter": bool(params["use_external_liquidity_filter"]),
            "use_prev_session_anchor_filter": bool(params["use_prev_session_anchor_filter"]),
            "use_session_array_refinement": bool(params["use_session_array_refinement"]),
            "trade_sessions": bool(params["trade_sessions"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "structure_lookback": int(params["structure_lookback"]),
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
        "funnel": _build_funnel_summary(signals.metadata),
    }


def _density_summary(strict_payload: dict[str, Any], lite_payload: dict[str, Any]) -> dict[str, Any]:
    strict_trades = int(strict_payload["metrics"]["total_trades"])
    lite_trades = int(lite_payload["metrics"]["total_trades"])
    strict_return = float(strict_payload["metrics"]["total_return_pct"])
    lite_return = float(lite_payload["metrics"]["total_return_pct"])
    return {
        "strict_trades": strict_trades,
        "lite_trades": lite_trades,
        "trade_gain": lite_trades - strict_trades,
        "trade_gain_multiple": round((lite_trades / strict_trades), 2) if strict_trades > 0 else 0.0,
        "strict_total_return_pct": strict_return,
        "lite_total_return_pct": lite_return,
        "return_delta_pct": round(lite_return - strict_return, 4),
        "toward_100_trade_gate": max(0, 100 - lite_trades),
    }


def _verdict(density: dict[str, Any], lite_payload: dict[str, Any]) -> tuple[str, str]:
    lite_trades = int(density["lite_trades"])
    lite_return = float(lite_payload["metrics"]["total_return_pct"])
    if lite_trades >= 100 and lite_return > 0:
        return (
            "LITE_ICT_REVERSAL_BASELINE_CLEARS_100_TRADE_GATE",
            "The lighter reversal lane clears the first 500-trades roadmap density gate while staying profitable.",
        )
    if lite_trades > int(density["strict_trades"]) and lite_return > 0:
        return (
            "LITE_ICT_REVERSAL_BASELINE_IMPROVES_DENSITY_BUT_STAYS_BELOW_100_TRADES",
            "The lighter reversal lane materially improves density versus the strict frontier, but it still does not clear the first 100-trade gate.",
        )
    if lite_trades > int(density["strict_trades"]):
        return (
            "LITE_ICT_REVERSAL_BASELINE_RAISES_DENSITY_WITHOUT_PROFITABLE_EDGE",
            "The lighter reversal lane increases activity versus the strict frontier, but the baseline is not yet strong enough to progress.",
        )
    return (
        "LITE_ICT_REVERSAL_BASELINE_FAILS_TO_IMPROVE_DENSITY",
        "Removing the heaviest pre-arm context blockers does not yet improve the strict frontier enough to justify the lite branch.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 500-trades roadmap lite ICT reversal baseline against the strict frontier."
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

    strict_params = (
        build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
            enable_smt=True
        )
    )
    lite_params = build_ict_lite_reversal_profile_params(enable_smt=True)

    strict_payload = _run_profile(
        merged,
        profile_name="strict_ict_frontier",
        params=strict_params,
        engine=engine,
    )
    lite_payload = _run_profile(
        merged,
        profile_name="lite_ict_reversal",
        params=lite_params,
        engine=engine,
    )

    density = _density_summary(strict_payload, lite_payload)
    verdict, interpretation = _verdict(density, lite_payload)

    output = {
        "analysis": "ict_lite_reversal_baseline",
        "risk_standard": {
            "reward_risk_gate": ">= 1.5:1",
            "engine": _engine_config_payload(engine),
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "strict_frontier": strict_payload,
        "lite_reversal": lite_payload,
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
