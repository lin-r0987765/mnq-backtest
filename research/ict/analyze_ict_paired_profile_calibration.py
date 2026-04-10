from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_research_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_paired_profile_calibration.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    context_relaxed_bundle = {
        "use_kill_zones": False,
        "use_daily_bias_filter": False,
        "use_premium_discount_filter": False,
        "use_external_liquidity_filter": False,
        "use_amd_filter": False,
        "use_macro_timing_windows": False,
        "use_prev_session_anchor_filter": False,
        "use_session_array_refinement": False,
    }
    return {
        "full_stack_smt": {"enable_smt": True, "overrides": {}},
        "no_smt": {"enable_smt": False, "overrides": {}},
        "no_daily_bias": {"enable_smt": True, "overrides": {"use_daily_bias_filter": False}},
        "no_premium_discount": {"enable_smt": True, "overrides": {"use_premium_discount_filter": False}},
        "no_amd": {"enable_smt": True, "overrides": {"use_amd_filter": False}},
        "no_macro": {"enable_smt": True, "overrides": {"use_macro_timing_windows": False}},
        "no_prev_session_anchor": {"enable_smt": True, "overrides": {"use_prev_session_anchor_filter": False}},
        "no_session_array": {"enable_smt": True, "overrides": {"use_session_array_refinement": False}},
        "no_external_liquidity": {"enable_smt": True, "overrides": {"use_external_liquidity_filter": False}},
        "no_kill_zones": {"enable_smt": True, "overrides": {"use_kill_zones": False}},
        "context_relaxed_bundle": {
            "enable_smt": True,
            "overrides": context_relaxed_bundle,
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        scored.append(
            {
                "label": label,
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "daily_bias_filtered_setups": int(payload["metadata"].get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(payload["metadata"].get("premium_discount_filtered_setups", 0)),
                "amd_filtered_setups": int(payload["metadata"].get("amd_filtered_setups", 0)),
                "macro_timing_filtered_sweeps": int(payload["metadata"].get("macro_timing_filtered_sweeps", 0)),
                "session_array_filtered_shifts": int(payload["metadata"].get("session_array_filtered_shifts", 0)),
                "smt_filtered_sweeps": int(payload["metadata"].get("smt_filtered_sweeps", 0)),
                "smt_missing_peer_bars": int(payload["metadata"].get("smt_missing_peer_bars", 0)),
            }
        )

    scored.sort(
        key=lambda row: (
            row["total_trades"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the ICT full-stack profile on paired primary/peer data.")
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(
        peer_symbol=args.peer_symbol,
        peer_csv=args.peer_csv,
        period=args.period,
    )
    merged = merge_peer_columns(intraday_df, peer_df)

    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in _variant_specs().items():
        params = build_ict_research_profile_params(
            enable_smt=bool(spec["enable_smt"]),
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "use_smt_filter": bool(params["use_smt_filter"]),
                **spec["overrides"],
            },
            "metrics": result.metrics,
            "metadata": {
                "bullish_sweeps": int(signals.metadata.get("bullish_sweeps", 0)),
                "bearish_sweeps": int(signals.metadata.get("bearish_sweeps", 0)),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
                "smt_filtered_sweeps": int(signals.metadata.get("smt_filtered_sweeps", 0)),
                "smt_missing_peer_bars": int(signals.metadata.get("smt_missing_peer_bars", 0)),
                "daily_bias_filtered_setups": int(signals.metadata.get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(signals.metadata.get("premium_discount_filtered_setups", 0)),
                "amd_filtered_setups": int(signals.metadata.get("amd_filtered_setups", 0)),
                "prev_session_anchor_filtered_setups": int(signals.metadata.get("prev_session_anchor_filtered_setups", 0)),
                "macro_timing_filtered_sweeps": int(signals.metadata.get("macro_timing_filtered_sweeps", 0)),
                "session_array_filtered_shifts": int(signals.metadata.get("session_array_filtered_shifts", 0)),
                "external_liquidity_filtered_sweeps": int(signals.metadata.get("external_liquidity_filtered_sweeps", 0)),
                "kill_zone_filtered_sweeps": int(signals.metadata.get("kill_zone_filtered_sweeps", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    ranked = _rank_variants(variants)
    best = ranked[0] if ranked else None
    full_stack = variants["full_stack_smt"]
    output = {
        "analysis": "ict_paired_profile_calibration",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "full_stack_smt": variants["full_stack_smt"],
        "variant_ranking": ranked,
        "best_variant": best,
        "verdict": (
            "PAIR_CALIBRATION_IDENTIFIED_A_HIGHER_ACTIVITY_PROFILE"
            if best and best["label"] != "full_stack_smt" and best["total_trades"] > 0
            else "FULL_STACK_REMAINS_TOO_RESTRICTIVE_ON_AVAILABLE_PAIRED_SAMPLE"
        ),
        "interpretation": (
            "The paired-data calibration found at least one relaxed profile with trades, so the next step should focus on calibrating the active ICT stack rather than adding another conceptual feature."
            if best and best["label"] != "full_stack_smt" and best["total_trades"] > 0
            else "The current full-stack ICT profile remains too restrictive on the available paired sample. More peer history or broader paired data is likely needed before adding new conceptual layers."
        ),
        "full_stack_zero_trade": int(full_stack["metrics"]["total_trades"]) == 0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
