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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_paired_activation_frontier.json"


def _minimal_structure_overrides(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "use_kill_zones": False,
        "use_daily_bias_filter": False,
        "use_premium_discount_filter": False,
        "use_external_liquidity_filter": False,
        "use_amd_filter": False,
        "use_macro_timing_windows": False,
        "use_prev_session_anchor_filter": False,
        "use_session_array_refinement": False,
    }
    if extra:
        overrides.update(extra)
    return overrides


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "full_stack_smt": {"enable_smt": True, "overrides": {}},
        "context_relaxed_bundle": {
            "enable_smt": True,
            "overrides": _minimal_structure_overrides(),
        },
        "minimal_structure_default": {
            "enable_smt": False,
            "overrides": _minimal_structure_overrides(),
        },
        "minimal_structure_loose_sweep": {
            "enable_smt": False,
            "overrides": _minimal_structure_overrides(
                {
                    "liq_sweep_lookback": 20,
                    "liq_sweep_threshold": 0.0005,
                }
            ),
        },
        "minimal_structure_very_loose": {
            "enable_smt": False,
            "overrides": _minimal_structure_overrides(
                {
                    "min_score_to_trade": 3,
                    "liq_sweep_lookback": 10,
                    "liq_sweep_threshold": 0.00025,
                }
            ),
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": label,
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "bullish_sweeps": int(payload["metadata"].get("bullish_sweeps", 0)),
                "bearish_sweeps": int(payload["metadata"].get("bearish_sweeps", 0)),
                "fvg_entries": int(payload["metadata"].get("fvg_entries", 0)),
                "ob_entries": int(payload["metadata"].get("ob_entries", 0)),
                "breaker_entries": int(payload["metadata"].get("breaker_entries", 0)),
                "ifvg_entries": int(payload["metadata"].get("ifvg_entries", 0)),
            }
        )

    ranked.sort(
        key=lambda row: (
            row["total_trades"] > 0,
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose what profile relaxation is required to activate ICT trades on paired data.")
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
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
                "daily_bias_filtered_setups": int(signals.metadata.get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(signals.metadata.get("premium_discount_filtered_setups", 0)),
                "amd_filtered_setups": int(signals.metadata.get("amd_filtered_setups", 0)),
                "macro_timing_filtered_sweeps": int(signals.metadata.get("macro_timing_filtered_sweeps", 0)),
                "session_array_filtered_shifts": int(signals.metadata.get("session_array_filtered_shifts", 0)),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
            },
        }

    ranked = _rank_variants(variants)
    best = ranked[0] if ranked else None
    loose = variants["minimal_structure_loose_sweep"]
    very_loose = variants["minimal_structure_very_loose"]

    verdict = "PAIRED_SAMPLE_REMAINS_FULLY_INACTIVE"
    interpretation = (
        "Even aggressive relaxation still produces no ICT trades on the available paired sample, so the next step must be broader paired data rather than feature layering."
    )
    context_relaxed = variants["context_relaxed_bundle"]
    minimal_default = variants["minimal_structure_default"]
    if int(context_relaxed["metrics"]["total_trades"]) > 0 and float(context_relaxed["metrics"]["total_return_pct"]) > 0:
        verdict = "BROADER_PAIRED_SAMPLE_REVEALS_ACTIVE_RELAXED_PROFILE"
        interpretation = (
            "The zero-trade issue is not structural deadness of the ICT lane. On the broader paired Alpaca sample, the full stack still produces zero trades, but a context-relaxed profile becomes active and positive. The next step should focus on calibrating which context filters can be reintroduced without killing activity, not on adding another conceptual feature."
        )
    elif int(minimal_default["metrics"]["total_trades"]) > 0 or int(loose["metrics"]["total_trades"]) > 0 or int(very_loose["metrics"]["total_trades"]) > 0:
        verdict = "ACTIVATION_REQUIRES_SWEEP_RELAXATION_BUT_QUALITY_REMAINS_WEAK"
        interpretation = (
            "The zero-trade issue is not purely caused by contextual filters. The paired sample only starts producing ICT trades after materially loosening sweep geometry, and the resulting sample is still weak. The next step should focus on broader paired-data calibration or sweep-geometry research, not another conceptual feature layer."
        )

    output = {
        "analysis": "ict_paired_activation_frontier",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "best_variant": best,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
