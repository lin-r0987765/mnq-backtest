from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_dual_speed_recovery_profile_params,
    build_ict_lite_reversal_qualified_continuation_density_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_lite_qualified_continuation_density.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "active_lite_frontier_base": {
            "builder": "fast_only",
            "overrides": {},
        },
        "continuation_on": {
            "builder": "fast_only",
            "overrides": {
                "enable_continuation_entry": True,
            },
        },
        "dual_speed_12_continuation": {
            "builder": "dual_speed",
            "overrides": {
                "slow_recovery_bars": 12,
                "enable_continuation_entry": True,
            },
        },
        "balanced_density_candidate": {
            "builder": "dual_speed",
            "overrides": {
                "slow_recovery_bars": 6,
                "enable_continuation_entry": True,
                "structure_lookback": 8,
                "fvg_min_gap_pct": 0.0005,
                "fvg_revisit_depth_ratio": 0.5,
            },
        },
        "qualified_continuation_density": {
            "builder": "qualified_density",
            "overrides": {},
        },
    }


def _build_params(builder_label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if builder_label == "dual_speed":
        return build_ict_lite_reversal_dual_speed_recovery_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    if builder_label == "qualified_density":
        return build_ict_lite_reversal_qualified_continuation_density_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    return build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
        enable_smt=True,
        overrides=dict(overrides),
    )


def _run_variant(merged, *, label: str, params: dict[str, Any], engine: BacktestEngine) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "label": label,
        "params": {
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "slow_recovery_enabled": bool(params["slow_recovery_enabled"]),
            "slow_recovery_bars": int(params["slow_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "enable_continuation_entry": bool(params["enable_continuation_entry"]),
            "min_reward_risk_ratio": float(params["min_reward_risk_ratio"]),
        },
        "metrics": result.metrics,
        "metadata": {
            key: int(signals.metadata.get(key, 0))
            for key in (
                "continuation_zone_refreshes",
                "continuation_entries",
                "fast_recovery_entries",
                "slow_recovery_entries",
                "sweep_expired_before_shift",
                "armed_setup_expired_before_retest",
                "delivery_missing_shifts",
                "fvg_depth_filtered_retests",
                "fvg_delay_filtered_retests",
                "bullish_entries",
                "bearish_entries",
            )
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        ranked.append(
            {
                "label": label,
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "structure_lookback": int(payload["params"]["structure_lookback"]),
                "slow_recovery_bars": int(payload["params"]["slow_recovery_bars"]),
                "fvg_min_gap_pct": float(payload["params"]["fvg_min_gap_pct"]),
                "fvg_revisit_depth_ratio": float(payload["params"]["fvg_revisit_depth_ratio"]),
                "enable_continuation_entry": bool(payload["params"]["enable_continuation_entry"]),
                "continuation_entries": int(metadata["continuation_entries"]),
                "slow_recovery_entries": int(metadata["slow_recovery_entries"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_trades"],
            row["total_return_pct"],
            row["profit_factor"],
        ),
        reverse=True,
    )
    return ranked


def _verdict(
    *,
    best_positive_density_variant: dict[str, Any] | None,
    best_balanced_variant: dict[str, Any] | None,
    density_gate: int,
) -> tuple[str, str]:
    if best_positive_density_variant is not None and best_positive_density_variant["total_trades"] >= density_gate:
        return (
            "QUALIFIED_CONTINUATION_DENSITY_BRANCH_IDENTIFIED",
            "A positive qualified continuation branch now pushes the ICT density lane into a materially higher trade-count regime.",
        )
    if best_balanced_variant is not None:
        return (
            "QUALIFIED_CONTINUATION_BALANCED_SURVIVOR_ONLY",
            "A positive continuation-plus-recovery branch exists, but it still does not reach the intended higher-density gate.",
        )
    if best_positive_density_variant is None:
        return (
            "QUALIFIED_CONTINUATION_DENSITY_REJECTED",
            "No positive continuation-plus-recovery density branch could beat the zero-density baseline constraints.",
        )
    return (
        "QUALIFIED_CONTINUATION_DENSITY_STILL_THIN",
        "Continuation-plus-recovery remains positive, but density still trails the intended next gate.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for a positive higher-density ICT continuation branch beyond the 18-trade lite frontier."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--density-gate", type=int, default=80)
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(
        initial_cash=100_000.0,
        fees_pct=0.0005,
        position_size_mode="capital_pct",
        capital_usage_pct=1.0,
        min_shares=40,
    )

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in _variant_specs().items():
        params = _build_params(spec["builder"], spec["overrides"])
        variants[label] = _run_variant(merged, label=label, params=params, engine=engine)

    ranked = _rank_variants(variants)
    best_positive_density_variant = next(
        (row for row in ranked if row["total_trades"] >= args.density_gate and row["total_return_pct"] > 0),
        None,
    )
    balanced_candidates = [
        row for row in ranked if row["total_trades"] >= 50 and row["total_return_pct"] > 0
    ]
    best_balanced_variant = None
    if balanced_candidates:
        best_balanced_variant = max(
            balanced_candidates,
            key=lambda row: (row["total_return_pct"], row["profit_factor"], row["total_trades"]),
        )
    verdict, interpretation = _verdict(
        best_positive_density_variant=best_positive_density_variant,
        best_balanced_variant=best_balanced_variant,
        density_gate=args.density_gate,
    )

    output = {
        "analysis": "ict_lite_qualified_continuation_density",
        "profile": "qualified_continuation_density_search",
        "risk_standard": {
            "initial_cash_usd": 100000,
            "capital_usage_pct": 1.0,
            "minimum_order_shares": 40,
            "reward_risk_floor": 1.5,
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "density_gate": int(args.density_gate),
        "variants": variants,
        "variant_ranking": ranked,
        "best_positive_density_variant": best_positive_density_variant,
        "best_balanced_variant": best_balanced_variant,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
