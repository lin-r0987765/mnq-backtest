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
    build_ict_lite_reversal_quick_swing_structure_repair_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_swing_structure_repair.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "active_lite_frontier": {"builder": "active_lite", "overrides": {}},
        "quick_density_rolling": {"builder": "quick_density", "overrides": {}},
        "quick_swing_2": {"builder": "quick_swing", "overrides": {"swing_threshold": 2}},
        "quick_swing_3": {"builder": "quick_swing", "overrides": {"swing_threshold": 3}},
        "quick_swing_4": {"builder": "quick_swing", "overrides": {"swing_threshold": 4}},
    }


def _build_params(builder_label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if builder_label == "active_lite":
        return build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    if builder_label == "quick_swing":
        return build_ict_lite_reversal_quick_swing_structure_repair_profile_params(
            enable_smt=False,
            overrides=dict(overrides),
        )
    return build_ict_lite_reversal_quick_density_repair_profile_params(
        enable_smt=False,
        overrides=dict(overrides),
    )


def _run_variant(merged, *, label: str, params: dict[str, Any], engine: BacktestEngine) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "label": label,
        "params": {
            "structure_reference_mode": str(params["structure_reference_mode"]),
            "swing_threshold": int(params["swing_threshold"]),
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "use_smt_filter": bool(params["use_smt_filter"]),
        },
        "metrics": result.metrics,
        "funnel": _build_funnel_summary(signals.metadata),
        "metadata": {
            "swing_structure_missing_reference": int(signals.metadata.get("swing_structure_missing_reference", 0)),
            "sweep_expired_before_shift": int(signals.metadata.get("sweep_expired_before_shift", 0)),
            "delivery_missing_shifts": int(signals.metadata.get("delivery_missing_shifts", 0)),
            "fvg_delay_filtered_retests": int(signals.metadata.get("fvg_delay_filtered_retests", 0)),
            "fvg_depth_filtered_retests": int(signals.metadata.get("fvg_depth_filtered_retests", 0)),
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "label": label,
                "structure_reference_mode": payload["params"]["structure_reference_mode"],
                "swing_threshold": payload["params"]["swing_threshold"],
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
            }
        )
    rows.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return rows


def _verdict(
    *,
    active_lite: dict[str, Any],
    quick_density: dict[str, Any],
    best_swing_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_swing_variant is None:
        return (
            "SWING_STRUCTURE_REPAIR_REJECTED",
            "No swing-structure variant improves the quick density branch enough to justify further promotion.",
        )

    active_trades = int(active_lite["metrics"]["total_trades"])
    quick_trades = int(quick_density["metrics"]["total_trades"])
    quick_return = float(quick_density["metrics"]["total_return_pct"])
    swing_trades = int(best_swing_variant["metrics"]["total_trades"])
    swing_return = float(best_swing_variant["metrics"]["total_return_pct"])

    if swing_trades >= 100 and swing_return > 0:
        return (
            "SWING_STRUCTURE_REPAIR_CLEARS_FIRST_DENSITY_GATE",
            "A swing-structure variant rescues expectancy while preserving 100+ trades.",
        )
    if swing_trades > active_trades and swing_return > 0:
        return (
            "SWING_STRUCTURE_REPAIR_POSITIVE_DENSITY_EXTENSION_IDENTIFIED",
            "A swing-structure variant restores positive expectancy while still improving density over the active lite frontier.",
        )
    if swing_return > quick_return and swing_trades >= max(active_trades, quick_trades // 2):
        return (
            "SWING_STRUCTURE_REPAIR_EXPECTANCY_RECOVERY_BUT_NOT_GATE_CLEAR",
            "Swing-structure repair materially improves the quick density branch, but it still does not clear a promotable density gate.",
        )
    return (
        "SWING_STRUCTURE_REPAIR_STILL_NOT_READY",
        "Swing-structure repair improves the branch only partially and still does not justify promotion.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay swing-structure architecture repairs on top of the quick density branch."
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

    variants: dict[str, dict[str, Any]] = {}
    for label, spec in _variant_specs().items():
        params = _build_params(spec["builder"], spec["overrides"])
        variants[label] = _run_variant(merged, label=label, params=params, engine=engine)

    ranked = _rank_variants(variants)
    best_swing_variant = next(
        (variants[row["label"]] for row in ranked if row["label"].startswith("quick_swing_")),
        None,
    )
    verdict, interpretation = _verdict(
        active_lite=variants["active_lite_frontier"],
        quick_density=variants["quick_density_rolling"],
        best_swing_variant=best_swing_variant,
    )

    output = {
        "analysis": "ict_swing_structure_repair",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "best_swing_variant": None if best_swing_variant is None else best_swing_variant["label"],
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
