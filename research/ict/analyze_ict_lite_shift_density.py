from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_shift_density.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "lite_frontier_base": {"overrides": {}},
        "structure_8": {"overrides": {"structure_lookback": 8}},
        "structure_10": {"overrides": {"structure_lookback": 10}},
        "structure_16": {"overrides": {"structure_lookback": 16}},
        "structure_20": {"overrides": {"structure_lookback": 20}},
    }


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        ranked.append(
            {
                "label": label,
                "structure_lookback": int(payload["params"]["structure_lookback"]),
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_gain_vs_base": int(metrics["total_trades"]) - base_trades,
                "shift_candidates": int(metadata["bullish_shift_candidates"]) + int(metadata["bearish_shift_candidates"]),
                "armed_setups": int(metadata["bullish_shifts"]) + int(metadata["bearish_shifts"]),
                "fvg_entries": int(metadata["fvg_entries"]),
                "ob_entries": int(metadata["ob_entries"]),
                "breaker_entries": int(metadata["breaker_entries"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_trades"],
            row["total_return_pct"],
            row["profit_factor"],
            row["shift_candidates"],
        ),
        reverse=True,
    )
    return ranked


def _verdict(
    *,
    base_trades: int,
    base_return: float,
    best_non_base: dict[str, Any] | None,
    best_robust_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_robust_variant is not None:
        return (
            "ROBUST_LITE_SHIFT_EXTENSION_IDENTIFIED",
            "At least one shift-density variant improves both activity and return versus the active lite frontier.",
        )
    if best_non_base is None:
        return (
            "LITE_SHIFT_NO_POSITIVE_VARIANTS",
            "No positive shift-density variant survived on top of the active lite frontier.",
        )
    if best_non_base["total_trades"] > base_trades and best_non_base["total_return_pct"] > 0:
        return (
            "LITE_SHIFT_DENSITY_EXTENSION_ONLY",
            "A shift-density variant improves activity on the active lite frontier, but not enough to replace the base cleanly.",
        )
    if best_non_base["total_return_pct"] > 0:
        return (
            "LITE_SHIFT_SURVIVOR_BUT_NOT_EXTENSION",
            "Positive shift-density variants survive on top of the active lite frontier, but none improve it enough to promote.",
        )
    return (
        "LITE_SHIFT_CALIBRATION_REJECTED",
        "The tested shift-density variants do not improve the active lite frontier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a shift-density calibration on top of the active lite ICT frontier."
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
        params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
            enable_smt=True,
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "structure_lookback": int(params["structure_lookback"]),
                "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
                "smt_threshold": float(params["smt_threshold"]),
                "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
                "displacement_body_min_pct": float(params["displacement_body_min_pct"]),
            },
            "metrics": result.metrics,
            "metadata": {
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
                "bullish_shift_candidates": int(signals.metadata.get("bullish_shift_candidates", 0)),
                "bearish_shift_candidates": int(signals.metadata.get("bearish_shift_candidates", 0)),
                "bullish_shifts": int(signals.metadata.get("bullish_shifts", 0)),
                "bearish_shifts": int(signals.metadata.get("bearish_shifts", 0)),
            },
        }

    base_label = "lite_frontier_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_return = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    best_robust_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] >= base_trades
            and row["total_return_pct"] > base_return
        ),
        None,
    )
    verdict, interpretation = _verdict(
        base_trades=base_trades,
        base_return=base_return,
        best_non_base=best_non_base,
        best_robust_variant=best_robust_variant,
    )

    output = {
        "analysis": "ict_lite_shift_density",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "base_label": base_label,
        "base_trades": base_trades,
        "base_total_return_pct": base_return,
        "best_non_base_variant": best_non_base,
        "best_robust_variant": best_robust_variant,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
