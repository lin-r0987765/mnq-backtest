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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_score_quality_repair.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "lite_frontier_base": {
            "overrides": {
                "score_sweep_depth_quality": 0.0,
                "score_displacement_quality": 0.0,
                "score_fvg_gap_quality": 0.0,
                "min_score_to_trade": 6.0,
            }
        },
        "quality_score_6": {
            "overrides": {
                "score_sweep_depth_quality": 1.0,
                "score_displacement_quality": 1.0,
                "score_fvg_gap_quality": 1.0,
                "min_score_to_trade": 6.0,
            }
        },
        "quality_score_7": {
            "overrides": {
                "score_sweep_depth_quality": 1.0,
                "score_displacement_quality": 1.0,
                "score_fvg_gap_quality": 1.0,
                "min_score_to_trade": 7.0,
            }
        },
        "quality_score_8": {
            "overrides": {
                "score_sweep_depth_quality": 1.0,
                "score_displacement_quality": 1.0,
                "score_fvg_gap_quality": 1.0,
                "min_score_to_trade": 8.0,
            }
        },
        "quality_score_9": {
            "overrides": {
                "score_sweep_depth_quality": 1.0,
                "score_displacement_quality": 1.0,
                "score_fvg_gap_quality": 1.0,
                "min_score_to_trade": 9.0,
            }
        },
        "quality_score_10": {
            "overrides": {
                "score_sweep_depth_quality": 1.0,
                "score_displacement_quality": 1.0,
                "score_fvg_gap_quality": 1.0,
                "min_score_to_trade": 10.0,
            }
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        ranked.append(
            {
                "label": label,
                "min_score_to_trade": float(payload["params"]["min_score_to_trade"]),
                "score_sweep_depth_quality": float(payload["params"]["score_sweep_depth_quality"]),
                "score_displacement_quality": float(payload["params"]["score_displacement_quality"]),
                "score_fvg_gap_quality": float(payload["params"]["score_fvg_gap_quality"]),
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_gain_vs_base": int(metrics["total_trades"]) - base_trades,
                "bullish_shifts": int(metadata["bullish_shifts"]),
                "bearish_shifts": int(metadata["bearish_shifts"]),
                "score_filtered_shifts": int(metadata["score_filtered_shifts"]),
                "score_quality_boosted_shifts": int(metadata["score_quality_boosted_shifts"]),
                "score_quality_bonus_total": float(metadata["score_quality_bonus_total"]),
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
    base_trades: int,
    base_return: float,
    base_score_filtered_shifts: int,
    best_non_base: dict[str, Any] | None,
    best_repaired_variant: dict[str, Any] | None,
    best_positive_density_variant: dict[str, Any] | None,
    best_robust_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_robust_variant is not None:
        return (
            "ROBUST_LITE_SCORE_QUALITY_EXTENSION_IDENTIFIED",
            "A quality-aware score variant improves both trade density and total return on the active lite frontier.",
        )
    if best_non_base is None:
        return (
            "LITE_SCORE_QUALITY_NO_POSITIVE_VARIANTS",
            "No positive quality-aware score variant survives on top of the active lite frontier.",
        )
    if best_repaired_variant is not None:
        return (
            "QUALITY_SCORE_SYSTEM_REPAIRED_BUT_NOT_EXTENSION",
            "The new quality-aware score finally filters weak shifts, but it does not improve the active lite frontier enough to promote.",
        )
    if best_positive_density_variant is not None and best_positive_density_variant["total_trades"] > base_trades:
        return (
            "LITE_SCORE_QUALITY_DENSITY_EXTENSION_ONLY",
            "A quality-aware score variant improves density on the active lite frontier, but not enough to replace the base cleanly.",
        )
    if best_non_base["total_return_pct"] > 0:
        return (
            "LITE_SCORE_QUALITY_SURVIVOR_BUT_NOT_EXTENSION",
            "Positive quality-aware score variants survive on the active lite frontier, but none improve it enough to promote.",
        )
    return (
        "LITE_SCORE_QUALITY_CALIBRATION_REJECTED",
        "The tested quality-aware score variants do not improve the active lite frontier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair the active lite ICT score system so setup quality can materially affect shift gating."
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
                "min_score_to_trade": float(params["min_score_to_trade"]),
                "score_sweep_depth_quality": float(params["score_sweep_depth_quality"]),
                "score_displacement_quality": float(params["score_displacement_quality"]),
                "score_fvg_gap_quality": float(params["score_fvg_gap_quality"]),
                "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
                "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            },
            "metrics": result.metrics,
            "metadata": {
                key: signals.metadata.get(key, 0)
                for key in (
                    "bullish_shifts",
                    "bearish_shifts",
                    "score_filtered_shifts",
                    "score_quality_boosted_shifts",
                    "score_quality_bonus_total",
                )
            },
        }

    base_label = "lite_frontier_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_return = float(variants[base_label]["metrics"]["total_return_pct"])
    base_score_filtered_shifts = int(variants[base_label]["metadata"]["score_filtered_shifts"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    best_repaired_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label and row["score_filtered_shifts"] > base_score_filtered_shifts
        ),
        None,
    )
    best_positive_density_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] > base_trades
            and row["total_return_pct"] > 0
        ),
        None,
    )
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
        base_score_filtered_shifts=base_score_filtered_shifts,
        best_non_base=best_non_base,
        best_repaired_variant=best_repaired_variant,
        best_positive_density_variant=best_positive_density_variant,
        best_robust_variant=best_robust_variant,
    )

    output = {
        "analysis": "ict_lite_score_quality_repair",
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
        "base_score_filtered_shifts": base_score_filtered_shifts,
        "best_non_base_variant": best_non_base,
        "best_repaired_variant": best_repaired_variant,
        "best_positive_density_variant": best_positive_density_variant,
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
