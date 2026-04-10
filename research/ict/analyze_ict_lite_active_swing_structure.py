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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_active_swing_structure.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "active_lite_rolling_base": {
            "overrides": {
                "structure_reference_mode": "rolling",
            }
        },
        "active_lite_swing_1": {
            "overrides": {
                "structure_reference_mode": "swing",
                "daily_bias_mode": "statistical",
                "swing_threshold": 1,
            }
        },
        "active_lite_swing_2": {
            "overrides": {
                "structure_reference_mode": "swing",
                "swing_threshold": 2,
            }
        },
        "active_lite_swing_3": {
            "overrides": {
                "structure_reference_mode": "swing",
                "swing_threshold": 3,
            }
        },
        "active_lite_swing_4": {
            "overrides": {
                "structure_reference_mode": "swing",
                "swing_threshold": 4,
            }
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": label,
                "structure_reference_mode": str(payload["params"]["structure_reference_mode"]),
                "swing_threshold": int(payload["params"]["swing_threshold"]),
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "trade_gain_vs_base": int(metrics["total_trades"]) - base_trades,
                "swing_structure_missing_reference": int(payload["metadata"]["swing_structure_missing_reference"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["total_trades"],
        ),
        reverse=True,
    )
    return ranked


def _verdict(
    *,
    base_trades: int,
    base_return: float,
    best_non_base: dict[str, Any] | None,
    best_positive_density_variant: dict[str, Any] | None,
    best_robust_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_robust_variant is not None:
        return (
            "ROBUST_ACTIVE_LITE_SWING_STRUCTURE_EXTENSION_IDENTIFIED",
            "A swing-structure variant improves both density and return on the active lite frontier.",
        )
    if best_non_base is None:
        return (
            "ACTIVE_LITE_SWING_STRUCTURE_NO_POSITIVE_VARIANTS",
            "No active-lite swing-structure variant survives on top of the rolling base.",
        )
    if best_positive_density_variant is not None and best_positive_density_variant["total_trades"] > base_trades:
        return (
            "ACTIVE_LITE_SWING_STRUCTURE_DENSITY_EXTENSION_ONLY",
            "A swing-structure variant improves density on the active lite frontier, but not enough to replace the rolling base cleanly.",
        )
    if best_non_base["total_return_pct"] > 0:
        return (
            "ACTIVE_LITE_SWING_STRUCTURE_SURVIVOR_BUT_NOT_EXTENSION",
            "Positive swing-structure variants survive on the active lite frontier, but none improve it enough to promote.",
        )
    if best_non_base["total_return_pct"] > base_return:
        return (
            "ACTIVE_LITE_SWING_STRUCTURE_EXPECTANCY_RECOVERY_ONLY",
            "Swing structure improves expectancy versus weaker branches, but still does not beat the active rolling lite frontier.",
        )
    return (
        "ACTIVE_LITE_SWING_STRUCTURE_REJECTED",
        "The tested swing-structure variants do not improve the active lite frontier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay swing-structure variants directly on the active lite ICT frontier."
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
                "structure_reference_mode": str(params["structure_reference_mode"]),
                "swing_threshold": int(params["swing_threshold"]),
                "structure_lookback": int(params["structure_lookback"]),
                "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
                "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
                "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
                "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            },
            "metrics": result.metrics,
            "funnel": _build_funnel_summary(signals.metadata),
            "metadata": {
                "swing_structure_missing_reference": int(signals.metadata.get("swing_structure_missing_reference", 0)),
                "bullish_shift_candidates": int(signals.metadata.get("bullish_shift_candidates", 0)),
                "bearish_shift_candidates": int(signals.metadata.get("bearish_shift_candidates", 0)),
                "delivery_missing_shifts": int(signals.metadata.get("delivery_missing_shifts", 0)),
                "score_filtered_shifts": int(signals.metadata.get("score_filtered_shifts", 0)),
            },
        }

    base_label = "active_lite_rolling_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_return = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
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
        best_non_base=best_non_base,
        best_positive_density_variant=best_positive_density_variant,
        best_robust_variant=best_robust_variant,
    )

    output = {
        "analysis": "ict_lite_active_swing_structure",
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
