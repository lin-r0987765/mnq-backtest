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
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
    build_ict_strict_soft_prev_session_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_prev_session_soft_filter.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "strict_prev_anchor_hard_base": {"builder": "hard", "overrides": {}},
        "prev_anchor_tolerance_0p10_hard": {
            "builder": "hard",
            "overrides": {"prev_session_anchor_tolerance": 0.10},
        },
        "prev_anchor_soft_penalty_0": {
            "builder": "soft",
            "overrides": {"prev_session_anchor_mismatch_score_penalty": 0.0},
        },
        "prev_anchor_soft_penalty_2": {
            "builder": "soft",
            "overrides": {"prev_session_anchor_mismatch_score_penalty": 2.0},
        },
        "prev_anchor_soft_penalty_3": {
            "builder": "soft",
            "overrides": {"prev_session_anchor_mismatch_score_penalty": 3.0},
        },
        "prev_anchor_filter_off_control": {
            "builder": "hard",
            "overrides": {"use_prev_session_anchor_filter": False},
        },
    }


def _build_params(builder_label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if builder_label == "soft":
        return build_ict_strict_soft_prev_session_profile_params(enable_smt=False, overrides=dict(overrides))
    return build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
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
            "use_prev_session_anchor_filter": bool(params["use_prev_session_anchor_filter"]),
            "prev_session_anchor_filter_mode": str(params["prev_session_anchor_filter_mode"]),
            "prev_session_anchor_tolerance": float(params["prev_session_anchor_tolerance"]),
            "prev_session_anchor_mismatch_score_penalty": float(
                params["prev_session_anchor_mismatch_score_penalty"]
            ),
            "use_smt_filter": bool(params["use_smt_filter"]),
        },
        "metrics": result.metrics,
        "funnel": _build_funnel_summary(signals.metadata),
        "metadata": {
            "prev_session_anchor_filtered_setups": int(
                signals.metadata.get("prev_session_anchor_filtered_setups", 0)
            ),
            "prev_session_anchor_mismatch_setups": int(
                signals.metadata.get("prev_session_anchor_mismatch_setups", 0)
            ),
            "prev_session_anchor_softened_setups": int(
                signals.metadata.get("prev_session_anchor_softened_setups", 0)
            ),
            "prev_session_anchor_score_penalty_shifts": int(
                signals.metadata.get("prev_session_anchor_score_penalty_shifts", 0)
            ),
            "score_filtered_shifts": int(signals.metadata.get("score_filtered_shifts", 0)),
            "bullish_shifts": int(signals.metadata.get("bullish_shifts", 0)),
            "bearish_shifts": int(signals.metadata.get("bearish_shifts", 0)),
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": label,
                "prev_session_anchor_filter_mode": payload["params"]["prev_session_anchor_filter_mode"],
                "prev_session_anchor_tolerance": float(payload["params"]["prev_session_anchor_tolerance"]),
                "prev_session_anchor_mismatch_score_penalty": float(
                    payload["params"]["prev_session_anchor_mismatch_score_penalty"]
                ),
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_gain_vs_base": int(metrics["total_trades"]) - base_trades,
                "prev_session_anchor_filtered_setups": int(
                    payload["metadata"]["prev_session_anchor_filtered_setups"]
                ),
                "prev_session_anchor_softened_setups": int(
                    payload["metadata"]["prev_session_anchor_softened_setups"]
                ),
                "score_filtered_shifts": int(payload["metadata"]["score_filtered_shifts"]),
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
    best_non_base: dict[str, Any] | None,
    best_positive_density_variant: dict[str, Any] | None,
    best_robust_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_robust_variant is not None:
        return (
            "ROBUST_PREV_SESSION_SOFT_FILTER_EXTENSION_IDENTIFIED",
            "A prev-session-anchor softening variant improves both activity and return on the strict context lane.",
        )
    if best_non_base is None:
        return (
            "PREV_SESSION_SOFT_FILTER_NO_POSITIVE_VARIANTS",
            "No prev-session-anchor softening variant survives on the strict context lane.",
        )
    if best_positive_density_variant is not None and best_positive_density_variant["total_trades"] > base_trades:
        return (
            "PREV_SESSION_SOFT_FILTER_DENSITY_EXTENSION_ONLY",
            "A prev-session-anchor softening variant improves density, but not cleanly enough to replace the strict base.",
        )
    if best_non_base["total_return_pct"] > 0:
        return (
            "PREV_SESSION_SOFT_FILTER_SURVIVOR_BUT_NOT_EXTENSION",
            "A prev-session-anchor softening variant stays positive, but none improve the strict base enough to promote.",
        )
    return (
        "PREV_SESSION_SOFT_FILTER_REJECTED",
        "Softening prev-session-anchor gating does not produce a promotable improvement on the strict context lane.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay prev-session-anchor soft-filter variants on the roadmap-aligned strict ICT lane."
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

    base_label = "strict_prev_anchor_hard_base"
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
        "analysis": "ict_prev_session_soft_filter",
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
