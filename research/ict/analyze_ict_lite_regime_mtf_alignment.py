from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_qualified_reversal_balance_profile_params,
    build_ict_lite_reversal_regime_mtf_alignment_profile_params,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_lite_regime_mtf_alignment.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "active_lite_quality_base": {
            "builder": "quality",
            "overrides": {},
        },
        "qualified_reversal_balance": {
            "builder": "balance",
            "overrides": {},
        },
        "regime_mtf_hard": {
            "builder": "regime",
            "overrides": {},
        },
        "regime_mtf_soft": {
            "builder": "regime",
            "overrides": {
                "higher_timeframe_alignment_mode": "soft",
                "higher_timeframe_mismatch_score_penalty": 1.5,
            },
        },
    }


def _build_params(builder_label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if builder_label == "balance":
        return build_ict_lite_reversal_qualified_reversal_balance_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    if builder_label == "regime":
        return build_ict_lite_reversal_regime_mtf_alignment_profile_params(
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
            "use_regime_adaptation": bool(params.get("use_regime_adaptation", False)),
            "use_higher_timeframe_alignment": bool(params.get("use_higher_timeframe_alignment", False)),
            "higher_timeframe_alignment_mode": str(params.get("higher_timeframe_alignment_mode", "hard")),
            "regime_high_fvg_min_gap_pct": float(params.get("regime_high_fvg_min_gap_pct", 0.0)),
            "regime_high_fvg_revisit_depth_ratio": float(
                params.get("regime_high_fvg_revisit_depth_ratio", 0.0)
            ),
            "regime_high_smt_threshold": float(params.get("regime_high_smt_threshold", 0.0)),
        },
        "metrics": result.metrics,
        "metadata": {
            key: int(signals.metadata.get(key, 0))
            for key in (
                "high_regime_bars",
                "higher_timeframe_filtered_setups",
                "higher_timeframe_softened_setups",
                "higher_timeframe_score_penalty_shifts",
                "fvg_depth_filtered_retests",
                "fvg_delay_filtered_retests",
                "smt_filtered_sweeps",
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
                "high_regime_bars": int(metadata["high_regime_bars"]),
                "higher_timeframe_filtered_setups": int(metadata["higher_timeframe_filtered_setups"]),
                "higher_timeframe_softened_setups": int(metadata["higher_timeframe_softened_setups"]),
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
    quality_base: dict[str, Any],
    reversal_balance: dict[str, Any],
    regime_best: dict[str, Any],
) -> tuple[str, str]:
    if (
        regime_best["total_return_pct"] > 0
        and regime_best["total_trades"] > quality_base["total_trades"]
        and regime_best["total_return_pct"] >= reversal_balance["total_return_pct"]
    ):
        return (
            "REGIME_MTF_REVERSAL_BRANCH_IDENTIFIED",
            "Regime adaptation plus 1H alignment creates a higher-trade reversal branch that keeps return quality at or above the existing balanced reversal lane.",
        )
    if regime_best["total_return_pct"] > 0 and regime_best["total_trades"] > quality_base["total_trades"]:
        return (
            "REGIME_MTF_REVERSAL_SURVIVOR_ONLY",
            "Regime adaptation plus 1H alignment improves trade count over the sparse quality lane, but it does not yet beat the balanced reversal branch on return quality.",
        )
    return (
        "REGIME_MTF_REVERSAL_REJECTED",
        "Regime adaptation plus 1H alignment does not currently improve the reversal lane enough to justify promotion.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a regime-aware 1H-aligned ICT reversal branch against the current quality and balanced reversal lanes."
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
    quality_base = next(row for row in ranked if row["label"] == "active_lite_quality_base")
    reversal_balance = next(row for row in ranked if row["label"] == "qualified_reversal_balance")
    regime_candidates = [row for row in ranked if row["label"].startswith("regime_mtf_")]
    regime_best = regime_candidates[0]
    verdict, interpretation = _verdict(
        quality_base=quality_base,
        reversal_balance=reversal_balance,
        regime_best=regime_best,
    )

    output = {
        "analysis": "ict_lite_regime_mtf_alignment",
        "profile": "regime_mtf_reversal_compare",
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
        "variants": variants,
        "variant_ranking": ranked,
        "quality_base": quality_base,
        "qualified_reversal_balance": reversal_balance,
        "best_regime_variant": regime_best,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
