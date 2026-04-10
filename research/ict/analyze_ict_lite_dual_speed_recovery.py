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
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_lite_dual_speed_recovery.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "fast_only_base": {
            "builder": "fast_only",
            "overrides": {},
        },
        "dual_speed_6": {
            "builder": "dual_speed",
            "overrides": {"slow_recovery_bars": 6},
        },
        "dual_speed_8": {
            "builder": "dual_speed",
            "overrides": {"slow_recovery_bars": 8},
        },
        "dual_speed_12": {
            "builder": "dual_speed",
            "overrides": {"slow_recovery_bars": 12},
        },
    }


def _build_params(builder_label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if builder_label == "dual_speed":
        return build_ict_lite_reversal_dual_speed_recovery_profile_params(
            enable_smt=True,
            overrides=dict(overrides),
        )
    return build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
        enable_smt=True,
        overrides=dict(overrides),
    )


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        metadata = payload["metadata"]
        ranked.append(
            {
                "label": label,
                "slow_recovery_enabled": bool(payload["params"]["slow_recovery_enabled"]),
                "slow_recovery_bars": int(payload["params"]["slow_recovery_bars"]),
                "total_trades": int(metrics["total_trades"]),
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
                "trade_gain_vs_base": int(metrics["total_trades"]) - base_trades,
                "fast_recovery_entries": int(metadata.get("fast_recovery_entries", 0)),
                "slow_recovery_entries": int(metadata.get("slow_recovery_entries", 0)),
                "sweep_expired_before_shift": int(metadata.get("sweep_expired_before_shift", 0)),
                "armed_setup_expired_before_retest": int(metadata.get("armed_setup_expired_before_retest", 0)),
                "delivery_missing_shifts": int(metadata.get("delivery_missing_shifts", 0)),
                "fvg_depth_filtered_retests": int(metadata.get("fvg_depth_filtered_retests", 0)),
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
    best_density_variant: dict[str, Any] | None,
    best_return_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    if best_density_variant is None:
        return (
            "DUAL_SPEED_RECOVERY_REJECTED",
            "No dual-speed recovery variant improves the active fast-only lite frontier.",
        )
    if (
        best_density_variant["total_trades"] > base_trades
        and best_density_variant["total_return_pct"] >= base_return
    ):
        return (
            "DUAL_SPEED_RECOVERY_ROBUST_EXTENSION_IDENTIFIED",
            "At least one dual-speed recovery variant improves both density and total return versus the active lite frontier.",
        )
    if best_density_variant["total_trades"] > base_trades and best_density_variant["total_return_pct"] > 0:
        return (
            "DUAL_SPEED_RECOVERY_DENSITY_EXTENSION_ONLY",
            "Dual-speed recovery improves density materially, but the higher-activity branch still trails the active lite frontier's fast-only return profile.",
        )
    if best_return_variant is not None and best_return_variant["label"] != "fast_only_base":
        return (
            "DUAL_SPEED_RECOVERY_SURVIVOR_BUT_NOT_EXTENSION",
            "A dual-speed recovery variant remains positive, but it does not improve enough over the active lite frontier to justify promotion.",
        )
    return (
        "DUAL_SPEED_RECOVERY_FAST_ONLY_REMAINS_BEST",
        "The active fast-only lite frontier remains the cleanest branch after dual-speed recovery comparison.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fast-only and dual-speed recovery variants on the active lite ICT frontier."
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
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
                "slow_recovery_enabled": bool(params["slow_recovery_enabled"]),
                "slow_recovery_bars": int(params["slow_recovery_bars"]),
                "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
                "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
                "min_reward_risk_ratio": float(params["min_reward_risk_ratio"]),
            },
            "metrics": result.metrics,
            "metadata": {
                key: int(signals.metadata.get(key, 0))
                for key in (
                    "fast_recovery_shifts",
                    "slow_recovery_shifts",
                    "fast_recovery_entries",
                    "slow_recovery_entries",
                    "bullish_shifts",
                    "bearish_shifts",
                    "bullish_retest_candidates",
                    "bearish_retest_candidates",
                    "delivery_missing_shifts",
                    "fvg_depth_filtered_retests",
                    "fvg_delay_filtered_retests",
                    "sweep_expired_before_shift",
                    "armed_setup_expired_before_retest",
                    "rr_filtered_entries",
                )
            },
        }

    base_label = "fast_only_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_return = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best_density_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] > base_trades
            and row["total_return_pct"] > 0
        ),
        None,
    )
    best_return_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label and row["total_return_pct"] > 0
        ),
        None,
    )
    verdict, interpretation = _verdict(
        base_trades=base_trades,
        base_return=base_return,
        best_density_variant=best_density_variant,
        best_return_variant=best_return_variant,
    )

    output = {
        "analysis": "ict_lite_dual_speed_recovery",
        "profile": "dual_speed_reversal_compare",
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
        "base_label": base_label,
        "base_trades": base_trades,
        "base_total_return_pct": base_return,
        "best_density_variant": best_density_variant,
        "best_return_variant": best_return_variant,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
