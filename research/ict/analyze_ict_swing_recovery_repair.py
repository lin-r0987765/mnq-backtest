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
    build_ict_lite_reversal_quick_swing_structure_repair_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_swing_recovery_repair.json"


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "swing_3_recovery_15": {
            "overrides": {
                "swing_threshold": 3,
                "liq_sweep_recovery_bars": 15,
                "slow_recovery_enabled": False,
                "slow_recovery_bars": 15,
            }
        },
        "swing_3_recovery_20": {
            "overrides": {
                "swing_threshold": 3,
                "liq_sweep_recovery_bars": 20,
                "slow_recovery_enabled": False,
                "slow_recovery_bars": 20,
            }
        },
        "swing_3_recovery_30": {
            "overrides": {
                "swing_threshold": 3,
                "liq_sweep_recovery_bars": 30,
                "slow_recovery_enabled": False,
                "slow_recovery_bars": 30,
            }
        },
        "swing_3_dual_12_30": {
            "overrides": {
                "swing_threshold": 3,
                "liq_sweep_recovery_bars": 12,
                "slow_recovery_enabled": True,
                "slow_recovery_bars": 30,
            }
        },
        "swing_2_dual_12_30": {
            "overrides": {
                "swing_threshold": 2,
                "liq_sweep_recovery_bars": 12,
                "slow_recovery_enabled": True,
                "slow_recovery_bars": 30,
            }
        },
    }


def _run_variant(merged, *, label: str, overrides: dict[str, Any], engine: BacktestEngine) -> dict[str, Any]:
    params = build_ict_lite_reversal_quick_swing_structure_repair_profile_params(
        enable_smt=False,
        overrides=dict(overrides),
    )
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    return {
        "label": label,
        "params": {
            "structure_reference_mode": str(params["structure_reference_mode"]),
            "swing_threshold": int(params["swing_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "slow_recovery_enabled": bool(params["slow_recovery_enabled"]),
            "slow_recovery_bars": int(params["slow_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
        },
        "metrics": result.metrics,
        "funnel": _build_funnel_summary(signals.metadata),
        "metadata": {
            "sweep_expired_before_shift": int(signals.metadata.get("sweep_expired_before_shift", 0)),
            "armed_setup_expired_before_retest": int(signals.metadata.get("armed_setup_expired_before_retest", 0)),
            "fast_recovery_shifts": int(signals.metadata.get("fast_recovery_shifts", 0)),
            "slow_recovery_shifts": int(signals.metadata.get("slow_recovery_shifts", 0)),
            "fast_recovery_entries": int(signals.metadata.get("fast_recovery_entries", 0)),
            "slow_recovery_entries": int(signals.metadata.get("slow_recovery_entries", 0)),
            "delivery_missing_shifts": int(signals.metadata.get("delivery_missing_shifts", 0)),
        },
    }


def _rank_variants(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "label": label,
                "swing_threshold": payload["params"]["swing_threshold"],
                "liq_sweep_recovery_bars": payload["params"]["liq_sweep_recovery_bars"],
                "slow_recovery_enabled": payload["params"]["slow_recovery_enabled"],
                "slow_recovery_bars": payload["params"]["slow_recovery_bars"],
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


def _verdict(best_variant: dict[str, Any]) -> tuple[str, str]:
    trades = int(best_variant["metrics"]["total_trades"])
    total_return = float(best_variant["metrics"]["total_return_pct"])
    if trades >= 100 and total_return > 0:
        return (
            "SWING_RECOVERY_REPAIR_CLEARS_FIRST_DENSITY_GATE",
            "A swing-based recovery variant clears the first 100-trade gate while staying profitable.",
        )
    if total_return > 0:
        return (
            "SWING_RECOVERY_REPAIR_POSITIVE_BUT_BELOW_GATE",
            "A swing-based recovery variant restores positive expectancy, but it still stays below the first 100-trade density gate.",
        )
    return (
        "SWING_RECOVERY_REPAIR_STILL_NEGATIVE",
        "Longer recovery windows improve the swing branch only partially and do not restore positive expectancy.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay longer recovery windows on top of the swing-structure ICT repair branch."
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
        variants[label] = _run_variant(
            merged,
            label=label,
            overrides=spec["overrides"],
            engine=engine,
        )

    ranked = _rank_variants(variants)
    best_label = ranked[0]["label"]
    best_variant = variants[best_label]
    verdict, interpretation = _verdict(best_variant)

    output = {
        "analysis": "ict_swing_recovery_repair",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "variants": variants,
        "variant_ranking": ranked,
        "best_variant": best_label,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
