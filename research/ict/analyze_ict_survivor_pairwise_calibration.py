from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_survivor_pairwise_calibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "survivor_base": {"enable_smt": True, "overrides": {}},
        "survivor_plus_premium_discount": {
            "enable_smt": True,
            "overrides": {"use_premium_discount_filter": True},
        },
        "survivor_plus_session_array": {
            "enable_smt": True,
            "overrides": {"use_session_array_refinement": True},
        },
        "survivor_plus_daily_bias": {
            "enable_smt": True,
            "overrides": {"use_daily_bias_filter": True},
        },
        "survivor_plus_premium_discount_and_session_array": {
            "enable_smt": True,
            "overrides": {
                "use_premium_discount_filter": True,
                "use_session_array_refinement": True,
            },
        },
        "survivor_plus_daily_bias_and_premium_discount": {
            "enable_smt": True,
            "overrides": {
                "use_daily_bias_filter": True,
                "use_premium_discount_filter": True,
            },
        },
        "survivor_plus_daily_bias_and_session_array": {
            "enable_smt": True,
            "overrides": {
                "use_daily_bias_filter": True,
                "use_session_array_refinement": True,
            },
        },
        "survivor_plus_daily_bias_and_premium_discount_and_session_array": {
            "enable_smt": True,
            "overrides": {
                "use_daily_bias_filter": True,
                "use_premium_discount_filter": True,
                "use_session_array_refinement": True,
            },
        },
    }


def _rank_variants(
    results: dict[str, dict[str, Any]],
    survivor_base_trades: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        trades = int(metrics["total_trades"])
        retention = (trades / survivor_base_trades) if survivor_base_trades > 0 else 0.0
        ranked.append(
            {
                "label": label,
                "total_trades": trades,
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_retention_vs_survivor_base": retention,
                "daily_bias_filtered_setups": int(payload["metadata"].get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(
                    payload["metadata"].get("premium_discount_filtered_setups", 0)
                ),
                "session_array_filtered_shifts": int(
                    payload["metadata"].get("session_array_filtered_shifts", 0)
                ),
                "prev_session_anchor_filtered_setups": int(
                    payload["metadata"].get("prev_session_anchor_filtered_setups", 0)
                ),
                "external_liquidity_filtered_sweeps": int(
                    payload["metadata"].get("external_liquidity_filtered_sweeps", 0)
                ),
                "smt_confirmed_sweeps": int(payload["metadata"].get("smt_confirmed_sweeps", 0)),
            }
        )

    ranked.sort(
        key=lambda row: (
            row["total_trades"] > 0,
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["trade_retention_vs_survivor_base"],
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate pairwise context additions on top of the broader paired-data ICT survivor base."
    )
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
        params = build_ict_paired_survivor_profile_params(
            enable_smt=bool(spec["enable_smt"]),
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "use_smt_filter": bool(params["use_smt_filter"]),
                "use_prev_session_anchor_filter": bool(params["use_prev_session_anchor_filter"]),
                "use_external_liquidity_filter": bool(params["use_external_liquidity_filter"]),
                **spec["overrides"],
            },
            "metrics": result.metrics,
            "metadata": {
                "daily_bias_filtered_setups": int(signals.metadata.get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(
                    signals.metadata.get("premium_discount_filtered_setups", 0)
                ),
                "prev_session_anchor_filtered_setups": int(
                    signals.metadata.get("prev_session_anchor_filtered_setups", 0)
                ),
                "session_array_filtered_shifts": int(
                    signals.metadata.get("session_array_filtered_shifts", 0)
                ),
                "external_liquidity_filtered_sweeps": int(
                    signals.metadata.get("external_liquidity_filtered_sweeps", 0)
                ),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    survivor_base_trades = int(variants["survivor_base"]["metrics"]["total_trades"])
    ranked = _rank_variants(variants, survivor_base_trades=survivor_base_trades)
    best = ranked[0] if ranked else None

    robust_trade_floor = max(3, int(round(survivor_base_trades * 0.5)))
    best_non_base = next((row for row in ranked if row["label"] != "survivor_base"), None)
    best_robust_extension = next(
        (
            row
            for row in ranked
            if row["label"] != "survivor_base"
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_survivor_base"] >= 0.5
        ),
        None,
    )

    verdict = "SURVIVOR_BASE_REMAINS_FRONTIER"
    interpretation = (
        "The broader paired ICT lane still works from the survivor base, but no pairwise context addition improved on that base without thinning activity or degrading quality. The next step should revisit SMT or sweep-geometry calibration from the survivor base rather than restoring more narrative filters."
    )
    if best_robust_extension and best_robust_extension["total_return_pct"] > 0:
        verdict = "ROBUST_PAIRWISE_SURVIVOR_EXTENSION_IDENTIFIED"
        interpretation = (
            "At least one pairwise extension survives on top of the broader paired-data survivor base without collapsing activity. The next step should preserve that extension and only then revisit deeper context or SMT recalibration."
        )
    elif best_non_base and best_non_base["total_trades"] > 0 and best_non_base["total_return_pct"] > 0:
        verdict = "THIN_PAIRWISE_SURVIVOR_EXTENSION_ONLY"
        interpretation = (
            "A positive pairwise extension exists, but only as a thinner continuation of the survivor base. The next step should prefer robustness over stacking more context back in."
        )

    output = {
        "analysis": "ict_survivor_pairwise_calibration",
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
        "best_non_base_variant": best_non_base,
        "best_robust_extension": best_robust_extension,
        "survivor_base_label": "survivor_base",
        "survivor_base_trades": survivor_base_trades,
        "robust_trade_floor": robust_trade_floor,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
