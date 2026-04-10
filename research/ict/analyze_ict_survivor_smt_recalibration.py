from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_loose_sweep_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_survivor_smt_recalibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "loose_sweep_base": {"enable_smt": True, "overrides": {}},
        "no_smt_control": {"enable_smt": False, "overrides": {}},
        "shorter_smt_lookback_10": {
            "enable_smt": True,
            "overrides": {"smt_lookback": 10},
        },
        "longer_smt_lookback_30": {
            "enable_smt": True,
            "overrides": {"smt_lookback": 30},
        },
        "longer_smt_lookback_40": {
            "enable_smt": True,
            "overrides": {"smt_lookback": 40},
        },
        "tighter_smt_threshold_0p0008": {
            "enable_smt": True,
            "overrides": {"smt_threshold": 0.0008},
        },
        "looser_smt_threshold_0p0015": {
            "enable_smt": True,
            "overrides": {"smt_threshold": 0.0015},
        },
        "shorter_lookback_looser_threshold": {
            "enable_smt": True,
            "overrides": {"smt_lookback": 10, "smt_threshold": 0.0015},
        },
        "longer_lookback_looser_threshold": {
            "enable_smt": True,
            "overrides": {"smt_lookback": 30, "smt_threshold": 0.0015},
        },
    }


def _rank_variants(
    results: dict[str, dict[str, Any]],
    base_trades: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in results.items():
        metrics = payload["metrics"]
        trades = int(metrics["total_trades"])
        retention = (trades / base_trades) if base_trades > 0 else 0.0
        ranked.append(
            {
                "label": label,
                "total_trades": trades,
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "trade_retention_vs_base": retention,
                "smt_confirmed_sweeps": int(payload["metadata"].get("smt_confirmed_sweeps", 0)),
                "smt_filtered_sweeps": int(payload["metadata"].get("smt_filtered_sweeps", 0)),
                "smt_missing_peer_bars": int(payload["metadata"].get("smt_missing_peer_bars", 0)),
                "external_liquidity_filtered_sweeps": int(
                    payload["metadata"].get("external_liquidity_filtered_sweeps", 0)
                ),
                "prev_session_anchor_filtered_setups": int(
                    payload["metadata"].get("prev_session_anchor_filtered_setups", 0)
                ),
                "session_array_filtered_shifts": int(
                    payload["metadata"].get("session_array_filtered_shifts", 0)
                ),
            }
        )

    ranked.sort(
        key=lambda row: (
            row["total_trades"] > 0,
            row["total_return_pct"] > 0,
            row["total_return_pct"],
            row["profit_factor"],
            row["trade_retention_vs_base"],
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalibrate SMT geometry on top of the paired ICT survivor-plus-session-array-loose-sweep profile."
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
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_params(
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
                "use_session_array_refinement": bool(params["use_session_array_refinement"]),
                "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
                **spec["overrides"],
            },
            "metrics": result.metrics,
            "metadata": {
                "bullish_sweeps": int(signals.metadata.get("bullish_sweeps", 0)),
                "bearish_sweeps": int(signals.metadata.get("bearish_sweeps", 0)),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
                "smt_filtered_sweeps": int(signals.metadata.get("smt_filtered_sweeps", 0)),
                "smt_missing_peer_bars": int(signals.metadata.get("smt_missing_peer_bars", 0)),
                "external_liquidity_filtered_sweeps": int(
                    signals.metadata.get("external_liquidity_filtered_sweeps", 0)
                ),
                "prev_session_anchor_filtered_setups": int(
                    signals.metadata.get("prev_session_anchor_filtered_setups", 0)
                ),
                "session_array_filtered_shifts": int(
                    signals.metadata.get("session_array_filtered_shifts", 0)
                ),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    base_label = "loose_sweep_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best = ranked[0] if ranked else None
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    robust_trade_floor = max(4, int(round(base_trades * 0.67))) if base_trades > 0 else 4
    best_robust_smt_extension = next(
        (
            row
            for row in ranked
            if row["label"] not in {base_label, "no_smt_control"}
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_base"] >= (2.0 / 3.0)
        ),
        None,
    )
    no_smt_control = next((row for row in ranked if row["label"] == "no_smt_control"), None)

    verdict = "LOOSE_SWEEP_BASE_REMAINS_SMT_FRONTIER"
    interpretation = (
        "The paired ICT lane now has a stronger loose-sweep geometry base, but no SMT recalibration improved that base without sacrificing too much activity or quality. The next step should avoid reopening SMT thresholds blindly."
    )
    if best_robust_smt_extension and best_robust_smt_extension["total_return_pct"] > 0:
        verdict = "ROBUST_SMT_EXTENSION_IDENTIFIED"
        interpretation = (
            "At least one SMT recalibration improves the survivor-plus-session-array-loose-sweep profile while preserving most of its activity. The next step should lock that SMT extension as the new paired ICT base before revisiting heavier narrative filters."
        )
    elif (
        no_smt_control
        and no_smt_control["total_trades"] >= robust_trade_floor
        and no_smt_control["total_return_pct"] > variants[base_label]["metrics"]["total_return_pct"]
    ):
        verdict = "SMT_GATE_STILL_TOO_RESTRICTIVE"
        interpretation = (
            "The loose-sweep paired ICT base performs better when SMT is removed entirely, which means the current SMT gate remains the bottleneck rather than the solution. The next step should not stack more SMT logic until the peer-divergence geometry itself is reconsidered."
        )

    output = {
        "analysis": "ict_survivor_smt_recalibration",
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
        "best_robust_smt_extension": best_robust_smt_extension,
        "no_smt_control": no_smt_control,
        "base_label": base_label,
        "base_trades": base_trades,
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
