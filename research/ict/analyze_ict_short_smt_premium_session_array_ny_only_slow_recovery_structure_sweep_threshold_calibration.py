from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_threshold_calibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "structure_frontier_base": {"overrides": {}},
        "sweep_threshold_0p0006": {"overrides": {"liq_sweep_threshold": 0.0006}},
        "sweep_threshold_0p0007": {"overrides": {"liq_sweep_threshold": 0.0007}},
        "sweep_threshold_0p0009": {"overrides": {"liq_sweep_threshold": 0.0009}},
        "sweep_threshold_0p0010": {"overrides": {"liq_sweep_threshold": 0.0010}},
        "sweep_threshold_0p0012": {"overrides": {"liq_sweep_threshold": 0.0012}},
    }


def _rank_variants(results: dict[str, dict[str, Any]], base_trades: int) -> list[dict[str, Any]]:
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
                "liq_sweep_threshold": float(payload["params"]["liq_sweep_threshold"]),
                "bullish_sweeps": int(payload["metadata"].get("bullish_sweeps", 0)),
                "bearish_sweeps": int(payload["metadata"].get("bearish_sweeps", 0)),
                "fvg_entries": int(payload["metadata"].get("fvg_entries", 0)),
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
        description="Calibrate liquidity-sweep threshold on top of the paired ICT slow-recovery structure frontier."
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
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_short_structure_params(
            enable_smt=True,
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
                "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
                "structure_lookback": int(params["structure_lookback"]),
                "smt_lookback": int(params["smt_lookback"]),
            },
            "metrics": result.metrics,
            "metadata": {
                "bullish_sweeps": int(signals.metadata.get("bullish_sweeps", 0)),
                "bearish_sweeps": int(signals.metadata.get("bearish_sweeps", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    base_label = "structure_frontier_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_total_return_pct = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best = ranked[0] if ranked else None
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    robust_trade_floor = max(3, int(round(base_trades * 0.6))) if base_trades > 0 else 3
    best_robust_variant = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_base"] >= 0.6
        ),
        None,
    )
    all_variants_match_base = all(
        int(payload["metrics"]["total_trades"]) == base_trades
        and float(payload["metrics"]["total_return_pct"]) == base_total_return_pct
        for payload in variants.values()
    )

    verdict = "STRUCTURE_FRONTIER_REMAINS_SWEEP_THRESHOLD_FRONTIER"
    interpretation = (
        "The structure-aware slow-recovery NY-only paired ICT base remains the strongest robust profile. Nearby liquidity-sweep threshold changes do not beat the current frontier."
    )
    if all_variants_match_base:
        verdict = "SWEEP_THRESHOLD_CALIBRATION_PLATEAU_ON_STRUCTURE_BASE"
        interpretation = (
            "All tested liquidity-sweep threshold variants collapse to the same structure-aware slow-recovery NY-only frontier result, which indicates a local plateau rather than a new edge."
        )
    elif best_robust_variant and best_robust_variant["total_return_pct"] > base_total_return_pct:
        verdict = "ROBUST_SWEEP_THRESHOLD_EXTENSION_IDENTIFIED_ON_STRUCTURE_BASE"
        interpretation = (
            "At least one liquidity-sweep threshold calibration improves the structure-aware slow-recovery NY-only paired ICT base while preserving most of its activity."
        )
    elif best_robust_variant and best_robust_variant["total_return_pct"] > 0:
        verdict = "ROBUST_SWEEP_THRESHOLD_SURVIVOR_BUT_NOT_FRONTIER_ON_STRUCTURE_BASE"
        interpretation = (
            "A liquidity-sweep threshold calibration survives on top of the structure-aware slow-recovery NY-only paired ICT base without collapsing activity, but it does not beat the frontier."
        )
    elif best_non_base and best_non_base["total_trades"] > 0 and best_non_base["total_return_pct"] > 0:
        verdict = "THIN_SWEEP_THRESHOLD_EXTENSION_ONLY_ON_STRUCTURE_BASE"
        interpretation = (
            "A positive liquidity-sweep threshold calibration exists on top of the structure-aware slow-recovery NY-only paired ICT base, but only as a thinner continuation."
        )

    output = {
        "analysis": "ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_threshold_calibration",
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
        "best_robust_sweep_threshold_variant": best_robust_variant,
        "base_label": base_label,
        "base_trades": base_trades,
        "base_total_return_pct": base_total_return_pct,
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
