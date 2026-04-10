from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_short_smt_premium_session_array_ny_only_slow_recovery_score_calibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "slow_recovery_base": {"overrides": {}},
        "min_score_5": {"overrides": {"min_score_to_trade": 5.0}},
        "min_score_7": {"overrides": {"min_score_to_trade": 7.0}},
        "min_score_8": {"overrides": {"min_score_to_trade": 8.0}},
        "min_score_9": {"overrides": {"min_score_to_trade": 9.0}},
        "min_score_5_lower_ote": {
            "overrides": {
                "min_score_to_trade": 5.0,
                "score_ote_zone": 1.0,
            },
        },
        "min_score_7_higher_fvg": {
            "overrides": {
                "min_score_to_trade": 7.0,
                "score_fvg": 3.0,
            },
        },
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
                "min_score_to_trade": float(payload["params"]["min_score_to_trade"]),
                "score_fvg": float(payload["params"]["score_fvg"]),
                "score_ote_zone": float(payload["params"]["score_ote_zone"]),
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
        description="Calibrate confluence score gating on top of the paired ICT NY-only slow-recovery frontier."
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
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_ny_only_slow_recovery_params(
            enable_smt=True,
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "min_score_to_trade": float(params["min_score_to_trade"]),
                "score_fvg": float(params["score_fvg"]),
                "score_ote_zone": float(params["score_ote_zone"]),
                "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
                "smt_lookback": int(params["smt_lookback"]),
            },
            "metrics": result.metrics,
            "metadata": {
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
                "bullish_shifts": int(signals.metadata.get("bullish_shifts", 0)),
                "bearish_shifts": int(signals.metadata.get("bearish_shifts", 0)),
            },
        }

    base_label = "slow_recovery_base"
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

    verdict = "SLOW_RECOVERY_BASE_REMAINS_SCORE_FRONTIER"
    interpretation = (
        "The slow-recovery NY-only paired ICT base remains the strongest robust profile. Nearby score-gating changes do not beat the current frontier."
    )
    if best_robust_variant and best_robust_variant["total_return_pct"] > base_total_return_pct:
        verdict = "ROBUST_SCORE_EXTENSION_IDENTIFIED_ON_SLOW_RECOVERY_BASE"
        interpretation = (
            "At least one score-gating calibration improves the slow-recovery NY-only paired ICT base while preserving most of its activity."
        )
    elif best_robust_variant and best_robust_variant["total_return_pct"] > 0:
        verdict = "ROBUST_SCORE_SURVIVOR_BUT_NOT_FRONTIER_ON_SLOW_RECOVERY_BASE"
        interpretation = (
            "A score-gating calibration survives on top of the slow-recovery NY-only paired ICT base without collapsing activity, but it does not beat the frontier."
        )
    elif best_non_base and best_non_base["total_trades"] > 0 and best_non_base["total_return_pct"] > 0:
        verdict = "THIN_SCORE_EXTENSION_ONLY_ON_SLOW_RECOVERY_BASE"
        interpretation = (
            "A positive score-gating calibration exists on top of the slow-recovery NY-only paired ICT base, but only as a thinner continuation."
        )

    output = {
        "analysis": "ict_short_smt_premium_session_array_ny_only_slow_recovery_score_calibration",
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
        "best_robust_score_variant": best_robust_variant,
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
