from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_survivor_geometry_calibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "survivor_plus_session_array_base": {"enable_smt": True, "overrides": {}},
        "shorter_sweep_lookback_30": {
            "enable_smt": True,
            "overrides": {"liq_sweep_lookback": 30},
        },
        "longer_sweep_lookback_70": {
            "enable_smt": True,
            "overrides": {"liq_sweep_lookback": 70},
        },
        "looser_sweep_threshold_0p0008": {
            "enable_smt": True,
            "overrides": {"liq_sweep_threshold": 0.0008},
        },
        "tighter_sweep_threshold_0p0012": {
            "enable_smt": True,
            "overrides": {"liq_sweep_threshold": 0.0012},
        },
        "looser_smt_threshold_0p0015": {
            "enable_smt": True,
            "overrides": {"smt_threshold": 0.0015},
        },
        "shorter_lookback_and_looser_sweep": {
            "enable_smt": True,
            "overrides": {"liq_sweep_lookback": 30, "liq_sweep_threshold": 0.0008},
        },
        "shorter_lookback_looser_sweep_looser_smt": {
            "enable_smt": True,
            "overrides": {
                "liq_sweep_lookback": 30,
                "liq_sweep_threshold": 0.0008,
                "smt_threshold": 0.0015,
            },
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
                "bullish_sweeps": int(payload["metadata"].get("bullish_sweeps", 0)),
                "bearish_sweeps": int(payload["metadata"].get("bearish_sweeps", 0)),
                "smt_confirmed_sweeps": int(payload["metadata"].get("smt_confirmed_sweeps", 0)),
                "smt_filtered_sweeps": int(payload["metadata"].get("smt_filtered_sweeps", 0)),
                "external_liquidity_filtered_sweeps": int(
                    payload["metadata"].get("external_liquidity_filtered_sweeps", 0)
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
        description="Calibrate sweep / SMT geometry on top of the paired ICT survivor-plus-session-array profile."
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
        params = build_ict_paired_survivor_plus_session_array_params(
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

    base_trades = int(variants["survivor_plus_session_array_base"]["metrics"]["total_trades"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best = ranked[0] if ranked else None
    best_non_base = next((row for row in ranked if row["label"] != "survivor_plus_session_array_base"), None)
    robust_trade_floor = max(3, int(round(base_trades * 0.75))) if base_trades > 0 else 3
    best_robust_geometry_extension = next(
        (
            row
            for row in ranked
            if row["label"] != "survivor_plus_session_array_base"
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_base"] >= 0.75
        ),
        None,
    )

    verdict = "SURVIVOR_PLUS_SESSION_ARRAY_BASE_REMAINS_FRONTIER"
    interpretation = (
        "The survivor-plus-session-array profile remains the best robust paired ICT geometry base. No sweep / SMT geometry tweak improved it without sacrificing too much activity or quality."
    )
    if best_robust_geometry_extension and best_robust_geometry_extension["total_return_pct"] > 0:
        verdict = "ROBUST_GEOMETRY_EXTENSION_IDENTIFIED"
        interpretation = (
            "At least one sweep / SMT geometry tweak improved the survivor-plus-session-array profile while preserving most of its activity. The next step should validate that geometry extension before reopening deeper narrative filters."
        )
    elif best_non_base and best_non_base["total_trades"] > 0 and best_non_base["total_return_pct"] > 0:
        verdict = "THIN_GEOMETRY_EXTENSION_ONLY"
        interpretation = (
            "A positive geometry tweak exists, but only as a thinner extension of the survivor-plus-session-array base. The next step should prefer robustness over chasing thinner improvements."
        )

    output = {
        "analysis": "ict_survivor_geometry_calibration",
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
        "best_robust_geometry_extension": best_robust_geometry_extension,
        "base_label": "survivor_plus_session_array_base",
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
