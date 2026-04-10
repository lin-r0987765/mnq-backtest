from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_short_smt_premium_kill_zone_calibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "premium_base": {"overrides": {}},
        "kill_zones_default": {
            "overrides": {"use_kill_zones": True},
        },
        "kill_zones_ny_am_only": {
            "overrides": {
                "use_kill_zones": True,
                "london_kill_start": 0,
                "london_kill_end": 0,
                "ny_am_kill_start": 10,
                "ny_am_kill_end": 11,
                "ny_pm_kill_start": 0,
                "ny_pm_kill_end": 0,
            },
        },
        "kill_zones_ny_only": {
            "overrides": {
                "use_kill_zones": True,
                "london_kill_start": 0,
                "london_kill_end": 0,
                "ny_am_kill_start": 10,
                "ny_am_kill_end": 11,
                "ny_pm_kill_start": 14,
                "ny_pm_kill_end": 15,
            },
        },
        "kill_zones_broader": {
            "overrides": {
                "use_kill_zones": True,
                "london_kill_start": 2,
                "london_kill_end": 5,
                "ny_am_kill_start": 9,
                "ny_am_kill_end": 12,
                "ny_pm_kill_start": 13,
                "ny_pm_kill_end": 16,
            },
        },
        "kill_zones_shifted": {
            "overrides": {
                "use_kill_zones": True,
                "london_kill_start": 4,
                "london_kill_end": 5,
                "ny_am_kill_start": 9,
                "ny_am_kill_end": 10,
                "ny_pm_kill_start": 13,
                "ny_pm_kill_end": 14,
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
                "kill_zone_filtered_sweeps": int(payload["metadata"].get("kill_zone_filtered_sweeps", 0)),
                "daily_bias_filtered_setups": int(payload["metadata"].get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(payload["metadata"].get("premium_discount_filtered_setups", 0)),
                "smt_confirmed_sweeps": int(payload["metadata"].get("smt_confirmed_sweeps", 0)),
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
        description="Calibrate kill-zone reintroduction on top of the paired ICT short-SMT premium base."
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
        params = build_ict_paired_survivor_plus_session_array_loose_sweep_short_smt_premium_params(
            enable_smt=True,
            overrides=dict(spec["overrides"]),
        )
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        signals = strategy.generate_signals(merged)
        variants[label] = {
            "params": {
                "use_premium_discount_filter": bool(params["use_premium_discount_filter"]),
                "use_kill_zones": bool(params["use_kill_zones"]),
                "smt_lookback": int(params["smt_lookback"]),
                "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
                "london_kill_start": int(params["london_kill_start"]),
                "london_kill_end": int(params["london_kill_end"]),
                "ny_am_kill_start": int(params["ny_am_kill_start"]),
                "ny_am_kill_end": int(params["ny_am_kill_end"]),
                "ny_pm_kill_start": int(params["ny_pm_kill_start"]),
                "ny_pm_kill_end": int(params["ny_pm_kill_end"]),
            },
            "metrics": result.metrics,
            "metadata": {
                "kill_zone_filtered_sweeps": int(signals.metadata.get("kill_zone_filtered_sweeps", 0)),
                "daily_bias_filtered_setups": int(signals.metadata.get("daily_bias_filtered_setups", 0)),
                "premium_discount_filtered_setups": int(signals.metadata.get("premium_discount_filtered_setups", 0)),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    base_label = "premium_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_total_return_pct = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best = ranked[0] if ranked else None
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    robust_trade_floor = max(3, int(round(base_trades * 0.6))) if base_trades > 0 else 3
    best_robust_kill_zone = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_base"] >= 0.6
        ),
        None,
    )

    verdict = "PREMIUM_BASE_REMAINS_KILL_ZONE_FRONTIER"
    interpretation = (
        "The short-SMT premium paired ICT base remains the strongest robust profile. Kill-zone timing still acts as a thinning filter rather than a robust extension."
    )
    if best_robust_kill_zone and best_robust_kill_zone["total_return_pct"] > base_total_return_pct:
        verdict = "ROBUST_KILL_ZONE_EXTENSION_IDENTIFIED"
        interpretation = (
            "At least one kill-zone calibration improves the short-SMT premium base while preserving most of its activity. The next step should preserve that kill-zone survivor before reopening additional context."
        )
    elif best_robust_kill_zone and best_robust_kill_zone["total_return_pct"] > 0:
        verdict = "ROBUST_KILL_ZONE_SURVIVOR_BUT_NOT_FRONTIER"
        interpretation = (
            "A kill-zone calibration survives on top of the short-SMT premium base without collapsing activity, but it does not beat the base itself. The next step should keep premium as the frontier and treat kill zones as a secondary optional branch."
        )
    elif best_non_base and best_non_base["total_trades"] > 0 and best_non_base["total_return_pct"] > 0:
        verdict = "THIN_KILL_ZONE_EXTENSION_ONLY"
        interpretation = (
            "A positive kill-zone calibration exists on top of the short-SMT premium base, but only as a thinner continuation. The next step should treat kill zones as optional refinement rather than a new frontier."
        )

    output = {
        "analysis": "ict_short_smt_premium_kill_zone_calibration",
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
        "best_robust_kill_zone_variant": best_robust_kill_zone,
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
