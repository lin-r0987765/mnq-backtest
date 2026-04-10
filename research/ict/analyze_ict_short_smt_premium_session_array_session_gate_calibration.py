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
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_short_smt_premium_session_array_session_gate_calibration.json"
)


def _variant_specs() -> dict[str, dict[str, Any]]:
    return {
        "session_array_premium_base": {"overrides": {}},
        "trade_sessions_off_control": {
            "overrides": {"trade_sessions": False},
        },
        "broader_sessions": {
            "overrides": {
                "trade_sessions": True,
                "london_open": 7,
                "london_close": 17,
                "ny_open": 12,
                "ny_close": 22,
            },
        },
        "narrower_london_ny_overlap": {
            "overrides": {
                "trade_sessions": True,
                "london_open": 9,
                "london_close": 15,
                "ny_open": 14,
                "ny_close": 20,
            },
        },
        "ny_only_core": {
            "overrides": {
                "trade_sessions": True,
                "london_open": 0,
                "london_close": 0,
                "ny_open": 14,
                "ny_close": 20,
            },
        },
        "london_only_core": {
            "overrides": {
                "trade_sessions": True,
                "london_open": 8,
                "london_close": 14,
                "ny_open": 0,
                "ny_close": 0,
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
                "session_array_filtered_shifts": int(payload["metadata"].get("session_array_filtered_shifts", 0)),
                "premium_discount_filtered_setups": int(
                    payload["metadata"].get("premium_discount_filtered_setups", 0)
                ),
                "smt_confirmed_sweeps": int(payload["metadata"].get("smt_confirmed_sweeps", 0)),
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
        description="Calibrate broad session gating on top of the paired ICT short-SMT premium-plus-session-array frontier."
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
                "trade_sessions": bool(params["trade_sessions"]),
                "london_open": int(params["london_open"]),
                "london_close": int(params["london_close"]),
                "ny_open": int(params["ny_open"]),
                "ny_close": int(params["ny_close"]),
                "use_session_array_refinement": bool(params["use_session_array_refinement"]),
                "use_premium_discount_filter": bool(params["use_premium_discount_filter"]),
                "smt_lookback": int(params["smt_lookback"]),
                "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            },
            "metrics": result.metrics,
            "metadata": {
                "session_array_filtered_shifts": int(signals.metadata.get("session_array_filtered_shifts", 0)),
                "premium_discount_filtered_setups": int(
                    signals.metadata.get("premium_discount_filtered_setups", 0)
                ),
                "smt_confirmed_sweeps": int(signals.metadata.get("smt_confirmed_sweeps", 0)),
                "fvg_entries": int(signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(signals.metadata.get("breaker_entries", 0)),
                "ifvg_entries": int(signals.metadata.get("ifvg_entries", 0)),
            },
        }

    base_label = "session_array_premium_base"
    base_trades = int(variants[base_label]["metrics"]["total_trades"])
    base_total_return_pct = float(variants[base_label]["metrics"]["total_return_pct"])
    ranked = _rank_variants(variants, base_trades=base_trades)
    best = ranked[0] if ranked else None
    best_non_base = next((row for row in ranked if row["label"] != base_label), None)
    robust_trade_floor = max(3, int(round(base_trades * 0.6))) if base_trades > 0 else 3
    best_robust_session_gate = next(
        (
            row
            for row in ranked
            if row["label"] != base_label
            and row["total_trades"] >= robust_trade_floor
            and row["trade_retention_vs_base"] >= 0.6
        ),
        None,
    )

    verdict = "SESSION_ARRAY_PREMIUM_BASE_REMAINS_SESSION_GATE_FRONTIER"
    interpretation = (
        "The short-SMT premium-plus-session-array paired ICT base remains the strongest robust session-gating profile. Nearby broad-session changes do not beat the frontier."
    )
    if best_robust_session_gate and best_robust_session_gate["total_return_pct"] > base_total_return_pct:
        verdict = "ROBUST_SESSION_GATE_EXTENSION_IDENTIFIED_ON_SESSION_ARRAY_BASE"
        interpretation = (
            "At least one broad session-gating calibration improves the short-SMT premium-plus-session-array base while preserving most of its activity."
        )
    elif best_robust_session_gate and best_robust_session_gate["total_return_pct"] > 0:
        verdict = "ROBUST_SESSION_GATE_SURVIVOR_BUT_NOT_FRONTIER_ON_SESSION_ARRAY_BASE"
        interpretation = (
            "A broad session-gating calibration survives on top of the short-SMT premium-plus-session-array base without collapsing activity, but it does not beat the frontier."
        )
    elif best_non_base and best_non_base["total_trades"] > 0 and best_non_base["total_return_pct"] > 0:
        verdict = "THIN_SESSION_GATE_EXTENSION_ONLY_ON_SESSION_ARRAY_BASE"
        interpretation = (
            "A positive broad session-gating calibration exists on top of the short-SMT premium-plus-session-array base, but only as a thinner continuation."
        )

    output = {
        "analysis": "ict_short_smt_premium_session_array_session_gate_calibration",
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
        "best_robust_session_gate_variant": best_robust_session_gate,
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
