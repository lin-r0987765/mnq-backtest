from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_position_sizing_compare.json"

def _shares_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    shares_used = [int(float(trade.get("shares", 0))) for trade in trades if float(trade.get("shares", 0)) > 0]
    avg_shares = round(float(sum(shares_used) / len(shares_used)), 4) if shares_used else 0.0
    return {
        "min_shares": min(shares_used) if shares_used else 0,
        "max_shares": max(shares_used) if shares_used else 0,
        "avg_shares": avg_shares,
        "entry_count": len(shares_used),
    }


def _rank_variants(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for label, payload in variants.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": label,
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "total_trades": int(metrics["total_trades"]),
            }
        )
    ranked.sort(key=lambda row: (row["total_return_pct"], row["profit_factor"], row["total_trades"]), reverse=True)
    return ranked


def _verdict(base_return: float, best_non_base: dict[str, Any]) -> tuple[str, str]:
    if float(best_non_base["total_return_pct"]) > base_return:
        return (
            "POSITION_SIZING_IMPACT_CONFIRMED",
            "Larger sizing standards materially change the measured return on the active lite frontier, so sizing must be treated as a first-class research dimension rather than a neutral presentation layer.",
        )
    return (
        "POSITION_SIZING_PLATEAU",
        "Changing sizing modes does not improve measured return on the active lite frontier, so the current research-size convention remains acceptable for ranking.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare active lite ICT frontier performance under 10-share, 40-share, and capital-based sizing conventions."
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

    params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(enable_smt=True)
    variant_engines = {
        "research_fixed_10": BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, position_size_mode="fixed", fixed_shares=10),
        "fixed_40_shares": BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, position_size_mode="fixed", fixed_shares=40),
        "capital_50pct_min40": BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, position_size_mode="capital_pct", capital_usage_pct=0.5, min_shares=40),
        "capital_100pct_min40": BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, position_size_mode="capital_pct", capital_usage_pct=1.0, min_shares=40),
    }
    variants: dict[str, dict[str, Any]] = {}
    for label, engine in variant_engines.items():
        strategy = ICTEntryModelStrategy(params=params)
        result = engine.run(strategy, merged)
        variants[label] = {
            "metrics": result.metrics,
            "shares_summary": _shares_summary(result.trades),
            "engine": {
                "position_size_mode": engine.position_size_mode,
                "fixed_shares": engine.fixed_shares,
                "capital_usage_pct": engine.capital_usage_pct,
                "min_shares": engine.min_shares,
            },
        }

    ranked = _rank_variants(variants)
    best_non_base = next(row for row in ranked if row["label"] != "research_fixed_10")
    verdict, interpretation = _verdict(
        base_return=float(variants["research_fixed_10"]["metrics"]["total_return_pct"]),
        best_non_base=best_non_base,
    )

    output = {
        "analysis": "ict_position_sizing_compare",
        "profile": "lite_ict_reversal_relaxed_smt_looser_sweep_faster_retest_frontier",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "risk_standard": {
            "initial_cash_usd": 100000,
            "minimum_order_shares": 40,
            "reward_risk_floor": 1.5,
        },
        "variants": variants,
        "ranking": ranked,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
