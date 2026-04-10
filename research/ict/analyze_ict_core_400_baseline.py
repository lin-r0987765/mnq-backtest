from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ict.analyze_ict_frontier_funnel import _build_funnel_summary
from research.ict.analyze_ict_lite_reversal_baseline import RESEARCH_STANDARD, _engine_config_payload
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_core_400_baseline_profile_params,
    build_ict_core_400_short_structure_refined_recovery_candidate_profile_params,
    build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params,
    build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params,
    build_ict_core_400_short_stat_bias_candidate_profile_params,
    build_ict_core_400_short_structure_bias_candidate_profile_params,
    build_ict_core_400_short_structure_bias_lb6_candidate_profile_params,
    build_ict_core_400_short_structure_refined_candidate_profile_params,
    build_ict_core_400_short_structure_refined_density_candidate_profile_params,
    build_ict_core_400_short_only_profile_params,
    build_ict_lite_reversal_qualified_continuation_density_profile_params,
    build_ict_lite_reversal_qualified_reversal_balance_profile_params,
)


_REGULARIZED_ROOT = PROJECT_ROOT / "regularized"
DEFAULT_INTRADAY = (
    _REGULARIZED_ROOT / "QQQ" / "qqq_5m_regular_et.csv"
    if (_REGULARIZED_ROOT / "QQQ" / "qqq_5m_regular_et.csv").exists()
    else PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
)
DEFAULT_PEER_CSV = (
    _REGULARIZED_ROOT / "SPY" / "spy_5m_regular_et.csv"
    if (_REGULARIZED_ROOT / "SPY" / "spy_5m_regular_et.csv").exists()
    else PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_core_400_baseline.json"


def _data_window(df) -> dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "trading_days": 0, "calendar_days": 0}
    start = pd.Timestamp(df.index.min())
    end = pd.Timestamp(df.index.max())
    return {
        "start": str(start),
        "end": str(end),
        "trading_days": int(len(set(df.index.date))),
        "calendar_days": int((end - start).days),
    }


def _period_label(df) -> str:
    window = _data_window(df)
    if window["start"] is None or window["end"] is None:
        return "empty"
    return f'{window["start"]} -> {window["end"]}'


def _sample_years(df) -> float:
    if df.empty:
        return 0.0
    start = pd.Timestamp(df.index.min())
    end = pd.Timestamp(df.index.max())
    return max((end - start).days / 365.25, 0.0)


def _side_breakdown(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "wins": 0.0,
            "engine_pnl": 0.0,
            "net_after_all_fees": 0.0,
            "gross_win": 0.0,
            "gross_loss": 0.0,
        }
    )
    for trade in trades:
        side = str(trade.get("side", "unknown"))
        pnl = float(trade.get("pnl", 0.0))
        entry_fee = float(trade.get("entry_fee", 0.0))
        net_after_all_fees = pnl - entry_fee
        grouped[side]["count"] += 1.0
        grouped[side]["engine_pnl"] += pnl
        grouped[side]["net_after_all_fees"] += net_after_all_fees
        if pnl > 0:
            grouped[side]["wins"] += 1.0
            grouped[side]["gross_win"] += pnl
        else:
            grouped[side]["gross_loss"] += abs(pnl)

    payload: dict[str, dict[str, Any]] = {}
    for side, stats in grouped.items():
        count = int(stats["count"])
        gross_loss = float(stats["gross_loss"])
        profit_factor = math.inf if gross_loss == 0.0 and count > 0 else (
            float(stats["gross_win"]) / gross_loss if gross_loss > 0 else 0.0
        )
        payload[side] = {
            "count": count,
            "win_rate_pct": round(float(stats["wins"]) / count * 100.0, 4) if count > 0 else 0.0,
            "engine_pnl_ex_entry_fees_usd": round(float(stats["engine_pnl"]), 4),
            "net_after_all_fees_usd": round(float(stats["net_after_all_fees"]), 4),
            "profit_factor": profit_factor if math.isinf(profit_factor) else round(profit_factor, 4),
        }
    return payload


def _run_profile(
    merged,
    *,
    label: str,
    params: dict[str, Any],
    engine: BacktestEngine,
    sample_years: float,
) -> dict[str, Any]:
    strategy = ICTEntryModelStrategy(params=params)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    metrics = dict(result.metrics)
    total_trades = int(metrics["total_trades"])
    trades_per_6y = total_trades / sample_years * 6.0 if sample_years > 0 else 0.0
    metadata = signals.metadata
    return {
        "label": label,
        "params": {
            "use_smt_filter": bool(params["use_smt_filter"]),
            "trade_sessions": bool(params["trade_sessions"]),
            "allow_long_entries": bool(params.get("allow_long_entries", True)),
            "allow_short_entries": bool(params.get("allow_short_entries", True)),
            "enable_continuation_entry": bool(params["enable_continuation_entry"]),
            "structure_reference_mode": str(params["structure_reference_mode"]),
            "structure_lookback": int(params["structure_lookback"]),
            "liq_sweep_threshold": float(params["liq_sweep_threshold"]),
            "liq_sweep_recovery_bars": int(params["liq_sweep_recovery_bars"]),
            "slow_recovery_enabled": bool(params["slow_recovery_enabled"]),
            "slow_recovery_bars": int(params["slow_recovery_bars"]),
            "fvg_min_gap_pct": float(params["fvg_min_gap_pct"]),
            "fvg_revisit_depth_ratio": float(params["fvg_revisit_depth_ratio"]),
            "fvg_revisit_min_delay_bars": int(params["fvg_revisit_min_delay_bars"]),
            "take_profit_rr": float(params["take_profit_rr"]),
            "stop_loss_atr_mult": float(params["stop_loss_atr_mult"]),
            "max_pending_setups_per_direction": int(params["max_pending_setups_per_direction"]),
            "max_reentries_per_setup": int(params["max_reentries_per_setup"]),
        },
        "metrics": metrics,
        "trades_per_6y_equivalent": round(trades_per_6y, 2),
        "side_breakdown": _side_breakdown(result.trades),
        "metadata": {
            key: int(metadata.get(key, 0))
            for key in (
                "bullish_sweeps",
                "bearish_sweeps",
                "bullish_shift_candidates",
                "bearish_shift_candidates",
                "long_entries",
                "short_entries",
                "continuation_entries",
                "fast_recovery_entries",
                "slow_recovery_entries",
                "reentry_entries",
                "reentry_stop_rearms",
                "sweep_expired_before_shift",
                "armed_setup_expired_before_retest",
                "delivery_missing_shifts",
                "fvg_depth_filtered_retests",
                "fvg_delay_filtered_retests",
            )
        },
        "funnel": _build_funnel_summary(metadata),
    }


def _rank_variants(results: list[dict[str, Any]], density_gate: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for payload in results:
        metrics = payload["metrics"]
        ranked.append(
            {
                "label": payload["label"],
                "total_trades": int(metrics["total_trades"]),
                "trades_per_6y_equivalent": float(payload["trades_per_6y_equivalent"]),
                "clears_density_gate": int(metrics["total_trades"]) >= density_gate,
                "total_return_pct": float(metrics["total_return_pct"]),
                "profit_factor": float(metrics["profit_factor"]),
                "win_rate_pct": float(metrics["win_rate_pct"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["clears_density_gate"],
            row["total_return_pct"] > 0,
            row["profit_factor"] > 1.0,
            row["total_trades"],
            row["total_return_pct"],
            row["profit_factor"],
        ),
        reverse=True,
    )
    return ranked


def _verdict(
    *,
    density_gate: int,
    core_baseline: dict[str, Any],
    core_short_only: dict[str, Any],
    best_positive_variant: dict[str, Any] | None,
) -> tuple[str, str]:
    baseline_trades = int(core_baseline["metrics"]["total_trades"])
    baseline_return = float(core_baseline["metrics"]["total_return_pct"])
    short_only_return = float(core_short_only["metrics"]["total_return_pct"])

    if baseline_trades >= density_gate and baseline_return > 0:
        return (
            "ICT_CORE_400_DENSITY_AND_EDGE_CONFIRMED",
            "The new ICT-core branch clears the full-sample density gate and stays profitable on the complete regularized history.",
        )
    if baseline_trades >= density_gate and short_only_return > 0:
        return (
            "ICT_CORE_400_DENSITY_CONFIRMED_LONG_SIDE_DRAG",
            "The ICT-core branch proves the 400-trades target is structurally reachable, but the long side currently drags the high-density lane negative.",
        )
    if baseline_trades >= density_gate:
        return (
            "ICT_CORE_400_DENSITY_CONFIRMED_BUT_EDGE_NOT_YET",
            "The ICT-core branch clears the density gate, but no full two-sided variant is profitable yet on the complete history.",
        )
    if best_positive_variant is not None:
        return (
            "ICT_CORE_400_EDGE_SURVIVES_ONLY_IN_LOWER_DENSITY_VARIANTS",
            "Some ICT-core variants remain profitable, but the full-sample density gate is still not cleared on the complete history.",
        )
    return (
        "ICT_CORE_400_REQUIRES_ARCHITECTURE_REPAIR",
        "Neither the complete ICT stack nor the current high-density ICT-core variants can yet satisfy the density-plus-edge target on the complete regularized history.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formalize and replay the high-density ICT-core baseline on the complete regularized history."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="full_local")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-trades-per-6y", type=int, default=400)
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    sample_years = _sample_years(intraday_df)
    density_gate = math.ceil(args.target_trades_per_6y * sample_years / 6.0) if sample_years > 0 else 0

    variant_specs: list[tuple[str, Callable[..., dict[str, Any]], bool, dict[str, Any]]] = [
        (
            "qualified_reversal_balance_no_smt",
            build_ict_lite_reversal_qualified_reversal_balance_profile_params,
            False,
            {},
        ),
        (
            "qualified_continuation_density_no_smt",
            build_ict_lite_reversal_qualified_continuation_density_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_baseline",
            build_ict_core_400_baseline_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_daily_bias",
            build_ict_core_400_baseline_profile_params,
            False,
            {"use_daily_bias_filter": True},
        ),
        (
            "ict_core_400_short_only",
            build_ict_core_400_short_only_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_bias_candidate",
            build_ict_core_400_short_structure_bias_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_bias_lb6_candidate",
            build_ict_core_400_short_structure_bias_lb6_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_candidate",
            build_ict_core_400_short_structure_refined_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_recovery_candidate",
            build_ict_core_400_short_structure_refined_recovery_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate",
            build_ict_core_400_short_structure_refined_recovery_sl135_daily_bias_lb8_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate",
            build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate",
            build_ict_core_400_short_structure_refined_capacity_threshold0265_pending4_sl135_daily_bias_lb8_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_capacity_pending4_candidate",
            build_ict_core_400_short_structure_refined_capacity_pending4_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_capacity_candidate",
            build_ict_core_400_short_structure_refined_capacity_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_structure_refined_density_candidate",
            build_ict_core_400_short_structure_refined_density_candidate_profile_params,
            False,
            {},
        ),
        (
            "ict_core_400_short_stat_bias_candidate",
            build_ict_core_400_short_stat_bias_candidate_profile_params,
            True,
            {},
        ),
        (
            "ict_core_400_reentry_1",
            build_ict_core_400_baseline_profile_params,
            False,
            {"max_reentries_per_setup": 1},
        ),
    ]

    payloads: list[dict[str, Any]] = []
    for label, builder, enable_smt, overrides in variant_specs:
        params = builder(enable_smt=enable_smt, overrides=dict(overrides))
        payloads.append(
            _run_profile(
                merged,
                label=label,
                params=params,
                engine=engine,
                sample_years=sample_years,
            )
        )

    ranking = _rank_variants(payloads, density_gate=density_gate)
    by_label = {payload["label"]: payload for payload in payloads}
    best_positive_variant = next(
        (
            row
            for row in ranking
            if row["total_return_pct"] > 0 and row["profit_factor"] > 1.0
        ),
        None,
    )
    verdict, interpretation = _verdict(
        density_gate=density_gate,
        core_baseline=by_label["ict_core_400_baseline"],
        core_short_only=by_label["ict_core_400_short_only"],
        best_positive_variant=best_positive_variant,
    )

    output = {
        "analysis": "ict_core_400_baseline",
        "target": {
            "trades_per_6y": int(args.target_trades_per_6y),
            "sample_years": round(sample_years, 4),
            "full_sample_density_gate": int(density_gate),
        },
        "risk_standard": {
            "reward_risk_gate": ">= 1.5:1",
            "engine": _engine_config_payload(engine),
        },
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": _period_label(intraday_df),
        "peer_fetch_period": args.period if not args.peer_csv else None,
        "data_window": _data_window(intraday_df),
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "profiles": payloads,
        "ranking": ranking,
        "best_positive_variant": best_positive_variant,
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
