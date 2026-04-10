from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns, resample_ohlcv
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_mtf_topdown_continuation_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_profile_params,
    build_ict_mtf_topdown_continuation_quality_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params,
    build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params,
    build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params,
    build_ict_mtf_topdown_continuation_timing_candidate_profile_params,
)
DEFAULT_INTRADAY = PROJECT_ROOT / "regularized" / "QQQ" / "qqq_1m_regular_et.csv"
if not DEFAULT_INTRADAY.exists():
    DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_1m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "regularized" / "SPY" / "spy_1m_regular_et.csv"
if not DEFAULT_PEER_CSV.exists():
    DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_1m_alpaca.csv"
DEFAULT_FIVE_MIN_REFERENCE = PROJECT_ROOT / "regularized" / "QQQ" / "qqq_5m_regular_et.csv"
if not DEFAULT_FIVE_MIN_REFERENCE.exists():
    DEFAULT_FIVE_MIN_REFERENCE = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qc_regime_prototypes"
    / "ict_mtf_topdown_continuation_baseline.json"
)
RESEARCH_STANDARD = {
    "initial_cash": 10_000.0,
    "fees_pct": 0.0005,
    "position_size_mode": "capital_pct",
    "capital_usage_pct": 1.0,
    "min_shares": 0,
}
PROMOTION_BASELINES = {
    "mtf_prototype": {
        "total_trades": 40,
        "total_return_pct": 13.4177,
        "profit_factor": 3.1879,
    },
    "active_lite_frontier_standardized": {
        "total_trades": 18,
        "total_return_pct": 12.6548,
    },
}


def _build_profile(profile: str) -> tuple[str, dict[str, Any]]:
    key = str(profile).strip().lower()
    if key == "execution_candidate":
        return (
            "mtf_topdown_continuation_execution_candidate",
            build_ict_mtf_topdown_continuation_execution_candidate_profile_params(enable_smt=False),
        )
    if key == "setup_execution_candidate":
        return (
            "mtf_topdown_continuation_setup_execution_candidate",
            build_ict_mtf_topdown_continuation_setup_execution_candidate_profile_params(enable_smt=False),
        )
    if key == "timing_candidate":
        return (
            "mtf_topdown_continuation_timing_candidate",
            build_ict_mtf_topdown_continuation_timing_candidate_profile_params(enable_smt=False),
        )
    if key == "regularized_long_only_candidate":
        return (
            "mtf_topdown_continuation_regularized_long_only_candidate",
            build_ict_mtf_topdown_continuation_regularized_long_only_candidate_profile_params(
                enable_smt=False
            ),
        )
    if key == "regularized_long_only_am_candidate":
        return (
            "mtf_topdown_continuation_regularized_long_only_am_candidate",
            build_ict_mtf_topdown_continuation_regularized_long_only_am_candidate_profile_params(
                enable_smt=False
            ),
        )
    if key == "quality_candidate":
        return (
            "mtf_topdown_continuation_quality_candidate",
            build_ict_mtf_topdown_continuation_quality_candidate_profile_params(enable_smt=False),
        )
    return (
        "mtf_topdown_continuation_baseline",
        build_ict_mtf_topdown_continuation_profile_params(enable_smt=False),
    )


def _engine_payload(engine: BacktestEngine) -> dict[str, Any]:
    return {
        "initial_cash": float(engine.initial_cash),
        "fees_pct": float(engine.fees_pct),
        "position_size_mode": str(engine.position_size_mode),
        "capital_usage_pct": float(engine.capital_usage_pct),
        "min_shares": int(engine.min_shares),
    }


def _data_window(df) -> dict[str, Any]:
    if df.empty:
        return {
            "start": None,
            "end": None,
            "trading_days": 0,
        }
    return {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "trading_days": int(len(set(df.index.date))),
    }


def _period_label(df) -> str:
    window = _data_window(df)
    if window["start"] is None or window["end"] is None:
        return "empty"
    return f'{window["start"]} -> {window["end"]}'


def _resolve_output_path(requested_output: str, *, profile_name: str) -> Path:
    requested_path = Path(requested_output)
    if requested_path != DEFAULT_OUTPUT:
        return requested_path
    if profile_name == "mtf_topdown_continuation_baseline":
        return requested_path
    if profile_name == "mtf_topdown_continuation_execution_candidate":
        return requested_path.with_name("ict_mtf_topdown_continuation_execution_candidate.json")
    if profile_name == "mtf_topdown_continuation_setup_execution_candidate":
        return requested_path.with_name("ict_mtf_topdown_continuation_setup_execution_candidate.json")
    if profile_name == "mtf_topdown_continuation_timing_candidate":
        return requested_path.with_name("ict_mtf_topdown_continuation_timing_candidate.json")
    if profile_name == "mtf_topdown_continuation_regularized_long_only_candidate":
        return requested_path.with_name("ict_mtf_topdown_continuation_regularized_long_only_candidate.json")
    if profile_name == "mtf_topdown_continuation_regularized_long_only_am_candidate":
        return requested_path.with_name("ict_mtf_topdown_continuation_regularized_long_only_am_candidate.json")
    return requested_path.with_name(f"{profile_name}.json")


def _resample_parity_summary(
    canonical_df,
    reference_path: Path,
) -> dict[str, Any]:
    if not reference_path.exists():
        return {
            "reference_available": False,
        }

    resampled = resample_ohlcv(canonical_df, "5m")
    reference = load_ohlcv_csv(reference_path)
    joined = resampled[["Open", "High", "Low", "Close"]].join(
        reference[["Open", "High", "Low", "Close"]],
        how="inner",
        lsuffix="_resampled",
        rsuffix="_reference",
    )
    if joined.empty:
        return {
            "reference_available": True,
            "matched_rows": 0,
        }

    parity: dict[str, float] = {}
    for column in ["Open", "High", "Low", "Close"]:
        diff = (
            joined[f"{column}_resampled"] - joined[f"{column}_reference"]
        ).abs()
        parity[f"{column.lower()}_max_abs_diff"] = round(float(diff.max()), 8)
        parity[f"{column.lower()}_mean_abs_diff"] = round(float(diff.mean()), 8)
    return {
        "reference_available": True,
        "matched_rows": int(len(joined)),
        **parity,
    }


def _promotion_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    trades = int(metrics["total_trades"])
    total_return = float(metrics["total_return_pct"])
    profit_factor = float(metrics["profit_factor"])
    return {
        "meets_trade_gate": trades >= PROMOTION_BASELINES["mtf_prototype"]["total_trades"],
        "meets_return_gate": total_return > PROMOTION_BASELINES["mtf_prototype"]["total_return_pct"],
        "meets_profit_factor_gate": profit_factor >= PROMOTION_BASELINES["mtf_prototype"]["profit_factor"],
        "beats_active_lite_return": total_return >= PROMOTION_BASELINES["active_lite_frontier_standardized"]["total_return_pct"],
        "beats_active_lite_trade_count": trades >= PROMOTION_BASELINES["active_lite_frontier_standardized"]["total_trades"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 1m-canonical ICT top-down continuation baseline."
    )
    parser.add_argument("--intraday-csv", default=str(DEFAULT_INTRADAY))
    parser.add_argument("--peer-symbol", default="SPY")
    parser.add_argument("--peer-csv", default=str(DEFAULT_PEER_CSV) if DEFAULT_PEER_CSV.exists() else None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--five-min-reference", default=str(DEFAULT_FIVE_MIN_REFERENCE))
    parser.add_argument(
        "--profile",
        choices=[
            "baseline",
            "quality_candidate",
            "execution_candidate",
            "setup_execution_candidate",
            "timing_candidate",
            "regularized_long_only_candidate",
            "regularized_long_only_am_candidate",
        ],
        default="baseline",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    intraday_df = load_ohlcv_csv(args.intraday_csv)
    peer_df = fetch_peer_data(peer_symbol=args.peer_symbol, peer_csv=args.peer_csv, period=args.period)
    merged = merge_peer_columns(intraday_df, peer_df)
    profile_name, params = _build_profile(args.profile)
    strategy = ICTEntryModelStrategy(params=params)
    engine = BacktestEngine(**RESEARCH_STANDARD)
    result = engine.run(strategy, merged)
    signals = strategy.generate_signals(merged)
    parity_summary = _resample_parity_summary(intraday_df, Path(args.five_min_reference))
    output_path = _resolve_output_path(args.output, profile_name=profile_name)

    output = {
        "analysis": profile_name,
        "profile": profile_name,
        "risk_standard": {
            "reward_risk_gate": ">= 1.5:1",
            "engine": _engine_payload(engine),
        },
        "promotion_baselines": PROMOTION_BASELINES,
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": _period_label(intraday_df),
        "peer_fetch_period": args.period if not args.peer_csv else None,
        "data_window": _data_window(intraday_df),
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "mtf_params": {
            "mtf_bias_daily_timeframe": str(params["mtf_bias_daily_timeframe"]),
            "mtf_bias_4h_timeframe": str(params["mtf_bias_4h_timeframe"]),
            "mtf_bias_1h_timeframe": str(params["mtf_bias_1h_timeframe"]),
            "mtf_setup_timeframe": str(params["mtf_setup_timeframe"]),
            "mtf_confirmation_timeframe": str(params["mtf_confirmation_timeframe"]),
            "mtf_trigger_timeframe": str(params["mtf_trigger_timeframe"]),
            "mtf_bias_lookback": int(params["mtf_bias_lookback"]),
            "mtf_setup_structure_lookback": int(params["mtf_setup_structure_lookback"]),
            "mtf_setup_fvg_min_gap_pct": float(params["mtf_setup_fvg_min_gap_pct"]),
            "mtf_setup_displacement_body_min_pct": float(params["mtf_setup_displacement_body_min_pct"]),
            "mtf_confirmation_close_ratio": float(params["mtf_confirmation_close_ratio"]),
            "mtf_confirmation_body_min_pct": float(params["mtf_confirmation_body_min_pct"]),
            "mtf_trigger_close_ratio": float(params["mtf_trigger_close_ratio"]),
            "mtf_trigger_body_min_pct": float(params["mtf_trigger_body_min_pct"]),
            "mtf_trigger_expiry_bars": int(params["mtf_trigger_expiry_bars"]),
            "mtf_timing_timezone": str(params["mtf_timing_timezone"]),
            "mtf_allowed_entry_weekdays": params["mtf_allowed_entry_weekdays"],
            "mtf_allowed_entry_hours": params["mtf_allowed_entry_hours"],
        },
        "metrics": result.metrics,
        "metadata": {
            key: int(signals.metadata.get(key, 0))
            for key in (
                "mtf_daily_blocked",
                "mtf_4h_blocked",
                "mtf_1h_blocked",
                "mtf_direction_long_allowed",
                "mtf_direction_short_allowed",
                "mtf_15m_setups",
                "mtf_5m_confirms",
                "mtf_1m_triggers",
                "mtf_setup_fvg_zones",
                "mtf_setup_ob_zones",
                "mtf_setup_missing_zone",
                "mtf_setup_missing_context",
                "mtf_setup_score_filtered",
                "mtf_setup_expired",
                "mtf_confirm_expired",
                "mtf_neutral_1h_high_quality_setups",
                "mtf_weekday_blocked",
                "mtf_hour_blocked",
                "rr_filtered_entries",
                "long_entries",
                "short_entries",
            )
        },
        "canonical_resample_parity": parity_summary,
        "promotion_summary": _promotion_summary(result.metrics),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
