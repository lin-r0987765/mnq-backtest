from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.base import BaseStrategy, StrategyResult
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    _compute_atr,
    _detect_fvg_zone,
    _in_kill_zone,
    _in_trade_session,
    _same_trading_day,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRADAY = PROJECT_ROOT / "alpaca" / "normalized" / "qqq_5m_alpaca.csv"
DEFAULT_PEER_CSV = PROJECT_ROOT / "alpaca" / "normalized" / "spy_5m_alpaca.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qc_regime_prototypes" / "ict_continuation_lane_compare.json"


@dataclass
class _ContinuationPending:
    direction: int
    breakout_index: int
    structure_level: float
    breakout_extreme: float
    expiry_index: int
    armed_index: int
    zone_lower: float
    zone_upper: float
    retest_seen: bool = False
    retest_touches: int = 0


class ICTContinuationProxyStrategy(BaseStrategy):
    name = "ICTContinuationProxyStrategy"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        if len(df) < 10:
            empty = pd.Series(False, index=df.index)
            return StrategyResult(empty.copy(), empty.copy(), empty.copy(), empty.copy(), metadata={})

        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        entries_long = pd.Series(False, index=df.index)
        exits_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits_short = pd.Series(False, index=df.index)

        structure_lookback = int(self.params["structure_lookback"])
        fvg_min_gap_pct = float(self.params["fvg_min_gap_pct"])
        fvg_max_age = int(self.params["fvg_max_age"])
        fvg_revisit_min_delay_bars = int(self.params["fvg_revisit_min_delay_bars"])
        fvg_revisit_depth_ratio = float(self.params["fvg_revisit_depth_ratio"])
        stop_loss_atr_mult = float(self.params["stop_loss_atr_mult"])
        take_profit_rr = float(self.params["take_profit_rr"])
        displacement_body_min_pct = float(self.params["displacement_body_min_pct"])
        trade_sessions = bool(self.params["trade_sessions"])
        london_open = int(self.params["london_open"])
        london_close = int(self.params["london_close"])
        ny_open = int(self.params["ny_open"])
        ny_close = int(self.params["ny_close"])
        use_kill_zones = bool(self.params["use_kill_zones"])
        kill_zone_timezone = str(self.params["kill_zone_timezone"])
        kill_windows = [
            (int(self.params["london_kill_start"]), int(self.params["london_kill_end"])),
            (int(self.params["ny_am_kill_start"]), int(self.params["ny_am_kill_end"])),
            (int(self.params["ny_pm_kill_start"]), int(self.params["ny_pm_kill_end"])),
        ]

        rolling_high = high.shift(1).rolling(structure_lookback, min_periods=structure_lookback).max()
        rolling_low = low.shift(1).rolling(structure_lookback, min_periods=structure_lookback).min()
        atr = _compute_atr(df, int(self.params["atr_period"]))

        metadata: dict[str, Any] = {
            "bullish_continuation_candidates": 0,
            "bearish_continuation_candidates": 0,
            "bullish_continuation_setups": 0,
            "bearish_continuation_setups": 0,
            "bullish_retest_candidates": 0,
            "bearish_retest_candidates": 0,
            "fvg_delay_filtered_retests": 0,
            "fvg_depth_filtered_retests": 0,
            "fvg_entries": 0,
            "long_entries": 0,
            "short_entries": 0,
        }

        position = 0
        active_stop = np.nan
        active_target = np.nan
        pending_long: _ContinuationPending | None = None
        pending_short: _ContinuationPending | None = None

        for idx, ts in enumerate(df.index):
            current_atr = float(atr.iat[idx]) if pd.notna(atr.iat[idx]) else 0.0
            price = float(close.iat[idx])
            next_is_new_day = (
                idx == len(df.index) - 1
                or not _same_trading_day(pd.Timestamp(ts), pd.Timestamp(df.index[idx + 1]))
            )
            in_session = _in_trade_session(
                pd.Timestamp(ts),
                trade_sessions,
                london_open,
                london_close,
                ny_open,
                ny_close,
            )
            in_kill_zone = _in_kill_zone(
                pd.Timestamp(ts),
                use_kill_zones,
                kill_zone_timezone,
                kill_windows,
            )

            if position > 0:
                if price <= active_stop or price >= active_target or next_is_new_day:
                    exits_long.iat[idx] = True
                    position = 0
                    active_stop = np.nan
                    active_target = np.nan
                    pending_long = None
                    pending_short = None
                    continue
            elif position < 0:
                if price >= active_stop or price <= active_target or next_is_new_day:
                    exits_short.iat[idx] = True
                    position = 0
                    active_stop = np.nan
                    active_target = np.nan
                    pending_long = None
                    pending_short = None
                    continue

            if next_is_new_day and position == 0:
                pending_long = None
                pending_short = None

            if pending_long is not None and idx > pending_long.expiry_index:
                pending_long = None
            if pending_short is not None and idx > pending_short.expiry_index:
                pending_short = None

            if position == 0 and in_session and in_kill_zone:
                prior_high = rolling_high.iat[idx]
                if pending_long is None and pd.notna(prior_high) and price > float(prior_high):
                    metadata["bullish_continuation_candidates"] += 1
                    candle_range = max(float(high.iat[idx]) - float(low.iat[idx]), 1e-12)
                    body_ratio = abs(float(close.iat[idx]) - float(open_.iat[idx])) / candle_range
                    if body_ratio >= displacement_body_min_pct:
                        zone = _detect_fvg_zone(
                            df,
                            max(0, idx - structure_lookback),
                            idx,
                            bullish=True,
                            min_gap_pct=fvg_min_gap_pct,
                        )
                        if zone is not None:
                            _, zone_lower, zone_upper = zone
                            pending_long = _ContinuationPending(
                                direction=1,
                                breakout_index=idx,
                                structure_level=float(prior_high),
                                breakout_extreme=float(high.iat[idx]),
                                expiry_index=idx + fvg_max_age,
                                armed_index=idx,
                                zone_lower=zone_lower,
                                zone_upper=zone_upper,
                            )
                            metadata["bullish_continuation_setups"] += 1

                prior_low = rolling_low.iat[idx]
                if pending_short is None and pd.notna(prior_low) and price < float(prior_low):
                    metadata["bearish_continuation_candidates"] += 1
                    candle_range = max(float(high.iat[idx]) - float(low.iat[idx]), 1e-12)
                    body_ratio = abs(float(close.iat[idx]) - float(open_.iat[idx])) / candle_range
                    if body_ratio >= displacement_body_min_pct:
                        zone = _detect_fvg_zone(
                            df,
                            max(0, idx - structure_lookback),
                            idx,
                            bullish=False,
                            min_gap_pct=fvg_min_gap_pct,
                        )
                        if zone is not None:
                            _, zone_lower, zone_upper = zone
                            pending_short = _ContinuationPending(
                                direction=-1,
                                breakout_index=idx,
                                structure_level=float(prior_low),
                                breakout_extreme=float(low.iat[idx]),
                                expiry_index=idx + fvg_max_age,
                                armed_index=idx,
                                zone_lower=zone_lower,
                                zone_upper=zone_upper,
                            )
                            metadata["bearish_continuation_setups"] += 1

            long_entry_executed = False
            if (
                position == 0
                and in_session
                and in_kill_zone
                and pending_long is not None
                and idx > pending_long.armed_index
                and float(low.iat[idx]) <= pending_long.zone_upper
                and float(high.iat[idx]) >= pending_long.zone_lower
            ):
                if not pending_long.retest_seen:
                    metadata["bullish_retest_candidates"] += 1
                    pending_long.retest_seen = True
                pending_long.retest_touches += 1
                if idx - pending_long.armed_index < fvg_revisit_min_delay_bars:
                    metadata["fvg_delay_filtered_retests"] += 1
                else:
                    gap_height = max(pending_long.zone_upper - pending_long.zone_lower, 1e-12)
                    required_touch = pending_long.zone_upper - gap_height * fvg_revisit_depth_ratio
                    if float(low.iat[idx]) > required_touch:
                        metadata["fvg_depth_filtered_retests"] += 1
                    else:
                        entries_long.iat[idx] = True
                        position = 1
                        atr_buffer = max(current_atr * stop_loss_atr_mult, price * 0.002)
                        active_stop = min(pending_long.zone_lower - current_atr * 0.1, price - atr_buffer)
                        if active_stop >= price:
                            active_stop = price - atr_buffer
                        active_target = price + (price - active_stop) * take_profit_rr
                        metadata["long_entries"] += 1
                        metadata["fvg_entries"] += 1
                        pending_long = None
                        pending_short = None
                        long_entry_executed = True

            if not long_entry_executed and (
                position == 0
                and in_session
                and in_kill_zone
                and pending_short is not None
                and idx > pending_short.armed_index
                and float(low.iat[idx]) <= pending_short.zone_upper
                and float(high.iat[idx]) >= pending_short.zone_lower
            ):
                if not pending_short.retest_seen:
                    metadata["bearish_retest_candidates"] += 1
                    pending_short.retest_seen = True
                pending_short.retest_touches += 1
                if idx - pending_short.armed_index < fvg_revisit_min_delay_bars:
                    metadata["fvg_delay_filtered_retests"] += 1
                else:
                    gap_height = max(pending_short.zone_upper - pending_short.zone_lower, 1e-12)
                    required_touch = pending_short.zone_lower + gap_height * fvg_revisit_depth_ratio
                    if float(high.iat[idx]) < required_touch:
                        metadata["fvg_depth_filtered_retests"] += 1
                    else:
                        entries_short.iat[idx] = True
                        position = -1
                        atr_buffer = max(current_atr * stop_loss_atr_mult, price * 0.002)
                        active_stop = max(pending_short.zone_upper + current_atr * 0.1, price + atr_buffer)
                        if active_stop <= price:
                            active_stop = price + atr_buffer
                        active_target = price - (active_stop - price) * take_profit_rr
                        metadata["short_entries"] += 1
                        metadata["fvg_entries"] += 1
                        pending_long = None
                        pending_short = None

            if entries_short.iat[idx]:
                continue

        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            metadata=metadata,
        )


def _compare_verdict(reversal_metrics: dict[str, float], continuation_metrics: dict[str, float]) -> tuple[str, str]:
    reversal_trades = int(reversal_metrics["total_trades"])
    continuation_trades = int(continuation_metrics["total_trades"])
    reversal_return = float(reversal_metrics["total_return_pct"])
    continuation_return = float(continuation_metrics["total_return_pct"])
    if continuation_trades > reversal_trades and continuation_return >= reversal_return:
        return (
            "ROBUST_CONTINUATION_LANE_OUTPERFORMS_REVERSAL",
            "The continuation proxy improves both density and total return versus the active lite reversal frontier.",
        )
    if continuation_trades > reversal_trades and continuation_return > 0:
        return (
            "CONTINUATION_LANE_DENSITY_CANDIDATE",
            "The continuation proxy improves trade density and stays positive, but it does not beat the active lite reversal frontier on return.",
        )
    if continuation_return > 0:
        return (
            "CONTINUATION_LANE_SURVIVES_BUT_NOT_DENSER",
            "The continuation proxy stays positive, but it does not improve density versus the active lite reversal frontier.",
        )
    return (
        "CONTINUATION_LANE_REJECTED",
        "The continuation proxy does not currently justify a promotion-ready branch versus the active lite reversal frontier.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the active lite reversal frontier against a BOS->FVG continuation proxy lane."
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
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    reversal_strategy = ICTEntryModelStrategy(params=params)
    reversal_result = engine.run(reversal_strategy, merged)
    reversal_signals = reversal_strategy.generate_signals(merged)

    continuation_strategy = ICTContinuationProxyStrategy(params=params)
    continuation_result = engine.run(continuation_strategy, merged)
    continuation_signals = continuation_strategy.generate_signals(merged)

    verdict, interpretation = _compare_verdict(reversal_result.metrics, continuation_result.metrics)
    output = {
        "analysis": "ict_continuation_lane_compare",
        "primary_intraday_csv": str(Path(args.intraday_csv)),
        "peer_symbol": args.peer_symbol,
        "peer_csv": args.peer_csv,
        "period": args.period,
        "rows_primary": len(intraday_df),
        "rows_peer": len(peer_df),
        "matched_peer_bars": int(merged["PeerHigh"].notna().sum()) if "PeerHigh" in merged.columns else 0,
        "active_lite_reversal_frontier": {
            "metrics": reversal_result.metrics,
            "metadata": {
                "fvg_entries": int(reversal_signals.metadata.get("fvg_entries", 0)),
                "ob_entries": int(reversal_signals.metadata.get("ob_entries", 0)),
                "breaker_entries": int(reversal_signals.metadata.get("breaker_entries", 0)),
                "bullish_sweeps": int(reversal_signals.metadata.get("bullish_sweeps", 0)),
                "bearish_sweeps": int(reversal_signals.metadata.get("bearish_sweeps", 0)),
                "bullish_shifts": int(reversal_signals.metadata.get("bullish_shifts", 0)),
                "bearish_shifts": int(reversal_signals.metadata.get("bearish_shifts", 0)),
            },
        },
        "continuation_proxy_lane": {
            "metrics": continuation_result.metrics,
            "metadata": continuation_signals.metadata,
        },
        "trade_delta_vs_reversal": int(continuation_result.metrics["total_trades"]) - int(reversal_result.metrics["total_trades"]),
        "return_delta_vs_reversal_pct": round(
            float(continuation_result.metrics["total_return_pct"]) - float(reversal_result.metrics["total_return_pct"]),
            4,
        ),
        "verdict": verdict,
        "interpretation": interpretation,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
