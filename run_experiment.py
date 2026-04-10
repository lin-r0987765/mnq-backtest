import json
from src.backtest.engine import BacktestEngine
from src.data.fetcher import fetch_peer_data, load_ohlcv_csv, merge_peer_columns
from src.strategies.ict_entry_model import (
    ICTEntryModelStrategy,
    build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params,
)

def run_experiment():
    intraday_df = load_ohlcv_csv("alpaca/normalized/qqq_5m_alpaca.csv")
    peer_df = fetch_peer_data(peer_symbol="SPY", peer_csv="alpaca/normalized/spy_5m_alpaca.csv", period="60d")
    merged = merge_peer_columns(intraday_df, peer_df)
    engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0005, size=10.0)

    base_params = {
        "structure_lookback": 8,
        "liq_sweep_threshold": 0.0006,
        "liq_sweep_recovery_bars": 4,
        "slow_recovery_enabled": True,
        "slow_recovery_bars": 12,
        "fvg_min_gap_pct": 0.0003,
        "fvg_revisit_depth_ratio": 0.5,
        "fvg_revisit_min_delay_bars": 4,
        "enable_continuation_entry": True,
        "min_reward_risk_ratio": 1.5,
    }

    variants = [
        ("Base_93", {}),
        ("Reduce_Delay", {"fvg_revisit_min_delay_bars": 1}),
        ("Reduce_Depth", {"fvg_revisit_depth_ratio": 0.0}),
        ("Reduce_Delay_Depth", {"fvg_revisit_min_delay_bars": 1, "fvg_revisit_depth_ratio": 0.0}),
        ("Struct5_LongRec", {"structure_lookback": 5, "slow_recovery_bars": 24, "fvg_revisit_min_delay_bars": 1, "fvg_revisit_depth_ratio": 0.0}),
        ("No_SMT", {"use_smt_filter": False, "structure_lookback": 5, "slow_recovery_bars": 24, "fvg_revisit_min_delay_bars": 1, "fvg_revisit_depth_ratio": 0.0}),
    ]

    for label, changes in variants:
        test_params = base_params.copy()
        test_params.update(changes)
        
        # Start with the correct baseline overrides
        final_params = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
            enable_smt=test_params.get("use_smt_filter", True),
            overrides=test_params
        )

        strategy = ICTEntryModelStrategy(params=final_params)
        result = engine.run(strategy, merged)
        metrics = result.metrics
        print(f"[{label}] Trades: {int(metrics['total_trades'])}, Return: {float(metrics['total_return_pct']):.4f}%, PF: {float(metrics['profit_factor']):.4f}")

if __name__ == "__main__":
    run_experiment()
