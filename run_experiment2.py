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

    base = build_ict_lite_reversal_relaxed_smt_looser_sweep_faster_retest_profile_params(
        enable_smt=False,
        overrides={
            "structure_lookback": 5,
            "liq_sweep_threshold": 0.0003,
            "liq_sweep_recovery_bars": 10,
            "slow_recovery_enabled": True,
            "slow_recovery_bars": 30,
            "fvg_min_gap_pct": 0.0001,
            "fvg_revisit_depth_ratio": 0.0,
            "fvg_revisit_min_delay_bars": 1,
            "enable_continuation_entry": True,
            "min_reward_risk_ratio": 1.0,  # Lower RR requirement
            "use_smt_filter": False,
        }
    )

    strategy = ICTEntryModelStrategy(params=base)
    result = engine.run(strategy, merged)
    metrics = result.metrics
    print(f"[Maximum Density] Trades: {int(metrics['total_trades'])}, Return: {float(metrics['total_return_pct']):.4f}%, PF: {float(metrics['profit_factor']):.4f}")

if __name__ == "__main__":
    run_experiment()
