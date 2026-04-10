from __future__ import annotations

import unittest

from research.ict.analyze_ict_strict_baseline_summary import _assemble_strict_summary
from src.backtest.engine import BacktestEngine


class ICTStrictBaselineSummaryTest(unittest.TestCase):
    def test_summary_payload_exposes_risk_block_and_sweeps(self) -> None:
        engine = BacktestEngine(
            initial_cash=100_000.0,
            fees_pct=0.0005,
            position_size_mode="capital_pct",
            capital_usage_pct=1.0,
            min_shares=40,
        )
        params = {
            "take_profit_rr": 4.0,
            "min_reward_risk_ratio": 1.5,
        }
        metadata = {
            "fvg_entries": 8,
            "ob_entries": 0,
            "breaker_entries": 0,
            "ifvg_entries": 0,
            "bullish_sweeps": 3,
            "bearish_sweeps": 5,
            "bullish_sweep_candidates": 10,
            "bearish_sweep_candidates": 8,
            "bullish_shift_candidates": 2,
            "bearish_shift_candidates": 3,
            "bullish_shifts": 2,
            "bearish_shifts": 2,
            "bullish_retest_candidates": 1,
            "bearish_retest_candidates": 1,
            "long_entries": 1,
            "short_entries": 1,
        }
        payload = _assemble_strict_summary(
            metrics={"total_trades": 8, "total_return_pct": 7.7249},
            metadata=metadata,
            params=params,
            engine=engine,
        )

        self.assertEqual(payload["risk_block"]["min_reward_risk_ratio"], 1.5)
        self.assertEqual(payload["risk_block"]["min_shares"], 40)
        self.assertEqual(payload["sweep_breakdown"]["accepted_sweeps"], 8)
        self.assertEqual(payload["delivery_breakdown"]["fvg_entries"], 8)


if __name__ == "__main__":
    unittest.main()
