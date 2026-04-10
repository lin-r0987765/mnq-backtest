from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_reversal_baseline import (
    RESEARCH_STANDARD,
    _density_summary,
    _engine_config_payload,
    _verdict,
)
from src.backtest.engine import BacktestEngine


class ICTLiteReversalBaselineTest(unittest.TestCase):
    def test_engine_config_payload_matches_table_risk_standard(self) -> None:
        engine = BacktestEngine(**RESEARCH_STANDARD)
        payload = _engine_config_payload(engine)
        self.assertEqual(payload["position_size_mode"], "capital_pct")
        self.assertEqual(payload["capital_usage_pct"], 1.0)
        self.assertEqual(payload["min_shares"], 40)

    def test_density_summary_reports_trade_gain_and_100_trade_gap(self) -> None:
        strict_payload = {"metrics": {"total_trades": 8, "total_return_pct": 0.2119}}
        lite_payload = {"metrics": {"total_trades": 120, "total_return_pct": 1.25}}
        summary = _density_summary(strict_payload, lite_payload)
        self.assertEqual(summary["trade_gain"], 112)
        self.assertEqual(summary["toward_100_trade_gate"], 0)
        self.assertEqual(summary["trade_gain_multiple"], 15.0)

    def test_verdict_marks_profitable_hundred_trade_clear(self) -> None:
        density = {
            "strict_trades": 8,
            "lite_trades": 120,
        }
        lite_payload = {"metrics": {"total_return_pct": 1.25}}
        verdict, _ = _verdict(density, lite_payload)
        self.assertEqual(verdict, "LITE_ICT_REVERSAL_BASELINE_CLEARS_100_TRADE_GATE")

    def test_verdict_marks_density_improvement_below_gate(self) -> None:
        density = {
            "strict_trades": 8,
            "lite_trades": 40,
        }
        lite_payload = {"metrics": {"total_return_pct": 0.5}}
        verdict, _ = _verdict(density, lite_payload)
        self.assertEqual(verdict, "LITE_ICT_REVERSAL_BASELINE_IMPROVES_DENSITY_BUT_STAYS_BELOW_100_TRADES")


if __name__ == "__main__":
    unittest.main()
