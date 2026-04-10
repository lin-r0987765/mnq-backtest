from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_session_gate_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArraySessionGateCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_session_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("session_array_premium_base", specs)
        self.assertIn("trade_sessions_off_control", specs)
        self.assertIn("broader_sessions", specs)
        self.assertIn("narrower_london_ny_overlap", specs)

    def test_rank_variants_prefers_positive_variants_by_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "session_array_premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "broader_sessions": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "trade_sessions_off_control": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.08, "profit_factor": 2.4},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "session_array_premium_base")
        self.assertEqual(ranked[2]["label"], "trade_sessions_off_control")
        self.assertAlmostEqual(ranked[2]["trade_retention_vs_base"], 1.4)


if __name__ == "__main__":
    unittest.main()
