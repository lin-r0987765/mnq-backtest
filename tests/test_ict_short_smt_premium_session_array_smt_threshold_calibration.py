from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_smt_threshold_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArraySmtThresholdCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_control_and_threshold_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("session_array_premium_base", specs)
        self.assertIn("smt_off_control", specs)
        self.assertIn("smt_tighter_0p0008", specs)
        self.assertIn("smt_looser_0p0015", specs)

    def test_rank_variants_prefers_positive_variants_by_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "session_array_premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "smt_tighter_0p0008": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "smt_off_control": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.12, "profit_factor": 1.8},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "smt_tighter_0p0008")
        self.assertEqual(ranked[2]["label"], "smt_off_control")
        self.assertAlmostEqual(ranked[2]["trade_retention_vs_base"], 1.6)


if __name__ == "__main__":
    unittest.main()
