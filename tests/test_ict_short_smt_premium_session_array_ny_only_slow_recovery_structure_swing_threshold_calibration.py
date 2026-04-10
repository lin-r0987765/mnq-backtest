from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_swing_threshold_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureSwingThresholdCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_swing_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("swing_threshold_2", specs)
        self.assertIn("swing_threshold_6", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1826, "profit_factor": 3.0},
                    "params": {"swing_threshold": 3},
                    "metadata": {"bullish_sweeps": 120, "bearish_sweeps": 100},
                },
                "swing_threshold_2": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"swing_threshold": 2},
                    "metadata": {"bullish_sweeps": 140, "bearish_sweeps": 110},
                },
                "swing_threshold_6": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"swing_threshold": 6},
                    "metadata": {"bullish_sweeps": 90, "bearish_sweeps": 70},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "swing_threshold_2")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
