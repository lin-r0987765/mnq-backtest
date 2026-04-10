from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_sweep_threshold_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureSweepThresholdCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_threshold_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("sweep_threshold_0p0006", specs)
        self.assertIn("sweep_threshold_0p0012", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1826, "profit_factor": 3.0},
                    "params": {"liq_sweep_threshold": 0.0008},
                    "metadata": {"bullish_sweeps": 100, "bearish_sweeps": 90},
                },
                "sweep_threshold_0p0007": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"liq_sweep_threshold": 0.0007},
                    "metadata": {"bullish_sweeps": 110, "bearish_sweeps": 95},
                },
                "sweep_threshold_0p0012": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"liq_sweep_threshold": 0.0012},
                    "metadata": {"bullish_sweeps": 80, "bearish_sweeps": 70},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "sweep_threshold_0p0007")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
