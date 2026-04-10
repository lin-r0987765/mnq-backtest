from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_sweep_geometry_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySweepGeometryCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_sweep_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("ny_only_base", specs)
        self.assertIn("sweep_shorter_lookback_35", specs)
        self.assertIn("sweep_faster_recovery_2", specs)
        self.assertIn("sweep_shorter_and_faster", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "ny_only_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.142, "profit_factor": 2.8},
                    "params": {"liq_sweep_lookback": 50, "liq_sweep_recovery_bars": 3},
                    "metadata": {},
                },
                "sweep_shorter_and_faster": {
                    "metrics": {"total_trades": 3, "total_return_pct": 0.145, "profit_factor": 2.0},
                    "params": {"liq_sweep_lookback": 35, "liq_sweep_recovery_bars": 2},
                    "metadata": {},
                },
                "sweep_longer_and_slower": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.08, "profit_factor": 1.7},
                    "params": {"liq_sweep_lookback": 70, "liq_sweep_recovery_bars": 4},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "sweep_shorter_and_faster")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.6)


if __name__ == "__main__":
    unittest.main()
