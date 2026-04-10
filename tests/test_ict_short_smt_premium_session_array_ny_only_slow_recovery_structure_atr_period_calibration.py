from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_atr_period_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureATRPeriodCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_atr_period_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("atr_period_10", specs)
        self.assertIn("atr_period_20", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1831, "profit_factor": 3.0, "avg_trade_pct": 0.0262},
                    "params": {"atr_period": 14},
                    "metadata": {"fvg_entries": 7},
                },
                "atr_period_12": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0, "avg_trade_pct": 0.0316},
                    "params": {"atr_period": 12},
                    "metadata": {"fvg_entries": 6},
                },
                "atr_period_20": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5, "avg_trade_pct": 0.0100},
                    "params": {"atr_period": 20},
                    "metadata": {"fvg_entries": 8},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "atr_period_12")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
