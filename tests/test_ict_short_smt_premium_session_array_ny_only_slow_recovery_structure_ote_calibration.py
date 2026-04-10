from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_ote_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureOTECalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_ote_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("ote_off_control", specs)
        self.assertIn("ote_tighter_higher_score", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1826, "profit_factor": 3.0},
                    "params": {"ote_fib_low": 0.618, "ote_fib_high": 0.786, "score_ote_zone": 2.0},
                    "metadata": {"ob_entries": 0},
                },
                "ote_tighter_band": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.19, "profit_factor": 2.0},
                    "params": {"ote_fib_low": 0.66, "ote_fib_high": 0.79, "score_ote_zone": 2.0},
                    "metadata": {"ob_entries": 0},
                },
                "ote_looser_band": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"ote_fib_low": 0.55, "ote_fib_high": 0.82, "score_ote_zone": 2.0},
                    "metadata": {"ob_entries": 0},
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "ote_tighter_band")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
