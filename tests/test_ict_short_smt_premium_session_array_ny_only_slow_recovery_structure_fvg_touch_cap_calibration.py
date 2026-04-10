from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_touch_cap_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureFvgTouchCapCalibrationTest(
    unittest.TestCase
):
    def test_variant_specs_include_base_and_touch_cap_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("fvg_touch_cap_1", specs)
        self.assertIn("fvg_touch_cap_5", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.2057, "profit_factor": 3.0},
                    "params": {"fvg_max_retest_touches": 0},
                    "metadata": {
                        "fvg_entries": 7,
                        "fvg_touch_filtered_retests": 0,
                        "bullish_sweeps": 151,
                        "bearish_sweeps": 122,
                    },
                },
                "fvg_touch_cap_2": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.21, "profit_factor": 2.0},
                    "params": {"fvg_max_retest_touches": 2},
                    "metadata": {
                        "fvg_entries": 6,
                        "fvg_touch_filtered_retests": 2,
                        "bullish_sweeps": 150,
                        "bearish_sweeps": 120,
                    },
                },
                "fvg_touch_cap_5": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"fvg_max_retest_touches": 5},
                    "metadata": {
                        "fvg_entries": 8,
                        "fvg_touch_filtered_retests": 4,
                        "bullish_sweeps": 155,
                        "bearish_sweeps": 125,
                    },
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "fvg_touch_cap_2")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
