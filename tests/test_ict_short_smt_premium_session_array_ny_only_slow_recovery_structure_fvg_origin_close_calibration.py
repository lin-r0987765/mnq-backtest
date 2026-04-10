from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_slow_recovery_structure_fvg_origin_close_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlySlowRecoveryStructureFvgOriginCloseCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_close_position_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("structure_frontier_base", specs)
        self.assertIn("fvg_origin_close_0p60", specs)
        self.assertIn("fvg_origin_close_0p95", specs)

    def test_rank_variants_prefers_positive_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "structure_frontier_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.2057, "profit_factor": 3.0},
                    "params": {"fvg_origin_close_position_min_pct": 0.0},
                    "metadata": {
                        "fvg_entries": 7,
                        "ifvg_entries": 0,
                        "fvg_origin_close_filtered_shifts": 0,
                        "bullish_sweeps": 151,
                        "bearish_sweeps": 122,
                    },
                },
                "fvg_origin_close_0p80": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.21, "profit_factor": 2.0},
                    "params": {"fvg_origin_close_position_min_pct": 0.80},
                    "metadata": {
                        "fvg_entries": 6,
                        "ifvg_entries": 0,
                        "fvg_origin_close_filtered_shifts": 1,
                        "bullish_sweeps": 151,
                        "bearish_sweeps": 122,
                    },
                },
                "fvg_origin_close_0p95": {
                    "metrics": {"total_trades": 8, "total_return_pct": 0.08, "profit_factor": 1.5},
                    "params": {"fvg_origin_close_position_min_pct": 0.95},
                    "metadata": {
                        "fvg_entries": 8,
                        "ifvg_entries": 0,
                        "fvg_origin_close_filtered_shifts": 4,
                        "bullish_sweeps": 151,
                        "bearish_sweeps": 122,
                    },
                },
            },
            base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "fvg_origin_close_0p80")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
