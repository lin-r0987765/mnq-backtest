from __future__ import annotations

import unittest

from research.ict.analyze_ict_survivor_geometry_calibration import _rank_variants, _variant_specs


class ICTSurvivorGeometryCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_geometry_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("survivor_plus_session_array_base", specs)
        self.assertIn("shorter_sweep_lookback_30", specs)
        self.assertIn("looser_sweep_threshold_0p0008", specs)
        self.assertIn("shorter_lookback_looser_sweep_looser_smt", specs)

    def test_rank_variants_prefers_positive_geometry_variant(self) -> None:
        ranked = _rank_variants(
            {
                "survivor_plus_session_array_base": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.05, "profit_factor": 1.9},
                    "metadata": {},
                },
                "shorter_sweep_lookback_30": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.08, "profit_factor": 2.1},
                    "metadata": {},
                },
                "looser_sweep_threshold_0p0008": {
                    "metrics": {"total_trades": 6, "total_return_pct": 0.04, "profit_factor": 1.4},
                    "metadata": {},
                },
            },
            base_trades=4,
        )
        self.assertEqual(ranked[0]["label"], "shorter_sweep_lookback_30")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 1.0)


if __name__ == "__main__":
    unittest.main()
