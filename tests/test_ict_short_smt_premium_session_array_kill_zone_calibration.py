from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_kill_zone_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayKillZoneCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_kill_zone_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("session_array_premium_base", specs)
        self.assertIn("kill_zones_default", specs)
        self.assertIn("kill_zones_ny_am_only", specs)
        self.assertIn("kill_zones_broader", specs)

    def test_rank_variants_prefers_positive_robust_kill_zone_variant(self) -> None:
        ranked = _rank_variants(
            {
                "session_array_premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "kill_zones_broader": {
                    "metrics": {"total_trades": 3, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "kill_zones_default": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "kill_zones_broader")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.6)


if __name__ == "__main__":
    unittest.main()
