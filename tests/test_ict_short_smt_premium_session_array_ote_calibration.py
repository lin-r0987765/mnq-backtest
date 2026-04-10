from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ote_calibration import _rank_variants, _variant_specs


class ICTShortSmtPremiumSessionArrayOteCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_ote_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("session_array_premium_base", specs)
        self.assertIn("ote_off_control", specs)
        self.assertIn("ote_tighter_band", specs)
        self.assertIn("ote_higher_score", specs)

    def test_rank_variants_prefers_higher_positive_return(self) -> None:
        ranked = _rank_variants(
            {
                "session_array_premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "ote_higher_score": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "ote_off_control": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.01, "profit_factor": 1.1},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "ote_higher_score")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.8)


if __name__ == "__main__":
    unittest.main()
