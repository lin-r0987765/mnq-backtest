from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_fvg_calibration import _rank_variants, _variant_specs


class ICTShortSmtPremiumSessionArrayFvgCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_fvg_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("session_array_premium_base", specs)
        self.assertIn("fvg_tighter_gap", specs)
        self.assertIn("fvg_looser_gap", specs)
        self.assertIn("fvg_shorter_age", specs)
        self.assertIn("fvg_looser_and_longer", specs)

    def test_rank_variants_sorts_positive_variants_by_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "session_array_premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "fvg_tighter_gap": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "fvg_looser_gap": {
                    "metrics": {"total_trades": 2, "total_return_pct": 0.2, "profit_factor": 10.0},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "fvg_looser_gap")
        self.assertAlmostEqual(ranked[1]["trade_retention_vs_base"], 0.8)


if __name__ == "__main__":
    unittest.main()
