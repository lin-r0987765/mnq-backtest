from __future__ import annotations

import unittest

from research.ict.analyze_ict_survivor_pairwise_calibration import _rank_variants, _variant_specs


class ICTSurvivorPairwiseCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_survivor_base_and_pairwise_additions(self) -> None:
        specs = _variant_specs()
        self.assertIn("survivor_base", specs)
        self.assertIn("survivor_plus_premium_discount", specs)
        self.assertIn("survivor_plus_session_array", specs)
        self.assertIn("survivor_plus_premium_discount_and_session_array", specs)
        self.assertTrue(bool(specs["survivor_base"]["enable_smt"]))

    def test_rank_variants_prefers_positive_robust_extension(self) -> None:
        ranked = _rank_variants(
            {
                "survivor_base": {
                    "metrics": {"total_trades": 7, "total_return_pct": 0.1, "profit_factor": 1.9},
                    "metadata": {},
                },
                "survivor_plus_premium_discount": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.12, "profit_factor": 2.0},
                    "metadata": {},
                },
                "survivor_plus_session_array": {
                    "metrics": {"total_trades": 4, "total_return_pct": 0.05, "profit_factor": 1.8},
                    "metadata": {},
                },
            },
            survivor_base_trades=7,
        )
        self.assertEqual(ranked[0]["label"], "survivor_plus_premium_discount")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_survivor_base"], 5 / 7)


if __name__ == "__main__":
    unittest.main()
