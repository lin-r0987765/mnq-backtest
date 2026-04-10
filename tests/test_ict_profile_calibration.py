from __future__ import annotations

import unittest

from research.ict.analyze_ict_paired_profile_calibration import _rank_variants, _variant_specs


class ICTProfileCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_full_stack_and_relaxed_bundle(self) -> None:
        specs = _variant_specs()
        self.assertIn("full_stack_smt", specs)
        self.assertIn("context_relaxed_bundle", specs)
        self.assertTrue(bool(specs["full_stack_smt"]["enable_smt"]))
        self.assertFalse(bool(specs["no_smt"]["enable_smt"]))

    def test_rank_variants_prefers_positive_trading_variant(self) -> None:
        ranked = _rank_variants(
            {
                "full_stack_smt": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
                "relaxed": {
                    "metrics": {"total_trades": 3, "total_return_pct": 1.2, "profit_factor": 1.1},
                    "metadata": {},
                },
            }
        )
        self.assertEqual(ranked[0]["label"], "relaxed")


if __name__ == "__main__":
    unittest.main()
