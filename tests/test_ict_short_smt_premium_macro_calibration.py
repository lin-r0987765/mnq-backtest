from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_macro_calibration import _rank_variants, _variant_specs


class ICTShortSmtPremiumMacroCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_premium_base_and_macro_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("premium_base", specs)
        self.assertIn("macro_default", specs)
        self.assertIn("macro_broader_windows", specs)
        self.assertIn("macro_second_window_only", specs)

    def test_rank_variants_prefers_positive_robust_macro_variant(self) -> None:
        ranked = _rank_variants(
            {
                "premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "macro_default": {
                    "metrics": {"total_trades": 3, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "macro_second_window_only": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "macro_default")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.6)


if __name__ == "__main__":
    unittest.main()
