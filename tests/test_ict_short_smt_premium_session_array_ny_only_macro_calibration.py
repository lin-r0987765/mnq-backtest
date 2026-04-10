from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_session_array_ny_only_macro_calibration import (
    _rank_variants,
    _variant_specs,
)


class ICTShortSmtPremiumSessionArrayNyOnlyMacroCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_base_and_macro_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("ny_only_base", specs)
        self.assertIn("macro_default", specs)
        self.assertIn("macro_first_window_only", specs)
        self.assertIn("macro_early_shifted", specs)

    def test_rank_variants_prefers_positive_variants_by_return_then_retention(self) -> None:
        ranked = _rank_variants(
            {
                "ny_only_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.14, "profit_factor": 2.8},
                    "metadata": {},
                },
                "macro_early_shifted": {
                    "metrics": {"total_trades": 3, "total_return_pct": 0.15, "profit_factor": 2.0},
                    "metadata": {},
                },
                "macro_default": {
                    "metrics": {"total_trades": 1, "total_return_pct": 0.03, "profit_factor": 1.2},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "macro_early_shifted")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.6)


if __name__ == "__main__":
    unittest.main()
