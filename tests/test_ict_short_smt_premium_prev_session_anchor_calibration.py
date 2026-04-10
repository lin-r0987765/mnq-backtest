from __future__ import annotations

import unittest

from research.ict.analyze_ict_short_smt_premium_prev_session_anchor_calibration import _rank_variants, _variant_specs


class ICTShortSmtPremiumPrevSessionAnchorCalibrationTest(unittest.TestCase):
    def test_variant_specs_include_anchor_control_and_tolerance_variants(self) -> None:
        specs = _variant_specs()
        self.assertIn("premium_base", specs)
        self.assertIn("anchor_off_control", specs)
        self.assertIn("anchor_tighter_0p03", specs)
        self.assertIn("anchor_looser_0p12", specs)

    def test_rank_variants_prefers_positive_robust_anchor_variant(self) -> None:
        ranked = _rank_variants(
            {
                "premium_base": {
                    "metrics": {"total_trades": 5, "total_return_pct": 0.13, "profit_factor": 2.7},
                    "metadata": {},
                },
                "anchor_looser_0p08": {
                    "metrics": {"total_trades": 3, "total_return_pct": 0.14, "profit_factor": 2.0},
                    "metadata": {},
                },
                "anchor_off_control": {
                    "metrics": {"total_trades": 0, "total_return_pct": 0.0, "profit_factor": 0.0},
                    "metadata": {},
                },
            },
            base_trades=5,
        )
        self.assertEqual(ranked[0]["label"], "anchor_looser_0p08")
        self.assertAlmostEqual(ranked[0]["trade_retention_vs_base"], 0.6)


if __name__ == "__main__":
    unittest.main()
