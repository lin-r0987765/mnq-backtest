from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_setup_density import _rank_variants, _verdict


class ICTLiteSetupDensityTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_faster_retest_base": {
                "params": {"fvg_max_age": 20},
                "metrics": {"total_trades": 18, "total_return_pct": 0.3529, "profit_factor": 4.6792},
                "metadata": {"fvg_entries": 13, "ob_entries": 4, "breaker_entries": 1, "bullish_shifts": 8, "bearish_shifts": 10},
            },
            "setup_age_30": {
                "params": {"fvg_max_age": 30},
                "metrics": {"total_trades": 20, "total_return_pct": 0.30, "profit_factor": 3.8},
                "metadata": {"fvg_entries": 15, "ob_entries": 4, "breaker_entries": 1, "bullish_shifts": 9, "bearish_shifts": 11},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "setup_age_30")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "setup_age_30", "total_trades": 19, "total_return_pct": 0.36},
            best_positive_density_variant={"label": "setup_age_30", "total_trades": 19, "total_return_pct": 0.36},
            best_robust_variant={"label": "setup_age_30", "total_trades": 19, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_SETUP_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "setup_age_30", "total_trades": 20, "total_return_pct": 0.30},
            best_positive_density_variant={"label": "setup_age_30", "total_trades": 20, "total_return_pct": 0.30},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_SETUP_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
