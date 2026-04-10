from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_revisit_depth import _rank_variants, _verdict


class ICTLiteFrontierRevisitDepthTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"fvg_revisit_depth_ratio": 0.5},
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247},
                "metadata": {"fvg_depth_filtered_retests": 22, "fvg_entries": 12, "ob_entries": 5, "breaker_entries": 1},
            },
            "depth_0p00": {
                "params": {"fvg_revisit_depth_ratio": 0.0},
                "metrics": {"total_trades": 20, "total_return_pct": 0.3496, "profit_factor": 3.3842},
                "metadata": {"fvg_depth_filtered_retests": 0, "fvg_entries": 14, "ob_entries": 5, "breaker_entries": 1},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "depth_0p00")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "depth_0p25", "total_trades": 19, "total_return_pct": 0.36},
            best_positive_density_variant={"label": "depth_0p25", "total_trades": 19, "total_return_pct": 0.36},
            best_robust_variant={"label": "depth_0p25", "total_trades": 19, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_FRONTIER_REVISIT_DEPTH_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            best_non_base={"label": "depth_0p00", "total_trades": 20, "total_return_pct": 0.3496},
            best_positive_density_variant={"label": "depth_0p00", "total_trades": 20, "total_return_pct": 0.3496},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_FRONTIER_REVISIT_DEPTH_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
