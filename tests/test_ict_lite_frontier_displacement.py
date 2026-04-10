from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_frontier_displacement import _rank_variants, _verdict


class ICTLiteFrontierDisplacementTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"displacement_body_min_pct": 0.1},
                "metrics": {"total_trades": 18, "total_return_pct": 0.3529, "profit_factor": 4.6792},
                "metadata": {"displacement_filtered_shifts": 2, "fvg_entries": 13, "ob_entries": 4, "breaker_entries": 1},
            },
            "displacement_0p00": {
                "params": {"displacement_body_min_pct": 0.0},
                "metrics": {"total_trades": 20, "total_return_pct": 0.28, "profit_factor": 3.2},
                "metadata": {"displacement_filtered_shifts": 0, "fvg_entries": 15, "ob_entries": 4, "breaker_entries": 1},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "displacement_0p00")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "displacement_0p05", "total_trades": 19, "total_return_pct": 0.36},
            best_positive_density_variant={"label": "displacement_0p05", "total_trades": 19, "total_return_pct": 0.36},
            best_robust_variant={"label": "displacement_0p05", "total_trades": 19, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_FRONTIER_DISPLACEMENT_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "displacement_0p00", "total_trades": 20, "total_return_pct": 0.28},
            best_positive_density_variant={"label": "displacement_0p00", "total_trades": 20, "total_return_pct": 0.28},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_FRONTIER_DISPLACEMENT_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
