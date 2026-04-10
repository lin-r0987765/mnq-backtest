from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_score_quality_repair import _rank_variants, _verdict


class ICTLiteScoreQualityRepairTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {
                    "min_score_to_trade": 6.0,
                    "score_sweep_depth_quality": 0.0,
                    "score_displacement_quality": 0.0,
                    "score_fvg_gap_quality": 0.0,
                },
                "metrics": {"total_trades": 18, "total_return_pct": 0.4353, "profit_factor": 5.0247},
                "metadata": {
                    "bullish_shifts": 8,
                    "bearish_shifts": 10,
                    "score_filtered_shifts": 0,
                    "score_quality_boosted_shifts": 0,
                    "score_quality_bonus_total": 0.0,
                },
            },
            "quality_score_7": {
                "params": {
                    "min_score_to_trade": 7.0,
                    "score_sweep_depth_quality": 1.0,
                    "score_displacement_quality": 1.0,
                    "score_fvg_gap_quality": 1.0,
                },
                "metrics": {"total_trades": 20, "total_return_pct": 0.31, "profit_factor": 3.0},
                "metadata": {
                    "bullish_shifts": 8,
                    "bearish_shifts": 10,
                    "score_filtered_shifts": 4,
                    "score_quality_boosted_shifts": 12,
                    "score_quality_bonus_total": 7.0,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "quality_score_7")

    def test_verdict_marks_repaired_but_not_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            base_score_filtered_shifts=0,
            best_non_base={
                "label": "quality_score_8",
                "total_trades": 18,
                "total_return_pct": 0.4353,
                "score_filtered_shifts": 3,
            },
            best_repaired_variant={
                "label": "quality_score_8",
                "total_trades": 18,
                "total_return_pct": 0.4353,
                "score_filtered_shifts": 3,
            },
            best_positive_density_variant=None,
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "QUALITY_SCORE_SYSTEM_REPAIRED_BUT_NOT_EXTENSION")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.4353,
            base_score_filtered_shifts=0,
            best_non_base={
                "label": "quality_score_7",
                "total_trades": 18,
                "total_return_pct": 0.45,
                "score_filtered_shifts": 2,
            },
            best_repaired_variant=None,
            best_positive_density_variant=None,
            best_robust_variant={
                "label": "quality_score_7",
                "total_trades": 18,
                "total_return_pct": 0.45,
                "score_filtered_shifts": 2,
            },
        )
        self.assertEqual(verdict, "ROBUST_LITE_SCORE_QUALITY_EXTENSION_IDENTIFIED")


if __name__ == "__main__":
    unittest.main()
