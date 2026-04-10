from __future__ import annotations

import unittest

from research.ict.analyze_ict_premium_discount_soft_filter import _rank_variants, _verdict


class ICTPremiumDiscountSoftFilterTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "strict_premium_hard_base": {
                "params": {
                    "premium_discount_filter_mode": "hard",
                    "premium_discount_neutral_band": 0.05,
                    "premium_discount_mismatch_score_penalty": 0.0,
                },
                "metrics": {"total_trades": 10, "total_return_pct": 0.18, "profit_factor": 2.0},
                "metadata": {
                    "premium_discount_filtered_setups": 20,
                    "premium_discount_softened_setups": 0,
                    "score_filtered_shifts": 0,
                },
            },
            "premium_soft_penalty_2": {
                "params": {
                    "premium_discount_filter_mode": "soft",
                    "premium_discount_neutral_band": 0.05,
                    "premium_discount_mismatch_score_penalty": 2.0,
                },
                "metrics": {"total_trades": 14, "total_return_pct": 0.22, "profit_factor": 2.3},
                "metadata": {
                    "premium_discount_filtered_setups": 0,
                    "premium_discount_softened_setups": 18,
                    "score_filtered_shifts": 3,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=10)
        self.assertEqual(ranked[0]["label"], "premium_soft_penalty_2")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=10,
            base_return=0.18,
            best_non_base={"label": "premium_soft_penalty_2", "total_trades": 14, "total_return_pct": 0.22},
            best_positive_density_variant={
                "label": "premium_soft_penalty_2",
                "total_trades": 14,
                "total_return_pct": 0.22,
            },
            best_robust_variant={"label": "premium_soft_penalty_2", "total_trades": 14, "total_return_pct": 0.22},
        )
        self.assertEqual(verdict, "ROBUST_PREMIUM_DISCOUNT_SOFT_FILTER_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=10,
            base_return=0.18,
            best_non_base={"label": "premium_soft_penalty_0", "total_trades": 16, "total_return_pct": 0.12},
            best_positive_density_variant={
                "label": "premium_soft_penalty_0",
                "total_trades": 16,
                "total_return_pct": 0.12,
            },
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "PREMIUM_DISCOUNT_SOFT_FILTER_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
