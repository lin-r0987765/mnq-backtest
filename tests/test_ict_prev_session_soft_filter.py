from __future__ import annotations

import unittest

from research.ict.analyze_ict_prev_session_soft_filter import _rank_variants, _verdict


class ICTPrevSessionSoftFilterTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "strict_prev_anchor_hard_base": {
                "params": {
                    "prev_session_anchor_filter_mode": "hard",
                    "prev_session_anchor_tolerance": 0.05,
                    "prev_session_anchor_mismatch_score_penalty": 0.0,
                },
                "metrics": {"total_trades": 10, "total_return_pct": 0.18, "profit_factor": 2.0},
                "metadata": {
                    "prev_session_anchor_filtered_setups": 9,
                    "prev_session_anchor_softened_setups": 0,
                    "score_filtered_shifts": 0,
                },
            },
            "prev_anchor_soft_penalty_2": {
                "params": {
                    "prev_session_anchor_filter_mode": "soft",
                    "prev_session_anchor_tolerance": 0.05,
                    "prev_session_anchor_mismatch_score_penalty": 2.0,
                },
                "metrics": {"total_trades": 14, "total_return_pct": 0.22, "profit_factor": 2.3},
                "metadata": {
                    "prev_session_anchor_filtered_setups": 0,
                    "prev_session_anchor_softened_setups": 12,
                    "score_filtered_shifts": 2,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=10)
        self.assertEqual(ranked[0]["label"], "prev_anchor_soft_penalty_2")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=10,
            base_return=0.18,
            best_non_base={"label": "prev_anchor_soft_penalty_2", "total_trades": 14, "total_return_pct": 0.22},
            best_positive_density_variant={
                "label": "prev_anchor_soft_penalty_2",
                "total_trades": 14,
                "total_return_pct": 0.22,
            },
            best_robust_variant={"label": "prev_anchor_soft_penalty_2", "total_trades": 14, "total_return_pct": 0.22},
        )
        self.assertEqual(verdict, "ROBUST_PREV_SESSION_SOFT_FILTER_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=10,
            base_return=0.18,
            best_non_base={"label": "prev_anchor_soft_penalty_0", "total_trades": 12, "total_return_pct": 0.12},
            best_positive_density_variant={
                "label": "prev_anchor_soft_penalty_0",
                "total_trades": 12,
                "total_return_pct": 0.12,
            },
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "PREV_SESSION_SOFT_FILTER_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
