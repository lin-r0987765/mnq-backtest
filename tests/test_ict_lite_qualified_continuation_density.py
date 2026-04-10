from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_qualified_continuation_density import _rank_variants, _verdict


class ICTLiteQualifiedContinuationDensityTest(unittest.TestCase):
    def test_rank_variants_prefers_highest_positive_trade_count(self) -> None:
        variants = {
            "active_lite_frontier_base": {
                "params": {
                    "structure_lookback": 12,
                    "slow_recovery_bars": 0,
                    "fvg_min_gap_pct": 0.0010,
                    "fvg_revisit_depth_ratio": 0.5,
                    "enable_continuation_entry": False,
                },
                "metrics": {
                    "total_trades": 18,
                    "total_return_pct": 13.8395,
                    "profit_factor": 6.3079,
                    "win_rate_pct": 72.2222,
                },
                "metadata": {
                    "continuation_entries": 0,
                    "slow_recovery_entries": 0,
                },
            },
            "qualified_continuation_density": {
                "params": {
                    "structure_lookback": 8,
                    "slow_recovery_bars": 12,
                    "fvg_min_gap_pct": 0.0005,
                    "fvg_revisit_depth_ratio": 0.5,
                    "enable_continuation_entry": True,
                },
                "metrics": {
                    "total_trades": 99,
                    "total_return_pct": 4.9880,
                    "profit_factor": 1.4148,
                    "win_rate_pct": 42.0,
                },
                "metadata": {
                    "continuation_entries": 23,
                    "slow_recovery_entries": 32,
                },
            },
        }
        ranked = _rank_variants(variants)
        self.assertEqual(ranked[0]["label"], "qualified_continuation_density")

    def test_verdict_marks_density_branch_identified_when_gate_is_met(self) -> None:
        verdict, _ = _verdict(
            best_positive_density_variant={
                "label": "qualified_continuation_density",
                "total_trades": 99,
                "total_return_pct": 4.9880,
            },
            best_balanced_variant={
                "label": "balanced_density_candidate",
                "total_trades": 60,
                "total_return_pct": 8.0209,
            },
            density_gate=80,
        )
        self.assertEqual(verdict, "QUALIFIED_CONTINUATION_DENSITY_BRANCH_IDENTIFIED")

    def test_verdict_marks_balanced_survivor_when_gate_is_missed(self) -> None:
        verdict, _ = _verdict(
            best_positive_density_variant=None,
            best_balanced_variant={
                "label": "balanced_density_candidate",
                "total_trades": 60,
                "total_return_pct": 8.0209,
            },
            density_gate=80,
        )
        self.assertEqual(verdict, "QUALIFIED_CONTINUATION_BALANCED_SURVIVOR_ONLY")


if __name__ == "__main__":
    unittest.main()
