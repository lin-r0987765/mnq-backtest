from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_reversal_balance_branch import _rank_variants, _verdict


class ICTLiteReversalBalanceBranchTest(unittest.TestCase):
    def test_rank_variants_prefers_stronger_return_quality_branch(self) -> None:
        variants = {
            "active_lite_quality_base": {
                "params": {
                    "structure_lookback": 12,
                    "slow_recovery_bars": 0,
                    "fvg_min_gap_pct": 0.0010,
                    "fvg_revisit_depth_ratio": 0.5,
                    "fvg_revisit_min_delay_bars": 2,
                    "take_profit_rr": 4.0,
                    "enable_continuation_entry": False,
                    "smt_threshold": 0.0015,
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
            "qualified_reversal_balance": {
                "params": {
                    "structure_lookback": 12,
                    "slow_recovery_bars": 12,
                    "fvg_min_gap_pct": 0.0003,
                    "fvg_revisit_depth_ratio": 0.5,
                    "fvg_revisit_min_delay_bars": 4,
                    "take_profit_rr": 3.0,
                    "enable_continuation_entry": False,
                    "smt_threshold": 0.0010,
                },
                "metrics": {
                    "total_trades": 40,
                    "total_return_pct": 13.4177,
                    "profit_factor": 3.1879,
                    "win_rate_pct": 60.0,
                },
                "metadata": {
                    "continuation_entries": 0,
                    "slow_recovery_entries": 14,
                },
            },
            "qualified_continuation_density": {
                "params": {
                    "structure_lookback": 8,
                    "slow_recovery_bars": 12,
                    "fvg_min_gap_pct": 0.0003,
                    "fvg_revisit_depth_ratio": 0.5,
                    "fvg_revisit_min_delay_bars": 4,
                    "take_profit_rr": 3.0,
                    "enable_continuation_entry": True,
                    "smt_threshold": 0.0013,
                },
                "metrics": {
                    "total_trades": 93,
                    "total_return_pct": 11.8319,
                    "profit_factor": 1.8131,
                    "win_rate_pct": 47.3118,
                },
                "metadata": {
                    "continuation_entries": 62,
                    "slow_recovery_entries": 34,
                },
            },
        }
        ranked = _rank_variants(variants)
        self.assertEqual(ranked[0]["label"], "active_lite_quality_base")
        self.assertEqual(ranked[1]["label"], "qualified_reversal_balance")

    def test_verdict_identifies_balanced_reversal_branch(self) -> None:
        verdict, _ = _verdict(
            quality_base={
                "label": "active_lite_quality_base",
                "total_trades": 18,
                "total_return_pct": 13.8395,
                "profit_factor": 6.3079,
            },
            reversal_balance={
                "label": "qualified_reversal_balance",
                "total_trades": 40,
                "total_return_pct": 13.4177,
                "profit_factor": 3.1879,
            },
            continuation_density={
                "label": "qualified_continuation_density",
                "total_trades": 93,
                "total_return_pct": 11.8319,
                "profit_factor": 1.8131,
            },
        )
        self.assertEqual(verdict, "QUALIFIED_REVERSAL_BALANCE_BRANCH_IDENTIFIED")

    def test_verdict_rejects_non_improving_branch(self) -> None:
        verdict, _ = _verdict(
            quality_base={
                "label": "active_lite_quality_base",
                "total_trades": 18,
                "total_return_pct": 13.8395,
                "profit_factor": 6.3079,
            },
            reversal_balance={
                "label": "qualified_reversal_balance",
                "total_trades": 18,
                "total_return_pct": 10.0,
                "profit_factor": 2.0,
            },
            continuation_density={
                "label": "qualified_continuation_density",
                "total_trades": 93,
                "total_return_pct": 11.8319,
                "profit_factor": 1.8131,
            },
        )
        self.assertEqual(verdict, "QUALIFIED_REVERSAL_BALANCE_REJECTED")


if __name__ == "__main__":
    unittest.main()
