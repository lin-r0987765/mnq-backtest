from __future__ import annotations

import unittest

from research.ict.analyze_ict_swing_recovery_repair import _rank_variants, _verdict


class ICTSwingRecoveryRepairTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_return_then_pf(self) -> None:
        variants = {
            "swing_3_recovery_15": {
                "params": {
                    "swing_threshold": 3,
                    "liq_sweep_recovery_bars": 15,
                    "slow_recovery_enabled": False,
                    "slow_recovery_bars": 15,
                },
                "metrics": {"total_trades": 20, "total_return_pct": -0.08, "profit_factor": 0.6, "win_rate_pct": 35.0},
            },
            "swing_3_recovery_20": {
                "params": {
                    "swing_threshold": 3,
                    "liq_sweep_recovery_bars": 20,
                    "slow_recovery_enabled": False,
                    "slow_recovery_bars": 20,
                },
                "metrics": {"total_trades": 35, "total_return_pct": 0.05, "profit_factor": 1.2, "win_rate_pct": 45.0},
            },
        }
        ranked = _rank_variants(variants)
        self.assertEqual(ranked[0]["label"], "swing_3_recovery_20")

    def test_verdict_marks_gate_clear(self) -> None:
        verdict, _ = _verdict({"metrics": {"total_trades": 120, "total_return_pct": 0.2}})
        self.assertEqual(verdict, "SWING_RECOVERY_REPAIR_CLEARS_FIRST_DENSITY_GATE")

    def test_verdict_marks_positive_below_gate(self) -> None:
        verdict, _ = _verdict({"metrics": {"total_trades": 40, "total_return_pct": 0.1}})
        self.assertEqual(verdict, "SWING_RECOVERY_REPAIR_POSITIVE_BUT_BELOW_GATE")


if __name__ == "__main__":
    unittest.main()
