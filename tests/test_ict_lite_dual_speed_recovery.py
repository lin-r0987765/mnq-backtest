from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_dual_speed_recovery import _rank_variants, _verdict


class ICTLiteDualSpeedRecoveryTest(unittest.TestCase):
    def test_rank_variants_prefers_best_positive_return(self) -> None:
        variants = {
            "fast_only_base": {
                "params": {"slow_recovery_enabled": False, "slow_recovery_bars": 0},
                "metrics": {
                    "total_trades": 18,
                    "total_return_pct": 13.8395,
                    "profit_factor": 6.3079,
                    "win_rate_pct": 72.2222,
                },
                "metadata": {
                    "fast_recovery_entries": 18,
                    "slow_recovery_entries": 0,
                    "sweep_expired_before_shift": 1000,
                    "armed_setup_expired_before_retest": 18,
                    "delivery_missing_shifts": 20,
                    "fvg_depth_filtered_retests": 22,
                },
            },
            "dual_speed_6": {
                "params": {"slow_recovery_enabled": True, "slow_recovery_bars": 6},
                "metrics": {
                    "total_trades": 28,
                    "total_return_pct": 10.2286,
                    "profit_factor": 2.9682,
                    "win_rate_pct": 53.5714,
                },
                "metadata": {
                    "fast_recovery_entries": 18,
                    "slow_recovery_entries": 10,
                    "sweep_expired_before_shift": 959,
                    "armed_setup_expired_before_retest": 24,
                    "delivery_missing_shifts": 23,
                    "fvg_depth_filtered_retests": 29,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "fast_only_base")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=13.8395,
            best_density_variant={"label": "dual_speed_6", "total_trades": 28, "total_return_pct": 14.1},
            best_return_variant={"label": "dual_speed_6", "total_trades": 28, "total_return_pct": 14.1},
        )
        self.assertEqual(verdict, "DUAL_SPEED_RECOVERY_ROBUST_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_when_return_trails_base(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=13.8395,
            best_density_variant={"label": "dual_speed_12", "total_trades": 45, "total_return_pct": 10.02},
            best_return_variant={"label": "dual_speed_12", "total_trades": 45, "total_return_pct": 10.02},
        )
        self.assertEqual(verdict, "DUAL_SPEED_RECOVERY_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
