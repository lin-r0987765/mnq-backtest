from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_smt_density import _rank_variants, _verdict


class ICTLiteSMTDensityTest(unittest.TestCase):
    def test_rank_variants_prioritizes_positive_density(self) -> None:
        variants = {
            "lite_base": {
                "params": {"use_smt_filter": True, "smt_lookback": 10, "smt_threshold": 0.001},
                "metrics": {"total_trades": 14, "total_return_pct": 0.2832, "profit_factor": 7.5},
                "metadata": {"smt_filtered_sweeps": 301, "fvg_entries": 11, "ob_entries": 2, "breaker_entries": 1},
            },
            "lite_smt_off_control": {
                "params": {"use_smt_filter": False, "smt_lookback": 10, "smt_threshold": 0.001},
                "metrics": {"total_trades": 20, "total_return_pct": 0.2, "profit_factor": 3.0},
                "metadata": {"smt_filtered_sweeps": 0, "fvg_entries": 15, "ob_entries": 3, "breaker_entries": 2},
            },
        }
        ranked = _rank_variants(variants, base_trades=14)
        self.assertEqual(ranked[0]["label"], "lite_smt_off_control")

    def test_verdict_marks_robust_extension_when_trades_and_return_improve(self) -> None:
        verdict, _ = _verdict(
            base_trades=14,
            base_return=0.2832,
            best_non_base={"label": "lite_smt_lookback_8", "total_trades": 18, "total_return_pct": 0.35},
            best_positive_density_variant={"label": "lite_smt_lookback_8", "total_trades": 18, "total_return_pct": 0.35},
            best_robust_variant={"label": "lite_smt_lookback_8", "total_trades": 18, "total_return_pct": 0.35},
        )
        self.assertEqual(verdict, "ROBUST_LITE_SMT_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension_when_return_lags(self) -> None:
        verdict, _ = _verdict(
            base_trades=14,
            base_return=0.2832,
            best_non_base={"label": "lite_smt_off_control", "total_trades": 20, "total_return_pct": 0.2},
            best_positive_density_variant={"label": "lite_smt_off_control", "total_trades": 20, "total_return_pct": 0.2},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_SMT_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
