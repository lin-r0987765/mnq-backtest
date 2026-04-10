from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_session_density import _rank_variants, _verdict


class ICTLiteSessionDensityTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_density(self) -> None:
        variants = {
            "lite_faster_retest_base": {
                "params": {"trade_sessions": True, "london_open": 0, "london_close": 0, "ny_open": 14, "ny_close": 20},
                "metrics": {"total_trades": 18, "total_return_pct": 0.3529, "profit_factor": 4.6792},
                "metadata": {"fvg_entries": 13, "ob_entries": 4, "breaker_entries": 1},
            },
            "session_off_control": {
                "params": {"trade_sessions": False, "london_open": 0, "london_close": 0, "ny_open": 14, "ny_close": 20},
                "metrics": {"total_trades": 30, "total_return_pct": 0.25, "profit_factor": 2.0},
                "metadata": {"fvg_entries": 20, "ob_entries": 6, "breaker_entries": 2},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "session_off_control")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "ny_13_21", "total_trades": 19, "total_return_pct": 0.36},
            best_positive_density_variant={"label": "ny_13_21", "total_trades": 19, "total_return_pct": 0.36},
            best_robust_variant={"label": "ny_13_21", "total_trades": 19, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_SESSION_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "session_off_control", "total_trades": 30, "total_return_pct": 0.25},
            best_positive_density_variant={"label": "session_off_control", "total_trades": 30, "total_return_pct": 0.25},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_SESSION_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
