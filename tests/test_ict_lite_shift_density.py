from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_shift_density import _rank_variants, _verdict


class ICTLiteShiftDensityTest(unittest.TestCase):
    def test_rank_variants_prefers_positive_trade_gain(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"structure_lookback": 12},
                "metrics": {"total_trades": 18, "total_return_pct": 0.3529, "profit_factor": 4.6792},
                "metadata": {
                    "fvg_entries": 13,
                    "ob_entries": 4,
                    "breaker_entries": 1,
                    "bullish_shift_candidates": 20,
                    "bearish_shift_candidates": 18,
                    "bullish_shifts": 12,
                    "bearish_shifts": 10,
                },
            },
            "structure_8": {
                "params": {"structure_lookback": 8},
                "metrics": {"total_trades": 20, "total_return_pct": 0.30, "profit_factor": 3.5},
                "metadata": {
                    "fvg_entries": 14,
                    "ob_entries": 5,
                    "breaker_entries": 1,
                    "bullish_shift_candidates": 25,
                    "bearish_shift_candidates": 22,
                    "bullish_shifts": 14,
                    "bearish_shifts": 12,
                },
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "structure_8")
        self.assertEqual(ranked[0]["shift_candidates"], 47)

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "structure_8", "total_trades": 19, "total_return_pct": 0.36},
            best_robust_variant={"label": "structure_8", "total_trades": 19, "total_return_pct": 0.36},
        )
        self.assertEqual(verdict, "ROBUST_LITE_SHIFT_EXTENSION_IDENTIFIED")

    def test_verdict_marks_density_only_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "structure_8", "total_trades": 20, "total_return_pct": 0.30},
            best_robust_variant=None,
        )
        self.assertEqual(verdict, "LITE_SHIFT_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
