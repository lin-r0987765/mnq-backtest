from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_rr_gate import _rank_variants, _verdict


class ICTLiteRRGateTest(unittest.TestCase):
    def test_rank_variants_prefers_more_active_positive_variant(self) -> None:
        variants = {
            "lite_frontier_base": {
                "params": {"take_profit_rr": 4.0, "min_reward_risk_ratio": 1.5},
                "metrics": {"total_trades": 18, "total_return_pct": 0.3529, "profit_factor": 4.6792},
                "metadata": {"rr_filtered_entries": 0, "fvg_entries": 13, "ob_entries": 4, "breaker_entries": 1},
            },
            "rr_gate_off_control": {
                "params": {"take_profit_rr": 4.0, "min_reward_risk_ratio": 0.0},
                "metrics": {"total_trades": 19, "total_return_pct": 0.36, "profit_factor": 4.1},
                "metadata": {"rr_filtered_entries": 0, "fvg_entries": 14, "ob_entries": 4, "breaker_entries": 1},
            },
        }
        ranked = _rank_variants(variants, base_trades=18)
        self.assertEqual(ranked[0]["label"], "rr_gate_off_control")

    def test_verdict_marks_robust_extension(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "rr_gate_2p0", "total_trades": 18, "total_return_pct": 0.36},
            best_robust_variant={"label": "rr_gate_2p0", "total_trades": 18, "total_return_pct": 0.36},
            rr_gate_control_variant={"label": "tp_rr_1p0_gate_1p5_control", "rr_filtered_entries": 5, "total_trades": 0},
        )
        self.assertEqual(verdict, "ROBUST_LITE_RR_GATE_EXTENSION_IDENTIFIED")

    def test_verdict_marks_implemented_plateau(self) -> None:
        verdict, _ = _verdict(
            base_trades=18,
            base_return=0.3529,
            best_non_base={"label": "rr_gate_4p0", "total_trades": 18, "total_return_pct": 0.3529},
            best_robust_variant=None,
            rr_gate_control_variant={"label": "tp_rr_1p0_gate_1p5_control", "rr_filtered_entries": 7, "total_trades": 0},
        )
        self.assertEqual(verdict, "RR_GATE_IMPLEMENTED_AND_PLATEAUED_ON_LITE_FRONTIER")


if __name__ == "__main__":
    unittest.main()
