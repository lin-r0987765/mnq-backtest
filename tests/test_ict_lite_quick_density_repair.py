from __future__ import annotations

import unittest

from research.ict.analyze_ict_lite_quick_density_repair import _density_summary, _verdict


class ICTLiteQuickDensityRepairTest(unittest.TestCase):
    def test_density_summary_reports_gains_against_strict_and_active_lite(self) -> None:
        strict_payload = {"metrics": {"total_trades": 8, "total_return_pct": 0.2119}}
        active_lite_payload = {"metrics": {"total_trades": 18, "total_return_pct": 0.4353}}
        quick_payload = {"metrics": {"total_trades": 98, "total_return_pct": 0.6}}

        summary = _density_summary(strict_payload, active_lite_payload, quick_payload)

        self.assertEqual(summary["gain_vs_strict"], 90)
        self.assertEqual(summary["gain_vs_active_lite"], 80)
        self.assertEqual(summary["toward_100_trade_gate"], 2)
        self.assertEqual(summary["multiple_vs_active_lite"], 5.44)

    def test_verdict_marks_first_density_gate_clear(self) -> None:
        density = {"quick_repair_trades": 120, "active_lite_trades": 18}
        active_lite_payload = {"metrics": {"total_return_pct": 0.4353}}
        quick_payload = {"metrics": {"total_return_pct": 0.5}}

        verdict, _ = _verdict(density, active_lite_payload, quick_payload)

        self.assertEqual(verdict, "QUICK_DENSITY_REPAIR_CLEARS_FIRST_DENSITY_GATE")

    def test_verdict_marks_density_only_extension(self) -> None:
        density = {"quick_repair_trades": 40, "active_lite_trades": 18}
        active_lite_payload = {"metrics": {"total_return_pct": 0.4353}}
        quick_payload = {"metrics": {"total_return_pct": 0.31}}

        verdict, _ = _verdict(density, active_lite_payload, quick_payload)

        self.assertEqual(verdict, "QUICK_DENSITY_REPAIR_DENSITY_EXTENSION_ONLY")


if __name__ == "__main__":
    unittest.main()
