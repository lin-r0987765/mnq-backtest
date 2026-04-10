from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from research.qc.analyze_qc_webide_result import build_report, count_same_bar_eod_reentries


class AnalyzeQcWebIdeResultTest(unittest.TestCase):
    def test_count_same_bar_eod_reentries(self) -> None:
        orders = pd.DataFrame(
            {
                "Time": [
                    "2026-01-01T20:50:00Z",
                    "2026-01-01T20:50:00Z",
                    "2026-01-02T20:50:00Z",
                ],
                "Tag": [
                    "ORB EOD Flatten",
                    "ORB Long",
                    "ORB Short",
                ],
            }
        )
        rows = count_same_bar_eod_reentries(orders)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "2026-01-01T20:50:00Z")

    def test_build_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prefix = "Sample Run"
            (base / f"{prefix}.json").write_text(
                json.dumps({"statistics": {"Net Profit": "1.23%", "Sharpe Ratio": "2.34", "Win Rate": "55%", "Drawdown": "1.5%"}}),
                encoding="utf-8",
            )
            (base / f"{prefix}_logs.txt").write_text(
                "QQQ ORB WebIDE init | version=v8-halfday-cutoff-verify | cutoff=exchange-hours-aware\n"
                "QQQ ORB WebIDE half-day detected | version=v8-halfday-cutoff-verify\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "Time": ["2026-01-01T20:50:00Z", "2026-01-01T20:50:00Z"],
                    "Tag": ["ORB EOD Flatten", "ORB Long"],
                }
            ).to_csv(base / f"{prefix}_orders.csv", index=False)
            pd.DataFrame({"ProfitLoss": [1.0]}).to_csv(base / f"{prefix}_trades.csv", index=False)

            report = build_report(base)
            self.assertTrue(report["contains_version_marker"])
            self.assertEqual(report["detected_versions"], ["v8-halfday-cutoff-verify"])
            self.assertTrue(report["contains_halfday_marker"])
            self.assertEqual(report["same_bar_eod_reentry_count"], 1)
            self.assertEqual(report["net_profit"], "1.23%")


if __name__ == "__main__":
    unittest.main()
