from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from research.qc.analyze_qc_regime_windows import (
    compute_profit_factor,
    compute_sharpe,
    load_daily_equity,
    rolling_windows,
)


class AnalyzeQcRegimeWindowsTest(unittest.TestCase):
    def test_compute_profit_factor(self) -> None:
        pnl = pd.Series([2.0, -1.0, 3.0, -2.0])
        self.assertAlmostEqual(compute_profit_factor(pnl), 5.0 / 3.0, places=6)

    def test_compute_sharpe_zero_std(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        self.assertEqual(compute_sharpe(returns), 0.0)

    def test_load_daily_equity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            payload = {
                "charts": {
                    "Strategy Equity": {
                        "series": {
                            "Equity": {
                                "values": [
                                    [1704067200, 100.0, 101.0, 99.0, 100.5],
                                    [1704153600, 100.5, 102.0, 100.0, 101.5],
                                ]
                            }
                        }
                    }
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            daily = load_daily_equity(path)
            self.assertEqual(len(daily), 2)
            self.assertEqual(float(daily.iloc[-1]["equity_close"]), 101.5)

    def test_rolling_windows(self) -> None:
        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        daily = pd.DataFrame(
            {
                "date": dates,
                "equity_close": 100000 * (1.0005 ** np.arange(len(dates))),
                "daily_return": [0.0] + [0.0005] * (len(dates) - 1),
            }
        )
        trades = pd.DataFrame(
            {
                "Exit Time": pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]),
                "Entry Time": pd.to_datetime(["2025-03-30", "2025-06-29", "2025-09-29"]),
                "P&L": [100.0, -50.0, 120.0],
                "Net P&L": [98.0, -52.0, 118.0],
                "Fees": [2.0, 2.0, 2.0],
                "IsWinNet": [1, 0, 1],
            }
        )
        out = rolling_windows(daily, trades, 6)
        self.assertGreaterEqual(len(out), 1)
        self.assertIn("sharpe", out.columns)
        self.assertIn("trades", out.columns)


if __name__ == "__main__":
    unittest.main()
