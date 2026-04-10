from __future__ import annotations

import unittest

from run_ict_walk_forward import _fold_windows, _summarize_folds, _verdict


class ICTWalkForwardTest(unittest.TestCase):
    def test_fold_windows_builds_expected_count(self) -> None:
        windows = _fold_windows(
            list(range(140)),
            train_days=60,
            validation_days=20,
            holdout_days=20,
            step_days=20,
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]["train_start"], 0)
        self.assertEqual(windows[-1]["holdout_end"], 140)

    def test_summarize_folds_aggregates_holdout_stats(self) -> None:
        folds = [
            {
                "fold": 1,
                "validation": {"total_return_pct": 0.20},
                "holdout": {"total_return_pct": 0.10, "sharpe_ratio": 1.2, "profit_factor": 2.0, "total_trades": 4},
            },
            {
                "fold": 2,
                "validation": {"total_return_pct": 0.30},
                "holdout": {"total_return_pct": -0.05, "sharpe_ratio": 0.4, "profit_factor": 1.2, "total_trades": 3},
            },
        ]
        summary = _summarize_folds(folds)
        self.assertEqual(summary["folds"], 2)
        self.assertEqual(summary["holdout_trade_total"], 7)
        self.assertEqual(summary["positive_holdout_fold_pct"], 50.0)
        self.assertEqual(summary["best_holdout_fold"]["fold"], 1)

    def test_verdict_marks_positive_oos_stability(self) -> None:
        verdict, _ = _verdict(
            {
                "folds": 5,
                "avg_holdout_return_pct": 0.12,
                "positive_holdout_fold_pct": 80.0,
            }
        )
        self.assertEqual(verdict, "ICT_WALK_FORWARD_POSITIVE_OOS_STABILITY_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
