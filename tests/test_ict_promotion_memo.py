from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMO_PATH = PROJECT_ROOT / "ICT_PROMOTION_MEMO.md"


class ICTPromotionMemoTest(unittest.TestCase):
    def test_memo_exists(self) -> None:
        self.assertTrue(MEMO_PATH.exists())

    def test_memo_contains_required_decision_points(self) -> None:
        text = MEMO_PATH.read_text(encoding="utf-8")
        self.assertIn("DO_NOT_PROMOTE_YET", text)
        self.assertIn("ict_strict_baseline_summary.json", text)
        self.assertIn("ict_walk_forward_results.json", text)
        self.assertIn("ict_combined_lanes.json", text)
        self.assertIn("18 trades", text)
        self.assertIn("1464 trades", text)


if __name__ == "__main__":
    unittest.main()
