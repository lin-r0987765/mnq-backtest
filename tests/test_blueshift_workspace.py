from __future__ import annotations

import unittest
from pathlib import Path

from blueshift.blueshift_library.orb_v26_runtime import build_config


class BlueshiftWorkspaceTest(unittest.TestCase):
    def test_runtime_defaults_match_v26_baseline(self) -> None:
        cfg = build_config()
        self.assertEqual(cfg["script_version"], "v26-profit-lock-blueshift")
        self.assertEqual(cfg["baseline_reference"], "v26-profit-lock")
        self.assertEqual(cfg["symbol"], "QQQ")
        self.assertEqual(cfg["orb_bars"], 4)
        self.assertEqual(cfg["trailing_pct"], 0.013)
        self.assertEqual(cfg["breakeven_trigger_mult"], 1.25)
        self.assertEqual(cfg["breakeven_active_minutes"], 180)
        self.assertEqual(cfg["profit_lock_trigger_mult"], 1.50)
        self.assertEqual(cfg["profit_lock_level_mult"], 0.25)
        self.assertFalse(cfg["orb_reentry_enabled"])

    def test_baseline_entry_file_contains_expected_markers(self) -> None:
        path = Path("blueshift/v26_profit_lock_blueshift.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn('from blueshift_library.orb_v26_runtime import handle_data, initialize_strategy', text)
        self.assertIn('"v26-profit-lock-blueshift"', text)
        self.assertIn('"v26-profit-lock"', text)
        self.assertIn('"orb_reentry_enabled": False', text)

    def test_evaluator_entry_file_contains_expected_markers(self) -> None:
        path = Path("blueshift/v26_orb_reentry_evaluator_blueshift.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn('"v26-orb-reentry-evaluator-blueshift"', text)
        self.assertIn('"baseline_reference": "v26-profit-lock"', text)
        self.assertIn('"research_only": True', text)
        self.assertIn('"orb_reentry_enabled": True', text)
        self.assertIn('"orb_reentry_arm_progress_mult": 1.0', text)
        self.assertIn('"orb_reentry_depth_mult": 0.25', text)
        self.assertIn('"orb_reentry_confirm_bars": 1', text)


if __name__ == "__main__":
    unittest.main()
