from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.live.config import LiveTradingConfig
from src.live.service import LiveTradingService


class LiveTradingServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "live.json"
        self.service = self._build_service()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_service(self, overrides: dict | None = None) -> LiveTradingService:
        payload = {
            "mode": "paper",
            "shared_secret": "secret-123",
            "default_volume": 10.0,
            "position_sizing_mode": "fixed",
            "account_equity": 100000.0,
            "risk_per_trade_pct": 0.0025,
            "max_notional_pct": 0.06,
            "contract_multiplier": 1.0,
            "min_volume": 1.0,
            "max_volume": 25.0,
            "volume_step": 1.0,
            "allow_signal_qty_override": False,
            "allow_default_fallback": True,
            "allowed_strategies": ["ORB"],
            "symbol_map": {"QQQ": "QQQ.US"},
            "event_log_path": str(self.root / "events.jsonl"),
            "paper_log_path": str(self.root / "paper.jsonl"),
            "paper_state_path": str(self.root / "state.json"),
            "risk_state_path": str(self.root / "risk.json"),
            "kill_switch_path": str(self.root / "kill_switch.json"),
            "dedupe_window_seconds": 3600,
            "cooldown_seconds": 0,
            "max_daily_entries_per_symbol": 10,
            "max_open_positions_per_symbol": 5,
            "paper_max_daily_loss": 0.0,
        }
        if overrides:
            payload.update(overrides)
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")
        return LiveTradingService(LiveTradingConfig.load(self.config_path))

    def test_entry_then_exit(self) -> None:
        entry = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "order_id": "orb-1"
            }
        )
        self.assertTrue(entry["result"]["success"])
        self.assertEqual(entry["broker_symbol"], "QQQ.US")

        exit_event = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "exit",
                "side": "buy",
                "order_id": "orb-1-exit"
            }
        )
        self.assertTrue(exit_event["result"]["success"])
        self.assertEqual(exit_event["result"]["details"]["closed_positions"], 1)

    def test_duplicate_signal_ignored(self) -> None:
        payload = {
            "secret": "secret-123",
            "strategy": "ORB",
            "symbol": "QQQ",
            "action": "entry",
            "side": "buy",
            "order_id": "dup-1",
            "event_time": "2026-04-04T14:30:00Z"
        }
        first = self.service.process_payload(payload)
        second = self.service.process_payload(payload)
        self.assertTrue(first["result"]["success"])
        self.assertTrue(second["result"]["details"]["duplicate"])

    def test_kill_switch_blocks_entry_but_allows_exit(self) -> None:
        self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "price": 100.0,
                "order_id": "orb-ks-1",
            }
        )
        self.service.risk.set_kill_switch(True, "manual halt")

        blocked = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "price": 101.0,
                "order_id": "orb-ks-2",
            }
        )
        self.assertFalse(blocked["result"]["success"])
        self.assertTrue(blocked["result"]["details"]["blocked"])

        exit_event = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "exit",
                "side": "buy",
                "price": 102.0,
                "order_id": "orb-ks-exit",
            }
        )
        self.assertTrue(exit_event["result"]["success"])
        self.assertEqual(exit_event["result"]["details"]["closed_positions"], 1)

    def test_cooldown_blocks_fast_reentry(self) -> None:
        self.service = self._build_service({"cooldown_seconds": 300})
        first = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "price": 100.0,
                "order_id": "orb-cooldown-1",
                "event_time": "2026-04-04T14:30:00Z",
            }
        )
        second = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "price": 100.5,
                "order_id": "orb-cooldown-2",
                "event_time": "2026-04-04T14:31:00Z",
            }
        )
        self.assertTrue(first["result"]["success"])
        self.assertFalse(second["result"]["success"])
        self.assertEqual(second["result"]["details"]["blocked_by"], "cooldown_seconds")

    def test_paper_daily_loss_blocks_new_entries(self) -> None:
        self.service = self._build_service({"paper_max_daily_loss": 50.0})
        self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "price": 100.0,
                "order_id": "orb-loss-1",
                "event_time": "2026-04-04T14:30:00Z",
            }
        )
        exit_event = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "exit",
                "side": "buy",
                "price": 94.0,
                "order_id": "orb-loss-exit",
                "event_time": "2026-04-04T15:00:00Z",
            }
        )
        self.assertEqual(exit_event["result"]["details"]["realized_pnl"], -60.0)

        blocked = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "qty": 10,
                "price": 95.0,
                "order_id": "orb-loss-2",
                "event_time": "2026-04-04T15:05:00Z",
            }
        )
        self.assertFalse(blocked["result"]["success"])
        self.assertEqual(blocked["result"]["details"]["blocked_by"], "paper_max_daily_loss")

    def test_risk_pct_sizing_uses_notional_cap(self) -> None:
        self.service = self._build_service(
            {
                "position_sizing_mode": "risk_pct",
                "account_equity": 100000.0,
                "risk_per_trade_pct": 0.0025,
                "max_notional_pct": 0.06,
                "volume_step": 1.0,
                "min_volume": 1.0,
                "max_volume": 25.0,
            }
        )
        entry = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "price": 585.12,
                "stop_loss": 582.40,
                "take_profit": 590.20,
                "order_id": "orb-risk-1",
                "event_time": "2026-04-04T15:30:00Z",
            }
        )
        self.assertTrue(entry["result"]["success"])
        sizing = entry["result"]["details"]["sizing"]
        self.assertEqual(sizing["sizing_mode"], "risk_pct")
        self.assertEqual(sizing["source"], "risk_budget")
        self.assertEqual(sizing["resolved_qty"], 10.0)

    def test_risk_pct_without_stop_is_blocked_when_fallback_disabled(self) -> None:
        self.service = self._build_service(
            {
                "position_sizing_mode": "risk_pct",
                "allow_default_fallback": False,
            }
        )
        blocked = self.service.process_payload(
            {
                "secret": "secret-123",
                "strategy": "ORB",
                "symbol": "QQQ",
                "action": "entry",
                "side": "buy",
                "price": 585.12,
                "order_id": "orb-risk-missing-stop",
                "event_time": "2026-04-04T15:30:00Z",
            }
        )
        self.assertFalse(blocked["result"]["success"])
        self.assertEqual(blocked["result"]["details"]["blocked_by"], "position_sizing")
        self.assertEqual(blocked["result"]["details"]["reason"], "missing_stop_loss")


if __name__ == "__main__":
    unittest.main()
