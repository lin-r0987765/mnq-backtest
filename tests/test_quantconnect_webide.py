from __future__ import annotations

import ast
from pathlib import Path
import unittest

from src.strategies.orb import ORBStrategy


class QuantConnectWebIDEFileTest(unittest.TestCase):
    def test_webide_constants_match_current_orb_defaults(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py")
        module = ast.parse(path.read_text(encoding="utf-8"))

        class_node = next(
            node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "QQQOrbWebIDE"
        )
        assignments: dict[str, object] = {}
        for node in class_node.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                try:
                    assignments[name] = ast.literal_eval(node.value)
                except Exception:
                    continue

        orb = ORBStrategy().params
        expected = {
            "SYMBOL": "QQQ",
            "INITIAL_CASH": 100000,
            "POSITION_SIZE_PCT": 0.25,
            "MIN_TRADE_QUANTITY": 1,
            "MAX_TRADE_QUANTITY": 500,
            "ORB_BARS": orb["orb_bars"],
            "PROFIT_RATIO": orb["profit_ratio"],
            "BREAKOUT_CONFIRM_PCT": orb["breakout_confirm_pct"],
            "ENTRY_DELAY_BARS": orb["entry_delay_bars"],
            # NOTE: QC uses 0.013 (independently optimized), local uses 0.015
            "TRAILING_PCT": 0.013,
            "BREAKEVEN_TRIGGER_MULT": 1.25,
            "BREAKEVEN_ACTIVE_MINUTES": 180,
            "EARLY_TIGHT_TRAIL_PCT": 0.013,
            "EARLY_TIGHT_TRAIL_MINUTES": 0,
            "CLOSE_BEFORE_MIN": orb["close_before_min"],
            "HTF_FILTER": orb["htf_filter"],
            "HTF_MODE": orb["htf_mode"],
            "HTF_EMA_FAST": orb["htf_ema_fast"],
            "HTF_EMA_SLOW": orb["htf_ema_slow"],
            "SKIP_SHORT_AFTER_UP_DAYS": orb["skip_short_after_up_days"],
            "SKIP_LONG_AFTER_UP_DAYS": orb["skip_long_after_up_days"],
            "MULTI_DAY_RANGE": orb["multi_day_range"],
            "MULTI_DAY_LOOKBACK": orb["multi_day_lookback"],
        }
        for key, value in expected.items():
            self.assertEqual(assignments.get(key), value, key)

    def test_webide_has_entry_cutoff_guard(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn("add_equity(", text)
        self.assertIn("DataNormalizationMode.ADJUSTED", text)
        self.assertNotIn("add_future(", text)
        self.assertNotIn("Futures.Indices.MICRO_NASDAQ_100_E_MINI", text)
        self.assertNotIn('getattr(self.future, "mapped", None)', text)
        self.assertNotIn("def on_symbol_changed_events", text)
        self.assertIn("def _entry_window_closed", text)
        self.assertIn("if self._entry_window_closed(bar.end_time):", text)
        self.assertIn("get_next_market_close", text)
        self.assertIn("current_session_market_close", text)
        self.assertIn('SCRIPT_VERSION = "v26-profit-lock"', text)
        self.assertIn('SYMBOL = "QQQ"', text)
        self.assertIn("cutoff=exchange-hours-aware", text)
        self.assertIn("security_type=equity", text)
        self.assertIn("REGIME_FILTER = True", text)
        self.assertIn('REGIME_MODE = "prev_day_up_and_mom3_positive"', text)
        self.assertIn("REGIME_ALLOW_SHORTS = False", text)
        self.assertIn("def _update_regime_state", text)
        self.assertIn("history(self.symbol", text)
        self.assertIn("prev_day_up", text)
        self.assertIn("mom3_positive", text)
        self.assertIn("prev_day_return_pct", text)
        self.assertIn("mom3_return_pct", text)
        self.assertIn("ENTRY_START_HOUR_UTC = 0", text)
        self.assertIn("ENTRY_END_HOUR_UTC = 17", text)
        self.assertIn("BREAKEVEN_TRIGGER_MULT = 1.25", text)
        self.assertIn("BREAKEVEN_ACTIVE_MINUTES = 180", text)
        self.assertIn("PROFIT_LOCK_TRIGGER_MULT = 1.50", text)
        self.assertIn("PROFIT_LOCK_LEVEL_MULT = 0.25", text)
        self.assertIn("EARLY_TIGHT_TRAIL_PCT = 0.013", text)
        self.assertIn("EARLY_TIGHT_TRAIL_MINUTES = 0", text)
        self.assertIn("def _entry_time_filter_passed", text)
        self.assertIn("self.ENTRY_START_HOUR_UTC <= h < self.ENTRY_END_HOUR_UTC", text)
        self.assertIn("def _active_trailing_pct", text)
        self.assertIn("def _breakeven_gate_active", text)
        self.assertIn("def _breakeven_trigger_points", text)
        self.assertIn("def _profit_lock_trigger_points", text)
        self.assertIn("def _profit_lock_points", text)
        self.assertIn("def _reset_position_state", text)
        self.assertIn("self.position_entry_time = bar.end_time", text)
        self.assertIn("self.entry_price = bar.close", text)
        self.assertIn("self.breakeven_activated = False", text)
        self.assertIn("self.profit_lock_activated = False", text)
        self.assertIn("breakeven_trigger_mult", text)
        self.assertIn("breakeven_active_minutes", text)
        self.assertIn("profit_lock_trigger_mult", text)
        self.assertIn("profit_lock_level_mult", text)
        self.assertIn("early_tight_trail_pct", text)
        self.assertIn("early_tight_trail_minutes", text)

    def test_webide_has_dynamic_sizing(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn("def TRADE_QUANTITY(self)", text)
        self.assertIn("@property", text)
        self.assertIn("price = self.securities[self.symbol].price", text)
        self.assertIn("POSITION_SIZE_PCT", text)
        self.assertIn("MIN_TRADE_QUANTITY", text)
        self.assertIn("MAX_TRADE_QUANTITY", text)
        self.assertIn("total_portfolio_value", text)
        self.assertIn("total_shares_traded", text)
        self.assertIn("sizing=dynamic", text)
        self.assertNotIn("contract_multiplier", text)
        self.assertNotIn("contract_notional", text)

    def test_webide_has_session_entry_cap(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/QQQ_ORB_WebIDE.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn("MAX_ENTRIES_PER_SESSION = 1", text)
        self.assertIn("session_entry_count", text)
        self.assertIn("reentry_blocked_count", text)
        self.assertIn("def _can_take_session_entry", text)
        self.assertIn("self.session_entry_count >= self.MAX_ENTRIES_PER_SESSION", text)
        self.assertIn("self.session_entry_count = 0", text)
        self.assertIn("self.session_entry_count += 1", text)
        self.assertIn("max_entries_per_session", text)

    def test_cli_main_has_entry_cutoff_guard(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/main.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn("add_equity(", text)
        self.assertIn("DataNormalizationMode.ADJUSTED", text)
        self.assertNotIn("add_future(", text)
        self.assertNotIn("Futures.Indices.MICRO_NASDAQ_100_E_MINI", text)
        self.assertNotIn('getattr(self.future, "mapped", None)', text)
        self.assertNotIn("def on_symbol_changed_events", text)
        self.assertIn("def _entry_window_closed", text)
        self.assertIn("if self._entry_window_closed(bar.end_time):", text)
        self.assertIn("get_next_market_close", text)
        self.assertIn("current_session_market_close", text)
        self.assertIn('SCRIPT_VERSION = "v26-profit-lock"', text)
        self.assertIn('SUPPORTED_SYMBOL = "QQQ"', text)
        self.assertIn("security_type=equity", text)
        self.assertIn("max_entries_per_session", text)
        self.assertIn("session_entry_count", text)
        self.assertIn("reentry_blocked_count", text)
        self.assertIn("entry_start_hour_utc", text)
        self.assertIn("entry_end_hour_utc", text)
        self.assertIn('self.ticker = (self.get_parameter("symbol") or self.SUPPORTED_SYMBOL).upper()', text)
        self.assertIn('self.regime_mode = self.get_parameter("regime_mode") or "prev_day_up_and_mom3_positive"', text)
        self.assertIn("prev_day_up", text)
        self.assertIn("mom3_positive", text)
        self.assertIn("prev_day_return_pct", text)
        self.assertIn("mom3_return_pct", text)
        self.assertIn("def _entry_time_filter_passed", text)
        self.assertIn("self.entry_start_hour_utc <= h < self.entry_end_hour_utc", text)
        self.assertIn('self.breakeven_trigger_mult = float(self.get_parameter("breakeven_trigger_mult") or 1.25)', text)
        self.assertIn('self.breakeven_active_minutes = int(self.get_parameter("breakeven_active_minutes") or 180)', text)
        self.assertIn('self.profit_lock_trigger_mult = float(self.get_parameter("profit_lock_trigger_mult") or 1.50)', text)
        self.assertIn('self.profit_lock_level_mult = float(self.get_parameter("profit_lock_level_mult") or 0.25)', text)
        self.assertIn("def _breakeven_gate_active", text)
        self.assertIn("def _breakeven_trigger_points", text)
        self.assertIn("def _profit_lock_trigger_points", text)
        self.assertIn("def _profit_lock_points", text)
        self.assertIn("def _reset_position_state", text)

    def test_v25_profit_lock_evaluator_ready(self) -> None:
        path = Path("lean/QQQ_ORB_DeepBacktest/QQQ_V25_ProfitLock_ProxyAnalyzer_WebIDE.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn('SCRIPT_VERSION = "v25-profit-lock-qc-evaluator"', text)
        self.assertIn("PROFIT_LOCK_TRIGGER_MULT = 1.50", text)
        self.assertIn("PROFIT_LOCK_LEVEL_MULT = 0.25", text)
        self.assertIn("baseline_reference=v25-timegated-be", text)
        self.assertIn("profit_lock_activation_count", text)
        self.assertIn("profit_lock_stop_exit_count", text)
        self.assertIn("def _profit_lock_trigger_points", text)
        self.assertIn("def _profit_lock_points", text)
        self.assertIn("ORB Long Profit Lock Stop", text)
        self.assertIn("ORB Short Profit Lock Stop", text)
        self.assertIn("QQQ v25 profit-lock evaluator init", text)


if __name__ == "__main__":
    unittest.main()
