from __future__ import annotations

import unittest

import pandas as pd

from src.backtest.engine import BacktestEngine
from src.strategies.base import BaseStrategy, StrategyResult


class _SingleLongTradeStrategy(BaseStrategy):
    name = "SingleLongTrade"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        entries_long = pd.Series([True, False, False], index=df.index)
        exits_long = pd.Series([False, False, True], index=df.index)
        entries_short = pd.Series([False, False, False], index=df.index)
        exits_short = pd.Series([False, False, False], index=df.index)
        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            metadata={},
        )


class _OpenLongAtEndStrategy(BaseStrategy):
    name = "OpenLongAtEnd"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        entries_long = pd.Series([False, False, True], index=df.index)
        exits_long = pd.Series([False, False, False], index=df.index)
        entries_short = pd.Series([False, False, False], index=df.index)
        exits_short = pd.Series([False, False, False], index=df.index)
        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            metadata={},
        )


class _OpenShortAtEndStrategy(BaseStrategy):
    name = "OpenShortAtEnd"

    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        entries_long = pd.Series([False, False, False], index=df.index)
        exits_long = pd.Series([False, False, False], index=df.index)
        entries_short = pd.Series([False, False, True], index=df.index)
        exits_short = pd.Series([False, False, False], index=df.index)
        return StrategyResult(
            entries_long=entries_long,
            exits_long=exits_long,
            entries_short=entries_short,
            exits_short=exits_short,
            metadata={},
        )


class BacktestEngineSizingTest(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range("2026-01-01 14:30:00+00:00", periods=3, freq="5min")
        self.df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 105.0],
                "High": [101.0, 105.0, 106.0],
                "Low": [99.0, 100.0, 104.0],
                "Close": [100.0, 101.0, 105.0],
                "Volume": [1000, 1000, 1000],
            },
            index=index,
        )

    def test_fixed_shares_mode_records_requested_share_count(self) -> None:
        engine = BacktestEngine(initial_cash=100_000.0, fees_pct=0.0, position_size_mode="fixed", fixed_shares=40)
        result = engine.run(_SingleLongTradeStrategy(), self.df)
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(int(result.trades[0]["shares"]), 40)

    def test_capital_pct_mode_enforces_minimum_shares(self) -> None:
        engine = BacktestEngine(
            initial_cash=100_000.0,
            fees_pct=0.0,
            position_size_mode="capital_pct",
            capital_usage_pct=0.1,
            min_shares=40,
        )
        result = engine.run(_SingleLongTradeStrategy(), self.df)
        self.assertEqual(len(result.trades), 1)
        self.assertGreaterEqual(int(result.trades[0]["shares"]), 40)

    def test_capital_pct_mode_skips_trade_when_minimum_is_unaffordable(self) -> None:
        engine = BacktestEngine(
            initial_cash=1_000.0,
            fees_pct=0.0,
            position_size_mode="capital_pct",
            capital_usage_pct=0.1,
            min_shares=40,
        )
        result = engine.run(_SingleLongTradeStrategy(), self.df)
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.metrics["total_trades"], 0)

    def test_manual_engine_exit_fee_uses_actual_share_count_and_records_timestamps(self) -> None:
        engine = BacktestEngine(
            initial_cash=100_000.0,
            fees_pct=0.01,
            position_size_mode="fixed",
            fixed_shares=40,
        )
        result = engine.run(_SingleLongTradeStrategy(), self.df)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(float(trade["entry_fee"]), 40.0)
        self.assertEqual(float(trade["exit_fee"]), 42.0)
        self.assertEqual(float(trade["total_fee"]), 82.0)
        self.assertEqual(float(trade["pnl"]), 158.0)
        self.assertEqual(str(trade["entry_time"]), "2026-01-01 14:30:00+00:00")
        self.assertEqual(str(trade["exit_time"]), "2026-01-01 14:40:00+00:00")
        self.assertEqual(int(trade["bars_held"]), 2)
        self.assertIsNone(trade["entry_stop"])
        self.assertIsNone(trade["entry_target"])

    def test_manual_engine_liquidates_open_long_on_final_bar(self) -> None:
        engine = BacktestEngine(
            initial_cash=10_000.0,
            fees_pct=0.0,
            position_size_mode="fixed",
            fixed_shares=10,
        )
        result = engine.run(_OpenLongAtEndStrategy(), self.df)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(str(trade["side"]), "long")
        self.assertEqual(int(trade["exit_index"]), 2)
        self.assertEqual(float(trade["pnl"]), 0.0)
        self.assertEqual(float(result.metrics["total_return_pct"]), 0.0)

    def test_manual_engine_liquidates_open_short_on_final_bar(self) -> None:
        engine = BacktestEngine(
            initial_cash=10_000.0,
            fees_pct=0.0,
            position_size_mode="fixed",
            fixed_shares=10,
        )
        result = engine.run(_OpenShortAtEndStrategy(), self.df)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(str(trade["side"]), "short")
        self.assertEqual(int(trade["exit_index"]), 2)
        self.assertEqual(float(trade["pnl"]), 0.0)
        self.assertEqual(float(result.metrics["total_return_pct"]), 0.0)


if __name__ == "__main__":
    unittest.main()
