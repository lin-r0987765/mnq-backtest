"""
Backtest engine wrapper around vectorbt (with pandas fallback).

Runs a strategy's signals through a vectorbt Portfolio simulation
and returns a rich BacktestResult containing both raw portfolio stats
and a normalised metrics dict ready for JSON serialisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy, StrategyResult

# ── vectorbt import (optional – fall back to manual simulation) ──────────────
try:
    import vectorbt as vbt
    _HAS_VBT = True
except ImportError:  # pragma: no cover
    _HAS_VBT = False


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    equity_curve: list[float]
    trades: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)  # vbt Portfolio or None


# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Wrap a BaseStrategy and run it on an OHLCV DataFrame.

    Parameters
    ----------
    initial_cash : float
        Starting account equity in USD (default 100_000).
    fees_pct : float
        Round-trip commission as fraction of notional (default 0.0005 = 0.05%).
    size : float
        Number of contracts / shares per trade (default 1).
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        fees_pct: float = 0.0005,
        size: float = 1.0,
    ):
        self.initial_cash = initial_cash
        self.fees_pct = fees_pct
        self.size = size

    # ------------------------------------------------------------------
    def run(self, strategy: BaseStrategy, df: pd.DataFrame) -> BacktestResult:
        """
        Run the strategy on *df* and return a BacktestResult.
        """
        signals: StrategyResult = strategy.generate_signals(df)

        if _HAS_VBT:
            return self._run_vbt(strategy, df, signals)
        else:
            return self._run_manual(strategy, df, signals)

    # ------------------------------------------------------------------
    # vectorbt path
    # ------------------------------------------------------------------
    def _run_vbt(
        self, strategy: BaseStrategy, df: pd.DataFrame, signals: StrategyResult
    ) -> BacktestResult:
        """Use vectorbt for simulation."""
        close = df["Close"]

        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=signals.entries_long,
                exits=signals.exits_long,
                short_entries=signals.entries_short,
                short_exits=signals.exits_short,
                size=self.size,
                fees=self.fees_pct,
                init_cash=self.initial_cash,
                freq="5min",
                sl_stop=signals.sl_stop if signals.sl_stop is not None else None,
                tp_stop=signals.tp_stop if signals.tp_stop is not None else None,
            )

            equity = pf.value()
            trades_df = pf.trades.records_readable

            metrics = self._extract_vbt_metrics(pf, equity)
            trades_list = self._trades_to_list(trades_df)

            return BacktestResult(
                strategy_name=strategy.name,
                params=strategy.get_params(),
                metrics=metrics,
                equity_curve=equity.tolist(),
                trades=trades_list,
                raw=pf,
            )
        except Exception as e:
            # Fallback on vectorbt errors
            return self._run_manual(strategy, df, signals)

    def _extract_vbt_metrics(self, pf: Any, equity: pd.Series) -> dict[str, float]:
        """Extract standardised metrics from a vbt Portfolio."""
        try:
            total_return = float(pf.total_return() * 100)
        except Exception:
            total_return = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)

        try:
            sharpe = float(pf.sharpe_ratio())
        except Exception:
            sharpe = _sharpe(equity)

        try:
            sortino = float(pf.sortino_ratio())
        except Exception:
            sortino = _sortino(equity)

        try:
            max_dd = float(pf.max_drawdown() * 100)
        except Exception:
            max_dd = _max_drawdown(equity)

        try:
            trades_rec = pf.trades.records_readable
            win_rate = float(
                (trades_rec["Return"] > 0).mean() * 100
            ) if len(trades_rec) > 0 else 0.0
            total_trades = len(trades_rec)
        except Exception:
            win_rate = 0.0
            total_trades = 0

        calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

        try:
            pf_factor = _profit_factor_from_vbt(pf)
        except Exception:
            pf_factor = 0.0

        avg_trade = total_return / total_trades if total_trades > 0 else 0.0

        return {
            "total_return_pct": round(total_return, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "win_rate_pct": round(win_rate, 4),
            "total_trades": total_trades,
            "profit_factor": round(pf_factor, 4),
            "avg_trade_pct": round(avg_trade, 4),
        }

    def _trades_to_list(self, trades_df: pd.DataFrame) -> list[dict[str, Any]]:
        if trades_df is None or len(trades_df) == 0:
            return []
        records = []
        for _, row in trades_df.iterrows():
            record: dict[str, Any] = {}
            for col in trades_df.columns:
                val = row[col]
                if isinstance(val, (pd.Timestamp,)):
                    record[col] = str(val)
                elif isinstance(val, float) and np.isnan(val):
                    record[col] = None
                else:
                    try:
                        record[col] = float(val)
                    except (TypeError, ValueError):
                        record[col] = str(val)
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Manual (pure-pandas) fallback path
    # ------------------------------------------------------------------
    def _run_manual(
        self, strategy: BaseStrategy, df: pd.DataFrame, signals: StrategyResult
    ) -> BacktestResult:
        """Simple bar-by-bar simulation – used when vectorbt is unavailable."""
        close = df["Close"].values
        n = len(close)

        cash = self.initial_cash
        position = 0.0   # +size = long, −size = short
        entry_price = 0.0
        equity = np.full(n, self.initial_cash)

        el = signals.entries_long.values.astype(bool)
        xl = signals.exits_long.values.astype(bool)
        es = signals.entries_short.values.astype(bool)
        xs = signals.exits_short.values.astype(bool)
        sl = signals.sl_stop.values if signals.sl_stop is not None else np.full(n, np.nan)
        tp = signals.tp_stop.values if signals.tp_stop is not None else np.full(n, np.nan)

        trades: list[dict[str, Any]] = []

        for i in range(n):
            price = close[i]
            fee = price * self.size * self.fees_pct

            # Check stops first
            if position > 0:
                triggered = False
                if not np.isnan(sl[i]) and price <= sl[i]:
                    triggered = True
                elif not np.isnan(tp[i]) and price >= tp[i]:
                    triggered = True
                elif xl[i]:
                    triggered = True
                if triggered:
                    pnl = (price - entry_price) * self.size - fee
                    cash += pnl
                    trades.append({"side": "long", "entry": entry_price, "exit": price, "pnl": pnl})
                    position = 0.0

            elif position < 0:
                triggered = False
                if not np.isnan(sl[i]) and price >= sl[i]:
                    triggered = True
                elif not np.isnan(tp[i]) and price <= tp[i]:
                    triggered = True
                elif xs[i]:
                    triggered = True
                if triggered:
                    pnl = (entry_price - price) * self.size - fee
                    cash += pnl
                    trades.append({"side": "short", "entry": entry_price, "exit": price, "pnl": pnl})
                    position = 0.0

            # New entries
            if position == 0:
                if el[i]:
                    entry_price = price
                    position = self.size
                    cash -= fee
                elif es[i]:
                    entry_price = price
                    position = -self.size
                    cash -= fee

            # Mark-to-market equity
            if position > 0:
                equity[i] = cash + position * (price - entry_price)
            elif position < 0:
                equity[i] = cash + abs(position) * (entry_price - price)
            else:
                equity[i] = cash

        equity_series = pd.Series(equity, index=df.index)
        metrics = _compute_metrics(equity_series, trades)

        return BacktestResult(
            strategy_name=strategy.name,
            params=strategy.get_params(),
            metrics=metrics,
            equity_curve=equity_series.tolist(),
            trades=trades,
            raw=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(equity: pd.Series, periods_per_year: int = 252 * 78) -> float:
    ret = equity.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * (periods_per_year ** 0.5))


def _sortino(equity: pd.Series, periods_per_year: int = 252 * 78) -> float:
    ret = equity.pct_change().dropna()
    downside = ret[ret < 0].std()
    if downside == 0:
        return 0.0
    return float(ret.mean() / downside * (periods_per_year ** 0.5))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100)


def _profit_factor_from_vbt(pf: Any) -> float:
    try:
        trades = pf.trades.records_readable
        wins = trades.loc[trades["Return"] > 0, "Return"].sum()
        losses = abs(trades.loc[trades["Return"] <= 0, "Return"].sum())
        return float(wins / losses) if losses > 0 else float("inf")
    except Exception:
        return 0.0


def _compute_metrics(
    equity: pd.Series, trades: list[dict[str, Any]]
) -> dict[str, float]:
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    sharpe = _sharpe(equity)
    sortino = _sortino(equity)
    max_dd = _max_drawdown(equity)
    calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

    total_trades = len(trades)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0.0

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_trade = total_return / total_trades if total_trades > 0 else 0.0

    return {
        "total_return_pct": round(float(total_return), 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
        "total_trades": total_trades,
        "profit_factor": round(profit_factor, 4),
        "avg_trade_pct": round(avg_trade, 4),
    }
