"""
Additional performance metric calculations.

All functions accept an equity curve as a pd.Series and/or a list of
trade dicts and return plain Python floats / dicts.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def annualised_return(equity: pd.Series, periods_per_year: int = 252 * 78) -> float:
    """
    Geometric annualised return.
    `periods_per_year` = 252 trading days × 78 five-minute bars per day.
    """
    n = len(equity)
    if n < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -100.0
    return float((total ** (periods_per_year / n) - 1) * 100)


def sharpe_ratio(equity: pd.Series, periods_per_year: int = 252 * 78) -> float:
    ret = equity.pct_change().dropna()
    if ret.std() == 0 or len(ret) < 2:
        return 0.0
    return float(ret.mean() / ret.std() * math.sqrt(periods_per_year))


def sortino_ratio(equity: pd.Series, periods_per_year: int = 252 * 78) -> float:
    ret = equity.pct_change().dropna()
    downside = ret[ret < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(ret.mean() / downside.std() * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Returns max drawdown as a negative percentage, e.g. -12.5."""
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min() * 100)


def calmar_ratio(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    ann_ret = annualised_return(equity)
    return float(ann_ret / abs(mdd)) if mdd != 0 else 0.0


def win_rate(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return float(wins / len(trades) * 100)


def profit_factor(trades: list[dict[str, Any]]) -> float:
    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) <= 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def avg_trade_return(
    trades: list[dict[str, Any]], initial_cash: float = 100_000.0
) -> float:
    if not trades:
        return 0.0
    avg_pnl = sum(t.get("pnl", 0) for t in trades) / len(trades)
    return float(avg_pnl / initial_cash * 100)


def compute_all_metrics(
    equity: pd.Series,
    trades: list[dict[str, Any]],
    initial_cash: float = 100_000.0,
) -> dict[str, float | int]:
    """Convenience wrapper – returns the full metrics dict."""
    total_return = float((equity.iloc[-1] / equity.iloc[0] - 1) * 100)
    mdd = max_drawdown(equity)

    return {
        "total_return_pct": round(total_return, 4),
        "annualised_return_pct": round(annualised_return(equity), 4),
        "sharpe_ratio": round(sharpe_ratio(equity), 4),
        "sortino_ratio": round(sortino_ratio(equity), 4),
        "calmar_ratio": round(calmar_ratio(equity), 4),
        "max_drawdown_pct": round(mdd, 4),
        "win_rate_pct": round(win_rate(trades), 4),
        "total_trades": len(trades),
        "profit_factor": round(profit_factor(trades), 4),
        "avg_trade_pct": round(avg_trade_return(trades, initial_cash), 4),
    }
