"""
Base strategy interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyResult:
    """Container for signals produced by a strategy."""

    entries_long: pd.Series   # Boolean series: True → enter long
    exits_long: pd.Series     # Boolean series: True → exit long
    entries_short: pd.Series  # Boolean series: True → enter short
    exits_short: pd.Series    # Boolean series: True → exit short
    sl_stop: pd.Series | None = None   # Stop-loss price per bar (optional)
    tp_stop: pd.Series | None = None   # Take-profit price per bar (optional)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all strategies."""

    name: str = "BaseStrategy"

    def __init__(self, params: dict[str, Any] | None = None):
        self.params: dict[str, Any] = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategyResult:
        """
        Given an OHLCV DataFrame (single session or multi-session),
        return entry/exit Boolean Series aligned to df.index.
        """

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"
