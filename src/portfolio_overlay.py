from __future__ import annotations

from typing import Any

import numpy as np

from src.backtest.engine import BacktestResult

DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_ACTIVE_REUSE_WEIGHT = 0.8


def combine_equity_static(
    eq1: list[float] | np.ndarray,
    eq2: list[float] | np.ndarray,
    w1: float = 0.5,
    w2: float = 0.5,
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> np.ndarray:
    """Combine two equity curves with fixed weights."""
    arr1 = np.asarray(eq1, dtype=float)
    arr2 = np.asarray(eq2, dtype=float)
    n = min(len(arr1), len(arr2))
    if n < 2:
        return arr1[:n] if len(arr1) > 0 else arr2[:n]

    ret1 = arr1[:n] / arr1[0]
    ret2 = arr2[:n] / arr2[0]
    return (w1 * ret1 + w2 * ret2) * initial_cash


def position_mask_from_result(result: BacktestResult) -> np.ndarray:
    """Extract a per-bar active-position mask from a backtest result."""
    if result.raw is not None:
        try:
            mask = result.raw.position_mask()
            if hasattr(mask, "to_numpy"):
                return mask.to_numpy().astype(bool).reshape(-1)
            return np.asarray(mask).astype(bool).reshape(-1)
        except Exception:
            pass

    eq = np.asarray(result.equity_curve, dtype=float)
    if len(eq) < 2:
        return np.zeros(len(eq), dtype=bool)
    return (np.abs(np.diff(eq, prepend=eq[0])) > 1e-12).astype(bool)


def combine_equity_active_reuse(
    eq1: list[float] | np.ndarray,
    eq2: list[float] | np.ndarray,
    mask1: list[bool] | np.ndarray,
    mask2: list[bool] | np.ndarray,
    active_weight: float = DEFAULT_ACTIVE_REUSE_WEIGHT,
    both_weights: tuple[float, float] = (0.5, 0.5),
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> np.ndarray:
    """Reuse idle allocation when only one strategy is holding a position."""
    arr1 = np.asarray(eq1, dtype=float)
    arr2 = np.asarray(eq2, dtype=float)
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)
    n = min(len(arr1), len(arr2), len(m1), len(m2))
    if n < 2:
        return arr1[:n] if len(arr1) > 0 else arr2[:n]

    ret1 = arr1[:n] / arr1[0]
    ret2 = arr2[:n] / arr2[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])
    combined_ret = np.ones(n)

    for i in range(1, n):
        if m1[i] and m2[i]:
            w1, w2 = both_weights
        elif m1[i] and not m2[i]:
            w1, w2 = active_weight, 0.0
        elif m2[i] and not m1[i]:
            w1, w2 = 0.0, active_weight
        else:
            w1, w2 = both_weights
        combined_ret[i] = combined_ret[i - 1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * initial_cash


def combine_results_active_reuse(
    result1: BacktestResult,
    result2: BacktestResult,
    active_weight: float = DEFAULT_ACTIVE_REUSE_WEIGHT,
    both_weights: tuple[float, float] = (0.5, 0.5),
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> np.ndarray:
    return combine_equity_active_reuse(
        result1.equity_curve,
        result2.equity_curve,
        position_mask_from_result(result1),
        position_mask_from_result(result2),
        active_weight=active_weight,
        both_weights=both_weights,
        initial_cash=initial_cash,
    )
