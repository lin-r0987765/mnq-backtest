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


# ── Kelly Criterion 動態倉位管理 ─────────────────────────────────────────

DEFAULT_KELLY_MULT = 0.5       # half-Kelly (conservative)
DEFAULT_KELLY_LOOKBACK = 10    # recent trades to compute Kelly from
DEFAULT_KELLY_MIN_SIZE = 0.3   # minimum size multiplier
DEFAULT_KELLY_MAX_SIZE = 1.5   # maximum size multiplier


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Calculate Kelly Criterion fraction.
    f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = avg_win / avg_loss
    """
    if avg_loss <= 0 or win_rate <= 0 or avg_win <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss
    f = (p * b - q) / b
    return max(0.0, min(f, 1.0))


def _apply_kelly_to_equity(
    equity_curve: list[float] | np.ndarray,
    trades: list[dict],
    lookback_trades: int = DEFAULT_KELLY_LOOKBACK,
    kelly_mult: float = DEFAULT_KELLY_MULT,
    min_size_mult: float = DEFAULT_KELLY_MIN_SIZE,
    max_size_mult: float = DEFAULT_KELLY_MAX_SIZE,
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> np.ndarray:
    """
    Apply Kelly criterion sizing to an equity curve.
    Uses rolling trade history to compute Kelly fraction,
    then scales the return contribution of each trade period.
    """
    eq = np.asarray(equity_curve, dtype=float)
    n = len(eq)
    if n < 2 or not trades:
        return eq.copy()

    trade_pnls = []
    for t in trades:
        pnl = 0.0
        if "pnl" in t:
            pnl = float(t.get("pnl", 0) or 0)
        elif "PnL" in t:
            pnl = float(t.get("PnL", 0) or 0)
        elif "Return" in t:
            pnl = float(t.get("Return", 0) or 0)
        trade_pnls.append(pnl)

    if not trade_pnls:
        return eq.copy()

    # Identify trade regions and compute Kelly sizing
    completed_trades = 0
    kelly_sizes: list[tuple[int, float]] = []
    current_trade_start = None

    for i in range(1, n):
        is_active = abs(eq[i] - eq[i - 1]) > 0.01

        if is_active and current_trade_start is None:
            current_trade_start = i
            if completed_trades >= 3:
                recent = trade_pnls[max(0, completed_trades - lookback_trades):completed_trades]
                wins = [p for p in recent if p > 0]
                losses = [abs(p) for p in recent if p <= 0]
                if wins and losses:
                    wr = len(wins) / len(recent)
                    avg_w = float(np.mean(wins))
                    avg_l = float(np.mean(losses))
                    kf = kelly_fraction(wr, avg_w, avg_l)
                    size_mult = max(min_size_mult, min(max_size_mult, kf * kelly_mult / 0.5))
                else:
                    size_mult = 1.0
            else:
                size_mult = 1.0
            kelly_sizes.append((i, size_mult))

        elif not is_active and current_trade_start is not None:
            current_trade_start = None
            completed_trades += 1

    if not kelly_sizes:
        return eq.copy()

    # Build per-bar size multiplier
    size_mults = np.ones(n)
    current_mult = 1.0
    size_idx = 0
    for i in range(n):
        while size_idx < len(kelly_sizes) and kelly_sizes[size_idx][0] <= i:
            current_mult = kelly_sizes[size_idx][1]
            size_idx += 1
        size_mults[i] = current_mult

    # Re-compute equity with scaled returns
    kelly_eq = np.full(n, initial_cash)
    for i in range(1, n):
        base_return = (eq[i] - eq[i - 1]) / eq[i - 1] if eq[i - 1] > 0 else 0
        kelly_eq[i] = kelly_eq[i - 1] * (1 + base_return * size_mults[i])

    return kelly_eq


def combine_results_active_reuse_kelly(
    result1: BacktestResult,
    result2: BacktestResult,
    active_weight: float = DEFAULT_ACTIVE_REUSE_WEIGHT,
    both_weights: tuple[float, float] = (0.5, 0.5),
    kelly_mult: float = DEFAULT_KELLY_MULT,
    lookback_trades: int = DEFAULT_KELLY_LOOKBACK,
    min_size_mult: float = DEFAULT_KELLY_MIN_SIZE,
    max_size_mult: float = DEFAULT_KELLY_MAX_SIZE,
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> np.ndarray:
    """Active Reuse + Kelly Criterion combined portfolio."""
    eq1_kelly = _apply_kelly_to_equity(
        result1.equity_curve, result1.trades,
        lookback_trades=lookback_trades, kelly_mult=kelly_mult,
        min_size_mult=min_size_mult, max_size_mult=max_size_mult,
        initial_cash=initial_cash,
    )
    eq2_kelly = _apply_kelly_to_equity(
        result2.equity_curve, result2.trades,
        lookback_trades=lookback_trades, kelly_mult=kelly_mult,
        min_size_mult=min_size_mult, max_size_mult=max_size_mult,
        initial_cash=initial_cash,
    )

    mask1 = position_mask_from_result(result1)
    mask2 = position_mask_from_result(result2)

    n = min(len(eq1_kelly), len(eq2_kelly), len(mask1), len(mask2))

    ret1 = eq1_kelly[:n] / eq1_kelly[0]
    ret2 = eq2_kelly[:n] / eq2_kelly[0]
    dret1 = np.diff(ret1, prepend=ret1[0])
    dret2 = np.diff(ret2, prepend=ret2[0])
    combined_ret = np.ones(n)

    for i in range(1, n):
        if mask1[i] and mask2[i]:
            w1, w2 = both_weights
        elif mask1[i] and not mask2[i]:
            w1, w2 = active_weight, 0.0
        elif mask2[i] and not mask1[i]:
            w1, w2 = 0.0, active_weight
        else:
            w1, w2 = both_weights
        combined_ret[i] = combined_ret[i - 1] + w1 * dret1[i] + w2 * dret2[i]

    return combined_ret * initial_cash
