"""Broker capability & symbol precision validation for MT5.

Queries MT5 symbol_info to validate:
- Symbol is tradeable (trade_mode)
- Volume constraints (min, max, step)
- Price precision (digits)
- Trade execution limits

This module prevents order failures from config/broker mismatches.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# MT5 trade mode constants (from MetaTrader5 docs)
TRADE_MODE_DISABLED = 0
TRADE_MODE_LONGONLY = 1
TRADE_MODE_SHORTONLY = 2
TRADE_MODE_CLOSEONLY = 3
TRADE_MODE_FULL = 4


@dataclass
class SymbolCapability:
    """Broker-reported constraints for a specific symbol."""

    symbol: str
    trade_allowed: bool = True
    trade_mode: int = TRADE_MODE_FULL
    trade_mode_desc: str = "FULL"
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    digits: int = 5
    point: float = 0.00001
    spread: int = 0
    trade_stops_level: int = 0
    trade_freeze_level: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def can_buy(self) -> bool:
        return self.trade_allowed and self.trade_mode in (
            TRADE_MODE_FULL,
            TRADE_MODE_LONGONLY,
        )

    @property
    def can_sell(self) -> bool:
        return self.trade_allowed and self.trade_mode in (
            TRADE_MODE_FULL,
            TRADE_MODE_SHORTONLY,
        )

    @property
    def can_trade(self) -> bool:
        return self.trade_allowed and self.trade_mode != TRADE_MODE_DISABLED


@dataclass
class VolumeValidation:
    """Result of volume validation against broker constraints."""

    valid: bool
    original_qty: float
    adjusted_qty: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def _trade_mode_description(mode: int) -> str:
    mapping = {
        TRADE_MODE_DISABLED: "DISABLED",
        TRADE_MODE_LONGONLY: "LONG_ONLY",
        TRADE_MODE_SHORTONLY: "SHORT_ONLY",
        TRADE_MODE_CLOSEONLY: "CLOSE_ONLY",
        TRADE_MODE_FULL: "FULL",
    }
    return mapping.get(mode, f"UNKNOWN({mode})")


def query_symbol_capability(mt5: Any, symbol: str) -> SymbolCapability:
    """Query MT5 for a symbol's trading constraints.

    Parameters
    ----------
    mt5 : MetaTrader5 module
    symbol : Broker symbol name (after symbol_map resolution)

    Returns
    -------
    SymbolCapability with broker-reported values.

    Raises
    ------
    RuntimeError if symbol_info fails.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"MT5 symbol_info returned None for: {symbol}")

    # Ensure symbol is visible in Market Watch
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"MT5 symbol_select failed for: {symbol}")
        # Re-query after selection
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5 symbol_info returned None after select: {symbol}")

    trade_mode = getattr(info, "trade_mode", TRADE_MODE_FULL)
    trade_allowed = bool(getattr(info, "trade_allowed", True))

    cap = SymbolCapability(
        symbol=symbol,
        trade_allowed=trade_allowed,
        trade_mode=trade_mode,
        trade_mode_desc=_trade_mode_description(trade_mode),
        volume_min=float(getattr(info, "volume_min", 0.01)),
        volume_max=float(getattr(info, "volume_max", 100.0)),
        volume_step=float(getattr(info, "volume_step", 0.01)),
        digits=int(getattr(info, "digits", 5)),
        point=float(getattr(info, "point", 0.00001)),
        spread=int(getattr(info, "spread", 0)),
        trade_stops_level=int(getattr(info, "trade_stops_level", 0)),
        trade_freeze_level=int(getattr(info, "trade_freeze_level", 0)),
        raw={
            "name": getattr(info, "name", symbol),
            "description": getattr(info, "description", ""),
            "path": getattr(info, "path", ""),
            "currency_base": getattr(info, "currency_base", ""),
            "currency_profit": getattr(info, "currency_profit", ""),
            "trade_calc_mode": getattr(info, "trade_calc_mode", None),
        },
    )

    logger.info(
        "Broker capability for %s: trade_mode=%s, vol=[%.4f, %.4f, step=%.4f], digits=%d",
        symbol,
        cap.trade_mode_desc,
        cap.volume_min,
        cap.volume_max,
        cap.volume_step,
        cap.digits,
    )
    return cap


def validate_volume(
    qty: float,
    capability: SymbolCapability,
    *,
    auto_adjust: bool = True,
) -> VolumeValidation:
    """Validate and optionally adjust volume to match broker constraints.

    Parameters
    ----------
    qty : Desired order volume.
    capability : Broker-reported symbol capability.
    auto_adjust : If True, round to valid step and clamp to [min, max].

    Returns
    -------
    VolumeValidation with validity flag and adjusted qty.
    """
    step = capability.volume_step
    if step <= 0:
        step = 0.01

    details = {
        "broker_volume_min": capability.volume_min,
        "broker_volume_max": capability.volume_max,
        "broker_volume_step": step,
        "original_qty": qty,
    }

    # Round to nearest valid step
    steps_count = math.floor((qty + 1e-12) / step)
    adjusted = round(steps_count * step, 8)

    if adjusted < capability.volume_min:
        if not auto_adjust:
            return VolumeValidation(
                valid=False,
                original_qty=qty,
                adjusted_qty=adjusted,
                message=f"Volume {qty} below broker min {capability.volume_min}",
                details=details,
            )
        # Try rounding up to min
        adjusted = capability.volume_min

    if adjusted > capability.volume_max > 0:
        if not auto_adjust:
            return VolumeValidation(
                valid=False,
                original_qty=qty,
                adjusted_qty=adjusted,
                message=f"Volume {qty} exceeds broker max {capability.volume_max}",
                details=details,
            )
        adjusted = capability.volume_max

    # Final step alignment
    steps_count = math.floor((adjusted + 1e-12) / step)
    adjusted = round(steps_count * step, 8)

    details["adjusted_qty"] = adjusted
    details["was_adjusted"] = abs(adjusted - qty) > 1e-8

    if adjusted < capability.volume_min:
        return VolumeValidation(
            valid=False,
            original_qty=qty,
            adjusted_qty=adjusted,
            message=f"Volume cannot meet broker min {capability.volume_min} after step alignment",
            details=details,
        )

    return VolumeValidation(
        valid=True,
        original_qty=qty,
        adjusted_qty=adjusted,
        message="OK" if not details["was_adjusted"] else f"Adjusted from {qty} to {adjusted}",
        details=details,
    )


def check_trade_direction(capability: SymbolCapability, side: str) -> tuple[bool, str]:
    """Check if the broker allows the requested trade direction.

    Returns
    -------
    (allowed, reason)
    """
    if not capability.can_trade:
        return False, f"Symbol {capability.symbol} trading is disabled (mode={capability.trade_mode_desc})"
    if side == "buy" and not capability.can_buy:
        return False, f"Symbol {capability.symbol} does not allow buy (mode={capability.trade_mode_desc})"
    if side == "sell" and not capability.can_sell:
        return False, f"Symbol {capability.symbol} does not allow sell (mode={capability.trade_mode_desc})"
    return True, "OK"
