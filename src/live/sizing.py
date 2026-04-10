from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.live.config import LiveTradingConfig
from src.live.models import LiveSignal


@dataclass
class SizingResult:
    success: bool
    qty: float | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class PositionSizer:
    def __init__(self, config: LiveTradingConfig):
        self.config = config

    def _round_down(self, value: float) -> float:
        step = self.config.volume_step
        if step <= 0:
            step = 1.0
        steps = math.floor((value + 1e-12) / step)
        rounded = steps * step
        return round(rounded, 8)

    def _fixed_qty(self, signal: LiveSignal) -> SizingResult:
        source = "config_default"
        qty = self.config.default_volume
        if self.config.allow_signal_qty_override and signal.qty is not None:
            source = "signal_override"
            qty = signal.qty
        qty = self._round_down(float(qty))
        if qty < self.config.min_volume:
            return SizingResult(
                False,
                message="Fixed quantity below minimum volume.",
                details={
                    "blocked_by": "position_sizing",
                    "sizing_mode": "fixed",
                    "requested_qty": signal.qty,
                    "resolved_qty": qty,
                    "min_volume": self.config.min_volume,
                },
            )
        return SizingResult(
            True,
            qty=qty,
            details={
                "sizing_mode": "fixed",
                "source": source,
                "requested_qty": signal.qty,
                "resolved_qty": qty,
            },
        )

    def _risk_qty(self, signal: LiveSignal) -> SizingResult:
        if self.config.allow_signal_qty_override and signal.qty is not None:
            qty = self._round_down(float(signal.qty))
            if qty < self.config.min_volume:
                return SizingResult(
                    False,
                    message="Signal override quantity below minimum volume.",
                    details={
                        "blocked_by": "position_sizing",
                        "sizing_mode": "risk_pct",
                        "source": "signal_override",
                        "requested_qty": signal.qty,
                        "resolved_qty": qty,
                        "min_volume": self.config.min_volume,
                    },
                )
            return SizingResult(
                True,
                qty=qty,
                details={
                    "sizing_mode": "risk_pct",
                    "source": "signal_override",
                    "requested_qty": signal.qty,
                    "resolved_qty": qty,
                },
            )

        if signal.price is None or signal.price <= 0:
            if self.config.allow_default_fallback:
                return self._fixed_qty(signal)
            return SizingResult(
                False,
                message="Risk sizing requires entry price.",
                details={
                    "blocked_by": "position_sizing",
                    "sizing_mode": "risk_pct",
                    "source": "risk_budget",
                    "reason": "missing_entry_price",
                },
            )

        if signal.stop_loss is None:
            if self.config.allow_default_fallback:
                return self._fixed_qty(signal)
            return SizingResult(
                False,
                message="Risk sizing requires stop loss.",
                details={
                    "blocked_by": "position_sizing",
                    "sizing_mode": "risk_pct",
                    "source": "risk_budget",
                    "reason": "missing_stop_loss",
                },
            )

        stop_distance = abs(float(signal.price) - float(signal.stop_loss))
        if stop_distance <= 0:
            if self.config.allow_default_fallback:
                return self._fixed_qty(signal)
            return SizingResult(
                False,
                message="Risk sizing requires positive stop distance.",
                details={
                    "blocked_by": "position_sizing",
                    "sizing_mode": "risk_pct",
                    "source": "risk_budget",
                    "reason": "invalid_stop_distance",
                },
            )

        contract_multiplier = max(self.config.contract_multiplier, 1e-12)
        risk_budget = self.config.account_equity * self.config.risk_per_trade_pct
        risk_per_unit = stop_distance * contract_multiplier
        risk_qty = risk_budget / risk_per_unit if risk_per_unit > 0 else float("inf")
        notional_per_unit = float(signal.price) * contract_multiplier
        qty_candidates = [risk_qty]

        notional_cap_qty = None
        if self.config.max_notional_pct > 0 and notional_per_unit > 0:
            notional_cap_qty = (self.config.account_equity * self.config.max_notional_pct) / notional_per_unit
            qty_candidates.append(notional_cap_qty)

        if self.config.max_volume > 0:
            qty_candidates.append(self.config.max_volume)

        raw_qty = min(candidate for candidate in qty_candidates if candidate > 0)
        qty = self._round_down(raw_qty)

        if qty < self.config.min_volume:
            return SizingResult(
                False,
                message="Sized quantity below minimum volume.",
                details={
                    "blocked_by": "position_sizing",
                    "sizing_mode": "risk_pct",
                    "source": "risk_budget",
                    "resolved_qty": qty,
                    "min_volume": self.config.min_volume,
                    "risk_budget": round(risk_budget, 6),
                    "risk_qty": round(risk_qty, 6),
                    "notional_cap_qty": round(notional_cap_qty, 6) if notional_cap_qty is not None else None,
                    "stop_distance": round(stop_distance, 6),
                },
            )

        return SizingResult(
            True,
            qty=qty,
            details={
                "sizing_mode": "risk_pct",
                "source": "risk_budget",
                "requested_qty": signal.qty,
                "resolved_qty": qty,
                "entry_price": float(signal.price),
                "stop_loss": float(signal.stop_loss),
                "stop_distance": round(stop_distance, 6),
                "risk_budget": round(risk_budget, 6),
                "risk_per_unit": round(risk_per_unit, 6),
                "risk_qty": round(risk_qty, 6),
                "notional_cap_qty": round(notional_cap_qty, 6) if notional_cap_qty is not None else None,
                "contract_multiplier": self.config.contract_multiplier,
                "account_equity": self.config.account_equity,
                "risk_per_trade_pct": self.config.risk_per_trade_pct,
                "max_notional_pct": self.config.max_notional_pct,
            },
        )

    def resolve(self, signal: LiveSignal) -> SizingResult:
        if signal.action != "entry":
            return SizingResult(True, qty=signal.qty)

        if self.config.position_sizing_mode == "risk_pct":
            return self._risk_qty(signal)
        return self._fixed_qty(signal)
