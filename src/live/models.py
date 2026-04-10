from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


VALID_ACTIONS = {"entry", "exit", "close_all", "heartbeat"}
VALID_SIDES = {"buy", "sell", "flat"}


@dataclass
class LiveSignal:
    secret: str
    strategy: str
    symbol: str
    action: str
    side: str = "flat"
    timeframe: str = ""
    qty: float | None = None
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_id: str = ""
    comment: str = ""
    event_time: str = ""
    broker_symbol: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LiveSignal":
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a JSON object.")

        secret = str(payload.get("secret", "")).strip()
        strategy = str(payload.get("strategy", "")).strip()
        symbol = str(payload.get("symbol", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        side = str(payload.get("side", "flat")).strip().lower()

        if not secret:
            raise ValueError("Missing secret.")
        if not strategy:
            raise ValueError("Missing strategy.")
        if not symbol:
            raise ValueError("Missing symbol.")
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        if side not in VALID_SIDES:
            raise ValueError(f"Unsupported side: {side}")

        def _to_float(name: str) -> float | None:
            value = payload.get(name)
            if value in (None, "", "null"):
                return None
            return float(value)

        return cls(
            secret=secret,
            strategy=strategy,
            symbol=symbol,
            action=action,
            side=side,
            timeframe=str(payload.get("timeframe", "")).strip(),
            qty=_to_float("qty"),
            price=_to_float("price"),
            stop_loss=_to_float("stop_loss"),
            take_profit=_to_float("take_profit"),
            order_id=str(payload.get("order_id", "")).strip(),
            comment=str(payload.get("comment", "")).strip(),
            event_time=str(payload.get("event_time", "")).strip(),
            raw=dict(payload),
        )

    def dedupe_key(self) -> str:
        parts = [
            self.strategy,
            self.symbol,
            self.action,
            self.side,
            self.order_id,
            self.event_time,
        ]
        return "|".join(part for part in parts if part)

    def event_dt_utc(self) -> datetime:
        raw = self.event_time.strip()
        if raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)


@dataclass
class ExecutionResult:
    success: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
