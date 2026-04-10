from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.live.config import LiveTradingConfig
from src.live.execution import MT5Executor, PaperExecutor
from src.live.models import ExecutionResult, LiveSignal
from src.live.risk import LiveRiskManager
from src.live.sizing import PositionSizer


class LiveTradingService:
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.event_log_path = config.resolve_path(config.event_log_path)
        self.executor = self._build_executor()
        self.risk = LiveRiskManager(config)
        self.sizer = PositionSizer(config)

    def _build_executor(self):
        if self.config.mode == "mt5":
            return MT5Executor()
        return PaperExecutor(
            log_path=self.config.resolve_path(self.config.paper_log_path),
            state_path=self.config.resolve_path(self.config.paper_state_path),
        )

    def _append_event(self, payload: dict[str, Any]) -> None:
        with open(self.event_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _validate_signal(self, signal: LiveSignal) -> None:
        if signal.secret != self.config.shared_secret:
            raise ValueError("Invalid shared secret.")
        if signal.strategy not in self.config.allowed_strategies:
            raise ValueError(f"Strategy not allowed: {signal.strategy}")

    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        signal = LiveSignal.from_payload(payload)
        self._validate_signal(signal)
        signal.broker_symbol = self.config.symbol_map.get(signal.symbol, signal.symbol)
        risk_snapshot_before = self.risk.snapshot()

        if self.risk.is_duplicate(signal):
            result = ExecutionResult(
                True,
                "Duplicate signal ignored.",
                {"duplicate": True, "risk": risk_snapshot_before},
            )
        else:
            sizing = self.sizer.resolve(signal)
            if not sizing.success:
                result = ExecutionResult(
                    False,
                    sizing.message,
                    {
                        "blocked": True,
                        "risk": risk_snapshot_before,
                        **sizing.details,
                    },
                )
            else:
                if signal.action == "entry" and sizing.qty is not None:
                    signal.qty = sizing.qty

                open_positions = 0
                if signal.action == "entry":
                    open_positions = self.executor.open_positions_count(signal.broker_symbol, self.config)
                risk_check = self.risk.check_entry(signal, open_positions)
                if not risk_check.allowed:
                    result = ExecutionResult(
                        False,
                        risk_check.reason,
                        {
                            "blocked": True,
                            "risk": risk_snapshot_before,
                            "sizing": sizing.details,
                            **(risk_check.details or {}),
                        },
                    )
                else:
                    result = self.executor.execute(signal, self.config)
                    result.details = {**result.details, "sizing": sizing.details}
                    if result.success:
                        self.risk.record_success(signal, result.details)

        risk_snapshot_after = self.risk.snapshot()
        result.details = {**result.details, "risk": risk_snapshot_after}

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode,
            "signal": signal.raw,
            "broker_symbol": signal.broker_symbol,
            "result": {
                "success": result.success,
                "message": result.message,
                "details": result.details,
            },
        }
        self._append_event(event)
        return event
