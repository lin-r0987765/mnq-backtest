from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.live.config import LiveTradingConfig
from src.live.models import LiveSignal


def _parse_utc(value: str) -> datetime | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""
    details: dict[str, Any] | None = None


class LiveRiskManager:
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.state_path = config.resolve_path(config.risk_state_path)
        self.kill_switch_path = config.resolve_path(config.kill_switch_path)
        self.state = self._load_state()
        self._normalize_day(self._current_day())
        self._save_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_state()
        if not isinstance(raw, dict):
            return self._default_state()
        state = self._default_state()
        state.update(raw)
        return state

    def _default_state(self) -> dict[str, Any]:
        return {
            "day": "",
            "entry_counts": {},
            "last_entry_at": {},
            "recent_signals": {},
            "realized_pnl_total": 0.0,
            "realized_pnl_by_strategy": {},
            "last_updated_at": "",
        }

    def _save_state(self) -> None:
        self.state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _signal_time(self, signal: LiveSignal | None = None) -> datetime:
        if signal is not None:
            parsed = _parse_utc(signal.event_time)
            if parsed is not None:
                return parsed
        return datetime.now(timezone.utc)

    def _current_day(self, signal: LiveSignal | None = None) -> str:
        return self._signal_time(signal).date().isoformat()

    def _normalize_day(self, day: str) -> None:
        if self.state.get("day") == day:
            return
        self.state["day"] = day
        self.state["entry_counts"] = {}
        self.state["last_entry_at"] = {}
        self.state["realized_pnl_total"] = 0.0
        self.state["realized_pnl_by_strategy"] = {}

    def _prune_recent_signals(self, now: datetime) -> None:
        window = max(int(self.config.dedupe_window_seconds), 0)
        if window <= 0:
            self.state["recent_signals"] = {}
            return
        keep_after = now - timedelta(seconds=window)
        recent = self.state.get("recent_signals", {})
        cleaned: dict[str, str] = {}
        for key, value in recent.items():
            parsed = _parse_utc(str(value))
            if parsed is not None and parsed >= keep_after:
                cleaned[key] = parsed.isoformat()
        self.state["recent_signals"] = cleaned

    def _entry_count_key(self, signal: LiveSignal) -> str:
        symbol = signal.broker_symbol or signal.symbol
        return f"{signal.strategy}|{symbol}"

    def _entry_side_key(self, signal: LiveSignal) -> str:
        symbol = signal.broker_symbol or signal.symbol
        return f"{signal.strategy}|{symbol}|{signal.side}"

    def kill_switch_status(self) -> dict[str, Any]:
        if not self.kill_switch_path.exists():
            return {"active": False, "reason": ""}
        try:
            raw = json.loads(self.kill_switch_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                active = bool(raw.get("disabled", True))
                reason = str(raw.get("reason", "")).strip()
                updated_at = str(raw.get("updated_at", "")).strip()
                return {
                    "active": active,
                    "reason": reason,
                    "updated_at": updated_at,
                }
        except Exception:
            pass
        return {"active": True, "reason": "kill switch file present"}

    def snapshot(self) -> dict[str, Any]:
        day = self._current_day()
        self._normalize_day(day)
        self._prune_recent_signals(datetime.now(timezone.utc))
        self._save_state()
        return {
            "day": self.state.get("day", day),
            "kill_switch": self.kill_switch_status(),
            "entry_counts": dict(self.state.get("entry_counts", {})),
            "realized_pnl_total": float(self.state.get("realized_pnl_total", 0.0)),
            "realized_pnl_by_strategy": dict(self.state.get("realized_pnl_by_strategy", {})),
            "recent_signal_count": len(self.state.get("recent_signals", {})),
        }

    def is_duplicate(self, signal: LiveSignal) -> bool:
        dedupe_key = signal.dedupe_key()
        if not dedupe_key:
            return False
        now = self._signal_time(signal)
        self._normalize_day(now.date().isoformat())
        self._prune_recent_signals(now)
        self._save_state()
        return dedupe_key in self.state.get("recent_signals", {})

    def check_entry(
        self,
        signal: LiveSignal,
        open_positions_count: int,
    ) -> RiskCheckResult:
        if signal.action != "entry":
            return RiskCheckResult(True)

        now = self._signal_time(signal)
        self._normalize_day(now.date().isoformat())
        self._prune_recent_signals(now)

        kill_switch = self.kill_switch_status()
        if kill_switch.get("active"):
            self._save_state()
            return RiskCheckResult(
                False,
                "Kill switch active.",
                {"blocked_by": "kill_switch", "reason": kill_switch.get("reason", "")},
            )

        if self.config.max_open_positions_per_symbol > 0:
            if open_positions_count >= self.config.max_open_positions_per_symbol:
                self._save_state()
                return RiskCheckResult(
                    False,
                    "Open position limit reached.",
                    {
                        "blocked_by": "max_open_positions_per_symbol",
                        "open_positions": open_positions_count,
                        "limit": self.config.max_open_positions_per_symbol,
                    },
                )

        if self.config.max_daily_entries_per_symbol > 0:
            key = self._entry_count_key(signal)
            current = int(self.state.get("entry_counts", {}).get(key, 0))
            if current >= self.config.max_daily_entries_per_symbol:
                self._save_state()
                return RiskCheckResult(
                    False,
                    "Daily entry limit reached.",
                    {
                        "blocked_by": "max_daily_entries_per_symbol",
                        "entry_count": current,
                        "limit": self.config.max_daily_entries_per_symbol,
                    },
                )

        if self.config.cooldown_seconds > 0:
            key = self._entry_side_key(signal)
            previous = _parse_utc(str(self.state.get("last_entry_at", {}).get(key, "")))
            if previous is not None:
                elapsed = (now - previous).total_seconds()
                if elapsed < self.config.cooldown_seconds:
                    self._save_state()
                    return RiskCheckResult(
                        False,
                        "Entry cooldown active.",
                        {
                            "blocked_by": "cooldown_seconds",
                            "elapsed_seconds": round(elapsed, 3),
                            "cooldown_seconds": self.config.cooldown_seconds,
                        },
                    )

        if self.config.paper_max_daily_loss > 0:
            realized = float(self.state.get("realized_pnl_total", 0.0))
            if realized <= -abs(self.config.paper_max_daily_loss):
                self._save_state()
                return RiskCheckResult(
                    False,
                    "Paper daily loss limit reached.",
                    {
                        "blocked_by": "paper_max_daily_loss",
                        "realized_pnl_total": realized,
                        "limit": -abs(self.config.paper_max_daily_loss),
                    },
                )

        self._save_state()
        return RiskCheckResult(True)

    def record_success(self, signal: LiveSignal, details: dict[str, Any]) -> None:
        now = self._signal_time(signal)
        day = now.date().isoformat()
        self._normalize_day(day)
        self._prune_recent_signals(now)

        dedupe_key = signal.dedupe_key()
        if dedupe_key:
            self.state.setdefault("recent_signals", {})[dedupe_key] = now.isoformat()

        if signal.action == "entry":
            count_key = self._entry_count_key(signal)
            entry_counts = self.state.setdefault("entry_counts", {})
            entry_counts[count_key] = int(entry_counts.get(count_key, 0)) + 1

            side_key = self._entry_side_key(signal)
            last_entry_at = self.state.setdefault("last_entry_at", {})
            last_entry_at[side_key] = now.isoformat()

        realized_pnl = details.get("realized_pnl")
        if realized_pnl not in (None, ""):
            pnl = float(realized_pnl)
            self.state["realized_pnl_total"] = round(float(self.state.get("realized_pnl_total", 0.0)) + pnl, 6)
            by_strategy = self.state.setdefault("realized_pnl_by_strategy", {})
            by_strategy[signal.strategy] = round(float(by_strategy.get(signal.strategy, 0.0)) + pnl, 6)

        self._save_state()

    def set_kill_switch(self, disabled: bool, reason: str = "") -> dict[str, Any]:
        if disabled:
            payload = {
                "disabled": True,
                "reason": reason,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.kill_switch_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            if self.kill_switch_path.exists():
                self.kill_switch_path.unlink()
        return self.kill_switch_status()
