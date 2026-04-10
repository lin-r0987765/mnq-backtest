from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MT5Config:
    terminal_path: str = ""
    login: int | None = None
    password: str = ""
    server: str = ""
    deviation: int = 20
    magic: int = 33033


@dataclass
class LiveTradingConfig:
    mode: str = "paper"
    bind_host: str = "0.0.0.0"
    bind_port: int = 8000
    webhook_path: str = "/webhook"
    shared_secret: str = "change-me"
    default_volume: float = 10.0
    position_sizing_mode: str = "fixed"
    account_equity: float = 100_000.0
    risk_per_trade_pct: float = 0.0025
    max_notional_pct: float = 0.06
    contract_multiplier: float = 1.0
    min_volume: float = 1.0
    max_volume: float = 0.0
    volume_step: float = 1.0
    allow_signal_qty_override: bool = False
    allow_default_fallback: bool = True
    allowed_strategies: list[str] = field(default_factory=lambda: ["ORB"])
    symbol_map: dict[str, str] = field(default_factory=dict)
    event_log_path: str = "iteration_logs/live_webhook_events.jsonl"
    paper_log_path: str = "iteration_logs/live_paper_orders.jsonl"
    paper_state_path: str = "iteration_logs/live_paper_state.json"
    risk_state_path: str = "iteration_logs/live_risk_state.json"
    kill_switch_path: str = "iteration_logs/live_kill_switch.json"
    dedupe_window_seconds: int = 3600
    cooldown_seconds: int = 300
    max_daily_entries_per_symbol: int = 3
    max_open_positions_per_symbol: int = 1
    paper_max_daily_loss: float = 0.0
    mt5: MT5Config = field(default_factory=MT5Config)

    @classmethod
    def load(cls, path: str | Path) -> "LiveTradingConfig":
        config_path = Path(path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))

        mt5_raw = raw.get("mt5", {})
        mt5 = MT5Config(
            terminal_path=str(mt5_raw.get("terminal_path", "")),
            login=int(mt5_raw["login"]) if mt5_raw.get("login") not in (None, "") else None,
            password=str(mt5_raw.get("password", "")),
            server=str(mt5_raw.get("server", "")),
            deviation=int(mt5_raw.get("deviation", 20)),
            magic=int(mt5_raw.get("magic", 33033)),
        )

        webhook_path = str(raw.get("webhook_path", "/webhook")).strip() or "/webhook"
        if not webhook_path.startswith("/"):
            webhook_path = "/" + webhook_path

        return cls(
            mode=str(raw.get("mode", "paper")).strip().lower(),
            bind_host=str(raw.get("bind_host", "0.0.0.0")).strip(),
            bind_port=int(raw.get("bind_port", 8000)),
            webhook_path=webhook_path,
            shared_secret=str(raw.get("shared_secret", "change-me")).strip(),
            default_volume=float(raw.get("default_volume", 10.0)),
            position_sizing_mode=str(raw.get("position_sizing_mode", "fixed")).strip().lower(),
            account_equity=float(raw.get("account_equity", 100_000.0)),
            risk_per_trade_pct=float(raw.get("risk_per_trade_pct", 0.0025)),
            max_notional_pct=float(raw.get("max_notional_pct", 0.06)),
            contract_multiplier=float(raw.get("contract_multiplier", 1.0)),
            min_volume=float(raw.get("min_volume", 1.0)),
            max_volume=float(raw.get("max_volume", 0.0)),
            volume_step=float(raw.get("volume_step", 1.0)),
            allow_signal_qty_override=bool(raw.get("allow_signal_qty_override", False)),
            allow_default_fallback=bool(raw.get("allow_default_fallback", True)),
            allowed_strategies=[str(item).strip() for item in raw.get("allowed_strategies", ["ORB"])],
            symbol_map={str(k): str(v) for k, v in raw.get("symbol_map", {}).items()},
            event_log_path=str(raw.get("event_log_path", "iteration_logs/live_webhook_events.jsonl")),
            paper_log_path=str(raw.get("paper_log_path", "iteration_logs/live_paper_orders.jsonl")),
            paper_state_path=str(raw.get("paper_state_path", "iteration_logs/live_paper_state.json")),
            risk_state_path=str(raw.get("risk_state_path", "iteration_logs/live_risk_state.json")),
            kill_switch_path=str(raw.get("kill_switch_path", "iteration_logs/live_kill_switch.json")),
            dedupe_window_seconds=int(raw.get("dedupe_window_seconds", 3600)),
            cooldown_seconds=int(raw.get("cooldown_seconds", 300)),
            max_daily_entries_per_symbol=int(raw.get("max_daily_entries_per_symbol", 3)),
            max_open_positions_per_symbol=int(raw.get("max_open_positions_per_symbol", 1)),
            paper_max_daily_loss=float(raw.get("paper_max_daily_loss", 0.0)),
            mt5=mt5,
        )

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
