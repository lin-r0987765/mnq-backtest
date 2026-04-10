from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.live.config import LiveTradingConfig
from src.live.models import ExecutionResult, LiveSignal

logger = logging.getLogger(__name__)


class BrokerExecutor(ABC):
    @abstractmethod
    def execute(self, signal: LiveSignal, config: LiveTradingConfig) -> ExecutionResult:
        raise NotImplementedError

    @abstractmethod
    def open_positions_count(self, symbol: str, config: LiveTradingConfig) -> int:
        raise NotImplementedError


class PaperExecutor(BrokerExecutor):
    def __init__(self, log_path: Path, state_path: Path):
        self.log_path = log_path
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> dict[str, list[dict[str, Any]]]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append_log(self, payload: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def open_positions_count(self, symbol: str, config: LiveTradingConfig) -> int:
        return len(self.state.get(symbol, []))

    def execute(self, signal: LiveSignal, config: LiveTradingConfig) -> ExecutionResult:
        symbol = signal.broker_symbol or signal.symbol
        qty = signal.qty if signal.qty is not None else config.default_volume
        now = datetime.now(timezone.utc).isoformat()
        symbol_positions = self.state.setdefault(symbol, [])

        record = {
            "timestamp": now,
            "mode": "paper",
            "strategy": signal.strategy,
            "symbol": symbol,
            "action": signal.action,
            "side": signal.side,
            "qty": qty,
            "price": signal.price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "order_id": signal.order_id,
            "comment": signal.comment,
        }

        if signal.action == "entry":
            position = {
                "id": signal.order_id or f"{signal.strategy}-{signal.side}-{len(symbol_positions)+1}",
                "strategy": signal.strategy,
                "side": signal.side,
                "qty": qty,
                "opened_at": now,
                "entry_price": signal.price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            }
            symbol_positions.append(position)
            self._save_state()
            self._append_log({**record, "result": "opened", "open_positions": len(symbol_positions)})
            return ExecutionResult(True, "Paper entry recorded.", {"open_positions": len(symbol_positions)})

        if signal.action in {"exit", "close_all"}:
            exit_price = signal.price
            if signal.action == "close_all":
                closed = len(symbol_positions)
                realized_pnl = 0.0
                for position in symbol_positions:
                    if exit_price is not None and position.get("entry_price") is not None:
                        if position["side"] == "buy":
                            realized_pnl += (exit_price - float(position["entry_price"])) * float(position["qty"])
                        else:
                            realized_pnl += (float(position["entry_price"]) - exit_price) * float(position["qty"])
                symbol_positions.clear()
            else:
                target_side = signal.side
                remaining = []
                closed = 0
                realized_pnl = 0.0
                for position in symbol_positions:
                    if target_side == "flat" or position["side"] == target_side:
                        closed += 1
                        if exit_price is not None and position.get("entry_price") is not None:
                            if position["side"] == "buy":
                                realized_pnl += (exit_price - float(position["entry_price"])) * float(position["qty"])
                            else:
                                realized_pnl += (float(position["entry_price"]) - exit_price) * float(position["qty"])
                    else:
                        remaining.append(position)
                self.state[symbol] = remaining
            self._save_state()
            details = {
                "closed_positions": closed,
                "realized_pnl": round(realized_pnl, 6),
            }
            self._append_log({**record, "result": "closed", **details})
            return ExecutionResult(True, "Paper exit recorded.", details)

        self._append_log({**record, "result": "heartbeat"})
        return ExecutionResult(True, "Heartbeat accepted.")


class MT5Executor(BrokerExecutor):
    def __init__(self) -> None:
        self._mt5 = None
        self._connected = False
        self._capability_cache: dict[str, Any] = {}

    def _load_mt5(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "MetaTrader5 package is not installed. Run: pip install -r requirements-live.txt"
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def open_positions_count(self, symbol: str, config: LiveTradingConfig) -> int:
        mt5 = self._ensure_connected(config)
        positions = mt5.positions_get(symbol=symbol) or []
        return len(positions)

    def _ensure_connected(self, config: LiveTradingConfig):
        mt5 = self._load_mt5()
        if self._connected:
            return mt5

        kwargs: dict[str, Any] = {}
        if config.mt5.terminal_path:
            kwargs["path"] = config.mt5.terminal_path
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

        if config.mt5.login:
            login_ok = mt5.login(
                login=config.mt5.login,
                password=config.mt5.password,
                server=config.mt5.server,
            )
            if not login_ok:
                raise RuntimeError(f"MT5 login() failed: {mt5.last_error()}")

        self._connected = True
        return mt5

    def _select_symbol(self, mt5: Any, symbol: str):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5 symbol_info failed for symbol: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed for symbol: {symbol}")
        return info

    def _order_filling(self, mt5: Any, info: Any) -> int:
        if hasattr(info, "filling_mode") and isinstance(info.filling_mode, int):
            return int(info.filling_mode)
        return int(mt5.ORDER_FILLING_RETURN)

    def get_symbol_capability(self, mt5: Any, symbol: str):
        """Query and cache broker capability for a symbol."""
        from src.live.broker import query_symbol_capability

        if symbol not in self._capability_cache:
            self._capability_cache[symbol] = query_symbol_capability(mt5, symbol)
        return self._capability_cache[symbol]

    def validate_entry(self, mt5: Any, signal: LiveSignal, config: LiveTradingConfig) -> ExecutionResult | None:
        """Pre-flight broker validation. Returns ExecutionResult on failure, None on pass."""
        from src.live.broker import check_trade_direction, validate_volume

        symbol = signal.broker_symbol or signal.symbol
        cap = self.get_symbol_capability(mt5, symbol)

        # 1. Check trade direction
        allowed, reason = check_trade_direction(cap, signal.side)
        if not allowed:
            return ExecutionResult(
                False,
                reason,
                {
                    "blocked_by": "broker_capability",
                    "trade_mode": cap.trade_mode_desc,
                    "symbol": symbol,
                },
            )

        # 2. Validate & adjust volume
        qty = signal.qty if signal.qty is not None else config.default_volume
        vol_result = validate_volume(qty, cap, auto_adjust=True)

        if not vol_result.valid:
            return ExecutionResult(
                False,
                vol_result.message,
                {
                    "blocked_by": "broker_volume_validation",
                    "symbol": symbol,
                    **vol_result.details,
                },
            )

        # Apply broker-adjusted volume back to signal
        if vol_result.details.get("was_adjusted"):
            logger.warning(
                "Volume adjusted by broker constraints for %s: %.4f -> %.4f",
                symbol,
                qty,
                vol_result.adjusted_qty,
            )
            signal.qty = vol_result.adjusted_qty

        return None  # All checks passed

    def _entry_request(self, mt5: Any, signal: LiveSignal, config: LiveTradingConfig) -> dict[str, Any]:
        symbol = signal.broker_symbol or signal.symbol
        info = self._select_symbol(mt5, symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"MT5 symbol_info_tick failed for symbol: {symbol}")

        if signal.side == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif signal.side == "sell":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            raise RuntimeError(f"Entry side must be buy or sell, got: {signal.side}")

        volume = signal.qty if signal.qty is not None else config.default_volume

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": config.mt5.deviation,
            "magic": config.mt5.magic,
            "comment": signal.comment or signal.strategy,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._order_filling(mt5, info),
        }
        if signal.stop_loss is not None:
            request["sl"] = signal.stop_loss
        if signal.take_profit is not None:
            request["tp"] = signal.take_profit
        return request

    def _close_positions(self, mt5: Any, signal: LiveSignal, config: LiveTradingConfig) -> ExecutionResult:
        symbol = signal.broker_symbol or signal.symbol
        self._select_symbol(mt5, symbol)
        positions = mt5.positions_get(symbol=symbol) or []
        if signal.action == "exit":
            if signal.side == "buy":
                positions = [pos for pos in positions if pos.type == mt5.POSITION_TYPE_BUY]
            elif signal.side == "sell":
                positions = [pos for pos in positions if pos.type == mt5.POSITION_TYPE_SELL]

        closed = []
        for position in positions:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise RuntimeError(f"MT5 symbol_info_tick failed while closing {symbol}")

            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": config.mt5.deviation,
                "magic": config.mt5.magic,
                "comment": signal.comment or f"Close {signal.strategy}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": int(mt5.ORDER_FILLING_RETURN),
            }
            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError(f"MT5 order_send returned None when closing {symbol}")
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(
                    f"MT5 close failed retcode={result.retcode}, comment={getattr(result, 'comment', '')}"
                )
            closed.append({"position": position.ticket, "retcode": result.retcode})

        return ExecutionResult(True, "MT5 close executed.", {"closed_positions": closed})

    def execute(self, signal: LiveSignal, config: LiveTradingConfig) -> ExecutionResult:
        mt5 = self._ensure_connected(config)

        if signal.action == "heartbeat":
            return ExecutionResult(True, "Heartbeat accepted.")

        if signal.action == "entry":
            # Pre-flight broker capability validation
            validation_failure = self.validate_entry(mt5, signal, config)
            if validation_failure is not None:
                return validation_failure

            request = self._entry_request(mt5, signal, config)
            result = mt5.order_send(request)
            if result is None:
                raise RuntimeError("MT5 order_send returned None.")
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(
                    f"MT5 order_send failed retcode={result.retcode}, comment={getattr(result, 'comment', '')}"
                )

            # Include broker capability info in result
            symbol = signal.broker_symbol or signal.symbol
            broker_info = {}
            if symbol in self._capability_cache:
                cap = self._capability_cache[symbol]
                broker_info = {
                    "broker_volume_min": cap.volume_min,
                    "broker_volume_max": cap.volume_max,
                    "broker_volume_step": cap.volume_step,
                    "broker_trade_mode": cap.trade_mode_desc,
                }

            return ExecutionResult(
                True,
                "MT5 entry executed.",
                {
                    "retcode": result.retcode,
                    "order": getattr(result, "order", None),
                    "deal": getattr(result, "deal", None),
                    "price": getattr(result, "price", None),
                    "broker": broker_info,
                },
            )

        if signal.action in {"exit", "close_all"}:
            return self._close_positions(mt5, signal, config)

        raise RuntimeError(f"Unsupported action: {signal.action}")
