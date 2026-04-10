#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.config import LiveTradingConfig
from src.live.service import LiveTradingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TradingView webhook bridge for paper or MT5 execution")
    parser.add_argument(
        "--config",
        default="live_trading.example.json",
        help="Path to live trading JSON config",
    )
    return parser.parse_args()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class LiveWebhookHandler(BaseHTTPRequestHandler):
    service: LiveTradingService
    webhook_path: str

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "mode": self.service.config.mode,
                    "webhook_path": self.webhook_path,
                    "position_sizing": {
                        "mode": self.service.config.position_sizing_mode,
                        "default_volume": self.service.config.default_volume,
                        "account_equity": self.service.config.account_equity,
                        "risk_per_trade_pct": self.service.config.risk_per_trade_pct,
                        "max_notional_pct": self.service.config.max_notional_pct,
                        "allow_signal_qty_override": self.service.config.allow_signal_qty_override,
                    },
                    "risk": self.service.risk.snapshot(),
                },
            )
            return
        _json_response(self, 404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        if self.path != self.webhook_path:
            _json_response(self, 404, {"ok": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            event = self.service.process_payload(payload)
            _json_response(self, 200, {"ok": True, "event": event})
        except ValueError as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = LiveTradingConfig.load(config_path)
    service = LiveTradingService(config)

    handler_cls = type(
        "ConfiguredLiveWebhookHandler",
        (LiveWebhookHandler,),
        {"service": service, "webhook_path": config.webhook_path},
    )
    server = ThreadingHTTPServer((config.bind_host, config.bind_port), handler_cls)

    print(f"Live trading bridge started on http://{config.bind_host}:{config.bind_port}")
    print(f"Webhook path: {config.webhook_path}")
    print(f"Mode: {config.mode}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live trading bridge...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
