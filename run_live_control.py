#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.config import LiveTradingConfig
from src.live.risk import LiveRiskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control kill switch and inspect live risk state")
    parser.add_argument(
        "--config",
        default="live_trading.example.json",
        help="Path to live trading JSON config",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current risk and kill switch status",
    )
    parser.add_argument(
        "--disable-entries",
        action="store_true",
        help="Enable kill switch and block new entries",
    )
    parser.add_argument(
        "--enable-entries",
        action="store_true",
        help="Disable kill switch and allow new entries",
    )
    parser.add_argument(
        "--reason",
        default="manual halt",
        help="Reason stored when disabling entries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = LiveTradingConfig.load(config_path)
    risk = LiveRiskManager(config)

    if args.disable_entries and args.enable_entries:
        raise SystemExit("Choose either --disable-entries or --enable-entries, not both.")

    output: dict[str, object] = {
        "config": str(config_path),
        "mode": config.mode,
        "position_sizing": {
            "mode": config.position_sizing_mode,
            "default_volume": config.default_volume,
            "account_equity": config.account_equity,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "max_notional_pct": config.max_notional_pct,
            "allow_signal_qty_override": config.allow_signal_qty_override,
            "allow_default_fallback": config.allow_default_fallback,
        },
    }

    if args.disable_entries:
        output["kill_switch"] = risk.set_kill_switch(True, args.reason)
    elif args.enable_entries:
        output["kill_switch"] = risk.set_kill_switch(False)
    else:
        output["kill_switch"] = risk.kill_switch_status()

    if args.status or not (args.disable_entries or args.enable_entries):
        output["risk"] = risk.snapshot()

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
