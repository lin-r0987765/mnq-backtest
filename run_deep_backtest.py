#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantConnect / LEAN 深度回測入口。

用途：
1. 檢查本機是否已安裝 LEAN CLI 與 Docker
2. 將目前 repo 的 ORB 預設參數同步成 LEAN parameter manifest
3. 產生可直接執行的 lean backtest 指令
4. 在條件齊備時直接執行深度回測
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import site
import subprocess
import sys
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path

from config import INITIAL_CAPITAL
from src.strategies.orb import ORBStrategy


PROJECT_ROOT = Path(__file__).resolve().parent
LEAN_PROJECT_DIR = PROJECT_ROOT / "lean" / "QQQ_ORB_DeepBacktest"
LEAN_MAIN = LEAN_PROJECT_DIR / "main.py"
LEAN_PARAM_PATH = LEAN_PROJECT_DIR / "parameters.generated.json"
LEAN_COMMAND_PATH = LEAN_PROJECT_DIR / "last_command.txt"
QC_WEB_IDE_PATH = LEAN_PROJECT_DIR / "QQQ_ORB_WebIDE.py"
DEFAULT_USER_ID_ENV = "QUANTCONNECT_USER_ID"
DEFAULT_API_TOKEN_ENV = "QUANTCONNECT_API_TOKEN"


SUPPORTED_ORB_KEYS = [
    "orb_bars",
    "profit_ratio",
    "breakout_confirm_pct",
    "entry_delay_bars",
    "trailing_pct",
    "close_before_min",
    "multi_day_range",
    "multi_day_lookback",
    "htf_filter",
    "htf_mode",
    "htf_ema_fast",
    "htf_ema_slow",
    "skip_short_after_up_days",
    "skip_long_after_up_days",
]

QC_FIXED_PARAMS = {
    "position_size_pct": "0.25",
    "min_trade_quantity": "1",
    "max_trade_quantity": "500",
    "max_entries_per_session": "1",
    "entry_start_hour_utc": "0",
    "entry_end_hour_utc": "17",
    "min_range_pct": "0.001",
    "regime_filter": "true",
    "regime_mode": "prev_day_up_and_mom3_positive",
    "regime_allow_shorts": "false",
    "regime_min_history_days": "4",
    "breakeven_trigger_mult": "1.25",
    "breakeven_active_minutes": "180",
    "profit_lock_trigger_mult": "1.50",
    "profit_lock_level_mult": "0.25",
    "early_tight_trail_pct": "0.013",
    "early_tight_trail_minutes": "0",
}


@dataclass
class PreflightStatus:
    lean_available: bool
    lean_path: str | None
    lean_logged_in: bool
    docker_available: bool
    project_exists: bool
    main_exists: bool

    @property
    def ready(self) -> bool:
        return (
            self.lean_available
            and self.lean_logged_in
            and self.docker_available
            and self.project_exists
            and self.main_exists
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuantConnect / LEAN QQQ ORB deep backtest")
    parser.add_argument("--symbol", default="QQQ", help="Lean parameter: symbol")
    parser.add_argument("--trade-quantity", type=int, default=10, help="Lean parameter: trade_quantity")
    parser.add_argument("--start", default="2017-04-03", help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-04-02", help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--cash", type=float, default=float(INITIAL_CAPITAL), help="Lean parameter: cash")
    parser.add_argument("--backtest-name", default="QQQ ORB Deep 8Y", help="LEAN backtest name")
    parser.add_argument("--check-only", action="store_true", help="Only run preflight and write command manifest")
    parser.add_argument("--print-command", action="store_true", help="Print the generated lean command")
    parser.add_argument("--login-from-env", action="store_true", help="Use env vars to run `lean login` before preflight")
    parser.add_argument("--user-id-env", default=DEFAULT_USER_ID_ENV, help="Environment variable for QuantConnect user id")
    parser.add_argument("--api-token-env", default=DEFAULT_API_TOKEN_ENV, help="Environment variable for QuantConnect API token")
    parser.add_argument("--no-download-data", action="store_true", help="Do not add --download-data")
    parser.add_argument(
        "--data-provider-historical",
        default="QuantConnect",
        choices=["QuantConnect", "Local", "Polygon"],
        help="Historical data provider passed to lean backtest",
    )
    parser.add_argument("--data-purchase-limit", type=int, default=None, help="Optional QCC purchase cap")
    return parser.parse_args()


def _split_date(value: str) -> tuple[int, int, int]:
    year_str, month_str, day_str = value.split("-")
    return int(year_str), int(month_str), int(day_str)


def load_orb_lean_parameters(
    *,
    symbol: str,
    trade_quantity: int,
    start: str,
    end: str,
    cash: float,
) -> dict[str, str]:
    params = ORBStrategy().params
    lean_params = {
        "symbol": symbol,
        "trade_quantity": str(int(trade_quantity)),
        "cash": str(float(cash)),
    }

    start_year, start_month, start_day = _split_date(start)
    end_year, end_month, end_day = _split_date(end)
    lean_params.update(
        {
            "start_year": str(start_year),
            "start_month": str(start_month),
            "start_day": str(start_day),
            "end_year": str(end_year),
            "end_month": str(end_month),
            "end_day": str(end_day),
        }
    )

    for key in SUPPORTED_ORB_KEYS:
        value = params[key]
        if isinstance(value, bool):
            lean_params[key] = "true" if value else "false"
        else:
            lean_params[key] = str(value)

    lean_params.update(QC_FIXED_PARAMS)
    return lean_params


def run_preflight() -> PreflightStatus:
    lean_path = find_lean_executable()
    return PreflightStatus(
        lean_available=lean_path is not None,
        lean_path=str(lean_path) if lean_path else None,
        lean_logged_in=check_lean_login(lean_path) if lean_path else False,
        docker_available=shutil.which("docker") is not None,
        project_exists=LEAN_PROJECT_DIR.exists(),
        main_exists=LEAN_MAIN.exists(),
    )


def build_lean_command(
    *,
    backtest_name: str,
    params: dict[str, str],
    download_data: bool,
    data_provider_historical: str,
    data_purchase_limit: int | None,
) -> list[str]:
    command = [
        str(find_lean_executable() or "lean"),
        "backtest",
        str(LEAN_PROJECT_DIR),
        "--backtest-name",
        backtest_name,
        "--data-provider-historical",
        data_provider_historical,
    ]
    if download_data:
        command.append("--download-data")
    if data_purchase_limit is not None:
        command.extend(["--data-purchase-limit", str(int(data_purchase_limit))])
    for key, value in params.items():
        command.extend(["--parameter", key, value])
    return command


def write_manifests(preflight: PreflightStatus, params: dict[str, str], command: list[str]) -> None:
    LEAN_PARAM_PATH.write_text(
        json.dumps(
            {
                "preflight": asdict(preflight),
                "parameters": params,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    LEAN_COMMAND_PATH.write_text(shlex.join(command), encoding="utf-8")


def print_preflight(preflight: PreflightStatus) -> None:
    print("LEAN preflight:")
    print(f"- lean CLI: {'OK' if preflight.lean_available else 'MISSING'}")
    print(f"- lean path: {preflight.lean_path or 'N/A'}")
    print(f"- lean login: {'OK' if preflight.lean_logged_in else 'MISSING'}")
    print(f"- Docker: {'OK' if preflight.docker_available else 'MISSING'}")
    print(f"- Project dir: {'OK' if preflight.project_exists else 'MISSING'}")
    print(f"- main.py: {'OK' if preflight.main_exists else 'MISSING'}")


def read_login_env(user_id_env: str, api_token_env: str) -> tuple[str | None, str | None]:
    user_id = os.getenv(user_id_env, "").strip() or None
    api_token = os.getenv(api_token_env, "").strip() or None
    return user_id, api_token


def login_with_credentials(lean_path: Path, *, user_id: str, api_token: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [str(lean_path), "login", "--user-id", user_id, "--api-token", api_token],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
    except Exception as exc:
        return False, str(exc)

    output = "\n".join([completed.stdout or "", completed.stderr or ""]).strip()
    return completed.returncode == 0, output


def find_lean_executable() -> Path | None:
    direct = shutil.which("lean")
    if direct:
        return Path(direct)

    candidates: list[Path] = []
    scripts_path = sysconfig.get_path("scripts")
    if scripts_path:
        candidates.append(Path(scripts_path) / "lean.exe")
        candidates.append(Path(scripts_path) / "lean")

    user_site = site.getusersitepackages()
    candidates.append(Path(user_site).parent / "Scripts" / "lean.exe")
    candidates.append(Path(user_site).parent / "Scripts" / "lean")

    appdata = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts"
    candidates.append(appdata / "lean.exe")
    candidates.append(appdata / "lean")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def check_lean_login(lean_path: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(lean_path), "whoami"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
        )
    except Exception:
        return False

    output = "\n".join([completed.stdout or "", completed.stderr or ""]).lower()
    if "not logged in" in output:
        return False
    return completed.returncode == 0


def main() -> int:
    args = parse_args()
    preflight = run_preflight()
    if args.login_from_env:
        if not preflight.lean_available or not preflight.lean_path:
            print("無法執行 env login，因為 lean CLI 尚未可用。")
            return 2
        if preflight.lean_logged_in:
            print("LEAN 已登入，略過 env login。")
        else:
            user_id, api_token = read_login_env(args.user_id_env, args.api_token_env)
            if not user_id or not api_token:
                print(
                    "找不到 QuantConnect 憑證環境變數。"
                    f" 需要 `{args.user_id_env}` 與 `{args.api_token_env}`。"
                )
                return 2
            ok, output = login_with_credentials(
                Path(preflight.lean_path),
                user_id=user_id,
                api_token=api_token,
            )
            if output:
                print(output)
            if not ok:
                print("LEAN env login 失敗。")
                return 2
            preflight = run_preflight()

    params = load_orb_lean_parameters(
        symbol=args.symbol,
        trade_quantity=args.trade_quantity,
        start=args.start,
        end=args.end,
        cash=args.cash,
    )
    command = build_lean_command(
        backtest_name=args.backtest_name,
        params=params,
        download_data=not args.no_download_data,
        data_provider_historical=args.data_provider_historical,
        data_purchase_limit=args.data_purchase_limit,
    )
    write_manifests(preflight, params, command)
    print_preflight(preflight)
    print(f"- Parameter manifest: {LEAN_PARAM_PATH}")
    print(f"- Command manifest: {LEAN_COMMAND_PATH}")

    if args.print_command or args.check_only:
        print("\nGenerated command:")
        print(shlex.join(command))

    if args.check_only:
        return 0

    if args.check_only:
        return 0

    if not preflight.ready:
        print("\n???? LEAN backtest?????????????")
        if not preflight.lean_available:
            print("- ??? lean CLI?")
        if preflight.lean_available and not preflight.lean_logged_in:
            print("- LEAN ????????? `lean login` ??? `--login-from-env`?")
        if not preflight.docker_available:
            print("- ??? Docker?")
        if not preflight.project_exists:
            print(f"- ???????: {LEAN_PROJECT_DIR}")
        if not preflight.main_exists:
            print(f"- ??????: {LEAN_MAIN}")
        return 2

    print("\nRunning command:")
    print(shlex.join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
