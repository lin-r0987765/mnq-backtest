from __future__ import annotations

from pathlib import Path
import os
import unittest
from unittest.mock import patch

from run_deep_backtest import (
    build_lean_command,
    load_orb_lean_parameters,
    login_with_credentials,
    read_login_env,
    run_preflight,
)


class DeepBacktestScriptTest(unittest.TestCase):
    def test_load_orb_parameters_matches_current_defaults(self) -> None:
        params = load_orb_lean_parameters(
            symbol="QQQ",
            trade_quantity=10,
            start="2017-04-03",
            end="2026-04-02",
            cash=100000.0,
        )
        self.assertEqual(params["symbol"], "QQQ")
        self.assertEqual(params["trade_quantity"], "10")
        self.assertEqual(params["orb_bars"], "4")
        self.assertEqual(params["profit_ratio"], "3.5")
        self.assertEqual(params["trailing_pct"], "0.013")
        self.assertEqual(params["position_size_pct"], "0.25")
        self.assertEqual(params["max_entries_per_session"], "1")
        self.assertEqual(params["regime_mode"], "prev_day_up_and_mom3_positive")
        self.assertEqual(params["breakeven_trigger_mult"], "1.25")
        self.assertEqual(params["breakeven_active_minutes"], "180")
        self.assertEqual(params["profit_lock_trigger_mult"], "1.50")
        self.assertEqual(params["profit_lock_level_mult"], "0.25")
        self.assertEqual(params["early_tight_trail_pct"], "0.013")
        self.assertEqual(params["early_tight_trail_minutes"], "0")
        self.assertEqual(params["htf_filter"], "true")
        self.assertEqual(params["skip_short_after_up_days"], "2")
        self.assertEqual(params["skip_long_after_up_days"], "3")

    def test_build_lean_command_contains_required_flags(self) -> None:
        project_dir = str(Path(__file__).resolve().parent / "lean" / "QQQ_ORB_DeepBacktest")
        command = build_lean_command(
            backtest_name="QQQ ORB Deep 8Y",
            params={"symbol": "QQQ", "orb_bars": "4"},
            download_data=True,
            data_provider_historical="QuantConnect",
            data_purchase_limit=25,
        )
        self.assertTrue(command[0].endswith("lean.exe") or command[0] == "lean")
        self.assertEqual(command[1:3], ["backtest", project_dir])
        self.assertIn("--download-data", command)
        self.assertIn("--data-provider-historical", command)
        self.assertIn("QuantConnect", command)
        self.assertIn("--data-purchase-limit", command)
        self.assertIn("25", command)
        self.assertGreaterEqual(command.count("--parameter"), 2)

    @patch("run_deep_backtest.find_lean_executable", return_value=None)
    @patch("run_deep_backtest.shutil.which")
    def test_preflight_reports_missing_lean(self, mock_which, mock_find_lean) -> None:
        def fake_which(binary: str):
            if binary == "docker":
                return "C:/Program Files/Docker/docker.exe"
            return None

        mock_which.side_effect = fake_which
        status = run_preflight()
        self.assertFalse(status.lean_available)
        self.assertIsNone(status.lean_path)
        self.assertTrue(status.docker_available)
        self.assertTrue(status.project_exists)
        self.assertTrue(status.main_exists)
        self.assertFalse(status.ready)
        mock_find_lean.assert_called_once()

    @patch("run_deep_backtest.check_lean_login", return_value=False)
    @patch("run_deep_backtest.shutil.which")
    def test_preflight_detects_user_scripts_lean(self, mock_which, mock_login) -> None:
        lean_path = str(Path.home() / "AppData" / "Roaming" / "Python" / "Python311" / "Scripts" / "lean.exe")

        def fake_which(binary: str):
            if binary == "docker":
                return "C:/Program Files/Docker/docker.exe"
            return None

        mock_which.side_effect = fake_which
        with patch("run_deep_backtest.site.getusersitepackages", return_value=str(Path.home() / "AppData" / "Roaming" / "Python" / "Python311" / "site-packages")):
            with patch("run_deep_backtest.Path.exists", autospec=True) as mock_exists:
                def fake_exists(path_obj):
                    return str(path_obj) == lean_path or "QQQ_ORB_DeepBacktest" in str(path_obj)
                mock_exists.side_effect = fake_exists
                status = run_preflight()

        self.assertTrue(status.lean_available)
        self.assertEqual(status.lean_path, lean_path)
        self.assertFalse(status.lean_logged_in)
        self.assertFalse(status.ready)
        mock_login.assert_called()

    @patch.dict(os.environ, {"QUANTCONNECT_USER_ID": "12345", "QUANTCONNECT_API_TOKEN": "secret"}, clear=True)
    def test_read_login_env(self) -> None:
        user_id, api_token = read_login_env("QUANTCONNECT_USER_ID", "QUANTCONNECT_API_TOKEN")
        self.assertEqual(user_id, "12345")
        self.assertEqual(api_token, "secret")

    @patch("run_deep_backtest.subprocess.run")
    def test_login_with_credentials_success(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Logged in successfully"
        mock_run.return_value.stderr = ""
        ok, output = login_with_credentials(Path("lean.exe"), user_id="123", api_token="abc")
        self.assertTrue(ok)
        self.assertIn("Logged in successfully", output)

    @patch("run_deep_backtest.subprocess.run")
    def test_login_with_credentials_failure(self, mock_run) -> None:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "Unauthorized"
        ok, output = login_with_credentials(Path("lean.exe"), user_id="123", api_token="bad")
        self.assertFalse(ok)
        self.assertIn("Unauthorized", output)


if __name__ == "__main__":
    unittest.main()
