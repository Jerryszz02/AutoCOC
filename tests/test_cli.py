from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
import io
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autococ.capture import ScreenshotCapture
from autococ.cli import main
from autococ.config import load_config
from autococ.device import ConnectedDevice, DisplayTarget
from autococ.errors import DeviceConnectionError
from autococ.reporting import RunStats, TaskResult
from autococ.scene import SceneSnapshot


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        path = self.root / "config.toml"
        path.write_text("", encoding="utf-8")
        config = load_config(path)
        self.config = replace(config, runtime=replace(config.runtime, screenshot_dir=self.root / "screenshots",
                                                     report_dir=self.root / "reports"))
        self.config_loader = self.patch("autococ.cli.load_config", return_value=self.config)
        self.patch("autococ.cli.setup_logging", return_value=Mock(spec=logging.Logger))
        self.manager = Mock()
        self.manager.connect.return_value = ConnectedDevice("test-device")
        self.manager_class = self.patch("autococ.cli.DeviceManager", return_value=self.manager)
        self.resolver = self.patch("autococ.cli.resolve_game_display", return_value=DisplayTarget(2, "physical-two"))
        self.capture = Mock()

        def frame(path: Path) -> ScreenshotCapture:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test-frame")
            return ScreenshotCapture(path, 2560, 1440, 0.1)

        self.capture.capture_screenshot_artifact.side_effect = frame
        self.capture_class = self.patch("autococ.cli.CaptureClient", return_value=self.capture)
        self.recognizer = Mock()
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("village", 0.95, Path(path), {"resources": {"gold": 100}})
        self.patch("autococ.cli.ScreenshotRecognizer", return_value=self.recognizer)

    def patch(self, target: str, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def invoke(self, *args: str) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(list(args))
        return code, output.getvalue(), error.getvalue()

    def test_offline_inspect_returns_clean_json_without_device_access(self) -> None:
        path = self.root / "offline.png"
        path.write_bytes(b"test-frame")
        code, output, error = self.invoke("inspect", "--image", str(path))
        self.assertEqual(code, 0, error)
        payload = json.loads(output)
        self.assertEqual(payload["scene"], "village")
        self.assertEqual(payload["confidence"], 0.95)
        self.assertEqual(payload["observations"]["resources"]["gold"], 100)
        self.manager_class.assert_not_called()
        self.capture_class.assert_not_called()
        self.resolver.assert_not_called()

    def test_gui_command_dispatches_without_device_connection_or_config_mutation(self) -> None:
        with patch("autococ.gui.main", return_value=0) as gui:
            code, _, _ = self.invoke("gui", "--config", "some-config.toml")
        self.assertEqual(code, 0)
        gui.assert_called_once_with(["--config", "some-config.toml"])
        self.manager_class.assert_not_called()
        self.config_loader.assert_not_called()

    def test_missing_offline_image_fails_without_device_access(self) -> None:
        code, output, error = self.invoke("inspect", "--image", str(self.root / "missing.png"))
        self.assertEqual(code, 1)
        self.assertIn("does not exist", error)
        self.assertEqual(output, "")
        self.manager_class.assert_not_called()

    def test_live_inspect_binds_both_display_ids_and_returns_json(self) -> None:
        code, output, error = self.invoke("inspect")
        self.assertEqual(code, 0, error)
        payload = json.loads(output)
        self.assertEqual(payload["logical_display_id"], 2)
        self.assertEqual(payload["physical_display_id"], "physical-two")
        self.assertEqual(payload["device_serial"], "test-device")
        self.assertTrue(Path(payload["screenshot_path"]).is_file())
        self.assertEqual(self.capture_class.call_args.kwargs, {"screenshot_display_id": "physical-two", "input_display_id": 2, "native": None})
        self.manager.adb.run.assert_not_called()

    def test_check_and_capture_require_actual_game_display(self) -> None:
        for command in ("check", "capture"):
            with self.subTest(command=command):
                code, output, error = self.invoke(command)
                self.assertEqual(code, 0, error)
                self.assertIn("Logical display: 2; physical display: physical-two", output)
                self.assertIn("2560x1440", output)
        self.assertEqual(self.resolver.call_count, 2)
        self.capture.capture_ui_xml_artifact.assert_not_called()
        self.manager.adb.run.assert_not_called()

    def test_absent_game_display_fails_without_implicit_launch_or_capture(self) -> None:
        self.resolver.side_effect = DeviceConnectionError("Game display unavailable; start the game")
        for command in ("check", "capture", "inspect"):
            with self.subTest(command=command):
                code, _, error = self.invoke(command)
                self.assertEqual(code, 1)
                self.assertIn("Game display unavailable", error)
        self.capture_class.assert_not_called()
        self.manager.adb.run.assert_not_called()

    def test_captures_with_identical_clock_never_reuse_old_path(self) -> None:
        with patch("autococ.cli.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 22, 12, 0, 0, 123456)
            self.assertEqual(self.invoke("capture")[0], 0)
            self.assertEqual(self.invoke("capture")[0], 0)
        paths = [call.args[0] for call in self.capture.capture_screenshot_artifact.call_args_list]
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertTrue(all(path.name.startswith("capture-20260922-120000-123456-") for path in paths))

    def test_sample_is_only_a_deprecated_capture_alias(self) -> None:
        runner = self.patch("autococ.cli.FlowRunner")
        code, output, error = self.invoke("sample")
        self.assertEqual(code, 0)
        self.assertIn("deprecated", error)
        self.assertIn("Screenshot:", output)
        self.assertNotIn("successes:", output)
        runner.assert_not_called()

    def test_once_overrides_cycle_limit_without_mutating_config(self) -> None:
        stats = RunStats()
        stats.record_task(TaskResult("collect", "skipped", "no collectibles"))
        runner = self.patch("autococ.cli.FlowRunner")
        runner.return_value.run_profile.return_value = stats
        code, _, _ = self.invoke("run", "--once", "--profile", "village-only")
        self.assertEqual(code, 0)
        self.assertEqual(runner.call_args.args[0].stop.max_runs, 1)
        self.assertEqual(self.config.stop.max_runs, 10)
        runner.return_value.run_profile.assert_called_once_with("village-only")
        self.resolver.assert_not_called()

    def test_dry_run_uses_real_planner_without_device_or_capture(self) -> None:
        code, output, error = self.invoke("run", "--dry-run", "--once", "--profile", "village-only")
        self.assertEqual(code, 0, error)
        self.assertIn("mode: dry-run", output)
        self.assertIn("successes: 0", output)
        self.assertIn("simulated: 4", output)
        self.manager_class.assert_not_called()
        self.capture_class.assert_not_called()
        payload = json.loads(next(self.config.runtime.report_dir.glob("run-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(payload["successes"], 0)

    def test_failed_and_empty_live_runs_return_nonzero(self) -> None:
        runner = self.patch("autococ.cli.FlowRunner")
        failed = RunStats()
        failed.record_task(TaskResult("initialization", "failed", "cannot connect game"))
        for stats in (failed, RunStats()):
            with self.subTest(failures=stats.failures):
                runner.return_value.run_profile.return_value = stats
                self.assertNotEqual(self.invoke("run")[0], 0)

    def test_real_success_or_confirmed_skip_can_return_zero(self) -> None:
        runner = self.patch("autococ.cli.FlowRunner")
        evidence = self.root / "verified.png"
        evidence.write_bytes(b"test-frame")
        for status in ("succeeded", "skipped"):
            with self.subTest(status=status):
                stats = RunStats()
                stats.record_task(TaskResult("collect", status, "verified", evidence=[evidence]))
                runner.return_value.run_profile.return_value = stats
                self.assertEqual(self.invoke("run")[0], 0)

    def test_user_interruption_returns_130(self) -> None:
        runner = self.patch("autococ.cli.FlowRunner")
        runner.return_value.run_profile.return_value = RunStats(stop_reason="interrupted by user")
        self.assertEqual(self.invoke("run")[0], 130)

    def test_image_argument_is_restricted_to_inspect(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["run", "--image", "example.png"])
        self.assertEqual(error.exception.code, 2)
        self.manager_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
