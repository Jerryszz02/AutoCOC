from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from autococ.config import load_config
from autococ.desktop import (DesktopController, RunOptions, desktop_config, load_options,
                             report_history, report_text, save_options)
from autococ.errors import ConfigError
from autococ.reporting import RunStats


class DesktopTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "config.toml"
        self.path.write_text("", encoding="utf-8")
        self.config = load_config(self.path)
        self.options = RunOptions.from_config(self.config)

    def test_options_reject_invalid_tasks_limits_and_modes(self):
        for changes in ({"tasks": ()}, {"tasks": ("battle",)}, {"tasks": ("launch", "launch")},
                        {"tasks": ("launch", "settle")}, {"max_runs": 0}, {"max_duration_sec": True},
                        {"min_expected_resources": -1}, {"max_searches": 0}, {"dry_run": "false"}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                replace(self.options, **changes).validate()

    def test_gui_overrides_preserve_config_and_anchor_output_paths(self):
        before = self.path.read_bytes()
        config = desktop_config(self.path, replace(self.options, tasks=("launch", "battle"), max_runs=2))
        self.assertTrue(config.runtime.dry_run)
        self.assertEqual(config.runtime.report_dir, self.root / "reports")
        self.assertEqual(config.vision.template_dir, self.root / "assets" / "templates")
        self.assertEqual(config.profiles["desktop"].enabled_tasks, ("launch", "battle"))
        self.assertEqual(config.stop.max_runs, 2)
        self.assertTrue(config.reporting.write_markdown)
        self.assertEqual(self.path.read_bytes(), before)

    def test_saved_options_round_trip_but_never_restore_live_mode(self):
        path = self.root / "desktop.json"
        options = replace(self.options, dry_run=False, max_runs=7, serial="127.0.0.1:1234")
        save_options(path, options)
        self.assertEqual(load_options(path, self.config), replace(options, dry_run=True))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dry_run"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertTrue(load_options(path, self.config).dry_run)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_malformed_settings_do_not_silently_enable_defaults(self):
        path = self.root / "desktop.json"
        for text in ("{", "[]", '{"tasks": ["launch"]}'):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ConfigError):
                load_options(path, self.config)

    def test_dry_run_worker_never_constructs_device_or_session(self):
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as manager, patch("autococ.flow.GameSession.connect") as connect:
            controller.start(self.path, self.options)
            controller.thread.join(5)
            self.assertFalse(controller.running)
            manager.assert_not_called()
            connect.assert_not_called()
        events = []
        while not controller.events.empty():
            events.append(controller.events.get_nowait())
        final = next(e for e in events if e["kind"] == "finished")
        self.assertEqual(final["summary"]["mode"], "dry-run")
        self.assertEqual(final["summary"]["successes"], 0)
        self.assertEqual(final["summary"]["simulated"], len(self.options.tasks))
        self.assertTrue(final["report"].is_file())
        reports, errors = report_history(self.root / "reports")
        self.assertEqual((len(reports), errors), (1, []))

    def test_double_start_rejected_and_stop_reaches_worker(self):
        controller = DesktopController()
        entered = Event()

        def run(profile):
            entered.set()
            controller.stop_event.wait(3)
            return RunStats(profile=profile, mode="dry-run", stop_reason="interrupted by user")

        with patch("autococ.desktop.FlowRunner") as runner:
            runner.return_value.run_profile.side_effect = run
            controller.start(self.path, self.options)
            self.assertTrue(entered.wait(2))
            with self.assertRaises(RuntimeError):
                controller.start(self.path, self.options)
            controller.stop()
            controller.thread.join(3)
            self.assertIs(runner.call_args.kwargs["stop_event"], controller.stop_event)
        self.assertFalse(controller.running)

    def test_connection_failure_is_reported_instead_of_hanging_ui(self):
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as manager:
            manager.return_value.connect.side_effect = OSError("test device unavailable")
            controller.start(self.path, replace(self.options, dry_run=False))
            controller.thread.join(3)
        self.assertFalse(controller.running)
        reports, _ = report_history(self.root / "reports")
        self.assertEqual(reports[0][1]["failures"], 1)
        self.assertIn("test device unavailable", reports[0][1]["stop_reason"])

    def test_corrupt_history_is_skipped_and_unknown_income_stays_unknown(self):
        good = {"mode": "live", "task_results": [], "resource_metrics": {"net_gold_elixir_per_hour": None}}
        (self.root / "run-1.json").write_text(json.dumps(good), encoding="utf-8")
        (self.root / "run-2.json").write_text('{"mode":"live","task_results":[1]}', encoding="utf-8")
        (self.root / "run-3.json").write_text("{", encoding="utf-8")
        reports, errors = report_history(self.root)
        self.assertEqual((len(reports), len(errors)), (1, 2))
        self.assertIn("经营净产出 / 小时：未知", report_text(good))
        self.assertIn("不适用（离线预演）", report_text({**good, "mode": "dry-run"}))
