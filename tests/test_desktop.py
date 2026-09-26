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
from autococ.routine_config import GoalConfig, ResourceFilter, RoutineConfig, TaskSpec


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
        loaded = load_options(path, self.config)
        self.assertEqual(replace(loaded, routine=None), replace(options, dry_run=True))
        self.assertIsNotNone(loaded.routine)
        self.assertEqual(next(task for task in loaded.routine.tasks if task.id == "resources").max_battles, 7)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dry_run"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertTrue(load_options(path, self.config).dry_run)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_routine_settings_round_trip_keeps_order_and_independent_goals(self):
        routine = RoutineConfig((
            TaskSpec("event", "event", goal=GoalConfig(adapter_path="events/current.toml", target=30),
                     resource_filter=ResourceFilter(enabled=False)),
            TaskSpec("resources", "resources", strategy="edrag_line",
                     resource_filter=ResourceFilter(min_gold=600000, min_total=None),
                     goal=GoalConfig(resource_targets={"gold": 24000000})),
            TaskSpec("clan_games", "clan_games", goal=GoalConfig(building_type="spell_factory", target=4))),
            maintenance_interval_sec=600)
        options = replace(self.options, routine=routine, dry_run=False)
        path = self.root / "desktop.json"
        save_options(path, options)
        loaded = load_options(path, self.config)
        self.assertEqual(loaded.routine, routine)
        self.assertTrue(loaded.dry_run)
        self.assertEqual(desktop_config(self.path, loaded).routine, routine)

    def test_routine_worker_uses_routine_entrypoint(self):
        routine = RoutineConfig((TaskSpec("resources", "resources"),))
        controller = DesktopController()
        stats = RunStats(profile="routine", mode="dry-run", stop_reason="completed")
        with patch("autococ.desktop.FlowRunner") as runner:
            runner.return_value.run_routine.return_value = stats
            controller.start(self.path, replace(self.options, routine=routine))
            controller.thread.join(3)
            runner.return_value.run_routine.assert_called_once_with(routine)
            runner.return_value.run_profile.assert_not_called()

    def test_multi_goal_dry_run_reports_order_without_contacting_device(self):
        routine = RoutineConfig((TaskSpec("collect", "collect"), TaskSpec("resources", "resources"),
                                 TaskSpec("event", "event"), TaskSpec("clan_games", "clan_games")))
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as device:
            controller.start(self.path, replace(self.options, routine=routine))
            controller.thread.join(5)
            device.assert_not_called()
        events = []
        while not controller.events.empty():
            events.append(controller.events.get_nowait())
        finished = next(event for event in events if event["kind"] == "finished")
        self.assertEqual([result["task"] for result in finished["summary"]["task_results"]],
                         ["collect", "resources", "event", "clan_games"])
        self.assertEqual(finished["summary"]["simulated"], 4)
        self.assertEqual(finished["summary"]["battles_completed"], 0)

    def test_missing_routine_file_fails_before_worker_or_device_connection(self):
        routine = RoutineConfig((TaskSpec("resources", "resources", strategy_file="missing.toml"),))
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as device, self.assertRaises(ConfigError):
            controller.start(self.path, replace(self.options, routine=routine, dry_run=False))
        self.assertIsNone(controller.thread)
        device.assert_not_called()

    def test_empty_selection_fails_before_live_connection(self):
        routine = RoutineConfig((TaskSpec("resources", "resources", enabled=False),))
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as device, self.assertRaises(ConfigError):
            controller.start(self.path, replace(self.options, routine=routine, dry_run=False))
        self.assertIsNone(controller.thread)
        device.assert_not_called()

    def test_relative_strategy_file_uses_selected_config_directory(self):
        (self.root / "battle.toml").write_text(
            'id = "single_edge"\nlabel = "单边"\narmy_mode = "captured"\n'
            '[[steps]]\naction = "deploy_troop"\nunit_id = "*"\ncount = "all"\nedge = "first"\n',
            encoding="utf-8")
        routine = RoutineConfig((TaskSpec("resources", "resources", strategy_file="battle.toml"),))
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as device:
            controller.start(self.path, replace(self.options, routine=routine))
            controller.thread.join(5)
            device.assert_not_called()
        self.assertFalse(controller.running)
        reports, errors = report_history(self.root / "reports")
        self.assertEqual(errors, [])
        self.assertEqual(reports[0][1]["simulated"], 1)

    def test_two_edge_desktop_configuration_and_saved_selection(self):
        options = replace(self.options, strategy="two_edge", tasks=("launch", "battle"))
        self.assertEqual(desktop_config(self.path, options).battle.strategy, "two_edge")
        path = self.root / "desktop.json"
        save_options(path, options)
        self.assertEqual(load_options(path, self.config).strategy, "two_edge")
        resource = next(task for task in load_options(path, self.config).routine.tasks if task.kind == "resources")
        self.assertFalse(resource.resource_filter.enabled)

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

    def test_cancelled_connection_is_not_reported_as_failure(self):
        from autococ.errors import StopRequested
        controller = DesktopController()
        with patch("autococ.desktop.DeviceManager") as manager, patch("autococ.desktop.FlowRunner") as runner:
            manager.return_value.connect.side_effect = StopRequested()
            controller.start(self.path, replace(self.options, dry_run=False))
            controller.thread.join(3)
            runner.assert_not_called()
        reports, _ = report_history(self.root / "reports")
        self.assertEqual(reports[0][1]["failures"], 0)
        self.assertEqual(reports[0][1]["cancelled"], 1)

    def test_corrupt_history_is_skipped_and_unknown_income_stays_unknown(self):
        good = {"mode": "live", "task_results": [], "resource_metrics": {"net_gold_elixir_per_hour": None}}
        (self.root / "run-1.json").write_text(json.dumps(good), encoding="utf-8")
        (self.root / "run-2.json").write_text('{"mode":"live","task_results":[1]}', encoding="utf-8")
        (self.root / "run-3.json").write_text("{", encoding="utf-8")
        reports, errors = report_history(self.root)
        self.assertEqual((len(reports), len(errors)), (1, 2))
        self.assertIn("经营净产出 / 小时：未知", report_text(good))
        self.assertIn("不适用（离线预演）", report_text({**good, "mode": "dry-run"}))
