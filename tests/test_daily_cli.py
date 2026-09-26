"""Daily CLI configuration and dry-run integration without an emulator."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from autococ.cli import main
from autococ.reporting import RunStats, TaskResult
from autococ.routine_config import GoalConfig, ResourceFilter, RoutineConfig, TaskSpec, routine_to_dict


STRATEGY = ('id = "single_edge"\nlabel = "单边"\narmy_mode = "captured"\n'
            '[[steps]]\naction = "deploy_troop"\nunit_id = "*"\ncount = "all"\nedge = "first"\n')
ADAPTER = ('id = "event-x"\nkind = "event"\ntitle = "活动"\nprogress_roi = [0, 0, 300, 300]\n'
           'entry_text = "活动"\nactive_label = "进行中"\nscoring_condition = "battle"\n')


class DailyCLITests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / "config.toml"
        self.config_path.write_text(f'[runtime]\nreport_dir = "{(self.root / "reports").as_posix()}"\n', encoding="utf-8")
        self.routine_dir = self.root / "separate-routine"
        self.routine_dir.mkdir()
        (self.routine_dir / "single_edge.toml").write_text(STRATEGY, encoding="utf-8")
        (self.routine_dir / "event.toml").write_text(ADAPTER, encoding="utf-8")

    def invoke(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(["run", "--config", str(self.config_path), *args])
        return code, output.getvalue(), error.getvalue()

    def test_toml_routine_resolves_its_own_files_and_runs_offline_once(self):
        routine_path = self.routine_dir / "routine.toml"
        routine_path.write_text(
            '[routine]\nmaintenance_interval_sec = 600\n'
            '[[routine.tasks]]\nid = "farm"\nkind = "resources"\nmax_battles = 6\n'
            'strategy_file = "single_edge.toml"\n'
            '[routine.tasks.resource_filter]\nenabled = true\nmin_gold = 500000\nmin_total = 700000\n'
            '[routine.tasks.goal.resource_targets]\ngold = 20000000\n'
            '[[routine.tasks]]\nid = "event_run"\nkind = "event"\nmax_battles = 3\n'
            '[routine.tasks.goal]\nadapter_path = "event.toml"\ntarget = 25\n', encoding="utf-8")
        with patch("autococ.cli.DeviceManager") as device, patch("autococ.cli.setup_logging", return_value=logging.getLogger("test.daily.cli")):
            code, output, error = self.invoke("--routine", str(routine_path), "--once", "--dry-run")
        self.assertEqual(code, 0, error)
        device.assert_not_called()
        self.assertIn("mode: dry-run", output)
        report = json.loads(next((self.root / "reports").glob("run-*.json")).read_text(encoding="utf-8"))
        self.assertEqual([result["task"] for result in report["task_results"]], ["farm", "event_run"])
        self.assertEqual(report["simulated"], 2)
        self.assertEqual(report["battles_completed"], 0)

    def test_inline_config_routine_resolves_strategy_against_config_directory(self):
        (self.root / "single_edge.toml").write_text(STRATEGY, encoding="utf-8")
        self.config_path.write_text(
            f'[runtime]\nreport_dir = "{(self.root / "reports").as_posix()}"\n'
            '[routine]\n[[routine.tasks]]\nid = "farm"\nkind = "resources"\n'
            'strategy_file = "single_edge.toml"\n', encoding="utf-8")
        with patch("autococ.cli.DeviceManager") as device, patch("autococ.cli.setup_logging", return_value=logging.getLogger("test.daily.cli")):
            code, output, error = self.invoke("--dry-run")
        self.assertEqual(code, 0, error)
        device.assert_not_called()
        self.assertIn("simulated: 1", output)

    def test_json_routine_override_preserves_independent_goals_and_limits(self):
        routine = RoutineConfig((
            TaskSpec("farm", "resources", strategy_file="single_edge.toml", max_battles=8,
                     resource_filter=ResourceFilter(min_gold=400000, min_total=None),
                     goal=GoalConfig(resource_targets={"gold": 24000000})),
            TaskSpec("event_run", "event", max_battles=6,
                     goal=GoalConfig(adapter_path="event.toml", target=50))),
            maintenance_interval_sec=600)
        routine_path = self.routine_dir / "routine.json"
        routine_path.write_text(json.dumps({"routine": routine_to_dict(routine)}), encoding="utf-8")
        override = self.routine_dir / "single_edge.toml"
        fake = RunStats(profile="daily", mode="dry-run", stop_reason="dry-run planning complete")
        with patch("autococ.cli.DeviceManager") as device, patch("autococ.cli.FlowRunner") as runner, \
             patch("autococ.cli.setup_logging", return_value=logging.getLogger("test.daily.cli")):
            runner.return_value.run_routine.return_value = fake
            code, _, error = self.invoke("--routine", str(routine_path), "--strategy-file", str(override),
                                         "--once", "--dry-run")
            passed = runner.call_args.args[0].routine
        self.assertEqual(code, 0, error)
        device.assert_not_called()
        by_id = {task.id: task for task in passed.tasks}
        self.assertEqual((by_id["farm"].max_battles, by_id["event_run"].max_battles), (1, 1))
        self.assertEqual(by_id["farm"].goal.resource_targets, {"gold": 24000000})
        self.assertEqual(by_id["farm"].resource_filter.min_gold, 400000)
        self.assertEqual(by_id["event_run"].goal.target, 50)
        self.assertEqual(Path(by_id["event_run"].goal.adapter_path), self.routine_dir / "event.toml")
        self.assertEqual(Path(by_id["farm"].strategy_file), override)
        self.assertEqual(Path(by_id["event_run"].strategy_file), override)

    def test_invalid_relative_strategy_fails_before_device_contact(self):
        routine_path = self.routine_dir / "routine.json"
        routine = RoutineConfig((TaskSpec("farm", "resources", strategy_file="missing.toml"),))
        routine_path.write_text(json.dumps(routine_to_dict(routine)), encoding="utf-8")
        with patch("autococ.cli.DeviceManager") as device, patch("autococ.cli.setup_logging") as logger:
            code, _, error = self.invoke("--routine", str(routine_path), "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("missing.toml", error)
        device.assert_not_called()
        logger.assert_not_called()

    def test_unsupported_goal_returns_nonzero_even_without_failure_counter(self):
        routine_path = self.routine_dir / "routine.json"
        routine_path.write_text(json.dumps(routine_to_dict(RoutineConfig((TaskSpec("farm", "resources"),)))), encoding="utf-8")
        result = RunStats(profile="daily", mode="dry-run", stop_reason="unsupported")
        result.record_task(TaskResult("farm", "not_supported", "missing calibrated adapter"))
        with patch("autococ.cli.DeviceManager") as device, patch("autococ.cli.FlowRunner") as runner, \
             patch("autococ.cli.setup_logging", return_value=logging.getLogger("test.daily.cli")):
            runner.return_value.run_routine.return_value = result
            code, _, _ = self.invoke("--routine", str(routine_path), "--dry-run")
        self.assertEqual(code, 1)
        device.assert_not_called()

    def test_all_disabled_or_malformed_routine_fails_before_connection(self):
        path = self.routine_dir / "invalid.json"
        for data in ([], routine_to_dict(RoutineConfig((TaskSpec("farm", "resources", enabled=False),)))):
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch("autococ.cli.DeviceManager") as device:
                code, _, error = self.invoke("--routine", str(path))
            self.assertEqual(code, 1, error)
            device.assert_not_called()

    def test_legacy_strategy_override_clears_configured_custom_strategy(self):
        self.config_path.write_text('[battle]\nstrategy_file="missing.toml"\n', encoding="utf-8")
        fake = RunStats(profile="core-loop", mode="dry-run")
        with patch("autococ.cli.FlowRunner") as runner, patch("autococ.cli.DeviceManager") as device:
            runner.return_value.run_profile.return_value = fake
            code, _, error = self.invoke("--strategy", "two_edge", "--dry-run")
        self.assertEqual(code, 0, error)
        self.assertEqual(runner.call_args.args[0].battle.strategy_file, "")
        device.assert_not_called()

    def test_capabilities_distinguishes_sample_version_surfaces_and_missing_coverage(self):
        output = io.StringIO()
        manifest = {"client": "test-client", "version_code": 123, "templates": [
            {"unit_id": "archer", "surface": "army", "state": "active"},
            {"unit_id": "archer", "surface": "battle", "state": "active"}]}
        with redirect_stdout(output), patch("autococ.cli.DeviceManager") as device, \
             patch("autococ.unit_catalog.template_manifest", return_value=manifest), \
             patch("autococ.unit_catalog.coverage", return_value={
                 "archer": {"recognition": "sampled", "templates": 2},
                 "giant": {"recognition": "unavailable", "templates": 0}}):
            code = main(["capabilities", "--config", str(self.config_path)])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["sample_client_version"], "test-client")
        self.assertEqual(result["units"]["archer"]["sampled_surfaces"], ["army", "battle"])
        self.assertEqual(result["units"]["giant"]["recognition"], "unavailable")
        self.assertEqual(result["units"]["giant"]["sampled_states"], [])
        device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
