from dataclasses import replace
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from autococ.config import ProfileConfig, load_config
from autococ.errors import DeviceConnectionError, FlowError
from autococ.flow import FlowRunner, return_to_village
from autococ.recovery import recover_connection
from autococ.reporting import TaskResult
from autococ.scene import SceneSnapshot
from autococ.session import GameSession


class FlowTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        path = self.root / "config.toml"
        path.write_text("", encoding="utf-8")
        config = load_config(path)
        self.config = replace(
            config,
            runtime=replace(config.runtime, screenshot_dir=self.root / "screenshots", report_dir=self.root / "reports"),
            stop=replace(config.stop, max_runs=1),
        )
        self.frame = self.root / "observed.png"
        self.frame.write_bytes(b"test-frame")
        self.session = Mock(spec=GameSession)
        self.session.config = self.config
        self.session.events_path = self.root / "events.jsonl"
        self.session.last_snapshot = self.snapshot()
        self.session.observe.return_value = self.session.last_snapshot
        self.factory = Mock(return_value=self.session)
        self.adb = _FailingADB()
        self.logger = Mock(spec=logging.Logger)

    def snapshot(self, scene: str = "village", confidence: float = 0.95) -> SceneSnapshot:
        return SceneSnapshot(scene, confidence, self.frame, {
            "source_resolution": [2560, 1440],
            "buttons": [{"name": "attack", "point": [50, 670], "text": "Attack"},
                        {"name": "shop", "point": [1200, 670], "text": "Shop"}],
        })

    def runner(self, tasks: tuple[str, ...] | None = None, **changes) -> FlowRunner:
        config = replace(self.config, **changes)
        if tasks is not None:
            config = replace(config, profiles={"test": ProfileConfig(tasks)})
        self.session.config = config
        return FlowRunner(config, self.adb, "test-device", logger=self.logger, session_factory=self.factory)

    def succeeded(self, task: str) -> TaskResult:
        return TaskResult(task, "succeeded", "postcondition verified", evidence=[self.frame])

    def test_legacy_profile_keeps_physical_battle_counts_and_unique_receipts(self):
        runner = self.runner(("battle",), stop=replace(self.config.stop, max_runs=2))
        successful = replace(self.succeeded("battle"), metrics={
            "returned_home": True, "rounds_completed": 1, "victory": False})
        partial = replace(successful, status="failed", reason="partial deployment",
                          metrics={**successful.metrics, "victory": True})
        with patch("autococ.combat.run_battle", side_effect=[successful, partial]):
            stats = runner.run_profile("test")
        self.assertEqual((stats.battles_completed, stats.battles_won), (2, 1))
        receipts = [r for r in stats.task_results if r.task == "battle"]
        self.assertEqual(len({r.metrics["battle_id"] for r in receipts}), 2)
        self.assertTrue(all(r.metrics["receipt_kind"] == "battle" for r in receipts))
        self.assertEqual(self.report(stats)["battles_completed"], 2)
        self.assertEqual(stats.failures, 1)

    def test_legacy_task_success_and_search_return_are_not_completed_battles(self):
        for metrics in ({}, {"returned_home": True}, {"returned_home": True, "rounds_completed": True}):
            with self.subTest(metrics=metrics):
                runner = self.runner(("battle",))
                with patch("autococ.combat.run_battle", return_value=replace(self.succeeded("battle"), metrics=metrics)):
                    stats = runner.run_profile("test")
                self.assertEqual(stats.battles_completed, 0)

    def recovery_frames(self, *, retry_button: bool = True,
                        after_scene: str = "village") -> tuple[SceneSnapshot, SceneSnapshot]:
        disconnected = SceneSnapshot("disconnected", 0.95, self.root / "disconnected.png", {
            "buttons": [{"name": "retry", "point": [355, 418], "bbox": [335, 406, 375, 430],
                         "confidence": 0.99}] if retry_button else [],
        })
        after = self.snapshot(after_scene)
        frames = iter([disconnected, after])

        def observe(label="observe"):
            self.session.last_snapshot = next(frames)
            return self.session.last_snapshot

        self.session.observe.side_effect = observe
        self.session.buttons.side_effect = GameSession.buttons
        self.session.deadline = self.session.task_deadline = float("inf")
        return disconnected, after

    def report(self, stats) -> dict:
        path = self.config.runtime.report_dir / f"run-{stats.run_id}.json"
        self.assertTrue(path.with_suffix(".md").is_file())
        return json.loads(path.read_text(encoding="utf-8"))

    def test_dry_run_has_only_simulations_and_never_initializes_session(self) -> None:
        runner = self.runner(runtime=replace(self.config.runtime, dry_run=True))
        with patch.object(runner, "_handler") as handler:
            stats = runner.run_profile("village-only")
        self.assertEqual(stats.mode, "dry-run")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.attempts, 0)
        self.assertEqual(stats.cycles, 0)
        self.assertEqual(stats.simulated, len(self.config.profiles["village-only"].enabled_tasks))
        self.assertTrue(all(result.status == "simulated" for result in stats.task_results))
        handler.assert_not_called()
        self.factory.assert_not_called()
        self.assertEqual(self.report(stats)["successes"], 0)

    def test_live_worker_is_closed_after_task_failure(self) -> None:
        runner = self.runner(("launch",))
        self.session.observe.side_effect = FlowError("capture interrupted")
        stats = runner.run_profile("test")
        self.assertEqual(stats.failures, 1)
        self.session.close.assert_called_once_with()

    def test_stop_before_initialization_never_connects(self) -> None:
        runner = self.runner(("launch",))
        runner.stop_event = Event()
        runner.stop_event.set()
        stats = runner.run_profile("test")
        self.factory.assert_not_called()
        self.assertIn("interrupted", stats.stop_reason)
        self.assertEqual(self.report(stats)["successes"], 0)

    def test_stop_after_verified_task_preserves_receipt_and_skips_next_task(self) -> None:
        runner = self.runner(("launch", "collect"))
        runner.stop_event = Event()
        events = []

        def progress(event):
            events.append(event)
            if event["kind"] == "task_result":
                runner.stop_event.set()

        runner.progress = progress
        with patch("autococ.flow.collect_resources") as collect:
            stats = runner.run_profile("test")
        collect.assert_not_called()
        self.session.close.assert_called_once()
        self.assertEqual(stats.successes, 1)
        self.assertIn("interrupted", stats.stop_reason)
        self.assertIs(self.factory.call_args.kwargs["stop_event"], runner.stop_event)
        self.assertEqual([e["task"] for e in events if e["kind"] == "task_started"], ["launch"])


    def test_provenance_is_captured_once_before_initialization_and_retained_on_failure(self) -> None:
        runner = self.runner(("launch",))
        provenance = {"source_fingerprint_sha256": "run-start-fingerprint"}
        with patch("autococ.flow.capture_run_provenance", return_value=provenance) as capture:
            def fail_connect(*args, **kwargs):
                capture.assert_called_once_with(runner.config, "test")
                raise DeviceConnectionError("no game display")

            self.factory.side_effect = fail_connect
            stats = runner.run_profile("test")
            capture.assert_called_once()
        self.assertEqual(self.report(stats)["provenance"], provenance)

    def test_dry_run_provenance_uses_effective_config_and_selected_profile(self) -> None:
        runner = self.runner(("launch", "collect"), runtime=replace(self.config.runtime, dry_run=True))
        stats = runner.run_profile("test")
        provenance = self.report(stats)["provenance"]
        self.assertEqual(provenance["profile"], "test")
        self.assertEqual(provenance["tasks"], ["launch", "collect"])
        self.assertEqual(len(provenance["effective_config_sha256"]), 64)
        self.assertIsNotNone(provenance["source_fingerprint_sha256"])
        self.factory.assert_not_called()

    def test_village_profile_never_resolves_or_invokes_battle_handler(self) -> None:
        runner = self.runner()
        names = []
        called = []

        def resolve(task: str):
            names.append(task)

            def perform(session):
                called.append(task)
                return TaskResult(task, "skipped", "confirmed no opportunity")

            return perform

        with patch.object(runner, "_handler", side_effect=resolve):
            stats = runner.run_profile("village-only")
        expected = list(self.config.profiles["village-only"].enabled_tasks)
        self.assertEqual(names, expected)
        self.assertEqual(called, expected)
        self.assertNotIn("battle", names)
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.skipped, len(expected))
        self.assertEqual(stats.cycles, 1)

    def test_profile_cycles_are_distinct_from_task_successes(self) -> None:
        runner = self.runner(("launch", "collect"), stop=replace(self.config.stop, max_runs=2))
        handlers = {name: Mock(side_effect=lambda session, name=name: self.succeeded(name)) for name in ("launch", "collect")}
        with patch.object(runner, "_handler", side_effect=handlers.__getitem__):
            stats = runner.run_profile("test")
        self.assertEqual(stats.cycles, 2)
        self.assertEqual(stats.attempts, 4)
        self.assertEqual(stats.successes, 4)
        self.assertEqual(stats.failures, 0)
        self.assertEqual([handler.call_count for handler in handlers.values()], [2, 2])
        self.assertTrue(all(result.elapsed_sec >= 0 for result in stats.task_results))
        self.assertIn("max_runs", stats.stop_reason)

    def test_failure_receipt_stops_following_tasks_and_preserves_prior_success(self) -> None:
        runner = self.runner(("launch", "collect", "battle"))
        handlers = {
            "launch": Mock(return_value=self.succeeded("launch")),
            "collect": Mock(return_value=TaskResult("collect", "failed", "inventory unverified", evidence=[self.frame])),
            "battle": Mock(return_value=self.succeeded("battle")),
        }
        with patch.object(runner, "_handler", side_effect=handlers.__getitem__):
            stats = runner.run_profile("test")
        self.assertEqual((stats.attempts, stats.successes, stats.failures, stats.cycles), (2, 1, 1, 1))
        handlers["battle"].assert_not_called()
        self.assertIn(self.frame, stats.failure_screenshots)
        payload = self.report(stats)
        self.assertEqual(payload["task_results"][-1]["reason"], "inventory unverified")
        self.assertEqual(payload["successes"], 1)
        self.assertEqual(payload["failures"], 1)

    def test_handler_exception_becomes_failed_task_receipt_with_evidence(self) -> None:
        runner = self.runner(("collect",))
        with patch.object(runner, "_handler", return_value=Mock(side_effect=FlowError("capture failed"))):
            stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 1)
        self.assertEqual(stats.task_results[0].task, "collect")
        self.assertEqual(stats.task_results[0].evidence, [self.frame])
        self.assertIn("capture failed", self.report(stats)["stop_reason"])

    def test_no_receipt_is_a_failure_not_implicit_success(self) -> None:
        runner = self.runner(("collect",))
        with patch.object(runner, "_handler", return_value=Mock(return_value=None)):
            stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 1)
        self.assertEqual(stats.task_results[0].status, "failed")

    def test_unknown_and_unimplemented_task_fail_before_session_connection(self) -> None:
        for task in ("nonexistent", "settle"):
            with self.subTest(task=task):
                runner = self.runner((task,))
                stats = runner.run_profile("test")
                self.assertEqual(stats.successes, 0)
                self.assertEqual(stats.failures, 1)
                self.assertEqual(stats.cycles, 0)
                self.assertEqual(self.report(stats)["successes"], 0)
        self.factory.assert_not_called()

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(FlowError, "Unknown profile"):
            self.runner().run_profile("missing")
        self.factory.assert_not_called()

    def test_initialization_failure_has_zero_success_and_a_report(self) -> None:
        runner = self.runner(("launch",))
        self.factory.side_effect = DeviceConnectionError("game display unavailable")
        stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 1)
        self.assertEqual(stats.cycles, 0)
        self.assertEqual(stats.task_results[0].task, "initialization")
        self.assertIn("game display unavailable", self.report(stats)["stop_reason"])

    def test_collection_encountering_enemy_village_does_not_act_or_attack(self) -> None:
        runner = self.runner(("collect",))
        self.session.last_snapshot = self.snapshot("enemy_village")
        self.session.observe.return_value = self.session.last_snapshot
        stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 1)
        self.session.tap.assert_not_called()
        self.session.click.assert_not_called()
        self.session.back.assert_not_called()
        self.assertEqual([result.task for result in stats.task_results], ["collect"])

    def test_unknown_launch_scene_does_not_trigger_blind_recovery(self) -> None:
        runner = self.runner(("launch",))
        self.session.last_snapshot = self.snapshot("unknown", 0.0)
        self.session.observe.return_value = self.session.last_snapshot
        self.session.wait_for.side_effect = FlowError("village recognition timed out")
        stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 1)
        self.session.back.assert_not_called()
        self.session.tap.assert_not_called()
        self.session.click.assert_not_called()

    def test_explicit_recover_uses_single_retry_and_requires_verified_village(self) -> None:
        runner = self.runner(("recover",))
        disconnected, village = self.recovery_frames()
        self.assertIs(runner._handler("recover"), recover_connection)
        stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("recover", "succeeded")])
        result = stats.task_results[0]
        self.assertEqual(result.evidence, [disconnected.screenshot_path, village.screenshot_path])
        self.assertEqual(result.metrics["retry_actions"], 1)
        self.assertTrue(result.metrics["village_verified"])
        self.session.click.assert_called_once_with(disconnected, "retry", region=(300, 270, 980, 460))
        self.session.back.assert_not_called()
        self.assertFalse(self.factory.call_args.kwargs["launch"])

    def test_disconnected_launch_recovers_once_and_keeps_startup_evidence(self) -> None:
        runner = self.runner(("launch",))
        disconnected, village = self.recovery_frames()
        stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("launch", "succeeded")])
        result = stats.task_results[0]
        self.assertTrue(result.metrics["startup_recovery"])
        self.assertEqual(result.metrics["retry_actions"], 1)
        self.assertEqual(result.evidence, [disconnected.screenshot_path, village.screenshot_path])
        self.assertEqual(self.session.observe.call_count, 2)
        self.session.click.assert_called_once()
        self.session.wait_for.assert_not_called()
        self.assertEqual(self.report(stats)["task_results"][0]["metrics"]["retry_actions"], 1)

    def test_disconnected_launch_without_retry_evidence_fails_before_next_task(self) -> None:
        runner = self.runner(("launch", "battle"))
        disconnected, _ = self.recovery_frames(retry_button=False)
        with patch("autococ.combat.run_battle") as battle:
            stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("launch", "failed")])
        result = stats.task_results[0]
        self.assertEqual(result.metrics["retry_actions"], 0)
        self.assertEqual(result.evidence, [disconnected.screenshot_path])
        self.session.click.assert_not_called()
        self.session.back.assert_not_called()
        battle.assert_not_called()
        self.assertEqual(self.report(stats)["failures"], 1)

    def test_failed_battle_is_neither_recovered_nor_replayed_automatically(self) -> None:
        runner = self.runner(("battle", "recover", "battle"), stop=replace(self.config.stop, max_runs=3))
        failure = TaskResult("battle", "failed", "Game interruption: disconnected", evidence=[self.frame])
        with patch("autococ.combat.run_battle", return_value=failure) as battle:
            with patch("autococ.flow.recover_connection") as recovery:
                stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("battle", "failed")])
        battle.assert_called_once_with(self.session)
        recovery.assert_not_called()
        self.assertEqual(stats.failures, 1)
        self.assertEqual(stats.cycles, 1)
        self.assertIn("disconnected", self.report(stats)["task_results"][0]["reason"])

    def test_failed_startup_recovery_retains_retry_and_after_frame_evidence(self) -> None:
        runner = self.runner(("launch", "battle"))
        disconnected, maintenance = self.recovery_frames(after_scene="maintenance")
        with patch("autococ.combat.run_battle") as battle:
            stats = runner.run_profile("test")
        result = stats.task_results[0]
        self.assertEqual((result.task, result.status), ("launch", "failed"))
        self.assertIn("maintenance", result.reason)
        self.assertEqual(result.metrics["retry_actions"], 1)
        self.assertFalse(result.metrics["village_verified"])
        self.assertEqual(result.evidence, [disconnected.screenshot_path, maintenance.screenshot_path])
        self.session.click.assert_called_once()
        battle.assert_not_called()
        payload = self.report(stats)
        self.assertEqual(payload["failures"], 1)
        self.assertEqual(payload["successes"], 0)
        self.assertEqual(payload["task_results"][0]["metrics"]["retry_actions"], 1)

    def test_return_to_village_for_other_tasks_never_implicitly_retries_connection(self) -> None:
        self.session.observe.return_value = self.snapshot("disconnected")
        self.session.wait_for.side_effect = FlowError("Game interruption: disconnected")
        with patch("autococ.flow.recover_connection") as recovery:
            with self.assertRaisesRegex(FlowError, "disconnected"):
                return_to_village(self.session)
        recovery.assert_not_called()
        self.session.click.assert_not_called()
        self.session.back.assert_not_called()

    def test_low_confidence_village_is_not_verified_launch_success(self) -> None:
        runner = self.runner(("launch",))
        self.session.last_snapshot = self.snapshot("village", 0.2)
        self.session.observe.return_value = self.session.last_snapshot
        self.session.wait_for.side_effect = FlowError("village recognition timed out")
        stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 0)
        self.assertEqual(stats.failures, 1)

    def test_success_screenshot_preference_preserves_verified_evidence(self) -> None:
        runner = self.runner(("collect",), reporting=replace(self.config.reporting, save_success_screenshots=True))
        with patch.object(runner, "_handler", return_value=Mock(return_value=self.succeeded("collect"))):
            stats = runner.run_profile("test")
        self.assertEqual(stats.success_screenshots, [self.frame])
        self.assertEqual(stats.events_path, self.session.events_path)
        self.assertEqual(stats.screenshot_resolution, (2560, 1440))

    def test_return_to_village_exits_request_then_chat(self) -> None:
        request, chat, village = self.snapshot("request"), self.snapshot("clan_chat"), self.snapshot()
        self.session.observe.return_value = request
        self.session.wait_for.side_effect = [chat, village]
        self.assertIs(return_to_village(self.session), village)
        self.assertEqual([call.args[0] for call in self.session.back.call_args_list], [request, chat])
        targets = [call.args[0] for call in self.session.wait_for.call_args_list]
        self.assertNotIn("request", targets[0])
        self.assertIn("clan_chat", targets[0])
        self.assertNotIn("clan_chat", targets[1])

    def test_recovery_does_not_repeat_back_without_scene_progress(self) -> None:
        self.session.observe.return_value = self.snapshot("request")
        self.session.wait_for.side_effect = FlowError("no scene change")
        with self.assertRaises(FlowError):
            return_to_village(self.session)
        self.session.back.assert_called_once()

    def test_recovery_cannot_loop_between_known_dialogs_forever(self) -> None:
        request, chat = self.snapshot("request"), self.snapshot("clan_chat")
        self.session.observe.return_value = request
        self.session.wait_for.side_effect = [chat, request, chat, request]
        with self.assertRaisesRegex(FlowError, "four"):
            return_to_village(self.session)
        self.assertEqual(self.session.back.call_count, 4)

    def test_begin_task_failure_is_attributed_to_selected_task(self) -> None:
        runner = self.runner(("collect",))
        handler = Mock(return_value=self.succeeded("collect"))
        self.session.begin_task.side_effect = FlowError("session deadline exceeded")
        with patch.object(runner, "_handler", return_value=handler):
            stats = runner.run_profile("test")
        self.assertEqual([result.task for result in stats.task_results], ["collect"])
        self.assertEqual(stats.failures, 1)
        handler.assert_not_called()
        self.assertNotIn("initialization", self.report(stats)["stop_reason"])

    def test_missing_resolution_does_not_overwrite_completed_receipt(self) -> None:
        runner = self.runner(("collect",))
        self.session.last_snapshot = SceneSnapshot("village", 0.95, self.frame)
        with patch.object(runner, "_handler", return_value=Mock(return_value=self.succeeded("collect"))):
            stats = runner.run_profile("test")
        self.assertEqual(stats.successes, 1)
        self.assertEqual(stats.failures, 0)
        self.assertIsNone(stats.screenshot_resolution)
        self.assertEqual(len(self.report(stats)["task_results"]), 1)

    def test_interrupt_preserves_previous_success_and_current_task_evidence(self) -> None:
        runner = self.runner(("launch", "collect", "battle"))
        handlers = {
            "launch": Mock(return_value=self.succeeded("launch")),
            "collect": Mock(side_effect=KeyboardInterrupt),
            "battle": Mock(return_value=self.succeeded("battle")),
        }
        with patch.object(runner, "_handler", side_effect=handlers.__getitem__):
            stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("launch", "succeeded"), ("collect", "failed")])
        self.assertTrue(stats.task_results[1].metrics["interrupted"])
        self.assertEqual(stats.task_results[1].evidence, [self.frame])
        handlers["battle"].assert_not_called()
        self.assertEqual(self.report(stats)["stop_reason"], "interrupted by user")

    def test_event_log_failure_preserves_completed_task_and_stops_next_task(self) -> None:
        runner = self.runner(("collect", "battle"))
        handlers = {task: Mock(return_value=self.succeeded(task)) for task in ("collect", "battle")}
        self.session.event.side_effect = OSError("disk write failed")
        with patch.object(runner, "_handler", side_effect=handlers.__getitem__):
            stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("collect", "succeeded"), ("event_log", "failed")])
        handlers["battle"].assert_not_called()
        self.assertEqual(self.report(stats)["successes"], 1)
        self.assertEqual(stats.failures, 1)

    def test_report_failure_does_not_mask_or_discard_task_results(self) -> None:
        runner = self.runner(("collect",))
        with patch.object(runner, "_handler", return_value=Mock(return_value=self.succeeded("collect"))):
            with patch("autococ.flow.write_report", side_effect=OSError("report directory unavailable")):
                stats = runner.run_profile("test")
        self.assertEqual([(result.task, result.status) for result in stats.task_results], [("collect", "succeeded"), ("report", "failed")])
        self.assertIn("report failed", stats.stop_reason)
        self.assertEqual(stats.successes, 1)
        self.assertEqual(stats.failures, 1)


class _FailingADB:
    def run(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("Unit tests must not run ADB commands")

    def run_bytes(self, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("Unit tests must not run ADB commands")


if __name__ == "__main__":
    unittest.main()
