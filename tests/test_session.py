from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from autococ.capture import CaptureClient, ScreenshotCapture
from autococ.adb import ADBResult
from autococ.config import MuMuConfig, load_config
from autococ.device import DisplayTarget
from autococ.errors import AdbError, CaptureError, DeviceConnectionError, FlowError, MuMuDisplayUnavailable, StopRequested
from autococ.evidence import EvidenceBudget
from autococ.locator import Bounds, LocatorResult
from autococ.scene import SceneSnapshot
from autococ.session import GameSession


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        config_path = self.root / "config.toml"
        config_path.write_text("", encoding="utf-8")
        config = load_config(config_path)
        self.config = replace(config, runtime=replace(config.runtime, poll_interval_sec=0.25))
        self.clock = Clock()
        clock_patch = patch("autococ.session.time.monotonic", side_effect=lambda: self.clock.now)
        sleep_patch = patch("autococ.session.time.sleep", side_effect=self.clock.advance)
        clock_patch.start()
        sleep_patch.start()
        self.addCleanup(clock_patch.stop)
        self.addCleanup(sleep_patch.stop)
        foreground_patch = patch("autococ.session.foreground_package", return_value="unknown")
        self.foreground_package = foreground_patch.start()
        self.addCleanup(foreground_patch.stop)
        self.adb = Mock()
        self.capture = Mock(spec=CaptureClient)
        self.capture.adb = self.adb
        self.capture.serial = "test-device"
        self.capture.input_display_id = 2
        self.capture.capture_screenshot_artifact.side_effect = self.capture_frame
        self.recognizer = Mock()
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("village", 0.95, Path(path))
        self.session = GameSession(self.config, self.capture, self.recognizer, self.root / "run")

    def capture_frame(self, path: Path) -> ScreenshotCapture:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-frame-" + path.name.encode("ascii"))
        return ScreenshotCapture(path, 2560, 1440, 0.01)

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.session.events_path.read_text(encoding="utf-8").splitlines()]

    def launch_calls(self):
        return [call for call in self.adb.run.call_args_list
                if call.args and call.args[0] != ["shell", "dumpsys", "package", self.config.game.package_name]]

    def test_bounded_wait_recycles_only_intermediate_polls_and_retains_result(self):
        self.session.evidence_budget = EvidenceBudget(self.session.directory / "frames", 300)
        selected = self.session.observe("selected")
        scenes = iter(["unknown"] * 30 + ["battle_result"])
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot(next(scenes), .95, path)
        result = self.session.wait_for({"battle_result"}, timeout_sec=20)
        self.assertTrue(selected.screenshot_path.exists())
        self.assertTrue(result.screenshot_path.exists())
        self.assertEqual(len(list((self.session.directory / "frames").glob("*.png"))), 2)
        self.assertEqual(result.observations["evidence_retention"], "retained")
        self.assertEqual(self.events()[-1]["kind"], "wait_evidence_retained")

    def test_bounded_wait_stop_preserves_last_poll(self):
        self.session.evidence_budget = EvidenceBudget(self.session.directory / "frames", 300)
        self.session.stop_event = Event()
        def recognize(path):
            self.session.stop_event.set()
            return SceneSnapshot("unknown", .95, path)
        self.recognizer.recognize.side_effect = recognize
        with self.assertRaises(StopRequested):
            self.session.wait_for({"battle_result"})
        self.assertTrue(self.session.last_snapshot.screenshot_path.exists())
        self.assertEqual(self.session.last_snapshot.observations["evidence_retention"], "retained")

    def package_calls(self):
        return [call for call in self.adb.run.call_args_list
                if call.args and call.args[0] == ["shell", "dumpsys", "package", self.config.game.package_name]]

    def test_unknown_or_low_confidence_observation_cannot_click(self) -> None:
        for scene, confidence in (("unknown", 0.99), ("village", 0.79)):
            with self.subTest(scene=scene, confidence=confidence):
                self.recognizer.recognize.side_effect = lambda path: SceneSnapshot(scene, confidence, Path(path))
                snapshot = self.session.observe()
                with self.assertRaises(FlowError):
                    self.session.tap(snapshot, (100, 200), reason="test")
        self.adb.run.assert_not_called()
        self.assertEqual(self.session.action_count, 0)

    def test_stop_blocks_capture_and_input_on_an_otherwise_valid_frame(self) -> None:
        snapshot = self.session.observe()
        self.session.stop_event = Event()
        self.session.stop_event.set()
        with self.assertRaises(StopRequested):
            self.session.tap(snapshot, (100, 200), reason="must not execute")
        with self.assertRaises(StopRequested):
            self.session.observe()
        self.adb.run.assert_not_called()
        self.assertEqual(self.capture.capture_screenshot_artifact.call_count, 1)

    def test_stop_during_foreground_query_prevents_launch(self) -> None:
        stop = Event()

        def foreground(*args, **kwargs):
            stop.set()
            return "other.package"

        self.foreground_package.side_effect = foreground
        with self.assertRaises(StopRequested):
            GameSession.connect(self.config, self.adb, "test", self.root / "cancelled", launch=True, stop_event=stop)
        self.adb.run.assert_not_called()

    def test_observation_records_capture_method_and_full_capture_elapsed_time(self) -> None:
        self.capture.capture_screenshot_artifact.side_effect = lambda path: ScreenshotCapture(
            path, 2560, 1440, .91, capture_method="raw_rgba", capture_elapsed_sec=.91)
        snapshot = self.session.observe()
        self.assertEqual(snapshot.observations["capture_method"], "raw_rgba")
        self.assertEqual(snapshot.observations["capture_elapsed_sec"], .91)
        self.assertEqual(self.events()[0]["observations"]["capture_method"], "raw_rgba")
        self.assertEqual(self.events()[0]["observations"]["capture_elapsed_sec"], .91)

    def test_legacy_screenshot_artifact_keeps_elapsed_time_without_inventing_method(self) -> None:
        snapshot = self.session.observe()
        self.assertEqual(snapshot.observations["capture_method"], "unknown")
        self.assertEqual(snapshot.observations["capture_elapsed_sec"], .01)

    def test_observation_times_capture_and_recognition_separately(self):
        def capture(path):
            self.clock.advance(.4)
            return ScreenshotCapture(path, 2560, 1440, .4)

        def recognize(path):
            self.clock.advance(4.7)
            return SceneSnapshot("village", .95, path)

        self.capture.capture_screenshot_artifact.side_effect = capture
        self.recognizer.recognize.side_effect = recognize
        snapshot = self.session.observe()
        self.assertAlmostEqual(snapshot.observations["recognition_elapsed_sec"], 4.7)
        self.assertAlmostEqual(snapshot.observations["observation_elapsed_sec"], 5.1)

    def test_nonfinite_confidence_cannot_click(self) -> None:
        for confidence in (float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("village", confidence, Path(path))
                snapshot = self.session.observe()
                with self.assertRaises(FlowError):
                    self.session.tap(snapshot, (100, 200), reason="invalid confidence")
        self.adb.run.assert_not_called()

    def test_explicit_battle_purpose_preserves_previous_snapshot_and_new_capture_age(self):
        previous = self.session.observe("before")
        slot = {"bbox": [100, 590, 190, 710]}
        self.recognizer.recognize_battle.side_effect = lambda path, **kwargs: SceneSnapshot("battle", .95, path)
        self.clock.advance(2)
        snapshot = self.session.observe("any-label", purpose="troop_count", slot=slot)
        self.recognizer.recognize_battle.assert_called_once_with(snapshot.screenshot_path,
            purpose="troop_count", slot=slot, previous=previous)
        self.assertEqual(snapshot.observations["observed_at_monotonic"], self.clock.now)
        self.assertIs(self.session.last_snapshot, snapshot)

    def test_stale_or_noncurrent_snapshot_cannot_click(self) -> None:
        previous = self.session.observe("previous")
        current = self.session.observe("current")
        with self.assertRaisesRegex(FlowError, "obsolete"):
            self.session.tap(previous, (100, 200), reason="obsolete")
        self.clock.advance(31)
        with self.assertRaises(FlowError):
            self.session.tap(current, (100, 200), reason="stale")
        self.adb.run.assert_not_called()

    def test_slow_recognition_does_not_refresh_age_of_old_screenshot(self) -> None:
        def slow_recognition(path: Path) -> SceneSnapshot:
            self.clock.advance(31)
            return SceneSnapshot("village", 0.95, Path(path))

        self.recognizer.recognize.side_effect = slow_recognition
        snapshot = self.session.observe()
        with self.assertRaises(FlowError):
            self.session.tap(snapshot, (100, 200), reason="recognition used an old screenshot")
        self.adb.run.assert_not_called()

    def test_boundary_coordinates_are_checked_before_adb(self) -> None:
        snapshot = self.session.observe()
        for point in ((-1, 0), (0, -1), (1280, 100), (100, 720)):
            with self.subTest(point=point), self.assertRaisesRegex(FlowError, "outside"):
                self.session.tap(snapshot, point, reason="invalid")
        self.adb.run.assert_not_called()
        for point in ((0, 0), (1279, 719)):
            snapshot = self.session.observe()
            self.session.tap(snapshot, point, reason="valid")
        self.assertEqual(self.adb.run.call_args.args[0], ["shell", "input", "-d", "2", "tap", "2558", "1438"])

    def battle_bar_frame(self):
        snapshot = self.session.observe("battle-toolbar")
        snapshot = replace(snapshot, scene="battle")
        snapshot.observations["battle"] = {"slots": [
            {"kind": "troop", "bbox": [x, 595, x + 89, 711], "point": [x + 44, 653],
             "count": 2, "confidence": 0, "evidence": {
                 "method": "independent_card_border", "count": {"confidence": .99}}}
            for x in (112, 208)]}
        self.session.last_snapshot = snapshot
        return snapshot

    def test_battle_bar_swipe_stays_in_toolbar_and_invalidates_old_coordinates(self):
        with patch.object(self.session.context, "swipe") as swipe:
            first = self.battle_bar_frame()
            self.session.swipe_battle_bar(first, direction="left", reason="Find a hidden spell")
            x1, y1, x2, y2, duration = swipe.call_args.args
            self.assertTrue(1280 > x1 > x2 > 0)
            self.assertTrue(595 < y1 == y2 < 711)
            self.assertTrue(300 <= duration <= 1000)
            self.assertIsNone(self.session.last_snapshot)
            with self.assertRaisesRegex(FlowError, "obsolete"):
                self.session.tap(first, (156, 653), reason="Old viewport must not be used")
            second = self.battle_bar_frame()
            self.session.swipe_battle_bar(second, direction="right", reason="Find own unit")
            self.assertLess(swipe.call_args.args[0], swipe.call_args.args[2])
        self.assertEqual(self.session.action_count, 2)
        self.assertEqual(len([e for e in self.events() if e["kind"] == "swipe_battle_bar"]), 2)
        self.adb.run.assert_not_called()

    def test_battle_bar_swipe_rejects_unverified_scene_anchor_and_stale_frame(self):
        for case in ("scene", "missing", "one_card", "map_card", "unknown", "direction", "stale", "stop"):
            with self.subTest(case=case):
                snapshot = self.battle_bar_frame()
                slots = snapshot.observations["battle"]["slots"]
                direction = "left"
                if case == "scene":
                    snapshot = replace(snapshot, scene="unknown")
                    self.session.last_snapshot = snapshot
                elif case == "missing":
                    snapshot.observations["battle"] = {}
                elif case == "one_card":
                    slots.pop()
                elif case == "map_card":
                    slots[0]["bbox"][1] = 450
                elif case == "unknown":
                    for card in slots:
                        card["evidence"]["count"]["confidence"] = .5
                elif case == "direction":
                    direction = "up"
                elif case == "stale":
                    self.clock.advance(31)
                else:
                    self.session.stop_event = Event()
                    self.session.stop_event.set()
                with self.assertRaises((FlowError, StopRequested)):
                    self.session.swipe_battle_bar(snapshot, direction=direction, reason="Unverified")
                self.session.stop_event = None
        self.adb.run.assert_not_called()
        self.assertEqual(self.session.action_count, 0)

    def test_template_point_uses_screenshot_scale_exactly_once(self) -> None:
        snapshot = self.session.observe()
        match = LocatorResult(None, "opencv_template", 0.99, (1000, 800), Bounds(980, 780, 1020, 820))
        with patch("autococ.session.find_template", return_value=match) as matcher:
            self.session.click_template(snapshot, "chat", roi=(400, 300, 700, 500))
        self.assertEqual(matcher.call_args.args[0], snapshot.screenshot_path)
        self.assertEqual(matcher.call_args.kwargs["roi_base_resolution"], (1280, 720))
        self.assertEqual(self.adb.run.call_args.args[0][-2:], ["1000", "800"])

    def test_template_at_last_source_pixel_stays_within_screen(self) -> None:
        snapshot = self.session.observe()
        match = LocatorResult(None, "opencv_template", 0.99, (2559, 1439), Bounds(2558, 1438, 2560, 1440))
        with patch("autococ.session.find_template", return_value=match):
            self.session.click_template(snapshot, "edge", roi=(1200, 650, 1280, 720))
        x, y = map(int, self.adb.run.call_args.args[0][-2:])
        self.assertTrue(2558 <= x < 2560)
        self.assertTrue(1438 <= y < 1440)

    def test_missing_or_ambiguous_button_cannot_click(self) -> None:
        snapshot = self.session.observe()
        with self.assertRaises(FlowError):
            self.session.click(snapshot, "attack")
        snapshot.observations["buttons"] = [
            {"name": "attack", "point": [100, 200], "text": "Attack"},
            {"name": "attack", "point": [200, 200], "text": "Attack"},
        ]
        with self.assertRaises(FlowError):
            self.session.click(snapshot, "attack")
        self.adb.run.assert_not_called()

    def test_wait_for_timeout_is_failure_and_never_blindly_backs(self) -> None:
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("unknown", 0.0, Path(path))
        with self.assertRaisesRegex(FlowError, "Expected"):
            self.session.wait_for({"village"}, timeout_sec=1)
        self.assertGreaterEqual(self.clock.now, 101)
        self.adb.run.assert_not_called()
        self.assertTrue(all(event["kind"] == "observation" for event in self.events()))

    def test_wait_for_rejects_target_observed_after_local_deadline(self) -> None:
        def slow_recognition(path: Path) -> SceneSnapshot:
            self.clock.advance(2)
            return SceneSnapshot("village", 0.95, Path(path))

        self.recognizer.recognize.side_effect = slow_recognition
        with self.assertRaises(FlowError):
            self.session.wait_for({"village"}, timeout_sec=1)

    def test_wait_for_accepts_target_before_deadline(self) -> None:
        self.assertEqual(self.session.wait_for({"village"}, timeout_sec=1).scene, "village")
        self.adb.run.assert_not_called()

    def test_back_requires_confirmed_permitted_scene(self) -> None:
        for scene in ("unknown", "village", "battle", "settlement"):
            with self.subTest(scene=scene):
                self.recognizer.recognize.side_effect = lambda path: SceneSnapshot(scene, 0.95, Path(path))
                snapshot = self.session.observe()
                with self.assertRaises(FlowError):
                    self.session.back(snapshot, reason="leave dialog")
        self.adb.run.assert_not_called()
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("clan_chat", 0.95, Path(path))
        self.session.back(self.session.observe(), reason="return to village")
        self.assertEqual(self.adb.run.call_args.args[0][-2:], ["keyevent", "4"])

    def test_expired_task_stops_before_screenshot_or_click(self) -> None:
        snapshot = self.session.observe()
        self.session.begin_task()
        self.clock.advance(self.config.runtime.task_timeout_sec + 1)
        before = self.capture.capture_screenshot_artifact.call_count
        with self.assertRaises(FlowError):
            self.session.observe()
        with self.assertRaises(FlowError):
            self.session.tap(snapshot, (100, 200), reason="expired")
        self.assertEqual(self.capture.capture_screenshot_artifact.call_count, before)
        self.adb.run.assert_not_called()

    def test_begin_task_cannot_reset_expired_session(self) -> None:
        self.session.deadline = self.clock.now
        with self.assertRaises(FlowError):
            self.session.begin_task()

    def test_capture_failure_invalidates_previous_observation(self) -> None:
        previous = self.session.observe()
        self.capture.capture_screenshot_artifact.side_effect = CaptureError("device disconnected")
        with self.assertRaises(CaptureError):
            self.session.observe()
        with self.assertRaises(FlowError):
            self.session.tap(previous, (100, 200), reason="cannot reuse observation after failed capture")
        self.adb.run.assert_not_called()

    def test_poll_interval_cannot_extend_local_wait_deadline(self) -> None:
        self.session.config = replace(self.config, runtime=replace(self.config.runtime, poll_interval_sec=60))
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("unknown", 0.0, Path(path))
        with self.assertRaises(FlowError):
            self.session.wait_for({"village"}, timeout_sec=1)
        self.assertEqual(self.clock.now, 101)
        self.assertEqual(self.capture.capture_screenshot_artifact.call_count, 1)

    def test_observation_action_observation_retains_distinct_evidence(self) -> None:
        before = self.session.observe("collect-before")
        self.session.tap(before, (100, 200), reason="recognized resource bubble")
        after = self.session.observe("collect-after")
        self.assertNotEqual(before.screenshot_path, after.screenshot_path)
        self.assertTrue(before.screenshot_path.is_file())
        self.assertTrue(after.screenshot_path.is_file())
        events = self.events()
        self.assertEqual([event["kind"] for event in events], ["observation", "tap", "observation"])
        self.assertEqual(events[1]["frame"], str(before.screenshot_path))
        self.assertEqual(events[1]["point"], [100, 200])
        self.assertEqual(events[2]["frame"], str(after.screenshot_path))
        self.assertEqual(self.session.action_count, 1)

    def test_failed_adb_does_not_increment_completed_action_count(self) -> None:
        snapshot = self.session.observe()
        self.adb.run.side_effect = AdbError("timeout")
        with self.assertRaises(AdbError):
            self.session.tap(snapshot, (100, 200), reason="test")
        self.assertEqual(self.session.action_count, 0)

    def test_chat_swipe_scales_once_and_records_source_frame(self) -> None:
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("clan_chat", 0.95, Path(path))
        snapshot = self.session.observe()
        self.session.swipe(snapshot, (180, 200), (180, 600), reason="scan old requests")
        self.assertEqual(self.adb.run.call_args.args[0],
                         ["shell", "input", "-d", "2", "swipe", "360", "400", "360", "1200", "500"])
        event = self.events()[-1]
        self.assertEqual(event["kind"], "swipe")
        self.assertEqual(event["frame"], str(snapshot.screenshot_path))
        self.assertEqual(event["start"], [180, 200])
        self.assertEqual(event["end"], [180, 600])
        self.assertEqual(self.session.action_count, 1)

    def test_chat_swipe_rejects_wrong_scene_or_stale_observation(self) -> None:
        for scene in ("village", "battle", "request", "unknown"):
            with self.subTest(scene=scene):
                self.recognizer.recognize.side_effect = lambda path: SceneSnapshot(scene, 0.95, Path(path))
                with self.assertRaises(FlowError):
                    self.session.swipe(self.session.observe(), (180, 200), (180, 600), reason="invalid")
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("clan_chat", 0.95, Path(path))
        old = self.session.observe()
        current = self.session.observe()
        with self.assertRaises(FlowError):
            self.session.swipe(old, (180, 200), (180, 600), reason="obsolete")
        self.clock.advance(31)
        with self.assertRaises(FlowError):
            self.session.swipe(current, (180, 200), (180, 600), reason="stale")
        self.adb.run.assert_not_called()

    def test_chat_swipe_rejects_input_navigation_short_and_reverse_gestures(self) -> None:
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("clan_chat", 0.95, Path(path))
        snapshot = self.session.observe()
        for start, end, duration in (((180, 200), (180, 680), 500), ((20, 200), (20, 600), 500),
                                     ((180, 600), (180, 200), 500), ((180, 200), (200, 600), 500),
                                     ((180, 200), (180, 250), 500), ((180, 200), (180, 600), 5000)):
            with self.subTest(start=start, end=end, duration=duration), self.assertRaises(FlowError):
                self.session.swipe(snapshot, start, end, duration_ms=duration, reason="invalid")
        self.adb.run.assert_not_called()

    def test_chat_swipe_obeys_deadline_and_does_not_count_failed_input(self) -> None:
        self.recognizer.recognize.side_effect = lambda path: SceneSnapshot("clan_chat", 0.95, Path(path))
        snapshot = self.session.observe()
        self.adb.run.side_effect = AdbError("timeout")
        with self.assertRaises(AdbError):
            self.session.swipe(snapshot, (180, 200), (180, 600), reason="input failure")
        self.assertEqual(self.session.action_count, 0)
        self.adb.run.reset_mock()
        self.session.task_deadline = self.clock.now
        with self.assertRaises(FlowError):
            self.session.swipe(snapshot, (180, 200), (180, 600), reason="expired")
        self.adb.run.assert_not_called()

    def test_launch_resolves_actual_display_after_start(self) -> None:
        order = []
        self.adb.run.side_effect = lambda args, **kwargs: order.append(
            "metadata" if args[:3] == ["shell", "dumpsys", "package"] else "launch")

        def actual_display(*args, **kwargs):
            order.append("resolve")
            return DisplayTarget(2, "physical-two")

        with patch("autococ.session.resolve_game_display", side_effect=actual_display), patch("autococ.session.ScreenshotRecognizer"):
            session = GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(order, ["launch", "resolve", "metadata"])
        self.assertEqual(session.capture.input_display_id, 2)
        self.assertEqual(session.capture.screenshot_display_id, "physical-two")
        self.assertTrue(session.capture.prefer_raw)

    def test_launch_retries_resolution_until_game_has_own_display(self) -> None:
        with patch("autococ.session.resolve_game_display", side_effect=[DeviceConnectionError("still loading"), DisplayTarget(2, "physical-two")]) as resolve:
            with patch("autococ.session.ScreenshotRecognizer"):
                session = GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(len(self.launch_calls()), 1)
        self.assertEqual(len(self.package_calls()), 1)
        self.assertEqual(session.capture.input_display_id, 2)

    def test_session_duration_includes_launch_time(self) -> None:
        config = replace(self.config, stop=replace(self.config.stop, max_duration_sec=30))
        self.adb.run.side_effect = lambda *args, **kwargs: self.clock.advance(20)
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")):
            with patch("autococ.session.ScreenshotRecognizer"):
                session = GameSession.connect(config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(session.deadline, 130)

    def test_already_foreground_game_is_connected_without_relaunch(self) -> None:
        self.foreground_package.return_value = self.config.game.package_name
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")) as resolve:
            with patch("autococ.session.ScreenshotRecognizer"):
                session = GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(self.launch_calls(), [])
        self.assertEqual(len(self.package_calls()), 1)
        resolve.assert_called_once()
        self.assertEqual(session.capture.input_display_id, 2)
        event = json.loads(session.events_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(event["launch_requested"])
        self.assertTrue(event["reused_foreground"])
        self.assertFalse(event["launch_issued"])

    def test_other_foreground_app_still_launches_and_records_the_action(self) -> None:
        self.foreground_package.return_value = "com.example.launcher"
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")):
            with patch("autococ.session.ScreenshotRecognizer"):
                session = GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(len(self.launch_calls()), 1)
        self.assertIn("monkey", self.launch_calls()[0].args[0])
        self.assertEqual(len(self.package_calls()), 1)
        event = json.loads(session.events_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(event["launch_issued"])
        self.assertFalse(event["reused_foreground"])

    def test_capture_only_connection_never_checks_or_changes_foreground_app(self) -> None:
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")):
            with patch("autococ.session.ScreenshotRecognizer"):
                GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=False)
        self.foreground_package.assert_not_called()
        self.assertEqual(self.launch_calls(), [])
        self.assertEqual(len(self.package_calls()), 1)

    def test_client_package_version_is_recorded_without_raw_dump(self) -> None:
        dump = "Package [com.supercell.clashofclans]\n    versionCode=180600008 minSdk=23\n    versionName=18.600.7\n"
        self.adb.run.return_value = ADBResult(("adb",), 0, dump, "")
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")), \
                patch("autococ.session.ScreenshotRecognizer"):
            session = GameSession.connect(self.config, self.adb, "test-device", self.root / "versioned", launch=False)
        self.assertEqual(session.client_version, "18.600.7")
        self.assertEqual(session.client_version_code, 180600008)
        event = json.loads(session.events_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(event["client_version"], "18.600.7")
        self.assertEqual(event["client_version_code"], 180600008)
        self.assertNotIn("Package [", str(event))

    def test_client_package_query_failure_does_not_block_connection(self) -> None:
        self.adb.run.side_effect = AdbError("package service busy")
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")), \
                patch("autococ.session.ScreenshotRecognizer"):
            session = GameSession.connect(self.config, self.adb, "test-device", self.root / "unknown-version", launch=False)
        self.assertEqual(session.client_version, "unknown")
        self.assertIsNone(session.client_version_code)
        self.assertEqual(len(self.package_calls()), 1)

    def test_vision_bundle_loads_distinct_scene_and_building_packages_before_search(self) -> None:
        config = replace(self.config, vision_agent=replace(self.config.vision_agent,
                         enabled=True, model_dir=str(self.root / "models")))
        scene_model, building_model = Mock(), Mock()
        for model in (scene_model, building_model):
            model.metadata = {"validation": {"status": "evaluated", "report_id": "offline-1"}}
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")), \
                patch("autococ.session.ScreenshotRecognizer") as recognizer, \
                patch("autococ.model_vision.SceneModel", return_value=scene_model) as load_scene, \
                patch("autococ.model_vision.BuildingModel", return_value=building_model) as load_building:
            session = GameSession.connect(config, self.adb, "test-device", self.root / "vision-run", launch=False)
        self.assertEqual(load_scene.call_args.args[0], self.root / "models" / "scene")
        self.assertEqual(load_building.call_args.args[0], self.root / "models" / "building")
        self.assertIs(session.prepared_building_model, building_model)
        self.assertIs(recognizer.return_value.scene_model, scene_model)
        self.assertIsNone(session.vision_model_error)

    def test_frozen_input_rejects_scaled_point_outside_physical_display(self) -> None:
        self.session.context.screen_resolution = (640, 360)
        self.session._active_frozen_plan = ("p", self.clock.now + 5, (1279, 719))
        with self.assertRaisesRegex(FlowError, "physical display"):
            self.session.frozen_battle_tap((1279, 719), timeout_sec=1,
                                           reason="prepared", plan_id="p")
        self.adb.run.assert_not_called()

    def test_frozen_input_recomputes_transport_timeout_immediately_before_send(self) -> None:
        self.session.context.screen_resolution = (1280, 720)
        self.session._active_frozen_plan = ("p", self.clock.now + 5, (100, 200))
        with patch.object(self.session.context, "_scale_point", side_effect=lambda point: (
                self.clock.advance(4.5) or point)):
            self.session.frozen_battle_tap((100, 200), timeout_sec=2,
                                           reason="prepared", plan_id="p")
        self.assertAlmostEqual(self.adb.run.call_args.kwargs["timeout_sec"], .5)
        self.adb.run.reset_mock()
        self.session._active_frozen_plan = ("p", self.clock.now + .25, (100, 200))
        with patch.object(self.session.context, "_scale_point", side_effect=lambda point: (
                self.clock.advance(.3) or point)):
            with self.assertRaisesRegex(FlowError, "expired before send"):
                self.session.frozen_battle_tap((100, 200), timeout_sec=1,
                                               reason="prepared", plan_id="p")
        self.adb.run.assert_not_called()

    def test_native_startup_reresolves_display_after_renderer_attachment_delay(self) -> None:
        config = replace(self.config, mumu=MuMuConfig(self.root, 1))
        native = Mock()
        with patch("autococ.session.resolve_game_display", side_effect=[
                DisplayTarget(2, "old"), DisplayTarget(3, "new")]) as resolve, \
                patch("autococ.session.MuMuClient", side_effect=[MuMuDisplayUnavailable("loading"), native]) as connect, \
                patch("autococ.session.ScreenshotRecognizer"):
            session = GameSession.connect(config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual([c.args[4] for c in connect.call_args_list], [2, 3])
        self.assertEqual(session.capture.screenshot_display_id, "new")
        self.assertIs(session.native, native)
        self.assertEqual(len(self.launch_calls()), 1)
        self.assertEqual(len(self.package_calls()), 1)
        native.tap.assert_not_called()

    def test_native_startup_does_not_retry_configuration_or_other_sdk_failures(self) -> None:
        config = replace(self.config, mumu=MuMuConfig(self.root, 1))
        for error in (DeviceConnectionError("wrong instance"), FlowError("capture timed out")):
            with self.subTest(error=error), \
                    patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "two")) as resolve, \
                    patch("autococ.session.MuMuClient", side_effect=error) as connect:
                with self.assertRaises(type(error)):
                    GameSession.connect(config, self.adb, "test-device", self.root / "connected", launch=False)
                self.assertEqual(resolve.call_count, 1)
                self.assertEqual(connect.call_count, 1)
        self.adb.run.assert_not_called()

    def test_native_readiness_wait_respects_session_limit_and_never_relaunches(self) -> None:
        config = replace(self.config, mumu=MuMuConfig(self.root, 1),
                         stop=replace(self.config.stop, max_duration_sec=2))
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "two")), \
                patch("autococ.session.MuMuClient", side_effect=MuMuDisplayUnavailable("loading")) as connect:
            with self.assertRaisesRegex(DeviceConnectionError, "deadline"):
                GameSession.connect(config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(self.clock.now, 102)
        self.assertEqual(connect.call_count, 2)
        self.assertEqual([c.kwargs["timeout_sec"] for c in connect.call_args_list], [2, 1])
        self.assertEqual(self.adb.run.call_count, 1)

    def test_late_native_initialization_is_closed_and_not_accepted(self) -> None:
        config = replace(self.config, mumu=MuMuConfig(self.root, 1))
        native = Mock()

        def late(*args, **kwargs):
            self.clock.advance(config.runtime.step_timeout_sec + 1)
            return native

        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "two")), \
                patch("autococ.session.MuMuClient", side_effect=late):
            with self.assertRaisesRegex(DeviceConnectionError, "deadline"):
                GameSession.connect(config, self.adb, "test-device", self.root / "connected", launch=False)
        native.close.assert_called_once_with()
        native.tap.assert_not_called()
        self.adb.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
