from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch

from autococ.capture import CaptureClient, ScreenshotCapture
from autococ.config import MuMuConfig, load_config
from autococ.device import DisplayTarget
from autococ.errors import AdbError, CaptureError, DeviceConnectionError, FlowError, MuMuDisplayUnavailable, StopRequested
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
        self.adb.run.side_effect = lambda *args, **kwargs: order.append("launch")

        def actual_display(*args, **kwargs):
            order.append("resolve")
            return DisplayTarget(2, "physical-two")

        with patch("autococ.session.resolve_game_display", side_effect=actual_display), patch("autococ.session.ScreenshotRecognizer"):
            session = GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(order, ["launch", "resolve"])
        self.assertEqual(session.capture.input_display_id, 2)
        self.assertEqual(session.capture.screenshot_display_id, "physical-two")
        self.assertTrue(session.capture.prefer_raw)

    def test_launch_retries_resolution_until_game_has_own_display(self) -> None:
        with patch("autococ.session.resolve_game_display", side_effect=[DeviceConnectionError("still loading"), DisplayTarget(2, "physical-two")]) as resolve:
            with patch("autococ.session.ScreenshotRecognizer"):
                session = GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=True)
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(self.adb.run.call_count, 1)
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
        self.adb.run.assert_not_called()
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
        self.assertEqual(self.adb.run.call_count, 1)
        self.assertIn("monkey", self.adb.run.call_args.args[0])
        event = json.loads(session.events_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(event["launch_issued"])
        self.assertFalse(event["reused_foreground"])

    def test_capture_only_connection_never_checks_or_changes_foreground_app(self) -> None:
        with patch("autococ.session.resolve_game_display", return_value=DisplayTarget(2, "physical-two")):
            with patch("autococ.session.ScreenshotRecognizer"):
                GameSession.connect(self.config, self.adb, "test-device", self.root / "connected", launch=False)
        self.foreground_package.assert_not_called()
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
        self.assertEqual(self.adb.run.call_count, 1)
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
