from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from autococ.errors import FlowError
from autococ.flow import return_to_village
from autococ.recovery import confirm_welcome_back, recover_connection, welcome_back_point
from autococ.reporting import RunStats, TaskResult
from autococ.scene import SceneSnapshot
from autococ.session import GameSession


RETRY = {"name": "retry", "text": "重试", "confidence": 0.99988,
         "bbox": [335, 406, 375, 430], "point": [355, 418]}
WELCOME_OCR = json.loads((Path(__file__).parent / "fixtures" /
                          "welcome_back_ocr_20260926.json").read_text(encoding="utf-8"))


def frame(index: int, scene: str, *, gems: int | None = None, confidence: float = 0.95) -> SceneSnapshot:
    buttons = [deepcopy(RETRY)] if scene == "disconnected" else [
        {"name": name, "point": [100, 600], "text": name} for name in ("attack", "shop")
    ] if scene == "village" else []
    return SceneSnapshot(scene, confidence, Path(f"recovery-{index}.png"), {
        "buttons": buttons, "resources": {"gems": gems},
    })


def welcome(index: int = 1) -> SceneSnapshot:
    return SceneSnapshot("popup", WELCOME_OCR["confidence"], Path(f"recovery-{index}.png"), {
        "ocr": deepcopy(WELCOME_OCR["ocr"]),
        "buttons": [{"name": "confirm", "point": [655, 601], "text": "确定",
                     "bbox": [637, 590, 673, 612], "confidence": .99999}],
    })


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


class FakeSession:
    buttons = staticmethod(GameSession.buttons)
    click = GameSession.click

    def __init__(self, frames: list[SceneSnapshot], clock: FakeClock, *, capture_seconds: float = 0) -> None:
        self.frames = iter(frames)
        self.clock = clock
        self.capture_seconds = capture_seconds
        self.last_snapshot = None
        self.deadline = self.task_deadline = 1000
        self.config = SimpleNamespace(runtime=SimpleNamespace(poll_interval_sec=1))
        self.taps = []

    def check_deadline(self) -> None:
        if self.clock.now >= min(self.deadline, self.task_deadline):
            raise FlowError("Session or task deadline exceeded")

    def observe(self, label: str = "observe") -> SceneSnapshot:
        self.check_deadline()
        self.clock.now += self.capture_seconds
        self.last_snapshot = next(self.frames, self.last_snapshot)
        return self.last_snapshot

    def tap(self, snapshot: SceneSnapshot, point: list[int], *, reason: str) -> None:
        if snapshot is not self.last_snapshot:
            raise AssertionError("Stale click")
        self.taps.append((snapshot.screenshot_path, point))

    def wait_for(self, *args, **kwargs):
        raise AssertionError("Recovery must not use interruption-sensitive wait_for")


def recover(frames: list[SceneSnapshot], *, capture_seconds: float = 0, deadline: float = 1000):
    clock = FakeClock()
    session = FakeSession(frames, clock, capture_seconds=capture_seconds)
    session.task_deadline = deadline
    with patch("autococ.recovery.time", clock):
        result = recover_connection(session)
    return result, session, clock


class RecoveryTests(unittest.TestCase):
    def test_real_welcome_ocr_is_confirmed_once_after_retry(self) -> None:
        popup = welcome()
        result, session, _ = recover([frame(0, "disconnected"), popup, popup,
                                      frame(3, "village")])
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(result.metrics["welcome_confirm_actions"], 1)
        self.assertTrue(result.metrics["village_verified"])
        self.assertEqual(session.taps, [(Path("recovery-0.png"), [355, 418]),
                                        (popup.screenshot_path, [655, 601])])

    def test_welcome_title_description_and_unique_ocr_button_are_required(self) -> None:
        for change in ("upgrade_title", "missing_description", "duplicate_confirm",
                       "low_title_confidence", "confirm_outside_bottom", "wrong_scene"):
            with self.subTest(change=change):
                popup = welcome()
                ocr = popup.observations["ocr"]
                if change == "upgrade_title":
                    ocr[0]["text"] = "将部落城堡升至 13 级？"
                elif change == "missing_description":
                    ocr.pop(1)
                elif change == "duplicate_confirm":
                    ocr.append(deepcopy(ocr[2]))
                elif change == "low_title_confidence":
                    ocr[0]["confidence"] = .7
                elif change == "confirm_outside_bottom":
                    ocr[2]["bbox"] = [637, 400, 673, 422]
                else:
                    popup = SceneSnapshot("village", .95, popup.screenshot_path, popup.observations)
                self.assertIsNone(welcome_back_point(popup))
                result, session, _ = recover([frame(0, "disconnected"), popup])
                self.assertEqual(result.status, "failed")
                self.assertEqual(len(session.taps), 1)
                self.assertEqual(result.metrics["welcome_confirm_actions"], 0)

    def test_obsolete_welcome_frame_cannot_authorize_click(self) -> None:
        popup = welcome()
        session = FakeSession([], FakeClock())
        session.last_snapshot = frame(2, "unknown")
        with self.assertRaisesRegex(FlowError, "obsolete"):
            confirm_welcome_back(session, popup)
        self.assertEqual(session.taps, [])

    def test_return_to_village_uses_narrow_welcome_confirmation(self) -> None:
        popup, village = welcome(), frame(2, "village")
        session = Mock()
        session.last_snapshot = popup
        session.wait_for.return_value = village
        self.assertIs(return_to_village(session, initial_snapshot=popup), village)
        session.tap.assert_called_once_with(popup, [655, 601],
                                             reason="Dismiss observed welcome-back summary")
        session.back.assert_not_called()
        self.assertNotIn("popup", session.wait_for.call_args.args[0])

    def test_current_initial_snapshot_is_reused_without_recapture(self) -> None:
        clock = FakeClock()
        before, after = frame(0, "disconnected"), frame(1, "village")
        session = FakeSession([after], clock)
        session.last_snapshot = before
        with patch("autococ.recovery.time", clock):
            result = recover_connection(session, initial_snapshot=before)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.evidence, [before.screenshot_path, after.screenshot_path])
        self.assertEqual(session.taps, [(before.screenshot_path, [355, 418])])

    def test_obsolete_initial_snapshot_cannot_authorize_retry(self) -> None:
        clock = FakeClock()
        old = frame(0, "disconnected")
        session = FakeSession([], clock)
        session.last_snapshot = frame(1, "unknown")
        with patch("autococ.recovery.time", clock):
            result = recover_connection(session, initial_snapshot=old)
        self.assertEqual(result.status, "failed")
        self.assertIn("obsolete", result.reason)
        self.assertEqual(session.taps, [])

    def test_old_disconnected_frames_are_tolerated_after_one_retry(self) -> None:
        frames = [frame(0, "disconnected"), frame(1, "disconnected"), frame(2, "disconnected"),
                  frame(3, "unknown"), frame(4, "village", gems=323)]
        result, session, _ = recover(frames)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(session.taps[0][1], [355, 418])
        self.assertEqual(result.metrics["retry_actions"], 1)
        self.assertTrue(result.metrics["village_verified"])
        self.assertEqual(result.metrics["before_frame"], "recovery-0.png")
        self.assertEqual(result.metrics["after_frame"], "recovery-4.png")
        self.assertIsNone(result.metrics["gems_before"])
        self.assertEqual(result.metrics["gems_after"], 323)
        self.assertIsNone(result.metrics["gems_delta"])
        self.assertEqual(len(result.evidence), 5)

    def test_persistent_same_frame_times_out_without_reclicking(self) -> None:
        result, session, clock = recover([frame(0, "disconnected")])
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.reason)
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(clock.now, 60)

    def test_unknown_after_retry_times_out_without_more_actions(self) -> None:
        result, session, _ = recover([frame(0, "disconnected"), frame(1, "unknown")])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["last_scene"], "unknown")
        self.assertEqual(len(session.taps), 1)

    def test_maintenance_stops_immediately(self) -> None:
        result, session, clock = recover([frame(0, "disconnected"), frame(1, "maintenance")])
        self.assertEqual(result.status, "failed")
        self.assertIn("maintenance", result.reason)
        self.assertEqual(clock.now, 0)
        self.assertEqual(len(session.taps), 1)

    def test_other_initial_scenes_never_receive_retry_click(self) -> None:
        for scene in ("village", "unknown", "battle", "maintenance"):
            with self.subTest(scene=scene):
                result, session, _ = recover([frame(0, scene)])
                self.assertEqual(result.status, "failed")
                self.assertEqual(session.taps, [])

    def test_retry_requires_unique_reliable_central_visual_evidence(self) -> None:
        for change in ("missing", "duplicate", "outside", "low_confidence", "point_outside_box", "missing_box"):
            with self.subTest(change=change):
                before = frame(0, "disconnected")
                button = before.observations["buttons"][0]
                if change == "missing":
                    before.observations["buttons"] = []
                elif change == "duplicate":
                    before.observations["buttons"].append(deepcopy(button))
                elif change == "outside":
                    button["point"] = [50, 50]
                elif change == "low_confidence":
                    button["confidence"] = 0.5
                elif change == "point_outside_box":
                    button["point"] = [800, 418]
                else:
                    button.pop("bbox")
                result, session, _ = recover([before])
                self.assertEqual(result.status, "failed")
                self.assertEqual(session.taps, [])

    def test_low_scene_confidence_cannot_authorize_retry(self) -> None:
        for confidence in (0.7, float("nan")):
            with self.subTest(confidence=confidence):
                result, session, _ = recover([frame(0, "disconnected", confidence=confidence)])
                self.assertEqual(result.status, "failed")
                self.assertEqual(session.taps, [])

    def test_village_must_have_both_unique_navigation_controls(self) -> None:
        for buttons in ([{"name": "attack"}], [{"name": "attack"}, {"name": "shop"}, {"name": "shop"}]):
            with self.subTest(buttons=buttons):
                home = frame(1, "village")
                home.observations["buttons"] = buttons
                result, _, _ = recover([frame(0, "disconnected"), home])
                self.assertEqual(result.status, "failed")
                self.assertFalse(result.metrics["village_verified"])

    def test_late_village_capture_does_not_bypass_sixty_second_limit(self) -> None:
        result, _, _ = recover([frame(0, "disconnected"), frame(1, "village")], capture_seconds=61)
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.reason)
        self.assertEqual(result.metrics["after_frame"], "recovery-1.png")

    def test_existing_task_deadline_is_preserved(self) -> None:
        result, session, clock = recover([frame(0, "disconnected")], deadline=3)
        self.assertEqual(result.status, "failed")
        self.assertEqual(clock.now, 3)
        self.assertEqual(session.task_deadline, 3)
        self.assertEqual(len(session.taps), 1)

    def test_available_gem_values_are_preserved_without_inventing_unknowns(self) -> None:
        result, _, _ = recover([frame(0, "disconnected", gems=323), frame(1, "village", gems=323)])
        self.assertEqual(result.metrics["gems_delta"], 0)
        self.assertEqual(result.metrics["gems_before"], 323)

    def test_recording_recovery_does_not_erase_prior_battle_failure(self) -> None:
        stats = RunStats()
        failure = TaskResult("battle", "failed", "disconnected", evidence=[Path("battle.png")])
        stats.record_task(failure)
        result, _, _ = recover([frame(0, "disconnected"), frame(1, "village")])
        stats.record_task(result)
        self.assertEqual(stats.failures, 1)
        self.assertIs(stats.task_results[0], failure)
        self.assertEqual(stats.task_results[0].status, "failed")
        self.assertEqual(stats.task_results[1].task, "recover")


if __name__ == "__main__":
    unittest.main()
