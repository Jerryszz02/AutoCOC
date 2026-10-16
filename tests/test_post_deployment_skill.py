"""One-shot current-client skill checks; all game I/O is an in-memory fake."""

from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from autococ.errors import CaptureError, FlowError, StopRequested
from autococ.scene import SceneSnapshot
from autococ.post_deployment_skill import execute_post_deployment_hero_skill


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
HEROES = ["grand_warden", "dragon_duke", "minion_prince", "archer_queen"]


def snapshot(number: int, *, scene: str = "battle", deployment: bool = False) -> SceneSnapshot:
    path = FIXTURES / f"hero_skill_035742_{number:05d}_18_647_2.png"
    now = time.monotonic()
    observations = {"source_resolution": [2560, 1440], "baseline_resolution": [1280, 720],
                    "capture_timestamp_basis": "capture_request_start",
                    "requested_at_monotonic": now, "observed_at_monotonic": now}
    if deployment:
        observations["deployment"] = {"verified": True, "completed": True, "deployed_units": 17,
                                      "heroes": [{"unit_id": hero, "verified": True} for hero in HEROES]}
    return SceneSnapshot(scene, .99, path, observations)


class FakeSession:
    def __init__(self, deployed: SceneSnapshot, frames: list[SceneSnapshot]):
        self.client_version = "18.647.2"
        self.deadline = time.monotonic() + 600
        self.task_deadline = self.deadline
        self.capture = SimpleNamespace(adb=None, serial="test", input_display_id=0)
        self.config = SimpleNamespace(game=SimpleNamespace(package_name="com.supercell.clashofclans",
                                                            baseline_resolution=(1280, 720)),
                                      runtime=SimpleNamespace(step_timeout_sec=5))
        self.last_snapshot = deployed
        self.post_deployment_skill_receipt = None
        self.frames = list(frames)
        self.taps = []
        self.fail_tap = False
        self.stop_on_check = None
        self.checks = 0
        self.tap_error = None

    def check_deadline(self):
        self.checks += 1
        if self.stop_on_check == self.checks:
            raise StopRequested("stopped")

    def observe(self, label: str, *, purpose: str = "full"):
        assert purpose == "full"
        item = self.frames.pop(0)
        now = time.monotonic()
        item.observations["requested_at_monotonic"] = now
        item.observations["observed_at_monotonic"] = now
        self.last_snapshot = item
        return item

    def tap(self, frame, point, *, reason):
        self.taps.append((frame, point, reason))
        if self.tap_error is not None:
            raise self.tap_error
        if self.fail_tap:
            raise RuntimeError("uncertain transport")


class PostDeploymentSkillTests(unittest.TestCase):
    def execute(self, session: FakeSession, deployed: SceneSnapshot, *, hero_id="grand_warden", delay=0):
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion",
                   return_value={"stationary": True}), \
             patch("autococ.post_deployment_skill._minimum_zoom_motion", return_value=True):
            return execute_post_deployment_hero_skill(session, deployed, hero_id=hero_id, delay_sec=delay)

    def test_ready_to_used_from_real_frames_taps_once_after_fresh_card_proof(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25), snapshot(26), snapshot(27)])
        latest, receipt = self.execute(session, deployed)
        self.assertEqual(latest.screenshot_path.name, "hero_skill_035742_00027_18_647_2.png")
        self.assertEqual((receipt["status"], receipt["verified"], receipt["completed"]),
                         ("verified", True, True))
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(session.taps[0][1], (691, 653))
        self.assertEqual(len(receipt["after_targets"]), 2)
        self.assertIsNone(receipt["after_targets"][0]["state"])
        self.assertEqual(len(receipt["evidence"]), 3)
        self.assertIs(session.post_deployment_skill_receipt, receipt)

    def test_incomplete_roster_zero_troops_wrong_version_hero_or_delay_never_taps(self):
        for change, hero, delay in (({"deployed_units": 0}, "grand_warden", 0),
                                    ({"heroes": [{"unit_id": hero, "verified": True}
                                                 for hero in HEROES[:3]]}, "grand_warden", 0),
                                    ({"heroes": [{"unit_id": "grand_warden", "verified": True}] * 4},
                                     "grand_warden", 0),
                                    ({}, "archer_queen", 0), ({}, "grand_warden", 31)):
            with self.subTest(change=change, hero=hero, delay=delay):
                deployed = snapshot(23, deployment=True)
                deployed.observations["deployment"].update(change)
                session = FakeSession(deployed, [])
                _, receipt = self.execute(session, deployed, hero_id=hero, delay=delay)
                self.assertFalse(receipt["completed"])
                self.assertFalse(receipt["attempted"])
                self.assertEqual(session.taps, [])
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [])
        session.client_version = "18.600.7"
        self.assertFalse(self.execute(session, deployed)[1]["completed"])

    def test_settlement_stale_account_stop_and_foreground_are_input_free(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25, scene="settlement")])
        latest, receipt = self.execute(session, deployed)
        self.assertEqual(latest.scene, "settlement")
        self.assertEqual(receipt["reason"], "battle_already_settled")
        self.assertFalse(receipt["attempted"])
        session = FakeSession(deployed, [snapshot(25, scene="account_in_use")])
        with self.assertRaisesRegex(Exception, "account_in_use"):
            self.execute(session, deployed)
        self.assertFalse(session.post_deployment_skill_receipt["attempted"])
        session = FakeSession(deployed, [snapshot(25)])
        session.stop_on_check = 1
        with self.assertRaises(StopRequested):
            self.execute(session, deployed)
        self.assertEqual(session.taps, [])
        session = FakeSession(deployed, [snapshot(25)])
        with patch("autococ.post_deployment_skill.foreground_package", return_value="wrong"):
            _, receipt = execute_post_deployment_hero_skill(session, deployed, hero_id="grand_warden", delay_sec=0)
        self.assertFalse(receipt["attempted"])
        self.assertEqual(session.taps, [])

    def test_transition_missing_book_or_hp_cannot_authorize_tap(self):
        for number in (26, 27):
            with self.subTest(number=number):
                deployed = snapshot(23, deployment=True)
                session = FakeSession(deployed, [snapshot(number)])
                _, receipt = self.execute(session, deployed)
                self.assertFalse(receipt["attempted"])
                self.assertEqual(session.taps, [])
        with TemporaryDirectory() as directory:
            ready = cv2.imdecode(np.fromfile(snapshot(25).screenshot_path, np.uint8), cv2.IMREAD_COLOR)
            for label, box in (("book", (650, 530, 690, 567)), ("hp", (654, 573, 731, 586))):
                with self.subTest(missing=label):
                    changed = ready.copy()
                    left, top, right, bottom = box
                    changed[top:bottom, left:right] = 30
                    path = Path(directory) / f"missing-{label}.png"
                    cv2.imencode(".png", changed)[1].tofile(path)
                    deployed = snapshot(23, deployment=True)
                    session = FakeSession(deployed, [replace(snapshot(25), screenshot_path=path)])
                    _, receipt = self.execute(session, deployed)
                    self.assertFalse(receipt["attempted"])
                    self.assertEqual(session.taps, [])

    def test_uncertain_tap_is_persisted_and_reinvocation_never_replays(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25)])
        session.fail_tap = True
        latest, receipt = self.execute(session, deployed)
        self.assertEqual(receipt["status"], "action_uncertain")
        self.assertTrue(receipt["attempted"])
        self.assertFalse(receipt["issued"])
        self.assertEqual(len(session.taps), 1)
        _, again = self.execute(session, latest)
        self.assertIs(again, receipt)
        self.assertEqual(len(session.taps), 1)

    def test_post_capture_failure_and_unverified_after_do_not_retap(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25), snapshot(25), snapshot(26)])
        _, receipt = self.execute(session, deployed)
        self.assertEqual(receipt["status"], "attempted_unverified")
        self.assertEqual(len(session.taps), 1)

        class FailedAfterCapture(FakeSession):
            def observe(self, label, *, purpose="full"):
                if label.endswith("after-1"):
                    self.last_snapshot = None  # GameSession.observe clears this before capture.
                    raise CaptureError("pixels unavailable after uncertain input")
                return super().observe(label, purpose=purpose)

        session = FailedAfterCapture(deployed, [snapshot(25)])
        latest, receipt = self.execute(session, deployed)
        self.assertEqual(latest.screenshot_path.name, "hero_skill_035742_00025_18_647_2.png")
        self.assertEqual(receipt["status"], "attempted_unverified")
        self.assertTrue(receipt["attempted"])
        self.assertTrue(receipt["issued"])
        self.assertFalse(receipt["verified"])
        self.assertIsNone(session.last_snapshot)
        self.assertEqual(len(session.taps), 1)
        _, again = self.execute(session, latest)
        self.assertIs(again, receipt)
        self.assertEqual(len(session.taps), 1)

    def test_account_conflict_from_tap_or_uncertain_transport_propagates(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25)])
        session.tap_error = FlowError("Game interruption: account_in_use")
        with self.assertRaisesRegex(FlowError, "account_in_use"):
            self.execute(session, deployed)
        self.assertTrue(session.post_deployment_skill_receipt["attempted"])
        self.assertEqual(len(session.taps), 1)

        class ChangedAccount(FakeSession):
            def tap(self, frame, point, *, reason):
                self.last_snapshot = snapshot(25, scene="account_in_use")
                super().tap(frame, point, reason=reason)

        session = ChangedAccount(deployed, [snapshot(25)])
        session.tap_error = RuntimeError("uncertain")
        with self.assertRaisesRegex(FlowError, "account_in_use"):
            self.execute(session, deployed)
        self.assertTrue(session.post_deployment_skill_receipt["attempted"])
        self.assertEqual(len(session.taps), 1)

    def test_stale_missing_timestamp_camera_display_and_reserve_reject_before_input(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25)])
        session.task_deadline = time.monotonic() + 80
        self.assertEqual(self.execute(session, deployed)[1]["reason"],
                         "insufficient_settlement_reserve_for_skill")
        self.assertEqual(session.taps, [])

        class StaleFrame(FakeSession):
            def observe(self, label, *, purpose="full"):
                item = super().observe(label, purpose=purpose)
                item.observations["observed_at_monotonic"] -= 9
                return item

        session = StaleFrame(deployed, [snapshot(25)])
        self.assertEqual(self.execute(session, deployed)[1]["reason"],
                         "fresh_skill_input_source_stale_or_changed")
        self.assertEqual(session.taps, [])

        class WrongDisplay(FakeSession):
            def observe(self, label, *, purpose="full"):
                item = super().observe(label, purpose=purpose)
                self.capture.input_display_id = 1
                return item

        session = WrongDisplay(deployed, [snapshot(25)])
        self.assertEqual(self.execute(session, deployed)[1]["reason"],
                         "deployment_source_or_display_changed")
        self.assertEqual(session.taps, [])

        no_timestamp = snapshot(25)
        no_timestamp.observations.pop("capture_timestamp_basis")
        session = FakeSession(deployed, [no_timestamp])
        self.assertFalse(self.execute(session, deployed)[1]["attempted"])
        self.assertEqual(session.taps, [])
        session = FakeSession(deployed, [snapshot(25)])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion", return_value={}), \
             patch("autococ.post_deployment_skill._minimum_zoom_motion", return_value=False):
            _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                            hero_id="grand_warden", delay_sec=0)
        self.assertEqual(receipt["reason"], "camera_not_strictly_stationary")
        self.assertEqual(session.taps, [])

    def test_stop_after_ready_proof_sends_zero_inputs_and_post_frames_must_advance(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25)])
        session.stop_on_check = 2
        with self.assertRaises(StopRequested):
            self.execute(session, deployed)
        self.assertEqual(session.taps, [])

        class RepeatedSecondPostTimestamp(FakeSession):
            def observe(self, label, *, purpose="full"):
                item = super().observe(label, purpose=purpose)
                if label.endswith("after-1"):
                    self.first_post_at = item.observations["requested_at_monotonic"]
                if label.endswith("after-2"):
                    # Still later than input_done_at and has a different path:
                    # only the rolling previous-post check can reject it.
                    item.observations["requested_at_monotonic"] = self.first_post_at
                return item

        session = RepeatedSecondPostTimestamp(deployed, [snapshot(25), snapshot(26), snapshot(27)])
        _, receipt = self.execute(session, deployed)
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["status"], "attempted_unverified")
        self.assertEqual(len(session.taps), 1)

    def test_changed_deployment_source_sha_and_missing_fresh_source_are_input_free(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "deployment.png"
            source.write_bytes(snapshot(23).screenshot_path.read_bytes())
            deployed = replace(snapshot(23, deployment=True), screenshot_path=source)

            class ChangedSource(FakeSession):
                def observe(self, label, *, purpose="full"):
                    item = super().observe(label, purpose=purpose)
                    source.write_bytes(source.read_bytes() + b"changed-after-initial-hash")
                    return item

            session = ChangedSource(deployed, [snapshot(25)])
            _, receipt = self.execute(session, deployed)
            self.assertEqual(receipt["reason"], "deployment_source_or_display_changed")
            self.assertEqual(session.taps, [])

            fresh = snapshot(25)
            fresh.observations["source_resolution"] = [1280, 720]
            session = FakeSession(deployed, [fresh])
            _, receipt = self.execute(session, deployed)
            self.assertFalse(receipt["attempted"])
            self.assertEqual(session.taps, [])
