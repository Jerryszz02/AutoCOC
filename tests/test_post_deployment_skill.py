"""One-shot current-client skill checks; all game I/O is an in-memory fake."""

from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import math
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


def stabilization_snapshot(number: int, *, deployment: bool = False) -> SceneSnapshot:
    original = snapshot(23, deployment=deployment)
    return replace(original, screenshot_path=FIXTURES /
                   f"hero_skill_stabilization_054932_{number:05d}_18_647_2.png")


def camera_motion(dx: float, dy: float, *, stationary: bool = True,
                  matches: int = 100, inliers: int = 90, rotation: float = 0.0) -> dict:
    return {"stationary": stationary, "matched_points": matches, "inliers": inliers,
            "affine": [[1.0, rotation, dx], [-rotation, 1.0, dy]]}


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

    def test_real_camera_borderline_allows_one_observation_but_dead_hp_never_taps(self):
        deployed = stabilization_snapshot(25, deployment=True)
        session = FakeSession(deployed, [stabilization_snapshot(26), stabilization_snapshot(27)])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"):
            latest, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                                 hero_id="grand_warden", delay_sec=0)
        self.assertEqual(latest.screenshot_path.name,
                         "hero_skill_stabilization_054932_00027_18_647_2.png")
        self.assertEqual(receipt["reason"], "current_warden_ready_with_live_hp_unverified")
        self.assertFalse(receipt["attempted"])
        self.assertEqual(session.taps, [])
        self.assertEqual(len(session.frames), 0)
        first = receipt["stabilization"]["first"]
        second = receipt["stabilization"]["second"]
        self.assertEqual(first["sha256"], hashlib.sha256(stabilization_snapshot(26).screenshot_path.read_bytes()).hexdigest())
        self.assertEqual(second["sha256"], hashlib.sha256(stabilization_snapshot(27).screenshot_path.read_bytes()).hexdigest())
        self.assertLess(first["capture_request_monotonic"], second["capture_request_monotonic"])
        first_motion = receipt["stabilization"]["first_motion"]
        second_motion = receipt["stabilization"]["second_motion"]
        self.assertTrue(first_motion["stationary"])
        self.assertAlmostEqual(math.hypot(first_motion["affine"][0][2], first_motion["affine"][1][2]),
                               3.018, delta=.002)
        self.assertTrue(second_motion["stationary"])
        self.assertLess(math.hypot(second_motion["affine"][0][2], second_motion["affine"][1][2]), 3)
        self.assertEqual(receipt["stabilization"]["status"], "strict_motion_verified")
        self.assertFalse(receipt["target_before"]["verified"])

    def test_one_bounded_stabilization_can_authorize_only_second_fresh_ready_card(self):
        # Synthetic state-machine timing only: the real 054932 second frame had
        # no live HP and is covered by the separate negative replay above.
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25), snapshot(24), snapshot(27)])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion",
                   side_effect=[camera_motion(3.1, 0.0), camera_motion(.2, 0.0)]):
            latest, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                                 hero_id="grand_warden", delay_sec=0)
        self.assertTrue(receipt["verified"])
        self.assertEqual(latest.screenshot_path.name, "hero_skill_035742_00027_18_647_2.png")
        self.assertEqual(receipt["before"]["frame"], str(snapshot(24).screenshot_path))
        self.assertEqual(receipt["stabilization"]["first"]["frame"], str(snapshot(25).screenshot_path))
        self.assertEqual(len(receipt["evidence"]), 3)
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(session.taps[0][0].screenshot_path.name, "hero_skill_035742_00024_18_647_2.png")
        self.assertEqual(session.taps[0][1], (691, 653))
        _, again = execute_post_deployment_hero_skill(session, latest, hero_id="grand_warden", delay_sec=0)
        self.assertIs(again, receipt)
        self.assertEqual(len(session.taps), 1)

    def test_stabilization_never_observes_again_after_large_or_weak_initial_motion(self):
        for motion in (camera_motion(4.0, 0.0), camera_motion(3.2, 0.0, stationary=False),
                       camera_motion(3.2, 0.0, inliers=70), camera_motion(3.2, 0.0, rotation=.003),
                       {"stationary": True, "affine": None}):
            with self.subTest(motion=motion):
                deployed = snapshot(23, deployment=True)
                session = FakeSession(deployed, [snapshot(25), snapshot(24)])
                with patch("autococ.post_deployment_skill.foreground_package",
                           return_value="com.supercell.clashofclans"), \
                     patch("autococ.post_deployment_skill.measure_camera_motion", return_value=motion):
                    _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                                    hero_id="grand_warden", delay_sec=0)
                self.assertEqual(receipt["reason"], "camera_not_strictly_stationary")
                self.assertEqual(len(session.frames), 1)
                self.assertEqual(session.taps, [])

    def test_second_stabilization_must_be_new_strict_battle_and_keep_reserve(self):
        first = camera_motion(3.1, 0.0)
        for second, expected in ((camera_motion(3.1, 0.0), "camera_not_strictly_stationary_after_one_observation"),
                                 (camera_motion(.1, 0.0, inliers=50, matches=100),
                                  "camera_not_strictly_stationary_after_one_observation")):
            with self.subTest(second_motion=second):
                deployed = snapshot(23, deployment=True)
                session = FakeSession(deployed, [snapshot(25), snapshot(24)])
                with patch("autococ.post_deployment_skill.foreground_package",
                           return_value="com.supercell.clashofclans"), \
                     patch("autococ.post_deployment_skill.measure_camera_motion",
                           side_effect=[first, second]):
                    _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                                    hero_id="grand_warden", delay_sec=0)
                self.assertEqual(receipt["reason"], expected)
                self.assertEqual(session.taps, [])
                self.assertEqual(len(session.frames), 0)

        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25), snapshot(24, scene="unknown")])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion", return_value=first):
            _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                            hero_id="grand_warden", delay_sec=0)
        self.assertIn("positive current battle", receipt["reason"])
        self.assertEqual(session.taps, [])

        session = FakeSession(deployed, [snapshot(25), snapshot(24, scene="settlement")])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion", return_value=first):
            _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                            hero_id="grand_warden", delay_sec=0)
        self.assertEqual(receipt["reason"], "battle_already_settled")
        self.assertEqual(session.taps, [])

        class RepeatedStabilizationTime(FakeSession):
            def observe(self, label, *, purpose="full"):
                item = super().observe(label, purpose=purpose)
                if label.endswith("skill-before"):
                    self.first_capture_at = item.observations["requested_at_monotonic"]
                if label.endswith("stabilized-before"):
                    item.observations["requested_at_monotonic"] = self.first_capture_at
                return item

        session = RepeatedStabilizationTime(deployed, [snapshot(25), snapshot(24)])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion", return_value=first):
            _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                            hero_id="grand_warden", delay_sec=0)
        self.assertIn("capture order is unverified", receipt["reason"])
        self.assertEqual(session.taps, [])

        class LostReserve(FakeSession):
            def observe(self, label, *, purpose="full"):
                item = super().observe(label, purpose=purpose)
                if label.endswith("stabilized-before"):
                    self.task_deadline = time.monotonic() + 80
                return item

        session = LostReserve(deployed, [snapshot(25), snapshot(24)])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion",
                   side_effect=[first, camera_motion(.1, 0.0)]):
            _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                            hero_id="grand_warden", delay_sec=0)
        self.assertEqual(receipt["reason"], "fresh_skill_input_source_stale_or_changed")
        self.assertEqual(session.taps, [])

        class ChangedDisplay(FakeSession):
            def observe(self, label, *, purpose="full"):
                item = super().observe(label, purpose=purpose)
                if label.endswith("stabilized-before"):
                    self.capture.input_display_id += 1
                return item

        session = ChangedDisplay(deployed, [snapshot(25), snapshot(24)])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion", return_value=first):
            _, receipt = execute_post_deployment_hero_skill(session, deployed,
                                                            hero_id="grand_warden", delay_sec=0)
        self.assertEqual(receipt["reason"], "stabilization_source_or_display_changed")
        self.assertEqual(session.taps, [])

    def test_stabilization_stop_and_account_conflict_send_zero_inputs(self):
        deployed = snapshot(23, deployment=True)
        session = FakeSession(deployed, [snapshot(25), snapshot(24)])
        session.stop_on_check = 2
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion",
                   return_value=camera_motion(3.1, 0.0)):
            with self.assertRaises(StopRequested):
                execute_post_deployment_hero_skill(session, deployed, hero_id="grand_warden", delay_sec=0)
        self.assertEqual(len(session.frames), 1)
        self.assertEqual(session.taps, [])

        session = FakeSession(deployed, [snapshot(25), snapshot(24, scene="account_in_use")])
        with patch("autococ.post_deployment_skill.foreground_package",
                   return_value="com.supercell.clashofclans"), \
             patch("autococ.post_deployment_skill.measure_camera_motion",
                   return_value=camera_motion(3.1, 0.0)):
            with self.assertRaisesRegex(FlowError, "account_in_use"):
                execute_post_deployment_hero_skill(session, deployed, hero_id="grand_warden", delay_sec=0)
        self.assertEqual(session.taps, [])

    def test_stabilization_fixture_source_hashes_and_native_pixels(self):
        metadata = json.loads((FIXTURES / "hero_skill_stabilization_054932_18_647_2.provenance.json")
                              .read_text(encoding="utf-8"))
        self.assertFalse(metadata["human_ground_truth"])
        self.assertFalse(metadata["dynamic_event_validation"])
        for name, record in metadata["fixtures"].items():
            fixture = FIXTURES / name
            self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), record["fixture_sha256"])
            original = Path("E:/projects/AutoCOC") / record["source_frame"]
            if original.is_file():
                self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), record["source_sha256"])
                raw = cv2.imdecode(np.fromfile(original, np.uint8), cv2.IMREAD_COLOR)
                expected = np.zeros_like(raw)
                expected[220:1060, 400:2180] = raw[220:1060, 400:2180]
                expected[1030:1440, :] = raw[1030:1440, :]
                saved = cv2.imdecode(np.fromfile(fixture, np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(saved, expected))

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
