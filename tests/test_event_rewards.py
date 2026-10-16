"""Versioned event reward evidence and one-input opt-in behavior."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import cv2

from autococ.errors import FlowError, StopRequested
from autococ.event_rewards import (claim_one_free_reward, close_reward_track,
                                   observe_reward_track, open_reward_track)
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "event_rewards_18_647_2"


def fixture_ocr(name: str) -> list[dict]:
    return json.loads((FIXTURES / f"{name}.ocr.json").read_text(encoding="utf-8"))


def recognized(name: str, *, ocr_name: str | None = None) -> SceneSnapshot:
    provider = Mock()
    provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                                       for item in fixture_ocr(ocr_name or name)]
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    snapshot = recognizer.recognize(FIXTURES / f"{name}.png")
    snapshot.observations["observed_at_monotonic"] = time.monotonic()
    return snapshot


class FakeSession:
    def __init__(self, snapshots, *, version="18.647.2"):
        self.snapshots = list(snapshots)
        self.client_version = version
        self.deadline = self.task_deadline = time.monotonic() + 60
        self.capture = Mock(adb=object(), serial="fake-serial")
        self.config = Mock()
        self.config.runtime.step_timeout_sec = 5
        self.config.game.package_name = "com.supercell.clashofclans"
        self.config.game.baseline_resolution = (1280, 720)
        self.last_snapshot = None
        self.taps = []
        self.template_clicks = []
        self.events = []
        self.observation_count = 0

    def check_deadline(self):
        if time.monotonic() >= min(self.deadline, self.task_deadline):
            raise FlowError("deadline")

    def observe(self, label, *, purpose="full"):
        self.check_deadline()
        snap = self.snapshots.pop(0)
        self.observation_count += 1
        captured = time.monotonic() + self.observation_count * .000001
        snap.observations["observed_at_monotonic"] = captured
        snap.observations["requested_at_monotonic"] = captured
        snap.observations["frame_id"] = f"fake-capture-{self.observation_count}"
        self.last_snapshot = snap
        return snap

    def tap(self, snapshot, point, *, reason):
        self.check_deadline()
        if snapshot is not self.last_snapshot:
            raise FlowError("stale")
        self.taps.append((tuple(point), reason))
        self.last_snapshot = None

    def click_template(self, snapshot, name, *, roi):
        if snapshot is not self.last_snapshot:
            raise FlowError("stale")
        self.template_clicks.append((name, roi))
        self.last_snapshot = None

    def event(self, kind, **values):
        self.events.append((kind, values))


class EventRewardTests(unittest.TestCase):
    def test_anonymous_fixture_and_template_provenance(self):
        provenance = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
        for row in provenance["fixtures"].values():
            self.assertEqual(hashlib.sha256((ROOT / row["derived_path"]).read_bytes()).hexdigest(),
                             row["derived_sha256"])
        for row in provenance["assets"].values():
            self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["sha256"])
        for row in provenance["ocr_fixtures"].values():
            self.assertEqual(hashlib.sha256((ROOT / row["derived_path"]).read_bytes()).hexdigest(),
                             row["derived_sha256"])
        roundtrip = json.loads((FIXTURES / "roundtrip_20261016_provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(roundtrip["human_review"])
        self.assertFalse(roundtrip["formal_acceptance"])
        for row in roundtrip["fixtures"].values():
            self.assertEqual(hashlib.sha256((ROOT / row["derived_path"]).read_bytes()).hexdigest(),
                             row["derived_sha256"])
            self.assertEqual(hashlib.sha256((ROOT / row["ocr_path"]).read_bytes()).hexdigest(),
                             row["ocr_sha256"])

    def test_real_before_animation_and_later_empty_states(self):
        cases = (("shiny_before", "shiny_before", "claimable", "claimable"),
                 ("shiny_animation", "shiny_before", "unknown", "claimable"),
                 ("glowy_before", "glowy_before", "no_opportunity", "claimable"),
                 ("glowy_animation", "glowy_before", "no_opportunity", "unknown"),
                 ("both_empty", "both_empty", "no_opportunity", "no_opportunity"))
        for frame, ocr, shiny, glowy in cases:
            with self.subTest(frame=frame):
                result = observe_reward_track(FIXTURES / f"{frame}.png", fixture_ocr(ocr),
                                              client_version="18.647.2")
                self.assertIsNotNone(result)
                self.assertEqual(result["targets"]["shiny"]["status"], shiny)
                self.assertEqual(result["targets"]["glowy"]["status"], glowy)

    def test_real_shifted_track_is_closable_but_original_targets_remain_unknown(self):
        result = observe_reward_track(FIXTURES / "roundtrip_track_shifted.png",
                                      fixture_ocr("roundtrip_track_shifted"), client_version="18.647.2")
        self.assertIsNotNone(result)
        self.assertFalse(result["position_verified"])
        self.assertEqual([result["targets"][name]["status"] for name in ("shiny", "glowy")],
                         ["unknown", "unknown"])
        snapshot = recognized("roundtrip_track_shifted")
        self.assertEqual(snapshot.scene, "popup")
        self.assertEqual([button["name"] for button in snapshot.observations["buttons"]],
                         ["close_event_rewards"])
        session = FakeSession([snapshot, SceneSnapshot("village", .95, FIXTURES / "roundtrip_panel_open.png",
                              {"village_type": "home", "resource_source": "village_inventory",
                               "buttons": [{"name": "attack"}, {"name": "shop"}]})])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            closed = close_reward_track(session)
        self.assertEqual(closed["status"], "home")
        self.assertEqual(len(session.taps), 1)

    def test_missing_original_milestone_never_becomes_claim_or_no_opportunity(self):
        old = fixture_ocr("shiny_before")
        omitted = [item for item in old if item["text"] not in {"600", "95"}]
        result = observe_reward_track(FIXTURES / "shiny_before.png", omitted, client_version="18.647.2")
        self.assertIsNotNone(result)
        self.assertFalse(result["position_verified"])
        self.assertTrue(all(row["status"] == "unknown" and row["button"] is None
                            for row in result["targets"].values()))
        shifted = FakeSession([recognized("roundtrip_track_shifted")])
        with patch("autococ.event_rewards.foreground_package", return_value=shifted.config.game.package_name):
            receipt = claim_one_free_reward(shifted, "shiny", enabled=True)
        self.assertEqual(receipt["status"], "unknown")
        self.assertFalse(receipt["ui_claim_acknowledged"])
        self.assertEqual(shifted.taps, [])

    def test_real_open_entry_and_conflicting_or_displaced_labels(self):
        home = SceneSnapshot("village", .95, FIXTURES / "roundtrip_panel_open.png",
                             {"village_type": "home", "resource_source": "village_inventory",
                              "buttons": [{"name": "attack"}, {"name": "shop"}]})
        panel = recognized("roundtrip_panel_open")
        shifted = recognized("roundtrip_track_shifted")
        session = FakeSession([home, panel, shifted])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            opened = open_reward_track(session)
        self.assertEqual(opened["status"], "opened")
        self.assertEqual(opened["entry_label"], "打开")
        self.assertEqual(session.taps, [((950, 580), "Open confirmed current event reward track")])

        for error in ("conflict", "displaced"):
            with self.subTest(error=error):
                panel = recognized("roundtrip_panel_open")
                if error == "conflict":
                    panel.observations["ocr"].append({"text": "领取", "confidence": .999,
                                                       "bbox": [926, 564, 974, 596]})
                else:
                    next(item for item in panel.observations["ocr"] if item["text"] == "打开")["bbox"] = [350, 564, 398, 596]
                rejected = FakeSession([home, panel])
                with patch("autococ.event_rewards.foreground_package",
                           return_value=rejected.config.game.package_name):
                    with self.assertRaises(FlowError):
                        open_reward_track(rejected)
                self.assertEqual(rejected.taps, [])

    def test_reward_track_requires_current_page_and_correct_target_evidence(self):
        path = FIXTURES / "shiny_before.png"
        base = fixture_ocr("shiny_before")
        self.assertIsNone(observe_reward_track(path, base, client_version="18.647.3"))
        for missing in ("裂隙之乱", "奖励清单", "购买活动令牌"):
            with self.subTest(missing=missing):
                self.assertIsNone(observe_reward_track(path, [x for x in base if x["text"] != missing],
                                                       client_version="18.647.2"))
        wrong = [dict(x, text="1251") if x["text"] == "1250" else x for x in base]
        self.assertEqual(observe_reward_track(path, wrong, client_version="18.647.2")
                         ["targets"]["shiny"]["status"], "unknown")
        duplicate = base + [next(x for x in base if x["text"] == "领取" and x["bbox"][0] < 800)]
        self.assertEqual(observe_reward_track(path, duplicate, client_version="18.647.2")
                         ["targets"]["shiny"]["status"], "unknown")
        # The checked milestone is visible before claiming; blanked OCR cannot turn it into a receipt.
        no_button = [x for x in base if x["text"] != "领取"]
        self.assertEqual(observe_reward_track(path, no_button, client_version="18.647.2")
                         ["targets"]["shiny"]["status"], "unknown")
        with TemporaryDirectory() as tmp:
            image = cv2.imread(str(path))
            image[26:82, 1092:1145] = 0
            hidden = Path(tmp) / "hidden.png"
            cv2.imwrite(str(hidden), image)
            self.assertIsNone(observe_reward_track(hidden, base, client_version="18.647.2"))
            image = cv2.imread(str(path))
            image[580:660, 675:800] = 0
            cv2.imwrite(str(hidden), image)
            self.assertNotEqual(observe_reward_track(hidden, base, client_version="18.647.2")
                                ["targets"]["shiny"]["status"], "claimable")

    def test_screenshot_recognizer_exposes_only_close_and_fails_closed_on_account(self):
        provider = Mock()
        texts = [OCRText(x["text"], x["confidence"], tuple(x["bbox"]))
                 for x in fixture_ocr("shiny_before")]
        provider.recognize.return_value = texts
        recognizer = ScreenshotRecognizer(provider=provider)
        recognizer.client_version = "18.647.2"
        path = FIXTURES / "shiny_before.png"
        current = recognizer.recognize(path)
        self.assertEqual(current.scene, "popup")
        self.assertEqual([x["name"] for x in current.observations["buttons"]], ["close_event_rewards"])
        self.assertIsNone(current.observations["resource_source"])
        self.assertTrue(all(x is None for x in current.observations["resources"].values()))
        wrong_version = ScreenshotRecognizer(provider=provider)
        wrong_version.client_version = "18.647.3"
        self.assertIsNone(wrong_version.recognize(path).observations["event_reward_track"])
        dialog = [OCRText("连接断开", .999, (330, 285, 426, 323)),
                  OCRText("另一个设备正在连接这个村庄。", .999, (330, 334, 576, 358)),
                  OCRText("重新载入", .999, (330, 408, 399, 430))]
        with patch("autococ.event_rewards.observe_reward_track", wraps=observe_reward_track) as observer:
            provider.recognize.return_value = texts + dialog
            full = recognizer.recognize(path)
            self.assertEqual(full.scene, "account_in_use")
            self.assertIsNone(full.observations["event_reward_track"])
            self.assertEqual(observer.call_count, 0)
            provider.recognize.return_value = texts + dialog[1:2]
            partial = recognizer.recognize(path)
            self.assertEqual(partial.scene, "unknown")
            self.assertIsNone(partial.observations["event_reward_track"])
            self.assertEqual(observer.call_count, 0)

    def test_one_claim_has_two_fresh_stable_ui_frames_but_no_inventory_credit(self):
        before = recognized("glowy_before")
        animation = recognized("glowy_animation", ocr_name="glowy_before")
        empty1 = recognized("both_empty", ocr_name="both_empty")
        empty2 = recognized("both_empty", ocr_name="both_empty")
        session = FakeSession([before, animation, empty1, empty2])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            result = claim_one_free_reward(session, "glowy", enabled=True)
        self.assertEqual(result["status"], "ui_acknowledged")
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(session.taps[0][0], (1047, 621))
        self.assertTrue(result["ui_claim_acknowledged"])
        self.assertFalse(result["reward_inventory_verified"])
        self.assertFalse(result["formal_success"])
        self.assertEqual([x["status"] for x in result["frames"]],
                         ["claimable", "unknown", "no_opportunity", "no_opportunity"])

    def test_unknown_stop_foreground_source_and_no_opportunity_never_tap(self):
        with self.assertRaises(FlowError):
            claim_one_free_reward(FakeSession([]), "shiny")
        for snapshot in (recognized("shiny_animation", ocr_name="shiny_before"),
                         recognized("both_empty", ocr_name="both_empty")):
            session = FakeSession([snapshot])
            with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
                result = claim_one_free_reward(session, "shiny", enabled=True)
            self.assertEqual(result["status"],
                             "unknown" if snapshot.observations["event_reward_track"]["targets"]["shiny"]["status"] == "unknown"
                             else "no_opportunity")
            self.assertEqual(session.taps, [])
        session = FakeSession([recognized("shiny_before")])
        with patch("autococ.event_rewards.foreground_package", return_value="wrong.package"):
            result = claim_one_free_reward(session, "shiny", enabled=True)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(session.taps, [])
        session = FakeSession([recognized("shiny_before")])
        with patch("autococ.event_rewards.foreground_package", side_effect=StopRequested("stop")):
            with self.assertRaises(StopRequested):
                claim_one_free_reward(session, "shiny", enabled=True)
        self.assertEqual(session.taps, [])
        session = FakeSession([recognized("shiny_before")])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name), \
             patch("autococ.event_rewards._sha256", side_effect=["changed", "changed"]):
            result = claim_one_free_reward(session, "shiny", enabled=True)
        self.assertEqual(session.taps, [])
        self.assertEqual(result["status"], "unknown")

    def test_tampered_close_wrong_baseline_and_reused_capture_are_not_authorized(self):
        current = recognized("shiny_before")
        current.observations["event_reward_track"]["button"]["point"] = [1000, 600]
        session = FakeSession([current])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            result = claim_one_free_reward(session, "shiny", enabled=True)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(session.taps, [])
        session = FakeSession([recognized("shiny_before")])
        session.config.game.baseline_resolution = (1920, 1080)
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            result = claim_one_free_reward(session, "shiny", enabled=True)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(session.taps, [])

        class ReusedCapture(FakeSession):
            def observe(self, label, *, purpose="full"):
                snapshot = super().observe(label, purpose=purpose)
                if self.observation_count == 3:
                    snapshot.observations["frame_id"] = "fake-capture-2"
                return snapshot

        session = ReusedCapture([recognized("glowy_before"),
                                 recognized("both_empty", ocr_name="both_empty"),
                                 recognized("both_empty", ocr_name="both_empty")])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            result = claim_one_free_reward(session, "glowy", enabled=True)
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["ui_claim_acknowledged"])
        self.assertTrue(any("not independent" in reason for reason in result["reasons"]))

    def test_same_session_target_is_one_shot_even_after_uncertain_transport(self):
        class UncertainInput(FakeSession):
            def tap(self, snapshot, point, *, reason):
                super().tap(snapshot, point, reason=reason)
                raise FlowError("transport result uncertain")

        session = UncertainInput([recognized("shiny_before"), recognized("shiny_before")])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            first = claim_one_free_reward(session, "shiny", enabled=True)
            second = claim_one_free_reward(session, "shiny", enabled=True)
        self.assertTrue(first["input_attempted"])
        self.assertFalse(first["input_sent"])
        self.assertEqual(second["status"], "unknown")
        self.assertIn("already_attempted_in_session", second["reasons"])
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(len(session.snapshots), 1)

    def test_navigation_requires_known_pages_and_close_requires_home(self):
        home = SceneSnapshot("village", .95, FIXTURES / "entry_panel.png",
                             {"village_type": "home", "resource_source": "village_inventory",
                              "buttons": [{"name": "attack"}, {"name": "shop"}]})
        panel = recognized("entry_panel")
        track = recognized("shiny_before")
        session = FakeSession([home, panel, track])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            opened = open_reward_track(session)
        self.assertEqual(opened["status"], "opened")
        self.assertEqual(len(session.template_clicks), 1)
        self.assertEqual(len(session.taps), 1)
        self.assertEqual(session.taps[0][0], (950, 580))
        self.assertFalse(opened["claim_attempted"])
        session = FakeSession([recognized("both_empty", ocr_name="both_empty"), home])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            closed = close_reward_track(session)
        self.assertEqual(closed["status"], "home")
        self.assertEqual(len(session.taps), 1)
        unknown = SceneSnapshot("unknown", 0, FIXTURES / "entry_panel.png", {"ocr": []})
        bad = FakeSession([home, unknown, unknown, unknown])
        with patch("autococ.event_rewards.foreground_package", return_value=bad.config.game.package_name):
            with self.assertRaises(FlowError):
                open_reward_track(bad)
        self.assertEqual(bad.taps, [])

    def test_navigation_rejects_wrong_account_and_late_observation(self):
        home = SceneSnapshot("village", .95, FIXTURES / "entry_panel.png",
                             {"village_type": "home", "resource_source": "village_inventory",
                              "buttons": [{"name": "attack"}, {"name": "shop"}]})
        wrong = FakeSession([home], version="18.647.3")
        with self.assertRaises(FlowError):
            open_reward_track(wrong)
        self.assertEqual(wrong.template_clicks, [])
        account = FakeSession([SceneSnapshot("account_in_use", .95, FIXTURES / "entry_panel.png", {})])
        with patch("autococ.event_rewards.foreground_package", return_value=account.config.game.package_name):
            with self.assertRaises(FlowError):
                open_reward_track(account)
        self.assertEqual(account.template_clicks, [])
        hidden_account = recognized("entry_panel")
        hidden_account.observations["account_in_use"] = {"message": "另一个设备正在连接这个村庄。"}
        account_behind_panel = FakeSession([home, hidden_account])
        with patch("autococ.event_rewards.foreground_package",
                   return_value=account_behind_panel.config.game.package_name):
            with self.assertRaises(FlowError):
                open_reward_track(account_behind_panel)
        self.assertEqual(account_behind_panel.taps, [])

        class ExpiringSession(FakeSession):
            def observe(self, label, *, purpose="full"):
                snapshot = super().observe(label, purpose=purpose)
                self.task_deadline = time.monotonic() - .1
                return snapshot

        expired = ExpiringSession([home])
        with patch("autococ.event_rewards.foreground_package", return_value=expired.config.game.package_name):
            with self.assertRaises(FlowError):
                open_reward_track(expired)
        self.assertEqual(expired.template_clicks, [])


if __name__ == "__main__":
    unittest.main()
