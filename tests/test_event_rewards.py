"""Versioned event reward evidence and one-input opt-in behavior."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import cv2

from autococ.errors import FlowError, StopRequested
from autococ.event_rewards import (EventPanelEntryUnconfirmed, claim_one_free_reward,
                                   close_reward_track, close_unentered_event_panel,
                                   locate_reward_track_position, observe_reward_track,
                                   open_reward_track)
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


def recognized_with_ocr(path: Path, rows: list[dict]) -> SceneSnapshot:
    provider = Mock()
    provider.recognize.return_value = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                                       for item in rows]
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    return recognizer.recognize(path)


def previous_event_panel() -> SceneSnapshot:
    previous = recognized("entry_panel")
    previous.observations["requested_at_monotonic"] = time.monotonic() - 1
    previous.observations["frame_id"] = "previous-event-list"
    return previous


def upper_row(*items: tuple[str, tuple[int, int, int, int]]) -> list[dict]:
    return [{"text": text, "bbox": list(box), "confidence": .999}
            for text, box in items]


def shifted_live_ocr() -> list[dict]:
    base = [item for item in fixture_ocr("both_empty")
            if not (item["bbox"][1] < 285 and item["text"] in
                    {"600", "95", "裂隙奖章", "璀璨矿石", "奖励清单"})]
    return base + upper_row(
        ("600", (234, 222, 284, 246)),
        ("95", (549, 220, 583, 246)),
        ("璀璨矿石", (526, 248, 593, 272)),
        ("25", (856, 222, 887, 246)),
        ("星辉矿石", (834, 248, 902, 272)),
        ("奖励清单裂隙奖章", (180, 246, 288, 271)),
    )


def initial_view_ocr() -> list[dict]:
    base = [item for item in fixture_ocr("both_empty")
            if not (item["bbox"][1] < 285 and item["text"] in
                    {"600", "95", "裂隙奖章", "璀璨矿石"})]
    return base + upper_row(
        ("100", (414, 220, 464, 246)),
        ("璀璨矿石", (401, 248, 470, 272)),
        ("800", (714, 220, 764, 246)),
        ("裂隙奖章", (701, 248, 770, 272)),
        ("12", (1032, 220, 1064, 246)),
    )


def terminal_view_112226_ocr() -> list[dict]:
    # Only the reviewed upper-row OCR/geometry is retained. The lower-row
    # glowy 120 is deliberately not a claim proof in this navigation test.
    rows = upper_row(
        ("800", (412, 220, 465, 246)),
        ("裂隙奖章", (400, 248, 468, 271)),
        ("120", (724, 220, 766, 245)),
        ("璀璨矿石", (706, 248, 774, 272)),
        ("35", (1038, 224, 1058, 242)),
        ("120", (721, 469, 762, 494)),
        ("璀璨矿石", (703, 498, 771, 522)),
    )
    rows[4]["confidence"] = .98671
    return rows


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
        self.context = Mock()
        self._input_attempt_generation = 0
        self.action_count = 0

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
    def test_upper_event_card_from_anonymized_105007_frame_opens_track(self):
        proof = json.loads((FIXTURES / "panel_upper_105007.provenance.json").read_text(encoding="utf-8"))
        fixture = FIXTURES / "panel_upper_105007.png"
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), proof["fixture_sha256"])
        self.assertEqual(hashlib.sha256((FIXTURES / "panel_upper_105007.ocr.json").read_bytes()).hexdigest(),
                         proof["ocr_fixture_sha256"])
        home = SceneSnapshot("village", .95, FIXTURES / "entry_panel.png",
                             {"village_type": "home", "resource_source": "village_inventory",
                              "buttons": [{"name": "attack"}, {"name": "shop"}]})
        session = FakeSession([home, recognized("panel_upper_105007"), recognized("shiny_before")])
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            opened = open_reward_track(session)
        self.assertEqual(opened["status"], "opened")
        self.assertEqual(opened["entry_label"], "领取")
        self.assertEqual(session.taps, [((950, 366), "Open confirmed current event reward track")])
        self.assertFalse(opened["formal_success"])

    def test_upper_event_card_rejects_ambiguous_or_unrelated_ocr(self):
        from autococ import event_rewards as module
        original = fixture_ocr("panel_upper_105007")
        self.assertEqual(module._event_card_entry(original)[2], "upper")
        for change in (lambda rows: rows.append(dict(next(x for x in rows if x["text"] == "领取"))),
                       lambda rows: rows.append(dict(next(x for x in rows if x["text"] == "裂隙之乱"))),
                       lambda rows: rows.append(dict(next(x for x in rows if x["text"] == "裂隙之乱"),
                                                     bbox=[220, 352, 292, 374])),
                       lambda rows: rows.append(dict(next(x for x in rows if x["text"] == "16073/16000"))),
                       lambda rows: next(x for x in rows if x["text"] == "16073/16000").update(text="15999/16000"),
                       lambda rows: next(x for x in rows if x["text"] == "16073/16000").update(text="16073/0"),
                       lambda rows: next(x for x in rows if x["text"] == "领取").update(bbox=[700, 351, 748, 382]),
                       lambda rows: next(x for x in rows if x["text"] == "裂隙之乱").update(confidence=.8)):
            with self.subTest(change=change):
                rows = json.loads(json.dumps(original))
                change(rows)
                self.assertIsNone(module._event_card_entry(rows))

    def test_confirmed_list_without_entry_proof_can_only_close_its_x(self):
        home = SceneSnapshot("village", .95, FIXTURES / "entry_panel.png",
                             {"village_type": "home", "resource_source": "village_inventory",
                              "buttons": [{"name": "attack"}, {"name": "shop"}], "ocr": []})
        panel = recognized("panel_upper_105007")
        panel.observations["ocr"] = [row for row in panel.observations["ocr"]
                                     if row["text"] != "16073/16000"]
        home_after1 = SceneSnapshot("village", .95, FIXTURES / "roundtrip_panel_open.png",
                                    dict(home.observations))
        home_after2 = SceneSnapshot("village", .95, FIXTURES / "roundtrip_track_shifted.png",
                                    dict(home.observations))
        with TemporaryDirectory() as tmp:
            fresh_path = Path(tmp) / "new-event-panel.png"
            copyfile(FIXTURES / "panel_upper_105007.png", fresh_path)
            fresh_panel = recognized_with_ocr(fresh_path, fixture_ocr("panel_upper_105007"))
            session = FakeSession([home, panel, fresh_panel, home_after1, home_after2])
            with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
                with self.assertRaises(EventPanelEntryUnconfirmed):
                    open_reward_track(session)
                self.assertEqual(session.taps, [])
                closed = close_unentered_event_panel(session)
        self.assertEqual(closed["status"], "home")
        self.assertEqual(session.taps, [((1055, 53), "Close verified unentered event list")])

    def test_unentered_event_list_closes_once_only_with_two_new_home_frames(self):
        panel = recognized("panel_upper_105007")
        home1 = SceneSnapshot("village", .95, FIXTURES / "entry_panel.png",
                              {"village_type": "home", "resource_source": "village_inventory",
                               "buttons": [{"name": "attack"}, {"name": "shop"}], "ocr": []})
        home2 = SceneSnapshot("village", .95, FIXTURES / "roundtrip_panel_open.png",
                              {"village_type": "home", "resource_source": "village_inventory",
                               "buttons": [{"name": "attack"}, {"name": "shop"}], "ocr": []})
        session = FakeSession([panel, home1, home2])
        session.last_snapshot = previous_event_panel()
        with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
            closed = close_unentered_event_panel(session)
        self.assertEqual(closed["status"], "home")
        self.assertEqual(len(closed["home_evidence"]), 2)
        self.assertEqual(session.taps, [((1055, 53), "Close verified unentered event list")])
        self.assertFalse(closed["formal_success"])

        repeated = FakeSession([recognized("panel_upper_105007"), home1, home1])
        repeated.last_snapshot = previous_event_panel()
        with patch("autococ.event_rewards.foreground_package",
                   return_value=repeated.config.game.package_name):
            with self.assertRaisesRegex(FlowError, "HOME frame is not new"):
                close_unentered_event_panel(repeated)
        self.assertEqual(len(repeated.taps), 1)

        wrong = FakeSession([recognized("panel_upper_105007")], version="18.647.3")
        with self.assertRaises(FlowError):
            close_unentered_event_panel(wrong)
        self.assertEqual(wrong.taps, [])

        for change in (lambda snap: snap.observations["ocr"].append(
                            dict(next(x for x in snap.observations["ocr"] if x["text"] == "X"))),
                       lambda snap: snap.observations.update(account_in_use={"message": "blocked"}),
                       lambda snap: snap.observations["ocr"].clear()):
            with self.subTest(change=change):
                panel = recognized("panel_upper_105007")
                change(panel)
                refused = FakeSession([panel])
                refused.last_snapshot = previous_event_panel()
                with patch("autococ.event_rewards.foreground_package",
                           return_value=refused.config.game.package_name):
                    with self.assertRaises(FlowError):
                        close_unentered_event_panel(refused)
                self.assertEqual(refused.taps, [])

        stopped = FakeSession([recognized("panel_upper_105007")])
        stopped.check_deadline = Mock(side_effect=StopRequested("stop"))
        with self.assertRaises(StopRequested):
            close_unentered_event_panel(stopped)
        self.assertEqual(stopped.taps, [])

    def test_position_pair_rejects_missing_or_nonfinite_source_identity(self):
        from autococ import event_rewards as module

        offset, pairs = module._upper_position(fixture_ocr("shifted_after_swipe_2"))
        self.assertEqual(offset, -181)
        self.assertEqual({row["amount"] for row in pairs}, {"95", "25"})

        first = Mock(screenshot_path=Path("first.png"), observations={
            "requested_at_monotonic": 1.0, "frame_id": "first"})
        second = Mock(screenshot_path=Path("second.png"), observations={
            "requested_at_monotonic": 2.0, "frame_id": "second"})
        self.assertTrue(module._fresh_pair(first, second))
        for field, value in (("requested_at_monotonic", float("nan")),
                             ("requested_at_monotonic", float("inf")),
                             ("frame_id", ""), ("frame_id", None)):
            with self.subTest(field=field, value=value):
                bad = dict(second.observations, **{field: value})
                self.assertFalse(module._fresh_pair(first, Mock(
                    screenshot_path=Path("second.png"), observations=bad)))
        self.assertEqual(module._start_position(initial_view_ocr())[0], "100_800")
        obscured_third = [row for row in initial_view_ocr() if row["text"] != "12"]
        self.assertIsNotNone(module._start_position(obscured_third))
        verified = {"position_verified": True, "position_offset_px": -180,
                    "targets": {"shiny": {"visible": True}}}
        jittered = {**verified, "position_offset_px": -181}
        self.assertTrue(module._stable_position(verified, jittered, "shiny"))
        self.assertFalse(module._stable_position(verified,
            {**verified, "position_offset_px": -182}, "shiny"))

    def test_reviewed_112226_terminal_viewport_binds_only_navigation_evidence(self):
        from autococ import event_rewards as module

        rows = terminal_view_112226_ocr()
        start = module._start_position(rows)
        self.assertEqual(start[0], "800_120_35")
        self.assertEqual(start[1], (438.5, 745))
        # The old claim position remains unproved on this viewport.
        self.assertIsNone(module._upper_position(rows)[0])
        for name, change in (
            ("missing_35", lambda values: values.pop(4)),
            ("weak_35", lambda values: values[4].update(confidence=.94)),
            ("duplicate_800", lambda values: values.append(dict(values[0]))),
            ("wrong_lower_row_only", lambda values: values[2].update(bbox=[724, 469, 766, 495])),
            ("wrong_upper_label", lambda values: values[3].update(text="星辉矿石")),
            ("offset_120", lambda values: values[2].update(bbox=[765, 220, 807, 245])),
        ):
            with self.subTest(name=name):
                modified = json.loads(json.dumps(rows))
                change(modified)
                self.assertIsNone(module._start_position(modified))

    def test_terminal_viewport_swipe_is_shiny_only_and_still_needs_old_after_position(self):
        # This checks the one-swipe control flow with fake observations, not
        # pixel equivalence to the 112226 frames or a completed live claim.
        def snapshot(name, ocr, *, positioned=False):
            return SceneSnapshot("popup", .95, FIXTURES / f"{name}.png", {
                "ocr": ocr,
                "event_reward_track": {
                    "position_verified": positioned,
                    "position_offset_px": 0 if positioned else None,
                    "targets": {"shiny": {"visible": positioned},
                                "glowy": {"visible": False}},
                },
            })

        before = lambda: [
            snapshot("shiny_before", terminal_view_112226_ocr()),
            snapshot("both_empty", terminal_view_112226_ocr()),
        ]
        wrong_target = FakeSession(before())
        with patch("autococ.event_rewards.foreground_package",
                   return_value=wrong_target.config.game.package_name), \
             patch("autococ.event_rewards._require_current", return_value={"status": "unknown"}):
            rejected = locate_reward_track_position(
                wrong_target, "glowy", enabled=True, task_id="free_glowy_ore")
        self.assertEqual(rejected["status"], "unknown")
        self.assertIn("terminal_viewport_target_not_supported", rejected["reasons"])
        wrong_target.context.swipe.assert_not_called()

        session = FakeSession(before() + [
            snapshot("glowy_before", [], positioned=True),
            snapshot("glowy_animation", [], positioned=True),
        ])
        with patch("autococ.event_rewards.foreground_package",
                   return_value=session.config.game.package_name), \
             patch("autococ.event_rewards._require_current", return_value={"status": "unknown"}):
            located = locate_reward_track_position(
                session, "shiny", enabled=True, task_id="free_shiny_ore")
        self.assertEqual(located["status"], "positioned")
        self.assertEqual(located["starting_viewport"], "800_120_35")
        self.assertEqual(session.context.swipe.call_count, 1)
        self.assertEqual(session.action_count, 1)

    def test_merged_list_text_needs_independent_icon_and_two_position_pairs(self):
        from autococ import event_rewards as module

        frame = FIXTURES / "shifted_after_swipe_2.png"
        rows = fixture_ocr("shifted_after_swipe_2")
        result = observe_reward_track(frame, rows, client_version="18.647.2")
        self.assertIsNotNone(result)
        self.assertEqual(result["anchors"]["list"]["method"], "fixed_icon_template")
        self.assertTrue(result["position_verified"])
        self.assertEqual(result["position_offset_px"], -181)
        self.assertEqual({row["amount"] for row in result["position_basis"]}, {"95", "25"})
        self.assertEqual(result["targets"]["shiny"]["rois"]["empty_roi"][0],
                         682 + result["position_offset_px"])
        original_score = module._template_score
        with patch("autococ.event_rewards._template_score",
                   side_effect=lambda image, name, roi: 0.0 if name == "list_icon"
                   else original_score(image, name, roi)):
            self.assertIsNone(observe_reward_track(frame, rows, client_version="18.647.2"))
        duplicate = rows + [dict(next(row for row in rows if row["text"] == "25"))]
        self.assertFalse(observe_reward_track(frame, duplicate, client_version="18.647.2")
                         ["position_verified"])
        missing = [row for row in rows if row["text"] != "星辉矿石"]
        self.assertFalse(observe_reward_track(frame, missing, client_version="18.647.2")
                         ["position_verified"])
        no_leading_number = [row for row in rows if row["text"] != "600"]
        self.assertFalse(observe_reward_track(frame, no_leading_number, client_version="18.647.2")
                         ["position_verified"])
        conflicting = [dict(row, bbox=[number + 20 if index in (0, 2) else number
                                       for index, number in enumerate(row["bbox"])])
                       if row["text"] in {"25", "星辉矿石"} else row for row in rows]
        self.assertFalse(observe_reward_track(frame, conflicting, client_version="18.647.2")
                         ["position_verified"])

    def test_no_opportunity_needs_two_distinct_stable_source_frames(self):
        with TemporaryDirectory() as tmp:
            first_path, second_path = (Path(tmp) / "one.png", Path(tmp) / "two.png")
            for path in (first_path, second_path):
                copyfile(FIXTURES / "both_empty.png", path)
            rows = fixture_ocr("both_empty")
            session = FakeSession([recognized_with_ocr(first_path, rows),
                                   recognized_with_ocr(second_path, rows)])
            with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
                result = claim_one_free_reward(session, "shiny", enabled=True)
            self.assertEqual(result["status"], "no_opportunity")
            self.assertEqual(len(result["frames"]), 2)
            self.assertEqual(session.taps, [])

    def test_live_shifted_no_opportunity_accepts_one_pixel_position_jitter(self):
        first = recognized("shifted_after_swipe_2")
        second = recognized("shifted_read_only_1")
        self.assertEqual(first.observations["event_reward_track"]["position_offset_px"], -181)
        self.assertEqual(second.observations["event_reward_track"]["position_offset_px"], -180)
        for target in ("shiny", "glowy"):
            with self.subTest(target=target):
                session = FakeSession([recognized("shifted_after_swipe_2"),
                                       recognized("shifted_read_only_1")])
                with patch("autococ.event_rewards.foreground_package",
                           return_value=session.config.game.package_name):
                    result = claim_one_free_reward(session, target, enabled=True)
                self.assertEqual(result["status"], "no_opportunity")
                self.assertEqual([row["status"] for row in result["frames"]],
                                 ["no_opportunity", "no_opportunity"])
                self.assertEqual(session.taps, [])

    def test_locator_one_prev_swipe_and_terminal_uncertainty(self):
        with TemporaryDirectory() as tmp:
            first_path, second_path = (Path(tmp) / "start-one.png", Path(tmp) / "start-two.png")
            for path in (first_path, second_path):
                copyfile(FIXTURES / "both_empty.png", path)
            first = recognized_with_ocr(first_path, initial_view_ocr())
            second = recognized_with_ocr(second_path, initial_view_ocr())
            session = FakeSession([first, second, recognized("shiny_before"), recognized("both_empty")])
            with patch("autococ.event_rewards.foreground_package", return_value=session.config.game.package_name):
                located = locate_reward_track_position(session, "shiny", enabled=True)
            self.assertEqual(located["status"], "positioned")
            self.assertEqual(located["offset_px"], 0)
            session.context.swipe.assert_called_once_with(340, 385, 920, 385, 550)
            self.assertFalse(located["claim_attempted"])

            uncertain = FakeSession([recognized_with_ocr(first_path, initial_view_ocr()),
                                     recognized_with_ocr(second_path, initial_view_ocr()),
                                     recognized_with_ocr(first_path, initial_view_ocr()),
                                     recognized_with_ocr(second_path, initial_view_ocr())])
            uncertain.context.swipe.side_effect = FlowError("transport uncertain")
            with patch("autococ.event_rewards.foreground_package", return_value=uncertain.config.game.package_name):
                failed = locate_reward_track_position(uncertain, "shiny", enabled=True)
                repeated = locate_reward_track_position(uncertain, "glowy", enabled=True,
                                                        task_id="other-task")
            self.assertEqual(failed["status"], "unknown")
            self.assertTrue(failed["input_attempted"])
            self.assertFalse(failed["input_sent"])
            self.assertIn("previous_navigation_transport_uncertain", repeated["reasons"])
            uncertain.context.swipe.assert_called_once()

            # A previous swipe must not hide a target already established in
            # two new frames (for example the next shiny/glowy task).
            positioned = FakeSession([recognized("shiny_before"), recognized("both_empty")])
            positioned._event_track_position_swipe_attempts = {("earlier", "shiny")}
            with patch("autococ.event_rewards.foreground_package",
                       return_value=positioned.config.game.package_name):
                result = locate_reward_track_position(positioned, "glowy", enabled=True,
                                                      task_id="next")
            self.assertEqual(result["status"], "positioned")
            positioned.context.swipe.assert_not_called()

    def test_second_task_can_swipe_after_track_reopens_at_start(self):
        with TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"start-{index}.png" for index in range(6)]
            for path in paths:
                copyfile(FIXTURES / "both_empty.png", path)
            starts = [recognized_with_ocr(path, initial_view_ocr()) for path in paths]
            session = FakeSession(starts[:2] + [recognized("shiny_before"),
                                                recognized("both_empty")]
                                  + starts[2:4] + [recognized("shiny_before"),
                                                    recognized("both_empty")]
                                  + starts[4:])
            with patch("autococ.event_rewards.foreground_package",
                       return_value=session.config.game.package_name):
                first = locate_reward_track_position(session, "shiny", enabled=True,
                                                     task_id="reward-shiny")
                second = locate_reward_track_position(session, "glowy", enabled=True,
                                                      task_id="reward-glowy")
                repeated = locate_reward_track_position(session, "shiny", enabled=True,
                                                        task_id="reward-shiny")
            self.assertEqual((first["status"], second["status"]), ("positioned", "positioned"))
            self.assertEqual(repeated["status"], "unknown")
            self.assertIn("one_swipe_already_attempted_for_task", repeated["reasons"])
            self.assertEqual(session.context.swipe.call_count, 2)
            self.assertEqual(session._event_track_position_swipe_attempts,
                             {("reward-shiny", "shiny"), ("reward-glowy", "glowy")})

    def test_locator_unknown_start_budget_and_stop_never_swipe(self):
        with TemporaryDirectory() as tmp:
            first_path, second_path = (Path(tmp) / "start-one.png", Path(tmp) / "start-two.png")
            for path in (first_path, second_path):
                copyfile(FIXTURES / "both_empty.png", path)
            start_rows = initial_view_ocr()
            wrong_rows = [dict(row, text="801") if row["text"] == "800" else row for row in start_rows]
            unknown = FakeSession([recognized_with_ocr(first_path, wrong_rows),
                                   recognized_with_ocr(second_path, wrong_rows)])
            with patch("autococ.event_rewards.foreground_package", return_value=unknown.config.game.package_name):
                result = locate_reward_track_position(unknown, "shiny", enabled=True)
            self.assertEqual(result["status"], "unknown")
            self.assertFalse(result["input_attempted"])
            unknown.context.swipe.assert_not_called()

            short = FakeSession([recognized_with_ocr(first_path, start_rows),
                                 recognized_with_ocr(second_path, start_rows)])
            short.task_deadline = time.monotonic() + 10
            with patch("autococ.event_rewards.foreground_package", return_value=short.config.game.package_name):
                result = locate_reward_track_position(short, "shiny", enabled=True)
            self.assertEqual(result["status"], "unknown")
            self.assertIn("insufficient_budget_for_one_swipe_and_after_frames", result["reasons"])
            short.context.swipe.assert_not_called()

            stopped = FakeSession([recognized_with_ocr(first_path, start_rows),
                                   recognized_with_ocr(second_path, start_rows)])
            with patch("autococ.event_rewards.foreground_package", side_effect=StopRequested("stop")):
                with self.assertRaises(StopRequested):
                    locate_reward_track_position(stopped, "shiny", enabled=True)
            stopped.context.swipe.assert_not_called()

    def test_anonymous_fixture_and_template_provenance(self):
        provenance = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
        calibration = provenance["position_calibration"]
        self.assertEqual(hashlib.sha256((ROOT / calibration["derived_ocr_path"]).read_bytes()).hexdigest(),
                         calibration["derived_ocr_sha256"])
        self.assertEqual(calibration["canonical_centers_x"]["25"], 1052)
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
        for missing in ("裂隙之乱", "购买活动令牌"):
            with self.subTest(missing=missing):
                self.assertIsNone(observe_reward_track(path, [x for x in base if x["text"] != missing],
                                                       client_version="18.647.2"))
        from autococ import event_rewards as module
        original_score = module._template_score
        with patch("autococ.event_rewards._template_score",
                   side_effect=lambda image, name, roi: 0.0 if name == "list_icon"
                   else original_score(image, name, roi)):
            self.assertIsNone(observe_reward_track(
                path, [x for x in base if x["text"] != "奖励清单"], client_version="18.647.2"))
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
            self.assertEqual(result["status"], "unknown")
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
