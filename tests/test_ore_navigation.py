"""Saved 18.647.2 blacksmith navigation evidence and bounded fake sessions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from autococ.errors import FlowError, StopRequested
from autococ.ocr import OCRText
from autococ.ore_inventory import observe_ore_inventory
from autococ.ore_navigation import (locate_blacksmith, read_current_ore_inventory,
                                     recognize_selected_blacksmith)
from autococ.scene import SceneSnapshot


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "blacksmith_navigation_18_647_2"
CAL_FIXTURES = ROOT / "tests" / "fixtures" / "blacksmith_navigation_123737_18_647_2"
SMALL_FIXTURES = ROOT / "tests" / "fixtures" / "blacksmith_navigation_small_120645_18_647_2"
ORE_FIXTURES = ROOT / "tests" / "fixtures" / "hero_equipment_18_647_2"
VERSION = "18.647.2"
TITLE_ROI = (535, 452, 740, 508)


def shot(name: str, *, frame_id: str | None = None) -> SceneSnapshot:
    observations = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    observations.update(frame_id=frame_id or name,
                        observed_at_monotonic=time.monotonic(),
                        baseline_resolution=[1280, 720])
    return SceneSnapshot(observations.pop("scene"), observations.pop("confidence"),
                         FIXTURES / f"{name}.png", observations)


def calibrated_shot(directory: Path, name: str) -> SceneSnapshot:
    observations = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    observations.update(frame_id=name, observed_at_monotonic=time.monotonic(),
                        baseline_resolution=[1280, 720])
    return SceneSnapshot(observations.pop("scene"), observations.pop("confidence"),
                         directory / f"{name}.png", observations)


def ore_shot(name: str, frame_id: str) -> SceneSnapshot:
    path = ORE_FIXTURES / f"{name}.png"
    ocr = json.loads((ORE_FIXTURES / f"{name}.ocr.json").read_text(encoding="utf-8"))
    observations = {"ocr": ocr, "frame_id": frame_id,
                    "observed_at_monotonic": time.monotonic(),
                    "baseline_resolution": [1280, 720], "account_in_use": None}
    observations["ore_inventory"] = observe_ore_inventory(path, ocr, client_version=VERSION)
    return SceneSnapshot("popup", .95, path, observations)


class FakeSession:
    def __init__(self, names: list[str]):
        self.names = list(names)
        self.last_snapshot = None
        self.client_version = VERSION
        self.deadline = self.task_deadline = time.monotonic() + 90
        self.config = Mock()
        self.config.game.baseline_resolution = (1280, 720)
        self.config.game.package_name = "com.supercell.clashofclans"
        self.config.runtime.step_timeout_sec = 5
        self.capture = Mock(adb=object(), serial="fake-serial")
        self.recognizer = Mock()
        self.recognizer.recognize_region.return_value = [OCRText("铁匠铺（10级）", .93856, TITLE_ROI)]
        self.taps = []
        self.observed = []

    def check_deadline(self):
        if time.monotonic() >= min(self.deadline, self.task_deadline):
            raise FlowError("deadline")

    def observe(self, label, *, purpose="full"):
        name = self.names.pop(0)
        self.observed.append(name)
        current = (ore_shot(name.removeprefix("ore_"), f"ore-{len(self.observed)}")
                   if name.startswith("ore_") else shot(name, frame_id=f"{name}-{len(self.observed)}"))
        self.last_snapshot = current
        return current

    def tap(self, current, point, *, reason):
        if current is not self.last_snapshot:
            raise FlowError("obsolete")
        self.taps.append((tuple(point), reason))
        self.last_snapshot = None


def give_each_capture_its_own_path(session: FakeSession, directory: Path) -> None:
    """Model saved pixels in distinct current captures without changing fixtures."""
    original = session.observe

    def observe(label, *, purpose="full"):
        snapshot = original(label, purpose=purpose)
        if snapshot.scene == "popup":
            return snapshot  # Ore evidence is already bound to its fixture source path.
        path = directory / f"{len(session.observed):02d}-{snapshot.screenshot_path.name}"
        path.write_bytes(snapshot.screenshot_path.read_bytes())
        fresh = SceneSnapshot(snapshot.scene, snapshot.confidence, path, snapshot.observations)
        session.last_snapshot = fresh
        return fresh

    session.observe = observe


def mark_captures_unknown(session: FakeSession, indices: set[int], *,
                          account: bool = False, conflict_ocr: bool = False,
                          stale: bool = False) -> None:
    """Return a real saved frame with a transient, incomplete scene reading."""
    original = session.observe

    def observe(label, *, purpose="full"):
        snapshot = original(label, purpose=purpose)
        if len(session.observed) not in indices:
            return snapshot
        observations = copy.deepcopy(snapshot.observations)
        observations["buttons"] = [{"name": "attack"}]
        if account:
            observations["account_in_use"] = {"message": True}
        if conflict_ocr:
            observations["ocr"].append({"text": "另一个设备正在连接这个村庄"})
        if stale:
            observations["observed_at_monotonic"] -= 10
        unknown = SceneSnapshot("unknown", snapshot.confidence,
                                snapshot.screenshot_path, observations)
        session.last_snapshot = unknown
        return unknown

    session.observe = observe


class OreNavigationTests(unittest.TestCase):
    def test_two_scale_train_fixture_provenance_and_anonymization(self):
        for directory in (CAL_FIXTURES, SMALL_FIXTURES):
            with self.subTest(directory=directory.name):
                provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["split"], "TRAIN")
                self.assertFalse(provenance["formal_acceptance"])
                self.assertFalse(provenance["account_verified"])
                self.assertEqual(len(provenance["prospective_train_declarations"]), 2)
                self.assertEqual(len({row["sha256"] for row in
                                      provenance["prospective_train_declarations"]}), 2)
                review = provenance["root_two_scale_review"]
                self.assertRegex(review["sha256"], r"^[0-9a-f]{64}$")
                template = provenance["template"]
                path = ROOT / template["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), template["sha256"])
                self.assertEqual(cv2.imread(str(path)).shape, (22, 18, 3))
                x1, y1, x2, y2 = provenance["transformation"]["retained_roi_xyxy"]
                for name, entry in provenance["fixtures"].items():
                    image = cv2.imread(str(ROOT / entry["derived_path"]))
                    self.assertEqual(image.shape, (720, 1280, 3))
                    self.assertEqual(hashlib.sha256((ROOT / entry["derived_path"]).read_bytes()).hexdigest(),
                                     entry["derived_sha256"])
                    self.assertEqual(hashlib.sha256((ROOT / entry["context_path"]).read_bytes()).hexdigest(),
                                     entry["context_sha256"])
                    self.assertEqual(hashlib.sha256(image[y1:y2, x1:x2].tobytes()).hexdigest(),
                                     entry["retained_roi_pixel_sha256"])
                    image[y1:y2, x1:x2] = (24, 24, 24)
                    self.assertTrue((image == (24, 24, 24)).all(), name)

    def test_verified_train_front_scales_and_obscuration(self):
        expected = ((CAL_FIXTURES, "home_calibrated", [753, 200], "verified_123737"),
                    (SMALL_FIXTURES, "home_small_07", [746, 210], "small_verified_120645"),
                    (SMALL_FIXTURES, "home_small_08", [746, 210], "small_verified_120645"),
                    (SMALL_FIXTURES, "home_small_13", [746, 210], "small_verified_120645"))
        for directory, name, point, asset_fragment in expected:
            with self.subTest(name=name):
                result = locate_blacksmith(calibrated_shot(directory, name), client_version=VERSION)
                self.assertIsNotNone(result)
                self.assertEqual(result["point"], point)
                self.assertIn(asset_fragment, result["template"])
                self.assertGreaterEqual(result["ncc"], .985)
                self.assertLessEqual(result["mad"], 3)
                self.assertGreaterEqual(result["ncc"] - result["runner_up"], .15)
        for directory, name in ((CAL_FIXTURES, "home_obscured"),
                                (SMALL_FIXTURES, "home_small_09"),
                                (SMALL_FIXTURES, "home_small_12")):
            with self.subTest(name=name):
                self.assertIsNone(locate_blacksmith(calibrated_shot(directory, name),
                                                     client_version=VERSION))

    def test_same_front_variants_are_one_candidate_but_distinct_fronts_conflict(self):
        current = calibrated_shot(SMALL_FIXTURES, "home_small_13")
        one = locate_blacksmith(current, client_version=VERSION)
        self.assertIsNotNone(one)
        from autococ import ore_navigation
        with patch.dict(ore_navigation._FRONTS,
                        {"same_front_copy": ore_navigation._FRONTS["small120645"]}):
            self.assertIsNotNone(locate_blacksmith(current, client_version=VERSION))
        source = cv2.imread(str(current.screenshot_path))
        template = cv2.imread(str(ROOT / "assets" / "templates" /
                                   "ore_navigation_blacksmith_front_small_verified_120645_18_647_2.png"))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "other-front.png"
            source[300:322, 500:518] = template
            self.assertTrue(cv2.imwrite(str(path), source))
            duplicate = SceneSnapshot(current.scene, current.confidence, path,
                                      copy.deepcopy(current.observations))
            self.assertIsNone(locate_blacksmith(duplicate, client_version=VERSION))

    def test_calibrated_front_rejects_neighbor_only_wrong_version_stale_and_outside_roi(self):
        current = calibrated_shot(SMALL_FIXTURES, "home_small_13")
        self.assertIsNone(locate_blacksmith(current, client_version="18.647.3"))
        current.observations["observed_at_monotonic"] -= 9
        self.assertIsNone(locate_blacksmith(current, client_version=VERSION))
        current = calibrated_shot(SMALL_FIXTURES, "home_small_13")
        source = cv2.imread(str(current.screenshot_path))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "neighbor.png"
            source[207:229, 751:769] = (24, 24, 24)
            self.assertTrue(cv2.imwrite(str(path), source))
            neighbor = SceneSnapshot(current.scene, current.confidence, path,
                                     copy.deepcopy(current.observations))
            self.assertIsNone(locate_blacksmith(neighbor, client_version=VERSION))
            template = cv2.imread(str(ROOT / "assets" / "templates" /
                                   "ore_navigation_blacksmith_front_small_verified_120645_18_647_2.png"))
            source[300:322, 100:118] = template
            self.assertTrue(cv2.imwrite(str(path), source))
            outside = SceneSnapshot(current.scene, current.confidence, path,
                                    copy.deepcopy(current.observations))
            self.assertIsNone(locate_blacksmith(outside, client_version=VERSION))

    def test_fixture_provenance_and_anonymization(self):
        p = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(p["formal_acceptance"])
        self.assertFalse(p["account_verified"])
        for entry in p["fixtures"].values():
            for path_key, hash_key in (("derived_path", "derived_sha256"),
                                       ("context_path", "context_sha256")):
                data = (ROOT / entry[path_key]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry[hash_key])
            image = cv2.imread(str(ROOT / entry["derived_path"]))
            self.assertEqual(image.shape[:2], (720, 1280))
            retained = {"home_zone": p["transformation"]["home_roi_xyxy"]}
            if "selected_menu" in entry["retained_roi_pixel_sha256"]:
                retained["selected_menu"] = p["transformation"]["selected_menu_roi_xyxy"]
            masked = image.copy()
            for name, (x1, y1, x2, y2) in retained.items():
                self.assertEqual(hashlib.sha256(image[y1:y2, x1:x2].tobytes()).hexdigest(),
                                 entry["retained_roi_pixel_sha256"][name])
                masked[y1:y2, x1:x2] = (24, 24, 24)
            self.assertTrue((masked == (24, 24, 24)).all())
        for template in p["templates"].values():
            self.assertEqual(hashlib.sha256((ROOT / template["path"]).read_bytes()).hexdigest(),
                             template["sha256"])

    def test_unique_visible_front_derives_current_point_and_occlusion_rejects(self):
        for name, point in (("home_old", [756, 212]),
                            ("home_new", [759, 207]),
                            ("home_visible", [759, 207])):
            with self.subTest(name=name):
                result = locate_blacksmith(shot(name), client_version=VERSION)
                self.assertIsNotNone(result)
                self.assertEqual(result["point"], point)
                self.assertGreaterEqual(result["ncc"], .985)
                self.assertFalse(result["formal_success"])
        for name in ("home_failed", "home_occluded"):
            self.assertIsNone(locate_blacksmith(shot(name), client_version=VERSION))

    def test_shifted_current_image_relocates_but_duplicate_or_zoom_rejects(self):
        source = cv2.imread(str(FIXTURES / "home_visible.png"))
        base = shot("home_visible")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "transformed.png"
            shifted = cv2.warpAffine(source, np.float32([[1, 0, 40], [0, 1, 25]]),
                                     (1280, 720), borderValue=(24, 24, 24))
            self.assertTrue(cv2.imwrite(str(path), shifted))
            moved = SceneSnapshot(base.scene, base.confidence, path, copy.deepcopy(base.observations))
            result = locate_blacksmith(moved, client_version=VERSION)
            self.assertIsNotNone(result)
            self.assertEqual(result["point"], [799, 232])

            duplicated = source.copy()
            duplicated[300:322, 500:518] = source[197:219, 758:776]
            self.assertTrue(cv2.imwrite(str(path), duplicated))
            duplicate = SceneSnapshot(base.scene, base.confidence, path, copy.deepcopy(base.observations))
            self.assertIsNone(locate_blacksmith(duplicate, client_version=VERSION))

            zoomed = cv2.warpAffine(source, cv2.getRotationMatrix2D((640, 360), 0, 1.08),
                                    (1280, 720), borderValue=(24, 24, 24))
            self.assertTrue(cv2.imwrite(str(path), zoomed))
            zoom = SceneSnapshot(base.scene, base.confidence, path, copy.deepcopy(base.observations))
            self.assertIsNone(locate_blacksmith(zoom, client_version=VERSION))

    def test_selected_title_and_equipment_require_text_and_icon(self):
        for name in ("selected_strong", "selected_mid"):
            with self.subTest(name=name):
                selected = recognize_selected_blacksmith(shot(name), client_version=VERSION)
                self.assertIsNotNone(selected)
                self.assertEqual(selected["title_source"], "full_ocr")
                self.assertEqual(selected["point"], [708, 562 if name == "selected_strong" else 561])
        weak = shot("selected_weak")
        self.assertIsNone(recognize_selected_blacksmith(weak, client_version=VERSION))
        reader = Mock(return_value=[{"text": "铁匠铺（10级）", "confidence": .93856,
                                     "bbox": list(TITLE_ROI)}])
        selected = recognize_selected_blacksmith(weak, client_version=VERSION, line_reader=reader)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["title_source"], "same_frame_line_roi")
        self.assertEqual(selected["full_title"]["confidence"], .89551)
        reader.assert_called_once_with(weak.screenshot_path, TITLE_ROI)
        for readings in ([],
                         [{"text": "铁匠铺（10级）", "confidence": .899,
                           "bbox": list(TITLE_ROI)}],
                         [{"text": "研究所（10级）", "confidence": .999,
                           "bbox": list(TITLE_ROI)}],
                         [{"text": "铁匠铺（10级）", "confidence": .999,
                           "bbox": list(TITLE_ROI)}] * 2):
            with self.subTest(readings=readings):
                self.assertIsNone(recognize_selected_blacksmith(
                    weak, client_version=VERSION, line_reader=Mock(return_value=readings)))
        conflicting = shot("selected_weak")
        conflicting.observations["ocr"].append({"text": "研究所（10级）", "confidence": .999,
                                                "bbox": [545, 460, 725, 499]})
        self.assertIsNone(recognize_selected_blacksmith(
            conflicting, client_version=VERSION, line_reader=reader))

        for mutation in (lambda s: s.observations["ocr"].__setitem__(0, {**s.observations["ocr"][0],
                                                                      "text": "研究所（10级）"}),
                         lambda s: s.observations["ocr"].append(copy.deepcopy(s.observations["ocr"][0])),
                         lambda s: s.observations["ocr"].__setitem__(2, {**s.observations["ocr"][2],
                                                                      "text": "升级"})):
            changed = shot("selected_strong")
            mutation(changed)
            self.assertIsNone(recognize_selected_blacksmith(changed, client_version=VERSION))
        self.assertIsNone(recognize_selected_blacksmith(shot("home_failed"), client_version=VERSION))

    def test_actual_immediate_post_tap_frame_is_transition_then_delayed_selection(self):
        immediate = shot("home_immediate_after_tap")
        delayed = shot("selected_delayed_after_tap")
        self.assertIsNone(recognize_selected_blacksmith(immediate, client_version=VERSION))
        self.assertIsNotNone(recognize_selected_blacksmith(delayed, client_version=VERSION))

    def test_wrong_version_account_foreground_and_stale_frames_never_tap(self):
        home = shot("home_visible")
        self.assertIsNone(locate_blacksmith(home, client_version="18.647.3"))
        self.assertIsNone(locate_blacksmith(home, client_version=VERSION,
                                            baseline_resolution=(1920, 1080)))
        home.observations["account_in_use"] = {"warning": True}
        self.assertIsNone(locate_blacksmith(home, client_version=VERSION))
        for mode in ("background", "stale", "changed_snapshot"):
            session = FakeSession(["home_visible"])
            def foreground(*args, **kwargs):
                if mode == "stale":
                    session.last_snapshot.observations["observed_at_monotonic"] -= 10
                if mode == "changed_snapshot":
                    session.last_snapshot = shot("home_new")
                return "other.package" if mode == "background" else session.config.game.package_name
            with patch("autococ.ore_navigation.foreground_package", side_effect=foreground):
                with self.assertRaises(FlowError):
                    read_current_ore_inventory(session)
            self.assertEqual(session.taps, [])

    def test_full_navigation_reads_two_fresh_frames_and_closes_home(self):
        session = FakeSession(["home_visible", "home_immediate_after_tap",
                               "selected_delayed_after_tap", "ore_first",
                               "ore_second", "home_new", "home_visible", "home_old"])
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=session.config.game.package_name), patch(
                   "autococ.ore_inventory.foreground_package",
                   return_value=session.config.game.package_name):
            result = read_current_ore_inventory(session)
        self.assertEqual(result["status"], "stable_inventory_read")
        self.assertEqual(result["inventory"]["ores"]["shiny"]["amount"], 15662)
        self.assertEqual(result["inventory"]["ores"]["starry"]["amount"], 77)
        self.assertEqual(result["close"]["status"], "home")
        self.assertEqual(len(result["home_evidence"]), 2)
        self.assertNotEqual(result["home_evidence"][0]["frame_id"],
                            result["home_evidence"][1]["frame_id"])
        self.assertEqual([point for point, _ in session.taps], [(759, 207), (708, 562), (1223, 81)])
        self.assertFalse(result["account_verified"])
        self.assertFalse(result["reward_inventory_verified"])
        self.assertFalse(result["formal_success"])

    def test_already_selected_shortcut_and_wrong_selection_stop(self):
        session = FakeSession(["selected_strong", "selected_mid", "ore_first", "ore_second", "home_new",
                               "home_visible", "home_old"])
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=session.config.game.package_name), patch(
                   "autococ.ore_inventory.foreground_package",
                   return_value=session.config.game.package_name):
            result = read_current_ore_inventory(session)
        self.assertTrue(result["returned_home"])
        self.assertEqual(len(session.taps), 2)

        wrong = FakeSession(["home_visible", "home_immediate_after_tap", "home_failed",
                             "home_occluded"])
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=wrong.config.game.package_name):
            with self.assertRaises(FlowError):
                read_current_ore_inventory(wrong)
        self.assertEqual([point for point, _ in wrong.taps], [(759, 207)])

        wrong_selected = FakeSession(["selected_strong"])
        original = wrong_selected.observe
        def changed(label, *, purpose="full"):
            value = original(label, purpose=purpose)
            value.observations["ocr"][0]["text"] = "研究所（10级）"
            return value
        wrong_selected.observe = changed
        with self.assertRaises(FlowError):
            read_current_ore_inventory(wrong_selected)
        self.assertEqual(wrong_selected.taps, [])

        wrong_after_tap = FakeSession(["home_visible", "selected_delayed_after_tap"])
        original = wrong_after_tap.observe
        def changed_after_tap(label, *, purpose="full"):
            value = original(label, purpose=purpose)
            if label == "ore-navigation-selected-1":
                value.observations["ocr"][0]["text"] = "研究所（10级）"
            return value
        wrong_after_tap.observe = changed_after_tap
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=wrong_after_tap.config.game.package_name):
            with self.assertRaises(FlowError):
                read_current_ore_inventory(wrong_after_tap)
        self.assertEqual([point for point, _ in wrong_after_tap.taps], [(759, 207)])

    def test_transition_account_stale_and_reused_frames_stop_without_extra_taps(self):
        for mode in ("account", "stale", "reused"):
            with self.subTest(mode=mode):
                names = (["home_visible", "home_immediate_after_tap"] if mode != "reused"
                         else ["home_visible", "home_visible"])
                session = FakeSession(names)
                original = session.observe
                def changed(label, *, purpose="full"):
                    value = original(label, purpose=purpose)
                    if label == "ore-navigation-selected-1":
                        if mode == "account":
                            value.observations["account_in_use"] = {"warning": True}
                        elif mode == "stale":
                            value.observations["observed_at_monotonic"] -= 10
                    return value
                session.observe = changed
                with patch("autococ.ore_navigation.foreground_package",
                           return_value=session.config.game.package_name):
                    with self.assertRaises(FlowError):
                        read_current_ore_inventory(session)
                self.assertEqual([point for point, _ in session.taps], [(759, 207)])

    def test_first_ore_page_never_opens_again_and_unknown_three_frames_stop(self):
        session = FakeSession(["selected_strong", "selected_mid", "home_immediate_after_tap",
                               "ore_first", "ore_second", "home_new",
                               "home_visible", "home_old"])
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=session.config.game.package_name), patch(
                   "autococ.ore_inventory.foreground_package",
                   return_value=session.config.game.package_name):
            result = read_current_ore_inventory(session)
        self.assertTrue(result["returned_home"])
        self.assertEqual([point for point, _ in session.taps], [(708, 562), (1223, 81)])

        unknown = FakeSession(["selected_strong", "selected_mid", "home_immediate_after_tap",
                               "home_failed"])
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=unknown.config.game.package_name):
            with self.assertRaises(FlowError):
                read_current_ore_inventory(unknown)
        self.assertEqual([point for point, _ in unknown.taps], [(708, 562)])

    def test_close_home_needs_two_new_ordered_clean_frames(self):
        for mode in ("account_overlay", "reused_path", "reused_id", "expired"):
            with self.subTest(mode=mode):
                names = (["selected_strong", "ore_first", "ore_second", "home_new",
                          "home_visible", "home_old"] if mode != "reused_path" else
                         ["selected_strong", "ore_first", "ore_second", "home_new",
                          "home_new", "home_old"])
                session = FakeSession(names)
                original = session.observe
                def changed(label, *, purpose="full"):
                    value = original(label, purpose=purpose)
                    if label == "ore-navigation-return-home-1":
                        if mode == "account_overlay":
                            value.observations["account_in_use"] = {"warning": True}
                        elif mode == "expired":
                            value.observations["observed_at_monotonic"] -= 10
                    if label == "ore-navigation-return-home-2" and mode == "reused_id":
                        value.observations["frame_id"] = "home_visible-5"
                    return value
                session.observe = changed
                with patch("autococ.ore_navigation.foreground_package",
                           return_value=session.config.game.package_name), patch(
                           "autococ.ore_inventory.foreground_package",
                           return_value=session.config.game.package_name):
                    with self.assertRaises(FlowError):
                        read_current_ore_inventory(session)
                self.assertEqual(len(session.taps), 2)

    def test_obstructed_home_reobserves_without_click_and_fails_closed(self):
        with TemporaryDirectory() as tmp:
            session = FakeSession(["home_occluded"] * 8)
            give_each_capture_its_own_path(session, Path(tmp))
            with patch("autococ.ore_navigation.time.sleep"):
                with self.assertRaisesRegex(FlowError, "eight current HOME captures"):
                    read_current_ore_inventory(session)
            self.assertEqual(len(session.observed), 8)
            self.assertEqual(session.taps, [])

    def test_three_obstructed_home_captures_then_clear_front_selects_once(self):
        names = (["home_occluded", "home_failed", "home_occluded", "home_visible",
                  "home_immediate_after_tap", "selected_delayed_after_tap", "ore_first",
                  "ore_second", "home_new", "home_visible", "home_old"])
        with TemporaryDirectory() as tmp:
            session = FakeSession(names)
            give_each_capture_its_own_path(session, Path(tmp))
            with patch("autococ.ore_navigation.time.sleep"), patch(
                    "autococ.ore_navigation.foreground_package",
                    return_value=session.config.game.package_name), patch(
                    "autococ.ore_inventory.foreground_package",
                    return_value=session.config.game.package_name):
                result = read_current_ore_inventory(session)
            self.assertEqual(result["status"], "stable_inventory_read")
            self.assertEqual(session.observed[:4], names[:4])
            self.assertEqual([point for point, reason in session.taps
                              if reason == "Select current visible blacksmith front"], [(759, 207)])

    def test_transient_unknown_then_fresh_home_reads_without_unknown_tap(self):
        names = ["home_visible", "home_visible", "home_immediate_after_tap",
                 "selected_delayed_after_tap", "ore_first", "ore_second",
                 "home_new", "home_visible", "home_old"]
        with TemporaryDirectory() as tmp:
            session = FakeSession(names)
            give_each_capture_its_own_path(session, Path(tmp))
            mark_captures_unknown(session, {1})
            with patch("autococ.ore_navigation.time.sleep"), patch(
                    "autococ.ore_navigation.foreground_package",
                    return_value=session.config.game.package_name), patch(
                    "autococ.ore_inventory.foreground_package",
                    return_value=session.config.game.package_name):
                result = read_current_ore_inventory(session)
            self.assertEqual(result["status"], "stable_inventory_read")
            self.assertEqual(session.observed[:2], ["home_visible", "home_visible"])
            self.assertEqual([point for point, reason in session.taps
                              if reason == "Select current visible blacksmith front"], [(759, 207)])

    def test_all_unknown_stops_after_eight_fresh_captures_without_input(self):
        with TemporaryDirectory() as tmp:
            session = FakeSession(["home_visible"] * 8)
            give_each_capture_its_own_path(session, Path(tmp))
            mark_captures_unknown(session, set(range(1, 9)))
            with patch("autococ.ore_navigation.time.sleep"), patch(
                    "autococ.ore_navigation.foreground_package",
                    return_value=session.config.game.package_name):
                with self.assertRaisesRegex(FlowError, "eight current HOME captures"):
                    read_current_ore_inventory(session)
            self.assertEqual(len(session.observed), 8)
            self.assertEqual(session.taps, [])

    def test_unknown_account_stale_foreground_and_other_scene_stop_before_input(self):
        for mode in ("account", "conflict_ocr", "stale", "foreground", "disconnected",
                     "account_scene", "popup"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                session = FakeSession(["home_visible"])
                give_each_capture_its_own_path(session, Path(tmp))
                if mode in {"account", "conflict_ocr", "stale", "foreground"}:
                    mark_captures_unknown(session, {1}, account=mode == "account",
                                          conflict_ocr=mode == "conflict_ocr",
                                          stale=mode == "stale")
                else:
                    original = session.observe
                    def other_scene(label, *, purpose="full"):
                        snapshot = original(label, purpose=purpose)
                        changed = SceneSnapshot("account_in_use" if mode == "account_scene" else mode,
                                                snapshot.confidence, snapshot.screenshot_path,
                                                snapshot.observations)
                        session.last_snapshot = changed
                        return changed
                    session.observe = other_scene
                foreground = ("wrong.package" if mode == "foreground"
                              else session.config.game.package_name)
                with patch("autococ.ore_navigation.foreground_package",
                           return_value=foreground):
                    with self.assertRaises(FlowError):
                        read_current_ore_inventory(session)
                self.assertEqual(len(session.observed), 1)
                self.assertEqual(session.taps, [])

    def test_front_observation_thirty_second_cap_and_nonhome_stop_without_input(self):
        with TemporaryDirectory() as tmp:
            session = FakeSession(["home_occluded", "home_visible"])
            give_each_capture_its_own_path(session, Path(tmp))
            clock = [time.monotonic()]
            fake_time = SimpleNamespace(monotonic=lambda: clock[0],
                                        sleep=lambda _seconds: clock.__setitem__(0, clock[0] + 31))
            with patch("autococ.ore_navigation.time", fake_time):
                with self.assertRaisesRegex(FlowError, "front observation deadline"):
                    read_current_ore_inventory(session)
            self.assertEqual(session.observed, ["home_occluded"])
            self.assertEqual(session.taps, [])

        session = FakeSession(["home_occluded", "home_visible"])
        original = session.observe
        def nonhome(label, *, purpose="full"):
            snapshot = original(label, purpose=purpose)
            if len(session.observed) == 1:
                return snapshot
            unknown = SceneSnapshot("disconnected", snapshot.confidence,
                                    snapshot.screenshot_path, snapshot.observations)
            session.last_snapshot = unknown
            return unknown
        session.observe = nonhome
        with TemporaryDirectory() as tmp:
            give_each_capture_its_own_path(session, Path(tmp))
            with patch("autococ.ore_navigation.time.sleep"):
                with self.assertRaisesRegex(FlowError, "not a current HOME"):
                    read_current_ore_inventory(session)
        self.assertEqual(session.observed, ["home_occluded", "home_visible"])
        self.assertEqual(session.taps, [])

    def test_stop_during_obstructed_front_wait_has_no_input(self):
        session = FakeSession(["home_occluded", "home_visible"])
        checked = 0
        def stop():
            nonlocal checked
            checked += 1
            if checked >= 3:
                raise StopRequested("requested stop")
        session.check_deadline = stop
        with self.assertRaises(StopRequested):
            read_current_ore_inventory(session)
        self.assertEqual(session.observed, ["home_occluded"])
        self.assertEqual(session.taps, [])


if __name__ == "__main__":
    unittest.main()
