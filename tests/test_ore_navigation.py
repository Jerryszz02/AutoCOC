"""Saved 18.647.2 blacksmith navigation evidence and bounded fake sessions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from autococ.errors import FlowError
from autococ.ocr import OCRText
from autococ.ore_inventory import observe_ore_inventory
from autococ.ore_navigation import (locate_blacksmith, read_current_ore_inventory,
                                     recognize_selected_blacksmith)
from autococ.scene import SceneSnapshot


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "blacksmith_navigation_18_647_2"
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


class OreNavigationTests(unittest.TestCase):
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
        session = FakeSession(["home_visible", "selected_strong", "ore_first",
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
        session = FakeSession(["selected_strong", "ore_first", "ore_second", "home_new",
                               "home_visible", "home_old"])
        with patch("autococ.ore_navigation.foreground_package",
                   return_value=session.config.game.package_name), patch(
                   "autococ.ore_inventory.foreground_package",
                   return_value=session.config.game.package_name):
            result = read_current_ore_inventory(session)
        self.assertTrue(result["returned_home"])
        self.assertEqual(len(session.taps), 2)

        wrong = FakeSession(["home_visible", "home_failed"])
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
        session = FakeSession(["home_occluded", "home_failed", "home_occluded"])
        with self.assertRaises(FlowError):
            read_current_ore_inventory(session)
        self.assertEqual(len(session.observed), 3)
        self.assertEqual(session.taps, [])


if __name__ == "__main__":
    unittest.main()
