"""Saved hero-equipment evidence: read-only inventory and guarded close."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

import cv2

from autococ.errors import FlowError
from autococ.ore_inventory import (close_ore_inventory, observe_ore_inventory,
                                   read_ore_inventory_snapshot, verify_stable_ore_inventory)
from autococ.scene import SceneSnapshot


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "hero_equipment_18_647_2"
VERSION = "18.647.2"


def ocr(name: str = "first") -> list[dict]:
    return json.loads((FIXTURES / f"{name}.ocr.json").read_text(encoding="utf-8"))


def snapshot(name: str, *, frame_id: str, observed: float) -> SceneSnapshot:
    path = FIXTURES / f"{name}.png"
    return SceneSnapshot("popup", .95, path,
                         {"ocr": ocr(name), "frame_id": frame_id,
                          "observed_at_monotonic": observed,
                          "baseline_resolution": [1280, 720], "account_in_use": None})


class FakeSession:
    def __init__(self, current: SceneSnapshot, subsequent: list[SceneSnapshot]):
        self.last_snapshot = current
        self.subsequent = list(subsequent)
        self.client_version = VERSION
        self.deadline = self.task_deadline = time.monotonic() + 30
        self.config = Mock()
        self.config.game.baseline_resolution = (1280, 720)
        self.config.game.package_name = "com.supercell.clashofclans"
        self.config.runtime.step_timeout_sec = 5
        self.capture = Mock(adb=object(), serial="fake-serial")
        self.taps: list[tuple] = []

    def check_deadline(self):
        if time.monotonic() >= min(self.deadline, self.task_deadline):
            raise FlowError("deadline")

    def tap(self, current, point, *, reason):
        if current is not self.last_snapshot:
            raise FlowError("obsolete")
        self.taps.append((tuple(point), reason))
        self.last_snapshot = None

    def observe(self, label, *, purpose="full"):
        current = self.subsequent.pop(0)
        self.last_snapshot = current
        return current


class OreInventoryTests(unittest.TestCase):
    def test_anonymous_fixture_and_template_provenance(self):
        provenance = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
        self.assertFalse(provenance["formal_acceptance"])
        self.assertFalse(provenance["reward_before_after"])
        for entry in provenance["fixtures"].values():
            for path_key, hash_key in (("derived_path", "derived_sha256"),
                                       ("ocr_path", "ocr_sha256")):
                path = ROOT / entry[path_key]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry[hash_key])
            image = cv2.imread(str(ROOT / entry["derived_path"]))
            self.assertEqual(image.shape[:2], (720, 1280))
            retained = provenance["transformation"]["retained_baseline_rois_xyxy"]
            for name, (x1, y1, x2, y2) in retained.items():
                pixels = image[y1:y2, x1:x2]
                self.assertEqual(hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 entry["retained_roi_pixel_sha256"][name])
            mask = image.copy()
            for x1, y1, x2, y2 in retained.values():
                mask[y1:y2, x1:x2] = (24, 24, 24)
            self.assertTrue((mask == (24, 24, 24)).all())
            self.assertEqual(len(json.loads((ROOT / entry["ocr_path"]).read_text(encoding="utf-8"))), 6)
        for entry in provenance["assets"].values():
            self.assertEqual(hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])

    def test_two_saved_pages_read_same_three_ores_without_claim_status(self):
        for name in ("first", "second"):
            with self.subTest(name=name):
                path = FIXTURES / f"{name}.png"
                result = observe_ore_inventory(path, ocr(name), client_version=VERSION)
                self.assertIsNotNone(result)
                self.assertEqual(result["balances"], {"shiny": 15662, "glowy": 1225, "starry": 77})
                self.assertEqual(result["capacities"], {"shiny": 50000, "glowy": 5000, "starry": 1000})
                self.assertEqual(result["frame_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(result["button"]["name"], "close_ore_inventory")
                self.assertFalse(result["account_verified"])
                self.assertFalse(result["reward_inventory_verified"])
                self.assertFalse(result["formal_success"])

    def test_version_page_anchors_and_icons_fail_closed(self):
        path = FIXTURES / "first.png"
        base = ocr()
        self.assertIsNone(observe_ore_inventory(path, base, client_version="18.647.3"))
        self.assertIsNone(observe_ore_inventory(path, base, client_version=VERSION,
                                                baseline_resolution=(1920, 1080)))
        for label in ("英雄装备", "拖动图标为英雄配置装备", "X"):
            with self.subTest(label=label):
                self.assertIsNone(observe_ore_inventory(path, [x for x in base if x["text"] != label],
                                                        client_version=VERSION))
                duplicate = base + [copy.deepcopy(next(x for x in base if x["text"] == label))]
                self.assertIsNone(observe_ore_inventory(path, duplicate, client_version=VERSION))
        with TemporaryDirectory() as directory:
            image = cv2.imread(str(path))
            for x1, y1, x2, y2 in ((399, 620, 430, 654), (565, 620, 597, 654),
                                   (734, 620, 767, 654), (1200, 58, 1247, 105)):
                with self.subTest(roi=(x1, y1, x2, y2)):
                    changed = image.copy()
                    changed[y1:y2, x1:x2] = 0
                    filename = Path(directory) / "changed.png"
                    self.assertTrue(cv2.imwrite(str(filename), changed))
                    self.assertIsNone(observe_ore_inventory(filename, base, client_version=VERSION))

    def test_dimmed_page_with_stale_original_ocr_is_not_inventory(self):
        with TemporaryDirectory() as directory:
            image = cv2.imread(str(FIXTURES / "first.png"))
            dimmed = (image.astype("float32") * .65).astype("uint8")
            filename = Path(directory) / "dimmed.png"
            self.assertTrue(cv2.imwrite(str(filename), dimmed))
            self.assertIsNone(observe_ore_inventory(filename, ocr(), client_version=VERSION))

    def test_count_missing_duplicate_weak_conflict_and_invalid_number(self):
        path = FIXTURES / "first.png"
        base = ocr()
        count = next(x for x in base if x["text"] == "1225/5000")
        cases = [
            [x for x in base if x is not count],
            base + [copy.deepcopy(count)],
            [dict(x, confidence=.989) if x is count else x for x in base],
            [dict(x, confidence=float("nan")) if x is count else x for x in base],
            [dict(x, confidence=True) if x is count else x for x in base],
            [dict(x, text="1225/5000 123") if x is count else x for x in base],
            [dict(x, text="5001/5000") if x is count else x for x in base],
            [dict(x, text="1225/50000") if x is count else x for x in base],
            [dict(x, text="0/0") if x is count else x for x in base],
            [dict(x, text="-1/5000") if x is count else x for x in base],
            [dict(x, text="1225/NaN") if x is count else x for x in base],
            [dict(x, bbox=[715, 626, 821, 648]) if x is count else x for x in base],
            base + [{"text": "9/5000", "confidence": .999, "bbox": [597, 626, 703, 648]}],
        ]
        for index, items in enumerate(cases):
            with self.subTest(case=index):
                self.assertIsNone(observe_ore_inventory(path, items, client_version=VERSION))

    def test_stable_read_requires_distinct_ordered_fresh_frames_and_same_values(self):
        now = time.monotonic()
        first = read_ore_inventory_snapshot(snapshot("first", frame_id="first-capture", observed=now - 2),
                                            client_version=VERSION)
        second = read_ore_inventory_snapshot(snapshot("second", frame_id="second-capture", observed=now - 1),
                                             client_version=VERSION)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        stable = verify_stable_ore_inventory(first, second)
        self.assertEqual(stable["status"], "stable_inventory_read")
        self.assertFalse(stable["reward_inventory_verified"])
        self.assertFalse(stable["account_verified"])
        for changed in (dict(second, frame_id=first["frame_id"]),
                        dict(second, frame=first["frame"]),
                        dict(second, observed_at_monotonic=first["observed_at_monotonic"]),
                        dict(second, observed_at_monotonic=now - 60),
                        dict(second, ores={**second["ores"], "glowy": {**second["ores"]["glowy"],
                                                                      "amount": 1226}}),
                        dict(second, ores={**second["ores"], "starry": {**second["ores"]["starry"],
                                                                       "capacity": 1001}})):
            self.assertIsNone(verify_stable_ore_inventory(first, changed))
        self.assertIsNone(verify_stable_ore_inventory(first, second, max_gap_sec=float("nan")))

    def test_stable_read_malformed_frame_and_reordered_ores(self):
        now = time.monotonic()
        first = read_ore_inventory_snapshot(snapshot("first", frame_id="one", observed=now - 2),
                                            client_version=VERSION)
        second = read_ore_inventory_snapshot(snapshot("second", frame_id="two", observed=now - 1),
                                             client_version=VERSION)
        for malformed in (None, 123, ""):
            self.assertIsNone(verify_stable_ore_inventory({**first, "frame": malformed}, second))
        missing = dict(first)
        missing.pop("frame")
        self.assertIsNone(verify_stable_ore_inventory(missing, second))
        reordered = {**second, "ores": {name: second["ores"][name]
                                         for name in ("starry", "shiny", "glowy")}}
        self.assertIsNotNone(verify_stable_ore_inventory(first, reordered))

    def test_close_only_current_recognized_page_and_fresh_home(self):
        current = snapshot("first", frame_id="live-frame", observed=time.monotonic())
        current.observations["ore_inventory"] = observe_ore_inventory(
            current.screenshot_path, current.observations["ocr"], client_version=VERSION)
        home = SceneSnapshot("village", .95, FIXTURES / "second.png",
                             {"village_type": "home", "resource_source": "village_inventory",
                              "buttons": [{"name": "attack"}, {"name": "shop"}]})
        session = FakeSession(current, [home])
        with patch("autococ.ore_inventory.foreground_package", return_value=session.config.game.package_name):
            receipt = close_ore_inventory(session, current)
        self.assertEqual(receipt["status"], "home")
        self.assertEqual(len(session.taps), 1)
        self.assertFalse(receipt["reward_inventory_verified"])

        for mutation in (lambda s: replace(s, scene="unknown"),
                         lambda s: s.observations.update(ore_inventory=None),
                         lambda s: s.observations.update(observed_at_monotonic=time.monotonic() - 31),
                         lambda s: s.observations.update(account_in_use={"warning": True}),
                         lambda s: s.observations["ore_inventory"].update(frame_sha256="0" * 64)):
            with self.subTest(mutation=mutation):
                altered = snapshot("first", frame_id="live-frame", observed=time.monotonic())
                altered.observations["ore_inventory"] = observe_ore_inventory(
                    altered.screenshot_path, altered.observations["ocr"], client_version=VERSION)
                updated = mutation(altered)
                if isinstance(updated, SceneSnapshot):
                    altered = updated
                blocked = FakeSession(altered, [])
                with patch("autococ.ore_inventory.foreground_package", return_value=blocked.config.game.package_name):
                    with self.assertRaises(FlowError):
                        close_ore_inventory(blocked, altered)
                self.assertEqual(blocked.taps, [])

    def test_close_rechecks_after_foreground_work(self):
        def current(path=FIXTURES / "first.png", *, age=0):
            now = time.monotonic()
            shot = SceneSnapshot("popup", .95, path,
                                 {"ocr": ocr(), "frame_id": "live-frame",
                                  "observed_at_monotonic": now - age,
                                  "baseline_resolution": [1280, 720], "account_in_use": None})
            shot.observations["ore_inventory"] = observe_ore_inventory(
                shot.screenshot_path, shot.observations["ocr"], client_version=VERSION)
            return shot

        for mode in ("local_timeout", "stale_age", "new_snapshot", "changed_frame"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                source = cv2.imread(str(FIXTURES / "first.png"))
                path = Path(directory) / "current.png"
                self.assertTrue(cv2.imwrite(str(path), source))
                shot = current(path, age=7.85 if mode == "stale_age" else 0)
                session = FakeSession(shot, [])

                def foreground(*args, **kwargs):
                    if mode == "local_timeout":
                        self.assertLessEqual(kwargs["timeout_sec"], .1)
                        time.sleep(.11)
                    elif mode == "stale_age":
                        time.sleep(.25)
                    elif mode == "new_snapshot":
                        session.last_snapshot = snapshot("second", frame_id="new", observed=time.monotonic())
                    elif mode == "changed_frame":
                        changed = source.copy()
                        changed[0, 0] = (0, 0, 0)
                        self.assertTrue(cv2.imwrite(str(path), changed))
                    return session.config.game.package_name

                with patch("autococ.ore_inventory.foreground_package", side_effect=foreground):
                    with self.assertRaises(FlowError):
                        close_ore_inventory(session, shot,
                                            timeout_sec=.1 if mode == "local_timeout" else 2)
                self.assertEqual(session.taps, [])


if __name__ == "__main__":
    unittest.main()
