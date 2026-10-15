"""Current battle hero toolbar, replayed from anonymous real preview crops."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "tests/fixtures/current_hero_toolbar_18_647_2_20261015.json"


def _replay(entry: dict, *, version: str = "18.647.2", erase: str | None = None):
    fixture = ROOT / "tests/fixtures" / entry["fixture"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == entry["fixture_sha256"]
    source = Path(r"E:\projects\AutoCOC") / entry["source_frame"]
    toolbar = cv2.imdecode(np.fromfile(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert toolbar.shape == (135, 1200, 3)
    if source.is_file():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["source_sha256"]
        original = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        original = cv2.resize(original, (1280, 720), interpolation=cv2.INTER_AREA)
        assert np.array_equal(toolbar, original[585:720, 40:1240])
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[585:720, 40:1240] = toolbar
    if erase == "left_side":
        image[625:695, 741:750] = 0
    elif erase == "bottom":
        image[701:716, 750:829] = 0
    elif erase == "pet":
        image[600:640, 746:784] = 0
    texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
             for item in entry["ocr"]]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "anonymous-toolbar.png"
        cv2.imencode(".png", image)[1].tofile(path)
        vision = ScreenshotRecognizer(provider=Mock())
        vision.client_version = version
        return vision._battle_observation(path, texts, "enemy_village")["slots"]


class CurrentHeroLayoutTests(unittest.TestCase):
    def test_pet_anchor_is_exactly_derived_from_the_recorded_source(self):
        anchor = json.loads(METADATA.read_text(encoding="utf-8"))["hero_pet_anchor"]
        template = ROOT / anchor["template"]
        self.assertEqual(hashlib.sha256(template.read_bytes()).hexdigest(), anchor["sha256"])
        source = Path(r"E:\projects\AutoCOC") / anchor["source_frame"]
        if source.is_file():
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             anchor["source_sha256"])
            image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
            x0, y0, x1, y1 = anchor["baseline_crop_xyxy"]
            crop = cv2.imdecode(np.fromfile(template, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(crop, image[y0:y1, x0:x1]))

    def test_five_troops_four_heroes_and_shifted_spells_remain_independent(self):
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        for entry in metadata["frames"]:
            with self.subTest(frame=entry["fixture"]):
                slots = _replay(entry)
                heroes = {slot["unit_id"]: slot for slot in slots if slot["kind"] == "hero"}
                self.assertEqual(set(heroes), {
                    "grand_warden", "dragon_duke", "minion_prince", "archer_queen"})
                duke = heroes["dragon_duke"]
                self.assertGreaterEqual(duke["confidence"], .9)
                self.assertTrue(duke["evidence"]["identity"]["version_verified"])
                self.assertTrue(duke["evidence"]["hero_pet_icon"])
                self.assertEqual({slot["unit_id"]: slot["count"] for slot in slots
                                  if slot["kind"] == "spell"},
                                 {"lightning_spell": 6, "earthquake_spell": 5})
                self.assertEqual([(slot["unit_id"], slot["count"]) for slot in slots
                                  if slot["kind"] == "troop" and slot["unit_id"] != "archer"],
                                 [("electro_dragon", 10), ("dragon", 1),
                                  ("baby_dragon", 1), ("balloon", 2)])
                self.assertEqual(1, sum(slot["kind"] == "troop" and slot["count"] == 3
                                        for slot in slots))
                boxes = [slot["bbox"] for slot in slots]
                self.assertEqual(len(boxes), len({tuple(box) for box in boxes}))
                for left, right in zip(sorted(boxes), sorted(boxes)[1:]):
                    self.assertLessEqual(left[2], right[0])

    def test_missing_independent_border_or_pet_never_infers_duke_from_army_order(self):
        entry = json.loads(METADATA.read_text(encoding="utf-8"))["frames"][1]
        for erase in ("left_side", "bottom", "pet"):
            with self.subTest(erase=erase):
                slots = _replay(entry, erase=erase)
                self.assertFalse(any(slot["unit_id"] == "dragon_duke" for slot in slots))

    def test_wrong_client_version_does_not_borrow_current_hero_identity(self):
        entry = json.loads(METADATA.read_text(encoding="utf-8"))["frames"][1]
        slots = _replay(entry, version="18.647.3")
        self.assertFalse(any(slot["unit_id"] == "dragon_duke" for slot in slots))


if __name__ == "__main__":
    unittest.main()
