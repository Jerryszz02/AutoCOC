"""Furnace battle identity from separate observed frames, with cached OCR only."""

import hashlib
import json
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.unit_catalog import recognize_card_identity, template_manifest
from autococ.vision import ScreenshotRecognizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
METADATA = ROOT / "tests/fixtures/furnace_battle_identity_18_647_2.json"
BOX = (287, 595, 375, 711)


def _read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FurnaceBattleTests(TestCase):
    def setUp(self):
        self.metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    def test_catalogue_and_anonymous_fixtures_match_source_pixels(self):
        entry, = [entry for entry in template_manifest(client_version="18.647.2")["templates"]
                  if entry["unit_id"] == "furnace" and entry["surface"] == "battle"]
        self.assertEqual(entry, self.metadata["template"])
        self.assertEqual(entry["state"], "active")
        self.assertFalse(self.metadata["human_reviewed"])
        self.assertFalse(self.metadata["counts_toward_acceptance"])
        template = ROOT / "assets/catalogs" / entry["path"]
        self.assertEqual(_sha(template), entry["template_sha256"])
        source = SOURCE_ROOT / entry["source_frame"]
        if source.is_file():
            self.assertEqual(_sha(source), entry["source_sha256"])
            resized = cv2.resize(_read(source), (1280, 720), interpolation=cv2.INTER_AREA)
            left, top, right, bottom = entry["source_bbox_1280"]
            self.assertTrue(np.array_equal(_read(template), resized[top:bottom, left:right]))
        detail = SOURCE_ROOT / entry["identity_evidence_frame"]
        if detail.is_file():
            self.assertEqual(_sha(detail), entry["identity_evidence_sha256"])
        self.assertEqual(entry["identity_evidence_title"], "4级烈焰熔炉")
        self.assertNotEqual(self.metadata["frames"][0]["source_frame"],
                            self.metadata["frames"][1]["source_frame"])
        for frame in self.metadata["frames"]:
            with self.subTest(frame=frame["path"]):
                actual = _read(ROOT / frame["path"])
                self.assertEqual(_sha(ROOT / frame["path"]), frame["sha256"])
                mask = np.zeros(actual.shape[:2], dtype=bool)
                for left, top, right, bottom in frame["retained_baseline_rectangles_xyxy"]:
                    mask[top:bottom, left:right] = True
                self.assertTrue(np.all(actual[~mask] == 45))
                source = SOURCE_ROOT / frame["source_frame"]
                if source.is_file():
                    self.assertEqual(_sha(source), frame["source_sha256"])
                    original = cv2.resize(_read(source), (1280, 720), interpolation=cv2.INTER_AREA)
                    self.assertTrue(np.array_equal(actual[mask], original[mask]))
                events = SOURCE_ROOT / frame["source_events"]
                if events.is_file():
                    self.assertEqual(_sha(events), frame["source_events_sha256"])
                    observation, = [event for line in events.read_text(encoding="utf-8").splitlines()
                                    if (event := json.loads(line)).get("kind") == "observation"
                                    and Path(event["frame"]) == source]
                    for text in frame["cached_ocr"]:
                        self.assertIn(text, observation["observations"]["ocr"])

    def test_separate_active_frame_identifies_by_appearance_not_position(self):
        image = _read(ROOT / self.metadata["frames"][1]["path"])
        identity = recognize_card_identity(image, BOX, "troop", surface="battle", client_version="18.647.2")
        self.assertEqual(identity["unit_id"], "furnace")
        self.assertGreaterEqual(identity["confidence"], .94)
        # Move only the actual card pixels; its new slot cannot supply identity.
        moved = np.full_like(image, 45)
        moved[595:711, 580:668] = image[595:711, 287:375]
        self.assertEqual(recognize_card_identity(
            moved, (580, 595, 668, 711), "troop", surface="battle",
            client_version="18.647.2")["unit_id"], "furnace")
        self.assertIsNone(recognize_card_identity(
            moved, BOX, "troop", surface="battle", client_version="18.647.2")["unit_id"])
        for box, expected in (((93, 595, 182, 711), "electro_dragon"),
                              ((190, 595, 278, 711), "dragon"),
                              ((383, 595, 472, 711), "balloon")):
            with self.subTest(neighbor=expected):
                self.assertEqual(recognize_card_identity(image, box, "troop", surface="battle",
                                                        client_version="18.647.2")["unit_id"], expected)

    def test_wrong_version_kind_surface_and_hidden_portrait_are_rejected(self):
        image = _read(ROOT / self.metadata["frames"][1]["path"])
        for version in (None, "18.647.3", "18.600.7"):
            with self.subTest(version=version):
                self.assertIsNone(recognize_card_identity(image, BOX, "troop", surface="battle",
                                                         client_version=version)["unit_id"])
        for kind, surface in (("hero", "battle"), ("spell", "battle"),
                              ("troop", "army"), ("troop", "current_picker")):
            with self.subTest(kind=kind, surface=surface):
                self.assertIsNone(recognize_card_identity(image, BOX, kind, surface=surface,
                                                         client_version="18.647.2")["unit_id"])
        hidden = image.copy()
        hidden[617:705, 292:370] = 45
        self.assertIsNone(recognize_card_identity(hidden, BOX, "troop", surface="battle",
                                                 client_version="18.647.2")["unit_id"])

    def test_real_toolbar_uses_count_ocr_independently_of_furnace_identity(self):
        recognizer = ScreenshotRecognizer(provider=None)
        recognizer.client_version = "18.647.2"
        for frame, countdown in zip(self.metadata["frames"], (17, 166)):
            with self.subTest(scene=frame["scene"]):
                texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                         for item in frame["cached_ocr"]]
                observed = recognizer._battle_observation(ROOT / frame["path"], texts, frame["scene"])
                self.assertEqual(observed["countdown_seconds"], countdown)
                troops = {slot["unit_id"]: slot["count"] for slot in observed["slots"]
                          if slot["kind"] == "troop"}
                self.assertEqual(troops, {"electro_dragon": 10, "dragon": 1,
                                         "furnace": 1, "balloon": 1})
                furnace, = [slot for slot in observed["slots"] if slot["unit_id"] == "furnace"]
                self.assertIsNone(furnace.get("selected"))
                self.assertIsNone(furnace.get("deployed"))
                # Identity must never turn a missing quantity header into x1.
                no_furnace_count = [item for item in texts
                                    if not (item.text == "x1" and 330 < item.bbox[0] < 380)]
                without_count = recognizer._battle_observation(
                    ROOT / frame["path"], no_furnace_count, frame["scene"])
                furnace, = [slot for slot in without_count["slots"] if slot["unit_id"] == "furnace"]
                self.assertIsNone(furnace["count"])
