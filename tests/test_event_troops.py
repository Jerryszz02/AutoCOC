"""Replay the observed gray event card without retaining a full battlefield PNG."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


class EventTroopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        fixture = Path(__file__).parent / "fixtures"
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.image[580:] = cv2.imdecode(np.fromfile(fixture / "event-battle-bar.png", dtype=np.uint8), cv2.IMREAD_COLOR)
        self.texts = json.loads((fixture / "event-battle-bar.json").read_text(encoding="utf-8"))
        self.path = Path(self.temporary.name) / "battle.png"

    def observe(self):
        okay, data = cv2.imencode(".png", self.image)
        self.assertTrue(okay)
        self.path.write_bytes(data.tobytes())
        texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in self.texts]
        return ScreenshotRecognizer(provider=Mock())._battle_observation(self.path, texts, "enemy_village")["slots"]

    def test_extra_forty_event_troops_are_distinct_from_the_owned_army(self):
        slots = self.observe()
        event = [slot for slot in slots if slot["source"] == "event"]
        self.assertEqual(len(event), 1)
        self.assertEqual((event[0]["kind"], event[0]["count"]), ("troop", 40))
        self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop" and slot["source"] == "army"], [10, 1, 1])

    def test_gray_card_without_known_portrait_is_not_guessed_from_quantity(self):
        self.image[627:679, 98:176] = 100
        self.assertFalse(any(slot["source"] == "event" for slot in self.observe()))

    def test_event_card_without_reliable_quantity_is_not_planned(self):
        self.texts = [item for item in self.texts if item["text"] != "x40"]
        self.assertFalse(any(slot["source"] == "event" for slot in self.observe()))
