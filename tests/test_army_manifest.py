from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.army_manifest import recognize_army_manifest
from autococ.ocr import OCRText, RapidOCRProvider


ANCHORS = [OCRText("我的军队", .99, (222, 63, 294, 90)),
           OCRText("已保存的配置", .99, (594, 64, 685, 85)),
           OCRText("精选库", .99, (972, 64, 1024, 86))]


class ArmyManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "army.png"
        self.provider = Mock()
        self.provider.recognize_line.return_value = []

    def make(self, troops=(10, 1, 2, 1), spells=(6, 5)):
        image = np.full((720, 1280, 3), (40, 70, 110), dtype=np.uint8)
        texts = list(ANCHORS)
        for values, x0, top in ((troops, 557, 194), (spells, 558, 373)):
            for index, count in enumerate(values):
                left = x0 + index * 103
                image[top:top + 97, left:left + 98] = (180, 90, 40)
                texts.append(OCRText(f"x{count}", .99, (left + 3, top + 2, left + 47, top + 25)))
        return image, texts

    def inspect(self, image, texts):
        cv2.imencode(".png", image)[1].tofile(self.path)
        return recognize_army_manifest(self.path, self.provider, texts)

    def test_card_counts_are_variable_and_not_the_sample_army(self) -> None:
        for troops, spells in (((3,), (2,)), ((7, 2, 4), (1, 3, 5)), ((9, 6, 4, 2, 1), (8, 2))):
            with self.subTest(troops=troops, spells=spells):
                result = self.inspect(*self.make(troops, spells))
                self.assertTrue(result["complete"], result["unknowns"])
                self.assertEqual([card["count"] for card in result["troops"]], list(troops))
                self.assertEqual([card["count"] for card in result["spells"]], list(spells))

    def test_missing_quantity_keeps_visible_card_unknown(self) -> None:
        image, texts = self.make()
        result = self.inspect(image, [item for item in texts if item.text != "x10"])
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["troops"]), 4)
        self.assertIsNone(result["troops"][0]["count"])

    def test_local_line_can_recover_a_missing_quantity(self) -> None:
        image, texts = self.make()
        self.provider.recognize_line.return_value = [OCRText("x10", .98, (560, 196, 600, 219))]
        result = self.inspect(image, [item for item in texts if item.text != "x10"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["troops"][0]["count"], 10)
        self.assertEqual(result["troops"][0]["evidence"]["count"]["source"], "line_roi")

    def test_neighbor_line_cannot_fill_a_missing_quantity(self) -> None:
        image, texts = self.make()
        self.provider.recognize_line.return_value = [OCRText("x10", .99, (662, 196, 704, 219))]
        result = self.inspect(image, [item for item in texts if item.text != "x10"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["troops"][0]["count"])

    def test_duplicate_conflicting_zero_and_low_counts_are_not_complete(self) -> None:
        image, texts = self.make()
        quantity = next(item for item in texts if item.text == "x10")
        variants = [texts + [replace(quantity, text="x3")],
                    [replace(item, text="x0") if item is quantity else item for item in texts]]
        for confidence in (.89, float("nan"), float("inf"), 1.01):
            variants.append([replace(item, confidence=confidence) if item is quantity else item for item in texts])
        for modified in variants:
            with self.subTest(texts=modified):
                result = self.inspect(image, modified)
                self.assertFalse(result["complete"])
                self.assertIsNone(result["troops"][0]["count"])

    def test_missing_middle_card_and_its_ocr_leave_a_detectable_gap(self) -> None:
        image, texts = self.make()
        image[194:291, 660:758] = (40, 70, 110)
        texts = [item for item in texts if not item.bbox or not 660 <= item.bbox[0] < 758]
        result = self.inspect(image, texts)
        self.assertFalse(result["complete"])
        self.assertIn("card_row_not_contiguous", [item["reason"] for item in result["unknowns"]])

    def test_clipped_extra_card_and_incomplete_borders_block_completion(self) -> None:
        image, texts = self.make()
        clipped = image.copy()
        clipped[194:291, 1240:1280] = (180, 90, 40)
        broken = image.copy()
        broken[194:250, 762:780] = (40, 70, 110)
        for modified in (clipped, broken):
            with self.subTest():
                self.assertFalse(self.inspect(modified, texts)["complete"])

    def test_missing_layout_anchor_is_unsupported(self) -> None:
        image, texts = self.make()
        result = self.inspect(image, texts[1:])
        self.assertFalse(result["supported_layout"])
        self.assertFalse(result["complete"])

    def test_clan_and_siege_quantities_do_not_change_numeric_army(self) -> None:
        image, texts = self.make()
        texts += [OCRText("x99", .99, (1040, 375, 1080, 400)), OCRText("x55", .99, (560, 565, 600, 588))]
        result = self.inspect(image, texts)
        self.assertTrue(result["complete"])
        self.assertEqual([card["count"] for card in result["troops"]], [10, 1, 2, 1])
        self.assertEqual([card["count"] for card in result["spells"]], [6, 5])

    def test_actual_army_recovers_balloon_quantity_with_local_ocr(self) -> None:
        root = Path(__file__).resolve().parents[1]
        folder = root / "reports/20260923-023613-726879-9389d9e7"
        path = folder / "frames/00006-battle-army-confirmation.png"
        if not path.is_file():
            self.skipTest("Optional live army fixture is absent")
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observation = next(event for event in events if event.get("kind") == "observation" and event["frame"].endswith(path.name))
        texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in observation["observations"]["ocr"]]
        result = recognize_army_manifest(path, RapidOCRProvider(), texts)
        self.assertTrue(result["complete"], result["unknowns"])
        self.assertEqual([card["count"] for card in result["troops"]], [10, 1, 2, 1])
        self.assertEqual([card["count"] for card in result["spells"]], [6, 5])
        self.assertEqual(result["troops"][2]["evidence"]["count"]["source"], "line_roi")
