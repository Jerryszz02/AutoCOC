from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.settlement import BONUS_TEMPLATE_CROPS, locate_number_line, recognize_earned_stars
from autococ.vision import ScreenshotRecognizer


class SparseSettlementNumberTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    fixture_directory = root / "reports/cleanup-partial-battle-20260924-021726"

    def setUp(self):
        self.path = self.fixture_directory / "frames/00003-natural-settlement.png"
        if not self.path.is_file():
            self.skipTest("Optional isolated-zero settlement fixture is absent")
        events = [json.loads(line) for line in (self.fixture_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed = next(item for item in events if item["kind"] == "observation" and Path(item["frame"]).name == self.path.name)
        self.texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in observed["observations"]["ocr"]]
        self.image = cv2.resize(cv2.imread(str(self.path)), (1280, 720), interpolation=cv2.INTER_AREA)

    def result(self, *, texts=None, image=None, tight_text="0", tight_confidence=.9956, wrong_row=False):
        factor = 2 if image is None else 1
        provider = Mock()
        provider.recognize.return_value = [replace(item, bbox=tuple(v * factor for v in item.bbox))
                                           for item in (self.texts if texts is None else texts)]

        def read_line(path, roi):
            if roi[2] - roi[0] > 40 * factor:
                return [OCRText("O", .60306, roi)]
            box = tuple(v * factor for v in (570, 388, 670, 424)) if wrong_row else roi
            return [OCRText(tight_text, tight_confidence, box)]

        provider.recognize_line.side_effect = read_line
        recognizer = ScreenshotRecognizer(provider=provider)
        if image is None:
            frame = recognizer.recognize(self.path)
        else:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "changed.png"
                cv2.imwrite(str(path), image)
                frame = recognizer.recognize(path)
        return frame.observations["settlement"], provider

    def test_isolated_zero_is_read_from_its_glyphs_not_inferred_from_inventory(self):
        result, provider = self.result()
        self.assertEqual(result["loot"], {"gold": 0, "elixir": 67192, "dark_elixir": 0})
        self.assertEqual(result["bonus"], {"gold": 0, "elixir": 0, "dark_elixir": 0})
        self.assertEqual(result["evidence"]["loot"]["gold"]["source"], "glyph_line_roi")
        self.assertEqual(result["evidence"]["loot"]["gold"]["text"], "0")
        self.assertEqual(provider.recognize_line.call_count, 2)
        wide, tight = [call.args[1] for call in provider.recognize_line.call_args_list]
        self.assertGreater(wide[2] - wide[0], 7 * (tight[2] - tight[0]))
        self.assertEqual(tight, (1284, 682, 1332, 730))

    def test_glyph_localization_alone_never_assigns_zero(self):
        for text, confidence in (("O", .999), ("D", .999), ("0", .89), ("0", float("nan")), ("0", float("inf"))):
            with self.subTest(text=text, confidence=confidence):
                result, _ = self.result(tight_text=text, tight_confidence=confidence)
                self.assertIsNone(result["loot"]["gold"])
                self.assertIsNone(result["evidence"]["layout"])
                self.assertTrue(all(value is None for value in result["bonus"].values()))

    def test_missing_character_does_not_trigger_tight_read_or_zero(self):
        image = self.image.copy()
        image[328:374, 479:672] = 0
        self.assertIsNone(locate_number_line(image, (479, 328, 672, 379)))
        result, provider = self.result(image=image)
        self.assertIsNone(result["loot"]["gold"])
        self.assertEqual(provider.recognize_line.call_count, 1)

    def test_conflicting_loot_readings_remain_unknown_without_preferred_reread(self):
        values = [OCRText("0", .999, (640, 339, 668, 367)), OCRText("10", .99, (640, 339, 668, 367))]
        result, provider = self.result(texts=self.texts + values)
        self.assertIsNone(result["loot"]["gold"])
        provider.recognize_line.assert_not_called()

    def test_tight_ocr_result_from_another_resource_row_is_ignored(self):
        result, _ = self.result(wrong_row=True)
        self.assertIsNone(result["loot"]["gold"])
        self.assertEqual(result["loot"]["elixir"], 67192)

    def test_clipped_and_multiline_glyph_regions_are_not_tightened(self):
        for box in ((650, 328, 672, 379), (479, 350, 672, 379)):
            with self.subTest(roi=box):
                self.assertIsNone(locate_number_line(self.image, box))
        changed = self.image.copy()
        cv2.rectangle(changed, (600, 331), (612, 345), (255, 255, 255), -1)
        self.assertIsNone(locate_number_line(changed, (479, 328, 672, 379)))


class VictorySettlementTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    fixture_directory = root / "reports/20260924-004347-675110-2e2b80bd"

    def setUp(self):
        self.path = self.fixture_directory / "frames/00061-battle-settlement-reread.png"
        if not self.path.is_file():
            self.skipTest("Optional real victory fixture is absent")
        events = [json.loads(line) for line in (self.fixture_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed = next(item for item in events if item["kind"] == "observation" and Path(item["frame"]).name == self.path.name)
        self.texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in observed["observations"]["ocr"]]
        self.image = cv2.resize(cv2.imread(str(self.path)), (1280, 720), interpolation=cv2.INTER_AREA)

    def result(self, *, texts=None, image=None, line=()):
        texts = self.texts if texts is None else texts
        provider = Mock()
        factor = 2 if image is None else 1
        provider.recognize.return_value = [replace(item, bbox=tuple(v * factor for v in item.bbox)) for item in texts]
        provider.recognize_line.return_value = [replace(item, bbox=tuple(v * factor for v in item.bbox)) for item in line]
        recognizer = ScreenshotRecognizer(provider=provider)
        if image is None:
            snapshot = recognizer.recognize(self.path)
        else:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "changed.png"
                cv2.imwrite(str(path), image)
                snapshot = recognizer.recognize(path)
        return snapshot.observations["settlement"], provider

    def test_real_victory_keeps_loot_bonus_and_earned_star_separate_and_reconciles(self):
        result, _ = self.result()
        self.assertEqual(result["percentage"], 42)
        self.assertEqual(result["stars"], 1)
        self.assertEqual(result["loot"], {"gold": 90196, "elixir": 226443, "dark_elixir": 495})
        self.assertEqual(result["bonus"], {"gold": 237850, "elixir": 237850, "dark_elixir": 1758})
        self.assertEqual(result["evidence"]["layout"], "victory_with_bonus_three_rows_v1")
        self.assertEqual(result["evidence"]["bonus"]["source"], "explicit_bonus_panel")
        report = json.loads((self.root / "reports/run-20260924-004347-675110-2e2b80bd.json").read_text(encoding="utf-8"))
        metrics = report["task_results"][-1]["metrics"]
        for resource in result["loot"]:
            cost = metrics["search_cost_gold"] if resource == "gold" else 0
            self.assertEqual(metrics["resources_before"][resource] + result["loot"][resource] + result["bonus"][resource] - cost,
                             metrics["resources_after"][resource])
        self.assertEqual(metrics["resources_before"]["gems"], metrics["resources_after"]["gems"])
        self.assertEqual(report["task_results"][-1]["status"], "failed", "An offline reread never rewrites the old failure")

    def test_missing_conflicting_unsigned_or_unreliable_bonus_stays_unknown(self):
        gold = next(item for item in self.texts if item.text == "+237850" and item.bbox[1] == 351)
        without = [item for item in self.texts if item is not gold]
        variants = [[], [replace(gold, text="237850")], [replace(gold, text="-237850")],
                    [gold, replace(gold, text="+238850")]]
        variants += [[replace(gold, confidence=value)] for value in (.89, float("nan"), float("inf"))]
        for variant in variants:
            with self.subTest(variant=variant):
                result, _ = self.result(texts=without + variant)
                self.assertIsNone(result["bonus"]["gold"])
                self.assertEqual(result["bonus"]["elixir"], 237850)
                self.assertEqual(result["loot"]["gold"], 90196)
                self.assertIsNone(result["evidence"]["layout"])

    def test_bonus_line_fallback_remains_bound_to_the_matched_icon(self):
        gold = next(item for item in self.texts if item.text == "+237850" and item.bbox[1] == 351)
        result, provider = self.result(texts=[item for item in self.texts if item is not gold], line=[gold])
        self.assertEqual(result["bonus"]["gold"], 237850)
        self.assertEqual(result["evidence"]["bonus"]["resources"]["gold"]["source"], "line_roi")
        provider.recognize_line.assert_called_once()
        wrong_row = replace(gold, bbox=(915, 387, 1004, 413))
        result, _ = self.result(texts=[item for item in self.texts if item is not gold], line=[wrong_row])
        self.assertIsNone(result["bonus"]["gold"])

    def test_missing_bonus_marker_does_not_assign_zero_or_parse_unlabelled_panel(self):
        result, _ = self.result(texts=[item for item in self.texts if "奖励" not in item.text])
        self.assertTrue(all(value is None for value in result["bonus"].values()))
        self.assertEqual(result["stars"], 1)

    def test_missing_bonus_icon_does_not_relabel_another_resource(self):
        for name, (left, top, right, bottom) in BONUS_TEMPLATE_CROPS.items():
            with self.subTest(resource=name):
                changed = self.image.copy()
                changed[top:bottom, left:right] = 0
                result, _ = self.result(image=changed)
                self.assertIsNone(result["bonus"][name])
                self.assertTrue(all(value is not None for key, value in result["bonus"].items() if key != name))

    def test_two_separated_copies_of_a_bonus_icon_are_ambiguous(self):
        image = self.image.copy()
        image[421:453, 1007:1036] = image[349:381, 1007:1036].copy()
        result, _ = self.result(image=image)
        self.assertIsNone(result["bonus"]["gold"])
        self.assertEqual(result["bonus"]["elixir"], 237850)

    def test_star_shape_counter_accepts_rotated_complete_stars(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                image = np.zeros((720, 1280, 3), np.uint8)
                for index in range(count):
                    points = []
                    for vertex in range(10):
                        angle = -math.pi / 2 + vertex * math.pi / 5 + (index - 1) * .12
                        radius = 60 if vertex % 2 == 0 else 27
                        points.append([round(510 + 135 * index + math.cos(angle) * radius), round(143 + math.sin(angle) * radius)])
                    cv2.fillPoly(image, [np.array(points, np.int32)], (240, 240, 240))
                self.assertEqual(recognize_earned_stars(image)["count"], count)

    def test_incomplete_or_unexplained_star_region_cannot_be_ignored(self):
        for mode in ("missing_tip", "extra_rectangle", "extra_circle", "blank"):
            with self.subTest(mode=mode):
                image = self.image.copy()
                if mode == "missing_tip":
                    image[80:116, 495:546] = 0
                elif mode == "extra_rectangle":
                    image[90:205, 610:705] = 240
                elif mode == "extra_circle":
                    cv2.circle(image, (660, 148), 50, (240, 240, 240), -1)
                else:
                    image[65:220, 430:845] = 0
                self.assertIsNone(recognize_earned_stars(image)["count"])

    def test_conflicting_outcomes_are_not_zero_stars_or_zero_bonus(self):
        result, _ = self.result(texts=self.texts + [OCRText("失败", .99, (608, 194, 671, 237))])
        self.assertIsNone(result["stars"])
        self.assertTrue(all(value is None for value in result["bonus"].values()))
