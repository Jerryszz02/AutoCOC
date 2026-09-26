from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.images import read_frame, read_template
from autococ.ocr import OCRText
from autococ.scene import SceneSnapshot
from autococ.vision import ScreenshotRecognizer
from autococ.battle_vision import selected_card_bbox


class BattleVisionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image = np.zeros((720, 1280, 3), np.uint8)
        rng = np.random.default_rng(42)
        for x0, y0, x1, y1 in ((540, 10, 740, 35), (25, 520, 140, 555), (100, 590, 190, 710)):
            self.image[y0:y1, x0:x1] = rng.integers(0, 255, (y1-y0, x1-x0, 3), dtype=np.uint8)
        self.slot = {"kind": "troop", "count": 10, "point": [145, 650], "bbox": [100, 590, 190, 710]}
        self.provider = Mock()
        self.provider.recognize_line_image.return_value = [OCRText("x0", .99)]
        self.reader = ScreenshotRecognizer(provider=self.provider)
        self.previous = SceneSnapshot("battle", .95, self.write("before", self.image), {
            "ocr": [{"text": "离战斗结束还有", "confidence": .99, "bbox": [540, 10, 740, 35]}],
            "buttons": [{"name": "end_battle", "bbox": [25, 520, 140, 555]}],
            "battle": {"slots": [self.slot]}})
        self.reader.recognize = Mock(side_effect=lambda path: SceneSnapshot("unknown", 0, path))

    def write(self, name, image):
        path = self.root / f"{name}.png"
        cv2.imencode(".png", image)[1].tofile(path)
        return path

    def observe(self, image=None, purpose="troop_count"):
        return self.reader.recognize_battle(self.write("after", self.image if image is None else image),
            purpose=purpose, slot=self.slot if purpose == "troop_count" else None, previous=self.previous)

    def test_current_count_is_read_without_full_ocr_and_not_copied(self):
        snapshot = self.observe()
        self.assertFalse(snapshot.observations["fast_fallback"])
        read = snapshot.observations["deployment_count_reads"][0]
        self.assertEqual(read["count"], 0)
        self.assertEqual(read["frame"], str(snapshot.screenshot_path))
        self.assertEqual(self.slot["count"], 10)
        self.reader.recognize.assert_not_called()

    def test_gray_header_gets_threshold_retry(self):
        self.provider.recognize_line_image.side_effect = [[OCRText("xO", .6)], [OCRText("x0", .98)]]
        snapshot = self.observe()
        self.assertEqual(snapshot.observations["deployment_count_reads"][0]["count"], 0)
        self.assertEqual(self.provider.recognize_line_image.call_count, 2)

    def test_unknown_low_confidence_or_conflicting_count_falls_back(self):
        for reads in ([], [OCRText("x0", .89)], [OCRText("x0", .99), OCRText("x1", .99)],
                      [OCRText("xO", .99)], [OCRText("x0", float("nan"))]):
            with self.subTest(reads=reads):
                self.provider.recognize_line_image.return_value = reads
                self.assertTrue(self.observe().observations["fast_fallback"])

    def test_dimmed_popup_and_clouds_cannot_authorize_input(self):
        images = [np.uint8(self.image * .6), self.image.copy(), self.image.copy()]
        images[1][200:400, 400:850] = 190
        images[2][150:490, 240:1050] = 255
        for image in images:
            self.assertEqual(self.observe(image).scene, "unknown")
        self.provider.recognize_line_image.assert_not_called()

    def test_moved_missing_or_other_card_requires_full_reacquisition(self):
        image = self.image.copy()
        image[590:710, 100:190] = 0
        image[590:710, 220:310] = self.image[590:710, 100:190]
        snapshot = self.observe(image)
        self.assertTrue(snapshot.observations["fast_fallback"])
        self.provider.recognize_line_image.assert_not_called()

    def test_waiting_battle_does_not_read_counts_but_settlement_does_full_recognition(self):
        self.assertFalse(self.observe(purpose="settlement").observations["fast_fallback"])
        self.provider.recognize_line_image.assert_not_called()
        self.reader.recognize.side_effect = lambda path: SceneSnapshot("settlement", .95, path)
        result = self.observe(np.zeros_like(self.image), purpose="settlement")
        self.assertEqual(result.scene, "settlement")
        self.assertTrue(result.observations["fast_fallback"])

    def test_same_path_overwrite_invalidates_frame_and_template_cache(self):
        path = self.write("cache", self.image)
        self.assertGreater(read_frame(path).max(), 0)
        self.assertGreater(read_template(path).max(), 0)
        self.write("cache", np.zeros_like(self.image))
        self.assertEqual(read_frame(path).max(), 0)
        self.assertEqual(read_template(path).max(), 0)

    def test_boundary_detection_requires_closed_current_selection_outline(self):
        self.assertIsNone(selected_card_bbox(self.previous.screenshot_path, self.slot["bbox"]))
        image = np.zeros_like(self.image)
        cv2.rectangle(image, (98, 583), (192, 703), (255, 255, 255), 2)
        self.assertIsNotNone(selected_card_bbox(self.write("selected", image), self.slot["bbox"]))
        image[620:680, 95:103] = 0
        self.assertIsNone(selected_card_bbox(self.write("partial", image), self.slot["bbox"]))
