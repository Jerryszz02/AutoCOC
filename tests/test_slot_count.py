from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.errors import SceneError
from autococ.vision import ScreenshotRecognizer


class SlotCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "frame.png"
        cv2.imencode(".png", np.zeros((1440, 2560, 3), dtype=np.uint8))[1].tofile(self.path)
        self.bbox = (180, 590, 250, 700)
        self.provider = Mock()
        self.provider.recognize.return_value = []
        self.provider.recognize_line.return_value = []
        self.reader = ScreenshotRecognizer(provider=self.provider)

    def test_explicit_local_zero_is_high_confidence_and_has_original_frame_evidence(self) -> None:
        self.provider.recognize_line.return_value = [OCRText("x0", .97, (0, 0, 140, 112))]
        result = self.reader.recognize_slot_count(self.path, self.bbox)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["confidence"], .97)
        self.assertEqual(result["frame"], str(self.path))
        self.assertEqual(result["slot_bbox"], list(self.bbox))
        self.assertEqual(result["roi"], [180, 574, 250, 630])
        self.assertEqual(result["readings"][0]["bbox"], result["roi"])
        self.assertEqual(result["readings"][0]["threshold"], 120)
        self.provider.recognize.assert_called_once_with(self.path, (360, 1148, 500, 1260))
        self.assertEqual(self.provider.recognize_line.call_args.args[1], (0, 0, 140, 112))

    def test_positive_local_count_does_not_require_preprocessing_to_succeed(self) -> None:
        self.provider.recognize.return_value = [OCRText("x12", .96, (370, 1160, 490, 1236))]
        result = self.reader.recognize_slot_count(self.path, self.bbox)
        self.assertEqual(result["count"], 12)
        self.assertEqual(result["readings"][0]["bbox"], (185, 580, 245, 618))

    def test_prior_quantity_bbox_tightens_line_crop_with_scaled_padding(self) -> None:
        count_bbox = (215, 590, 245, 616)
        self.provider.recognize_line.return_value = [OCRText("x0", .97)]
        result = self.reader.recognize_slot_count(self.path, self.bbox, count_bbox=count_bbox)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["count_bbox"], list(count_bbox))
        self.assertEqual(result["line_roi"], [197, 586, 249, 624])
        self.assertEqual(result["readings"][0]["bbox"], result["line_roi"])
        self.assertEqual(result["readings"][0]["source"], "count_bbox_threshold_line")
        self.assertEqual(self.provider.recognize_line.call_args.args[1], (0, 0, 104, 76))
        self.provider.recognize.assert_called_once_with(self.path, (360, 1148, 500, 1260))

    def test_prior_quantity_bbox_must_belong_to_this_card_header(self) -> None:
        for bbox in ((280, 590, 310, 616), (215, 650, 245, 676), (245, 590, 215, 616)):
            with self.subTest(bbox=bbox), self.assertRaises(SceneError):
                self.reader.recognize_slot_count(self.path, self.bbox, count_bbox=bbox)
        self.provider.recognize.assert_not_called()

    def test_current_weak_detection_expands_crop_without_proving_value(self) -> None:
        self.provider.recognize.return_value = [OCRText("x0", .83, (400, 1180, 496, 1252))]
        self.provider.recognize_line.return_value = [OCRText("x0", .97)]
        result = self.reader.recognize_slot_count(self.path, self.bbox, count_bbox=(215, 590, 245, 616))
        self.assertEqual(result["line_roi"], [196, 586, 250, 630])
        self.assertEqual(result["count"], 0)
        self.provider.recognize_line.return_value = []
        self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox, count_bbox=(215, 590, 245, 616))["count"])

    def test_tight_crop_still_rejects_missing_ambiguous_and_conflicting_readings(self) -> None:
        bbox = (215, 590, 245, 616)
        for readings in ([], [OCRText("x0", .89844)], [OCRText("xO", .99)],
                         [OCRText("x0", .99), OCRText("x1", .99)]):
            with self.subTest(readings=readings):
                self.provider.recognize_line.return_value = readings
                self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox, count_bbox=bbox)["count"])
        self.provider.recognize.return_value = [OCRText("x1", .99, (430, 1180, 490, 1232))]
        self.provider.recognize_line.return_value = [OCRText("x0", .99)]
        self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox, count_bbox=bbox)["count"])

    def test_missing_or_low_confidence_count_is_unknown_not_zero(self) -> None:
        for confidence in (.89, float("nan"), float("inf"), 1.01):
            with self.subTest(confidence=confidence):
                self.provider.recognize_line.return_value = [OCRText("x0", confidence)]
                self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox)["count"])
        self.provider.recognize_line.return_value = []
        self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox)["count"])

    def test_conflicting_high_confidence_counts_are_unknown(self) -> None:
        self.provider.recognize.return_value = [OCRText("x1", .99, (370, 1160, 490, 1236))]
        self.provider.recognize_line.return_value = [OCRText("x0", .99)]
        self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox)["count"])

    def test_matching_independent_header_reads_are_accepted(self) -> None:
        self.provider.recognize.return_value = [OCRText("X 0", .94, (370, 1160, 490, 1236))]
        self.provider.recognize_line.return_value = [OCRText("×0", .98)]
        self.assertEqual(self.reader.recognize_slot_count(self.path, self.bbox)["count"], 0)

    def test_letter_o_and_incomplete_or_mixed_text_never_become_zero(self) -> None:
        for text in ("xO", "XO", "×O", "0", "x-1", "x1 x0", "x", ""):
            with self.subTest(text=text):
                self.provider.recognize_line.return_value = [OCRText(text, .99)]
                self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox)["count"])
        self.provider.recognize.return_value = [OCRText("xO", .99, (370, 1160, 490, 1236))]
        self.provider.recognize_line.return_value = [OCRText("x0", .99)]
        self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox)["count"])

    def test_adjacent_card_number_is_outside_original_header(self) -> None:
        self.provider.recognize.return_value = [OCRText("x0", .99, (580, 1160, 680, 1236))]
        self.assertIsNone(self.reader.recognize_slot_count(self.path, self.bbox)["count"])

    def test_real_last_unit_and_three_gray_zero_frames_with_local_ocr(self) -> None:
        folder = Path(__file__).resolve().parents[1] / "reports/20260923-012829-323923-b0c8cf92/frames"
        if not folder.is_dir():
            self.skipTest("Optional live exhausted-card fixture is absent")
        reader = ScreenshotRecognizer()
        for number, expected in ((16, 1), (17, 0), (18, 0), (19, 0)):
            with self.subTest(frame=number):
                path = next(folder.glob(f"{number:05d}-*.png"))
                result = reader.recognize_slot_count(path, (92, 591, 183, 708))
                self.assertEqual(result["count"], expected)
                self.assertGreaterEqual(result["confidence"], .9)
                self.assertEqual(result["frame"], str(path))
                self.assertTrue(any(item["text"] == f"x{expected}" and item["confidence"] >= .9
                                    for item in result["readings"]))
                tight = reader.recognize_slot_count(path, (92, 591, 183, 708), count_bbox=(150, 588, 180, 616))
                self.assertEqual(tight["count"], expected)
                self.assertGreaterEqual(tight["confidence"], .9)

    def test_real_balloon_positive_and_gray_zero_use_prior_numeric_bbox(self) -> None:
        folder = Path(__file__).resolve().parents[1] / "reports/20260923-022800-094669-1aa03c4b/frames"
        if not folder.is_dir():
            self.skipTest("Optional live balloon count fixture is absent")
        reader = ScreenshotRecognizer()
        for number, expected in ((16, 2), (17, 0), (18, 0), (19, 0)):
            with self.subTest(frame=number):
                path = next(folder.glob(f"{number:05d}-*.png"))
                result = reader.recognize_slot_count(path, (285, 591, 377, 713), count_bbox=(336, 590, 375, 618))
                self.assertEqual(result["count"], expected)
                line = next(item for item in result["readings"] if item["source"] == "count_bbox_threshold_line")
                self.assertEqual(line["text"], f"x{expected}")
                self.assertGreaterEqual(line["confidence"], .9)
                self.assertEqual(result["frame"], str(path))
        blank = reader.recognize_slot_count(self.path, (285, 591, 377, 713), count_bbox=(336, 590, 375, 618))
        self.assertIsNone(blank["count"], "A historical quantity bbox is not evidence of a current number")

    def test_real_gray_electro_zero_is_wider_and_lower_than_previous_one(self) -> None:
        folder = Path(__file__).resolve().parents[1] / "reports/20260923-023613-726879-9389d9e7/frames"
        if not folder.is_dir():
            self.skipTest("Optional shifted gray electro count fixture is absent")
        reader = ScreenshotRecognizer()
        for number in (15, 16, 17):
            with self.subTest(frame=number):
                path = next(folder.glob(f"{number:05d}-*.png"))
                result = reader.recognize_slot_count(path, (92, 591, 183, 708), count_bbox=(150, 590, 180, 616))
                self.assertEqual(result["count"], 0)
                line = next(item for item in result["readings"] if item["source"] == "count_bbox_threshold_line")
                self.assertEqual(line["text"], "x0")
                self.assertGreaterEqual(line["confidence"], .9)
                self.assertEqual(result["line_roi"], [131, 586, 183, 626])
                left, top, right, bottom = result["line_roi"]
                self.assertLessEqual(left, 138)
                self.assertLessEqual(top, 592)
                self.assertGreaterEqual(right, 180)
                self.assertGreaterEqual(bottom, 622)


if __name__ == "__main__":
    unittest.main()
