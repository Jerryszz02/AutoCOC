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


class EnemyResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "hud.png"
        self.provider = Mock()
        self.provider.recognize_line.return_value = []
        self.recognizer = ScreenshotRecognizer(provider=self.provider)
        self.write_hud()

    def write_hud(self, *, missing=(), dy=0, resolution=(1280, 720)):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for name, left, top in (("gold", 31, 99), ("elixir", 31, 137), ("dark_elixir", 34, 178)):
            if name in missing:
                continue
            template = cv2.imread(str(ROOT / "assets/templates" / f"enemy_{name}.png"))
            height, width = template.shape[:2]
            image[top + dy:top + dy + height, left:left + width] = template
        cv2.imwrite(str(self.path), cv2.resize(image, resolution, interpolation=cv2.INTER_NEAREST))

    def read(self, texts, resolution=(1280, 720)):
        return self.recognizer._enemy_resources(self.path, texts, resolution)

    def test_higher_confidence_defender_levels_do_not_replace_any_resource(self):
        texts = [
            OCRText("1688575", .99994, (63, 98, 172, 124)),
            OCRText("897 982", .93363, (61, 136, 156, 162)),
            OCRText("15439", .99987, (60, 172, 138, 202)),
            OCRText("94", .99995, (86, 118, 108, 134)),
            OCRText("93", 1, (86, 156, 108, 172)),
            OCRText("71", 1, (86, 194, 108, 210)),
            OCRText("99999999", 1, (1148, 69, 1228, 92)),
        ]
        values, evidence = self.read(texts)
        self.assertEqual(values, {"gold": 1688575, "elixir": 897982, "dark_elixir": 15439, "gems": None})
        self.assertEqual(evidence["gold"]["text"], "1688575")
        self.assertIn("resource_icon", evidence["gold"])
        self.provider.recognize_line.assert_not_called()

    def test_missing_amount_does_not_turn_a_defender_level_into_loot(self):
        values, _ = self.read([OCRText("94", 1, (86, 118, 108, 134)),
                               OCRText("93", 1, (63, 141, 82, 157))])
        self.assertTrue(all(value is None for value in values.values()))
        self.assertEqual(self.provider.recognize_line.call_count, 3)

    def test_no_icon_means_unknown_even_with_valid_looking_numeric_rows(self):
        self.write_hud(missing=("gold", "elixir", "dark_elixir"))
        values, evidence = self.read([OCRText("999999", 1, (63, 98, 170, 124))])
        self.assertTrue(all(value is None for value in values.values()))
        self.assertEqual(evidence, {})
        self.provider.recognize_line.assert_not_called()

    def test_missing_dark_icon_never_substitutes_an_elixir_icon(self):
        self.write_hud(missing=("dark_elixir",))
        values, _ = self.read([OCRText("999", 1, (60, 172, 105, 202))])
        self.assertIsNone(values["dark_elixir"])
        self.assertEqual(self.provider.recognize_line.call_count, 2)

    def test_small_amount_and_zero_are_valid_without_magnitude_guessing(self):
        values, _ = self.read([OCRText("94", .95, (63, 98, 99, 124)),
                               OCRText("0", .95, (61, 136, 80, 162)),
                               OCRText("0", .95, (60, 172, 79, 202))])
        self.assertEqual((values["gold"], values["elixir"], values["dark_elixir"]), (94, 0, 0))

    def test_conflicting_or_fragmented_same_row_numbers_stay_unknown(self):
        for other in (OCRText("9999999", 1, (63, 98, 172, 124)),
                      OCRText("575", 1, (139, 98, 172, 124))):
            with self.subTest(other=other):
                values, evidence = self.read([OCRText("1688", .95, (63, 98, 115, 124)), other])
                self.assertIsNone(values["gold"])
                self.assertNotIn("gold", evidence)

    def test_rows_follow_icons_when_the_hud_moves(self):
        self.write_hud(dy=7)
        values, evidence = self.read([OCRText("125", .95, (63, 105, 110, 131))])
        self.assertEqual(values["gold"], 125)
        self.assertEqual(evidence["gold"]["resource_icon"]["bbox_at_1280x720"], (31, 106, 59, 136))

    def test_custom_baseline_and_native_resolution_are_scaled_once(self):
        self.write_hud(resolution=(2560, 1440))
        self.recognizer = ScreenshotRecognizer(provider=self.provider, baseline_resolution=(2560, 1440))
        values, _ = self.read([OCRText("1688575", .99, (126, 196, 344, 248)),
                               OCRText("94", 1, (172, 236, 216, 268))], (2560, 1440))
        self.assertEqual(values["gold"], 1688575)

    def test_line_fallback_uses_icon_anchored_crop_above_defender_level(self):
        self.write_hud(resolution=(2560, 1440))
        self.provider.recognize_line.side_effect = lambda path, roi: [OCRText("1688575", .99, roi)] if roi[1] == 192 else []
        values, evidence = self.read([OCRText("94", 1, (86, 118, 108, 134))], (2560, 1440))
        self.assertEqual(values["gold"], 1688575)
        self.provider.recognize_line.assert_any_call(self.path, (120, 192, 448, 244))
        self.assertEqual(evidence["gold"]["source"], "line_roi")
        self.assertEqual(evidence["gold"]["bbox"], (60, 96, 224, 122))

    def test_inspected_live_frame_and_recorded_ocr_regression(self):
        run = ROOT / "reports/20260923-012829-323923-b0c8cf92"
        path = run / "frames/00010-battle-candidate.png"
        if not path.is_file():
            self.skipTest("Optional inspected live fixture is absent")
        events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        event = next(event for event in events if event.get("kind") == "observation"
                     and event["frame"].endswith("00010-battle-candidate.png"))
        texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"])) for item in event["observations"]["ocr"]]
        values, _ = self.recognizer._enemy_resources(path, texts, (2560, 1440))
        self.assertEqual(values, {"gold": 1688575, "elixir": 897982, "dark_elixir": 15439, "gems": None})


if __name__ == "__main__":
    unittest.main()
