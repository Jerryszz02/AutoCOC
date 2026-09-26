"""The current-army picker must not be confused with a saved-plan editor."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf
from unittest.mock import patch
from importlib.util import find_spec

import cv2
import numpy as np

from autococ.army_editor import _picker_count, recognize_army_editor
from autococ.ocr import OCRText
from autococ.ocr import RapidOCRProvider
from autococ.vision import ScreenshotRecognizer


class CurrentPickerTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "picker.png"
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.image[460:665, 50:1200] = 230

    def tearDown(self):
        self.directory.cleanup()

    def recognize(self, texts, identities=None):
        cv2.imwrite(str(self.path), self.image)
        identities = identities or {}

        def identity(_image, box, kind, **_kwargs):
            unit_id = identities.get((kind, box[0]))
            return {"unit_id": unit_id, "confidence": .98 if unit_id else None,
                    "reason": "portrait_template" if unit_id else "no_supported_template"}

        with patch("autococ.army_editor.recognize_card_identity", side_effect=identity):
            return recognize_army_editor(self.path, None, texts, client_version="18.600.7")

    def minus(self, x, y):
        self.image[y-13:y+13, x-13:x+13] = (15, 20, 210)
        self.image[y-2:y+2, x-8:x+8] = (245, 245, 245)

    def test_troop_picker_counts_only_independently_named_top_cards(self):
        self.minus(636, 210)
        self.minus(740, 210)
        texts = [OCRText("我的军队", .99, (210, 60, 310, 90)),
                 OCRText("已保存的配置", .99, (570, 60, 710, 90)),
                 OCRText("325/335", .99, (580, 150, 650, 180)),
                 OCRText("x10", .99, (562, 198, 592, 219)),
                 OCRText("x1", .99, (666, 198, 690, 219))]
        editor = self.recognize(texts, {("troop", 558): "electro_dragon",
                                        ("troop", 662): "dragon_rider"})
        self.assertEqual(editor["surface"], "current_picker")
        self.assertEqual(editor["editing_kind"], "troop")
        self.assertTrue(editor["complete_kinds"]["troop"])
        self.assertEqual([(card["unit_id"], card["count"]) for card in editor["cards"]],
                         [("electro_dragon", 10), ("dragon_rider", 1)])
        self.assertEqual([control["action"] for control in editor["controls"]],
                         ["close_picker", "decrement", "decrement"])

    def test_missing_identity_blocks_complete_count_and_decrement(self):
        self.minus(636, 210)
        editor = self.recognize([
            OCRText("我的军队", .99, (210, 60, 310, 90)),
            OCRText("已保存的配置", .99, (570, 60, 710, 90)),
            OCRText("335/335", .99, (580, 150, 650, 180)),
            OCRText("x10", .99, (562, 198, 592, 219))])
        self.assertFalse(editor["complete_kinds"]["troop"])
        self.assertEqual([control["action"] for control in editor["controls"]],
                         ["close_picker"])

    def test_shifted_spell_picker_uses_separate_page_anchors(self):
        self.minus(636, 303)
        self.minus(740, 303)
        texts = [OCRText("4/4", .99, (50, 66, 95, 90)),
                 OCRText("强化军队", .99, (890, 45, 960, 75)),
                 OCRText("强化英雄", .99, (1080, 45, 1160, 75)),
                 OCRText("335/335", .99, (580, 66, 660, 94)),
                 OCRText("11/11", .99, (580, 245, 635, 275)),
                 OCRText("x6", .99, (562, 290, 590, 311)),
                 OCRText("x5", .99, (666, 290, 690, 311))]
        editor = self.recognize(texts, {("spell", 558): "lightning_spell",
                                        ("spell", 662): "totem_spell"})
        self.assertEqual((editor["surface"], editor["editing_kind"]),
                         ("current_picker", "spell"))
        self.assertEqual(editor["capacities"]["spell"], {"used": 11, "total": 11})
        self.assertTrue(editor["complete_kinds"]["spell"])
        self.assertEqual([control["action"] for control in editor["controls"]],
                         ["close_picker", "decrement", "decrement"])
        self.assertEqual(editor["controls"][0]["point"], [1050, 240])

    def test_conflicting_ocr_counts_cannot_be_overridden_by_x1_sample(self):
        sample = cv2.imread(str(Path(__file__).resolve().parents[1] /
                               "assets/catalogs/current_picker_x1_full_18_600_7.png"),
                            cv2.IMREAD_GRAYSCALE)
        self.image[194:224, 663:708] = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        count, evidence = _picker_count(
            None, self.path,
            [OCRText("x1", .99, (665, 198, 688, 220)),
             OCRText("x2", .99, (665, 198, 688, 220))],
            (662, 194, 760, 292), (1280, 720), self.image,
            "18.600.7", "dragon_rider")
        self.assertIsNone(count)
        self.assertEqual(evidence["source"], "full_ocr")

    def test_extra_digit_after_x1_blocks_sample_fallback(self):
        source = Path(__file__).resolve().parents[1] / "tests/fixtures/army_current_new_20260926.png"
        image = cv2.resize(cv2.imread(str(source)), (1280, 720))
        # Use the independently sampled troop-picker header and add a second
        # numeral to its right. The x1 glyph remains, but the whole header
        # must no longer match an exact one-unit quantity.
        picker = Path(__file__).resolve().parents[1] / "assets/catalogs/current_picker_x1_full_18_600_7.png"
        full = cv2.imread(str(picker), cv2.IMREAD_GRAYSCALE)
        image[194:224, 663:708] = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
        image[197:219, 695:706] = image[197:219, 675:686]
        count, evidence = _picker_count(None, self.path, [],
                                         (662, 194, 760, 292), (1280, 720),
                                         image, "18.600.7", "dragon_rider")
        self.assertIsNone(count)
        self.assertGreaterEqual(evidence["x1_sample_score"], .9)
        self.assertLess(evidence["x1_full_score"], .96)

    def test_selected_count_must_balance_observed_capacity(self):
        self.minus(636, 303)
        editor = self.recognize([
            OCRText("4/4", .99, (50, 66, 95, 90)),
            OCRText("强化军队", .99, (890, 45, 960, 75)),
            OCRText("强化英雄", .99, (1080, 45, 1160, 75)),
            OCRText("11/11", .99, (580, 245, 635, 275)),
            OCRText("x6", .99, (562, 290, 590, 311))],
            {("spell", 558): "lightning_spell"})
        self.assertFalse(editor["complete_kinds"]["spell"])
        self.assertIn("picker_capacity_or_housing_mismatch", editor["unknowns"])

    def test_local_contrast_agreement_reads_quantity_after_full_ocr_misses(self):
        class Provider:
            def __init__(self, responses):
                self.responses = iter(responses)

            def recognize_line(self, _path, _roi):
                return []

            def recognize_line_image(self, _image):
                return [next(self.responses)]

        provider = Provider([OCRText("x7", .803), OCRText("x7", .787),
                             OCRText("x7", .813)])
        count, evidence = _picker_count(provider, self.path, [],
                                         (558, 287, 656, 385), (1280, 720),
                                         self.image, "18.600.7", "lightning_spell")
        self.assertEqual((count, evidence["source"]), (7, "contrast_consensus"))
        provider = Provider([OCRText("x7", .803), OCRText("x8", .787),
                             OCRText("x7", .813)])
        count, evidence = _picker_count(provider, self.path, [],
                                         (558, 287, 656, 385), (1280, 720),
                                         self.image, "18.600.7", "lightning_spell")
        self.assertIsNone(count)

    def test_conflicting_contrast_votes_block_version_sample_fallback(self):
        class Provider:
            def __init__(self):
                self.reads = iter((OCRText("x7", .803), OCRText("x8", .787),
                                   OCRText("x7", .813)))

            def recognize_line(self, _path, _roi):
                return []

            def recognize_line_image(self, _image):
                return [next(self.reads)]

        sample = cv2.imread(str(Path(__file__).resolve().parents[1] /
                               "assets/catalogs/current_picker_x6_18_600_7.png"),
                            cv2.IMREAD_GRAYSCALE)
        self.image[287:317, 559:604] = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        count, _ = _picker_count(Provider(), self.path, [],
                                 (558, 287, 656, 385), (1280, 720),
                                 self.image, "18.600.7", "lightning_spell")
        self.assertIsNone(count)


if __name__ == "__main__":
    import unittest
    unittest.main()


@skipIf(find_spec("rapidocr") is None, "bundled RapidOCR is unavailable")
class SampledCurrentPickerTests(TestCase):
    def test_real_x7_and_capacity_gray_to_active_samples(self):
        recognizer = ScreenshotRecognizer(provider=RapidOCRProvider())
        recognizer.client_version = "18.600.7"
        root = Path(__file__).resolve().parent / "fixtures"
        cases = (
            ("current_picker_spell_x7_20260926.png", 7, 4, 11, False),
            ("current_picker_spell_x6_x4_20260926.png", 6, 4, 10, True),
        )
        for filename, lightning, totem, used, enabled in cases:
            with self.subTest(frame=filename):
                snapshot = recognizer.recognize(root / filename, battle_details=False)
                editor = snapshot.observations["army_editor"]
                self.assertEqual((snapshot.scene, editor["surface"], editor["editing_kind"]),
                                 ("training", "current_picker", "spell"))
                self.assertTrue(editor["complete_kinds"]["spell"])
                self.assertEqual(editor["capacities"]["spell"],
                                 {"used": used, "total": 11})
                cards = {card["unit_id"]: card for card in editor["cards"]}
                self.assertEqual((cards["lightning_spell"]["count"],
                                  cards["totem_spell"]["count"]),
                                 (lightning, totem))
                increments = {item["unit_id"]: item for item in editor["controls"]
                              if item["action"] == "increment"}
                self.assertEqual(set(increments), {"lightning_spell", "totem_spell"})
                self.assertTrue(all(item["enabled"] is enabled and
                                    item["capacity_blocked"] is not enabled
                                    for item in increments.values()))
                if lightning == 7:
                    self.assertEqual(cards["lightning_spell"]["evidence"]["count"]["source"],
                                     "contrast_consensus")
