from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.army_editor import recognize_army_editor, recognize_hero_loadout, visual_fingerprint_matches
from autococ.ocr import OCRText


class ArmyEditorRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name) / "frame.png"
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        self.directory.cleanup()

    def read(self, texts):
        cv2.imwrite(str(self.path), self.image)
        return recognize_army_editor(self.path, None, texts)

    def test_current_tab_only_emits_named_navigation(self):
        editor = self.read([OCRText("我的军队", .99, (210, 60, 310, 90)),
                            OCRText("已保存的配置", .99, (570, 60, 710, 90))])
        self.assertEqual(editor["surface"], "current")
        self.assertTrue(editor["ready"])
        self.assertEqual([(x["action"], x["point"]) for x in editor["controls"]],
                         [("open_saved", [640, 75])])
        self.assertEqual(editor["cards"], [])

    def test_saved_page_controls_do_not_make_unidentified_preset_complete(self):
        editor = self.read([OCRText("我的军队", .99, (210, 60, 310, 90)),
                            OCRText("已保存的配置", .99, (570, 60, 710, 90)),
                            OCRText("使用", .99, (1150, 170, 1190, 195)),
                            OCRText("编辑", .99, (1150, 230, 1190, 255)),
                            OCRText("新建", .99, (1150, 670, 1190, 695))])
        self.assertEqual(editor["surface"], "saved")
        self.assertTrue(editor["ready"])
        self.assertFalse(editor["presets"][0]["complete"])
        self.assertEqual(editor["controls"][0]["action"], "create_preset")

    def test_edit_title_anchors_capacity_but_does_not_invent_cards(self):
        editor = self.read([OCRText("编辑军队配置1", .99, (570, 52, 710, 80)),
                            OCRText("270/335", .99, (570, 150, 670, 180))])
        self.assertEqual(editor["surface"], "edit")
        self.assertEqual(editor["preset_id"], "1")
        self.assertEqual(editor["capacities"]["troop"], {"used": 270, "total": 335})
        self.assertFalse(editor["complete"])
        self.assertEqual(editor["controls"], [])

    def test_picker_needs_visual_panel_and_capacity_anchor(self):
        self.image[460:665, 50:1200] = 230
        editor = self.read([OCRText("270/335", .99, (570, 70, 670, 95))])
        self.assertEqual(editor["surface"], "picker")
        self.assertTrue(editor["ready"])
        self.assertEqual(editor["controls"], [])

    def test_without_page_anchors_remains_unknown(self):
        editor = self.read([OCRText("编辑", .99, (1150, 230, 1190, 255))])
        self.assertEqual(editor["surface"], "unknown")
        self.assertFalse(editor["ready"])

    def test_hero_loadout_fingerprint_follows_verified_identity_not_column(self):
        boxes = [(22, 184, 147, 670), (153, 184, 278, 670)]
        rng = np.random.default_rng(25)
        icons = {name: [rng.integers(0, 256, (66, 109, 3), dtype=np.uint8),
                        rng.integers(0, 256, (56, 51, 3), dtype=np.uint8),
                        rng.integers(0, 256, (56, 51, 3), dtype=np.uint8)]
                 for name in ("barbarian_king", "archer_queen")}

        def frame(order, filename):
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cards = []
            for hero_id, box in zip(order, boxes):
                l, t, r, b = box
                regions = ((l+8, b-135, r-8, b-69),
                           (l+8, b-66, l+59, b-10),
                           (r-59, b-66, r-8, b-10))
                for icon, (il, it, ir, ib) in zip(icons[hero_id], regions):
                    image[it:ib, il:ir] = icon
                cards.append({"unit_id": hero_id, "kind": "hero", "source": "army",
                              "count": 1, "confidence": .99, "card_bbox": list(box)})
            path = Path(self.directory.name) / filename
            cv2.imwrite(str(path), image)
            return path, cards

        first_path, first_cards = frame(("barbarian_king", "archer_queen"), "first.png")
        second_path, second_cards = frame(("archer_queen", "barbarian_king"), "second.png")
        texts = [OCRText("我的军队", .99, (210, 60, 310, 90)),
                 OCRText("2/4", .99, (50, 150, 98, 176))]
        first = recognize_hero_loadout(first_path, texts, first_cards)
        second = recognize_hero_loadout(second_path, texts, second_cards)
        self.assertTrue(first["complete"])
        self.assertTrue(second["complete"])
        for hero_id in icons:
            for kind in ("pet_visual", "equipment_1_visual", "equipment_2_visual"):
                self.assertTrue(visual_fingerprint_matches(
                    first["hero_loadout"][hero_id][kind], second["hero_loadout"][hero_id][kind]))
        self.assertFalse(visual_fingerprint_matches(
            first["hero_loadout"]["barbarian_king"]["pet_visual"],
            second["hero_loadout"]["archer_queen"]["pet_visual"]))

    def test_hero_identity_missing_blocks_visual_association(self):
        cv2.imwrite(str(self.path), self.image)
        cards = [{"unit_id": None, "kind": "hero", "source": "army", "count": 1,
                  "confidence": .99, "card_bbox": [22, 184, 147, 670]}]
        result = recognize_hero_loadout(
            self.path, [OCRText("我的军队", .99, (210, 60, 310, 90)),
                        OCRText("1/4", .99, (50, 150, 98, 176))], cards)
        self.assertFalse(result["complete"])
        self.assertEqual(result["unknowns"], ["hero_card_identity_or_geometry_unverified"])

    def test_saved_overview_cannot_masquerade_as_current_hero_page(self):
        cv2.imwrite(str(self.path), self.image)
        cards = [{"unit_id": "barbarian_king", "kind": "hero", "source": "army",
                  "count": 1, "confidence": .99, "card_bbox": [22, 184, 147, 670]}]
        result = recognize_hero_loadout(self.path, [
            OCRText("我的军队", .99, (210, 60, 310, 90)),
            OCRText("使用", .99, (1150, 170, 1190, 195))], cards)
        self.assertFalse(result["complete"])
        self.assertEqual(result["unknowns"], ["hero_loadout_page_unanchored"])

    def test_hero_capacity_must_equal_named_cards(self):
        cv2.imwrite(str(self.path), self.image)
        cards = [{"unit_id": "barbarian_king", "kind": "hero", "source": "army",
                  "count": 1, "confidence": .99, "card_bbox": [22, 184, 147, 670]}]
        result = recognize_hero_loadout(self.path, [
            OCRText("我的军队", .99, (210, 60, 310, 90)),
            OCRText("2/4", .99, (50, 150, 98, 176))], cards)
        self.assertFalse(result["complete"])
        self.assertEqual(result["unknowns"], ["hero_capacity_or_card_count_unverified"])


if __name__ == "__main__":
    unittest.main()
