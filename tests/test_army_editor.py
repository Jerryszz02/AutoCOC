from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

import cv2
import numpy as np

from autococ.army_editor import recognize_army_editor, recognize_hero_loadout, visual_fingerprint_matches
from autococ.army_manifest import recognize_army_manifest
from autococ.ocr import OCRText, RapidOCRProvider


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

    def test_current_save_icon_needs_visual_glyph(self):
        self.image[54:100, 388:443] = (20, 20, 20)
        self.image[61:83, 404:428] = (235, 235, 235)
        editor = self.read([OCRText("我的军队", .99, (210, 60, 310, 90)),
                            OCRText("已保存的配置", .99, (570, 60, 710, 90))])
        self.assertIn("save_current", [item["action"] for item in editor["controls"]])

    def test_save_dialog_only_exposes_unique_visible_empty_enabled_slot(self):
        self.image[:] = (75, 95, 115)
        self.image[110:240, 100:350] = (235, 70, 35)
        self.image[305:430, 1120:1230] = (150, 175, 195)
        self.image[455:580, 1120:1230] = (75, 75, 75)
        self.image[610:710, 1120:1230] = (75, 75, 75)
        texts = [OCRText("将军队保存为新配置", .99, (545, 55, 735, 85))]
        texts.extend(OCRText("保存军队", .99, (1145, y, 1215, y + 25))
                     for y in (180, 360, 512, 660))
        editor = self.read(texts)
        self.assertEqual(editor["surface"], "save_current")
        self.assertTrue(editor["ready"])
        self.assertEqual([(item["action"], item["point"]) for item in editor["controls"]],
                         [("save_to_empty", [1180, 372])])
        self.image[305:430, 100:350] = (235, 70, 35)
        editor = self.read(texts)
        self.assertFalse(editor["ready"])
        self.assertEqual(editor["controls"], [])
        self.image[305:430, 100:350] = (75, 95, 115)
        self.image[305:410, 100:310] = (95, 95, 95)
        self.image[305:410, 100:310:3] = (225, 225, 225)
        editor = self.read(texts)
        self.assertFalse(editor["ready"])  # A gray card still occupies the row.
        self.image[305:410, 100:310] = (75, 95, 115)
        editor = self.read(texts + [OCRText("x2", .99, (200, 360, 225, 380))])
        self.assertFalse(editor["ready"])  # Explicit quantity also forbids an empty slot.

    def test_saved_page_controls_do_not_make_unidentified_preset_complete(self):
        editor = self.read([OCRText("我的军队", .99, (210, 60, 310, 90)),
                            OCRText("已保存的配置", .99, (570, 60, 710, 90)),
                            OCRText("军队配置1", .99, (45, 130, 140, 155)),
                            OCRText("使用", .99, (1150, 170, 1190, 195)),
                            OCRText("编辑", .99, (1150, 230, 1190, 255)),
                            OCRText("新建", .99, (1150, 670, 1190, 695))])
        self.assertEqual(editor["surface"], "saved")
        self.assertTrue(editor["ready"])
        self.assertFalse(editor["presets"][0]["complete"])
        self.assertEqual(editor["presets"][0]["preset_id"], "1")
        self.assertTrue(editor["occupied_preset_ids_complete"])
        self.assertEqual([item["action"] for item in editor["controls"]],
                         ["open_current", "create_preset"])

    def test_edit_title_anchors_capacity_but_does_not_invent_cards(self):
        editor = self.read([OCRText("编辑军队配置1", .99, (570, 52, 710, 80)),
                            OCRText("270/335", .99, (570, 150, 670, 180))])
        self.assertEqual(editor["surface"], "edit")
        self.assertEqual(editor["preset_id"], "1")
        self.assertEqual(editor["capacities"]["troop"], {"used": 270, "total": 335})
        self.assertFalse(editor["complete"])
        self.assertEqual(editor["controls"], [])

    def test_ocr_misread_spell_capacity_is_rejected(self):
        editor = self.read([OCRText("编辑军队配置3", .99, (570, 52, 710, 80)),
                            OCRText("0/335", .99, (570, 150, 670, 180)),
                            OCRText("0/110", .99, (570, 290, 670, 320))])
        self.assertNotIn("spell", editor["capacities"])

    def test_picker_needs_visual_panel_and_capacity_anchor(self):
        self.image[460:665, 50:1200] = 230
        editor = self.read([OCRText("270/335", .99, (570, 70, 670, 95))])
        self.assertEqual(editor["surface"], "picker")
        self.assertTrue(editor["ready"])
        self.assertEqual(editor["controls"], [])

    def test_current_troop_picker_exposes_only_sampled_active_archer(self):
        fixture_root = Path(__file__).parent / "fixtures"
        metadata = json.loads((fixture_root / "troop_picker_18_647_2.json").read_text(encoding="utf-8"))
        provider = RapidOCRProvider()
        for frame in metadata["frames"]:
            with self.subTest(frame=frame["fixture"]):
                path = fixture_root / frame["fixture"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 frame["fixture_sha256"])
                texts = [OCRText(**item) for item in frame["ocr"]]
                editor = recognize_army_editor(path, provider, texts,
                                               client_version="18.647.2")
                self.assertEqual(editor["surface"], "current_picker")
                self.assertTrue(editor["ready"])
                self.assertTrue(editor["complete_kinds"]["troop"], editor["unknowns"])
                self.assertEqual([(card["unit_id"], card["count"]) for card in editor["cards"]],
                                 [("electro_dragon", 10), ("dragon", 1),
                                  ("baby_dragon", 1), ("balloon", 2)])
                self.assertEqual(editor["unit_housing_space"],
                                 {"electro_dragon": 30, "dragon": 20,
                                  "baby_dragon": 10, "balloon": 5, "archer": 1})
                additions = [control for control in editor["controls"]
                             if control["action"] == "increment"]
                self.assertEqual([(control["unit_id"], control["point"],
                                   control["enabled"]) for control in additions],
                                 [("archer", [355, 625], True)])
                self.assertEqual(editor["unit_availability"]["archer"],
                                 {"available": True, "reason": "active_picker_card"})
                self.assertTrue(additions[0]["evidence"]["identity"]["version_verified"])

                wrong_version = recognize_army_editor(path, provider, texts,
                                                       client_version="18.647.3")
                self.assertFalse(wrong_version["complete_kinds"]["troop"])
                self.assertFalse(any(control["action"] == "increment"
                                     for control in wrong_version["controls"]))

        # A count-only frame never establishes a complete current group.
        frame = metadata["frames"][0]
        path = fixture_root / frame["fixture"]
        texts = [OCRText(**item) for item in frame["ocr"]]
        incomplete = recognize_army_editor(path, None, texts, client_version="18.647.2")
        self.assertFalse(incomplete["complete_kinds"]["troop"])

        # A neighbouring gray portrait cannot substitute for the archer card.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image[590:650, 300:375] = 0
        cv2.imencode(".png", image)[1].tofile(self.path)
        masked = recognize_army_editor(self.path, provider, texts, client_version="18.647.2")
        self.assertFalse(any(control["action"] == "increment"
                             for control in masked["controls"]))

    def test_current_archer_picker_replay_requires_five_named_counted_cards(self):
        fixture_root = Path(__file__).parent / "fixtures"
        metadata = json.loads((fixture_root / "army_archer_18_647_2.json")
                              .read_text(encoding="utf-8"))
        frame = metadata["frames"][0]
        path = fixture_root / frame["fixture"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         frame["fixture_sha256"])
        texts = [OCRText(**item) for item in frame["ocr"]]
        provider = RapidOCRProvider()
        editor = recognize_army_editor(path, provider, texts, client_version="18.647.2")
        expected = [("electro_dragon", 10), ("dragon", 1), ("baby_dragon", 1),
                    ("balloon", 2), ("archer", 1)]
        self.assertEqual(editor["surface"], "current_picker")
        self.assertEqual(editor["capacities"]["troop"], {"used": 341, "total": 343})
        self.assertTrue(editor["complete_kinds"]["troop"], editor["unknowns"])
        self.assertEqual([(card["unit_id"], card["count"]) for card in editor["cards"]],
                         expected)
        self.assertEqual([(control["unit_id"], control["enabled"])
                          for control in editor["controls"] if control["action"] == "increment"],
                         [("archer", True)])

        unsupported = recognize_army_editor(path, provider, texts, client_version="18.647.3")
        self.assertFalse(unsupported["complete_kinds"]["troop"])
        self.assertFalse(any(control["action"] == "increment"
                             for control in unsupported["controls"]))
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        missing_portrait = image.copy()
        missing_portrait[226:268, 991:1036] = 0
        cv2.imencode(".png", missing_portrait)[1].tofile(self.path)
        unknown = recognize_army_editor(self.path, provider, texts, client_version="18.647.2")
        self.assertFalse(unknown["complete_kinds"]["troop"])
        self.assertIsNone(unknown["cards"][-1]["unit_id"])
        missing_count = image.copy()
        missing_count[194:226, 968:1005] = 0
        cv2.imencode(".png", missing_count)[1].tofile(self.path)
        without_archer_count = [item for item in texts if not (
            item.text == "x1" and item.bbox and item.bbox[0] >= 968 and item.bbox[1] < 230)]
        unknown = recognize_army_editor(self.path, provider, without_archer_count,
                                        client_version="18.647.2")
        self.assertFalse(unknown["complete_kinds"]["troop"])
        self.assertIsNone(unknown["cards"][-1]["count"])

    def test_current_archer_army_replay_needs_quantity_and_portrait(self):
        fixture_root = Path(__file__).parent / "fixtures"
        metadata = json.loads((fixture_root / "army_archer_18_647_2.json")
                              .read_text(encoding="utf-8"))
        frame = metadata["frames"][1]
        path = fixture_root / frame["fixture"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         frame["fixture_sha256"])
        texts = [OCRText(**item) for item in frame["ocr"]]
        provider = RapidOCRProvider()
        capacities = {"troops": {"used": 341, "capacity": 343}}

        def read(target, items, version="18.647.2"):
            return recognize_army_manifest(target, provider, items, groups=("troops",),
                                           capacities=capacities, client_version=version)

        army = read(path, texts)
        self.assertTrue(army["complete"], army["unknowns"])
        self.assertTrue(army["identity_complete"])
        self.assertEqual([(card["unit_id"], card["count"]) for card in army["troops"]],
                         [("electro_dragon", 10), ("dragon", 1), ("baby_dragon", 1),
                          ("balloon", 2), ("archer", 1)])
        self.assertEqual(army["troops"][-1]["evidence"]["identity"]["version_verified"],
                         True)
        self.assertFalse(read(path, texts, "18.647.3")["identity_complete"])

        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        missing_portrait = image.copy()
        missing_portrait[452:536, 1982:2072] = 0
        cv2.imencode(".png", missing_portrait)[1].tofile(self.path)
        unknown = read(self.path, texts)
        self.assertFalse(unknown["identity_complete"])
        self.assertIsNone(unknown["troops"][-1]["unit_id"])

        missing_count = image.copy()
        missing_count[388:452, 1936:2010] = 0
        cv2.imencode(".png", missing_count)[1].tofile(self.path)
        without_archer_count = [item for item in texts if not (
            item.text == "x1" and item.bbox and item.bbox[0] >= 968 and item.bbox[1] < 230)]
        unknown = read(self.path, without_archer_count)
        self.assertFalse(unknown["complete"])
        self.assertIsNone(unknown["troops"][-1]["count"])

    def test_current_archer_gray_picker_cannot_offer_enabled_increment(self):
        fixture_root = Path(__file__).parent / "fixtures"
        metadata = json.loads((fixture_root / "troop_picker_archer_gray_18_647_2.json")
                              .read_text(encoding="utf-8"))
        path = fixture_root / metadata["fixture"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         metadata["fixture_sha256"])
        source_path = Path(r"E:\projects\AutoCOC") / metadata["source_frame"]
        if source_path.is_file():
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(),
                             metadata["source_sha256"])
            source = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            source = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
            fixture = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            for left, top, right, bottom in metadata["retained_rois_1280"]:
                self.assertTrue(np.array_equal(fixture[top:bottom, left:right],
                                               source[top:bottom, left:right]))
        texts = [OCRText(**item) for item in metadata["ocr"]]
        provider = RapidOCRProvider()
        editor = recognize_army_editor(path, provider, texts, client_version="18.647.2")
        self.assertEqual(editor["surface"], "current_picker")
        self.assertEqual(editor["capacities"]["troop"], {"used": 343, "total": 343})
        self.assertTrue(editor["complete_kinds"]["troop"], editor["unknowns"])
        self.assertEqual([(card["unit_id"], card["count"]) for card in editor["cards"]],
                         [("electro_dragon", 10), ("dragon", 1), ("baby_dragon", 1),
                          ("balloon", 2), ("archer", 3)])
        increments = [control for control in editor["controls"]
                      if control["action"] == "increment"]
        self.assertTrue(increments)
        self.assertFalse(any(control["enabled"] for control in increments))
        self.assertTrue(all(control["capacity_blocked"] for control in increments))
        archer = next(control for control in increments if control["unit_id"] == "archer")
        self.assertEqual(archer["evidence"]["state"], "capacity_gray")
        self.assertTrue(archer["evidence"]["identity"]["version_verified"])
        self.assertEqual(archer["point"], [355, 625])

        wrong_version = recognize_army_editor(path, provider, texts,
                                              client_version="18.647.3")
        self.assertFalse(wrong_version["complete_kinds"]["troop"])
        self.assertFalse(any(control["action"] == "increment"
                             for control in wrong_version["controls"]))
        contradictory = [OCRText("342/343", item.confidence, item.bbox)
                         if item.text == "343/343" else item for item in texts]
        mismatch = recognize_army_editor(path, provider, contradictory,
                                         client_version="18.647.2")
        self.assertFalse(mismatch["complete_kinds"]["troop"])
        self.assertFalse(any(control["action"] == "increment"
                             for control in mismatch["controls"]))
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image[595:637, 313:360] = 0
        cv2.imencode(".png", image)[1].tofile(self.path)
        missing = recognize_army_editor(self.path, provider, texts,
                                        client_version="18.647.2")
        self.assertFalse(any(control["action"] == "increment"
                             and control["unit_id"] == "archer"
                             for control in missing["controls"]))

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
