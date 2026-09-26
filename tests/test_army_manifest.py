from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.army_manifest import recognize_army_heroes, recognize_army_manifest, recognize_army_siege
from autococ.army_editor import recognize_hero_loadout, visual_fingerprint_matches
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

    def inspect(self, image, texts, *, capacities=None):
        cv2.imencode(".png", image)[1].tofile(self.path)
        return recognize_army_manifest(self.path, self.provider, texts, capacities=capacities)

    def test_pure_spell_row_requires_independent_zero_capacity_header(self) -> None:
        image, texts = self.make(troops=(), spells=(2,))
        without_header = self.inspect(image, texts)
        self.assertFalse(without_header["complete"])
        self.assertIn("no_complete_card_borders",
                      [item["reason"] for item in without_header["unknowns"]])
        capacities = {"troops": {"used": 0, "capacity": 335}}
        empty = self.inspect(image, texts, capacities=capacities)
        self.assertTrue(empty["complete"], empty["unknowns"])
        self.assertEqual(empty["troops"], [])
        self.assertEqual(empty["spells"][0]["count"], 2)
        self.assertEqual(empty["layout_evidence"]["rows"]["troops"]["zero_capacity_header"],
                         capacities["troops"])
        conflict, conflict_texts = self.make(troops=(1,), spells=(2,))
        observed = self.inspect(conflict, conflict_texts, capacities=capacities)
        self.assertFalse(observed["complete"])
        self.assertIn("visible_cards_conflict_with_zero_capacity",
                      [item["reason"] for item in observed["unknowns"]])

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

    def test_live_spell_contrast_recovers_all_four_counts_without_capacity_guess(self) -> None:
        path = Path(__file__).parent / "fixtures/army_confirmation_counts_20260926.png"
        # Only the panel is retained in this fixture; the account/header and
        # village behind it were removed. Full OCR missed x3 and had low
        # confidence on the final x2 in the real 18.600.7 confirmation frame.
        texts = list(ANCHORS) + [
            OCRText("x8", .99, (560, 196, 590, 220)),
            OCRText("x2", .99, (665, 196, 695, 220)),
            OCRText("x1", .99, (767, 196, 795, 220)),
            OCRText("x1", .99, (870, 196, 898, 220)),
            OCRText("x2", .99, (664, 376, 692, 398)),
            OCRText("x1", .99, (768, 378, 790, 398)),
            OCRText("x2", .87, (868, 374, 898, 399)),
        ]
        result = recognize_army_manifest(path, RapidOCRProvider(), texts)
        self.assertTrue(result["complete"], result["unknowns"])
        self.assertEqual([x["count"] for x in result["troops"]], [8, 2, 1, 1])
        self.assertEqual([x["count"] for x in result["spells"]], [3, 2, 1, 2])
        self.assertEqual(result["spells"][0]["evidence"]["count"]["source"], "contrast_line_roi")
        self.assertEqual(result["spells"][3]["evidence"]["count"]["source"], "contrast_line_roi")

    def test_real_large_hero_cards_bind_stable_ids_to_loadout_without_column_identity(self) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        observed = []
        for filename in ("army_heroes_first_frame_20260926.png",
                         "army_heroes_second_frame_20260926.png"):
            path = fixtures / filename
            heroes = recognize_army_heroes(path, {"used": 4, "capacity": 4},
                                           client_version="18.600.7")
            self.assertTrue(heroes["complete"], heroes["unknowns"])
            self.assertEqual([card["unit_id"] for card in heroes["cards"]],
                             ["grand_warden", "dragon_duke", "minion_prince", "archer_queen"])
            texts = [OCRText("我的军队", .99, (222, 63, 294, 90)),
                     OCRText("4/4", .99, (50, 150, 98, 176))]
            loadout = recognize_hero_loadout(path, texts, heroes["cards"])
            self.assertTrue(loadout["complete"], loadout["unknowns"])
            observed.append(loadout["hero_loadout"])
        for unit_id in observed[0]:
            for icon in ("pet_visual", "equipment_1_visual", "equipment_2_visual"):
                self.assertTrue(visual_fingerprint_matches(
                    observed[0][unit_id][icon], observed[1][unit_id][icon]))

    def test_swapped_hero_cards_change_ids_with_portraits_and_missing_edge_blocks_completion(self) -> None:
        source = Path(__file__).parent / "fixtures/army_heroes_first_frame_20260926.png"
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        first, last = image[185:668, 23:148].copy(), image[185:668, 413:538].copy()
        image[185:668, 23:148], image[185:668, 413:538] = last, first
        with TemporaryDirectory() as directory:
            path = Path(directory) / "swapped.png"
            cv2.imencode(".png", image)[1].tofile(path)
            swapped = recognize_army_heroes(path, {"used": 4, "capacity": 4},
                                            client_version="18.600.7")
            self.assertTrue(swapped["complete"], swapped["unknowns"])
            self.assertEqual([card["unit_id"] for card in swapped["cards"]],
                             ["archer_queen", "dragon_duke", "minion_prince", "grand_warden"])
            image[185:668, 18:149] = 0
            cv2.imencode(".png", image)[1].tofile(path)
            broken = recognize_army_heroes(path, {"used": 4, "capacity": 4},
                                           client_version="18.600.7")
            self.assertFalse(broken["complete"])
            self.assertEqual(len(broken["cards"]), 3)
        mismatched = recognize_army_heroes(source, {"used": 4, "capacity": 4},
                                           client_version="18.600.8")
        self.assertFalse(mismatched["complete"])
        self.assertTrue(all(card["unit_id"] is None for card in mismatched["cards"]))

    def test_real_siege_cards_verify_quantity_without_naming_unknown_machines(self) -> None:
        for name in ("army_confirmation_counts_20260926.png",
                     "army_current_second_frame_20260926.png"):
            with self.subTest(frame=name):
                path = Path(__file__).parent / "fixtures" / name
                result = recognize_army_siege(path, RapidOCRProvider(),
                                              {"used": 3, "capacity": 3},
                                              client_version="18.600.7")
                self.assertEqual([card["count"] for card in result["cards"]], [1, 1, 1])
                self.assertEqual([card["unit_id"] for card in result["cards"]],
                                 ["siege_barracks", None, None])
                self.assertTrue(all(card["level"] is None and card["available"] is None
                                    for card in result["cards"]))
                self.assertFalse(result["complete"])
                self.assertEqual(result["unknowns"].count("siege_identity_unavailable"), 2)
                self.assertTrue(all(card["evidence"]["count_reads"] for card in result["cards"]))

    def test_siege_identity_requires_observed_capacity_quantity_and_client_version(self) -> None:
        path = Path(__file__).parent / "fixtures/army_confirmation_counts_20260926.png"
        for capacity, version, reason in (
            (None, "18.600.7", "siege_capacity_unreadable"),
            ({"used": 0, "capacity": 3}, "18.600.7", "empty_siege_row_not_calibrated"),
            ({"used": 2, "capacity": 3}, "18.600.7", "siege_card_count_disagrees_with_capacity"),
            ({"used": 3, "capacity": 3}, "18.600.8", "siege_client_version_unverified"),
        ):
            with self.subTest(capacity=capacity, version=version):
                result = recognize_army_siege(path, RapidOCRProvider(), capacity,
                                              client_version=version)
                self.assertFalse(result["complete"])
                self.assertIn(reason, result["unknowns"])
        unreadable = recognize_army_siege(path, Mock(), {"used": 3, "capacity": 3},
                                          client_version="18.600.7")
        self.assertFalse(unreadable["complete"])
        self.assertTrue(all(card["count"] is None for card in unreadable["cards"]))
