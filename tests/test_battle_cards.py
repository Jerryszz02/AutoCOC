from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from autococ.ocr import OCRText
from autococ.vision import ScreenshotRecognizer


class BattleCardTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def fixture(self, directory: str, number: int) -> tuple[Path, list[OCRText], str]:
        folder = self.root / directory
        if not folder.is_dir():
            self.skipTest("Optional live battle-card fixture is absent")
        frame = next((folder / "frames").glob(f"{number:05d}-*.png"))
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed = next(event for event in events if event.get("kind") == "observation"
                        and Path(event["frame"]).name == frame.name)
        texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
                 for item in observed["observations"]["ocr"] if item.get("bbox")]
        return frame, texts, observed["scene"]

    def observe(self, fixture: tuple[Path, list[OCRText], str]) -> list[dict]:
        return ScreenshotRecognizer(provider=Mock())._battle_observation(*fixture)["slots"]

    def assert_independent(self, slots: list[dict]) -> None:
        for first, second in zip(slots, slots[1:]):
            self.assertLessEqual(first["bbox"][2], second["bbox"][0])
        for slot in slots:
            left, top, right, bottom = slot["bbox"]
            self.assertGreaterEqual(right - left, 80)
            self.assertGreaterEqual(bottom - top, 105)

    def test_live_scout_recovers_leftmost_meteor_card_at_screen_edge(self) -> None:
        # The real 18.600.7 scout frame's first card has only .724 cyan side
        # support where it meets the map. Its own x8 label, edge gradients and
        # bottom border must locate it without inferring a missing army slot.
        path = self.root / "tests/fixtures/scout_battle_bar_20260926.png"
        texts = [OCRText(value, .98, bbox) for value, bbox in (
            ("x8", (61, 592, 104, 622)), ("x2", (160, 592, 200, 621)),
            ("x1", (264, 594, 295, 621)), ("x1", (358, 592, 392, 622)),
        )]
        slots = self.observe((path, texts, "enemy_village"))
        self.assert_independent(slots)
        self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop"], [8, 2, 1, 1])
        self.assertEqual(slots[0]["bbox"], [16, 595, 104, 711])

    def test_zoomed_battle_bar_refines_header_overlap_without_losing_neighbors(self) -> None:
        path = self.root / "tests/fixtures/zoomed_battle_bar_20260926.png"
        texts = [OCRText(value, .99, bbox) for value, bbox in (
            ("x8", (61, 592, 104, 622)), ("x2", (161, 592, 200, 621)),
            ("x1", (264, 594, 295, 621)), ("x1", (360, 594, 392, 621)),
        )]
        vision = ScreenshotRecognizer(provider=Mock())
        vision.client_version = "18.600.7"
        slots = vision._battle_observation(path, texts, "enemy_village")["slots"]
        self.assert_independent(slots)
        troops = [slot for slot in slots if slot["kind"] == "troop"]
        self.assertEqual([(slot["count"], slot["unit_id"]) for slot in troops],
                         [(8, "meteor_golem"), (2, "bowler"), (1, "wall_breaker"), (1, "archer")])

    def test_swiped_real_bar_recovers_all_four_spell_cards_without_guessing_low_ocr(self) -> None:
        expected = ["totem_spell", "overgrowth_spell", "revival_spell", "freeze_spell"]
        for label, revival_confidence, revival_count in (("first", .99878, 1),
                                                          ("middle", .82878, None),
                                                          ("last", .99878, 1)):
            with self.subTest(frame=label):
                path = self.root / f"tests/fixtures/swiped_battle_bar_{label}_20260926.png"
                texts = [OCRText(value, confidence, bbox) for value, confidence, bbox in (
                    ("x3", .9988, (930, 592, 970, 622)),
                    ("x2", .99802, (1027, 592, 1066, 622)),
                    ("x1", revival_confidence, (1128, 593, 1164, 622)),
                    ("x2", .9984, (1221, 592, 1260, 622)),
                )]
                vision = ScreenshotRecognizer(provider=Mock())
                vision.client_version = "18.600.7"
                slots = vision._battle_observation(path, texts, "enemy_village")["slots"]
                spells = [slot for slot in slots if slot["kind"] == "spell"]
                self.assertEqual([slot["unit_id"] for slot in spells], expected)
                self.assertEqual([slot["count"] for slot in spells], [3, 2, revival_count, 2])
                self.assert_independent(slots)
                vision.client_version = "18.600.8"
                mismatched = vision._battle_observation(path, texts, "enemy_village")["slots"]
                self.assertFalse(any(slot["unit_id"] in expected for slot in mismatched))

    def test_background_bridges_do_not_duplicate_siege_or_split_last_spell(self) -> None:
        fixtures = (("reports/live-20260922/terrain-stop-check-015411", 1),
                    ("reports/20260923-015217-869637-e2129181", 13))
        for directory, number in fixtures:
            with self.subTest(directory=directory):
                slots = self.observe(self.fixture(directory, number))
                self.assertEqual(Counter(slot["kind"] for slot in slots),
                                 Counter(troop=4, siege=1, hero=4, spell=2))
                self.assert_independent(slots)
                siege = next(slot for slot in slots if slot["kind"] == "siege")
                self.assertEqual(siege["bbox"], [498, 595, 587, 711])
                self.assertEqual(siege["point"], [542, 653])
                last = slots[-1]
                self.assertEqual(last["bbox"], [1094, 595, 1183, 711])
                self.assertEqual(last["count"], 5)
                self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop"], [10, 1, 2, 1])

    def test_normal_selected_and_gray_frames_keep_explicit_card_counts(self) -> None:
        for number, troop_counts in ((11, [10, 1, 2, 1]), (16, [1, 1, 2, 1]), (19, [1, 2, 1])):
            with self.subTest(frame=number):
                slots = self.observe(self.fixture("reports/20260923-012829-323923-b0c8cf92", number))
                self.assert_independent(slots)
                self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop"], troop_counts)
                self.assertEqual(Counter(slot["kind"] for slot in slots),
                                 Counter(troop=len(troop_counts), siege=1, hero=4, spell=2))
                self.assertFalse(any(slot["count"] == 0 for slot in slots), "A missing gray contour is not an explicit zero")

    def test_received_clan_spell_is_distinct_from_own_recipe(self) -> None:
        for folder, expected in (("reports/live-20260922/reconnected-precheck-230909", 3),
                                 ("reports/live-20260922/current-result-observation-231122", 1)):
            with self.subTest(folder=folder):
                slots = self.observe(self.fixture(folder, 1))
                clan = [slot for slot in slots if slot["source"] == "clan_reinforcement"]
                self.assertEqual(len(clan), 1)
                self.assertEqual(clan[0]["kind"], "spell")
                self.assertEqual(clan[0]["count"], expected)
                self.assertGreaterEqual(clan[0]["evidence"]["clan_badge"]["confidence"], .95)
        own = self.observe(self.fixture("reports/20260923-021847-763016-3632dc81", 9))
        self.assertTrue(all(slot["source"] == "army" for slot in own))

    def test_hero_icon_anchors_recover_complete_borders_in_scout_background(self) -> None:
        slots = self.observe(self.fixture("reports/20260923-015217-869637-e2129181", 10))
        self.assert_independent(slots)
        heroes = [slot for slot in slots if slot["kind"] == "hero"]
        self.assertEqual(len(heroes), 4)
        for hero in heroes:
            self.assertIsNone(hero["count"])
            self.assertGreaterEqual(hero["evidence"]["hero_pet_icon"]["confidence"], .9)

    def test_hero_icon_without_card_borders_does_not_create_a_slot(self) -> None:
        template = cv2.imdecode(np.fromfile(self.root / "assets/templates/battle_hero_pet_0.png", dtype=np.uint8), cv2.IMREAD_COLOR)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "icon.png"
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            height, width = template.shape[:2]
            image[600:600 + height, 604:604 + width] = template
            cv2.imencode(".png", image)[1].tofile(path)
            self.assertEqual(self.observe((path, [], "battle")), [])

    def test_intermediate_selected_hero_scale_keeps_its_identity(self) -> None:
        slots = self.observe(self.fixture("reports/20260924-003409-344711-0e4a3e7a", 24))
        heroes = [slot for slot in slots if slot["kind"] == "hero"]
        self.assertEqual(len(heroes), 4)
        self.assert_independent(slots)
        warden = heroes[0]
        self.assertEqual(warden["bbox"], [500, 590, 593, 713])
        self.assertEqual(warden["evidence"]["hero_pet_icon"]["scale"], 1.05)
        self.assertGreaterEqual(warden["evidence"]["hero_pet_icon"]["confidence"], .98)

    def test_smaller_selected_queen_still_requires_its_complete_white_border(self) -> None:
        fixture = self.fixture("reports/20260924-004347-675110-2e2b80bd", 37)
        slots = self.observe(fixture)
        queen = [slot for slot in slots if slot["kind"] == "hero" and 790 <= slot["bbox"][0] <= 800]
        self.assertEqual(len(queen), 1)
        self.assertEqual(queen[0]["bbox"][:3], [793, 595, 880])
        self.assert_independent(slots)
        image = cv2.resize(cv2.imread(str(fixture[0])), (1280, 720), interpolation=cv2.INTER_AREA)
        image[620:700, 790:799] = (25, 25, 25)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing-side.png"
            cv2.imwrite(str(path), image)
            broken = self.observe((path, fixture[1], fixture[2]))
            self.assertFalse(any(slot["kind"] == "hero" and 790 <= slot["bbox"][0] <= 800 for slot in broken))

    def test_blue_background_preserves_every_numeric_troop_and_spell_card(self) -> None:
        slots = self.observe(self.fixture("reports/20260923-021847-763016-3632dc81", 9))
        self.assert_independent(slots)
        self.assertEqual(Counter(slot["kind"] for slot in slots), Counter(troop=4, siege=1, hero=4, spell=2))
        self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop"], [10, 1, 2, 1])
        self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "spell"], [6, 5])
        self.assertEqual([slot["bbox"][0] for slot in slots if slot["kind"] in {"troop", "spell"}],
                         [93, 190, 287, 383, 998, 1094])

    def test_selected_enlarged_hero_keeps_identity_and_other_numeric_cards(self) -> None:
        before = self.observe(self.fixture("reports/20260923-021847-763016-3632dc81", 14))
        selected = self.observe(self.fixture("reports/20260923-021847-763016-3632dc81", 15))
        for slots in (before, selected):
            self.assert_independent(slots)
            self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop"], [10, 1, 2])
            self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "spell"], [6, 5])
            self.assertEqual(sum(slot["kind"] == "hero" for slot in slots), 4)
        hero = next(slot for slot in selected if slot["kind"] == "hero")
        original = next(slot for slot in before if slot["kind"] == "hero")
        self.assertEqual(hero["bbox"], [594, 586, 692, 715])
        self.assertEqual(hero["evidence"]["hero_pet_icon"]["scale"], 1.1)
        self.assertGreater(hero["evidence"]["hero_pet_icon"]["confidence"], .96)
        self.assertTrue(all(abs(a - b) < 25 for a, b in zip(original["point"], hero["point"])))
        self.assertIsNone(hero["count"])
        self.assertNotIn("deployed", hero)
        self.assertNotIn("ability_ready", hero)

    def test_selected_wall_breaker_thin_white_border_keeps_its_numeric_card(self) -> None:
        slots = self.observe(self.fixture("reports/20260923-235238-479708-a0aa6220", 17))
        self.assert_independent(slots)
        selected = next(slot for slot in slots if 180 <= slot["bbox"][0] <= 195)
        self.assertEqual(selected["kind"], "troop")
        self.assertEqual(selected["count"], 1)
        self.assertEqual(selected["bbox"], [185, 587, 283, 715])
        self.assertEqual([slot["count"] for slot in slots if slot["kind"] == "troop"], [1, 1])

    def test_broken_white_side_does_not_supply_a_selected_card(self) -> None:
        frame, texts, scene = self.fixture("reports/20260923-235238-479708-a0aa6220", 17)
        image = cv2.resize(cv2.imdecode(np.fromfile(frame, dtype=np.uint8), cv2.IMREAD_COLOR),
                           (1280, 720), interpolation=cv2.INTER_AREA)
        image[638:669, 278:285] = 0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken-side.png"
            cv2.imencode(".png", image)[1].tofile(path)
            slots = self.observe((path, texts, scene))
        self.assertFalse(any(180 <= slot["bbox"][0] <= 195 for slot in slots))

    def test_cyan_siege_header_recovers_card_connected_to_hero_background(self) -> None:
        for number in (10, 16, 17):
            with self.subTest(frame=number):
                slots = self.observe(self.fixture("reports/20260923-235238-479708-a0aa6220", number))
                self.assert_independent(slots)
                siege = [slot for slot in slots if slot["kind"] == "siege"]
                self.assertEqual(len(siege), 1)
                self.assertEqual(siege[0]["bbox"], [401, 595, 490, 711])
                self.assertEqual(sum(slot["kind"] == "hero" for slot in slots), 4)

    def test_cyan_header_without_long_side_borders_cannot_locate_siege(self) -> None:
        frame, texts, scene = self.fixture("reports/20260923-235238-479708-a0aa6220", 17)
        image = cv2.resize(cv2.imdecode(np.fromfile(frame, dtype=np.uint8), cv2.IMREAD_COLOR),
                           (1280, 720), interpolation=cv2.INTER_AREA)
        image[625:695, 398:408] = 0
        image[625:695, 484:494] = 0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "header-only.png"
            cv2.imencode(".png", image)[1].tofile(path)
            slots = self.observe((path, texts, scene))
        self.assertFalse(any(slot["kind"] == "siege" for slot in slots))

    def test_blue_background_recovery_does_not_trust_low_or_conflicting_counts(self) -> None:
        frame, texts, scene = self.fixture("reports/20260923-021847-763016-3632dc81", 9)
        original = next(item for item in texts if item.text == "x10")
        low = [replace(item, confidence=.89) if item is original else item for item in texts]
        self.assertFalse(any(slot["bbox"][0] == 93 for slot in self.observe((frame, low, scene))))
        conflict = self.observe((frame, texts + [replace(original, text="x0")], scene))
        self.assertIsNone(next(slot for slot in conflict if slot["bbox"][0] == 93)["count"])

    def test_conflicting_or_unreliable_quantity_stays_unknown(self) -> None:
        frame, texts, scene = self.fixture("reports/live-20260922/terrain-stop-check-015411", 1)
        original = next(item for item in texts if item.text == "x10")
        variants = [texts + [replace(original, text="x0")]]
        for confidence in (.89, float("nan"), float("inf")):
            variants.append([replace(item, confidence=confidence) if item is original else item for item in texts])
        for modified in variants:
            with self.subTest(readings=modified):
                slots = self.observe((frame, modified, scene))
                self.assertEqual(slots[0]["kind"], "troop")
                self.assertIsNone(slots[0]["count"])

    def test_a_wide_colored_region_does_not_imply_multiple_cards(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wide.png"
            hsv = np.zeros((720, 1280, 3), dtype=np.uint8)
            hsv[595:711, 180:429] = [105, 200, 180]
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imencode(".png", image)[1].tofile(path)
            self.assertEqual(self.observe((path, [], "battle")), [])
            self.assertEqual(self.observe((path, [OCRText("x10", .99, (380, 596, 427, 620))], "battle")), [])


if __name__ == "__main__":
    unittest.main()
