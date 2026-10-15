import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.unit_catalog import ArmyRecipe, ArmyRequirement, coverage, recognize_card_identity, recognize_large_hero_identity, template_manifest
from autococ.army_manifest import recognize_army_heroes
from autococ.locator import scale_box
from autococ.ocr import OCRText, RapidOCRProvider
from autococ.scene import SCENE_BATTLE, SCENE_ENEMY_VILLAGE
from autococ.vision import ScreenshotRecognizer


class UnitCatalogTests(unittest.TestCase):
    def test_recipe_rejects_unknown_and_duplicate_units(self):
        with self.assertRaises(ValueError):
            ArmyRequirement("imaginary_dragon", 1)
        with self.assertRaises(ValueError):
            ArmyRecipe((ArmyRequirement("lightning_spell", 2),
                        ArmyRequirement("lightning_spell", 3)))

    def test_template_identity_is_independent_of_card_position(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(4)
            portrait = rng.integers(0, 255, (36, 36, 3), dtype=np.uint8)
            cv2.imencode(".png", portrait)[1].tofile(root / "lightning.png")
            (root / "unit_templates.json").write_text(json.dumps({"client": "18.600.7", "templates": [
                {"unit_id": "lightning_spell", "surface": "battle", "path": "lightning.png"}]}))
            image = np.zeros((720, 1280, 3), np.uint8)
            for left in (200, 500):
                image[625:661, left + 20:left + 56] = portrait
                observation = recognize_card_identity(image, (left, 595, left + 90, 710),
                                                       "spell", root=root, threshold=.99,
                                                       client_version="18.600.7")
                self.assertEqual(observation["unit_id"], "lightning_spell")
                no_version = recognize_card_identity(image, (left, 595, left + 90, 710),
                                                     "spell", root=root)
                self.assertIsNone(no_version["unit_id"])
                self.assertFalse(no_version["version_verified"])
            self.assertEqual(coverage(root)["lightning_spell"]["recognition"], "sampled")
            self.assertEqual(coverage(root)["freeze_spell"]["recognition"], "unavailable")

    def test_missing_and_ambiguous_samples_remain_unknown(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(7)
            portrait = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imencode(".png", portrait)[1].tofile(root / "same.png")
            image[630:662, 220:252] = portrait
            manifest = {"client": "18.600.7", "templates": [
                {"unit_id": "lightning_spell", "surface": "battle", "path": "same.png"},
                {"unit_id": "freeze_spell", "surface": "battle", "path": "same.png"}]}
            (root / "unit_templates.json").write_text(json.dumps(manifest))
            observed = recognize_card_identity(image, (200, 595, 290, 710), "spell", root=root,
                                               client_version="18.600.7")
            self.assertIsNone(observed["unit_id"])
            self.assertEqual(observed["reason"], "ambiguous_identity")

    def test_second_real_army_frame_matches_versioned_cards(self):
        path = Path(__file__).parent / "fixtures/army_current_second_frame_20260926.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        rows = (("troop", 194, ("meteor_golem", "bowler", "wall_breaker", "archer")),
                ("spell", 373, ("totem_spell", "overgrowth_spell", "revival_spell", "freeze_spell")))
        for kind, top, expected in rows:
            for index, unit_id in enumerate(expected):
                with self.subTest(kind=kind, unit_id=unit_id):
                    left = 557 + index * 103
                    observed = recognize_card_identity(
                        image, (left, top, left + 98, top + 97), kind,
                        surface="army", client_version="18.600.7")
                    self.assertEqual(observed["unit_id"], unit_id)
                    self.assertTrue(observed["version_verified"])
                    self.assertGreaterEqual(observed["confidence"], .99)
                    wrong_version = recognize_card_identity(
                        image, (left, top, left + 98, top + 97), kind,
                        surface="army", client_version="18.600.8")
                    self.assertIsNone(wrong_version["unit_id"])
                    self.assertEqual(wrong_version["reason"], "client_version_unverified")

    def test_new_army_samples_are_versioned_independently(self):
        path = Path(__file__).parent / "fixtures/army_current_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        self.assertEqual(template_manifest(client_version="18.647.2")["client"], "18.647.2")
        self.assertEqual(template_manifest()["client"], "18.600.7")
        for kind, top, left, unit_id in (
            ("troop", 194, 557, "electro_dragon"),
            ("troop", 194, 660, "dragon"),
            ("troop", 194, 762, "baby_dragon"),
            ("troop", 194, 865, "balloon"),
            ("spell", 373, 558, "lightning_spell"),
            ("spell", 373, 661, "earthquake_spell"),
        ):
            with self.subTest(unit_id=unit_id):
                box = (left, top, left + 98, top + 97)
                observed = recognize_card_identity(image, box, kind, surface="army",
                                                   client_version="18.647.2")
                self.assertEqual(observed["unit_id"], unit_id)
                self.assertTrue(observed["version_verified"])
                unsupported = recognize_card_identity(image, box, kind, surface="army",
                                                      client_version="18.647.3")
                self.assertIsNone(unsupported["unit_id"])
                self.assertEqual(unsupported["reason"], "client_version_unverified")
                battle_surface = recognize_card_identity(image, box, kind, surface="battle",
                                                         client_version="18.647.2")
                self.assertIsNone(battle_surface["unit_id"])
                self.assertEqual(battle_surface["reason"], "below_identity_threshold")
                self.assertLess(battle_surface["confidence"], .94)
        self.assertEqual(coverage(client_version="18.647.2")["dragon"]["recognition"], "sampled")
        self.assertEqual(coverage()["dragon"]["recognition"], "unavailable")

    def test_new_dragon_duke_sample_matches_independent_selected_view(self):
        path = Path(__file__).parent / "fixtures/hero_dragon_duke_selected_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_large_hero_identity(image, (153, 185, 277, 668),
                                               client_version="18.647.2")
        self.assertEqual(result["unit_id"], "dragon_duke")
        self.assertGreaterEqual(result["evidence"][0]["face_score"], .85)
        self.assertGreaterEqual(result["evidence"][0]["color_score"], .8)
        wrong_version = recognize_large_hero_identity(image, (153, 185, 277, 668),
                                                       client_version="18.600.8")
        self.assertIsNone(wrong_version["unit_id"])

    def test_new_grand_warden_sample_matches_independent_selected_view(self):
        path = Path(__file__).parent / "fixtures/hero_grand_warden_selected_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_large_hero_identity(image, (23, 185, 147, 668),
                                               client_version="18.647.2")
        self.assertEqual(result["unit_id"], "grand_warden")
        self.assertGreaterEqual(result["evidence"][0]["face_score"], .85)
        self.assertGreaterEqual(result["evidence"][0]["color_score"], .8)
        self.assertTrue(all(candidate["face_score"] < .85
                            for candidate in result["evidence"][1:]))
        wrong_version = recognize_large_hero_identity(image, (23, 185, 147, 668),
                                                       client_version="18.600.8")
        self.assertIsNone(wrong_version["unit_id"])

    def test_current_hero_roster_heldout_frames_keep_all_four_identities(self):
        expected = ("grand_warden", "dragon_duke", "minion_prince", "archer_queen")
        metadata = json.loads((Path(__file__).parent / "fixtures/hero_roster_holdout_18_647_2_20261015.json")
                              .read_text(encoding="utf-8"))
        for number in (7, 8):
            path = Path(__file__).parent / f"fixtures/hero_roster_holdout_{number:05d}_18_647_2_20261015.png"
            source = metadata["frames"][number - 7]
            self.assertEqual(path.name, source["fixture"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["fixture_sha256"])
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            source_path = Path(r"E:\projects\AutoCOC") / source["source_frame"]
            if source_path.is_file():
                self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), source["source_sha256"])
                original = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                original = cv2.resize(original, (1280, 720), interpolation=cv2.INTER_AREA)
                self.assertTrue(np.array_equal(image, original[185:668, 23:537]))
            for index, unit_id in enumerate(expected):
                with self.subTest(frame=number, unit_id=unit_id):
                    left = index * 130
                    result = recognize_large_hero_identity(
                        image, (left, 0, left + 124, 483), client_version="18.647.2")
                    self.assertEqual(result["unit_id"], unit_id, result)
                    self.assertGreaterEqual(result["confidence"], .9)
                    self.assertEqual(result["reason"], "face_and_upper_card_visual_profile")
                    self.assertIsNone(recognize_large_hero_identity(
                        image, (left, 0, left + 124, 483), client_version="18.647.3")["unit_id"])
            # Preserve the observed card pixels and geometry while masking the
            # account and village before testing the full hero-roster reader.
            anonymous = np.zeros((720, 1280, 3), dtype=np.uint8)
            anonymous[185:668, 23:537] = image
            with TemporaryDirectory() as folder:
                replay = Path(folder) / "anonymous-roster.png"
                cv2.imencode(".png", anonymous)[1].tofile(replay)
                roster = recognize_army_heroes(replay, {"used": 4, "capacity": 4},
                                               client_version="18.647.2")
            self.assertTrue(roster["complete"], roster["unknowns"])
            self.assertEqual([card["unit_id"] for card in roster["cards"]], list(expected))

        template = next(entry for entry in template_manifest(client_version="18.647.2")["templates"]
                        if entry.get("path") == "units/grand_warden_army_color_roster_18_647_2.png")
        template_path = Path(__file__).parents[1] / "assets/catalogs" / template["path"]
        self.assertEqual(hashlib.sha256(template_path.read_bytes()).hexdigest(), template["template_sha256"])
        source_path = Path(r"E:\projects\AutoCOC") / template["source_frame"]
        if source_path.is_file():
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), template["source_sha256"])
            original = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            original = cv2.resize(original, (1280, 720), interpolation=cv2.INTER_AREA)
            left, top, right, bottom = template["source_bbox_1280"]
            sampled = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(sampled, original[top:bottom, left:right]))

    def test_new_minion_prince_sample_matches_independent_selected_view(self):
        path = Path(__file__).parent / "fixtures/hero_minion_prince_selected_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_large_hero_identity(image, (283, 185, 407, 668),
                                               client_version="18.647.2")
        self.assertEqual(result["unit_id"], "minion_prince")
        self.assertGreaterEqual(result["evidence"][0]["face_score"], .85)
        self.assertGreaterEqual(result["evidence"][0]["color_score"], .8)

    def test_new_archer_queen_sample_matches_independent_selected_view(self):
        path = Path(__file__).parent / "fixtures/hero_archer_queen_selected_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_large_hero_identity(image, (413, 185, 537, 668),
                                               client_version="18.647.2")
        self.assertEqual(result["unit_id"], "archer_queen")
        self.assertGreaterEqual(result["evidence"][0]["face_score"], .85)
        self.assertGreaterEqual(result["evidence"][0]["color_score"], .8)

    def test_unknown_client_cannot_verify_empty_manifest(self):
        path = Path(__file__).parent / "fixtures/hero_archer_queen_selected_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        hero = recognize_large_hero_identity(image, (413, 185, 537, 668),
                                             client_version="unknown")
        self.assertIsNone(hero["unit_id"])
        self.assertFalse(hero["version_verified"])
        self.assertEqual(hero["reason"], "client_version_unverified")
        card = recognize_card_identity(image, (413, 185, 513, 305), "hero",
                                       client_version="unknown")
        self.assertIsNone(card["unit_id"])
        self.assertFalse(card["version_verified"])

    def test_current_siege_barracks_matches_independent_army_frame(self):
        path = Path(__file__).parent / "fixtures/siege_barracks_army_heldout_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_card_identity(image, (1016, 372, 1111, 470), "siege",
                                         surface="army", client_version="18.647.2")
        self.assertEqual(result["unit_id"], "siege_barracks")
        self.assertTrue(result["version_verified"])
        self.assertGreaterEqual(result["confidence"], .94)
        unsupported = recognize_card_identity(image, (1016, 372, 1111, 470), "siege",
                                              surface="battle", client_version="18.647.2")
        self.assertIsNone(unsupported["unit_id"])

    def test_current_sky_wagon_matches_independent_army_frame(self):
        path = Path(__file__).parent / "fixtures/sky_wagon_army_heldout_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_card_identity(image, (1083, 372, 1178, 470), "siege",
                                         surface="army", client_version="18.647.2")
        self.assertEqual(result["unit_id"], "sky_wagon")
        self.assertGreaterEqual(result["confidence"], .94)

    def test_current_troop_launcher_matches_independent_army_frame(self):
        path = Path(__file__).parent / "fixtures/troop_launcher_army_heldout_18_647_2_20261015.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = recognize_card_identity(image, (1150, 372, 1245, 470), "siege",
                                         surface="army", client_version="18.647.2")
        self.assertEqual(result["unit_id"], "troop_launcher")
        self.assertGreaterEqual(result["confidence"], .94)

    def test_current_battle_cards_match_independent_scout_bar(self):
        # Both fixtures are unmodified HUD crops. One is a later frame of the
        # source scout; the second is a separate camera/base scouting run.
        cards = [
            ("electro_dragon", "troop", (93, 595, 182, 711)),
            ("dragon", "troop", (190, 595, 278, 711)),
            ("baby_dragon", "troop", (287, 595, 375, 711)),
            ("balloon", "troop", (383, 595, 472, 711)),
            ("sky_wagon", "siege", (491, 595, 580, 711)),
            ("grand_warden", "hero", (599, 595, 687, 711)),
            ("dragon_duke", "hero", (696, 595, 784, 711)),
            ("minion_prince", "hero", (792, 595, 881, 711)),
            ("archer_queen", "hero", (889, 595, 978, 711)),
            ("lightning_spell", "spell", (998, 595, 1086, 711)),
            ("earthquake_spell", "spell", (1094, 595, 1183, 711)),
        ]
        for suffix in ("", "_2"):
            path = Path(__file__).parent / f"fixtures/battle_bar_heldout_18_647_2_20261015{suffix}.png"
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            for unit_id, kind, (left, top, right, bottom) in cards:
                box = (left - 85, top - 590, right - 85, bottom - 590)
                with self.subTest(frame=path.name, unit=unit_id):
                    result = recognize_card_identity(image, box, kind, surface="battle",
                                                     client_version="18.647.2")
                    self.assertEqual(result["unit_id"], unit_id)
                    self.assertGreaterEqual(result["confidence"], .98)
                    self.assertIsNone(recognize_card_identity(
                        image, box, kind, surface="battle", client_version="unknown")["unit_id"])

    def test_real_battle_bar_binds_remaining_troops_after_electro_deploys(self):
        provider = RapidOCRProvider()
        recognizer = ScreenshotRecognizer(provider=provider)
        recognizer.client_version = "18.600.7"
        fixtures = Path(__file__).parent / "fixtures"
        for filename, scene, expected in (
            ("daily_battle_bar_before_20260926.png", SCENE_ENEMY_VILLAGE,
             [("electro_dragon", 10), ("dragon_rider", 1), ("balloon", 2)]),
            ("daily_battle_bar_after_20260926.png", SCENE_BATTLE,
             [("dragon_rider", 1), ("balloon", 2)]),
        ):
            with self.subTest(frame=filename):
                path = fixtures / filename
                texts = [OCRText(item.text, item.confidence, scale_box(
                    item.bbox, from_resolution=(2560, 1440), to_resolution=(1280, 720)))
                    for item in provider.recognize(path)]
                battle = recognizer._battle_observation(path, texts, scene)
                troops = [(card["unit_id"], card["count"]) for card in battle["slots"]
                          if card["kind"] == "troop"]
                self.assertEqual(troops, expected)
                self.assertTrue(all(card["source"] == "army" for card in battle["slots"]
                                    if card["kind"] == "troop"))


if __name__ == "__main__":
    unittest.main()
