import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.unit_catalog import ArmyRecipe, ArmyRequirement, coverage, recognize_card_identity
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
            (root / "unit_templates.json").write_text(json.dumps({"client": "test", "templates": [
                {"unit_id": "lightning_spell", "surface": "battle", "path": "lightning.png"}]}))
            image = np.zeros((720, 1280, 3), np.uint8)
            for left in (200, 500):
                image[625:661, left + 20:left + 56] = portrait
                observation = recognize_card_identity(image, (left, 595, left + 90, 710),
                                                       "spell", root=root, threshold=.99)
                self.assertEqual(observation["unit_id"], "lightning_spell")
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
            manifest = {"client": "test", "templates": [
                {"unit_id": "lightning_spell", "surface": "battle", "path": "same.png"},
                {"unit_id": "freeze_spell", "surface": "battle", "path": "same.png"}]}
            (root / "unit_templates.json").write_text(json.dumps(manifest))
            observed = recognize_card_identity(image, (200, 595, 290, 710), "spell", root=root)
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
