from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.hero_state import recognize_hero_state


class HeroStateTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    slot = [599, 595, 688, 711]

    def test_selected_border_with_shaded_segment_requires_a_closed_outline(self) -> None:
        folder = self.root / "tests/fixtures"
        slot = [502, 592, 591, 711]
        before = recognize_hero_state(folder / "hero-border-before.png", slot)
        selected = recognize_hero_state(folder / "hero-border-selected.png", slot)
        self.assertFalse(before["selected"])
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["state"], "selected")
        self.assertIsNone(selected["deployed"])
        self.assertIsNone(selected["ability_ready"])

    def fixture(self, number: int) -> Path:
        folder = self.root / "reports/live-20260922/hero-state-023106/frames"
        matches = list(folder.glob(f"{number:05d}-*.png"))
        if not matches:
            self.skipTest("Optional controlled hero-state fixture is absent")
        return matches[0]

    def changed(self, image: np.ndarray) -> dict:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "changed.png"
            cv2.imencode(".png", image)[1].tofile(path)
            return recognize_hero_state(path, self.slot)

    def baseline_image(self) -> np.ndarray:
        return cv2.resize(cv2.imdecode(np.fromfile(self.fixture(4), dtype=np.uint8), cv2.IMREAD_COLOR),
                          (1280, 720), interpolation=cv2.INTER_AREA)

    def test_white_selection_is_separate_from_deployment_and_ability(self) -> None:
        before = recognize_hero_state(self.fixture(1), self.slot)
        selected = recognize_hero_state(self.fixture(2), self.slot)
        self.assertFalse(before["selected"])
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["state"], "selected")
        for result in (before, selected):
            self.assertIsNone(result["deployed"])
            self.assertIsNone(result["ability_ready"])
            self.assertIsNone(result["ability_used"])
            self.assertIsNone(result["defeated"])

    def test_both_stable_deployed_frames_have_health_and_warden_ready_evidence(self) -> None:
        for number in (3, 4):
            with self.subTest(frame=number):
                result = recognize_hero_state(self.fixture(number), self.slot)
                self.assertTrue(result["deployed"])
                self.assertTrue(result["ability_ready"])
                self.assertFalse(result["selected"])
                self.assertEqual(result["state"], "ability_ready")
                self.assertGreaterEqual(min(result["evidence"]["health_bar"]["frame_support"]), .85)
                self.assertEqual(result["evidence"]["ability_ready"]["hero"], "grand_warden")
                self.assertEqual(result["evidence"]["ability_ready"]["equipment_layout"], "observed_gold_book")
                self.assertFalse(result["ability_used"])
                self.assertFalse(result["defeated"])

    def test_old_selected_hero_with_deployed_siege_does_not_inherit_its_health_bar(self) -> None:
        path = self.root / "reports/20260923-021847-763016-3632dc81/frames/00015-deploy-support-selected.png"
        if not path.is_file():
            self.skipTest("Optional older selected-hero negative is absent")
        result = recognize_hero_state(path, self.slot)
        self.assertTrue(result["selected"])
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_intermediate_selection_animation_is_identified_without_deployment(self) -> None:
        path = self.root / "reports/20260924-003409-344711-0e4a3e7a/frames/00024-deploy-support-selected.png"
        if not path.is_file():
            self.skipTest("Optional intermediate selected-hero fixture is absent")
        result = recognize_hero_state(path, [502, 595, 590, 711])
        self.assertTrue(result["selected"])
        self.assertEqual(result["evidence"]["pet_anchor"]["scale"], 1.05)
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_deployed_pet_icons_over_map_background_still_require_health(self) -> None:
        folder = self.root / "reports/20260924-004347-675110-2e2b80bd/frames"
        for number, slot in ((31, [599, 595, 687, 711]), (35, [696, 595, 784, 711])):
            with self.subTest(frame=number):
                paths = list(folder.glob(f"{number:05d}-*.png"))
                if not paths:
                    self.skipTest("Optional varied-background hero fixture is absent")
                result = recognize_hero_state(paths[0], slot)
                self.assertTrue(result["deployed"])
                self.assertEqual(result["evidence"]["pet_anchor"]["comparison"], "opaque_pet_foreground")
                self.assertIsNot(result["ability_used"], True)
                image = cv2.resize(cv2.imread(str(paths[0])), (1280, 720), interpolation=cv2.INTER_AREA)
                image[567:585, slot[0]:slot[2]] = (25, 25, 25)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "missing-health.png"
                    cv2.imwrite(str(path), image)
                    self.assertIsNone(recognize_hero_state(path, slot)["deployed"])

    def test_green_patch_without_dark_hp_frame_is_not_deployment(self) -> None:
        image = self.baseline_image()
        image[571:574, 604:684] = (220, 220, 220)
        image[582:585, 604:684] = (220, 220, 220)
        result = self.changed(image)
        self.assertIn("pet_anchor", result["evidence"])
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_gray_ability_button_alone_does_not_prove_used(self) -> None:
        image = self.baseline_image()
        image[526:568, 600:642] = cv2.cvtColor(cv2.cvtColor(image[526:568, 600:642], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        result = self.changed(image)
        self.assertTrue(result["deployed"])
        self.assertIsNone(result["ability_ready"])
        self.assertIsNone(result["ability_used"])

    def test_controlled_ability_use_gray_card_is_alive_not_defeated(self) -> None:
        folder = self.root / "reports/live-20260922/warden-activate-023930/frames"
        if not folder.is_dir():
            self.skipTest("Optional controlled ability-use fixture is absent")
        before = recognize_hero_state(folder / "00001-before.png", self.slot)
        self.assertTrue(before["ability_ready"])
        self.assertFalse(before["ability_used"])
        for number in (3, 4):
            with self.subTest(frame=number):
                result = recognize_hero_state(folder / f"{number:05d}-after-ability.png", self.slot)
                self.assertEqual(result["state"], "ability_used")
                self.assertTrue(result["deployed"])
                self.assertTrue(result["ability_used"])
                self.assertFalse(result["ability_ready"])
                self.assertFalse(result["defeated"])
                self.assertIn("health_bar", result["evidence"])
                self.assertEqual(result["evidence"]["ability_used"]["hero"], "grand_warden")

    def test_first_ability_flash_frame_is_not_stable_ready_or_used(self) -> None:
        path = self.root / "reports/live-20260922/warden-activate-023930/frames/00002-after-ability.png"
        if not path.is_file():
            self.skipTest("Optional ability-flash fixture is absent")
        result = recognize_hero_state(path, self.slot)
        self.assertIsNone(result["ability_ready"])
        self.assertIsNone(result["ability_used"])
        self.assertIsNone(result["defeated"])

    def test_gray_portrait_without_hp_cannot_prove_used_or_defeated(self) -> None:
        path = self.root / "reports/live-20260922/warden-activate-023930/frames/00004-after-ability.png"
        if not path.is_file():
            self.skipTest("Optional used portrait fixture is absent")
        image = cv2.resize(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR),
                           (1280, 720), interpolation=cv2.INTER_AREA)
        image[571:585, 604:684] = (35, 35, 35)
        result = self.changed(image)
        self.assertIn("pet_anchor", result["evidence"])
        self.assertIsNone(result["deployed"])
        self.assertIsNone(result["ability_used"])
        self.assertIsNone(result["defeated"])

    def test_active_book_alone_without_warden_card_glow_is_not_ready(self) -> None:
        image = self.baseline_image()
        image[631:680, 603:685] = (70, 60, 80)
        result = self.changed(image)
        self.assertTrue(result["deployed"])
        self.assertIsNone(result["ability_ready"])

    def test_other_heroes_do_not_inherit_warden_ability_state(self) -> None:
        for slot in ([696, 595, 784, 711], [792, 595, 881, 711], [889, 595, 978, 711]):
            with self.subTest(slot=slot):
                result = recognize_hero_state(self.fixture(4), slot)
                self.assertIn("pet_anchor", result["evidence"])
                self.assertIsNone(result["ability_ready"])
                self.assertIsNone(result["ability_used"])

    def test_warden_glow_animation_phases_keep_both_positive_ready_anchors(self) -> None:
        folder = self.root / "reports/20260924-004347-675110-2e2b80bd/frames"
        for number in (27, 28, 29, 30, 31):
            with self.subTest(frame=number):
                paths = list(folder.glob(f"{number:05d}-*.png"))
                if not paths:
                    self.skipTest("Optional Warden glow-animation fixture is absent")
                result = recognize_hero_state(paths[0], [502, 595, 591, 711])
                self.assertTrue(result["deployed"])
                self.assertTrue(result["ability_ready"])
                self.assertFalse(result["ability_used"])
                self.assertGreaterEqual(result["evidence"]["ability_ready"]["active_equipment"]["confidence"], .9)
                self.assertGreaterEqual(result["evidence"]["ability_ready"]["card_book_glow"]["confidence"], .9)

    def test_missing_identity_and_invalid_card_geometry_stay_unknown(self) -> None:
        result = self.changed(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["deployed"])
        result = recognize_hero_state(self.fixture(4), [500, 300, 700, 500])
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["deployed"])

    def test_source_and_coordinate_scaling_preserve_state(self) -> None:
        result = recognize_hero_state(self.fixture(4), [1198, 1190, 1376, 1422], baseline_resolution=(2560, 1440))
        self.assertTrue(result["ability_ready"])
        resized = self.changed(self.baseline_image())
        self.assertTrue(resized["ability_ready"])
        self.assertEqual(result["evidence"]["health_bar"]["fill_bbox"], resized["evidence"]["health_bar"]["fill_bbox"])


class SampledHeroAbilityTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    folder = root / "reports/hero-ability-samples-20260924-015325-792650/frames"
    cases = (("barbarian_king", 599, 14, 16), ("minion_prince", 696, 20, 22), ("archer_queen", 792, 26, 28))

    def fixture(self, number):
        paths = list(self.folder.glob(f"{number:05d}-*.png"))
        if not paths:
            self.skipTest("Optional controlled three-hero ability fixture is absent")
        return paths[0]

    def read(self, number, left, image=None):
        slot = [left, 595, left + 89, 711]
        if image is None:
            return recognize_hero_state(self.fixture(number), slot)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "changed.png"
            cv2.imwrite(str(path), image)
            return recognize_hero_state(path, slot)

    def image(self, number):
        return cv2.resize(cv2.imread(str(self.fixture(number))), (1280, 720), interpolation=cv2.INTER_AREA)

    def test_controlled_clicks_produce_ready_then_stable_used_states(self):
        for hero, left, before, after in self.cases:
            with self.subTest(hero=hero):
                ready = self.read(before, left)
                self.assertTrue(ready["deployed"])
                self.assertTrue(ready["ability_ready"])
                self.assertFalse(ready["ability_used"])
                self.assertEqual(ready["evidence"]["ability_ready"]["hero"], hero)
                for number in (after, after + 1):
                    used = self.read(number, left)
                    self.assertTrue(used["deployed"])
                    self.assertTrue(used["ability_used"])
                    self.assertFalse(used["ability_ready"])
                    self.assertFalse(used["defeated"])
                    self.assertEqual(used["evidence"]["ability_used"]["hero"], hero)
                # Enlarged first animation frames do not justify another click.
                transient = self.read(before + 1, left)
                self.assertIsNone(transient["ability_ready"])
                self.assertIsNone(transient["ability_used"])

    def test_undeployed_and_selected_heroes_are_not_ready_or_used(self):
        for hero, left, before, _ in self.cases:
            with self.subTest(hero=hero):
                for number in (before - 2, before - 1):
                    state = self.read(number, left)
                    self.assertIsNone(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])

    def test_removing_health_preserves_identity_but_cancels_ability_evidence(self):
        for hero, left, before, after in self.cases:
            for number in (before, after):
                with self.subTest(hero=hero, frame=number):
                    image = self.image(number)
                    x0, y0, x1, y1 = self.read(number, left)["evidence"]["health_bar"]["fill_bbox"]
                    image[y0:y1, x0:x1] = 25
                    state = self.read(number, left, image)
                    self.assertIn("pet_anchor", state["evidence"])
                    self.assertIsNone(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])
                    self.assertIsNone(state["defeated"])

    def test_equipment_and_portrait_or_weapon_are_both_required(self):
        regions = (("ready_equipment", 0, 528, 42, 569), ("ready_portrait", 40, 594, 82, 646),
                   ("used_portrait", 40, 594, 82, 646), ("used_weapon", 35, 647, 82, 679))
        for hero, left, before, after in self.cases:
            for name, x0, y0, x1, y1 in regions:
                with self.subTest(hero=hero, removed=name):
                    number = before if name.startswith("ready") else after
                    image = self.image(number)
                    image[y0:y1, left + x0:left + x1] = 25
                    state = self.read(number, left, image)
                    self.assertTrue(state["deployed"])
                    self.assertIsNone(state["ability_ready"])
                    self.assertIsNone(state["ability_used"])

    def test_pet_reassignment_does_not_transfer_another_hero_ability(self):
        image, donor = self.image(14), self.image(20)
        a = self.read(14, 599)["evidence"]["pet_anchor"]["bbox"]
        b = self.read(20, 696)["evidence"]["pet_anchor"]["bbox"]
        image[a[1]:a[3], a[0]:a[2]] = cv2.resize(donor[b[1]:b[3], b[0]:b[2]], (a[2] - a[0], a[3] - a[1]))
        state = self.read(14, 599, image)
        self.assertTrue(state["deployed"])
        self.assertTrue(state["evidence"]["pet_anchor"]["template"].endswith("battle_hero_pet_2.png"))
        self.assertIsNone(state["ability_ready"])
        self.assertIsNone(state["ability_used"])

    def test_empty_and_unrecognized_red_health_do_not_prove_ability_use(self):
        for left in (599, 696):
            with self.subTest(left=left):
                state = self.read(29, left)
                self.assertIsNone(state["deployed"])
                self.assertIsNone(state["ability_used"])
                self.assertIsNone(state["defeated"])


if __name__ == "__main__":
    unittest.main()
