from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.hero_state import recognize_siege_state


class SiegeStateTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    slot = [498, 595, 587, 711]

    def fixture(self, number):
        folder = self.root / "reports/20260923-021847-763016-3632dc81/frames"
        paths = list(folder.glob(f"{number:05d}-*.png"))
        if not paths:
            self.skipTest("Optional live siege fixture is absent")
        return paths[0]

    def test_ready_and_selected_cards_are_not_deployment(self):
        for number in (12, 13):
            with self.subTest(frame=number):
                self.assertIsNone(recognize_siege_state(self.fixture(number), self.slot)["deployed"])

    def test_release_control_and_card_health_confirm_deployment(self):
        result = recognize_siege_state(self.fixture(15), self.slot)
        self.assertTrue(result["deployed"])
        self.assertIn("health_bar", result["evidence"])
        self.assertGreaterEqual(result["evidence"]["release_control"]["confidence"], .9)

    def test_neither_release_control_nor_health_alone_proves_deployment(self):
        original = cv2.resize(cv2.imdecode(np.fromfile(self.fixture(15), dtype=np.uint8), cv2.IMREAD_COLOR), (1280, 720))
        for roi in ((506, 572, 585, 589), (530, 650, 578, 706)):
            with self.subTest(erased=roi), TemporaryDirectory() as directory:
                changed = original.copy()
                left, top, right, bottom = roi
                changed[top:bottom, left:right] = (25, 25, 25)
                path = Path(directory) / "changed.png"
                cv2.imencode(".png", changed)[1].tofile(path)
                self.assertIsNone(recognize_siege_state(path, self.slot)["deployed"])

    def test_neighbor_selected_hero_cannot_supply_siege_proof(self):
        self.assertIsNone(recognize_siege_state(self.fixture(15), [599, 595, 688, 711])["deployed"])

    def test_smaller_siege_hp_and_release_over_another_map_require_stable_frame(self):
        folder = self.root / "reports/20260924-003409-344711-0e4a3e7a/frames"
        stable = folder / "00022-deploy-siege-verify.png"
        if not stable.is_file():
            self.skipTest("Optional smaller siege fixture is absent")
        slot = [401, 595, 490, 711]
        transition = recognize_siege_state(folder / "00021-deploy-siege-verify.png", slot)
        self.assertIsNone(transition["deployed"])
        result = recognize_siege_state(stable, slot)
        self.assertTrue(result["deployed"])
        self.assertGreaterEqual(result["evidence"]["release_control"]["confidence"], .99)
        image = cv2.resize(cv2.imread(str(stable)), (1280, 720), interpolation=cv2.INTER_AREA)
        for roi in ((410, 571, 485, 590), (433, 650, 482, 707)):
            with self.subTest(erased=roi), TemporaryDirectory() as directory:
                changed = image.copy()
                left, top, right, bottom = roi
                changed[top:bottom, left:right] = (25, 25, 25)
                path = Path(directory) / "changed.png"
                cv2.imwrite(str(path), changed)
                self.assertIsNone(recognize_siege_state(path, slot)["deployed"])

    def test_bobbing_release_control_may_extend_below_old_crop(self):
        path = self.root / "reports/20260924-004347-675110-2e2b80bd/frames/00023-deploy-siege-verify.png"
        if not path.is_file():
            self.skipTest("Optional low release-control fixture is absent")
        result = recognize_siege_state(path, [401, 595, 490, 711])
        self.assertTrue(result["deployed"])
        self.assertGreater(result["evidence"]["release_control"]["bbox"][3], 710)
