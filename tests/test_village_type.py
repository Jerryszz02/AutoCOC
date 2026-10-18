from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.vision import recognize_village_type


class VillageTypeTests(unittest.TestCase):
    fixtures = Path(__file__).parent / "fixtures"

    def test_real_home_variant_with_stars_has_positive_anchor(self):
        result = recognize_village_type(self.fixtures / "home_village_stars_anchor_20260926.png")
        self.assertEqual(result["type"], "home")
        self.assertGreaterEqual(result["scores"]["home"], .92)
        self.assertLess(result["scores"]["builder_base"], .5)

    def test_real_builder_base_anchor_is_not_home(self):
        result = recognize_village_type(self.fixtures / "builder_base_anchor_20260926.png")
        self.assertEqual(result["type"], "builder_base")
        self.assertGreaterEqual(result["scores"]["builder_base"], .92)
        self.assertLess(result["scores"]["home"], .5)

    def test_new_home_battle_icon_has_positive_anchor(self):
        template = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets/catalogs/village_home_battle_18_647_2.png"))
        image = np.zeros((720, 1280, 3), np.uint8)
        image[590:665, 45:120] = template
        with TemporaryDirectory() as directory:
            path = Path(directory) / "new-home.png"
            cv2.imwrite(str(path), image)
            result = recognize_village_type(path, client_version="18.647.2")
        self.assertEqual(result["type"], "home")
        self.assertEqual(result["point"], [82, 627])
        self.assertGreaterEqual(result["scores"]["home"], .92)

    def test_checklist_variant_requires_current_version_and_heldout_icon(self):
        icon = cv2.imread(str(self.fixtures / "home_battle_checklist_heldout_18_647_2.png"))
        image = np.zeros((720, 1280, 3), np.uint8)
        image[570:700, 25:145] = icon
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checklist-home.png"
            cv2.imwrite(str(path), image)
            current = recognize_village_type(path, client_version="18.647.2")
            unknown = recognize_village_type(path, client_version="unknown")
        self.assertEqual(current["type"], "home")
        self.assertEqual(current["point"], [82, 627])
        self.assertGreaterEqual(current["scores"]["home"], .92)
        self.assertEqual(unknown["type"], "unknown")

    def test_new_home_anchor_does_not_identify_builder_base(self):
        result = recognize_village_type(self.fixtures / "builder_base_anchor_20260926.png")
        self.assertEqual(result["type"], "builder_base")
        self.assertLess(result["scores"]["home"], .92)

    def test_no_positive_anchor_stays_unknown(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            cv2.imencode(".png", np.zeros((720, 1280, 3), np.uint8))[1].tofile(path)
            self.assertEqual(recognize_village_type(path)["type"], "unknown")


if __name__ == "__main__":
    unittest.main()
