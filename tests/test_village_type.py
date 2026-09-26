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

    def test_no_positive_anchor_stays_unknown(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            cv2.imencode(".png", np.zeros((720, 1280, 3), np.uint8))[1].tofile(path)
            self.assertEqual(recognize_village_type(path)["type"], "unknown")


if __name__ == "__main__":
    unittest.main()
