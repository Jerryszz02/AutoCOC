"""Bounded 18.647.2 blacksmith front calibration at the observed HOME zoom."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import cv2
import numpy as np

from autococ.ore_navigation import locate_blacksmith
from autococ.scene import SceneSnapshot


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/blacksmith_navigation_zoom103_18_647_2"
TEMPLATE = ROOT / "assets/templates/ore_navigation_blacksmith_front_zoom103_18_647_2.png"
VERSION = "18.647.2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shot(index: int, path: Path | None = None) -> SceneSnapshot:
    return SceneSnapshot(
        "village", .9, path or FIXTURES / f"home_zoom103_{index}.png",
        {"ocr": [], "village_type": "home", "resource_source": "village_inventory",
         "account_in_use": None, "buttons": [{"name": "shop"}, {"name": "attack"}],
         "frame_id": f"zoom103-{index}-{time.monotonic_ns()}",
         "observed_at_monotonic": time.monotonic(),
         "baseline_resolution": [1280, 720]},
    )


class OreNavigationZoom103Tests(unittest.TestCase):
    def test_exact_source_binding_and_minimal_anonymization(self):
        p = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(p["client_version"], VERSION)
        self.assertFalse(p["formal_acceptance"])
        self.assertFalse(p["new_run_selected_blacksmith_verified"])
        self.assertEqual(p["prior_confirmed_selection"]["run"],
                         "20261016-224105-027202-f362718d")
        self.assertEqual(p["source_mapping_diagnostic"]["affine_inliers"], 342)
        self.assertAlmostEqual(p["source_mapping_diagnostic"]["map_scale"], 1.029241699878419)
        self.assertEqual(p["template"]["physical_bbox_xyxy"], [762, 192, 780, 214])
        self.assertEqual(_sha256(TEMPLATE), p["template"]["sha256"])
        asset = cv2.imread(str(TEMPLATE))
        self.assertEqual(asset.shape, (22, 18, 3))
        self.assertEqual(hashlib.sha256(asset.tobytes()).hexdigest(),
                         p["template"]["source_pixel_sha256"])
        x0, y0, x1, y1 = p["transformation"]["retained_roi_xyxy"]
        for index, entry in p["fixtures"].items():
            with self.subTest(index=index):
                path = ROOT / entry["derived_path"]
                self.assertEqual(_sha256(path), entry["derived_sha256"])
                image = cv2.imread(str(path))
                self.assertEqual(image.shape, (720, 1280, 3))
                self.assertEqual(hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest(),
                                 entry["retained_roi_pixel_sha256"])
                image[y0:y1, x0:x1] = (24, 24, 24)
                self.assertTrue(np.all(image == 24))

    def test_clean_current_fronts_pass_but_occluded_third_frame_rejects(self):
        for index in (48, 49):
            with self.subTest(index=index):
                result = locate_blacksmith(_shot(index), client_version=VERSION)
                self.assertIsNotNone(result)
                self.assertEqual(result["template"], TEMPLATE.name)
                self.assertEqual(result["match_roi"], [762, 192, 780, 214])
                self.assertEqual(result["point"], [763, 202])
                self.assertGreaterEqual(result["ncc"], .985)
                self.assertLessEqual(result["mad"], 3)
                self.assertGreaterEqual(result["ncc"] - result["runner_up"], .15)
                self.assertFalse(result["formal_success"])
        self.assertIsNone(locate_blacksmith(_shot(50), client_version=VERSION))

    def test_duplicate_front_other_zoom_and_wrong_version_reject(self):
        original = cv2.imread(str(FIXTURES / "home_zoom103_48.png"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.png"
            duplicated = original.copy()
            duplicated[300:322, 500:518] = original[192:214, 762:780]
            self.assertTrue(cv2.imwrite(str(path), duplicated))
            self.assertIsNone(locate_blacksmith(_shot(48, path), client_version=VERSION))

            zoomed = cv2.warpAffine(original, cv2.getRotationMatrix2D((640, 360), 0, 1.08),
                                    (1280, 720), borderValue=(24, 24, 24))
            self.assertTrue(cv2.imwrite(str(path), zoomed))
            self.assertIsNone(locate_blacksmith(_shot(48, path), client_version=VERSION))

        self.assertIsNone(locate_blacksmith(_shot(48), client_version="18.647.3"))
        self.assertIsNone(locate_blacksmith(_shot(48), client_version=VERSION,
                                            baseline_resolution=(1920, 1080)))


if __name__ == "__main__":
    unittest.main()
