"""Current-client quantity fallback uses sampled pixels, never capacity guesses."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from autococ.army_manifest import _x1_template_paths, recognize_army_manifest
from autococ.ocr import OCRText
from autococ.unit_catalog import CATALOG_ROOT


FIXTURES = Path(__file__).parent / "fixtures"
ANCHORS = [OCRText("我的军队", .99, (222, 63, 294, 90)),
           OCRText("已保存的配置", .99, (594, 64, 685, 85)),
           OCRText("精选库", .99, (972, 64, 1024, 86))]


class CurrentArmyQuantityTests(unittest.TestCase):
    def setUp(self):
        self.provider = Mock()
        self.provider.recognize_line.return_value = []
        self.provider.recognize_line_image.return_value = []
        self.path = FIXTURES / "army_balloon_x1_18_647_2.png"

    def inspect(self, path=None, *, texts=None, version="18.647.2"):
        return recognize_army_manifest(path or self.path, self.provider,
                                       ANCHORS if texts is None else texts,
                                       client_version=version, groups=("troops",))

    def test_new_sample_recovers_only_observed_balloon_x1_without_ocr(self):
        with patch("autococ.army_manifest._x1_template_paths", return_value=None):
            self.assertIsNone(self.inspect()["troops"][3]["count"])
        result = self.inspect()
        self.assertEqual([card["count"] for card in result["troops"]], [None, None, None, 1])
        count = result["troops"][3]["evidence"]["count"]
        self.assertEqual(count["source"], "sampled_x1_header")
        self.assertEqual(count["x1_header_client_version"], "18.647.2")
        self.assertGreaterEqual(count["x1_header_scores"]["independent_glyph"], .85)
        self.assertGreaterEqual(count["x1_header_scores"]["whole_header"], .94)
        self.assertFalse(result["complete"])  # Other missing counts stay unknown.

    def test_current_x10_and_x2_do_not_match_sampled_one(self):
        result = self.inspect(FIXTURES / "army_current_18_647_2_20261015.png")
        self.assertIsNone(result["troops"][0]["count"])
        self.assertIsNone(result["troops"][3]["count"])

    def test_current_troop_header_cannot_authorize_a_spell_quantity(self):
        image = cv2.imdecode(np.fromfile(self.path, np.uint8), cv2.IMREAD_COLOR)
        # Even identical sampled pixels moved into the spell row must not use
        # the troop-only current-version fallback.
        image[373:470, 558:656] = image[194:291, 865:963]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "spell-header.png"
            cv2.imencode(".png", image)[1].tofile(path)
            result = recognize_army_manifest(path, self.provider, ANCHORS,
                                             client_version="18.647.2", groups=("spells",))
        self.assertEqual(len(result["spells"]), 1)
        self.assertIsNone(result["spells"][0]["count"])
        self.assertNotIn("x1_header_scores", result["spells"][0]["evidence"]["count"])

    def test_trailing_digit_and_missing_quantity_are_rejected(self):
        image = cv2.imdecode(np.fromfile(self.path, np.uint8), cv2.IMREAD_COLOR)
        extra = image.copy()
        extra[197:220, 897:905] = image[197:220, 883:891]
        blank = image.copy()
        blank[198:220, 869:899] = (20, 20, 20)
        with TemporaryDirectory() as folder:
            for name, altered in (("extra_digit", extra), ("blank", blank)):
                with self.subTest(name=name):
                    path = Path(folder) / (name + ".png")
                    cv2.imencode(".png", altered)[1].tofile(path)
                    self.assertIsNone(self.inspect(path)["troops"][3]["count"])

    def test_conflicting_exact_ocr_is_not_overridden_by_pixels(self):
        for counts in ((1, 2), (0,), (2,)):
            with self.subTest(counts=counts):
                texts = ANCHORS + [OCRText(f"x{count}", .99, (869, 197, 895, 218))
                                   for count in counts]
                result = self.inspect(texts=texts)["troops"][3]
                self.assertEqual(result["count"], 2 if counts == (2,) else None)
                self.assertNotEqual(result["evidence"]["count"]["source"], "sampled_x1_header")

    def test_contrast_conflict_does_not_fall_back_to_one(self):
        self.provider.recognize_line_image.return_value = [OCRText("x1", .99), OCRText("x2", .99)]
        self.assertIsNone(self.inspect()["troops"][3]["count"])

    def test_missing_or_unknown_version_has_no_template_fallback(self):
        for version in (None, "18.647.3", "unknown"):
            with self.subTest(version=version):
                self.assertIsNone(_x1_template_paths(version))
                self.assertIsNone(self.inspect(version=version)["troops"][3]["count"])

    def test_version_and_hash_bound_assets_fail_closed(self):
        original = json.loads((CATALOG_ROOT / "army_count_provenance_18_647_2.json").read_text(encoding="utf8"))
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            for filename in original["samples"]:
                (directory / filename).write_bytes((CATALOG_ROOT / filename).read_bytes())
            provenance_path = directory / "army_count_provenance_18_647_2.json"
            with patch("autococ.army_manifest.CATALOG_ROOT", directory):
                self.assertIsNone(_x1_template_paths("18.647.2"))
                for field, value in (("client", "18.600.7"), ("observed_count", 2)):
                    modified = dict(original, **{field: value})
                    provenance_path.write_text(json.dumps(modified), encoding="utf8")
                    self.assertIsNone(_x1_template_paths("18.647.2"))
                provenance_path.write_text(json.dumps(original), encoding="utf8")
                self.assertIsNotNone(_x1_template_paths("18.647.2"))
                first = directory / next(iter(original["samples"]))
                first.write_bytes(first.read_bytes() + b"changed")
                self.assertIsNone(_x1_template_paths("18.647.2"))

    def test_assets_and_anonymous_fixture_match_recorded_hashes(self):
        provenance = json.loads((CATALOG_ROOT / "army_count_provenance_18_647_2.json").read_text(encoding="utf8"))
        self.assertFalse(provenance["human_reviewed"])
        self.assertEqual(provenance["reviewer_kind"], "agent_visual")
        for filename, sample in provenance["samples"].items():
            self.assertEqual(hashlib.sha256((CATALOG_ROOT / filename).read_bytes()).hexdigest(), sample["sha256"])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), provenance["fixture"]["sha256"])
        image = cv2.imdecode(np.fromfile(self.path, np.uint8), cv2.IMREAD_COLOR)
        for left, top, right, bottom in provenance["fixture"]["preserved_rois_1280"]:
            image[top:bottom, left:right] = 0
        self.assertFalse(np.any(image))


if __name__ == "__main__":
    unittest.main()
