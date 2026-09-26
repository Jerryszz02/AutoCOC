import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.building_vision import building_coverage, detect_buildings
from scripts.prepare_building_samples import prepare, validate_variant


class BuildingVisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(8)
        self.alive = rng.integers(0, 255, (34, 38, 3), dtype=np.uint8)
        self.rubble = rng.integers(0, 255, (34, 38, 3), dtype=np.uint8)
        self.background = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        for name, image in (("alive.png", self.alive), ("rubble.png", self.rubble)):
            cv2.imencode(".png", image)[1].tofile(self.root / name)
        (self.root / "building_templates.json").write_text(json.dumps({"client": "test", "levels": [
            {"type": "air_defense", "level": 1, "zoom": "minimum", "alive": "alive.png",
             "destroyed": "rubble.png", "validation": {"status": "validated",
                 "source_alive": "source-alive.png", "source_destroyed": "source-destroyed.png",
                 "heldout_alive": "holdout-alive.png", "heldout_destroyed": "holdout-destroyed.png"}}]}))

    def frame(self, name, image):
        path = self.root / name
        cv2.imencode(".png", image)[1].tofile(path)
        return path

    def test_disappearance_does_not_prove_destruction(self):
        image = self.background.copy()
        image[200:234, 300:338] = self.alive
        before = detect_buildings(self.frame("before.png", image), root=self.root, threshold=.99)
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0]["state"], "alive")
        image[200:234, 300:338] = self.background[200:234, 300:338]
        after = detect_buildings(self.frame("after.png", image), root=self.root,
                                 previous=before, threshold=.99)
        self.assertEqual(after[0]["state"], "unknown")
        self.assertEqual(after[0]["building_id"], before[0]["building_id"])

    def test_fresh_rubble_is_positive_destruction_evidence(self):
        image = self.background.copy()
        image[200:234, 300:338] = self.alive
        before = detect_buildings(self.frame("before.png", image), root=self.root, threshold=.99)
        image[200:234, 300:338] = self.rubble
        after = detect_buildings(self.frame("after.png", image), root=self.root,
                                 previous=before, threshold=.99)
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]["state"], "destroyed")
        self.assertEqual(after[0]["evidence"]["method"], "positive_rubble_template")
        self.assertIs(after[0]["evidence"]["camera_motion"]["stationary"], True)
        self.assertEqual(building_coverage(self.root)["air_defense"]["recognition"], "sampled")
        self.assertEqual(building_coverage(self.root)["spell_factory"]["recognition"], "unavailable")

    def test_rubble_after_camera_pan_cannot_inherit_previous_building(self):
        image = self.background.copy()
        image[200:234, 300:338] = self.alive
        before = detect_buildings(self.frame("pan-before.png", image), root=self.root, threshold=.99)
        shifted = np.roll(image, 40, axis=1)
        shifted[200:234, 300:338] = self.rubble
        after = detect_buildings(self.frame("pan-after.png", shifted), root=self.root,
                                 previous=before, threshold=.99)
        self.assertFalse(any(item["state"] == "destroyed" for item in after))

    def test_existing_files_without_independent_validation_are_not_coverage(self):
        (self.root / "building_templates.json").write_text(json.dumps({"client": "test", "levels": [
            {"type": "air_defense", "level": "unknown_variant", "alive": "alive.png",
             "destroyed": "rubble.png"}]}))
        image = self.background.copy()
        image[200:234, 300:338] = self.alive
        self.assertEqual(building_coverage(self.root)["air_defense"]["recognition"], "unavailable")
        self.assertEqual(detect_buildings(self.frame("unvalidated.png", image), root=self.root), [])

    def test_live_version_must_match_before_any_alive_or_destroyed_match_is_used(self):
        path = self.root / "building_templates.json"
        manifest = json.loads(path.read_text())
        manifest["client_version"] = "18.600.7"
        path.write_text(json.dumps(manifest))
        image = self.background.copy()
        image[200:234, 300:338] = self.alive
        frame = self.frame("version-alive.png", image)
        observed = detect_buildings(frame, root=self.root, client_version="18.600.7")
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0]["evidence"]["version_verified"])
        self.assertFalse(detect_buildings(frame, root=self.root)[0]["evidence"]["version_verified"])
        for version in ("unknown", "", "19.0.0"):
            with self.subTest(version=version):
                self.assertEqual(detect_buildings(frame, root=self.root, previous=observed,
                                                 client_version=version), [])
                self.assertEqual(building_coverage(self.root, client_version=version)["air_defense"]["recognition"],
                                 "unavailable")

    def test_sample_promotion_rejects_unverified_or_mixed_versions_without_writing(self):
        manifest = self.root / "building_templates.json"
        original = manifest.read_bytes()
        spec = self.root / "spec.json"
        for version in (None, "unknown", "18.600.7"):
            with self.subTest(version=version):
                spec.write_text(json.dumps({"client_version": version, "variants": [{}]}))
                with self.assertRaisesRegex(ValueError, "client_version|different client versions"):
                    prepare(spec, self.root)
                self.assertEqual(manifest.read_bytes(), original)
                self.assertFalse((self.root / "buildings").exists())

    def test_uncertain_camera_frame_retains_anchor_until_positive_recheck(self):
        import autococ.terrain as terrain
        image = self.background.copy()
        image[200:234, 300:338] = self.alive
        before = detect_buildings(self.frame("anchor.png", image), root=self.root, threshold=.99)
        image[200:234, 300:338] = self.background[200:234, 300:338]
        middle = self.frame("flash.png", image)
        image[200:234, 300:338] = self.rubble
        after = self.frame("rubble-after-flash.png", image)
        calls = []
        original = terrain.measure_camera_motion
        def motion(first, second):
            calls.append((str(first), str(second)))
            return {"stationary": None} if str(second) == str(middle) else original(first, second)
        terrain.measure_camera_motion = motion
        self.addCleanup(setattr, terrain, "measure_camera_motion", original)
        unknown = detect_buildings(middle, root=self.root, previous=before, threshold=.99)
        self.assertEqual(unknown[0]["state"], "unknown")
        self.assertEqual(unknown[0]["anchor_frame"], str(before[0]["frame"]))
        verified = detect_buildings(after, root=self.root, previous=unknown, threshold=.99)
        self.assertEqual(verified[0]["state"], "destroyed")
        self.assertEqual(calls[-1][0], str(before[0]["frame"]))

    def test_reviewed_sample_requires_independent_alive_and_rubble_pairs(self):
        annotations = {}
        for run in ("source", "heldout"):
            directory = self.root / run / "frames"
            directory.mkdir(parents=True)
            background = self.background.copy()
            for state, crop in (("alive", self.alive), ("destroyed", self.rubble)):
                image = background.copy()
                image[200:234, 300:338] = crop
                path = directory / f"{state}.png"
                cv2.imencode(".png", image)[1].tofile(path)
                annotations[f"{run}_{state}"] = {"frame": str(path),
                                                  "bbox": [300, 200, 338, 234]}
        negative = self.root / "negative.png"
        cv2.imencode(".png", self.background)[1].tofile(negative)
        entry, alive, destroyed = validate_variant({
            "type": "air_defense", "level": "unknown_variant", "zoom": "minimum",
            "reviewed_label": True, **annotations,
            "negative": {"frame": str(negative), "bbox": [400, 250, 438, 284]},
        }, self.root)
        self.assertEqual(entry["validation"]["status"], "validated")
        self.assertGreater(entry["validation"]["scores"]["alive_holdout"], .94)
        self.assertEqual(alive.shape, destroyed.shape)
        with self.assertRaisesRegex(ValueError, "reviewer"):
            validate_variant({"type": "air_defense", "level": 1}, self.root)


if __name__ == "__main__":
    unittest.main()
