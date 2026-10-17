import unittest
import numpy as np

from training.vision.export import _manifest, _validate_raw_building
from training.vision.train import BUILDING_PREPROCESS, TrainingError


class ExportContractTests(unittest.TestCase):
    def test_raw_detector_keeps_all_candidates_and_rejects_capped_or_wrong_head(self):
        raw = np.zeros((1, 5, 8400), dtype=np.float32)
        raw[:, 2:4, :] = 8
        report = _validate_raw_building(raw, 1)
        self.assertEqual(report["candidates"], 8400)
        self.assertTrue(report["raw_untruncated"])
        raw[0, 2, 0] = -1
        raw[0, 4, 0] = .01
        self.assertEqual(_validate_raw_building(raw, 1)["invalid_geometry_candidates"], 1)
        with self.assertRaisesRegex(TrainingError, "not raw"):
            _validate_raw_building(np.zeros((1, 300, 6), dtype=np.float32), 1)
        with self.assertRaisesRegex(TrainingError, "bound"):
            _validate_raw_building(np.zeros((1, 5, 120001), dtype=np.float32), 1)

    def test_manifest_requires_explicit_output_contract(self):
        config = {"client_versions": ["verified-test"], "output": {"format": "yolo_xywh_classes", "coordinates": "input_pixels", "min_confidence": .5, "nms_iou": .5,
                  "evaluation_raw_provenance": {"unfiltered": True, "no_topk": True, "no_nms": True}},
                  "code_version": "test", "data_version": "test-data", "prototype_only": True,
                  "snapshot_sha256": "b" * 64, "split_sha256": "c" * 64,
                  "source_width_range": [40, 40], "source_height_range": [32, 32], "zoom": "minimum",
                  "classes": ["town_hall"], "preprocess": BUILDING_PREPROCESS}
        manifest = _manifest(config, "building", "a" * 64)
        self.assertEqual(manifest["validation"], {"status": "unvalidated", "prototype_only": True,
                                                   "snapshot_sha256": "b" * 64, "split_sha256": "c" * 64})
        self.assertEqual(manifest["classes"], ["town_hall"])
        self.assertNotIn("evaluation_raw_provenance", manifest["output"])
        config["output"]["format"] = "xyxy_score_class"
        with self.assertRaisesRegex(TrainingError, "raw xywh"):
            _manifest(config, "building", "a" * 64)


if __name__ == "__main__":
    unittest.main()
