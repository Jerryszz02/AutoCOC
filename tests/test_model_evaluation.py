import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from training.vision.evaluation import (compare_stability, evaluate_independent, load_config,
                                        raw_predictions, score_predictions)


class FakeRawModel:
    def __init__(self, rows, *, provenance=True):
        self.manifest = {"classes": ["town_hall"], "output": {
            "evaluation_raw_provenance": {"unfiltered": True, "no_topk": True, "no_nms": True}
            if provenance else {}}}
        self.output_format = "xyxy_score_class"
        self.rows = rows
        self.calls = 0
        self.min_confidence = .99  # must have no effect on raw AP
        self.max_candidates = 1

    def _prepare(self, image):
        return np.zeros((1, 3, 32, 32), dtype=np.float32), (1., 1., 0., 0.)

    def _run(self, tensor):
        result = self.rows[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return np.asarray(result, dtype=np.float32)


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))[1].tofile(self.root / "frame.png")
        self.config = load_config()

    def test_low_score_true_positive_and_runtime_threshold_invariance(self):
        truth = [[{"class": "town_hall", "bbox": [2, 2, 12, 12]}]]
        model = FakeRawModel([[[2, 2, 12, 12, .02, 0]]])
        first = evaluate_independent(model, [{"image": "frame.png", "boxes": truth[0],
                                              "group_id": "layout-1", "subset": "occluded"}],
                                     self.root, self.config)
        self.assertEqual(first["per_class"]["town_hall"]["ap50"], 1)
        self.assertEqual(first["per_class"]["town_hall"]["truth_groups"], 1)
        self.assertEqual(first["subset_counts"], {"occluded": 1})
        self.assertEqual(first["subset_breakdown"]["occluded"]["truth_instances_by_class"], {"town_hall": 1})
        self.assertEqual(first["status"], "full_protocol")
        model.min_confidence = .001
        model.rows = [[[2, 2, 12, 12, .02, 0]]]
        model.calls = 0
        second = evaluate_independent(model, [{"image": "frame.png", "boxes": truth[0],
                                               "group_id": "layout-1"}], self.root, self.config)
        self.assertEqual(first["mAP50_95"], second["mAP50_95"])

    def test_duplicate_matching_and_fresh_iou_threshold(self):
        truth = [[{"class": "town_hall", "bbox": [0, 0, 10, 10]}]]
        predictions = [[{"class": "town_hall", "bbox": [0, 0, 10, 10], "confidence": .9},
                        {"class": "town_hall", "bbox": [2, 0, 12, 10], "confidence": .8}]]
        result = score_predictions(predictions, truth, [(32, 32)], ["town_hall"], [.5, .75])
        self.assertEqual(result["per_class"]["town_hall"]["ap50"], 1)
        self.assertEqual(result["per_class"]["town_hall"]["ap_by_iou"]["0.75"], 1)
        # Higher-scored false positive precedes the only true match.
        predictions[0][0]["bbox"] = [20, 20, 30, 30]
        result = score_predictions(predictions, truth, [(32, 32)], ["town_hall"], [.5])
        self.assertAlmostEqual(result["mAP50"], .5)

    def test_candidate_truncation_and_unverified_provenance_downgrade(self):
        config = copy.deepcopy(self.config)
        config["max_candidates"] = 1
        model = FakeRawModel([[[2, 2, 12, 12, .9, 0], [18, 18, 28, 28, .8, 0]]], provenance=False)
        row = {"image": "frame.png", "boxes": [{"class": "town_hall", "bbox": [2, 2, 12, 12]}]}
        report = evaluate_independent(model, [row], self.root, config)
        self.assertFalse(report["full_ap_eligible"])
        self.assertIn("raw_provenance_unverified", report["limitations"])
        self.assertIn("evaluation_truncated", report["limitations"])
        self.assertEqual(report["truncation"]["candidate_truncation"], 1)

    def test_inference_failure_truth_remains_in_ap_denominator(self):
        rows = [[[2, 2, 12, 12, .9, 0]], RuntimeError("failed")]
        model = FakeRawModel(rows)
        inputs = [{"image": "frame.png", "boxes": [{"class": "town_hall", "bbox": [2, 2, 12, 12]}],
                   "group_id": str(index)} for index in range(2)]
        report = evaluate_independent(model, inputs, self.root, self.config)
        self.assertAlmostEqual(report["mAP50"], 51 / 101)
        self.assertEqual(report["per_class"]["town_hall"]["truth_instances"], 2)
        self.assertEqual(report["per_class"]["town_hall"]["truth_groups"], 2)
        self.assertEqual(report["failed"], 1)
        self.assertFalse(report["full_ap_eligible"])

    def test_raw_yolo_decoding_has_independent_score_floor(self):
        model = FakeRawModel([[[7, 7, 10, 10, .02]]])
        model.output_format = "yolo_xywh_classes"
        model.rows = [np.asarray([[[7, 7, 10, 10, .02], [20, 20, 10, 10, .0001]]], dtype=np.float32)]
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        predictions, meta = raw_predictions(model, image, self.config)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["bbox"], (2., 2., 12., 12.))
        self.assertEqual(meta["candidate_truncation"], 0)

    def test_stability_comparison_needs_complete_raw_ap(self):
        base = {"full_ap_eligible": True, "protocol": {"sha256": "frozen"},
                "per_class": {"town_hall": {"ap50": .8, "ap50_95": .6}}}
        lower = {"full_ap_eligible": True, "protocol": {"score_floor": .0001,
                    "max_candidates": 400000, "max_detections": 400000},
                 "per_class": {"town_hall": {"ap50": .8005, "ap50_95": .6005}}}
        self.assertTrue(compare_stability(base, lower)["passed"])
        lower["full_ap_eligible"] = False
        self.assertFalse(compare_stability(base, lower)["passed"])


if __name__ == "__main__":
    unittest.main()
