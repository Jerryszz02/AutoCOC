import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.building_vision import detect_model_buildings
from autococ.model_vision import (BuildingModel, BuildingObservation, ModelUnavailable,
                                  SceneModel, StaleModelFrame, infer_core_geometry)
from scripts.evaluate_model_package import evaluate


class _IO:
    name = "pixels"


class FakeSession:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.inputs = []

    def get_inputs(self):
        return [_IO()]

    def get_outputs(self):
        return [_IO()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        self.inputs.append(feed["pixels"])
        return [self.output]


class ModelVisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_file = self.root / "model.onnx"
        self.model_file.write_bytes(b"fake onnx model")
        self.digest = hashlib.sha256(self.model_file.read_bytes()).hexdigest()

    def package(self, kind, classes, output, **metadata):
        manifest = {"schema_version": 1, "kind": kind, "client_versions": ["18.600.7"],
            "classes": classes, "model_file": "model.onnx", "sha256": self.digest,
            "code_version": "test-code", "data_version": "test-data",
            "validation": {"status": "unvalidated"},
            "applicability": {"screen_scope": "full_ui" if kind == "scene" else "battlefield",
                              "source_width_range": [20, 200],
                              "source_height_range": [20, 100], "zoom": "minimum"},
            "preprocess": {"width": 100, "height": 100, "color": "RGB",
                           "resize": "letterbox", "scale": 1 / 255,
                           "mean": [0, 0, 0], "std": [1, 1, 1], "pad_value": 114},
            "max_frame_age_seconds": 2, "output": output, **metadata}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_scene_logits_unknown_and_preprocess(self):
        self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .7, "min_margin": .2})
        session = FakeSession([[0, 3]])
        model = SceneModel(self.root, client_version="18.600.7", session_factory=lambda _: session)
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # BGR blue becomes RGB third channel.
        result = model.classify(image, captured_at=10, now=11)
        self.assertEqual(result.scene, "battle")
        self.assertFalse(result.unknown)
        self.assertEqual(session.inputs[0].shape, (1, 3, 100, 100))
        self.assertAlmostEqual(float(session.inputs[0][0, 2, 50, 50]), 1)
        self.assertAlmostEqual(float(session.inputs[0][0, 2, 0, 0]), 114 / 255)
        session.output = np.array([[0, 0]], dtype=np.float32)
        self.assertTrue(model.classify(image, captured_at=10, now=11).unknown)
        with self.assertRaises(StaleModelFrame):
            model.classify(image, captured_at=10, now=13)

    def test_hash_version_classes_and_missing_package_reject(self):
        self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .5, "min_margin": .1})
        with self.assertRaisesRegex(ModelUnavailable, "incompatible"):
            SceneModel(self.root, client_version="unknown", session_factory=lambda _: FakeSession([[1, 0]]))
        self.model_file.write_bytes(b"tampered")
        with self.assertRaisesRegex(ModelUnavailable, "SHA-256"):
            SceneModel(self.root, session_factory=lambda _: FakeSession([[1, 0]]))
        with self.assertRaisesRegex(ModelUnavailable, "manifest"):
            SceneModel(self.root / "missing")

    def test_metadata_requires_explicit_applicability_and_validation(self):
        manifest = self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .5, "min_margin": .1})
        manifest.pop("applicability")
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ModelUnavailable, "applicability"):
            SceneModel(self.root, session_factory=lambda _: FakeSession([[1, 0]]))
        manifest["applicability"] = {"screen_scope": "full_ui", "source_width_range": [100, 100],
                                      "source_height_range": [50, 50], "zoom": "minimum"}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        model = SceneModel(self.root, session_factory=lambda _: FakeSession([[1, 0]]))
        self.assertEqual(model.metadata["validation"]["status"], "unvalidated")
        with self.assertRaisesRegex(ModelUnavailable, "applicability"):
            model.classify(np.zeros((20, 20, 3), dtype=np.uint8), captured_at=10, now=11)

    def test_detection_letterbox_restore_nms_and_core(self):
        self.package("building", ["town_hall", "air_defense"],
                     {"format": "xyxy_score_class", "coordinates": "input_pixels",
                      "min_confidence": .5, "nms_iou": .4})
        # Source 200x100 is scaled to 100x50 and padded by 25 vertically.
        session = FakeSession([[[20, 35, 40, 55, .95, 0],
                                [21, 36, 41, 56, .8, 0],
                                [60, 35, 80, 55, .9, 1]]])
        model = BuildingModel(self.root, session_factory=lambda _: session)
        detections = model.detect(np.zeros((100, 200, 3), dtype=np.uint8),
                                  captured_at=10, now=11)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].bbox, (40, 20, 80, 60))
        core = infer_core_geometry(detections, {"core_classes": {"town_hall": .9},
                                                "min_core_targets": 1},
                                   source_resolution=(200, 100))
        self.assertEqual(core.center, (384, 288))
        self.assertEqual(core.bbox, (256, 144, 512, 432))
        self.assertEqual(core.evidence, ("town_hall",))
        self.assertIsNone(infer_core_geometry(detections, {}, source_resolution=(200, 100)))
        self.assertIsNone(infer_core_geometry(detections, {"core_classes": {"town_hall": .99}},
                                              source_resolution=(200, 100)))

    def test_standard_yolo_output_and_invalid_frame_mixing(self):
        self.package("building", ["town_hall", "air_defense"],
                     {"format": "yolo_xywh_classes", "coordinates": "input_pixels",
                      "min_confidence": .5, "nms_iou": .5})
        # Channel-first YOLO output [1, 4 + number_of_classes, N].
        session = FakeSession([[[30], [45], [20], [20], [.92], [.1]]])
        model = BuildingModel(self.root, session_factory=lambda _: session)
        detected = model.detect(np.zeros((100, 200, 3), dtype=np.uint8),
                                captured_at=10, now=11)
        self.assertEqual(detected[0].bbox, (40, 20, 80, 60))
        other = BuildingObservation("town_hall", (40, 20, 80, 60), .95, 9, model.model_sha256)
        self.assertIsNone(infer_core_geometry((*detected, other),
            {"core_classes": {"town_hall": .5}}, source_resolution=(200, 100)))

    def test_building_wrapper_keeps_model_evidence_separate(self):
        self.package("building", ["town_hall"],
                     {"format": "xyxy_score_class", "coordinates": "input_pixels",
                      "min_confidence": .5, "nms_iou": .5})
        model = BuildingModel(self.root, session_factory=lambda _: FakeSession([[[20, 35, 40, 55, .95, 0]]]))
        frame = self.root / "frame.png"
        cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))[1].tofile(frame)
        self.assertEqual(detect_model_buildings(frame, model=model, captured_at=None)["status"], "unavailable")
        import time
        result = detect_model_buildings(frame, model=model, captured_at=time.monotonic())
        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["detections"][0]["state"], "visible")
        self.assertEqual(result["detections"][0]["evidence"]["model_sha256"], self.digest)

    def test_offline_evaluation_reports_rejects_and_failed_rows(self):
        self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .7, "min_margin": .2})
        model = SceneModel(self.root, session_factory=lambda _: FakeSession([[0, 3]]))
        cv2.imencode(".png", np.zeros((20, 20, 3), dtype=np.uint8))[1].tofile(self.root / "scene.png")
        report = evaluate(model, [{"image": "scene.png", "label": "battle"},
                                  {"image": "missing.png", "label": "village"}], self.root)
        self.assertEqual(report["per_class"]["battle"]["tp"], 1)
        self.assertEqual(report["per_class"]["battle"]["f1"], 1)
        self.assertEqual(report["per_class"]["village"]["fn"], 1)
        self.assertEqual(report["confusion_matrix"]["village"]["inference_error"], 1)
        self.assertEqual(report["coverage"], .5)
        self.assertEqual(report["coverage_valid_labeled"], .5)
        self.assertEqual(report["valid_labeled_samples"], 2)
        self.assertEqual(report["failed"], 1)
        self.assertEqual([item["phase"] for item in report["attempts"]], ["cold", "warm"])
        self.assertEqual(report["attempts"][0]["device"], "CPU")
        self.assertEqual(report["attempts"][0]["input_size"], [100, 100])
        self.assertEqual(report["timing"]["failed_attempts"], 1)
        self.assertIsNotNone(report["timing"]["p95_seconds"])

    def test_scene_rejection_rate_and_building_ap50_location(self):
        self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .7, "min_margin": .2})
        scene_session = FakeSession([[0, 3]])
        scene_model = SceneModel(self.root, session_factory=lambda _: scene_session)
        cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))[1].tofile(self.root / "frame.png")
        original = scene_session.run
        calls = 0
        def alternating(names, feed):
            nonlocal calls
            calls += 1
            scene_session.output = np.array([[0, 3]] if calls == 1 else [[0, 0]], dtype=np.float32)
            return original(names, feed)
        scene_session.run = alternating
        scene_report = evaluate(scene_model, [{"image": "frame.png", "label": "battle"},
                                              {"image": "frame.png", "label": "village"}], self.root)
        self.assertEqual(scene_report["coverage"], .5)
        self.assertEqual(scene_report["rejection_rate"], .5)
        self.assertEqual(scene_report["unknown_rejections"], 1)

        self.package("building", ["town_hall"],
                     {"format": "xyxy_score_class", "coordinates": "input_pixels",
                      "min_confidence": .5, "nms_iou": .5})
        building_session = FakeSession([[[70, 35, 90, 55, .95, 0],
                                         [20, 35, 40, 55, .9, 0]]])
        building_model = BuildingModel(self.root, session_factory=lambda _: building_session)
        report = evaluate(building_model,
            [{"image": "frame.png", "boxes": [{"class": "town_hall", "bbox": [40, 20, 80, 60]}]}],
            self.root)
        self.assertEqual(report["per_class"]["town_hall"]["tp"], 1)
        self.assertEqual(report["per_class"]["town_hall"]["fp"], 1)
        self.assertAlmostEqual(report["per_class"]["town_hall"]["ap50"], .5)
        self.assertAlmostEqual(report["mAP50"], .5)
        self.assertEqual(report["per_class"]["town_hall"]["mean_center_error_px"], 0)

    def test_failed_building_inference_remains_in_recall_and_ap_denominator(self):
        self.package("building", ["town_hall"],
                     {"format": "xyxy_score_class", "coordinates": "input_pixels",
                      "min_confidence": .5, "nms_iou": .5})
        session = FakeSession([[[20, 35, 40, 55, .9, 0]]])
        calls = 0
        original = session.run
        def fail_second(names, feed):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test inference failure")
            return original(names, feed)
        session.run = fail_second
        model = BuildingModel(self.root, session_factory=lambda _: session)
        cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))[1].tofile(self.root / "present.png")
        truth = [{"class": "town_hall", "bbox": [40, 20, 80, 60]}]
        report = evaluate(model, [{"image": "present.png", "boxes": truth},
                                  {"image": "present.png", "boxes": truth},
                                  {"image": "present.png", "boxes": [{"class": "bad", "bbox": [1, 2, 3, 4]}]}],
                          self.root)
        self.assertEqual(report["per_class"]["town_hall"]["tp"], 1)
        self.assertEqual(report["per_class"]["town_hall"]["fn"], 1)
        self.assertEqual(report["per_class"]["town_hall"]["recall"], .5)
        self.assertLess(report["per_class"]["town_hall"]["ap50"], 1)
        self.assertEqual(report["valid_labeled_samples"], 2)
        self.assertEqual(report["timing"]["failed_attempts"], 2)
        self.assertEqual(report["attempts"][1]["status"], "inference_error")
        self.assertEqual(report["attempts"][2]["status"], "invalid_label")

    def test_valid_image_model_failure_and_bad_scene_label(self):
        self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .7, "min_margin": .2})
        session = FakeSession([[0, 3]])
        session.output = np.array([[float("nan"), 3]], dtype=np.float32)
        model = SceneModel(self.root, session_factory=lambda _: session)
        cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))[1].tofile(self.root / "valid.png")
        report = evaluate(model, [{"image": "valid.png", "label": "battle"},
                                  {"image": "valid.png", "label": "not_a_scene"}], self.root)
        self.assertEqual(report["confusion_matrix"]["battle"]["inference_error"], 1)
        self.assertEqual(report["per_class"]["battle"]["fn"], 1)
        self.assertNotIn("not_a_scene", report["per_class"])
        self.assertEqual(report["valid_labeled_samples"], 1)
        self.assertEqual(report["attempts"][0]["status"], "inference_error")
        self.assertEqual(report["attempts"][1]["phase"], "preflight")
        self.assertGreaterEqual(report["attempts"][0]["inference_elapsed_seconds"], 0)

    def test_expired_after_inference_is_rejected_even_with_injected_now(self):
        import time
        self.package("scene", ["village", "battle"],
                     {"format": "logits", "min_confidence": .7, "min_margin": .2},
                     max_frame_age_seconds=.01)
        session = FakeSession([[0, 3]])
        original_run = session.run
        def slow_run(names, feed):
            time.sleep(.03)
            return original_run(names, feed)
        session.run = slow_run
        model = SceneModel(self.root, session_factory=lambda _: session)
        with self.assertRaises(StaleModelFrame):
            model.classify(np.zeros((50, 100, 3), dtype=np.uint8),
                           captured_at=10, now=10)

    def test_unicode_model_and_frame_paths(self):
        self.package("building", ["town_hall"],
                     {"format": "xyxy_score_class", "coordinates": "input_pixels",
                      "min_confidence": .5, "nms_iou": .5})
        package = self.root / "中文模型"
        package.mkdir()
        (package / "manifest.json").write_bytes((self.root / "manifest.json").read_bytes())
        (package / "model.onnx").write_bytes(self.model_file.read_bytes())
        seen = []
        def factory(path):
            seen.append(path)
            return FakeSession([[[20, 35, 40, 55, .95, 0]]])
        model = BuildingModel(package, session_factory=factory)
        self.assertIn("中文模型", seen[0])
        frame = self.root / "中文截图.png"
        cv2.imencode(".png", np.zeros((100, 200, 3), dtype=np.uint8))[1].tofile(frame)
        import time
        self.assertEqual(detect_model_buildings(frame, model=model,
                         captured_at=time.monotonic())["status"], "recognized")


if __name__ == "__main__":
    unittest.main()
