"""Synthetic contract checks only; no U model or gameplay readiness claim."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from autococ.dynamic_vision import DynamicModel
from training.vision.dynamic_snapshot import freeze_dynamic_snapshot, verify_dynamic_snapshot
from training.vision.snapshot import TrainingError
from training.vision.u_export import _validate_output, export_u_position
from training.vision.u_train import U_PREPROCESS, stage_u_rois, train_u_position, u_config, u_rows
from training.vision.data import sha256
from tests.test_dynamic_evaluation import u_window


def config(snapshot):
    return {"kind": "u_position", "classes": snapshot["metadata"]["classes"],
            "client_versions": snapshot["metadata"]["client_versions"], "prototype_only": True,
            "code_version": "synthetic-u-contract-test", "architecture": "yolo26n", "batch": 2,
            "preprocess": U_PREPROCESS,
            "applicability": {"roi_kind": "u_position", "source_width_range": [20, 20],
                              "source_height_range": [20, 20], "requires_stable_camera": True,
                              "requires_verified_zoom": True},
            "output": {"format": "xyxy_score_class", "min_confidence": .25,
                       "max_detections": 300}, "max_frame_age_seconds": 1.0,
            "event_rules": {"consecutive_frames": 2, "min_iou": .4,
                            "max_gap_seconds": 1.0, "max_missed_frames": 1,
                            "event_min_confidence": .95, "max_frame_age_seconds": .5}}


class UTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        rows = []
        for split, offset in (("train", 25), ("val", 85), ("test", 145)):
            row = u_window(source, f"u-{split}", boxes=[
                {"frame_id": "f1", "label": "hero:king", "bbox": [4, 5, 12, 13]},
                {"frame_id": "f2", "label": "troop_group", "bbox": [3, 4, 15, 16]}])
            row["split"] = split
            for index, frame in enumerate(row["frames"]):
                path = source / frame["image"]
                image = np.full((20, 20, 3), offset + index, np.uint8)
                path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
                frame["sha256"] = frame["parent_image_sha256"] = sha256(path)
                frame["boxes_source_frame_sha256"] = frame["sha256"]
            rows.append(row)
        self.snapshot_path = self.root / "snapshot"
        self.snapshot = freeze_dynamic_snapshot(source, rows, self.snapshot_path,
                                                kind="u_position", prototype_only=True)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config(self.snapshot)), encoding="utf-8")

    def test_frozen_boxes_are_roi_transformed_and_empty_frame_is_explicit_negative(self):
        rows = u_rows(self.snapshot)
        self.assertEqual(len(rows["train"]), 3)
        self.assertEqual(len(rows["val"]), 3)
        self.assertEqual(len(rows["test"]), 3)
        self.assertEqual([len(row["boxes"]) for row in rows["train"]], [1, 1, 0])
        counts = stage_u_rois(self.snapshot_path, self.root / "staged-run", rows,
                              self.snapshot["metadata"]["classes"])
        self.assertEqual(counts, {"train": 3, "val": 3})
        label = (self.root / "staged-run" / "staged" / "labels" / "train-000000.txt").read_text()
        values = [float(value) for value in label.split()]
        # ROI [2,2,18,18] -> source box [4,5,12,13], YOLO normalized center/size.
        self.assertEqual(int(values[0]), 0)
        np.testing.assert_allclose(values[1:], [.375, .4375, .5, .5])
        self.assertEqual((self.root / "staged-run" / "staged" / "labels" /
                          "train-000002.txt").read_text(), "")
        self.assertEqual(verify_dynamic_snapshot(self.snapshot_path, kind="u_position")["snapshot_sha256"],
                         self.snapshot["snapshot_sha256"])

    def test_reject_config_drift_and_non_one_to_one_outputs(self):
        broken = config(self.snapshot)
        broken["classes"] = ["troop_group", "hero:king"]
        self.config_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "frozen classes"):
            u_config(self.snapshot, self.config_path)
        raw = np.zeros((1, 5, 8400), np.float32)
        with self.assertRaisesRegex(TrainingError, r"\[1,N,6\]"):
            _validate_output(raw, 2)

    @unittest.skipUnless(os.environ.get("AUTOCOC_DYNAMIC_U_INTEGRATION") == "1",
                         "isolated local-weight training environment only")
    def test_synthetic_train_export_real_cpu_loader_and_resume_binding(self):
        pretrained = Path(os.environ["AUTOCOC_U_LOCAL_WEIGHT"])
        run = self.root / "run"
        report = train_u_position(self.snapshot_path, self.config_path, run, device="cpu",
                                  pretrained=pretrained, max_epochs=1, deadline_seconds=300)
        self.assertEqual(report["train_frames"], 3)
        self.assertEqual(report["epochs_completed"], 1)
        package = self.root / "package"
        exported = export_u_position(run, self.snapshot_path, package)
        self.assertTrue(exported["same_tensor_parity"])
        self.assertEqual(exported["provider"], "CPUExecutionProvider")
        self.assertEqual(DynamicModel(package, client_version="18.647.2").kind, "u_position")
        changed = config(self.snapshot)
        changed["code_version"] = "changed"
        self.config_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "differs"):
            train_u_position(self.snapshot_path, self.config_path, run, device="cpu",
                             pretrained=pretrained, resume=run / "resume.pt",
                             max_epochs=1, deadline_seconds=300)


if __name__ == "__main__":
    unittest.main()
