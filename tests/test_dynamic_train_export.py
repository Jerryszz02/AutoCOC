"""Synthetic contract tests only; these fixtures are never R model readiness."""

import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import cv2

from autococ.dynamic_vision import BUILDING_STATES, CARD_STATES, DynamicFrame, DynamicModel
from training.vision.dynamic_export import export_dynamic_state
from training.vision.dynamic_snapshot import freeze_dynamic_snapshot, verify_dynamic_snapshot
from training.vision.dynamic_train import R_PREPROCESS, _config, _rows, roi_tensor, train_dynamic_state
from training.vision.data import sha256
from training.vision.snapshot import TrainingError
from tests.test_dynamic_evaluation import window


def config(snapshot: dict) -> dict:
    return {"kind": "r_building", "classes": list(BUILDING_STATES),
            "client_versions": ["18.647.2"], "prototype_only": True,
            "code_version": "synthetic-r-contract-test", "architecture": "mobilenet_v3_small",
            "batch": 2, "preprocess": R_PREPROCESS,
            "applicability": {"roi_kind": "r_building", "source_width_range": [20, 20],
                              "source_height_range": [20, 20], "requires_stable_camera": True,
                              "requires_verified_zoom": True},
            "output": {"format": "logits", "min_confidence": .0},
            "max_frame_age_seconds": 1.0,
            "event_rules": {"consecutive_frames": 2, "min_iou": .4,
                            "max_gap_seconds": 1.0, "max_missed_frames": 1,
                            "event_min_confidence": .95, "max_frame_age_seconds": .5}}


class DynamicTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        rows = [window(source, "r-train", True, split="train"),
                window(source, "r-val", False, split="val"),
                window(source, "r-test", False, split="test")]
        for group_index, row in enumerate(rows, start=1):
            for frame_index, frame in enumerate(row["frames"], start=1):
                path = source / frame["image"]
                path.write_bytes(cv2.imencode(".png", np.full((20, 20, 3),
                    group_index * 30 + frame_index, np.uint8))[1].tobytes())
                frame["sha256"] = frame["parent_image_sha256"] = sha256(path)
                frame["state_source_frame_sha256"] = frame["sha256"]
        self.snapshot_path = self.root / "snapshot"
        self.snapshot = freeze_dynamic_snapshot(source, rows, self.snapshot_path,
                                                kind="r_building", prototype_only=True)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config(self.snapshot)), encoding="utf-8")

    def test_overlapping_reviewed_windows_have_one_sampling_weight_per_source_roi_target(self):
        original = deepcopy(next(row for row in self.snapshot["windows"] if row["split"] == "train"))
        duplicate = deepcopy(original)
        duplicate["window_id"] = "same-pixels-another-event"
        inventory = {"window_components": {original["window_id"]: "source-component",
                                           duplicate["window_id"]: "source-component"}}
        repeated = {"windows": [original, duplicate], "inventory": inventory}
        self.assertEqual(len(_rows(repeated)["train"]), 3)
        duplicate["frames"][0]["state_label"] = "rubble"
        with self.assertRaisesRegex(TrainingError, "Conflicting.*R supervision"):
            _rows(repeated)
        duplicate["frames"][0]["state_label"] = original["frames"][0]["state_label"]
        for frame in duplicate["frames"]:
            frame["roi_bbox"] = [3, 3, 18, 18]
        self.assertEqual(len(_rows(repeated)["train"]), 6)

    def test_verified_frame_labels_and_split_are_training_inputs(self):
        self.assertEqual(self.snapshot["metadata"]["classes"], list(BUILDING_STATES))
        self.assertEqual({row["split"] for row in self.snapshot["windows"]},
                         {"train", "val", "test"})
        frame = self.snapshot["windows"][0]["frames"][0]
        tensor = roi_tensor(self.snapshot_path, frame)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        broken = json.loads(self.config_path.read_text())
        broken["classes"] = ["intact", "rubble"]
        self.config_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "differ from frozen snapshot"):
            _config(self.snapshot, self.config_path)
        broken = config(self.snapshot)
        broken["applicability"]["source_width_range"] = [21, 21]
        self.config_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "outside declared applicability"):
            _config(self.snapshot, self.config_path)

    @unittest.skipUnless(os.environ.get("AUTOCOC_DYNAMIC_INTEGRATION") == "1",
                         "isolated training environment only")
    def test_synthetic_r_train_resume_binding_and_cpu_onnx_loader(self):
        run = self.root / "run"
        result = train_dynamic_state(self.snapshot_path, self.config_path, run,
                                     device="cpu", max_epochs=1, deadline_seconds=120,
                                     pretrained=None)
        self.assertEqual(result["epochs_completed"], 1)
        self.assertEqual(result["train_frames"], 3)
        self.assertEqual(result["validation_frames"], 3)
        self.assertEqual(result["status"], "prototype_only")
        package = self.root / "package"
        exported = export_dynamic_state(run, self.snapshot_path, package)
        self.assertTrue(exported["same_tensor_parity"])
        self.assertEqual(exported["provider"], "CPUExecutionProvider")
        self.assertEqual(exported["validation_frames_compared"], 3)
        loader = DynamicModel(package, client_version="18.647.2")
        frame = next(row for row in self.snapshot["windows"] if row["split"] == "val")["frames"][0]
        tensor = roi_tensor(self.snapshot_path, frame)[None]
        seen = []
        original = loader.session

        class Spy:
            def run(self, outputs, inputs):
                seen.append(inputs[loader.input_name])
                return original.run(outputs, inputs)

        loader.session = Spy()
        import cv2
        image = cv2.imdecode(np.fromfile(self.snapshot_path / frame["image"], dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        state = DynamicFrame(frame["session_id"], frame["camera_epoch"], frame["frame_id"],
                             frame["captured_at_monotonic"], frame["client_version"],
                             image.shape[1], image.shape[0], True, True)
        loader.infer(image, tuple(frame["roi_bbox"]), state,
                     target_id=frame["target_id"], offline_replay=True)
        np.testing.assert_array_equal(seen[0], tensor)
        # A different frozen split must fail before package creation or resume writes.
        modified = json.loads(self.config_path.read_text())
        modified["code_version"] = "changed"
        self.config_path.write_text(json.dumps(modified), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "differs"):
            train_dynamic_state(self.snapshot_path, self.config_path, run, device="cpu",
                                resume=run / "last.pt", max_epochs=1,
                                deadline_seconds=120, pretrained=None)
        self.assertEqual(verify_dynamic_snapshot(self.snapshot_path, kind="r_building")["snapshot_sha256"],
                         self.snapshot["snapshot_sha256"])

    @unittest.skipUnless(os.environ.get("AUTOCOC_DYNAMIC_INTEGRATION") == "1",
                         "isolated training environment only")
    def test_card_uses_separate_vocabulary_and_package_kind(self):
        rows = deepcopy(self.snapshot["windows"])
        for row in rows:
            row["kind"] = "r_card"
            row["action_kind"] = "skill_available"
            for index, frame in enumerate(row["frames"]):
                frame["state_label"] = ("unknown" if index == 0 else
                                        "available" if row["truth_event"] else "exhausted")
                frame["target_id"] = frame["state_target_id"] = "hero_skill:archer_queen"
                frame["hero_deployment_truth"] = {
                    "hero_id": "archer_queen", "card_identity_verified": True,
                    "deployed": bool(row["truth_event"] and index > 0),
                    "review_status": "reviewed", "review_method": "agent_visual",
                    "reviewer": "synthetic-fixture", "review_evidence": "synthetic card context only",
                    "source_frame_sha256": frame["parent_image_sha256"],
                    "roi_bbox": frame["roi_bbox"][:], "frame_id": frame["frame_id"],
                    "session_id": frame["session_id"], "camera_epoch": frame["camera_epoch"],
                    "captured_at_monotonic": frame["captured_at_monotonic"],
                    "client_version": frame["client_version"],
                }
        card_root = self.root / "card-snapshot"
        frozen = freeze_dynamic_snapshot(self.snapshot_path, rows, card_root,
                                         kind="r_card", prototype_only=True)
        self.assertEqual(frozen["metadata"]["classes"], list(CARD_STATES))
        card_config = config(frozen)
        card_config["kind"] = "r_card"
        card_config["classes"] = list(CARD_STATES)
        card_config["applicability"]["roi_kind"] = "r_card"
        card_config["applicability"]["requires_stable_camera"] = False
        card_config["applicability"]["requires_verified_zoom"] = False
        path = self.root / "card-config.json"
        path.write_text(json.dumps(card_config), encoding="utf-8")
        run = self.root / "card-run"
        report = train_dynamic_state(card_root, path, run, device="cpu", max_epochs=1,
                                     deadline_seconds=120, pretrained=None)
        self.assertEqual(report["kind"], "r_card")
        exported = export_dynamic_state(run, card_root, self.root / "card-package")
        self.assertTrue(exported["same_tensor_parity"])
        self.assertEqual(DynamicModel(self.root / "card-package",
                                      client_version="18.647.2").classes, list(CARD_STATES))


if __name__ == "__main__":
    unittest.main()
