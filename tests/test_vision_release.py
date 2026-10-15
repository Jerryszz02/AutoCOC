import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.validate_vision_release import DEFAULT_ACCEPTANCE, validate_release
from scripts.evaluate_model_package import select_frozen_rows
from training.vision.data import _pixel_hash, export, leakage_groups, sha256
from training.vision.snapshot import TrainingError, verify_snapshot


class VisionReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.acceptance = json.loads(DEFAULT_ACCEPTANCE.read_text(encoding="utf-8"))
        scene_classes = ["enemy_village", "battle", "settlement", "popup", "disconnected", "maintenance"]
        self.acceptance["scope"] = {"frozen": True, "client_versions": ["v1"],
                                    "scene_classes": scene_classes,
                                    "building_classes": ["town_hall"], "dynamic_required": False,
                                    "evaluation_protocol_sha256": "a" * 64,
                                    "frozen_at_utc": "2026-10-15T00:00:00+00:00"}
        thresholds = self.acceptance["thresholds"]
        for key in ("scene_test_samples_per_class", "scene_test_groups_per_class",
                    "unknown_test_samples", "unknown_test_groups", "building_test_instances_per_class",
                    "building_test_groups_per_class", "town_hall_instances", "town_hall_groups",
                    "parity_reviewed_images", "core_entry_test_groups", "windows_hot_model_images"):
            thresholds[key] = 1
        thresholds["building_ap50_min"] = .5
        thresholds["building_map50_95_min"] = .5
        self.scene_dir, self.building_dir = self.snapshots()
        self.scene_report = {
            "kind": "scene", "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "classes": scene_classes,
            "model_contract_sha256": "1" * 64,
            "metadata": self.metadata(self.scene_dir),
            "evaluation_scope": self.scope(self.scene_dir),
            "evaluation": {"kind": "scene", "failed": 0, "samples": 7,
                "coverage_known": 1., "unknown_truth_samples": 1, "unknown_truth_groups": 1,
                "unknown_actionable_false_accepts": 0,
                "confusion_matrix": {label: {label: 1} for label in scene_classes},
                "per_class": {label: {"tp": 1, "fn": 0, "f1": 1., "recall": 1., "truth_groups": 1}
                              for label in scene_classes}},
        }
        self.building_report = {
            "kind": "building", "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "classes": ["town_hall"],
            "model_contract_sha256": "2" * 64,
            "metadata": self.metadata(self.building_dir),
            "evaluation_scope": self.scope(self.building_dir),
            "evaluation": {"kind": "building", "protocol": "runtime_point", "failed": 0,
                "samples": 7, "per_class": {"town_hall": {"tp": 7, "fn": 0,
                    "precision": 1., "recall": 1., "truth_groups": 7}}},
            "independent_ap": {"full_ap_eligible": True, "failed": 0,
                "protocol": {"sha256": "a" * 64, "protocol_id": "test"},
                "mAP50": 1., "mAP50_95": 1.,
                "per_class": {"town_hall": {"ap50": 1., "ap50_95": 1.,
                    "center_error_p95_normalized": 0.}}
            },
        }
        self.parity = {"reviewed_images": 1, "scene_decision_agreement": 1.,
                       "model_sha256": {"scene": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "building": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
                       "max_building_precision_drop": 0., "max_building_recall_drop": 0.,
                       "box_error_p95_normalized": 0.}
        self.ap_stability = {"kind": "building", "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "model_contract_sha256": "2" * 64,
            "metadata": self.metadata(self.building_dir),
            "evaluation": {"samples": 1},
            "independent_ap": {"full_ap_eligible": True},
            "evaluation_scope": {**self.scope(self.building_dir), "split": "val", "test_image_count": 1},
            "evaluation_stability": {"passed": True, "max_ap_delta": 0.,
                "base_protocol_sha256": "a" * 64}}
        self.windows = {"platform": "Windows", "complete_chain": True,
                        "model_sha256": {"scene": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "building": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},
                        "hot_model_images": 1, "action_types": {"deploy": {
                            "p95_seconds": .2, "max_seconds": .3, "timeouts": 0,
                            "failures": 0, "rejections": 0, "stale_discards": 0}}}
        self.core_entry = {"test_layout_groups": 1, "target_recall": 1.,
                           "entry_set_hit": 1., "illegal_deployments": 0,
                           "building_model_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}

    def snapshots(self):
        source = self.root / "source"
        source.mkdir()
        samples, annotations = [], []
        labels = ["enemy_village", "enemy_village", "enemy_village", "battle", "settlement",
                  "popup", "disconnected", "maintenance", "unknown"]
        for index, label in enumerate(labels):
            sample_id = f"S{index + 1:03d}"
            image = np.random.default_rng(index + 19).integers(0, 256, (32, 40, 3), dtype=np.uint8)
            path = source / f"{sample_id}.png"
            path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
            samples.append({"schema_version": 1, "sample_id": sample_id, "image_path": path.name,
                "file_sha256": sha256(path), "pixel_sha256": _pixel_hash(image), "width": 40, "height": 32,
                "source_domain": "fixture", "source_ref": sample_id, "source_session": sample_id,
                "battle_id": sample_id, "layout_group_id": sample_id, "parent_sample_id": None,
                "client_version": "v1", "review_status": "reviewed", "review_method": "human",
                "reviewer": "fixture", "review_evidence": "fixture", "split": "unassigned"})
            annotations.append({"schema_version": 1, "sample_id": sample_id,
                "scene": {"label": label, "complete": True, "review_status": "reviewed",
                    "review_method": "human", "reviewer": "fixture", "review_evidence": "fixture"},
                "buildings": {"complete": True, "review_status": "reviewed", "review_method": "human",
                    "reviewer": "fixture", "review_evidence": "fixture", "objects": [{
                        "object_id": sample_id + "-th", "class_name": "town_hall", "bbox": [4, 6, 20, 22],
                        "ignore": False, "review_status": "reviewed"}]}})
        groups, _ = leakage_groups(source, samples)
        assignments = {row["sample_id"]: "train" if i == 0 else "val" if i == 1 else "test"
                       for i, row in enumerate(samples)}
        split_base = {"schema_version": 1, "prototype_only": False,
            "sample_assignments": assignments, "sample_groups": groups,
            "group_assignments": {groups[sample_id]: assigned for sample_id, assigned in assignments.items()}}
        scene_dir, building_dir = self.root / "scene", self.root / "building"
        export(source, samples, annotations, {**split_base, "task": "scene"}, scene_dir,
               task="scene", data_version="v001")
        export(source, samples, annotations, {**split_base, "task": "building"}, building_dir,
               task="building", data_version="v001", classes=["town_hall"])
        return scene_dir, building_dir

    def metadata(self, root):
        verified = verify_snapshot(root, root.name)
        return {"data_version": "v001", "model_contract_sha256": "1" * 64 if root.name == "scene" else "2" * 64,
            "validation": {"prototype_only": False,
            "snapshot_sha256": verified["snapshot_sha256"], "split_sha256": verified["split_sha256"]}}

    def scope(self, root):
        verified = verify_snapshot(root, root.name)
        return {"status": "frozen_split", "split": "test", "data_version": "v001",
                "dataset_sha256": verified["snapshot_sha256"],
                "split_sha256": verified["split_sha256"], "test_image_count": 7}

    def check(self, **overrides):
        args = {"acceptance": self.acceptance, "scene_report": self.scene_report,
                "building_report": self.building_report, "ap_stability": self.ap_stability,
                "parity": self.parity,
                "windows": self.windows, "core_entry": self.core_entry,
                "scene_dataset": self.scene_dir, "building_dataset": self.building_dir}
        args.update(overrides)
        return validate_release(**args)

    def test_complete_static_evidence_can_pass(self):
        report = self.check()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["evidence"]["dynamic_scope"], "not_requested_for_static_release")
        self.assertEqual(report["evidence"]["scene_model_sha256"], "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
        self.assertEqual(report["evidence"]["building_dataset_sha256"],
                         self.scope(self.building_dir)["dataset_sha256"])

    def test_missing_evidence_and_low_sample_counts_are_insufficient(self):
        self.assertEqual(self.check(parity=None)["status"], "INSUFFICIENT_DATA")
        self.assertEqual(self.check(ap_stability=None)["status"], "INSUFFICIENT_DATA")
        self.assertEqual(self.check(scene_dataset=None)["status"], "INSUFFICIENT_DATA")
        insufficient = copy.deepcopy(self.scene_report)
        insufficient["evaluation"]["per_class"]["battle"]["truth_groups"] = 0
        report = self.check(scene_report=insufficient)
        self.assertEqual(report["status"], "INSUFFICIENT_DATA")
        self.assertTrue(any("battle test groups" in item for item in report["missing"]))

    def test_measured_failure_is_fail(self):
        bad = copy.deepcopy(self.building_report)
        bad["evaluation"]["per_class"]["town_hall"]["recall"] = .5
        report = self.check(building_report=bad)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("town_hall recall" in item for item in report["failures"]))

    def test_prototype_hash_leakage_and_raw_ap_cannot_pass(self):
        bad = copy.deepcopy(self.building_report)
        bad["independent_ap"]["full_ap_eligible"] = False
        self.assertEqual(self.check(building_report=bad)["status"], "INSUFFICIENT_DATA")
        (self.scene_dir / "snapshot.json").write_text(json.dumps({"schema_version": 1, "task": "scene",
            "prototype_only": True, "client_versions": ["unknown"], "data_version": "v001"}))
        report = self.check()
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["evidence"]["scene_snapshot_verified"])

    def test_unverified_eval_scope_and_dynamic_requirement(self):
        scene = copy.deepcopy(self.scene_report)
        scene["evaluation_scope"] = {"status": "unverified"}
        self.assertEqual(self.check(scene_report=scene)["status"], "FAIL")
        self.acceptance["scope"]["dynamic_required"] = True
        self.assertEqual(self.check()["status"], "INSUFFICIENT_DATA")

    def test_train_image_cannot_be_substituted_into_test_list(self):
        train_file, test_file = self.scene_dir / "train.txt", self.scene_dir / "test.txt"
        original_train = train_file.read_text(encoding="utf-8").splitlines()
        original_test = test_file.read_text(encoding="utf-8").splitlines()
        train_file.write_text(original_test[0] + "\n", encoding="utf-8")
        test_file.write_text("\n".join([original_train[0], *original_test[1:]]) + "\n", encoding="utf-8")
        sums = self.scene_dir / "SHA256SUMS"
        entries = []
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            if relative in {"train.txt", "test.txt"}:
                digest = sha256(self.scene_dir / relative)
            entries.append(f"{digest}  {relative}")
        sums.write_text("\n".join(entries) + "\n", encoding="utf-8")
        with self.assertRaises(TrainingError):
            select_frozen_rows(self.scene_dir, "scene", "test", self.scene_dir / "reviewed-scenes.jsonl")
        report = self.check()
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["evidence"]["scene_snapshot_verified"])

    def test_model_validation_absent_or_conflicting(self):
        missing = copy.deepcopy(self.scene_report)
        del missing["metadata"]["validation"]["snapshot_sha256"]
        report = self.check(scene_report=missing)
        self.assertEqual(report["status"], "INSUFFICIENT_DATA")
        self.assertTrue(any("training snapshot hash missing" in item for item in report["missing"]))
        conflict = copy.deepcopy(self.scene_report)
        conflict["metadata"]["validation"]["prototype_only"] = True
        conflict["metadata"]["validation"]["snapshot_sha256"] = "0" * 64
        report = self.check(scene_report=conflict)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("prototype_only mismatch" in item for item in report["failures"]))

    def test_ap_quality_limits_require_freezing_then_enforce(self):
        self.acceptance["thresholds"]["building_ap50_min"] = None
        self.assertEqual(self.check()["status"], "INSUFFICIENT_DATA")
        self.acceptance["thresholds"]["building_ap50_min"] = .8
        bad = copy.deepcopy(self.building_report)
        bad["independent_ap"]["per_class"]["town_hall"]["ap50"] = .7
        report = self.check(building_report=bad)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("town_hall AP50" in item for item in report["failures"]))

    def test_model_contract_report_metadata_conflict_fails(self):
        scene = copy.deepcopy(self.scene_report)
        scene["metadata"]["model_contract_sha256"] = "0" * 64
        result = self.check(scene_report=scene)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("scene model contract metadata" in issue for issue in result["failures"]))


if __name__ == "__main__":
    unittest.main()

