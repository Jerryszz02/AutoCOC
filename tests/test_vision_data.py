import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from training.vision.data import (DatasetError, _pixel_hash, _verify_near_duplicate_reviews,
                                  export, image_path,
                                  import_annotations, import_legacy, inspect, leakage_groups, sha256,
                                  split, stage_sources, validate)


class VisionDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "素材"
        self.root.mkdir()
        self.samples, self.annotations = [], []
        for index in range(4):
            sample_id = f"S{index + 1:03d}"
            rng = np.random.default_rng(index + 1)
            image = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
            path = self.root / f"{sample_id}.png"
            path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
            self.samples.append({
                "schema_version": 1, "sample_id": sample_id, "image_path": path.name,
                "file_sha256": sha256(path), "pixel_sha256": _pixel_hash(image),
                "width": 40, "height": 32, "source_domain": "local",
                "source_ref": f"frame{index}", "source_session": f"session{index}",
                "battle_id": f"battle{index}", "layout_group_id": f"layout{index}",
                "parent_sample_id": None, "client_version": "current-test",
                "review_status": "reviewed", "reviewer": "fixture", "review_evidence": "fixture generation",
                "review_method": "human",
                "split": "unassigned",
            })
            self.annotations.append({
                "schema_version": 1, "sample_id": sample_id,
                "scene": {"label": "enemy_village", "complete": True, "review_status": "reviewed",
                          "reviewer": "fixture", "review_evidence": "fixture generation", "review_method": "human"},
                "buildings": {"complete": True, "review_status": "reviewed",
                              "reviewer": "fixture", "review_evidence": "fixture generation", "review_method": "human", "objects": [
                    {"object_id": f"{sample_id}-th", "class_name": "town_hall",
                     "bbox": [4, 6, 20, 22], "ignore": False, "review_status": "reviewed"}]},
            })

    def test_legacy_import_keeps_ids_and_guesses_pending(self) -> None:
        legacy = self.root / "legacy.jsonl"
        row = {"sample_id": "S001", "file": "S001.png", "copy_sha256": self.samples[0]["file_sha256"],
               "pixel_sha256": self.samples[0]["pixel_sha256"], "width": 40, "height": 32,
               "source_session": "session0", "game_version": None,
               "scene_candidate": "enemy_village", "captured_at_unix": 1.0}
        original = json.dumps(row) + "\n"
        legacy.write_text(original, encoding="utf-8")
        output = self.root / "manifest.jsonl"
        report = import_legacy(self.root, legacy, output)
        imported = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["imported"], 1)
        self.assertEqual(imported["sample_id"], "S001")
        self.assertEqual(imported["review_status"], "pending")
        self.assertIsNone(imported["client_version"])
        self.assertEqual(imported["legacy_scene_candidate"], "enemy_village")
        self.assertEqual(legacy.read_text(encoding="utf-8"), original)
        with self.assertRaises(DatasetError):
            import_legacy(self.root, legacy, output)

    def test_stage_sources_copies_verified_images_without_changing_groups(self) -> None:
        manifest = self.root / "source-manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in self.samples) + "\n", encoding="utf-8")
        output = Path(self.temp.name) / "stage"
        report = stage_sources([(self.root, manifest)], output)
        rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        self.assertEqual(report["samples"], 4)
        self.assertEqual(rows[0]["source_session"], self.samples[0]["source_session"])
        self.assertEqual(rows[0]["image_path"], "source-00/S001.png")
        self.assertTrue(validate(output, rows, self.annotations)["valid"])
        with self.assertRaisesRegex(DatasetError, "overwrite"):
            stage_sources([(self.root, manifest)], output)

    def test_validate_rejects_tamper_unknown_and_partial_labels(self) -> None:
        self.assertTrue(validate(self.root, self.samples, self.annotations)["valid"])
        broken = copy.deepcopy(self.annotations)
        broken[0]["buildings"]["objects"][0]["bbox"] = [4, 6, 41, 22]
        broken[1]["scene"]["label"] = "invented"
        broken[2]["buildings"]["complete"] = False
        report = validate(self.root, self.samples, broken)
        self.assertFalse(report["valid"])
        self.assertTrue(any("invalid bbox" in error for error in report["errors"]))
        self.assertTrue(any("unknown scene" in error for error in report["errors"]))
        self.assertTrue(any("incomplete" in error for error in report["errors"]))
        (self.root / "S001.png").write_bytes(b"changed")
        self.assertTrue(any("SHA-256 mismatch" in error for error in validate(self.root, self.samples, self.annotations)["errors"]))

    def test_transitive_groups_and_unknown_version_block_split(self) -> None:
        linked = copy.deepcopy(self.samples)
        linked[1]["layout_group_id"] = linked[0]["layout_group_id"]
        linked[2]["parent_sample_id"] = linked[1]["sample_id"]
        groups, _ = leakage_groups(self.root, linked)
        self.assertEqual(groups["S001"], groups["S003"])
        linked[3]["leakage_group_id"] = "human-reviewed-same-layout"
        linked[0]["leakage_group_id"] = "human-reviewed-same-layout"
        groups, _ = leakage_groups(self.root, linked)
        self.assertEqual(groups["S001"], groups["S004"])
        with self.assertRaisesRegex(DatasetError, "independent groups"):
            split(self.root, linked, self.annotations, task="building", near_duplicates=False)
        unknown = copy.deepcopy(self.samples)
        unknown[0]["client_version"] = None
        result = split(self.root, unknown, self.annotations, task="scene", near_duplicates=False)
        self.assertNotIn("S001", result["sample_assignments"])

    def test_export_portable_yolo_and_evaluation_ground_truth(self) -> None:
        split_doc = split(self.root, self.samples, self.annotations, task="building")
        output = Path(self.temp.name) / "snapshot"
        report = export(self.root, self.samples, self.annotations, split_doc, output,
                        task="building", data_version="v001", classes=["town_hall"])
        self.assertEqual(report["sample_count"], 4)
        self.assertTrue((output / "SHA256SUMS").is_file())
        self.assertEqual(json.loads((output / "class_mapping.json").read_text())["output_indices"], {"town_hall": 0})
        label = (output / "labels" / "S001.txt").read_text().strip().split()
        self.assertEqual(label[0], "0")
        self.assertAlmostEqual(float(label[1]), .3)
        self.assertAlmostEqual(float(label[2]), .4375)
        reviewed = json.loads((output / "reviewed-buildings.jsonl").read_text().splitlines()[0])
        self.assertEqual(reviewed["boxes"][0]["bbox"], [4, 6, 20, 22])
        exported_source = json.loads((output / "reviewed-manifest.jsonl").read_text().splitlines()[0])
        self.assertEqual(exported_source["review_method"], "human")
        self.assertEqual(exported_source["snapshot_image_path"], "images/S001.png")
        self.assertIn(exported_source["snapshot_split"], {"train", "val", "test"})
        with self.assertRaisesRegex(DatasetError, "overwrite"):
            export(self.root, self.samples, self.annotations, split_doc, output,
                   task="building", data_version="v001", classes=["town_hall"])

    def test_split_balances_large_groups_without_breaking_leakage(self) -> None:
        samples, labels = [], []
        for group in range(19):
            for frame in range(50 if group == 0 else 1):
                sample_id = f"group-{group}-frame-{frame}"
                row = copy.deepcopy(self.samples[0])
                row.update(sample_id=sample_id, file_sha256=f"file-{sample_id}",
                           pixel_sha256=f"pixels-{sample_id}", source_ref=sample_id,
                           source_session=f"session-{group}", battle_id=f"battle-{group}",
                           layout_group_id=f"layout-{group}")
                samples.append(row)
                label = copy.deepcopy(self.annotations[0])
                label["sample_id"] = sample_id
                labels.append(label)
        result = split(self.root, samples, labels, task="scene", near_duplicates=False)
        repeated = split(self.root, list(reversed(samples)), labels, task="scene", near_duplicates=False)
        self.assertEqual(result["sample_assignments"], repeated["sample_assignments"])
        self.assertEqual({result["sample_assignments"][f"group-0-frame-{i}"] for i in range(50)}, {"train"})
        self.assertEqual(result["counts"]["train"], 50)
        self.assertEqual(set(result["counts"]), {"train", "val", "test"})
        self.assertEqual(result["algorithm"], "group_size_greedy_v2")

    def test_ignore_and_group_crossing_block_export(self) -> None:
        split_doc = split(self.root, self.samples, self.annotations, task="building")
        labels = copy.deepcopy(self.annotations)
        labels[0]["buildings"]["objects"][0]["ignore"] = True
        labels[0]["buildings"]["objects"][0]["ignore_reason"] = "uncertain"
        with self.assertRaisesRegex(DatasetError, "not eligible"):
            export(self.root, self.samples, labels, split_doc, Path(self.temp.name) / "bad",
                   task="building", data_version="v001", classes=["town_hall"])
        split_doc["sample_groups"]["S001"] = "wrong"
        with self.assertRaisesRegex(DatasetError, "grouping changed"):
            export(self.root, self.samples, self.annotations, split_doc, Path(self.temp.name) / "bad2",
                   task="building", data_version="v001", classes=["town_hall"])

    def test_prototype_only_version_gate_and_annotation_import(self) -> None:
        unknown = copy.deepcopy(self.samples)
        for row in unknown:
            row["client_version"] = None
            row["review_method"] = "agent_visual"
        agent_labels = copy.deepcopy(self.annotations)
        for row in agent_labels:
            row["scene"]["review_method"] = "agent_visual"
        with self.assertRaisesRegex(DatasetError, "No reviewed"):
            split(self.root, unknown, agent_labels, task="scene", near_duplicates=False)
        split_doc = split(self.root, unknown, agent_labels, task="scene", prototype_only=True)
        output = Path(self.temp.name) / "prototype"
        with self.assertRaisesRegex(DatasetError, "mode mismatch"):
            export(self.root, unknown, agent_labels, split_doc, output,
                   task="scene", data_version="smoke-v1")
        export(self.root, unknown, agent_labels, split_doc, output,
               task="scene", data_version="smoke-v1", prototype_only=True)
        snapshot = json.loads((output / "snapshot.json").read_text())
        self.assertTrue(snapshot["prototype_only"])
        self.assertEqual(snapshot["client_versions"], ["unknown"])
        self.assertEqual(snapshot["review_methods"], ["agent_visual"])
        card = (output / "DATASET_CARD.md").read_text(encoding="utf-8")
        self.assertIn("Unknown client versions may be present", card)
        self.assertNotIn("Unknown client versions are excluded", card)
        base = [{"schema_version": 1, "sample_id": "S001", "scene": self.annotations[0]["scene"]}]
        incoming = [{"schema_version": 1, "sample_id": "S001", "buildings": self.annotations[0]["buildings"]}]
        merged = import_annotations(base, incoming, self.samples)
        self.assertEqual(merged[0]["scene"]["reviewer"], "fixture")
        self.assertEqual(merged[0]["buildings"]["review_evidence"], "fixture generation")
        conflict = copy.deepcopy(incoming)
        conflict[0]["buildings"]["objects"][0]["bbox"] = [1, 1, 2, 2]
        with self.assertRaisesRegex(DatasetError, "conflicting"):
            import_annotations(merged, conflict, self.samples)

    def test_building_prototype_uses_session_when_layout_unreviewed(self) -> None:
        samples = copy.deepcopy(self.samples)
        labels = copy.deepcopy(self.annotations)
        for row, label in zip(samples, labels):
            row["layout_group_id"] = None
            row["client_version"] = "unknown"
            row["review_method"] = "agent_visual"
            label["buildings"]["review_method"] = "agent_visual"
        with self.assertRaisesRegex(DatasetError, "No reviewed"):
            split(self.root, samples, labels, task="building", near_duplicates=False)
        split_doc = split(self.root, samples, labels, task="building", prototype_only=True)
        self.assertEqual(split_doc["group_count"], 4)
        self.assertEqual(len(split_doc["sample_assignments"]), 4)

        provisional = copy.deepcopy(self.samples)
        for row in provisional:
            row["layout_group_id"] = f"provisional:{row['source_session']}"
        with self.assertRaisesRegex(DatasetError, "No reviewed"):
            split(self.root, provisional, self.annotations, task="building", near_duplicates=False)

    def test_near_duplicate_review_requires_source_bound_visual_verdict(self) -> None:
        near = [{"sample_ids": ["S001", "S002"], "dhash_distance": 4}]
        pair = {"sample_ids": ["S001", "S002"], "dhash_distance": 4,
                "source_file_sha256": {row["sample_id"]: row["file_sha256"]
                                       for row in self.samples[:2]},
                "disposition": "distinct", "reviewer": "fixture",
                "review_method": "agent_visual", "review_evidence": "Visible layouts differ"}
        with self.assertRaisesRegex(DatasetError, "need review"):
            _verify_near_duplicate_reviews(near, self.samples, [], prototype_only=True)
        with self.assertRaisesRegex(DatasetError, "source-bound"):
            _verify_near_duplicate_reviews(near, self.samples,
                                           [{**pair, "source_file_sha256": {}}], prototype_only=True)
        with self.assertRaisesRegex(DatasetError, "source-bound"):
            _verify_near_duplicate_reviews(near, self.samples, [pair], prototype_only=False)
        self.assertEqual(_verify_near_duplicate_reviews(near, self.samples, [pair],
                                                        prototype_only=True), [pair])

    def test_path_traversal_rejected(self) -> None:
        with self.assertRaises(DatasetError):
            image_path(self.root, "../outside.png")


if __name__ == "__main__":
    unittest.main()
