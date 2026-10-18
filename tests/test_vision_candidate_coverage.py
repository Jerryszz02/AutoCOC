from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.vision_candidate_coverage import main as candidate_coverage_main
from training.vision.candidate_coverage import candidate_coverage_report
from training.vision.coverage import coverage_report
from training.vision.data import (DatasetError, _pixel_hash, leakage_groups, load_taxonomy,
                                  sha256, split, write_jsonl)


class CandidateCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "images"
        self.root.mkdir()
        self.manifest: list[dict] = []
        self.annotations: list[dict] = []
        self.classes = ["town_hall", "air_defense"]
        self.taxonomy = load_taxonomy()
        for index in range(5):
            sample_id = f"C{index + 1:03d}"
            image = np.random.default_rng(index + 91).integers(
                0, 256, (48, 64, 3), dtype=np.uint8)
            path = self.root / f"{sample_id}.png"
            path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
            self.manifest.append({
                "schema_version": 1, "sample_id": sample_id, "image_path": path.name,
                "file_sha256": sha256(path), "pixel_sha256": _pixel_hash(image),
                "width": 64, "height": 48, "source_domain": "local",
                "source_ref": f"frame-{index}", "source_session": f"session-{index}",
                "battle_id": f"battle-{index}", "layout_group_id": f"layout-{index}",
                "parent_sample_id": None, "client_version": "18.647.2",
                "review_status": "reviewed", "reviewer": "fixture-agent",
                "review_evidence": f"candidate source {index}",
                "review_method": "agent_visual", "split": "unassigned",
            })
            buildings = {
                "complete": False,
                "complete_classes": list(self.classes),
                "review_status": "reviewed", "reviewer": "fixture-agent",
                "review_evidence": f"complete scope {index}",
                "review_method": "agent_visual",
                "objects": [{
                    "object_id": f"{sample_id}-th", "class_name": "town_hall",
                    "bbox": [5, 6, 27, 28], "ignore": False, "review_status": "reviewed",
                }],
            }
            if index == 0:
                buildings["objects"].append({
                    "object_id": f"{sample_id}-ad", "class_name": "air_defense",
                    "bbox": [32, 7, 48, 23], "ignore": False, "review_status": "reviewed",
                })
            self.annotations.append({
                "schema_version": 1, "sample_id": sample_id,
                "scene": {"label": "enemy_village", "complete": True,
                          "review_status": "reviewed", "reviewer": "fixture-agent",
                          "review_evidence": f"scene candidate {index}",
                          "review_method": "agent_visual"},
                "buildings": buildings,
            })

    def _report(self, manifest: list[dict] | None = None,
                annotations: list[dict] | None = None) -> dict:
        return candidate_coverage_report(
            self.root, manifest if manifest is not None else self.manifest,
            annotations if annotations is not None else self.annotations,
            task="building", classes=self.classes, taxonomy=self.taxonomy, seed=42,
        )

    def test_rare_class_and_agent_visual_qualification_are_separate(self) -> None:
        report = self._report()
        self.assertEqual(report["candidate_inventory"]["definite_independent_groups"], 5)
        self.assertEqual(report["classes"]["town_hall"]["candidate_positives"]["independent_groups"], 5)
        self.assertEqual(report["classes"]["town_hall"]["prototype_eligible_positives"][
            "missing_independent_groups_to_split_minimum"], 0)
        ad = report["classes"]["air_defense"]
        self.assertEqual(ad["candidate_positives"]["positive_images"], 1)
        self.assertEqual(ad["candidate_positives"]["independent_groups"], 1)
        self.assertEqual(ad["reviewed_complete"]["images"], 5)
        self.assertIn("not formal eligibility", ad["reviewed_complete"]["meaning"])
        self.assertEqual(ad["prototype_eligible_positives"][
            "missing_independent_groups_to_split_minimum"], 2)
        self.assertEqual(ad["formal_precondition_positives"][
            "missing_independent_groups_to_split_minimum"], 3)
        self.assertEqual(report["qualification"]["prototype"]["eligible_samples"], 5)
        self.assertEqual(report["qualification"]["formal_preconditions"]["eligible_samples"], 0)
        forecast = report["qualification"]["prototype"]["forecast"]
        self.assertEqual(report["qualification"]["prototype"]["status"],
                         "available_with_class_gaps")
        self.assertEqual(len(forecast["classes_missing_positive_groups"]["air_defense"]), 2)
        self.assertFalse(report["qualification"]["prototype"]["formal_eligibility_claim"])
        self.assertFalse(report["qualification"]["formal_preconditions"]["formal_eligibility_claim"])
        self.assertFalse(report["formal_success"])
        self.assertFalse(report["snapshot_created"])
        self.assertFalse(report["split_file_written"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["heldout_test_accessed"])

    def test_class_completeness_is_not_inferred_from_an_empty_box_list(self) -> None:
        annotations = copy.deepcopy(self.annotations)
        annotations[4]["buildings"]["complete_classes"] = ["town_hall"]
        report = self._report(annotations=annotations)
        ad = report["classes"]["air_defense"]
        self.assertEqual(ad["candidate_positives"]["positive_images"], 1)
        self.assertEqual(ad["reviewed_complete"]["images"], 4)
        self.assertEqual(report["qualification"]["prototype"]["eligible_samples"], 4)
        self.assertEqual(report["qualification"]["prototype"]["eligible_independent_groups"], 4)

    def test_preview_matches_existing_group_split_without_dividing_groups(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest[1]["source_session"] = manifest[0]["source_session"]
        manifest[1]["battle_id"] = manifest[0]["battle_id"]
        manifest[1]["layout_group_id"] = manifest[0]["layout_group_id"]
        existing = split(self.root, manifest, self.annotations, task="building",
                         seed=42, prototype_only=True, classes=self.classes,
                         taxonomy=self.taxonomy)
        report = self._report(manifest=manifest)
        forecast = report["qualification"]["prototype"]["forecast"]
        self.assertEqual(forecast["algorithm"], existing["algorithm"])
        self.assertEqual(forecast["images"], existing["counts"])
        self.assertEqual(sum(forecast["independent_groups"].values()), existing["group_count"])
        self.assertEqual(existing["sample_assignments"]["C001"], existing["sample_assignments"]["C002"])
        for split_name in ("train", "val", "test"):
            members = {sample_id for sample_id, assigned in existing["sample_assignments"].items()
                       if assigned == split_name}
            self.assertEqual(forecast["by_split"][split_name]["images"], len(members))
            self.assertEqual(
                forecast["by_split"][split_name]["independent_groups"],
                len({existing["sample_groups"][sample_id] for sample_id in members}),
            )
        reordered = self._report(manifest=list(reversed(manifest)),
                                 annotations=list(reversed(self.annotations)))
        self.assertEqual(reordered["qualification"]["prototype"]["forecast"], forecast)

    def test_custom_taxonomy_is_used_by_both_split_forecasts(self) -> None:
        taxonomy = copy.deepcopy(self.taxonomy)
        taxonomy["building_classes"].append({"id": 20, "name": "custom_defense"})
        manifest = copy.deepcopy(self.manifest)
        annotations = copy.deepcopy(self.annotations)
        for row, annotation in zip(manifest, annotations):
            row["review_method"] = "human"
            annotation["buildings"].update(
                complete=False, complete_classes=["custom_defense"], review_method="human")
            if annotation["buildings"]["objects"]:
                for obj in annotation["buildings"]["objects"]:
                    obj["review_status"] = "reviewed"
            if row["sample_id"] == "C001":
                annotation["buildings"]["objects"].append({
                    "object_id": "C001-custom", "class_name": "custom_defense",
                    "bbox": [49, 25, 61, 40], "ignore": False, "review_status": "reviewed",
                })
        report = candidate_coverage_report(
            self.root, manifest, annotations, task="building", classes=["custom_defense"],
            taxonomy=taxonomy, seed=42,
        )
        for qualification in ("prototype", "formal_preconditions"):
            forecast = report["qualification"][qualification]["forecast"]
            expected = split(self.root, manifest, annotations, task="building", seed=42,
                             prototype_only=qualification == "prototype",
                             classes=["custom_defense"], taxonomy=taxonomy)
            self.assertIsNotNone(forecast)
            self.assertEqual(forecast["images"], expected["counts"])
            self.assertEqual(forecast["algorithm"], expected["algorithm"])

    def test_near_duplicate_blocks_preview_until_source_bound_review(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        for index, shade in ((0, 100), (1, 101)):
            image = np.full((48, 64, 3), shade, dtype=np.uint8)
            path = self.root / manifest[index]["image_path"]
            path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
            manifest[index]["file_sha256"] = sha256(path)
            manifest[index]["pixel_sha256"] = _pixel_hash(image)
        blocked = self._report(manifest=manifest)
        self.assertTrue(blocked["near_duplicate_candidates"])
        pair = blocked["near_duplicate_candidates"][0]
        self.assertEqual(blocked["qualification"]["prototype"]["status"],
                         "blocked_unresolved_near_duplicates")
        self.assertIn(pair["sample_ids"],
                      blocked["qualification"]["prototype"]["unresolved_near_duplicate_pairs"])

        by_id = {row["sample_id"]: row for row in manifest}
        reviews = [{
            "sample_ids": item["sample_ids"],
            "dhash_distance": item["dhash_distance"],
            "disposition": "distinct",
            "source_file_sha256": {
                sample_id: by_id[sample_id]["file_sha256"] for sample_id in item["sample_ids"]
            },
            "reviewer": "fixture-agent",
            "review_evidence": "synthetic distinct-pair fixture",
            "review_method": "agent_visual",
        } for item in blocked["near_duplicate_candidates"]]
        reviewed = candidate_coverage_report(
            self.root, manifest, self.annotations, task="building", classes=self.classes,
            taxonomy=self.taxonomy, seed=42, near_duplicate_reviews=reviews,
        )
        self.assertIn(reviewed["qualification"]["prototype"]["status"],
                      {"available", "available_with_class_gaps"})
        self.assertEqual(reviewed["qualification"]["formal_preconditions"]["eligible_samples"], 0)

    def test_bad_review_is_rejected_even_when_no_samples_are_eligible(self) -> None:
        annotations = copy.deepcopy(self.annotations)
        for item in annotations:
            item["buildings"]["review_status"] = "pending"
        stale_review = [{"sample_ids": ["C900", "C901"]}]
        with self.assertRaisesRegex(DatasetError, "Stale near-duplicate review pair"):
            candidate_coverage_report(
                self.root, self.manifest, annotations, task="building", classes=self.classes,
                taxonomy=self.taxonomy, near_duplicate_reviews=stale_review,
            )

        manifest_path = self.base / "low-manifest.jsonl"
        annotations_path = self.base / "low-annotations.jsonl"
        review_path = self.base / "bad-review.json"
        output = self.base / "low-report.json"
        write_jsonl(manifest_path, self.manifest)
        write_jsonl(annotations_path, annotations)
        review_path.write_text(json.dumps(stale_review), encoding="utf-8")
        args = ["--root", str(self.root), "--manifest", str(manifest_path),
                "--annotations", str(annotations_path), "--task", "building",
                "--classes", *self.classes, "--near-duplicate-review", str(review_path),
                "--output", str(output)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(candidate_coverage_main(args), 2)
        self.assertFalse(output.exists())

    def test_assigned_and_frozen_inputs_are_rejected(self) -> None:
        assigned = copy.deepcopy(self.manifest)
        assigned[0]["split"] = "test"
        with self.assertRaisesRegex(DatasetError, "unassigned"):
            self._report(manifest=assigned)
        frozen_fields = copy.deepcopy(self.manifest)
        frozen_fields[0]["snapshot_split"] = "test"
        with self.assertRaisesRegex(DatasetError, "frozen manifest fields"):
            self._report(manifest=frozen_fields)
        marker = self.root / "snapshot.json"
        marker.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "frozen/assigned"):
            self._report()

    def test_cli_writes_new_report_only_and_does_not_change_inputs(self) -> None:
        manifest_path = self.base / "candidate-manifest.jsonl"
        annotations_path = self.base / "candidate-annotations.jsonl"
        write_jsonl(manifest_path, self.manifest)
        write_jsonl(annotations_path, self.annotations)
        image_hashes = {path.name: sha256(path) for path in self.root.glob("*.png")}
        manifest_hash, annotations_hash = sha256(manifest_path), sha256(annotations_path)
        output = self.base / "reports" / "coverage.json"
        args = [
            "--root", str(self.root), "--manifest", str(manifest_path),
            "--annotations", str(annotations_path), "--task", "building",
            "--classes", *self.classes, "--seed", "42", "--output", str(output),
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(candidate_coverage_main(args), 0)
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["source_files"]["manifest"]["sha256"], manifest_hash)
        self.assertEqual(saved["source_files"]["annotations"]["sha256"], annotations_hash)
        self.assertEqual(sha256(manifest_path), manifest_hash)
        self.assertEqual(sha256(annotations_path), annotations_hash)
        self.assertEqual({path.name: sha256(path) for path in self.root.glob("*.png")}, image_hashes)
        original_output_hash = sha256(output)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(candidate_coverage_main(args), 2)
        self.assertEqual(sha256(output), original_output_hash)

    def test_cli_review_is_protected_as_input_and_input_hashes_must_stay_stable(self) -> None:
        manifest_path = self.base / "candidate-manifest.jsonl"
        annotations_path = self.base / "candidate-annotations.jsonl"
        review_path = self.base / "near-duplicate-review.json"
        review_path.write_text("[]", encoding="utf-8")
        write_jsonl(manifest_path, self.manifest)
        write_jsonl(annotations_path, self.annotations)
        base_args = ["--root", str(self.root), "--manifest", str(manifest_path),
                     "--annotations", str(annotations_path), "--task", "building",
                     "--classes", *self.classes, "--near-duplicate-review", str(review_path)]
        collision_args = [*base_args, "--output", str(review_path)]
        before_review = sha256(review_path)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(candidate_coverage_main(collision_args), 2)
        self.assertEqual(sha256(review_path), before_review)

        output = self.base / "unstable-input-report.json"
        original_sha = sha256
        calls = 0

        def change_manifest_hash(path: Path) -> str:
            nonlocal calls
            if Path(path).resolve() == manifest_path.resolve():
                calls += 1
                if calls == 2:
                    return "changed-during-report"
            return original_sha(path)

        args = [*base_args, "--output", str(output)]
        with patch("scripts.vision_candidate_coverage.sha256", side_effect=change_manifest_hash):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(candidate_coverage_main(args), 2)
        self.assertFalse(output.exists())

    def test_image_changed_during_grouping_rejects_report(self) -> None:
        original_groups = leakage_groups
        changed_path = self.root / self.manifest[0]["image_path"]

        def change_after_grouping(root: Path, manifest: list[dict], *,
                                  near_duplicates: bool = False) -> tuple[dict, list]:
            result = original_groups(root, manifest, near_duplicates=near_duplicates)
            different = np.random.default_rng(999).integers(
                0, 256, (48, 64, 3), dtype=np.uint8)
            changed_path.write_bytes(cv2.imencode(".png", different)[1].tobytes())
            return result

        with patch("training.vision.candidate_coverage.leakage_groups",
                   side_effect=change_after_grouping):
            with self.assertRaisesRegex(DatasetError,
                                        "Candidate images changed during report generation"):
                self._report()

    def test_cli_does_not_publish_report_when_image_changes_during_grouping(self) -> None:
        manifest_path = self.base / "candidate-manifest.jsonl"
        annotations_path = self.base / "candidate-annotations.jsonl"
        output = self.base / "coverage.json"
        write_jsonl(manifest_path, self.manifest)
        write_jsonl(annotations_path, self.annotations)
        changed_path = self.root / self.manifest[0]["image_path"]
        original_groups = leakage_groups

        def change_after_grouping(root: Path, manifest: list[dict], *,
                                  near_duplicates: bool = False) -> tuple[dict, list]:
            result = original_groups(root, manifest, near_duplicates=near_duplicates)
            different = np.random.default_rng(999).integers(
                0, 256, (48, 64, 3), dtype=np.uint8)
            changed_path.write_bytes(cv2.imencode(".png", different)[1].tobytes())
            return result

        args = ["--root", str(self.root), "--manifest", str(manifest_path),
                "--annotations", str(annotations_path), "--task", "building",
                "--classes", *self.classes, "--output", str(output)]
        stderr = io.StringIO()
        with patch("training.vision.candidate_coverage.leakage_groups",
                   side_effect=change_after_grouping):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(candidate_coverage_main(args), 2)
        self.assertIn("Candidate images changed during report generation", stderr.getvalue())
        self.assertFalse(output.exists())

    def test_cli_rejects_nonobject_taxonomy_with_clear_exit(self) -> None:
        manifest_path = self.base / "candidate-manifest.jsonl"
        annotations_path = self.base / "candidate-annotations.jsonl"
        taxonomy_path = self.base / "bad-taxonomy.json"
        output = self.base / "bad-taxonomy-report.json"
        write_jsonl(manifest_path, self.manifest)
        write_jsonl(annotations_path, self.annotations)
        args = ["--root", str(self.root), "--manifest", str(manifest_path),
                "--annotations", str(annotations_path), "--task", "building",
                "--classes", *self.classes, "--taxonomy", str(taxonomy_path),
                "--output", str(output)]
        for payload in ("[]", "null", '{"schema_version": 1, "building_classes": [1]}'):
            taxonomy_path.write_text(payload, encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                self.assertEqual(candidate_coverage_main(args), 2)
            self.assertIn("Taxonomy must be an object", stderr.getvalue())
            self.assertFalse(output.exists())


class FrozenCoverageTests(unittest.TestCase):
    def test_building_class_instances_count_only_matching_boxes(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            scene, building = root / "scene", root / "building"
            scene.mkdir()
            building.mkdir()
            (scene / "taxonomy.json").write_text(
                json.dumps({"scene_classes": [], "unknown_scene": "unknown"}), encoding="utf-8")
            (building / "class_mapping.json").write_text(
                json.dumps({"output_indices": {"town_hall": 0, "air_defense": 1}}),
                encoding="utf-8")
            rows = [
                {"sample_id": "one", "image": "images/one.png", "group_id": "one",
                 "boxes": [{"class": "town_hall"}, {"class": "town_hall"},
                           {"class": "air_defense"}]},
                {"sample_id": "two", "image": "images/two.png", "group_id": "two",
                 "boxes": [{"class": "air_defense"}]},
            ]
            sources = [{"sample_id": row["sample_id"], "source_session": row["sample_id"],
                        "layout_group_id": row["sample_id"], "review_method": "human",
                        "client_version": "18.647.2"} for row in rows]

            def read_fixture(path: Path) -> list[dict]:
                if path.parent == scene:
                    return []
                return sources if path.name == "reviewed-manifest.jsonl" else rows

            def verified(root: Path, task: str) -> dict:
                return {"paths": {"train": [], "val": [],
                                  "test": [] if task == "scene" else [row["image"] for row in rows]},
                        "snapshot_sha256": "snapshot", "split_sha256": "split",
                        "metadata": {"prototype_only": False, "review_policy": "human"}}

            acceptance = Path(__file__).resolve().parents[1] / "training/vision/acceptance-v1.json"
            with patch("training.vision.coverage.verify_snapshot", side_effect=verified), \
                 patch("training.vision.coverage.read_jsonl", side_effect=read_fixture):
                report = coverage_report(scene, building, acceptance_path=acceptance,
                                         current_client_version="18.647.2")
            classes = report["building"]["classes"]
            self.assertEqual(report["building"]["splits"]["test"]["instances"], 4)
            self.assertEqual(classes["town_hall"]["splits"]["test"]["instances"], 2)
            self.assertEqual(classes["air_defense"]["splits"]["test"]["instances"], 2)
            self.assertEqual(classes["town_hall"]["human_current_version"]["test"]["instances"], 2)
            self.assertEqual(classes["air_defense"]["formal_test_deficit"]["test_instances"], 48)


if __name__ == "__main__":
    unittest.main()
