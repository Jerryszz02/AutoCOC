import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autococ.model_vision import _ONNXModel
from training.vision.train import (SCENE_PREPROCESS, TrainingError, _config, _scene_rows,
                                   _advance_scene_patience, _set_scene_training_phase,
                                   _stage_building_inputs,
                                   _validate_scene_mode,
                                   prepare_bgr, train_building, train_scene, verify_snapshot)


class TrainingContractTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("AUTOCOC_TRAINING_INTEGRATION") == "1", "training environment only")
    def test_head_freezes_feature_bn_buffers_but_updates_classifier_then_finetune_updates_features(self):
        import torch

        class TinyScene(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1),
                                                    torch.nn.BatchNorm2d(4),
                                                    torch.nn.AdaptiveAvgPool2d(1))
                self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))

            def forward(self, images):
                return self.classifier(self.features(images))

        torch.manual_seed(7)
        model = TinyScene()
        images = torch.randn(8, 3, 8, 8) + 3
        labels = torch.tensor([0, 1] * 4)
        bn = model.features[1]
        before_mean = bn.running_mean.clone()
        before_var = bn.running_var.clone()
        before_count = bn.num_batches_tracked.clone()
        before_features = model.features[0].weight.detach().clone()
        before_classifier = model.classifier[1].weight.detach().clone()

        _set_scene_training_phase(model, "head")
        self.assertFalse(model.features.training)
        self.assertTrue(model.classifier.training)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
        self.assertTrue(torch.equal(bn.running_mean, before_mean))
        self.assertTrue(torch.equal(bn.running_var, before_var))
        self.assertTrue(torch.equal(bn.num_batches_tracked, before_count))
        self.assertTrue(torch.equal(model.features[0].weight, before_features))
        self.assertFalse(torch.equal(model.classifier[1].weight, before_classifier))

        _set_scene_training_phase(model, "finetune")
        self.assertTrue(model.features.training)
        self.assertTrue(all(p.requires_grad for p in model.features.parameters()))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(images), labels).backward()
        optimizer.step()
        self.assertFalse(torch.equal(bn.running_mean, before_mean))
        self.assertFalse(torch.equal(bn.running_var, before_var))
        self.assertGreater(int(bn.num_batches_tracked), int(before_count))
        self.assertFalse(torch.equal(model.features[0].weight, before_features))

    def test_head_stalls_do_not_consume_finetune_patience_and_resume_does_not_reset_twice(self):
        best, bad = (.8, .8), 2  # head's last checkpoint, after two stagnant epochs
        for epoch in range(5, 12):
            best, bad, improved = _advance_scene_patience((.1, .1), best, bad,
                                                          epoch=epoch, head_epochs=5)
            self.assertFalse(improved)
            self.assertEqual(best, (.8, .8))
            self.assertEqual(bad, epoch - 4)
            if epoch == 5:
                boundary_checkpoint = {"epoch": epoch, "phase": "finetune", "bad_epochs": bad,
                                       "best_score": best}
                best, bad = boundary_checkpoint["best_score"], boundary_checkpoint["bad_epochs"]
            if epoch == 8:
                middle_checkpoint = {"epoch": epoch, "phase": "finetune", "bad_epochs": bad,
                                     "best_score": best}
                best, bad = middle_checkpoint["best_score"], middle_checkpoint["bad_epochs"]
            self.assertEqual(bad >= 7, epoch == 11)

    @unittest.skipUnless(os.environ.get("AUTOCOC_TRAINING_INTEGRATION") == "1", "training environment only")
    def test_scene_resume_across_head_boundary_and_mid_finetune_keeps_seven_fine_misses(self):
        import torch

        class TinyScene(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1),
                                                    torch.nn.BatchNorm2d(4),
                                                    torch.nn.AdaptiveAvgPool2d(1))
                self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))

            def forward(self, images):
                return self.classifier(self.features(images))

        metadata = json.loads((self.root / "snapshot.json").read_text())
        metadata["prototype_only"] = True
        (self.root / "snapshot.json").write_text(json.dumps(metadata))
        split = json.loads((self.root / "split.json").read_text())
        split["prototype_only"] = True
        (self.root / "split.json").write_text(json.dumps(split))
        self.hash_snapshot()
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({
            "task": "scene", "data_version": "test", "client_versions": ["test-client"],
            "source_width_range": [40, 40], "source_height_range": [32, 32],
            "zoom": "minimum", "code_version": "test-code", "classes": ["battle", "village"],
            "output": {"format": "logits", "min_confidence": .5, "min_margin": .1},
            "prototype_only": True, "batch": 1}), encoding="utf-8")
        run = self.root / "run"
        scores = [(.5, .5), (.6, .6), (.7, .7)] + [(.1, .1)] * 9
        measured = []
        stop_after = [5]

        def metric(_truth, _prediction, _classes):
            score = scores[len(measured)]
            measured.append(score)
            return {"macro_f1": score[0], "minimum_recall": score[1]}

        def now():
            return 1000.0 if len(measured) >= stop_after[0] else 0.0

        with (patch("training.vision.train._scene_model", side_effect=lambda *_: (TinyScene(), None)),
              patch("training.vision.train.prepare_bgr", return_value=np.ones((3, 8, 8), dtype=np.float32)),
              patch("training.vision.train._metric", side_effect=metric),
              patch("training.vision.train.time.monotonic", side_effect=now)):
            for limit, expected_bad in ((5, 2), (6, 1), (9, 4), (12, 7)):
                stop_after[0] = limit
                result = train_scene(self.root, config_path, run, device="cpu", seed=42,
                                     resume=run / "last.pt" if run.exists() else None,
                                     prototype=True, development=True, max_epochs=20,
                                     deadline_seconds=120, pretrained=None)
                self.assertEqual(result["epochs_completed"], limit)
                last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(last["bad_epochs"], expected_bad)
                self.assertEqual(tuple(last["best_score"]), (.7, .7))
                self.assertEqual(last["phase"], "head" if limit == 5 else "finetune")
            self.assertEqual(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["epoch"], 2)
            recipe = json.loads((run / "training_config.json").read_text())
            self.assertEqual(recipe["head_features_mode"], "eval_frozen_parameters_and_buffers")
            self.assertEqual(recipe["finetune_patience_policy"], "reset_at_boundary_keep_global_best")
            old = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
            old_identity = {key: value for key, value in recipe.items()
                            if key not in {"deadline_seconds", "trainer_recipe_version",
                                           "head_features_mode", "finetune_features_mode",
                                           "finetune_patience", "finetune_patience_policy"}}
            old["signature"] = hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
            torch.save(old, run / "last.pt")
            with self.assertRaisesRegex(TrainingError, "Resume config or snapshot differs"):
                train_scene(self.root, config_path, run, device="cpu", seed=42,
                            resume=run / "last.pt", prototype=True, development=True,
                            max_epochs=20, deadline_seconds=120, pretrained=None)

    def test_development_scene_mode_remains_explicit_and_bounded(self):
        _validate_scene_mode(True, True, 20)
        with self.assertRaisesRegex(TrainingError, "Prototype smoke"):
            _validate_scene_mode(True, False, 20)
        with self.assertRaisesRegex(TrainingError, "prototype snapshot"):
            _validate_scene_mode(False, True, 20)
        with self.assertRaisesRegex(TrainingError, "epoch cap"):
            _validate_scene_mode(True, True, None)
        with self.assertRaisesRegex(TrainingError, "epoch cap"):
            _validate_scene_mode(True, True, 36)
        with self.assertRaisesRegex(TrainingError, "positive"):
            _validate_scene_mode(True, True, 0)

    def test_building_staging_preserves_frozen_inputs_and_rejects_modified_copy(self):
        (self.root / "labels").mkdir()
        (self.root / "labels" / "a.txt").write_text("0 0.5 0.5 0.3 0.3125\n")
        run = self.root / "run"
        paths = _stage_building_inputs(self.root, run, ["images/a.png"])
        self.assertEqual(paths, [run / "staged" / "images" / "a.png"])
        self.assertEqual((run / "staged" / "labels" / "a.txt").read_bytes(),
                         (self.root / "labels" / "a.txt").read_bytes())
        (run / "staged" / "labels" / "a.txt").write_text("changed")
        with self.assertRaisesRegex(TrainingError, "differs from snapshot"):
            _stage_building_inputs(self.root, run, ["images/a.png"])

    def test_building_development_requires_prototype_and_bounded_epochs(self):
        args = dict(device="cpu", seed=42, resume=None, deadline_seconds=60,
                    pretrained=None, development=True)
        with self.assertRaisesRegex(TrainingError, "prototype snapshot"):
            train_building(self.root, self.root / "unused.json", self.root / "run",
                           prototype=False, max_epochs=20, **args)
        with self.assertRaisesRegex(TrainingError, "1..100"):
            train_building(self.root, self.root / "unused.json", self.root / "run",
                           prototype=True, max_epochs=101, **args)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        image = np.zeros((32, 40, 3), dtype=np.uint8)
        image[..., 0] = 80
        for name in ("a", "b", "c"):
            (self.root / "images" / f"{name}.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
        for split, name in (("train", "a"), ("val", "b"), ("test", "c")):
            (self.root / f"{split}.txt").write_text(f"images/{name}.png\n")
        (self.root / "DATASET_CARD.md").write_text("Task: scene\n")
        (self.root / "split.json").write_text(json.dumps({"task": "scene", "sample_groups": {"a": "A", "b": "B", "c": "C"},
                                                    "sample_assignments": {"a": "train", "b": "val", "c": "test"},
                                                    "group_assignments": {"A": "train", "B": "val", "C": "test"}}))
        (self.root / "snapshot.json").write_text(json.dumps({"schema_version": 1, "task": "scene", "data_version": "test", "prototype_only": False, "client_versions": ["test-client"]}))
        (self.root / "taxonomy.json").write_text(json.dumps({"scene_classes": ["battle", "village"]}))
        rows = [{"image": f"images/{name}.png", "label": "battle", "sample_id": name, "group_id": name.upper()} for name in "abc"]
        (self.root / "reviewed-scenes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.root / "reviewed-manifest.jsonl").write_text("".join(json.dumps({
            "sample_id": name, "snapshot_image_path": f"images/{name}.png", "snapshot_split": split,
            "computed_leakage_group_id": name.upper(), "client_version": "test-client",
            "file_sha256": hashlib.sha256((self.root / "images" / f"{name}.png").read_bytes()).hexdigest(),
            "width": 40, "height": 32,
            "review_status": "reviewed", "review_method": "human", "reviewer": "fixture",
            "review_evidence": "synthetic fixture"}) + "\n"
            for split, name in (("train", "a"), ("val", "b"), ("test", "c"))))
        (self.root / "annotations").mkdir()
        for name in "abc":
            (self.root / "annotations" / f"{name}.jsonl").write_text(json.dumps({
                "sample_id": name,
                "scene": {"label": "battle", "complete": True, "review_status": "reviewed",
                          "review_method": "human", "reviewer": "fixture", "review_evidence": "synthetic fixture"},
                "buildings": {"complete": True, "review_status": "reviewed", "review_method": "human",
                              "reviewer": "fixture", "review_evidence": "synthetic fixture", "objects": [
                                  {"class_name": "town_hall", "bbox": [14, 11, 26, 21],
                                   "review_status": "reviewed", "ignore": False}]}}) + "\n")
        self.hash_snapshot()

    def hash_snapshot(self):
        lines = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.root).as_posix()}")
        (self.root / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    def test_frozen_snapshot_hash_and_group_gate(self):
        snapshot = verify_snapshot(self.root, "scene")
        self.assertEqual(len(_scene_rows(self.root, snapshot, ["battle"])["train"]), 1)
        rows = [json.loads(line) for line in (self.root / "reviewed-scenes.jsonl").read_text().splitlines()]
        rows[1]["group_id"] = "A"
        (self.root / "reviewed-scenes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(TrainingError, "hash mismatch"):
            verify_snapshot(self.root, "scene")
        self.hash_snapshot()
        with self.assertRaisesRegex(TrainingError, "split/group"):
            verify_snapshot(self.root, "scene")

    def test_preprocess_matches_runtime_letterbox_and_normalization(self):
        image = np.random.default_rng(12).integers(0, 256, (57, 103, 3), dtype=np.uint8)
        model = object.__new__(_ONNXModel)
        model.width, model.height = 320, 192
        model.resize, model.color = "letterbox", "RGB"
        model.scale, model.pad_value = SCENE_PREPROCESS["scale"], 114
        model.mean = np.asarray(SCENE_PREPROCESS["mean"], dtype=np.float32).reshape(1, 1, 3)
        model.std = np.asarray(SCENE_PREPROCESS["std"], dtype=np.float32).reshape(1, 1, 3)
        model.manifest = {"applicability": {"source_width_range": [1, 200], "source_height_range": [1, 200]}}
        actual, _ = model._prepare(image)
        np.testing.assert_array_equal(prepare_bgr(image, SCENE_PREPROCESS), actual[0])

    def test_prototype_snapshot_cannot_be_promoted_to_formal_training(self):
        metadata = json.loads((self.root / "snapshot.json").read_text())
        metadata["prototype_only"] = True
        (self.root / "snapshot.json").write_text(json.dumps(metadata))
        split = json.loads((self.root / "split.json").read_text())
        split["prototype_only"] = True
        (self.root / "split.json").write_text(json.dumps(split))
        self.hash_snapshot()
        snapshot = verify_snapshot(self.root, "scene")
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"task": "scene", "data_version": "test", "client_versions": ["test-client"],
                                      "source_width_range": [40, 40], "source_height_range": [32, 32],
                                      "zoom": "minimum", "code_version": "test-code", "classes": ["battle", "village"],
                                      "output": {"format": "logits", "min_confidence": .8, "min_margin": .1}}))
        with self.assertRaisesRegex(TrainingError, "Prototype snapshot"):
            _config(snapshot, "scene", config_path, prototype=False)
        self.assertTrue(_config(snapshot, "scene", config_path, prototype=True)["prototype_only"])

    def test_formal_snapshot_rejects_agent_only_annotation(self):
        path = self.root / "annotations" / "a.jsonl"
        annotation = json.loads(path.read_text())
        annotation["scene"]["review_method"] = "agent_visual"
        path.write_text(json.dumps(annotation))
        self.hash_snapshot()
        with self.assertRaisesRegex(TrainingError, "complete reviewed evidence"):
            verify_snapshot(self.root, "scene")

    def test_building_snapshot_rejects_rehashed_label_mismatch(self):
        (self.root / "DATASET_CARD.md").write_text("Task: building\n")
        metadata = json.loads((self.root / "snapshot.json").read_text())
        metadata["task"] = "building"
        (self.root / "snapshot.json").write_text(json.dumps(metadata))
        split = json.loads((self.root / "split.json").read_text())
        split["task"] = "building"
        (self.root / "split.json").write_text(json.dumps(split))
        (self.root / "class_mapping.json").write_text(json.dumps({
            "output_indices": {"town_hall": 0}, "stable_ids": {"town_hall": 1}}))
        (self.root / "labels").mkdir()
        for name in "abc":
            (self.root / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.3 0.3125\n")
        rows = [{"image": f"images/{name}.png", "sample_id": name, "group_id": name.upper(),
                 "boxes": [{"class": "town_hall", "bbox": [14, 11, 26, 21]}]} for name in "abc"]
        (self.root / "reviewed-buildings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.hash_snapshot()
        verify_snapshot(self.root, "building")
        (self.root / "labels" / "a.txt").write_text("0 0.6 0.5 0.3 0.3125\n")
        self.hash_snapshot()
        with self.assertRaisesRegex(TrainingError, "YOLO labels differ"):
            verify_snapshot(self.root, "building")

    @unittest.skipUnless(os.environ.get("AUTOCOC_TRAINING_INTEGRATION") == "1", "training environment only")
    def test_synthetic_scene_checkpoint_and_real_onnx_loader(self):
        # Technical integration fixture only; never evidence for T2 data smoke.
        import torch
        from training.vision.export import export_package
        metadata = json.loads((self.root / "snapshot.json").read_text())
        metadata["prototype_only"] = True
        (self.root / "snapshot.json").write_text(json.dumps(metadata))
        split = json.loads((self.root / "split.json").read_text())
        split["prototype_only"] = True
        (self.root / "split.json").write_text(json.dumps(split))
        outputs = TemporaryDirectory()
        self.addCleanup(outputs.cleanup)
        out_root = Path(outputs.name)
        config_path = out_root / "config.json"
        config_path.write_text(json.dumps({"task": "scene", "data_version": "test", "client_versions": ["test-client"],
                                      "source_width_range": [40, 40], "source_height_range": [32, 32],
                                      "zoom": "minimum", "code_version": "test-code", "classes": ["battle", "village"],
                                      "output": {"format": "logits", "min_confidence": .5, "min_margin": .1},
                                      "prototype_only": True, "batch": 1}))
        self.hash_snapshot()
        run = out_root / "run"
        result = train_scene(self.root, config_path, run,
                             device="cuda:0" if torch.cuda.is_available() else "cpu",
                             seed=42, resume=None, prototype=True, max_epochs=1,
                             deadline_seconds=120, pretrained=None)
        self.assertEqual(result["epochs_completed"], 1)
        self.assertTrue((run / "last.pt").is_file())
        package = out_root / "scene-package"
        report = export_package(run, self.root, package)
        self.assertTrue(report["same_tensor_parity"])
        self.assertEqual(report["status"], "prototype_only")

    @unittest.skipUnless(os.environ.get("AUTOCOC_BUILDING_INTEGRATION") == "1", "training environment only")
    def test_synthetic_building_checkpoint_and_raw_onnx_loader(self):
        # Tiny technical fixture. It is not a reviewed-image T2 smoke result.
        from training.vision.export import export_package
        weight = Path(os.environ["AUTOCOC_YOLO_WEIGHT"])
        (self.root / "DATASET_CARD.md").write_text("Task: building\n")
        metadata = json.loads((self.root / "snapshot.json").read_text())
        metadata.update({"task": "building", "prototype_only": True})
        (self.root / "snapshot.json").write_text(json.dumps(metadata))
        split = json.loads((self.root / "split.json").read_text())
        split.update({"task": "building", "prototype_only": True})
        (self.root / "split.json").write_text(json.dumps(split))
        (self.root / "class_mapping.json").write_text(json.dumps({"output_indices": {"town_hall": 0}, "stable_ids": {"town_hall": 1}}))
        (self.root / "labels").mkdir()
        rows = []
        for name in "abc":
            (self.root / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.3 0.3125\n")
            rows.append({"image": f"images/{name}.png", "sample_id": name,
                         "boxes": [{"class": "town_hall", "bbox": [14, 11, 26, 21]}],
                         "group_id": name.upper()})
        (self.root / "reviewed-buildings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        outputs = TemporaryDirectory()
        self.addCleanup(outputs.cleanup)
        out_root = Path(outputs.name)
        config_path = out_root / "building-config.json"
        config_path.write_text(json.dumps({"task": "building", "data_version": "test", "client_versions": ["test-client"],
                                      "source_width_range": [40, 40], "source_height_range": [32, 32],
                                      "zoom": "minimum", "code_version": "test-code", "architecture": "yolo11n", "batch": 1,
                                      "output": {"format": "yolo_xywh_classes", "coordinates": "input_pixels",
                                                 "min_confidence": .25, "nms_iou": .5}, "prototype_only": True}))
        self.hash_snapshot()
        run = out_root / "building-run"
        result = train_building(self.root, config_path, run, device="0", seed=42,
                                resume=None, prototype=True, max_epochs=1,
                                deadline_seconds=120, pretrained=weight)
        self.assertGreaterEqual(result["epochs_completed"], 1)
        self.assertTrue(Path(result["last_checkpoint"]).is_file())
        self.assertTrue(Path(result["resume_checkpoint"]).is_file())
        import torch
        self.assertIsNotNone(torch.load(result["resume_checkpoint"], map_location="cpu", weights_only=False)["optimizer"])
        report = export_package(run, self.root, out_root / "building-package")
        self.assertTrue(report["same_tensor_parity"])
        self.assertTrue(report["output"]["raw_untruncated"])


if __name__ == "__main__":
    unittest.main()
