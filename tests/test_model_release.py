"""Release packaging and live-use checks use local fake ONNX bytes only."""

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autococ.model_release import require_released_model
from autococ.model_vision import BuildingModel, ModelUnavailable, SceneModel, model_contract_sha256
from scripts.release_vision_models import release_models, verify_and_revalidate
from scripts.validate_vision_release import main as validate_main
from tests.test_model_vision import FakeSession


class ModelReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = {}
        self.hashes = {}
        for kind in ("scene", "building"):
            package = self.root / "source" / kind
            package.mkdir(parents=True)
            weights = package / "model.onnx"
            weights.write_bytes(("fake " + kind).encode())
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            self.sources[kind], self.hashes[kind] = package, digest
            manifest = {"schema_version": 1, "kind": kind, "client_versions": ["v1"],
                "classes": ["battle"] if kind == "scene" else ["town_hall"],
                "model_file": "model.onnx", "sha256": digest, "code_version": "test",
                "data_version": "v001", "validation": {"status": "unvalidated",
                    "prototype_only": False, "snapshot_sha256": ("a" if kind == "scene" else "b") * 64,
                    "split_sha256": ("c" if kind == "scene" else "d") * 64},
                "applicability": {"screen_scope": "full_ui" if kind == "scene" else "battlefield",
                    "source_width_range": [20, 200], "source_height_range": [20, 200], "zoom": "minimum"},
                "preprocess": {"width": 100, "height": 100, "color": "RGB", "resize": "letterbox",
                    "scale": 1 / 255, "mean": [0, 0, 0], "std": [1, 1, 1], "pad_value": 114},
                "max_frame_age_seconds": 2,
                "output": {"format": "logits", "min_confidence": .5, "min_margin": .1} if kind == "scene"
                    else {"format": "xyxy_score_class", "coordinates": "input_pixels",
                          "min_confidence": .5, "nms_iou": .5}}
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        from tests.test_vision_release import VisionReleaseTests
        fixture = VisionReleaseTests("test_complete_static_evidence_can_pass")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for kind in ("scene", "building"):
            manifest_file = self.sources[kind] / "manifest.json"
            manifest = json.loads(manifest_file.read_text())
            report = fixture.scene_report if kind == "scene" else fixture.building_report
            manifest["classes"] = report["classes"]
            manifest["validation"].update(fixture.metadata(fixture.scene_dir if kind == "scene" else fixture.building_dir)["validation"])
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            contract = model_contract_sha256(manifest)
            report["sha256"] = self.hashes[kind]
            report["model_contract_sha256"] = contract
            report["metadata"]["model_contract_sha256"] = contract
            report["metadata"]["validation"] = manifest["validation"].copy()
        fixture.ap_stability["sha256"] = self.hashes["building"]
        fixture.ap_stability["model_contract_sha256"] = fixture.building_report["model_contract_sha256"]
        fixture.ap_stability["metadata"] = fixture.building_report["metadata"].copy()
        fixture.parity["model_sha256"] = self.hashes.copy()
        fixture.windows["model_sha256"] = self.hashes.copy()
        fixture.core_entry["building_model_sha256"] = self.hashes["building"]
        self.report_path = self.root / "pass.json"
        files = {}
        for key, value in (("acceptance_config", fixture.acceptance),
                           ("scene_report", fixture.scene_report),
                           ("building_report", fixture.building_report),
                           ("ap_stability_report", fixture.ap_stability),
                           ("parity_report", fixture.parity), ("windows_report", fixture.windows),
                           ("core_entry_report", fixture.core_entry)):
            path = self.root / f"{key}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            files[key] = path
        arguments = ["--acceptance-config", str(files["acceptance_config"])]
        for key in ("scene_report", "building_report", "ap_stability_report", "parity_report",
                    "windows_report", "core_entry_report"):
            arguments.extend(["--" + key.replace("_", "-"), str(files[key])])
        arguments.extend(["--scene-dataset", str(fixture.scene_dir),
                          "--building-dataset", str(fixture.building_dir),
                          "--output", str(self.report_path)])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(validate_main(arguments), 0)
        self.report = json.loads(self.report_path.read_text())
        self.output = self.root / "released"

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def model(self, kind):
        cls = SceneModel if kind == "scene" else BuildingModel
        output = [[1.]] if kind == "scene" else [[[1, 2, 3, 4, .9, 0]]]
        return cls(self.output / kind, client_version="v1",
                   session_factory=lambda _: FakeSession(output))

    def test_valid_bundle_is_bound_and_source_stays_unvalidated(self):
        result = release_models(self.sources["scene"], self.sources["building"],
                                self.report_path, self.output)
        self.assertEqual(result["status"], "released")
        self.assertEqual(verify_and_revalidate(self.output / "release_report.json")["status"], "PASS")
        self.assertTrue((self.output / "release_report.inputs" / "scene_report.json").is_file())
        for kind in ("scene", "building"):
            self.assertEqual(require_released_model(self.model(kind))["status"], "PASS")
            source = json.loads((self.sources[kind] / "manifest.json").read_text())
            self.assertEqual(source["validation"]["status"], "unvalidated")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            release_models(self.sources["scene"], self.sources["building"], self.report_path, self.output)

    def test_rejects_prototype_wrong_hash_and_failed_report_before_copy(self):
        manifest_path = self.sources["scene"] / "manifest.json"
        original = json.loads(manifest_path.read_text())
        bad = json.loads(manifest_path.read_text())
        bad["validation"]["prototype_only"] = True
        manifest_path.write_text(json.dumps(bad))
        with self.assertRaises(ModelUnavailable):
            release_models(*self.sources.values(), self.report_path, self.output)
        manifest_path.write_text(json.dumps(original))
        self.report["evidence"]["building_model_sha256"] = "0" * 64
        self.write_report()
        with self.assertRaises(ModelUnavailable):
            release_models(*self.sources.values(), self.report_path, self.output)
        self.report["evidence"]["building_model_sha256"] = self.hashes["building"]
        self.report["status"] = "FAIL"
        self.write_report()
        with self.assertRaises(ModelUnavailable):
            release_models(*self.sources.values(), self.report_path, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_report_and_tampering_block_live_use(self):
        release_models(self.sources["scene"], self.sources["building"], self.report_path, self.output)
        model = self.model("building")
        report_file = self.output / "release_report.json"
        original = report_file.read_bytes()
        report_file.unlink()
        with self.assertRaisesRegex(ModelUnavailable, "cannot be read"):
            require_released_model(model)
        report_file.write_bytes(original + b" ")
        with self.assertRaisesRegex(ModelUnavailable, "SHA-256 mismatch"):
            require_released_model(model)
        report_file.write_bytes(original)
        manifest_file = self.output / "building" / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["validation"]["snapshot_sha256"] = "0" * 64
        manifest_file.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ModelUnavailable, "snapshot_sha256"):
            require_released_model(self.model("building"))

    def test_changed_source_weights_cannot_be_promoted(self):
        (self.sources["building"] / "model.onnx").write_bytes(b"different weights")
        with self.assertRaisesRegex(ModelUnavailable, "SHA-256 mismatch"):
            release_models(self.sources["scene"], self.sources["building"],
                           self.report_path, self.output)
        self.assertFalse(self.output.exists())

    def test_handwritten_pass_and_changed_audit_input_are_rejected(self):
        handwritten = self.root / "handwritten.json"
        handwritten.write_text(json.dumps({key: value for key, value in self.report.items()
                                           if key != "input_manifest"}), encoding="utf-8")
        with self.assertRaisesRegex(ModelUnavailable, "input audit manifest"):
            release_models(self.sources["scene"], self.sources["building"],
                           handwritten, self.output)
        audit_file = self.root / self.report["input_manifest"]["files"]["scene_report"]["path"]
        audit_file.write_bytes(audit_file.read_bytes() + b" ")
        with self.assertRaisesRegex(ModelUnavailable, "input SHA-256 mismatch"):
            release_models(self.sources["scene"], self.sources["building"],
                           self.report_path, self.output)

    def test_behavior_contract_rejects_threshold_and_class_edits(self):
        release_models(self.sources["scene"], self.sources["building"],
                       self.report_path, self.output)
        scene_file = self.output / "scene" / "manifest.json"
        original = json.loads(scene_file.read_text())
        changed = json.loads(scene_file.read_text())
        changed["output"]["min_confidence"] = .9
        scene_file.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ModelUnavailable, "model_contract_sha256"):
            require_released_model(self.model("scene"))
        changed = json.loads(json.dumps(original))
        changed["classes"] = list(reversed(changed["classes"]))
        scene_file.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ModelUnavailable, "model_contract_sha256"):
            require_released_model(self.model("scene"))


if __name__ == "__main__":
    unittest.main()
