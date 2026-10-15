import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from autococ.dynamic_vision import (CARD_STATES, BUILDING_STATES, DynamicFrame,
                                    DynamicModel, DynamicUnavailable,
                                    dynamic_contract_sha256)


class Session:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def get_inputs(self):
        return [type("Input", (), {"name": "image"})()]

    def run(self, _, inputs):
        self.calls.append(inputs["image"])
        return [self.output]


def package(root: Path, kind="r_building", output=None):
    root.mkdir()
    weight = root / "model.onnx"
    weight.write_bytes(b"fixture-dynamic-weight")
    classes = (list(BUILDING_STATES) if kind == "r_building" else
               list(CARD_STATES) if kind == "r_card" else ["hero:archer_queen", "troop_group"])
    manifest = {"schema_version": 1, "kind": kind, "classes": classes,
        "client_versions": ["18.647.2"], "model_file": "model.onnx",
        "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        "max_frame_age_seconds": 1.0,
        "event_rules": {"consecutive_frames": 2, "min_iou": .4,
                         "max_gap_seconds": 1.0, "max_missed_frames": 1,
                         "event_min_confidence": .95, "max_frame_age_seconds": .5},
        "applicability": {"roi_kind": kind, "source_width_range": [100, 100],
                          "source_height_range": [100, 100], "requires_stable_camera": True,
                          "requires_verified_zoom": kind != "r_card"},
        "preprocess": {"width": 224 if kind != "u_position" else 128,
                       "height": 224 if kind != "u_position" else 128,
                       "color": "RGB", "resize": "stretch", "scale": 1 / 255,
                       "mean": [0, 0, 0], "std": [1, 1, 1]},
        "output": {"format": "probabilities" if kind != "u_position" else "xyxy_score_class",
                   "min_confidence": .7, "max_detections": 10},
        "validation": {"status": "unvalidated", "prototype_only": True,
                       "snapshot_sha256": "a" * 64, "split_sha256": "b" * 64}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    session = Session(output if output is not None else np.array([[.95, .02, .02, .01]], dtype=np.float32))
    return DynamicModel(root, client_version="18.647.2", session_factory=lambda _: session), session


def frame(**changes):
    return DynamicFrame(**{"session_id": "s", "camera_epoch": "c1", "frame_id": "f1",
        "captured_at_monotonic": 100., "client_version": "18.647.2", "width": 100,
        "height": 100, "camera_stable": True, "zoom_verified": True, **changes})


def test_roi_source_coordinates_and_capture_start_age_survive_inference():
    with TemporaryDirectory() as temp:
        model, session = package(Path(temp) / "r")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(DynamicUnavailable, match="age"):
            model.infer(image, (10, 20, 90, 100), frame(), target_id="hall-1", now_monotonic=102)
        assert not session.calls
        result = model.infer(image, (10, 20, 90, 100), frame(), target_id="hall-1",
                             now_monotonic=100.5)[0]
        assert result.label == "intact"
        assert result.bbox == (10, 20, 90, 100)
        assert result.target_id == "hall-1"
        assert result.captured_at_monotonic == 100
        assert result.frame_id == "f1" and result.camera_epoch == "c1"
        assert session.calls[0].shape == (1, 3, 224, 224)


def test_dynamic_package_rejects_missing_epoch_zoom_bad_hash_and_provenance():
    with TemporaryDirectory() as temp:
        root = Path(temp) / "r"
        model, _ = package(root)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for bad_frame in (frame(camera_epoch=""), frame(zoom_verified=False),
                          frame(camera_stable=False)):
            with pytest.raises(DynamicUnavailable):
                model.infer(image, (10, 20, 90, 100), bad_frame,
                            target_id="hall", now_monotonic=100.1)
        with pytest.raises(DynamicUnavailable, match="target_id"):
            model.infer(image, (10, 20, 90, 100), frame(), now_monotonic=100.1)
        (root / "model.onnx").write_bytes(b"changed")
        with pytest.raises(DynamicUnavailable, match="SHA-256"):
            DynamicModel(root, client_version="18.647.2", session_factory=lambda _: Session(None))


def test_u_boxes_map_back_to_original_source_pixels():
    with TemporaryDirectory() as temp:
        raw = np.array([[[32, 32, 96, 96, .9, 0]]], dtype=np.float32)
        model, _ = package(Path(temp) / "u", kind="u_position", output=raw)
        result = model.infer(np.zeros((100, 100, 3), dtype=np.uint8), (10, 20, 90, 100),
                             frame(), now_monotonic=100.1)[0]
        assert result.label == "hero:archer_queen"
        assert result.bbox == (30, 40, 70, 80)
        assert result.target_id is None


def test_u_invalid_confidence_cannot_be_silently_filtered():
    with TemporaryDirectory() as temp:
        model, _ = package(Path(temp) / "u", kind="u_position",
                           output=np.array([[[32, 32, 96, 96, 1.2, 0]]], dtype=np.float32))
        with pytest.raises(DynamicUnavailable, match="confidence"):
            model.infer(np.zeros((100, 100, 3), dtype=np.uint8), (10, 20, 90, 100),
                        frame(), now_monotonic=100.1)


def test_event_rules_are_frozen_in_model_contract_and_required():
    with TemporaryDirectory() as temp:
        root = Path(temp) / "r"
        model, _ = package(root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["event_rules"]["event_min_confidence"] = .99
        assert dynamic_contract_sha256(manifest) != model.contract_sha256
        del manifest["event_rules"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(DynamicUnavailable, match="event rules"):
            DynamicModel(root, client_version="18.647.2", session_factory=lambda _: Session(None))
