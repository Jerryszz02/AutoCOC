"""Export separately trained R-state heads as CPU FP32 ONNX packages."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .data import sha256
from .dynamic_snapshot import verify_dynamic_snapshot
from .dynamic_train import R_PREPROCESS, _config, _rows, roi_tensor
from .snapshot import TrainingError
from .train import _scene_model, _signature


def export_dynamic_state(training_dir: Path, snapshot_root: Path, package_dir: Path, *,
                         atol: float = 1e-3, rtol: float = 1e-3, opset: int = 17) -> dict:
    import torch
    from autococ.dynamic_vision import DynamicFrame, DynamicModel

    training_dir, snapshot_root, package_dir = (Path(path).resolve() for path in
                                                (training_dir, snapshot_root, package_dir))
    config = json.loads((training_dir / "training_config.json").read_text(encoding="utf-8"))
    snapshot = verify_dynamic_snapshot(snapshot_root, kind=config["kind"],
                                       prototype_only=config["prototype_only"])
    temp_config = training_dir / "training_config.json"
    # Revalidate the saved, explicit runtime contract against the current
    # frozen snapshot without altering either file.
    if (_config(snapshot, temp_config)["preprocess"] != R_PREPROCESS or
            config.get("snapshot_sha256") != snapshot["snapshot_sha256"] or
            config.get("split_sha256") != snapshot["split_sha256"]):
        raise TrainingError("R export snapshot or preprocessing differs from training")
    if package_dir.exists():
        raise TrainingError(f"Refusing to overwrite R package: {package_dir}")
    checkpoint_path = training_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise TrainingError("R best checkpoint missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    identity = {key: value for key, value in config.items() if key != "deadline_seconds"}
    if checkpoint.get("signature") != _signature(identity):
        raise TrainingError("R checkpoint signature differs from training config")
    if config["kind"] not in {"r_building", "r_card"}:
        raise TrainingError("Unsupported R kind")
    val_rows = _rows(snapshot)["val"]
    if not val_rows:
        raise TrainingError("R export needs frozen validation frames")
    root = Path(snapshot["root"])
    model, _ = _scene_model(config["classes"], None)
    model.load_state_dict(checkpoint["model"])
    model.eval().cpu()
    sample = torch.from_numpy(roi_tensor(root, val_rows[0]["frame"])[None])
    package_dir.mkdir(parents=True)
    torch.onnx.export(model, sample, str(package_dir / "model.onnx"),
                      input_names=["pixels"], output_names=["logits"],
                      opset_version=opset, dynamo=False)
    manifest = {"schema_version": 1, "kind": config["kind"],
                "classes": config["classes"], "client_versions": config["client_versions"],
                "model_file": "model.onnx", "sha256": sha256(package_dir / "model.onnx"),
                "max_frame_age_seconds": config["max_frame_age_seconds"],
                "event_rules": config["event_rules"],
                "applicability": config["applicability"],
                "preprocess": config["preprocess"], "output": config["output"],
                "validation": {"status": "unvalidated", "prototype_only": config["prototype_only"],
                               "snapshot_sha256": snapshot["snapshot_sha256"],
                               "split_sha256": snapshot["split_sha256"]}}
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # The product loader checks hash, class order, preprocess, applicability,
    # event rules and the CPU ONNX session rather than a test-only mock.
    eligible = None
    for row in val_rows:
        frame = row["frame"]
        if (not config["applicability"]["requires_stable_camera"] or frame["camera_stable"]) and (
                not config["applicability"]["requires_verified_zoom"] or frame["zoom_verified"]):
            eligible = frame
            break
    if eligible is None:
        raise TrainingError("No applicable frozen validation frame for R loader smoke")
    loader = DynamicModel(package_dir, client_version=eligible["client_version"])
    source = (eligible["image"] if eligible["sha256"] == eligible["parent_image_sha256"]
              else eligible["parent_image"])
    image = cv2.imdecode(np.fromfile(root / source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise TrainingError("Cannot decode R loader smoke source frame")
    dynamic_frame = DynamicFrame(eligible["session_id"], eligible["camera_epoch"],
                                 eligible["frame_id"], eligible["captured_at_monotonic"],
                                 eligible["client_version"], image.shape[1], image.shape[0],
                                 eligible["camera_stable"], eligible["zoom_verified"])
    loader.infer(image, tuple(eligible["roi_bbox"]), dynamic_frame,
                 target_id=eligible["target_id"], offline_replay=True)
    errors, compared = [], 0
    selected = val_rows[:100]
    for row in selected:
        tensor = roi_tensor(root, row["frame"])[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).numpy()
        actual = np.asarray(loader.session.run(None, {loader.input_name: tensor})[0])
        if expected.shape != (1, len(config["classes"])) or actual.shape != expected.shape:
            raise TrainingError(f"R ONNX logits shape differs: {expected.shape}/{actual.shape}")
        errors.append(float(np.max(np.abs(expected - actual))))
        compared += int(np.allclose(expected, actual, atol=atol, rtol=rtol))
    report = {"kind": config["kind"], "status": "prototype_only" if config["prototype_only"] else "unevaluated",
              "checkpoint_sha256": sha256(checkpoint_path), "onnx_sha256": manifest["sha256"],
              "snapshot_sha256": snapshot["snapshot_sha256"],
              "provider": loader.session.get_providers()[0], "opset": opset,
              "validation_frames_compared": len(selected), "frames_within_tolerance": compared,
              "same_tensor_parity": compared == len(selected),
              "max_absolute_error": max(errors), "atol": atol, "rtol": rtol,
              "parity_sample_sufficient": config["prototype_only"] or len(selected) >= 100,
              "loader_smoke_frame_id": eligible["frame_id"],
              "validation_status": "unvalidated"}
    (package_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (package_dir / "MODEL_CARD.md").write_text(
        f"# {config['kind']} offline ROI state model\n\nStatus: {report['status']} / unvalidated.\n"
        f"Snapshot SHA-256: {report['snapshot_sha256']}\n"
        f"ONNX SHA-256: {report['onnx_sha256']}\n"
        "Parity is export compatibility only. Event recall, false triggers, latency and real battles remain untested.\n",
        encoding="utf-8")
    if not report["same_tensor_parity"]:
        raise TrainingError(f"R PyTorch/ONNX parity failed: max error {max(errors)}")
    return report
