"""CPU FP32 ONNX export of the YOLO26 one-to-one U-position head."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .data import sha256
from .dynamic_snapshot import verify_dynamic_snapshot
from .snapshot import TrainingError
from .train import _offline_ultralytics, _signature
from .u_train import U_PREPROCESS, u_config, u_rows


def _validate_output(value: np.ndarray, classes: int, max_detections: int = 300) -> None:
    if (value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 6 or
            not 1 <= value.shape[1] <= max_detections or not np.isfinite(value).all()):
        raise TrainingError(f"U one-to-one output must be finite [1,N,6], got {value.shape}")
    candidates = value[0][value[0, :, 4] >= 0.001]
    if (len(candidates) and (np.any(candidates[:, 4] > 1) or np.any(candidates[:, 4] < 0) or
            np.any(candidates[:, 5] != np.floor(candidates[:, 5])) or
            np.any(candidates[:, 5] < 0) or np.any(candidates[:, 5] >= classes))):
        raise TrainingError("U one-to-one confidence/class contract failed")


def _source_image(root: Path, row: dict) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(root / row["source"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise TrainingError("U frozen validation source cannot be decoded")
    return image


def _manifest(config: dict, snapshot: dict, digest: str) -> dict:
    return {"schema_version": 1, "kind": "u_position", "classes": config["classes"],
            "client_versions": config["client_versions"], "model_file": "model.onnx", "sha256": digest,
            "max_frame_age_seconds": config["max_frame_age_seconds"],
            "event_rules": config["event_rules"], "applicability": config["applicability"],
            "preprocess": U_PREPROCESS, "output": config["output"],
            "validation": {"status": "unvalidated", "prototype_only": config["prototype_only"],
                           "snapshot_sha256": snapshot["snapshot_sha256"],
                           "split_sha256": snapshot["split_sha256"]}}


def export_u_position(training_dir: Path, snapshot_root: Path, package_dir: Path, *,
                      atol: float = 1e-3, rtol: float = 1e-3, opset: int = 17) -> dict:
    from autococ.dynamic_vision import DynamicFrame, DynamicModel
    import torch

    training_dir, snapshot_root, package_dir = (Path(path).resolve() for path in
                                                (training_dir, snapshot_root, package_dir))
    config_path = training_dir / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot = verify_dynamic_snapshot(snapshot_root, kind="u_position",
                                       prototype_only=config.get("prototype_only"))
    if (u_config(snapshot, config_path)["preprocess"] != U_PREPROCESS or
            config.get("snapshot_sha256") != snapshot["snapshot_sha256"] or
            config.get("split_sha256") != snapshot["split_sha256"]):
        raise TrainingError("U export snapshot/config differs from training")
    rows = u_rows(snapshot)["val"]
    if not rows:
        raise TrainingError("U export needs frozen validation ROI frames")
    if package_dir.exists():
        raise TrainingError(f"Refusing to overwrite U package: {package_dir}")
    checkpoint = training_dir / "run" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise TrainingError("U best checkpoint missing")
    training_report = json.loads((training_dir / "training_report.json").read_text(encoding="utf-8"))
    if (training_report.get("best_checkpoint_sha256") != sha256(checkpoint) or
            training_report.get("snapshot_sha256") != snapshot["snapshot_sha256"] or
            training_report.get("prototype_only") is not config["prototype_only"]):
        raise TrainingError("U best checkpoint or provenance differs from training report")
    resume_metadata = json.loads((training_dir / "resume.metadata.json").read_text(encoding="utf-8"))
    environment = json.loads((training_dir / "environment.json").read_text(encoding="utf-8"))
    args = json.loads((training_dir / "effective_train_args.json").read_text(encoding="utf-8"))
    recipe = {key: value for key, value in args.items() if key not in
              {"data", "epochs", "device", "project", "name", "exist_ok", "resume", "seed"}}
    signature = _signature({"config": {k: v for k, v in config.items() if k != "deadline_seconds"},
                            "recipe": recipe, "ultralytics_version": environment["packages"]["ultralytics"],
                            "training_recipe_version": 1})
    if resume_metadata.get("signature") != signature:
        raise TrainingError("U training signature differs from saved checkpoint identity")
    package_dir.mkdir(parents=True)
    YOLO = _offline_ultralytics(package_dir / "ultralytics_config")
    torch.set_num_threads(2)
    export_source = package_dir / "export_source.pt"
    shutil.copyfile(checkpoint, export_source)
    yolo = YOLO(str(export_source))
    if list(yolo.names.values()) != config["classes"]:
        raise TrainingError("U checkpoint class order differs from frozen snapshot")
    exported = Path(yolo.export(format="onnx", imgsz=640, batch=1, dynamic=False,
                                half=False, nms=False, max_det=300, simplify=False,
                                opset=opset, device="cpu"))
    shutil.copyfile(exported, package_dir / "model.onnx")
    export_source.unlink()
    if exported != package_dir / "model.onnx":
        exported.unlink()
    manifest = _manifest(config, snapshot, sha256(package_dir / "model.onnx"))
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    root = Path(snapshot["root"])
    eligible = [row for row in rows if row["frame"]["camera_stable"] and
                (not config["applicability"]["requires_verified_zoom"] or row["frame"]["zoom_verified"])]
    if not eligible:
        raise TrainingError("U export has no applicable frozen validation ROI")
    loader = DynamicModel(package_dir, client_version=eligible[0]["frame"]["client_version"])
    # Ultralytics selects this same one-to-one branch for nms=False export.
    head = yolo.model.model[-1]
    if not hasattr(head, "one2one_cv2"):
        raise TrainingError("U checkpoint lacks the YOLO26 one-to-one head")
    head.end2end = True
    head.export = True
    head.format = "onnx"
    head.max_det = 300
    yolo.model.eval().cpu()
    compared, errors = 0, []
    for row in eligible[:100]:
        frame = row["frame"]
        image = _source_image(root, row)
        info = DynamicFrame(frame["session_id"], frame["camera_epoch"], frame["frame_id"],
                            frame["captured_at_monotonic"], frame["client_version"],
                            image.shape[1], image.shape[0], frame["camera_stable"], frame["zoom_verified"])
        # This uses the real loader's ROI resize, validation and original-coordinate restoration.
        loader.infer(image, tuple(frame["roi_bbox"]), info, offline_replay=True)
        x0, y0, x1, y1 = frame["roi_bbox"]
        crop = cv2.resize(image[y0:y1, x0:x1], (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None])
        with torch.inference_mode():
            reference = yolo.model(torch.from_numpy(tensor))
        reference = np.asarray(reference.detach().cpu())
        actual = loader.session.run(None, {loader.input_name: tensor})[0]
        _validate_output(reference, len(config["classes"]))
        _validate_output(actual, len(config["classes"]))
        if reference.shape != actual.shape:
            raise TrainingError(f"U PyTorch/ONNX shape mismatch: {reference.shape}/{actual.shape}")
        errors.append(float(np.max(np.abs(reference - actual))))
        compared += int(np.allclose(reference, actual, atol=atol, rtol=rtol))
    report = {"kind": "u_position", "status": "prototype_only" if config["prototype_only"] else "unevaluated",
              "validation_status": "unvalidated", "provider": loader.session.get_providers()[0],
              "checkpoint_sha256": sha256(checkpoint), "onnx_sha256": manifest["sha256"],
              "snapshot_sha256": snapshot["snapshot_sha256"], "split_sha256": snapshot["split_sha256"],
              "images_compared": len(eligible[:100]), "images_within_tolerance": compared,
              "max_absolute_error": max(errors), "same_tensor_parity": compared == len(eligible[:100]),
              "atol": atol, "rtol": rtol, "opset": opset,
              "output": {"format": "xyxy_score_class", "shape": list(actual.shape),
                         "head": "YOLO26 one-to-one nms=False"}}
    (package_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (package_dir / "MODEL_CARD.md").write_text(
        "# U position model\n\nStatus: " + report["status"] + "/unvalidated.\n"
        "CPU FP32 ONNX one-to-one output; parity proves export compatibility only. "
        "Accuracy, event timing and live gameplay remain untested.\n", encoding="utf-8")
    if not report["same_tensor_parity"]:
        raise TrainingError(f"U PyTorch/ONNX parity failed: max error {max(errors)}")
    return report
