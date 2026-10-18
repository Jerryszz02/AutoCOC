"""CPU FP32 ONNX package export and same-tensor PyTorch/ONNX parity checks."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .data import sha256
from .train import TrainingError, _offline_ultralytics, _scene_model, prepare_bgr, verify_snapshot


def _probe(snapshot_root: Path, snapshot: dict) -> np.ndarray:
    relative = snapshot["paths"]["val"][0]
    path = snapshot_root / relative
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise TrainingError(f"Cannot decode validation probe: {path}")
    return image


def _tensor(output):
    import torch
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TrainingError("Unexpected PyTorch model output")
    return output.detach().cpu().numpy().astype(np.float32)


def _manifest(config: dict, kind: str, digest: str) -> dict:
    versions = config.get("client_versions")
    if not isinstance(versions, list) or not versions or not all(isinstance(x, str) and x for x in versions):
        raise TrainingError("Explicit verified client_versions required")
    output = config.get("output")
    if not isinstance(output, dict):
        raise TrainingError("Explicit runtime output thresholds required in training config")
    if kind == "scene" and output.get("format") != "logits":
        raise TrainingError("Scene export must use logits")
    if kind == "building" and (output.get("format") != "yolo_xywh_classes" or output.get("coordinates") != "input_pixels"):
        raise TrainingError("Building export requires raw xywh class scores in input pixels")
    code_version = config.get("code_version")
    if not isinstance(code_version, str) or not code_version.strip():
        raise TrainingError("Explicit code_version required")
    output = {key: value for key, value in output.items() if key != "evaluation_raw_provenance"}
    manifest = {"schema_version": 1, "kind": kind, "client_versions": versions,
                "code_version": code_version, "data_version": config["data_version"],
                "validation": {"status": "unvalidated", "prototype_only": bool(config["prototype_only"]),
                               "snapshot_sha256": config["snapshot_sha256"],
                               "split_sha256": config["split_sha256"]},
                "applicability": {"screen_scope": "full_ui" if kind == "scene" else "battlefield",
                                  "source_width_range": config["source_width_range"],
                                  "source_height_range": config["source_height_range"],
                                  "zoom": config["zoom"]},
                "classes": config["classes"], "model_file": "model.onnx", "sha256": digest,
                "max_frame_age_seconds": config.get("max_frame_age_seconds", 2.0),
                "preprocess": config["preprocess"], "output": output}
    return manifest


def _validate_raw_building(output: np.ndarray, classes: int) -> dict:
    if output.ndim != 3 or output.shape[0] != 1:
        raise TrainingError(f"Detector output must be rank-3 batch=1, got {output.shape}")
    if output.shape[1] == 4 + classes:
        candidates = output.shape[2]
        channels_first = True
    elif output.shape[2] == 4 + classes:
        candidates = output.shape[1]
        channels_first = False
    else:
        raise TrainingError(f"Detector output is not raw 4+C class scores: {output.shape}")
    if output.size > 600000 or candidates <= 0:
        raise TrainingError(f"Detector output exceeds runtime bound: {output.shape}")
    boxes = output[0, :4, :] if channels_first else output[0, :, :4].T
    scores = output[0, 4:, :] if channels_first else output[0, :, 4:].T
    if not np.isfinite(output).all() or (scores < 0).any() or (scores > 1).any():
        raise TrainingError("Detector output violates finite xywh/probability contract")
    # YOLO26 can emit a few negative widths/heights for very low-score
    # background anchors. They remain part of the raw dense tensor; runtime
    # and independent AP both discard invalid geometry after score decoding.
    invalid_geometry = (boxes[2] <= 0) | (boxes[3] <= 0)
    return {"shape": list(output.shape), "candidates": int(candidates),
            "elements": int(output.size), "channels_first": channels_first,
            "raw_untruncated": True,
            "invalid_geometry_candidates": int(invalid_geometry.sum()),
            "max_invalid_geometry_score": float(scores[:, invalid_geometry].max()) if invalid_geometry.any() else None}


def _raw_graph_provenance(path: Path) -> dict:
    """Conservatively certify the exported graph has no selection/filter ops.

    This is only one part of provenance; the caller also verifies it requested
    raw nms=None output and checks the resulting 4+C dense tensor.
    """
    import onnx
    graph = onnx.load(str(path)).graph
    ops = {node.op_type for node in graph.node}
    blockers = sorted(ops & {"TopK", "NonMaxSuppression", "NonZero", "Where", "ThresholdedRelu"})
    return {"verified": not blockers, "blocked_by_ops": blockers,
            "observed_ops": sorted(ops)}


def export_package(training_dir: Path, snapshot_root: Path, package_dir: Path, *,
                   checkpoint: Path | None = None, opset: int = 17,
                   atol: float = 1e-3, rtol: float = 1e-3) -> dict:
    import torch
    from autococ.model_vision import BuildingModel, SceneModel

    training_dir, snapshot_root, package_dir = training_dir.resolve(), snapshot_root.resolve(), package_dir.resolve()
    config = json.loads((training_dir / "training_config.json").read_text(encoding="utf-8"))
    kind = config["task"]
    if kind not in {"scene", "building"}:
        raise TrainingError("Unsupported task")
    snapshot = verify_snapshot(snapshot_root, kind)
    if config.get("snapshot_sha256") != snapshot["snapshot_sha256"] or config.get("split_sha256") != snapshot["split_sha256"]:
        raise TrainingError("Training snapshot differs from export snapshot")
    if package_dir.exists():
        raise TrainingError(f"Refusing to overwrite package: {package_dir}")
    ckpt = (checkpoint or (training_dir / "best.pt" if kind == "scene" else training_dir / "run" / "weights" / "best.pt")).resolve()
    if not ckpt.is_file():
        raise TrainingError(f"Checkpoint missing: {ckpt}")
    package_dir.mkdir(parents=True)
    image = _probe(snapshot_root, snapshot)
    if kind == "scene":
        model, _ = _scene_model(config["classes"], None)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.eval().cpu()
        sample = torch.from_numpy(prepare_bgr(image, config["preprocess"])[None])
        with torch.inference_mode():
            expected = _tensor(model(sample))
        torch.onnx.export(model, sample, str(package_dir / "model.onnx"),
                          input_names=["pixels"], output_names=["logits"],
                          opset_version=opset, dynamo=False)
    else:
        YOLO = _offline_ultralytics(package_dir / "ultralytics_config")
        export_source = package_dir / "export_source.pt"
        shutil.copyfile(ckpt, export_source)
        yolo = YOLO(str(export_source))
        yolo.model.eval().cpu()
        sample = torch.from_numpy(prepare_bgr(image, config["preprocess"])[None])
        with torch.inference_mode():
            expected = _tensor(yolo.model(sample))
        _validate_raw_building(expected, len(config["classes"]))
        exported = Path(yolo.export(format="onnx", imgsz=640, batch=1, dynamic=False,
                                    half=False, nms=None, simplify=False, opset=opset, device="cpu"))
        shutil.copyfile(exported, package_dir / "model.onnx")
        export_source.unlink()
        if exported != package_dir / "model.onnx":
            exported.unlink()
    manifest = _manifest(config, kind, sha256(package_dir / "model.onnx"))
    raw_provenance = None
    if kind == "building":
        raw_provenance = _raw_graph_provenance(package_dir / "model.onnx")
        if raw_provenance["verified"]:
            manifest["output"]["evaluation_raw_provenance"] = {
                "unfiltered": True, "no_topk": True, "no_nms": True}
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # This exercises the real product loader, including SHA-256, shape, class,
    # applicability and CPUExecutionProvider checks.
    loader = (SceneModel if kind == "scene" else BuildingModel)(package_dir, client_version=config["client_versions"][0])
    actual = loader._run(loader._prepare(image)[0])
    if kind == "scene":
        if expected.shape != (1, len(config["classes"])) or actual.shape != expected.shape:
            raise TrainingError(f"Scene output shape mismatch: {expected.shape} / {actual.shape}")
        loader.classify(image, captured_at=1, now=1)
        contract = {"shape": list(actual.shape)}
    else:
        contract = _validate_raw_building(actual, len(config["classes"]))
        if actual.shape != expected.shape:
            raise TrainingError(f"Detector output shape mismatch: {expected.shape} / {actual.shape}")
        loader.detect(image, captured_at=1, now=1)
    errors, compared, decision_matches = [], 0, 0
    # Smoke checks one validation frame; development compares every held-out
    # validation frame. Neither path touches the test split.
    parity_limit = None if config.get("development") else (1 if config["prototype_only"] else 100)
    parity_images = snapshot["paths"]["val"][:parity_limit]
    for relative in parity_images:
        path = snapshot_root / relative
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise TrainingError(f"Cannot decode parity image: {path}")
        input_tensor, _ = loader._prepare(frame)
        with torch.inference_mode():
            pytorch_output = _tensor((model if kind == "scene" else yolo.model)(torch.from_numpy(input_tensor)))
        onnx_output = loader._run(input_tensor)
        if pytorch_output.shape != onnx_output.shape:
            raise TrainingError(f"Parity shape mismatch for {relative}: {pytorch_output.shape} / {onnx_output.shape}")
        if kind == "building":
            _validate_raw_building(onnx_output, len(config["classes"]))
        errors.append(float(np.max(np.abs(pytorch_output - onnx_output))))
        decision_matches += int(kind != "scene" or np.argmax(pytorch_output) == np.argmax(onnx_output))
        compared += int(np.allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol))
    close = compared == len(parity_images)
    absolute_error = max(errors)
    sufficient = config["prototype_only"] or len(parity_images) >= 100
    report = {"status": "prototype_only" if config["prototype_only"] else "unevaluated",
              "kind": kind, "checkpoint_sha256": sha256(ckpt), "onnx_sha256": manifest["sha256"],
              "snapshot_sha256": snapshot["snapshot_sha256"], "opset": opset,
              "provider": loader.session.get_providers()[0], "same_tensor_parity": close,
              "max_absolute_error": absolute_error, "atol": atol, "rtol": rtol,
              "probe_image": snapshot["paths"]["val"][0], "output": contract,
              "raw_graph_provenance": raw_provenance,
              "images_compared": len(parity_images), "images_within_tolerance": compared,
              "scene_decision_agreement": decision_matches / len(parity_images) if kind == "scene" else None,
              "parity_sample_sufficient": sufficient, "validation_status": "unvalidated"}
    (package_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (package_dir / "MODEL_CARD.md").write_text(
        f"# {kind} model\n\nStatus: {report['status']} / unvalidated.\n"
        f"Checkpoint SHA-256: {report['checkpoint_sha256']}\n"
        f"ONNX SHA-256: {report['onnx_sha256']}\n"
        f"Snapshot SHA-256: {report['snapshot_sha256']}\n"
        "Parity checks export compatibility only; accuracy and live gameplay are untested.\n",
        encoding="utf-8")
    if not close:
        raise TrainingError(f"PyTorch/ONNX parity failed: max absolute error {absolute_error}")
    return report
