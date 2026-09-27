"""Strict, local ONNX inference for versioned scene and building model packages.

The model package owns its labels, transforms and thresholds. No default label
list is inferred from a network output, and a missing package is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .scene import (SCENE_UNKNOWN, SCENE_BATTLE, SCENE_CLAN_CHAT, SCENE_DISCONNECTED,
                    SCENE_DONATION, SCENE_ENEMY_VILLAGE, SCENE_MAINTENANCE, SCENE_POPUP,
                    SCENE_REQUEST, SCENE_SEARCH, SCENE_SETTLEMENT, SCENE_STARTING,
                    SCENE_TRAINING, SCENE_VILLAGE)

SUPPORTED_SCENES = frozenset({SCENE_BATTLE, SCENE_CLAN_CHAT, SCENE_DISCONNECTED,
    SCENE_DONATION, SCENE_ENEMY_VILLAGE, SCENE_MAINTENANCE, SCENE_POPUP,
    SCENE_REQUEST, SCENE_SEARCH, SCENE_SETTLEMENT, SCENE_STARTING,
    SCENE_TRAINING, SCENE_VILLAGE})


class ModelUnavailable(RuntimeError):
    """The installed model cannot safely be used for this client or input."""


class StaleModelFrame(ModelUnavailable):
    """The image is older than the model package's maximum allowed age."""


@dataclass(frozen=True)
class ScenePrediction:
    scene: str
    confidence: float
    unknown: bool
    reason: str | None
    frame_captured_at: float
    model_sha256: str


@dataclass(frozen=True)
class BuildingObservation:
    kind: str
    bbox: tuple[float, float, float, float]  # original image pixels
    confidence: float
    frame_captured_at: float
    model_sha256: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)


@dataclass(frozen=True)
class CoreGeometry:
    center: tuple[float, float]  # baseline pixels
    bbox: tuple[float, float, float, float]  # baseline pixels
    confidence: float
    evidence: tuple[str, ...]


def _number(value: object, *, minimum: float = 0, maximum: float = math.inf) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ModelUnavailable(f"Invalid model metadata number: {value!r}")
    return float(value)


def _manifest(root: Path, kind: str, client_version: str | None) -> tuple[dict, Path, str]:
    root = Path(root).resolve()
    try:
        data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelUnavailable(f"Model manifest unavailable: {root / 'manifest.json'}") from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1 or data.get("kind") != kind:
        raise ModelUnavailable("Unsupported model manifest schema or kind")
    versions = data.get("client_versions")
    if not isinstance(versions, list) or not versions or not all(isinstance(v, str) and v for v in versions):
        raise ModelUnavailable("Model client_versions must be explicit")
    if client_version is not None and client_version not in versions:
        raise ModelUnavailable("Game client version is unverified or incompatible")
    labels = data.get("classes")
    if not isinstance(labels, list) or not labels or not all(isinstance(s, str) and s for s in labels) or len(set(labels)) != len(labels):
        raise ModelUnavailable("Model classes must be unique nonempty names")
    for field in ("code_version", "data_version"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ModelUnavailable(f"Model {field} must be explicit")
    validation = data.get("validation")
    if not isinstance(validation, dict) or validation.get("status") not in {"unvalidated", "evaluated"}:
        raise ModelUnavailable("Model validation status must be explicit")
    if validation["status"] == "evaluated" and (not isinstance(validation.get("report_id"), str) or not validation["report_id"].strip()):
        raise ModelUnavailable("Evaluated model requires a validation report ID")
    applicability = data.get("applicability")
    expected_scope = "full_ui" if kind == "scene" else "battlefield"
    if not isinstance(applicability, dict) or applicability.get("screen_scope") != expected_scope or not isinstance(applicability.get("zoom"), str) or not applicability["zoom"]:
        raise ModelUnavailable("Model screen scope and zoom applicability must be explicit")
    for field in ("source_width_range", "source_height_range"):
        bounds = applicability.get(field)
        if not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not int or not 1 <= v <= 16384 for v in bounds) or bounds[0] > bounds[1]:
            raise ModelUnavailable(f"Invalid applicability {field}")
    filename, digest = data.get("model_file"), data.get("sha256")
    if not isinstance(filename, str) or not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ModelUnavailable("Model file and SHA-256 are required")
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ModelUnavailable("Model file is missing or outside package")
    try:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise ModelUnavailable("Cannot read model file for SHA-256 verification") from exc
    if actual != digest.lower():
        raise ModelUnavailable("Model SHA-256 mismatch")
    return data, path, actual


class _ONNXModel:
    def __init__(self, root: str | Path, kind: str, *, client_version: str | None = None,
                 session_factory: Callable[[str], object] | None = None) -> None:
        self.manifest, path, self.model_sha256 = _manifest(Path(root), kind, client_version)
        self.metadata = {"kind": kind, "model_sha256": self.model_sha256,
                         "code_version": self.manifest["code_version"],
                         "data_version": self.manifest["data_version"],
                         "validation": self.manifest["validation"],
                         "applicability": self.manifest["applicability"],
                         "client_versions": self.manifest["client_versions"]}
        prep = self.manifest.get("preprocess")
        if not isinstance(prep, dict):
            raise ModelUnavailable("Missing preprocess metadata")
        if type(prep.get("width")) is not int or type(prep.get("height")) is not int:
            raise ModelUnavailable("Model input dimensions must be integers")
        self.width = int(_number(prep["width"], minimum=1, maximum=4096))
        self.height = int(_number(prep["height"], minimum=1, maximum=4096))
        self.color = prep.get("color")
        self.resize = prep.get("resize")
        self.scale = _number(prep.get("scale"), minimum=0.0000001)
        if self.color not in {"RGB", "BGR"} or self.resize not in {"stretch", "letterbox"}:
            raise ModelUnavailable("Unsupported color or resize transform")
        mean, std = prep.get("mean"), prep.get("std")
        if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
            raise ModelUnavailable("Preprocess mean/std must have three channels")
        self.mean = np.array([_number(v) for v in mean], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([_number(v, minimum=0.0000001) for v in std], dtype=np.float32).reshape(1, 1, 3)
        pad = prep.get("pad_value", 114)
        if type(pad) is not int:
            raise ModelUnavailable("Pad value must be an integer")
        self.pad_value = int(_number(pad, maximum=255))
        self.max_frame_age_seconds = _number(self.manifest.get("max_frame_age_seconds"), minimum=0.001)
        if session_factory is None:
            try:
                import onnxruntime as ort
                session_factory = lambda filename: ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
            except ImportError as exc:
                raise ModelUnavailable("onnxruntime unavailable") from exc
        try:
            self.session = session_factory(str(path))
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as exc:
            raise ModelUnavailable(f"Unable to load ONNX model: {exc}") from exc
        try:
            input_shape = getattr(self.session.get_inputs()[0], "shape", None)
        except Exception as exc:
            raise ModelUnavailable(f"Unable to inspect ONNX input shape: {exc}") from exc
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) != 4:
                raise ModelUnavailable("Unexpected ONNX input rank")
            expected = (1, 3, self.height, self.width)
            if any(type(observed) is int and observed != wanted
                   for observed, wanted in zip(input_shape, expected, strict=True)):
                raise ModelUnavailable("ONNX input shape conflicts with package metadata")

    def _check_age(self, captured_at: float, now: float | None) -> None:
        captured_at = _number(captured_at)
        now = time.monotonic() if now is None else _number(now)
        if captured_at > now or now - captured_at > self.max_frame_age_seconds:
            raise StaleModelFrame("Image capture time is future or stale")

    def _inference_start(self, captured_at: float, now: float | None) -> tuple[float, float | None]:
        self._check_age(captured_at, now)
        return time.monotonic(), now

    def _inference_finish(self, captured_at: float, started: float,
                          initial_now: float | None) -> None:
        elapsed = time.monotonic() - started
        effective_now = None if initial_now is None else initial_now + elapsed
        self._check_age(captured_at, effective_now)

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
            raise ModelUnavailable("Expected nonempty uint8 BGR image")
        source_h, source_w = image.shape[:2]
        applicable = self.manifest["applicability"]
        if not (applicable["source_width_range"][0] <= source_w <= applicable["source_width_range"][1]
                and applicable["source_height_range"][0] <= source_h <= applicable["source_height_range"][1]):
            raise ModelUnavailable("Image resolution outside model applicability")
        try:
            if self.resize == "stretch":
                transformed = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                sx, sy, dx, dy = self.width / source_w, self.height / source_h, 0.0, 0.0
            else:
                factor = min(self.width / source_w, self.height / source_h)
                scaled_w, scaled_h = max(1, round(source_w * factor)), max(1, round(source_h * factor))
                dx, dy = (self.width - scaled_w) // 2, (self.height - scaled_h) // 2
                transformed = np.full((self.height, self.width, 3), self.pad_value, dtype=np.uint8)
                transformed[dy:dy + scaled_h, dx:dx + scaled_w] = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                sx, sy = scaled_w / source_w, scaled_h / source_h
            if self.color == "RGB":
                transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            raise ModelUnavailable(f"Cannot preprocess image: {exc}") from exc
        tensor = ((transformed.astype(np.float32) * self.scale - self.mean) / self.std).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(tensor), (sx, sy, float(dx), float(dy))

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        try:
            output = np.asarray(self.session.run([self.output_name], {self.input_name: tensor})[0], dtype=np.float32)
        except Exception as exc:
            raise ModelUnavailable(f"ONNX inference failed: {exc}") from exc
        if not np.all(np.isfinite(output)):
            raise ModelUnavailable("ONNX output contains non-finite values")
        return output


class SceneModel(_ONNXModel):
    """ONNX classifier with a package-defined unknown threshold and margin."""

    def __init__(self, root: str | Path, *, client_version: str | None = None,
                 session_factory: Callable[[str], object] | None = None) -> None:
        super().__init__(root, "scene", client_version=client_version, session_factory=session_factory)
        if not set(self.manifest["classes"]).issubset(SUPPORTED_SCENES):
            raise ModelUnavailable("Scene class is not mapped to a supported scene")
        output = self.manifest.get("output")
        if not isinstance(output, dict) or output.get("format") not in {"logits", "probabilities"}:
            raise ModelUnavailable("Unsupported scene output format")
        self.output_format = output["format"]
        self.min_confidence = _number(output.get("min_confidence"), maximum=1)
        self.min_margin = _number(output.get("min_margin"), maximum=1)

    def classify(self, image: np.ndarray, *, captured_at: float,
                 now: float | None = None) -> ScenePrediction:
        started, initial_now = self._inference_start(captured_at, now)
        tensor, _ = self._prepare(image)
        scores = self._run(tensor)
        self._inference_finish(captured_at, started, initial_now)
        labels = self.manifest["classes"]
        if scores.shape != (1, len(labels)):
            raise ModelUnavailable("Unexpected scene output shape")
        values = scores[0]
        if self.output_format == "logits":
            shifted = values - values.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
        else:
            if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, atol=1e-3):
                raise ModelUnavailable("Invalid scene probabilities")
            probabilities = values
        order = np.argsort(probabilities)[::-1]
        best, second = int(order[0]), float(probabilities[order[1]]) if len(order) > 1 else 0.0
        confidence = float(probabilities[best])
        unknown = confidence < self.min_confidence or confidence - second < self.min_margin
        self._inference_finish(captured_at, started, initial_now)
        return ScenePrediction(SCENE_UNKNOWN if unknown else labels[best], confidence, unknown,
                               "below_confidence_or_margin" if unknown else None,
                               captured_at, self.model_sha256)


class BuildingModel(_ONNXModel):
    """Detector for postprocessed xyxy rows or standard YOLO raw xywh scores."""

    def __init__(self, root: str | Path, *, client_version: str | None = None,
                 session_factory: Callable[[str], object] | None = None) -> None:
        super().__init__(root, "building", client_version=client_version, session_factory=session_factory)
        output = self.manifest.get("output")
        if not isinstance(output, dict) or output.get("format") not in {"xyxy_score_class", "yolo_xywh_classes"} or output.get("coordinates") != "input_pixels":
            raise ModelUnavailable("Unsupported building output format or coordinates")
        self.output_format = output["format"]
        self.min_confidence = _number(output.get("min_confidence"), maximum=1)
        self.nms_iou = _number(output.get("nms_iou"), maximum=1)
        overrides = output.get("class_min_confidence", {})
        if not isinstance(overrides, dict) or any(label not in self.manifest["classes"] for label in overrides):
            raise ModelUnavailable("Invalid per-class thresholds")
        self.class_thresholds = {label: _number(value, maximum=1) for label, value in overrides.items()}
        candidate_limit = output.get("max_candidates", 1000)
        detection_limit = output.get("max_detections", 300)
        if type(candidate_limit) is not int or type(detection_limit) is not int:
            raise ModelUnavailable("Detection limits must be integers")
        self.max_candidates = int(_number(candidate_limit, minimum=1, maximum=1000))
        self.max_detections = int(_number(detection_limit, minimum=1, maximum=300))

    def detect(self, image: np.ndarray, *, captured_at: float,
               now: float | None = None) -> tuple[BuildingObservation, ...]:
        started, initial_now = self._inference_start(captured_at, now)
        tensor, (sx, sy, dx, dy) = self._prepare(image)
        rows = self._run(tensor)
        self._inference_finish(captured_at, started, initial_now)
        if rows.ndim == 3 and rows.shape[0] == 1:
            rows = rows[0]
        if rows.ndim != 2:
            raise ModelUnavailable("Unexpected building output shape")
        if rows.shape[0] * rows.shape[1] > 600000:
            raise ModelUnavailable("Building output exceeds bounded candidate count")
        if self.output_format == "yolo_xywh_classes":
            expected = 4 + len(self.manifest["classes"])
            if rows.shape[0] == expected:
                rows = rows.T
            if rows.shape[1] != expected:
                raise ModelUnavailable("Unexpected YOLO output shape")
            if (rows[:, 4:] < 0).any() or (rows[:, 4:] > 1).any():
                raise ModelUnavailable("Invalid YOLO class scores")
            classes = rows[:, 4:].argmax(axis=1)
            scores = rows[np.arange(len(rows)), classes + 4]
            rows = np.column_stack((rows[:, 0] - rows[:, 2] / 2,
                                    rows[:, 1] - rows[:, 3] / 2,
                                    rows[:, 0] + rows[:, 2] / 2,
                                    rows[:, 1] + rows[:, 3] / 2,
                                    scores, classes))
        elif rows.shape[1] != 6:
            raise ModelUnavailable("Unexpected building output shape")
        height, width = image.shape[:2]
        candidates: list[BuildingObservation] = []
        for x0, y0, x1, y1, score, class_id in rows:
            if not 0 <= score <= 1:
                raise ModelUnavailable("Invalid building confidence")
            if class_id != int(class_id) or not 0 <= class_id < len(self.manifest["classes"]):
                raise ModelUnavailable("Invalid building class ID")
            label = self.manifest["classes"][int(class_id)]
            if score < self.class_thresholds.get(label, self.min_confidence):
                continue
            box = (max(0.0, min(float(width), (x0 - dx) / sx)),
                   max(0.0, min(float(height), (y0 - dy) / sy)),
                   max(0.0, min(float(width), (x1 - dx) / sx)),
                   max(0.0, min(float(height), (y1 - dy) / sy)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            candidates.append(BuildingObservation(label, box, float(score), captured_at, self.model_sha256))
        kept: list[BuildingObservation] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True)[:self.max_candidates]:
            if not any(candidate.kind == prior.kind and _iou(candidate.bbox, prior.bbox) > self.nms_iou for prior in kept):
                kept.append(candidate)
                if len(kept) >= self.max_detections:
                    break
        self._inference_finish(captured_at, started, initial_now)
        return tuple(kept)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(1e-9, area_a + area_b - overlap)


def infer_core_geometry(
    detections: Sequence[BuildingObservation], profile: Mapping[str, object], *,
    source_resolution: tuple[int, int], baseline_resolution: tuple[int, int] = (1280, 720),
) -> CoreGeometry | None:
    """Aggregate profile-selected visible targets; None means unknown core."""
    classes = profile.get("core_classes")
    if not isinstance(classes, dict) or not classes:
        return None
    count = profile.get("min_core_targets", 1)
    if type(count) is not int or count < 1:
        return None
    sw, sh = source_resolution
    bw, bh = baseline_resolution
    if min(sw, sh, bw, bh) <= 0:
        raise ValueError("Invalid geometry resolution")
    selected: list[tuple[BuildingObservation, float]] = []
    identities = {(d.frame_captured_at, d.model_sha256) for d in detections}
    if len(identities) > 1:
        return None
    for detection in detections:
        threshold = classes.get(detection.kind)
        if threshold is None:
            continue
        if detection.confidence >= _number(threshold, maximum=1):
            selected.append((detection, detection.confidence))
    if len(selected) < count:
        return None
    total = sum(weight for _, weight in selected)
    if total <= 0:
        return None
    x_scale, y_scale = bw / sw, bh / sh
    center = (sum(d.center[0] * weight for d, weight in selected) / total * x_scale,
              sum(d.center[1] * weight for d, weight in selected) / total * y_scale)
    padding = _number(profile.get("core_padding_pixels", 0), maximum=max(bw, bh))
    box = (max(0.0, min(d.bbox[0] * x_scale for d, _ in selected) - padding),
           max(0.0, min(d.bbox[1] * y_scale for d, _ in selected) - padding),
           min(float(bw), max(d.bbox[2] * x_scale for d, _ in selected) + padding),
           min(float(bh), max(d.bbox[3] * y_scale for d, _ in selected) + padding))
    evidence = tuple(sorted({d.kind for d, _ in selected}))
    return CoreGeometry(center, box, min(d.confidence for d, _ in selected), evidence)
