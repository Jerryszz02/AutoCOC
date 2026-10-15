"""Offline-first R/U model packages with source-frame ROI and age provenance.

Nothing in this module is wired into battle inputs. Dynamic packages must be
independently trained, evaluated, released and explicitly enabled later.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable

import cv2
import numpy as np


BUILDING_STATES = ("intact", "damaged", "rubble", "occlusion_unknown")
CARD_STATES = ("available", "selected", "exhausted", "unknown")
KINDS = frozenset({"r_building", "r_card", "u_position"})


class DynamicUnavailable(RuntimeError):
    """A frame or package lacks evidence needed for dynamic inference."""


@dataclass(frozen=True)
class DynamicFrame:
    session_id: str
    camera_epoch: str
    frame_id: str
    captured_at_monotonic: float  # capture request start, never crop/inference time
    client_version: str
    width: int
    height: int
    camera_stable: bool
    zoom_verified: bool


@dataclass(frozen=True)
class DynamicObservation:
    kind: str
    label: str
    bbox: tuple[float, float, float, float]  # original source-image pixels
    confidence: float
    session_id: str
    camera_epoch: str
    frame_id: str
    captured_at_monotonic: float
    roi_bbox: tuple[int, int, int, int]
    target_id: str | None
    model_sha256: str
    contract_sha256: str


def dynamic_contract_sha256(manifest: dict) -> str:
    fields = ("kind", "classes", "client_versions", "applicability", "preprocess",
              "output", "max_frame_age_seconds", "event_rules")
    raw = json.dumps({key: manifest.get(key) for key in fields}, sort_keys=True,
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite(value: object, *, lower: float = 0, upper: float = math.inf) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise DynamicUnavailable(f"Invalid dynamic numeric value: {value!r}")
    return float(value)


def _box(box: object, width: int, height: int) -> tuple[int, int, int, int]:
    if (not isinstance(box, (tuple, list)) or len(box) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise DynamicUnavailable("ROI requires four finite source-image coordinates")
    x0, y0, x1, y1 = (int(v) for v in box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise DynamicUnavailable("ROI is outside original source image")
    return x0, y0, x1, y1


class DynamicModel:
    def __init__(self, root: str | Path, *, client_version: str,
                 session_factory: Callable[[str], object] | None = None) -> None:
        self.root = Path(root).resolve()
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DynamicUnavailable("Dynamic manifest is unavailable") from exc
        m = self.manifest
        if not isinstance(m, dict) or m.get("schema_version") != 1 or m.get("kind") not in KINDS:
            raise DynamicUnavailable("Unsupported dynamic kind/schema")
        self.kind = m["kind"]
        self.classes = m.get("classes")
        expected = BUILDING_STATES if self.kind == "r_building" else CARD_STATES if self.kind == "r_card" else None
        if (not isinstance(self.classes, list) or not self.classes or
                any(not isinstance(item, str) or not item for item in self.classes) or
                len(set(self.classes)) != len(self.classes) or
                (expected is not None and tuple(self.classes) != expected) or
                (self.kind == "u_position" and not all(item == "troop_group" or item.startswith("hero:")
                                                       for item in self.classes))):
            raise DynamicUnavailable("Dynamic classes are not the declared state/position vocabulary")
        versions = m.get("client_versions")
        if not isinstance(versions, list) or client_version not in versions or not client_version:
            raise DynamicUnavailable("Dynamic package has no verified matching client version")
        self.client_version = client_version
        validation = m.get("validation")
        if (not isinstance(validation, dict) or
                validation.get("status") not in {"unvalidated", "evaluated"} or
                type(validation.get("prototype_only")) is not bool or
                (validation["status"] == "evaluated" and not validation.get("report_id"))):
            raise DynamicUnavailable("Dynamic validation/prototype provenance is missing")
        for field in ("snapshot_sha256", "split_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(validation.get(field, ""))) is None:
                raise DynamicUnavailable(f"Dynamic training {field} is missing")
        self.validation = validation
        applicability = m.get("applicability")
        if (not isinstance(applicability, dict) or applicability.get("roi_kind") != self.kind or
                type(applicability.get("requires_stable_camera")) is not bool or
                type(applicability.get("requires_verified_zoom")) is not bool or
                (self.kind in {"r_building", "u_position"} and
                 applicability["requires_stable_camera"] is not True)):
            raise DynamicUnavailable("Dynamic ROI applicability is missing")
        self.requires_stable_camera = applicability.get("requires_stable_camera") is True
        self.requires_verified_zoom = applicability.get("requires_verified_zoom") is True
        for key in ("source_width_range", "source_height_range"):
            bounds = applicability.get(key)
            if (not isinstance(bounds, list) or len(bounds) != 2 or
                    any(type(v) is not int or v < 1 for v in bounds) or bounds[0] > bounds[1]):
                raise DynamicUnavailable(f"Invalid dynamic {key}")
        self.applicability = applicability
        prep = m.get("preprocess")
        if (not isinstance(prep, dict) or type(prep.get("width")) is not int or
                type(prep.get("height")) is not int or prep["width"] < 1 or prep["height"] < 1 or
                prep.get("color") not in {"RGB", "BGR"} or prep.get("resize") != "stretch"):
            raise DynamicUnavailable("Dynamic preprocess must declare fixed stretch ROI")
        if self.kind != "u_position" and (prep["width"], prep["height"]) != (224, 224):
            raise DynamicUnavailable("R state ROI must be 224x224")
        self.preprocess = prep
        self.scale = _finite(prep.get("scale"), lower=1e-9)
        mean, std = prep.get("mean"), prep.get("std")
        if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
            raise DynamicUnavailable("Dynamic mean/std need three channels")
        self.mean = np.asarray([_finite(v) for v in mean], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray([_finite(v, lower=1e-9) for v in std], dtype=np.float32).reshape(1, 1, 3)
        output = m.get("output")
        if (not isinstance(output, dict) or output.get("format") not in
                ({"logits", "probabilities"} if expected else {"xyxy_score_class"})):
            raise DynamicUnavailable("Dynamic output format is unsupported")
        self.output = output
        self.min_confidence = _finite(output.get("min_confidence"), upper=1)
        self.max_detections = int(_finite(output.get("max_detections", 1 if expected else 100), lower=1, upper=300))
        self.max_age = _finite(m.get("max_frame_age_seconds"), lower=.001, upper=10)
        rules = m.get("event_rules")
        if not isinstance(rules, dict) or rules.get("consecutive_frames") != 2:
            raise DynamicUnavailable("Dynamic package requires frozen two-frame event rules")
        try:
            from .dynamic_tracking import DynamicTracker
            tracker = DynamicTracker(**{key: rules[key] for key in
                ("min_iou", "max_gap_seconds", "max_missed_frames",
                 "event_min_confidence", "max_frame_age_seconds")})
        except (KeyError, TypeError, ValueError) as exc:
            raise DynamicUnavailable("Dynamic package event rules are invalid") from exc
        if tracker.max_frame_age_seconds > self.max_age:
            raise DynamicUnavailable("Event-rule frame age exceeds model frame age")
        self.event_rules = rules
        filename, digest = m.get("model_file"), m.get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise DynamicUnavailable("Dynamic weight name and SHA-256 are required")
        model_path = (self.root / filename).resolve()
        if not model_path.is_relative_to(self.root) or not model_path.is_file():
            raise DynamicUnavailable("Dynamic weight path escapes package or is missing")
        with model_path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise DynamicUnavailable("Dynamic weight SHA-256 mismatch")
        self.model_sha256 = actual
        self.contract_sha256 = dynamic_contract_sha256(m)
        if session_factory is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise DynamicUnavailable("ONNX Runtime is unavailable") from exc
            session_factory = lambda path: ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.session = session_factory(str(model_path))
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise DynamicUnavailable("Dynamic model requires exactly one input")
        self.input_name = inputs[0].name

    def infer(self, image: np.ndarray, roi_bbox: tuple[int, int, int, int],
              frame: DynamicFrame, *, target_id: str | None = None,
              now_monotonic: float | None = None, offline_replay: bool = False) -> list[DynamicObservation]:
        now = time.monotonic() if now_monotonic is None else _finite(now_monotonic)
        capture = _finite(frame.captured_at_monotonic)
        if (not frame.session_id or not frame.camera_epoch or not frame.frame_id or
                frame.client_version != self.client_version or
                not offline_replay and (now < capture or now - capture > self.max_age)):
            raise DynamicUnavailable("Dynamic frame identity, version or capture-start age is invalid")
        if (self.requires_stable_camera and not frame.camera_stable or
                self.requires_verified_zoom and not frame.zoom_verified):
            raise DynamicUnavailable("Dynamic ROI lacks stable camera/verified zoom")
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or
                image.shape[1] != frame.width or image.shape[0] != frame.height):
            raise DynamicUnavailable("Dynamic image does not match original frame dimensions")
        for value, key in ((frame.width, "source_width_range"), (frame.height, "source_height_range")):
            low, high = self.applicability[key]
            if not low <= value <= high:
                raise DynamicUnavailable("Dynamic frame outside declared resolution")
        x0, y0, x1, y1 = _box(roi_bbox, frame.width, frame.height)
        if self.kind != "u_position" and (not isinstance(target_id, str) or not target_id):
            raise DynamicUnavailable("R state requires a stable target_id from reviewed/static evidence")
        crop = cv2.resize(image[y0:y1, x0:x1],
                          (self.preprocess["width"], self.preprocess["height"]),
                          interpolation=cv2.INTER_LINEAR)
        if self.preprocess["color"] == "RGB":
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = ((crop.astype(np.float32) * self.scale - self.mean) / self.std)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 1:
            raise DynamicUnavailable("Dynamic model returned multiple output tensors")
        raw = np.asarray(outputs[0])

        def observation(label: str, box: tuple[float, float, float, float], score: float) -> DynamicObservation:
            return DynamicObservation(self.kind, label, box, score, frame.session_id,
                                      frame.camera_epoch, frame.frame_id, capture,
                                      (x0, y0, x1, y1), target_id,
                                      self.model_sha256, self.contract_sha256)

        if self.kind != "u_position":
            if raw.shape != (1, len(self.classes)) or not np.isfinite(raw).all():
                raise DynamicUnavailable("R output must be one finite score per declared state")
            values = raw[0].astype(np.float64)
            if self.output["format"] == "logits":
                values = np.exp(values - values.max())
                values /= values.sum()
            elif (values < 0).any() or abs(values.sum() - 1) > .001:
                raise DynamicUnavailable("R probability vector is invalid")
            index = int(values.argmax())
            score = float(values[index])
            return [observation(self.classes[index], (x0, y0, x1, y1), score)] if score >= self.min_confidence else []
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 6 or not np.isfinite(raw).all():
            raise DynamicUnavailable("U output must be finite [1,N,6]")
        if raw.shape[1] > self.max_detections:
            raise DynamicUnavailable("U output exceeds frozen max_detections")
        result = []
        for left, top, right, bottom, score, class_index in raw[0]:
            if not 0 <= score <= 1:
                raise DynamicUnavailable("U confidence is outside [0,1]")
            if score < self.min_confidence:
                continue
            if (class_index != int(class_index) or not 0 <= int(class_index) < len(self.classes) or
                    not 0 <= left < right <= self.preprocess["width"] or
                    not 0 <= top < bottom <= self.preprocess["height"]):
                raise DynamicUnavailable("U box/class is outside ROI output space")
            box = (x0 + left * (x1 - x0) / self.preprocess["width"],
                   y0 + top * (y1 - y0) / self.preprocess["height"],
                   x0 + right * (x1 - x0) / self.preprocess["width"],
                   y0 + bottom * (y1 - y0) / self.preprocess["height"])
            result.append(observation(self.classes[int(class_index)], box, float(score)))
        return result
