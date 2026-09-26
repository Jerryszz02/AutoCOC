"""Versioned enemy-building recognition with positive destruction evidence.

No building is inferred from a gap in a later screenshot. Unsupported levels,
zoom, occlusion and missing templates remain unknown to callers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .errors import SceneError


CATALOG_ROOT = Path(__file__).resolve().parents[2] / "assets" / "catalogs"
BUILDING_TYPES = frozenset({"air_defense", "spell_factory"})


def _validated(entry: dict) -> bool:
    validation = entry.get("validation")
    return (isinstance(validation, dict) and validation.get("status") == "validated" and
            isinstance(validation.get("source_alive"), str) and
            isinstance(validation.get("source_destroyed"), str) and
            isinstance(validation.get("heldout_alive"), str) and
            isinstance(validation.get("heldout_destroyed"), str) and
            validation["source_alive"] != validation["heldout_alive"] and
            validation["source_destroyed"] != validation["heldout_destroyed"])


def building_manifest(root: Path = CATALOG_ROOT) -> dict:
    path = root / "building_templates.json"
    if not path.is_file():
        return {"client": "unknown", "levels": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("levels"), list):
        raise ValueError("Invalid building template manifest")
    return data


def _client_matches(manifest: dict, client_version: str | None) -> bool:
    # None is reserved for explicit offline replay. Production passes the
    # independently read package version, including "unknown" on failure.
    return (client_version is None or isinstance(client_version, str)
            and client_version not in {"", "unknown"}
            and manifest.get("client_version") == client_version)


def building_coverage(root: Path = CATALOG_ROOT, *, client_version: str | None = None) -> dict[str, dict[str, object]]:
    result = {kind: {"recognition": "unavailable", "levels": []}
              for kind in BUILDING_TYPES}
    manifest = building_manifest(root)
    if not _client_matches(manifest, client_version):
        return {kind: {**value, "reason": "client_version_unverified_or_mismatched"}
                for kind, value in result.items()}
    for entry in manifest["levels"]:
        kind = entry.get("type")
        if kind not in result or not _validated(entry):
            continue
        files = [entry.get("alive"), entry.get("destroyed")]
        if all(isinstance(file, str) and (root / file).resolve().is_relative_to(root.resolve())
               and (root / file).is_file() for file in files):
            result[kind]["levels"].append(entry.get("level"))
            result[kind]["recognition"] = "sampled"
    return result


def _read(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def _iou(first: Sequence[int], second: Sequence[int]) -> float:
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    area = max(0, x1 - x0) * max(0, y1 - y0)
    size_a = (first[2] - first[0]) * (first[3] - first[1])
    size_b = (second[2] - second[0]) * (second[3] - second[1])
    return area / max(1, size_a + size_b - area)


def _matches(image: np.ndarray, template: np.ndarray, *, threshold: float) -> list[tuple[float, list[int]]]:
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        return []
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if gray.std() < 12:
        return []
    score = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    peaks = cv2.dilate(score, np.ones((9, 9), dtype=np.uint8))
    ys, xs = np.where((score >= threshold) & (score == peaks))
    return [(float(score[y, x]), [int(x), int(y), int(x + template.shape[1]), int(y + template.shape[0])])
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True) if math.isfinite(float(score[y, x]))]


def _id(kind: str, point: Sequence[int]) -> str:
    return f"{kind}:{round(point[0] / 20)}:{round(point[1] / 20)}"


def detect_buildings(
    screenshot_path: str | Path, *, previous: Sequence[dict] = (),
    root: Path = CATALOG_ROOT, baseline_resolution: tuple[int, int] = (1280, 720),
    threshold: float = .94, client_version: str | None = None,
) -> list[dict[str, object]]:
    """Return current positive matches and unresolved previously tracked sites.

    Destruction requires a *fresh* rubble template match at the tracked site
    and positive evidence that the camera stayed fixed since the alive frame.
    A never-seen building is only admitted from an alive template. The caller
    must use the same fixed battle zoom for all frames.
    """
    manifest = building_manifest(root)
    if not _client_matches(manifest, client_version) or not manifest["levels"]:
        return []
    frame = Path(screenshot_path)
    source = _read(frame)
    if source is None:
        raise ValueError(f"Unable to decode building screenshot: {frame}")
    image = cv2.resize(source, baseline_resolution, interpolation=cv2.INTER_AREA)
    entries = manifest["levels"]
    previous = tuple(previous)
    camera_motion = None
    anchor_frame = None
    if previous:
        frames = {item.get("anchor_frame", item.get("frame"))
                  for item in previous if isinstance(item, dict)}
        if len(frames) == 1:
            anchor_frame = next(iter(frames))
            try:
                from .terrain import measure_camera_motion
                camera_motion = measure_camera_motion(anchor_frame, frame)
            except (OSError, ValueError, TypeError, SceneError):
                camera_motion = None
        if camera_motion is not None and camera_motion.get("stationary") is False:
            # A measured pan/zoom invalidates coordinates. An *uncertain*
            # comparison instead retains unknown sites, always anchored to
            # their last positive alive frame for later reevaluation.
            previous = ()
    results: list[dict[str, object]] = []
    for entry in entries:
        kind = entry.get("type")
        if (kind not in BUILDING_TYPES or not _validated(entry) or
                not (type(entry.get("level")) is int and entry["level"] > 0 or
                     entry.get("level") == "unknown_variant")):
            continue
        relative = entry.get("alive")
        if not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            continue
        template = _read(path)
        if template is None:
            continue
        for score, box in _matches(image, template, threshold=threshold):
            point = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
            results.append({"building_id": _id(kind, point), "type": kind,
                            "level": entry["level"], "point": point, "bbox": box,
                            "confidence": round(score, 5), "state": "alive", "status": "alive",
                            "frame": str(frame), "anchor_frame": str(frame), "evidence": {"template": str(path),
                            "method": "positive_alive_template", "zoom": entry.get("zoom"),
                            "template_client_version": manifest.get("client_version", "unknown"),
                            "version_verified": client_version is not None}})
    kept: list[dict[str, object]] = []
    for match in sorted(results, key=lambda item: item["confidence"], reverse=True):
        if not any(_iou(match["bbox"], prior["bbox"]) > .3 for prior in kept):
            kept.append(match)
    for prior in previous:
        if prior.get("type") not in BUILDING_TYPES or prior.get("state") == "destroyed":
            continue
        point = prior.get("point")
        box = prior.get("bbox")
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or not isinstance(box, (list, tuple)) or len(box) != 4):
            continue
        if camera_motion is None or camera_motion.get("stationary") is not True:
            kept.append({"building_id": prior.get("building_id") or _id(prior["type"], point),
                         "type": prior["type"], "level": prior.get("level"),
                         "point": list(point), "bbox": list(box), "confidence": None,
                         "state": "unknown", "status": "unknown", "frame": str(frame),
                         "anchor_frame": prior.get("anchor_frame", prior.get("frame")),
                         "evidence": {"method": "camera_unverified", "camera_motion": camera_motion}})
            continue
        current = next((item for item in kept if item["type"] == prior["type"]
                        and math.dist(item["point"], point) <= 25), None)
        if current is not None:
            current["building_id"] = prior.get("building_id") or current["building_id"]
            continue
        rubble: list[tuple[float, str]] = []
        for entry in entries:
            if (not _validated(entry) or entry.get("type") != prior["type"] or
                    entry.get("level") != prior.get("level")):
                continue
            relative = entry.get("destroyed")
            if not isinstance(relative, str):
                continue
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                continue
            template = _read(path)
            if template is None:
                continue
            pad = max(20, int(max(template.shape[:2]) * .4))
            left, top = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
            right, bottom = min(image.shape[1], int(box[2]) + pad), min(image.shape[0], int(box[3]) + pad)
            for score, candidate in _matches(image[top:bottom, left:right], template, threshold=threshold):
                center = [left + (candidate[0] + candidate[2]) // 2,
                          top + (candidate[1] + candidate[3]) // 2]
                if math.dist(center, point) <= max(25, pad):
                    rubble.append((score, str(path)))
        state = "destroyed" if rubble else "unknown"
        best = max(rubble, default=(0.0, None))
        kept.append({"building_id": prior.get("building_id") or _id(prior["type"], point),
                     "type": prior["type"], "level": prior.get("level"), "point": list(point),
                     "bbox": list(box), "confidence": round(best[0], 5) if rubble else None,
                     "state": state, "status": state, "frame": str(frame),
                     "anchor_frame": prior.get("anchor_frame", prior.get("frame")),
                     "evidence": {"method": "positive_rubble_template" if rubble else "unresolved_after_absence",
                                  "template": best[1], "previous_frame": prior.get("frame"),
                                  "camera_motion": camera_motion,
                                  "template_client_version": manifest.get("client_version", "unknown"),
                                  "version_verified": client_version is not None}})
    return sorted(kept, key=lambda item: (item["type"], item["point"][1], item["point"][0]))
