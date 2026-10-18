"""Describe visible buildings for finite prebattle choices, without game-state guesses."""

from __future__ import annotations

from collections import Counter
import math
from typing import Mapping, Sequence

from .model_vision import BuildingObservation


def describe_visible_layout(
    detections: Sequence[BuildingObservation], *, classes: Sequence[str],
    frame_id: str, captured_at: float, model_sha256: str,
    source_resolution: tuple[int, int], baseline_resolution: tuple[int, int],
    entries: Mapping[int, tuple[float, float]],
) -> dict:
    """Keep missing detections unknown; distances are screen pixels, not attack ranges."""
    if (not isinstance(classes, (tuple, list)) or not classes
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)
            or not frame_id or not model_sha256
            or type(captured_at) not in (int, float) or not math.isfinite(captured_at)
            or len(detections) > 300):
        raise ValueError("Invalid visible-layout source contract")
    dimensions = (*source_resolution, *baseline_resolution)
    if len(dimensions) != 4 or any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("Invalid visible-layout resolution")
    sw, sh = source_resolution
    bw, bh = baseline_resolution
    buildings = []
    for item in detections:
        box = item.bbox
        if (item.kind not in classes or item.model_sha256 != model_sha256
                or type(item.frame_captured_at) not in (int, float)
                or not math.isfinite(item.frame_captured_at) or item.frame_captured_at != captured_at
                or type(item.confidence) not in (int, float)
                or not math.isfinite(item.confidence) or not 0 <= item.confidence <= 1
                or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or not 0 <= box[0] < box[2] <= sw or not 0 <= box[1] < box[3] <= sh):
            raise ValueError("Building observation differs from the current layout source")
        scaled = [box[0] * bw / sw, box[1] * bh / sh, box[2] * bw / sw, box[3] * bh / sh]
        buildings.append({"kind": item.kind, "bbox": scaled,
                          "center": [(scaled[0] + scaled[2]) / 2, (scaled[1] + scaled[3]) / 2],
                          "confidence": item.confidence})
    buildings.sort(key=lambda item: (item["kind"], *item["bbox"]))
    by_entry = {}
    for edge, point in entries.items():
        if (type(edge) is not int or len(point) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
                or not 0 <= point[0] < bw or not 0 <= point[1] < bh):
            raise ValueError("Invalid candidate entry point")
        nearest = {}
        for kind in classes:
            matches = [item for item in buildings if item["kind"] == kind]
            if not matches:
                nearest[kind] = None
                continue
            target = min(matches, key=lambda item: math.dist(point, item["center"]))
            nearest[kind] = {"distance_pixels": round(math.dist(point, target["center"]), 2),
                             "bbox": target["bbox"], "confidence": target["confidence"]}
        by_entry[str(edge)] = {"point": list(point), "nearest_observed_by_class": nearest}
    return {"schema_version": 1, "frame_id": frame_id, "captured_at": captured_at,
            "model_sha256": model_sha256, "coordinate_space": "baseline_pixels",
            "resolution": list(baseline_resolution), "supported_classes": list(classes),
            "observed_counts": dict(sorted(Counter(item["kind"] for item in buildings).items())),
            "buildings": buildings, "entries": by_entry,
            "coverage": "visible_detections_only", "missing_detection_means_absent": False,
            "unknown": ["undetected_buildings", "hidden_traps", "building_levels",
                        "remaining_hitpoints", "air_sweeper_direction", "attack_ranges", "tile_scale"]}
