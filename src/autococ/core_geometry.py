"""Opt-in, bounded grouping of visible core targets in one preview frame."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .model_vision import BuildingObservation


_FIELDS = {"enabled", "link_distance_pixels", "separation_margin_pixels",
           "min_targets_per_core", "max_cores", "max_selected_detections", "max_candidates"}
_EPS = 1e-6


def validate_multi_core_profile(profile: Mapping[str, object]) -> dict | None:
    """Missing/disabled means the original single-core planner, with no defaults."""
    if "multi_core" not in profile:
        return None
    raw = profile["multi_core"]
    if not isinstance(raw, dict) or type(raw.get("enabled")) is not bool:
        raise ValueError("multi_core.enabled must be an explicit boolean")
    if raw["enabled"] is False:
        if set(raw) != {"enabled"}:
            raise ValueError("Disabled multi_core accepts only enabled=false")
        return None
    if set(raw) != _FIELDS:
        raise ValueError("Enabled multi_core requires all bounded parameters and no extras")
    for key, lower, upper in (("link_distance_pixels", 20, 400),
                              ("separation_margin_pixels", 1, 100)):
        value = raw[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid multi_core.{key}")
    for key, lower, upper in (("min_targets_per_core", 1, 4), ("max_cores", 2, 3),
                              ("max_selected_detections", 2, 24), ("max_candidates", 1, 24)):
        value = raw[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid multi_core.{key}")
    if raw["min_targets_per_core"] * 2 > raw["max_selected_detections"]:
        raise ValueError("Multi-core detection budget cannot form two accepted groups")
    classes = profile.get("core_classes")
    if (not isinstance(classes, dict) or not classes or
            any(not isinstance(name, str) or not name or type(threshold) not in (int, float)
                or not math.isfinite(threshold) or not 0 < threshold <= 1
                for name, threshold in classes.items())):
        raise ValueError("Enabled multi_core requires finite positive core class thresholds")
    padding = profile.get("core_padding_pixels", 0)
    if type(padding) not in (int, float) or not math.isfinite(padding) or not 0 <= padding <= 1280:
        raise ValueError("Invalid multi_core core padding")
    return dict(raw)


def infer_multi_core_geometry(
    detections: Sequence[BuildingObservation], profile: Mapping[str, object], *,
    captured_at: float, model_sha256: str, source_resolution: tuple[int, int],
    baseline_resolution: tuple[int, int] = (1280, 720),
) -> tuple[dict, ...]:
    """Reject ambiguous bridges rather than joining unrelated visible targets."""
    settings = validate_multi_core_profile(profile)
    if settings is None:
        raise ValueError("Multi-core geometry is not enabled")
    sw, sh = source_resolution
    bw, bh = baseline_resolution
    if (any(type(value) is not int or value <= 0 for value in (sw, sh, bw, bh)) or
            type(captured_at) not in (int, float) or not math.isfinite(captured_at) or
            not isinstance(model_sha256, str) or not model_sha256):
        raise ValueError("Invalid multi-core frame or model identity")
    if len(detections) > 300:
        raise ValueError("Multi-core input detection count exceeds model output bound")
    selected = []
    for item in detections:
        if (not isinstance(item, BuildingObservation) or
                type(item.frame_captured_at) not in (int, float) or
                not math.isfinite(item.frame_captured_at) or
                item.frame_captured_at != captured_at or item.model_sha256 != model_sha256 or
                not isinstance(item.kind, str) or not item.kind or
                type(item.confidence) not in (int, float) or not math.isfinite(item.confidence)
                or not 0 <= item.confidence <= 1 or
                not isinstance(item.bbox, (tuple, list)) or len(item.bbox) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v)
                                           for v in item.bbox) or
                not (0 <= item.bbox[0] < item.bbox[2] <= sw and
                     0 <= item.bbox[1] < item.bbox[3] <= sh)):
            raise ValueError("Multi-core detections are not from one valid current frame/model")
        threshold = profile["core_classes"].get(item.kind)
        if threshold is not None and item.confidence >= threshold:
            if item.confidence <= 0:
                raise ValueError("Multi-core target has zero weight")
            selected.append(item)
    if not 2 <= len(selected) <= settings["max_selected_detections"]:
        raise ValueError("Multi-core target count is unsupported")
    xscale, yscale = bw / sw, bh / sh
    selected.sort(key=lambda item: (item.center[0] * xscale, item.center[1] * yscale,
                                    item.kind, item.bbox, -item.confidence))
    points = [(item.center[0] * xscale, item.center[1] * yscale) for item in selected]
    link = float(settings["link_distance_pixels"])
    margin = float(settings["separation_margin_pixels"])
    graph = [set() for _ in selected]
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            distance = math.dist(points[i], points[j])
            if abs(distance - link) <= _EPS or abs(distance - (link + margin)) <= _EPS:
                raise ValueError("Multi-core target distance lies on a grouping boundary")
            if distance < link:
                graph[i].add(j)
                graph[j].add(i)
    groups = []
    unseen = set(range(len(selected)))
    while unseen:
        stack = [min(unseen)]
        component = set()
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index)
            stack.extend(sorted(graph[index] - component, reverse=True))
        unseen -= component
        groups.append(sorted(component))
    if not 2 <= len(groups) <= settings["max_cores"]:
        raise ValueError("Multi-core group count is unsupported")
    if any(len(group) < settings["min_targets_per_core"] for group in groups):
        raise ValueError("A multi-core group has too few visible targets")
    for group in groups:
        if any(math.dist(points[i], points[j]) >= link - _EPS
               for i in group for j in group if i < j):
            raise ValueError("Multi-core bridge chain has ambiguous diameter")
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            if min(math.dist(points[i], points[j]) for i in groups[a] for j in groups[b]) <= link + margin + _EPS:
                raise ValueError("Multi-core groups lack clear separation")
    padding = float(profile.get("core_padding_pixels", 0))
    cores = []
    for group in groups:
        members = [selected[i] for i in group]
        total = sum(item.confidence for item in members)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Multi-core group has invalid target weights")
        center = (sum(item.center[0] * item.confidence for item in members) / total * xscale,
                  sum(item.center[1] * item.confidence for item in members) / total * yscale)
        box = (max(0., min(item.bbox[0] * xscale for item in members) - padding),
               max(0., min(item.bbox[1] * yscale for item in members) - padding),
               min(float(bw), max(item.bbox[2] * xscale for item in members) + padding),
               min(float(bh), max(item.bbox[3] * yscale for item in members) + padding))
        cores.append({"center": center, "bbox": box,
                      "confidence": min(item.confidence for item in members),
                      "evidence": tuple(sorted({item.kind for item in members})),
                      "detections": tuple({"kind": item.kind, "bbox": item.bbox,
                                           "confidence": item.confidence,
                                           "frame_captured_at": item.frame_captured_at,
                                           "model_sha256": item.model_sha256} for item in members)})
    cores.sort(key=lambda core: (core["center"][0], core["center"][1], core["bbox"]))
    return tuple(cores)
