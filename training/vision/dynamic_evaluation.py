"""Offline R/U event evaluation; never a Windows action-latency release claim."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics

import cv2
import numpy as np

from autococ.dynamic_tracking import DynamicTracker
from autococ.dynamic_vision import DynamicFrame, DynamicModel, DynamicUnavailable

from .data import DatasetError, image_path
from .dynamic_data import EVENT_BY_KIND, verify_dynamic_windows


def _iou(a, b) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    denominator = ((a[2] - a[0]) * (a[3] - a[1]) +
                   (b[2] - b[0]) * (b[3] - b[1]) - overlap)
    return overlap / denominator if denominator > 0 else 0.


def evaluate_dynamic(model: DynamicModel, root: Path, windows: list[dict], *,
                     prototype_only: bool, split: str = "test") -> dict:
    if split not in {"train", "val", "test"}:
        raise DatasetError("Invalid dynamic evaluation split")
    inventory = verify_dynamic_windows(root, windows, kind=model.kind, prototype_only=prototype_only)
    selected = [window for window in windows if window["split"] == split]
    if not selected:
        raise DatasetError(f"Dynamic evaluation split {split!r} has no reviewed windows")
    counts = Counter({"tp": 0, "fp": 0, "fn": 0, "early": 0,
                      "inference_failures": 0, "unknown_frames": 0})
    delays, attempts = [], []
    for window in selected:
        tracker = DynamicTracker(**{key: model.event_rules[key] for key in
            ("min_iou", "max_gap_seconds", "max_missed_frames",
             "event_min_confidence", "max_frame_age_seconds")})
        predicted_events = []
        truth_by_frame = {}
        if model.kind == "u_position":
            for truth in window["truth_boxes"]:
                truth_by_frame.setdefault(truth["frame_id"], []).append(truth)
        for frame_row in window["frames"]:
            source_image = (frame_row["image"] if frame_row["parent_image_sha256"] == frame_row["sha256"]
                            else frame_row["parent_image"])
            path = image_path(root, source_image)
            try:
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise DynamicUnavailable("Reviewed dynamic frame cannot be decoded")
                frame = DynamicFrame(frame_row["session_id"], frame_row["camera_epoch"],
                                     frame_row["frame_id"], frame_row["captured_at_monotonic"],
                                     frame_row["client_version"], image.shape[1], image.shape[0],
                                     frame_row.get("camera_stable") is True,
                                     frame_row.get("zoom_verified") is True)
                observations = model.infer(image, frame_row["roi_bbox"], frame,
                                           target_id=frame_row.get("target_id"), offline_replay=True)
                step = tracker.update(frame, observations, offline_replay=True)
                predicted_events.extend(step.events)
                if not observations:
                    counts["unknown_frames"] += 1
            except (OSError, ValueError, DynamicUnavailable) as exc:
                counts["inference_failures"] += 1
                tracker = DynamicTracker(**{key: model.event_rules[key] for key in
                    ("min_iou", "max_gap_seconds", "max_missed_frames",
                     "event_min_confidence", "max_frame_age_seconds")})  # failure breaks continuity
                attempts.append({"window_id": window["window_id"], "frame_id": frame_row["frame_id"],
                                 "status": "inference_failure", "reason": str(exc)})
                observations = []
            if model.kind == "u_position":
                truth = truth_by_frame.get(frame_row["frame_id"], [])
                used = set()
                for prediction in sorted(observations, key=lambda item: item.confidence, reverse=True):
                    candidates = [(index, _iou(prediction.bbox, item["bbox"]))
                                  for index, item in enumerate(truth)
                                  if index not in used and item["label"] == prediction.label]
                    matches = [(index, score) for index, score in candidates if score >= .5]
                    if matches:
                        index, _ = max(matches, key=lambda item: item[1])
                        used.add(index)
                        counts["tp"] += 1
                    else:
                        counts["fp"] += 1
                counts["fn"] += len(truth) - len(used)
        if model.kind in EVENT_BY_KIND:
            confirmed = [event for event in predicted_events if event.kind == window["action_kind"]]
            first = confirmed[0] if confirmed else None
            positive = window["truth_event"]
            early = first is not None and positive and first.confirmed_captured_at_monotonic < window["event_at_monotonic"]
            if first is not None and not early and positive:
                counts["tp"] += 1
                delays.append(first.confirmed_captured_at_monotonic - window["event_at_monotonic"])
            elif positive:
                counts["fn"] += 1
            if first is not None and (not positive or early):
                counts["fp"] += 1
            counts["early"] += int(early)
            attempts.append({"window_id": window["window_id"], "truth_event": positive,
                             "predicted_event": first is not None, "early": early,
                             "first_confirmed_frame_id": first.confirmed_frame_id if first else None})
    denominator = counts["tp"] + counts["fn"]
    return {"schema_version": 1, "kind": model.kind, "split": split,
            "status": "offline_only", "prototype_only": prototype_only,
            "model_sha256": model.model_sha256, "model_contract_sha256": model.contract_sha256,
            "event_rules": model.event_rules,
            "inventory": inventory, "evaluated_windows": len(selected),
            "evaluated_groups": inventory["groups_by_split"][split],
            "metrics": {**dict(counts), "event_recall": counts["tp"] / denominator if denominator else None,
                        "confirmation_delay_median_seconds": statistics.median(delays) if delays else None,
                        "confirmation_delay_max_seconds": max(delays) if delays else None},
            "attempts": attempts,
            "limits": ["Offline replay bypasses real-time frame-age enforcement",
                       "No capture-to-input or real Windows action latency is measured",
                       "No real model or battle release is implied by this report"]}
