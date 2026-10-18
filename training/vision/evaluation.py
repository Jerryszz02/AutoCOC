"""Independent raw-output AP protocol, separate from runtime detector settings."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from autococ.model_vision import ModelUnavailable


DEFAULT_CONFIG = Path(__file__).with_name("evaluation-v1.json")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or not config.get("protocol_id"):
        raise ValueError("Unsupported evaluation protocol")
    for key in ("score_floor", "nms_iou"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("max_candidates", "max_detections"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    values = config.get("iou_thresholds")
    if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not 0 < v <= 1 for v in values):
        raise ValueError("Invalid iou_thresholds")
    config["sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def iou(a: tuple[float, ...] | list[float], b: tuple[float, ...] | list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / max(1e-9, area_a + area_b - overlap)


def _valid_box(box: Any) -> bool:
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            and box[2] > box[0] and box[3] > box[1])


def _image(root: Path, relative: Any) -> np.ndarray:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Image path must be relative")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Image outside reviewed root")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {relative}")
    return image


def raw_predictions(model, image: np.ndarray, config: dict) -> tuple[list[dict], dict]:
    """Decode _run output with evaluation limits, never reading runtime thresholds."""
    tensor, (sx, sy, dx, dy) = model._prepare(image)
    rows = model._run(tensor)
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2 or rows.size > 600000:
        raise ModelUnavailable("Unexpected or oversized raw building output")
    classes = model.manifest["classes"]
    decoded: list[tuple[float, str, tuple[float, float, float, float]]] = []
    if model.output_format == "yolo_xywh_classes":
        expected = 4 + len(classes)
        if rows.shape[0] == expected:
            rows = rows.T
        if rows.shape[1] != expected or (rows[:, 4:] < 0).any() or (rows[:, 4:] > 1).any():
            raise ModelUnavailable("Invalid raw YOLO output")
        for row in rows:
            x, y, w, h = map(float, row[:4])
            for class_id, score in enumerate(row[4:]):
                if score >= config["score_floor"]:
                    decoded.append((float(score), classes[class_id],
                                    (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    elif model.output_format == "xyxy_score_class":
        if rows.shape[1] != 6:
            raise ModelUnavailable("Invalid raw xyxy output")
        for row in rows:
            x0, y0, x1, y1, score, class_id = map(float, row)
            if not 0 <= score <= 1 or not class_id.is_integer() or not 0 <= class_id < len(classes):
                raise ModelUnavailable("Invalid raw building score/class")
            if score >= config["score_floor"]:
                decoded.append((score, classes[int(class_id)], (x0, y0, x1, y1)))
    else:
        raise ModelUnavailable("Unsupported raw output format")
    height, width = image.shape[:2]
    candidates = []
    for score, label, (x0, y0, x1, y1) in decoded:
        box = (max(0., min(float(width), (x0 - dx) / sx)),
               max(0., min(float(height), (y0 - dy) / sy)),
               max(0., min(float(width), (x1 - dx) / sx)),
               max(0., min(float(height), (y1 - dy) / sy)))
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append({"class": label, "bbox": box, "confidence": score})
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    candidate_count = len(candidates)
    candidates = candidates[:config["max_candidates"]]
    kept = []
    by_class: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_class[candidate["class"]].append(candidate)
    for entries in by_class.values():
        boxes = np.asarray([item["bbox"] for item in entries], dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        remaining = np.arange(len(entries))  # each class already sorted by confidence
        while len(remaining):
            first = int(remaining[0])
            kept.append(entries[first])
            others = remaining[1:]
            if not len(others):
                break
            x0 = np.maximum(boxes[first, 0], boxes[others, 0])
            y0 = np.maximum(boxes[first, 1], boxes[others, 1])
            x1 = np.minimum(boxes[first, 2], boxes[others, 2])
            y1 = np.minimum(boxes[first, 3], boxes[others, 3])
            overlap = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
            ratio = overlap / np.maximum(1e-9, areas[first] + areas[others] - overlap)
            remaining = others[ratio <= config["nms_iou"]]
    kept.sort(key=lambda item: item["confidence"], reverse=True)
    detection_count = len(kept)
    kept = kept[:config["max_detections"]]
    return kept, {"decoded_candidates": candidate_count,
                  "candidate_truncation": max(0, candidate_count - config["max_candidates"]),
                  "detection_truncation": max(0, detection_count - config["max_detections"])}


def _ap(records: list[tuple[float, bool]], truth_count: int) -> float | None:
    if truth_count == 0:
        return None
    tp = fp = 0
    curve = []
    for _, matched in records:
        tp += int(matched)
        fp += int(not matched)
        curve.append((tp / truth_count, tp / (tp + fp)))
    return sum(max((precision for recall, precision in curve if recall >= point / 100), default=0.)
               for point in range(101)) / 101


def score_predictions(predictions: list[list[dict]], truths: list[list[dict]], sizes: list[tuple[int, int]],
                      classes: list[str], thresholds: list[float]) -> dict:
    """Global confidence ranking and fresh one-to-one matches per IoU threshold."""
    if not (len(predictions) == len(truths) == len(sizes)):
        raise ValueError("Prediction/truth/size row counts differ")
    results = {}
    for label in classes:
        truth_count = sum(sum(item["class"] == label for item in boxes) for boxes in truths)
        ranked = sorted(((pred["confidence"], image_index, pred["bbox"])
                         for image_index, boxes in enumerate(predictions) for pred in boxes if pred["class"] == label),
                        key=lambda item: item[0], reverse=True)
        ap_by_threshold = {}
        errors = []
        corners = []
        for threshold in thresholds:
            matched: set[tuple[int, int]] = set()
            records = []
            for confidence, image_index, box in ranked:
                choices = [(i, iou(box, truth["bbox"])) for i, truth in enumerate(truths[image_index])
                           if truth["class"] == label and (image_index, i) not in matched]
                best = max(choices, key=lambda item: item[1], default=None)
                success = best is not None and best[1] >= threshold
                records.append((confidence, success))
                if success:
                    matched.add((image_index, best[0]))
                    if abs(threshold - .5) < 1e-9:
                        actual = truths[image_index][best[0]]["bbox"]
                        longest = max(sizes[image_index])
                        errors.append(math.dist(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                                                ((actual[0] + actual[2]) / 2, (actual[1] + actual[3]) / 2)) / longest)
                        corners.append(sum(abs(a - b) for a, b in zip(box, actual, strict=True)) / (4 * longest))
            ap_by_threshold[f"{threshold:.2f}"] = _ap(records, truth_count)
        values = [value for value in ap_by_threshold.values() if value is not None]
        results[label] = {"truth_instances": truth_count, "predictions": len(ranked),
                          "ap50": ap_by_threshold.get("0.50"),
                          "ap50_95": sum(values) / len(values) if values else None,
                          "ap_by_iou": ap_by_threshold,
                          "center_error_p95_normalized": float(np.percentile(errors, 95)) if errors else None,
                          "corner_error_p95_normalized": float(np.percentile(corners, 95)) if corners else None,
                          "matched_at_50": len(errors)}
    defined50 = [item["ap50"] for item in results.values() if item["ap50"] is not None]
    defined5095 = [item["ap50_95"] for item in results.values() if item["ap50_95"] is not None]
    return {"per_class": results, "mAP50": sum(defined50) / len(defined50) if defined50 else None,
            "mAP50_95": sum(defined5095) / len(defined5095) if defined5095 else None}


def evaluate_independent(model, rows: list[dict], root: Path, config: dict) -> dict:
    classes = model.manifest["classes"]
    predictions, truths, sizes = [], [], []
    attempts, errors = [], []
    group_sets = defaultdict(set)
    subset_counts = Counter()
    subset_truth: dict[str, Counter] = defaultdict(Counter)
    subset_groups: dict[str, set[str]] = defaultdict(set)
    truncation = Counter()
    for index, row in enumerate(rows):
        truth = row.get("boxes") if isinstance(row, dict) else None
        if not isinstance(truth, list) or not all(isinstance(item, dict) and item.get("class") in classes
                and _valid_box(item.get("bbox")) for item in truth):
            errors.append({"row": index + 1, "reason": "Missing reviewed building boxes"})
            attempts.append({"row": index + 1, "status": "invalid_label"})
            continue
        truths.append(truth)
        group = row.get("group_id")
        subset = row.get("subset") or "unspecified"
        subset_counts[str(subset)] += 1
        if group:
            subset_groups[str(subset)].add(str(group))
        for item in truth:
            subset_truth[str(subset)][item["class"]] += 1
            if group:
                group_sets[item["class"]].add(str(group))
        try:
            image = _image(root, row.get("image"))
            detected, counts = raw_predictions(model, image, config)
            sizes.append((image.shape[1], image.shape[0]))
            predictions.append(detected)
            truncation.update(counts)
            attempts.append({"row": index + 1, "status": "success", **counts})
        except (OSError, ValueError, ModelUnavailable, RuntimeError) as exc:
            # Truth remains in the denominator, even when inference fails.
            sizes.append((1, 1))
            predictions.append([])
            errors.append({"row": index + 1, "reason": str(exc)})
            attempts.append({"row": index + 1, "status": "inference_error", "reason": str(exc)})
    scored = score_predictions(predictions, truths, sizes, classes, config["iou_thresholds"])
    for label, item in scored["per_class"].items():
        item["truth_groups"] = len(group_sets[label])
    provenance = model.manifest.get("output", {}).get("evaluation_raw_provenance", {})
    declared_raw = all(provenance.get(key) is True for key in ("unfiltered", "no_topk", "no_nms"))
    truncated = truncation["candidate_truncation"] + truncation["detection_truncation"]
    full_protocol = declared_raw and (not config["require_untruncated"] or not truncated) and not errors
    return {"status": "full_protocol" if full_protocol else "diagnostic_only",
            "full_ap_eligible": full_protocol,
            "limitations": (["raw_provenance_unverified"] if not declared_raw else [])
                           + (["evaluation_truncated"] if truncated else [])
                           + (["evaluation_errors"] if errors else []),
            "protocol": {key: value for key, value in config.items()},
            "raw_provenance": provenance,
            "samples": len(rows), "valid_labeled_samples": len(truths),
            "failed": len(errors), "errors": errors, "attempts": attempts,
            "truncation": dict(truncation), "subset_counts": dict(subset_counts),
            "subset_breakdown": {name: {"samples": count,
                "truth_instances_by_class": dict(subset_truth[name]),
                "groups": len(subset_groups[name])} for name, count in subset_counts.items()},
            **scored}


def compare_stability(base: dict, looser: dict, *, tolerance: float = .001) -> dict:
    """Compare validation-set AP under a lower floor and larger independent caps."""
    changes = {}
    for label, item in base.get("per_class", {}).items():
        other = looser.get("per_class", {}).get(label, {})
        for metric in ("ap50", "ap50_95"):
            left, right = item.get(metric), other.get(metric)
            if left is None or right is None:
                changes[f"{label}.{metric}"] = None
            else:
                changes[f"{label}.{metric}"] = abs(left - right)
    observed = [value for value in changes.values() if value is not None]
    maximum = max(observed, default=None)
    passed = (base.get("full_ap_eligible") is True and looser.get("full_ap_eligible") is True
              and bool(changes) and len(observed) == len(changes)
              and maximum is not None and maximum <= tolerance)
    return {"passed": passed, "max_ap_delta": maximum, "per_metric_delta": changes,
            "tolerance": tolerance,
            "base_protocol_sha256": base.get("protocol", {}).get("sha256"),
            "looser_score_floor": looser.get("protocol", {}).get("score_floor"),
            "looser_max_candidates": looser.get("protocol", {}).get("max_candidates"),
            "looser_max_detections": looser.get("protocol", {}).get("max_detections")}
