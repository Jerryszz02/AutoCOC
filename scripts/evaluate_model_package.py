"""Validate and independently score a local, already trained ONNX model package.

Input JSONL rows reference reviewed local images. This command does not create
labels or train/export models. It reports counts, including failed inference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autococ.model_vision import BuildingModel, ModelUnavailable, SceneModel
from training.vision.evaluation import compare_stability, evaluate_independent, load_config
from training.vision.snapshot import TrainingError, verify_snapshot


def select_frozen_rows(root: Path, task: str, split: str, reviewed_jsonl: Path) -> tuple[list[dict], dict]:
    """Verify the whole snapshot before selecting one split's reviewed truth."""
    root = root.resolve()
    if reviewed_jsonl.resolve().parent != root:
        raise ValueError("Reviewed JSONL must be at frozen snapshot root")
    expected = f"reviewed-{'scenes' if task == 'scene' else 'buildings'}.jsonl"
    if reviewed_jsonl.name != expected:
        raise ValueError(f"Expected {expected}")
    verified = verify_snapshot(root, task)
    paths = set(verified["paths"][split])
    rows = [json.loads(line) for line in reviewed_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("image") in paths]
    if len(selected) != len(paths) or {row.get("image") for row in selected} != paths:
        raise ValueError("Reviewed rows do not exactly cover frozen split")
    return selected, verified


def _image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def _reviewed_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Missing image path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Image path outside reviewed dataset")
    return path


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    return overlap / max(1e-9, (a[2] - a[0]) * (a[3] - a[1])
                     + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def _valid_box(box: object) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value) for value in box)
            and box[2] > box[0] and box[3] > box[1])


def _runtime_metadata(model) -> dict:
    try:
        providers = list(model.session.get_providers())
    except (AttributeError, RuntimeError):
        providers = []
    provider = providers[0] if providers else "unknown"
    device = ("CUDA" if "CUDA" in provider else "DirectML" if "Dml" in provider
              else "CPU" if "CPU" in provider else "unknown")
    return {"device": device, "providers": providers,
            "input_size": [model.width, model.height]}


def _timing_summary(attempts: list[dict]) -> dict:
    elapsed = [item["offline_elapsed_seconds"] for item in attempts]
    warm = [item["offline_elapsed_seconds"] for item in attempts if item["phase"] == "warm"]
    cold = next((item["offline_elapsed_seconds"] for item in attempts if item["phase"] == "cold"), None)
    return {"scope": "offline file decode, model inference and metric matching; excludes live capture, decision and input",
            "cold_first_seconds": cold,
            "warm_p50_seconds": float(np.percentile(warm, 50)) if warm else None,
            "p50_seconds": float(np.percentile(elapsed, 50)) if elapsed else None,
            "p95_seconds": float(np.percentile(elapsed, 95)) if elapsed else None,
            "max_seconds": max(elapsed) if elapsed else None,
            "failed_attempts": sum(item["status"] != "success" for item in attempts)}


def _attempt(index: int, phase: str, started: float, inference_elapsed: float | None,
             status: str, runtime: dict, reason: str | None = None) -> dict:
    ended = time.monotonic()
    return {"row": index + 1, "phase": phase, "status": status,
            "reason": reason, "offline_elapsed_seconds": ended - started,
            "inference_elapsed_seconds": inference_elapsed,
            **runtime}


def evaluate(model, rows: list[dict], root: Path, iou_threshold: float = .5) -> dict:
    errors: list[dict] = []
    attempts: list[dict] = []
    runtime = _runtime_metadata(model)
    if isinstance(model, SceneModel):
        matrix: dict[str, Counter] = defaultdict(Counter)
        truth_groups: dict[str, set[str]] = defaultdict(set)
        subset_counts = Counter()
        image_attempts = 0
        for index, row in enumerate(rows):
            started, inference_started = time.monotonic(), None
            inference_elapsed = None
            valid_label = None
            phase = "preflight"
            try:
                if not isinstance(row, dict) or row.get("label") not in {*model.manifest["classes"], "unknown"}:
                    raise ValueError("Missing reviewed scene label")
                valid_label = row["label"]
                if row.get("group_id"):
                    truth_groups[valid_label].add(str(row["group_id"]))
                subset_counts[str(row.get("subset") or "unspecified")] += 1
                phase = "cold" if image_attempts == 0 else "warm"
                image_attempts += 1
                image = _image(_reviewed_path(root, row.get("image")))
                inference_started = time.monotonic()
                prediction = model.classify(image, captured_at=started)
                inference_elapsed = time.monotonic() - inference_started
                matrix[valid_label][prediction.scene] += 1
                attempts.append(_attempt(index, phase, started, inference_elapsed, "success", runtime))
            except (OSError, ValueError, ModelUnavailable) as exc:
                errors.append({"row": index + 1, "reason": str(exc)})
                if valid_label is not None:
                    matrix[valid_label]["inference_error"] += 1
                if inference_started is not None and inference_elapsed is None:
                    inference_elapsed = time.monotonic() - inference_started
                attempts.append(_attempt(index, phase, started, inference_elapsed,
                                         "inference_error" if valid_label is not None else "invalid_label",
                                         runtime, str(exc)))
        labels = sorted(set(matrix) | {key for predictions in matrix.values() for key in predictions})
        counts = {label: {"tp": matrix[label][label],
                          "fp": sum(matrix[other][label] for other in matrix if other != label),
                          "fn": sum(count for predicted, count in matrix[label].items() if predicted != label)}
                  for label in labels if label not in {"unknown", "inference_error"}}
        valid_labeled = sum(sum(row.values()) for row in matrix.values())
        evaluated = sum(item["status"] == "success" for item in attempts)
        rejected = sum(row.get("unknown", 0) for row in matrix.values())
        known_truth = sum(sum(values.values()) for label, values in matrix.items() if label != "unknown")
        known_accepted = sum(count for label, values in matrix.items() if label != "unknown"
                             for predicted, count in values.items() if predicted not in {"unknown", "inference_error"})
        actionable = {"search", "enemy_village", "battle", "village", "training"}
        per_class = _metrics(counts)
        for label, values in per_class.items():
            values["truth_groups"] = len(truth_groups[label])
        return {"kind": "scene", "samples": len(rows), "failed": len(errors), "errors": errors,
                "valid_labeled_samples": valid_labeled, "evaluated": evaluated,
                "coverage": (evaluated - rejected) / max(1, len(rows)),
                "coverage_valid_labeled": (evaluated - rejected) / max(1, valid_labeled),
                "coverage_known": known_accepted / max(1, known_truth),
                "rejection_rate": rejected / max(1, evaluated),
                "confusion_matrix": {label: dict(matrix[label]) for label in sorted(matrix)},
                "per_class": per_class, "subset_counts": dict(subset_counts),
                "unknown_truth_samples": sum(matrix.get("unknown", {}).values()),
                "unknown_truth_groups": len(truth_groups["unknown"]),
                "unknown_rejections": sum(row.get("unknown", 0) for label, row in matrix.items() if label != "unknown"),
                "unknown_false_accepts": sum(value for predicted, value in matrix.get("unknown", {}).items()
                                             if predicted not in {"unknown", "inference_error"}),
                "unknown_actionable_false_accepts": sum(value for predicted, value in matrix.get("unknown", {}).items()
                                                          if predicted in actionable),
                "attempts": attempts, "timing": _timing_summary(attempts)}
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in model.manifest["classes"]}
    center_errors: dict[str, list[float]] = defaultdict(list)
    corner_errors: dict[str, list[float]] = defaultdict(list)
    truth_groups: dict[str, set[str]] = defaultdict(set)
    subset_counts = Counter()
    image_attempts = 0
    for index, row in enumerate(rows):
        started, inference_started = time.monotonic(), None
        inference_elapsed = None
        truth = None
        phase = "preflight"
        try:
            truth = row.get("boxes") if isinstance(row, dict) else None
            if not isinstance(truth, list) or not all(isinstance(box, dict) and box.get("class") in counts
                    and _valid_box(box.get("bbox")) for box in truth):
                raise ValueError("Missing reviewed building boxes")
            for box in truth:
                if row.get("group_id"):
                    truth_groups[box["class"]].add(str(row["group_id"]))
            subset_counts[str(row.get("subset") or "unspecified")] += 1
            phase = "cold" if image_attempts == 0 else "warm"
            image_attempts += 1
            image = _image(_reviewed_path(root, row.get("image")))
            inference_started = time.monotonic()
            predictions = model.detect(image, captured_at=started)
            inference_elapsed = time.monotonic() - inference_started
            unmatched = set(range(len(truth)))
            for prediction in predictions:
                matches = [(i, _iou(prediction.bbox, truth[i]["bbox"])) for i in unmatched
                           if truth[i]["class"] == prediction.kind]
                match = max(matches, key=lambda item: item[1], default=None)
                if match is not None and match[1] >= iou_threshold:
                    counts[prediction.kind]["tp"] += 1
                    unmatched.remove(match[0])
                    actual = truth[match[0]]["bbox"]
                    center_errors[prediction.kind].append(math.dist(
                        ((prediction.bbox[0] + prediction.bbox[2]) / 2,
                         (prediction.bbox[1] + prediction.bbox[3]) / 2),
                        ((actual[0] + actual[2]) / 2, (actual[1] + actual[3]) / 2)))
                    corner_errors[prediction.kind].append(sum(abs(a - b) for a, b in zip(
                        prediction.bbox, actual, strict=True)) / 4)
                else:
                    counts[prediction.kind]["fp"] += 1
            for i in unmatched:
                counts[truth[i]["class"]]["fn"] += 1
            attempts.append(_attempt(index, phase, started, inference_elapsed, "success", runtime))
        except (OSError, ValueError, ModelUnavailable) as exc:
            errors.append({"row": index + 1, "reason": str(exc)})
            if isinstance(truth, list) and all(isinstance(box, dict) and box.get("class") in counts
                    and _valid_box(box.get("bbox")) for box in truth):
                for box in truth:
                    counts[box["class"]]["fn"] += 1
                status = "inference_error"
            else:
                status = "invalid_label"
            if inference_started is not None and inference_elapsed is None:
                inference_elapsed = time.monotonic() - inference_started
            attempts.append(_attempt(index, phase, started, inference_elapsed, status, runtime, str(exc)))
    per_class = _metrics(counts)
    for label in per_class:
        per_class[label]["truth_groups"] = len(truth_groups[label])
        per_class[label]["mean_center_error_px"] = (sum(center_errors[label]) / len(center_errors[label])
                                                      if center_errors[label] else None)
        per_class[label]["mean_corner_error_px"] = (sum(corner_errors[label]) / len(corner_errors[label])
                                                      if corner_errors[label] else None)
    return {"kind": "building", "protocol": "runtime_point", "samples": len(rows), "failed": len(errors), "errors": errors,
            "iou_threshold": iou_threshold, "per_class": per_class,
            "subset_counts": dict(subset_counts),
            "valid_labeled_samples": sum(item["status"] != "invalid_label" for item in attempts),
            "attempts": attempts, "timing": _timing_summary(attempts)}


def _metrics(counts: dict[str, dict[str, int]]) -> dict:
    return {label: {**values,
        "precision": values["tp"] / max(1, values["tp"] + values["fp"]),
        "recall": values["tp"] / max(1, values["tp"] + values["fn"]),
        "f1": (2 * values["tp"] / max(1, 2 * values["tp"] + values["fp"] + values["fn"]))}
        for label, values in sorted(counts.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--kind", choices=("scene", "building"), required=True)
    parser.add_argument("--client-version", required=True)
    parser.add_argument("--reviewed-jsonl", type=Path)
    parser.add_argument("--dataset-dir", type=Path,
                        help="Frozen snapshot root; required for a verified test-set report")
    parser.add_argument("--split", choices=("train", "val", "test"),
                        help="Evaluate exactly this frozen split from --dataset-dir")
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--evaluation-config", type=Path,
                        help="Frozen independent AP protocol; building only")
    parser.add_argument("--stability-check", action="store_true",
                        help="On validation split, rerun AP with lower score floor and larger caps")
    args = parser.parse_args(argv)
    try:
        if not 0 < args.iou <= 1:
            raise ValueError("--iou must be within (0, 1]")
        if bool(args.dataset_dir) != bool(args.split):
            raise ValueError("--dataset-dir and --split must be supplied together")
        if args.dataset_dir is not None and args.reviewed_jsonl is None:
            raise ValueError("--dataset-dir requires --reviewed-jsonl")
        if args.stability_check and (args.kind != "building" or args.split != "val"):
            raise ValueError("--stability-check requires building model and frozen val split")
        model_type = SceneModel if args.kind == "scene" else BuildingModel
        load_started = time.monotonic()
        model = model_type(args.model_dir, client_version=args.client_version)
        load_seconds = time.monotonic() - load_started
        result = {"status": "package_loadable", "kind": args.kind, "sha256": model.model_sha256,
                  "model_contract_sha256": model.metadata["model_contract_sha256"],
                  "classes": model.manifest["classes"],
                  "metadata": model.metadata,
                  "model_load_seconds": load_seconds,
                  "offline_acceptance": "not_established"}
        if args.reviewed_jsonl is not None:
            rows = [json.loads(line) for line in args.reviewed_jsonl.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            reviewed_root = args.reviewed_jsonl.parent
            result["evaluation_scope"] = {"status": "unverified"}
            if args.dataset_dir is not None:
                reviewed_root = args.dataset_dir.resolve()
                rows, verified = select_frozen_rows(reviewed_root, args.kind, args.split, args.reviewed_jsonl)
                result["evaluation_scope"] = {"status": "frozen_split", "split": args.split,
                    "data_version": verified["metadata"].get("data_version"),
                    "dataset_sha256": verified["snapshot_sha256"],
                    "split_sha256": verified["split_sha256"],
                    "reviewed_jsonl_sha256": hashlib.sha256(args.reviewed_jsonl.read_bytes()).hexdigest(),
                    "image_count": len(rows)}
                if args.split == "test":
                    result["evaluation_scope"]["test_image_count"] = len(rows)
            result["evaluation"] = evaluate(model, rows, reviewed_root, args.iou)
            if args.kind == "building":
                config = load_config(args.evaluation_config) if args.evaluation_config else load_config()
                result["independent_ap"] = evaluate_independent(model, rows, reviewed_root, config)
                if args.stability_check:
                    looser = dict(config)
                    looser["score_floor"] = config["score_floor"] / 10
                    looser["max_candidates"] = config["max_candidates"] * 2
                    looser["max_detections"] = config["max_detections"] * 2
                    looser["sha256"] = "derived-from-" + config["sha256"]
                    result["evaluation_stability"] = compare_stability(
                        result["independent_ap"], evaluate_independent(model, rows, reviewed_root, looser))
    except (OSError, ValueError, ModelUnavailable, TrainingError) as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (result.get("evaluation", {}).get("failed", 0) == 0
                 and result.get("independent_ap", {}).get("failed", 0) == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
