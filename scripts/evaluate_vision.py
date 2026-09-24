"""Evaluate local screenshots against independently reviewed human labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

from autococ import scene as scene_types
from autococ.config import load_config
from autococ.vision import ScreenshotRecognizer


SCENES = {value for name, value in vars(scene_types).items() if name.startswith("SCENE_")}
ACTIONABLE_SCENES = SCENES - {"unknown", "starting", "maintenance", "disconnected"}
RESOURCES = {"gold", "elixir", "dark_elixir", "gems"}
ACCEPTANCE_THRESHOLD = 0.8


@dataclass
class EvaluationRow:
    path: str
    expected_scene: str
    predicted_scene: str
    confidence: float | None
    expected_resources: dict[str, int] = field(default_factory=dict)
    predicted_resources: dict[str, object] = field(default_factory=dict)
    expected_resource_source: str | None = None
    predicted_resource_source: str | None = None
    pixel_sha256: str | None = None
    error: str | None = None


def summarize_predictions(rows: list[EvaluationRow]) -> dict[str, object]:
    true_classes = sorted({row.expected_scene for row in rows})
    predicted_classes = sorted({row.predicted_scene for row in rows} | set(true_classes))
    matrix = {truth: {predicted: 0 for predicted in predicted_classes} for truth in true_classes}
    for row in rows:
        matrix[row.expected_scene][row.predicted_scene] += 1
    per_class = {}
    for label in true_classes:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth].get(label, 0) for truth in true_classes)
        per_class[label] = {
            "support": support,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support,
            "f1": 2 * true_positive / (support + predicted),
        }
    accepted = [row for row in rows if row.error is None and isinstance(row.confidence, (int, float))
                and math.isfinite(row.confidence) and ACCEPTANCE_THRESHOLD <= row.confidence <= 1]
    numeric = {resource: {"correct": 0, "total": 0} for resource in sorted(RESOURCES)}
    source_labeled = [row for row in rows if row.expected_resource_source is not None]
    for row in rows:
        correct_source = row.expected_resource_source is None or row.expected_resource_source == row.predicted_resource_source
        for resource, expected in row.expected_resources.items():
            predicted = row.predicted_resources.get(resource)
            numeric[resource]["total"] += 1
            numeric[resource]["correct"] += int(row.error is None and correct_source and type(predicted) is int and predicted == expected)
    for values in numeric.values():
        values["exact_ratio"] = values["correct"] / values["total"] if values["total"] else None
    numeric_total = sum(values["total"] for values in numeric.values())
    numeric_correct = sum(values["correct"] for values in numeric.values())
    unknown_rows = [row for row in rows if row.expected_scene == "unknown"]
    return {
        "sample_count": len(rows),
        "recognition_errors": sum(row.error is not None for row in rows),
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro_f1": sum(values["f1"] for values in per_class.values()) / len(per_class) if per_class else None,
        "macro_f1_classes": true_classes,
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "accepted_count": len(accepted),
        "accepted_accuracy": sum(row.expected_scene == row.predicted_scene for row in accepted) / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(rows) if rows else None,
        "numeric_correct": numeric_correct,
        "numeric_total": numeric_total,
        "numeric_exact_ratio": numeric_correct / numeric_total if numeric_total else None,
        "numeric_by_resource": numeric,
        "resource_source_total": len(source_labeled),
        "resource_source_correct": sum(row.error is None and row.expected_resource_source == row.predicted_resource_source for row in source_labeled),
        "unknown_sample_count": len(unknown_rows),
        "unknown_actionable_false_positives": sum(row.expected_scene == "unknown" and row.predicted_scene in ACTIONABLE_SCENES for row in accepted),
        "actionable_scenes": sorted(ACTIONABLE_SCENES),
    }


def _pixel_hash(path: Path) -> str:
    import cv2
    import numpy as np

    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"Unable to decode screenshot: {path}: {exc}") from exc
    if image is None:
        raise ValueError(f"Unable to decode screenshot: {path}")
    header = f"{image.shape[1]}x{image.shape[0]}:bgr8\n".encode("ascii")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def evaluate_manifest(path: Path, recognizer: ScreenshotRecognizer) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Manifest must be an object containing a samples list")
    unique: dict[str, dict] = {}
    excluded, duplicates = [], []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Sample {index} must be an object")
        if not entry.get("scene"):
            excluded.append({"index": index, "path": entry.get("path"), "reason": "no human scene label"})
            continue
        if entry["scene"] not in SCENES or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"Sample {index} needs a known scene label and a nonempty path")
        resources = entry.get("resources", {})
        if not isinstance(resources, dict) or any(key not in RESOURCES or (value is not None and (type(value) is not int or value < 0)) for key, value in resources.items()):
            raise ValueError(f"Sample {index} resources must contain nonnegative integers or null")
        resources = {key: value for key, value in resources.items() if value is not None}
        source = entry.get("resource_source")
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError(f"Sample {index} resource_source must be a nonempty string or null")
        image_path = (path.parent / entry["path"]).resolve()
        error = None
        try:
            pixel_hash = _pixel_hash(image_path)
        except (OSError, ValueError) as exc:
            pixel_hash, error = None, str(exc)
        key = pixel_hash or f"unreadable:{image_path}"
        sample = {"path": image_path, "scene": entry["scene"], "resources": resources,
                  "resource_source": source, "pixel_sha256": pixel_hash, "error": error}
        if key not in unique:
            unique[key] = sample
            continue
        previous = unique[key]
        conflicts = previous["scene"] != sample["scene"] or (
            previous["resource_source"] is not None and source is not None and previous["resource_source"] != source
        ) or any(name in previous["resources"] and previous["resources"][name] != value for name, value in resources.items())
        if conflicts:
            raise ValueError(f"Conflicting human labels for the same image: {previous['path']} and {image_path}")
        previous["resources"].update(resources)
        previous["resource_source"] = previous["resource_source"] or source
        duplicates.append({"index": index, "path": str(image_path), "same_image_as": str(previous["path"]), "pixel_sha256": pixel_hash})

    rows = []
    for sample in unique.values():
        row = EvaluationRow(str(sample["path"]), sample["scene"], "__error__", None,
                            expected_resources=sample["resources"], expected_resource_source=sample["resource_source"],
                            pixel_sha256=sample["pixel_sha256"], error=sample["error"])
        if row.error is None:
            try:
                snapshot = recognizer.recognize(sample["path"])
                if snapshot.scene not in SCENES:
                    raise ValueError(f"Recognizer returned an unknown scene category: {snapshot.scene}")
                row.predicted_scene = snapshot.scene
                row.confidence = snapshot.confidence if math.isfinite(snapshot.confidence) else None
                row.predicted_resources = snapshot.observations.get("resources", {})
                row.predicted_resource_source = snapshot.observations.get("resource_source")
                if not isinstance(row.predicted_resources, dict):
                    raise ValueError("Recognizer resources must be an object")
            except Exception as exc:
                row.error = f"{type(exc).__name__}: {exc}"
                row.predicted_scene, row.confidence, row.predicted_resources = "__error__", None, {}
        rows.append(row)
    return {
        "scope": "Only the unique human-labeled images in this manifest; not live game verification or full acceptance.",
        "manifest": str(path.resolve()),
        "manifest_entries": len(entries),
        "excluded_unlabeled": excluded,
        "duplicate_entries": duplicates,
        "metrics": summarize_predictions(rows),
        "samples": [asdict(row) for row in rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest with human-reviewed labels")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", type=Path, help="New JSON result path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        recognizer = ScreenshotRecognizer(config.ocr, config.game.baseline_resolution)
        report = evaluate_manifest(args.manifest, recognizer)
        output = args.output or config.runtime.report_dir / f"vision-evaluation-{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid4().hex[:8]}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        print(json.dumps({"output": str(output), "scope": report["scope"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if report["metrics"]["recognition_errors"] or not report["metrics"]["sample_count"] else 0
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
