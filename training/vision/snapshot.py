"""Frozen reviewed dataset verification shared by training and evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .data import read_jsonl, sha256


class TrainingError(ValueError):
    pass


def verify_snapshot(root: Path, task: str) -> dict:
    """Require a complete frozen snapshot and verify every listed file."""
    root = root.resolve()
    card, sums = root / "DATASET_CARD.md", root / "SHA256SUMS"
    if not card.is_file() or not sums.is_file() or f"Task: {task}" not in card.read_text(encoding="utf-8"):
        raise TrainingError("Expected a frozen reviewed snapshot for this task")
    entries = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = (root / relative).resolve()
        if not separator or len(digest) != 64 or not path.is_relative_to(root) or not path.is_file():
            raise TrainingError(f"Invalid snapshot hash entry: {line}")
        if sha256(path) != digest:
            raise TrainingError(f"Snapshot hash mismatch: {relative}")
        entries[relative] = digest
    required = {"split.json", "snapshot.json", "taxonomy.json", "train.txt", "val.txt", "test.txt",
                "reviewed-manifest.jsonl",
                "reviewed-scenes.jsonl" if task == "scene" else "reviewed-buildings.jsonl"}
    if not required <= entries.keys():
        raise TrainingError(f"Snapshot missing hash entries: {sorted(required - entries.keys())}")
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    if split.get("task") != task or not split.get("sample_groups"):
        raise TrainingError("Split task/group provenance missing")
    metadata = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("task") != task or not metadata.get("data_version") or not metadata.get("client_versions"):
        raise TrainingError("Snapshot provenance missing")
    if metadata.get("prototype_only") is not split.get("prototype_only", False):
        raise TrainingError("Snapshot/split prototype status differs")
    if not metadata["prototype_only"] and "unknown" in metadata["client_versions"]:
        raise TrainingError("Formal snapshot cannot have unknown client version")
    paths = {}
    for name in ("train", "val", "test"):
        paths[name] = [line.strip().replace("\\", "/") for line in
                       (root / f"{name}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not paths[name] or len(paths[name]) != len(set(paths[name])):
            raise TrainingError(f"Empty or duplicate {name} split")
        for relative in paths[name]:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or relative not in entries:
                raise TrainingError(f"Unhashed or unsafe image: {relative}")
    if set(paths["train"]) & set(paths["val"]) or set(paths["train"]) & set(paths["test"]) or set(paths["val"]) & set(paths["test"]):
        raise TrainingError("Image appears in more than one split")
    reviewed_name = "reviewed-scenes.jsonl" if task == "scene" else "reviewed-buildings.jsonl"
    reviewed = read_jsonl(root / reviewed_name)
    source_rows = read_jsonl(root / "reviewed-manifest.jsonl")
    sources = {row.get("sample_id"): row for row in source_rows}
    if len(sources) != len(source_rows) or len(sources) != len(reviewed):
        raise TrainingError("Reviewed source provenance does not cover snapshot")
    expected_paths = {path for selected in paths.values() for path in selected}
    if (len(reviewed) != len(expected_paths) or {row.get("image") for row in reviewed} != expected_paths
            or len({row.get("sample_id") for row in reviewed}) != len(reviewed)):
        raise TrainingError("Reviewed labels do not exactly cover frozen split images")
    image_splits = {path: name for name, selected in paths.items() for path in selected}
    sample_assignments = split.get("sample_assignments", {})
    sample_groups = split.get("sample_groups", {})
    group_assignments = split.get("group_assignments", {})
    if set(sample_assignments) != {row["sample_id"] for row in reviewed} or set(sample_groups) != set(sample_assignments):
        raise TrainingError("Split sample assignments do not cover reviewed images")
    class_indices = None
    if task == "building":
        if "class_mapping.json" not in entries:
            raise TrainingError("Building class mapping omitted from frozen hashes")
        mapping = json.loads((root / "class_mapping.json").read_text(encoding="utf-8"))
        class_indices = mapping.get("output_indices")
        if (not isinstance(class_indices, dict) or not class_indices
                or set(class_indices.values()) != set(range(len(class_indices)))):
            raise TrainingError("Invalid building class mapping")
    for row in reviewed:
        sample_id, group_id, image = row.get("sample_id"), row.get("group_id"), row["image"]
        if not isinstance(sample_id, str) or not isinstance(group_id, str):
            raise TrainingError("Reviewed row lacks sample/group ID")
        assigned = image_splits[image]
        if (sample_assignments.get(sample_id) != assigned or sample_groups.get(sample_id) != group_id
                or group_assignments.get(group_id) != assigned):
            raise TrainingError(f"{sample_id}: split/group assignment differs from reviewed data")
        source = sources.get(sample_id)
        methods = {"human", "agent_visual"} if metadata["prototype_only"] else {"human"}
        if (not isinstance(source, dict) or source.get("snapshot_image_path") != image
                or source.get("snapshot_split") != assigned or source.get("computed_leakage_group_id") != group_id
                or source.get("file_sha256") != entries[image]
                or source.get("review_status") != "reviewed" or source.get("review_method") not in methods
                or not source.get("reviewer") or not source.get("review_evidence")
                or (source.get("client_version") or "unknown") not in metadata["client_versions"]):
            raise TrainingError(f"{sample_id}: source review provenance or version mismatch")
        annotation_rel = f"annotations/{sample_id}.jsonl"
        if annotation_rel not in entries:
            raise TrainingError(f"{sample_id}: annotation omitted from frozen hashes")
        original = read_jsonl(root / annotation_rel)
        if len(original) != 1 or original[0].get("sample_id") != sample_id:
            raise TrainingError(f"{sample_id}: original annotation mismatch")
        item = original[0].get("scene" if task == "scene" else "buildings")
        if (not isinstance(item, dict) or item.get("complete") is not True or item.get("review_status") != "reviewed"
                or item.get("review_method") not in methods or not item.get("reviewer") or not item.get("review_evidence")):
            raise TrainingError(f"{sample_id}: annotation lacks complete reviewed evidence")
        if task == "scene":
            if item.get("label") != row.get("label"):
                raise TrainingError(f"{sample_id}: scene label differs from original annotation")
        else:
            objects = item.get("objects", [])
            boxes = row.get("boxes", [])
            if (any(obj.get("ignore") or obj.get("review_status") != "reviewed" for obj in objects)
                    or not isinstance(boxes, list) or len(objects) != len(boxes)):
                raise TrainingError(f"{sample_id}: building annotation incomplete")
            expected_boxes = [(obj.get("class_name"), obj.get("bbox")) for obj in objects]
            reviewed_boxes = [(box.get("class"), box.get("bbox")) for box in boxes]
            if expected_boxes != reviewed_boxes:
                raise TrainingError(f"{sample_id}: building boxes differ from original annotation")
            label_rel = f"labels/{sample_id}.txt"
            if label_rel not in entries:
                raise TrainingError(f"{sample_id}: YOLO labels omitted from frozen hashes")
            width, height = source.get("width"), source.get("height")
            if (not isinstance(width, int) or not isinstance(height, int)
                    or width < 1 or height < 1):
                raise TrainingError(f"{sample_id}: source dimensions missing")
            expected_labels = []
            for class_name, bbox in reviewed_boxes:
                if (class_name not in class_indices or not isinstance(bbox, list) or len(bbox) != 4
                        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)):
                    raise TrainingError(f"{sample_id}: invalid reviewed building box")
                x0, y0, x1, y1 = bbox
                if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                    raise TrainingError(f"{sample_id}: reviewed building box outside image")
                expected_labels.append((class_indices[class_name], (x0 + x1) / (2 * width),
                                        (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height))
            lines = [line.split() for line in (root / label_rel).read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != len(expected_labels):
                raise TrainingError(f"{sample_id}: YOLO labels differ from reviewed boxes")
            try:
                actual_labels = [(int(parts[0]), *[float(value) for value in parts[1:]])
                                 for parts in lines if len(parts) == 5]
            except ValueError as exc:
                raise TrainingError(f"{sample_id}: malformed YOLO labels") from exc
            if len(actual_labels) != len(expected_labels) or any(
                    actual[0] != expected[0] or any(not math.isclose(a, b, abs_tol=1e-6)
                                                    for a, b in zip(actual[1:], expected[1:]))
                    for actual, expected in zip(actual_labels, expected_labels)):
                raise TrainingError(f"{sample_id}: YOLO labels differ from reviewed boxes")
    return {"root": str(root), "snapshot_sha256": sha256(sums), "paths": paths,
            "metadata": metadata,
            "split_sha256": entries["split.json"]}


