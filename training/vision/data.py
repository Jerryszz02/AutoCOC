"""Offline, fail-closed dataset preparation for AutoCOC vision models."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any

import cv2
import numpy as np


TAXONOMY = Path(__file__).with_name("taxonomy.json")
SPLITS = ("train", "val", "test")
REVIEWED = "reviewed"


class DatasetError(ValueError):
    """A dataset cannot be used without correcting its evidence."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise DatasetError(f"Missing JSONL: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DatasetError(f"Image path must be relative: {relative!r}")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DatasetError(f"Image escapes data root: {relative}")
    return path


def decode(path: Path) -> np.ndarray:
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise DatasetError(f"Cannot decode image: {path}: {exc}") from exc
    if image is None:
        raise DatasetError(f"Cannot decode image: {path}")
    return image


def _pixel_hash(image: np.ndarray) -> str:
    # Match the historical catalog's Pillow RGB + str(size) convention.
    header = str((image.shape[1], image.shape[0])).encode("ascii")
    return hashlib.sha256(header + cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()


def load_taxonomy(path: Path = TAXONOMY) -> dict:
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    if taxonomy.get("schema_version") != 1:
        raise DatasetError("Unsupported taxonomy schema_version")
    classes = taxonomy.get("building_classes", [])
    ids = [item.get("id") for item in classes]
    names = [item.get("name") for item in classes]
    if any(type(value) is not int or value <= 0 for value in ids) or len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise DatasetError("Taxonomy requires unique, positive stable IDs and names")
    return taxonomy


def import_legacy(root: Path, legacy: Path, output: Path) -> dict:
    """Import the historical catalog without changing it or treating guesses as truth."""
    rows, seen = [], set()
    for old in read_jsonl(legacy):
        sample_id = old.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise DatasetError(f"Duplicate/missing legacy sample_id: {sample_id!r}")
        seen.add(sample_id)
        path = image_path(root, old.get("file"))
        if not path.is_file():
            raise DatasetError(f"Missing legacy image: {path}")
        rows.append({
            "schema_version": 1, "sample_id": sample_id,
            "image_path": old["file"].replace("\\", "/"),
            "file_sha256": old.get("copy_sha256"), "pixel_sha256": old.get("pixel_sha256"),
            "width": old.get("width"), "height": old.get("height"),
            "parent_sample_id": None, "source_domain": "local",
            "source_ref": old.get("source"), "source_session": old.get("source_session"),
            "battle_id": None, "layout_group_id": old.get("layout_group"),
            "leakage_group_id": None, "client_version": old.get("game_version"),
            "language": None, "zoom": None, "camera_id": None, "army_profile_id": None,
            "capture_started_monotonic": None, "capture_finished_monotonic": None,
            "session_clock_id": None, "captured_at_utc": None,
            "video_time_ms": None, "timestamp_quality": "legacy_observation_only",
            "legacy_observed_at_unix": old.get("captured_at_unix"),
            "annotation_version": None, "review_status": "pending",
            "reviewer": None, "review_evidence": None, "review_method": None,
            "exclusion_reason": None,
            "intended_uses": old.get("intended_uses", []),
            "split": "unassigned", "data_version": None,
            "legacy_scene_candidate": old.get("scene_candidate"),
        })
    if output.exists():
        raise DatasetError(f"Refusing to overwrite manifest: {output}")
    write_jsonl(output, rows)
    return {"imported": len(rows), "manifest": str(output), "legacy_unchanged": True}


def stage_sources(sources: list[tuple[Path, Path]], output: Path) -> dict:
    """Copy verified manifests and originals into one new, portable data root."""
    output = Path(output).resolve()
    if output.exists():
        raise DatasetError(f"Refusing to overwrite staged source root: {output}")
    if not sources:
        raise DatasetError("At least one source root and manifest are required")
    staged, audit, seen = [], [], set()
    for index, (root, manifest_path) in enumerate(sources):
        root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
        rows = read_jsonl(manifest_path)
        checks = validate(root, rows, [], verify_pixels=True)
        if not checks["valid"]:
            raise DatasetError(f"Source {index} failed validation: {'; '.join(checks['errors'][:5])}")
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise DatasetError(f"Duplicate staged sample_id: {sample_id}")
            seen.add(sample_id)
            # Prefixing source roots prevents path collision while retaining
            # every source image's byte and pixel hash and original reference.
            relative = f"source-{index:02d}/{row['image_path']}"
            staged.append({**row, "image_path": relative})
        audit.append({"source_index": index, "root": str(root), "manifest": str(manifest_path),
                      "manifest_sha256": sha256(manifest_path), "samples": len(rows)})
    output.mkdir(parents=True)
    try:
        for row in staged:
            index = int(row["image_path"].split("/", 1)[0][7:])
            original = sources[index][0]
            original_row_path = row["image_path"].split("/", 1)[1]
            source = image_path(Path(original), original_row_path)
            destination = image_path(output, row["image_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(destination) != row["file_sha256"]:
                raise DatasetError(f"Staged copy hash changed: {row['sample_id']}")
        write_jsonl(output / "manifest.jsonl", staged)
        (output / "source-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    except Exception:
        (output / "INCOMPLETE").write_text("Staging failed; do not use this directory.\n", encoding="utf-8")
        raise
    return {"output": str(output), "samples": len(staged),
            "source_count": len(sources), "manifest_sha256": sha256(output / "manifest.jsonl")}


def _ids(rows: list[dict], key: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in indexed:
            raise DatasetError(f"Duplicate/missing {key}: {value!r}")
        indexed[value] = row
    return indexed


def _valid_box(box: Any, width: int, height: int) -> bool:
    return (type(width) is int and width > 0 and type(height) is int and height > 0
            and isinstance(box, list) and len(box) == 4
            and all(type(value) in (float, int) and math.isfinite(value) for value in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def import_annotations(existing: list[dict], incoming: list[dict], manifest: list[dict]) -> list[dict]:
    """Merge task annotations without changing a prior verdict or provenance."""
    known = _ids(manifest, "sample_id")
    result = {key: dict(value) for key, value in _ids(existing, "sample_id").items()}
    for sample_id, row in _ids(incoming, "sample_id").items():
        if sample_id not in known:
            raise DatasetError(f"{sample_id}: annotation has no manifest sample")
        if row.get("schema_version") != 1:
            raise DatasetError(f"{sample_id}: unsupported annotation schema_version")
        for task in ("scene", "buildings"):
            item = row.get(task)
            if isinstance(item, dict) and item.get("review_status") == REVIEWED:
                if not item.get("reviewer") or not item.get("review_evidence") or item.get("review_method") not in {"human", "agent_visual"}:
                    raise DatasetError(f"{sample_id}: reviewed {task} needs reviewer, review_evidence and review_method")
        current = result.get(sample_id, {"schema_version": 1, "sample_id": sample_id})
        for key, value in row.items():
            if key in current and current[key] != value:
                raise DatasetError(f"{sample_id}: conflicting annotation field {key}; resolve in review log")
            current[key] = value
        result[sample_id] = current
    return [result[key] for key in sorted(result)]


def validate(root: Path, manifest: list[dict], annotations: list[dict], *,
             taxonomy: dict | None = None, verify_pixels: bool = True) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    errors, warnings = [], []
    if not manifest:
        errors.append("Manifest is empty")
    try:
        samples, labels = _ids(manifest, "sample_id"), _ids(annotations, "sample_id")
    except DatasetError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "sample_count": len(manifest)}
    known_scenes = set(taxonomy["scene_classes"]) | {taxonomy["unknown_scene"]}
    known_buildings = {item["name"] for item in taxonomy["building_classes"]}
    for sample_id in labels.keys() - samples.keys():
        errors.append(f"{sample_id}: annotation has no manifest sample")
    for sample_id, row in samples.items():
        prefix = f"{sample_id}: "
        if row.get("schema_version") != 1:
            errors.append(prefix + "unsupported manifest schema_version")
        if type(row.get("width")) is not int or row["width"] <= 0 or type(row.get("height")) is not int or row["height"] <= 0:
            errors.append(prefix + "invalid image dimensions")
        if row.get("review_status") not in {"pending", "reviewed", "rejected"}:
            errors.append(prefix + "invalid review_status")
        try:
            path = image_path(root, row.get("image_path"))
            if not path.is_file():
                raise DatasetError("image missing")
            if sha256(path) != row.get("file_sha256"):
                errors.append(prefix + "file SHA-256 mismatch")
            if verify_pixels:
                image = decode(path)
                if [image.shape[1], image.shape[0]] != [row.get("width"), row.get("height")]:
                    errors.append(prefix + "image dimensions mismatch")
                if _pixel_hash(image) != row.get("pixel_sha256"):
                    errors.append(prefix + "pixel SHA-256 mismatch")
        except (OSError, ValueError) as exc:
            errors.append(prefix + str(exc))
        parent = row.get("parent_sample_id")
        if parent is not None and parent not in samples:
            errors.append(prefix + "missing parent_sample_id")
        if row.get("review_status") == REVIEWED and not row.get("reviewer"):
            errors.append(prefix + "reviewed sample needs reviewer")
        if row.get("review_status") == REVIEWED and not row.get("review_evidence"):
            errors.append(prefix + "reviewed sample needs review_evidence")
        if row.get("review_status") == REVIEWED and row.get("review_method") not in {"human", "agent_visual"}:
            errors.append(prefix + "reviewed sample needs review_method human/agent_visual")
        annotation = labels.get(sample_id)
        if not annotation:
            warnings.append(prefix + "no annotation")
            continue
        if annotation.get("schema_version") != 1:
            errors.append(prefix + "unsupported annotation schema_version")
        scene = annotation.get("scene")
        if scene is not None:
            if not isinstance(scene, dict):
                errors.append(prefix + "scene annotation must be object")
            elif scene.get("label") not in known_scenes:
                errors.append(prefix + "unknown scene label")
            if isinstance(scene, dict) and scene.get("complete") is not True and scene.get("review_status") == REVIEWED:
                errors.append(prefix + "reviewed scene is incomplete")
            if isinstance(scene, dict) and scene.get("review_status") == REVIEWED and (not scene.get("reviewer") or not scene.get("review_evidence") or scene.get("review_method") not in {"human", "agent_visual"}):
                errors.append(prefix + "reviewed scene needs reviewer, review_evidence and review_method")
        buildings = annotation.get("buildings")
        if buildings is not None:
            if not isinstance(buildings, dict):
                errors.append(prefix + "buildings annotation must be object")
                continue
            complete_classes = buildings.get("complete_classes")
            if complete_classes is not None:
                try:
                    _mapping(taxonomy, complete_classes)
                except DatasetError as exc:
                    errors.append(prefix + f"invalid buildings.complete_classes: {exc}")
                if buildings.get("complete") is not False:
                    errors.append(prefix + "buildings.complete_classes requires complete=false")
            if (buildings.get("complete") is not True and buildings.get("review_status") == REVIEWED
                    and complete_classes is None):
                errors.append(prefix + "reviewed buildings are incomplete")
            if buildings.get("review_status") == REVIEWED and (not buildings.get("reviewer") or not buildings.get("review_evidence") or buildings.get("review_method") not in {"human", "agent_visual"}):
                errors.append(prefix + "reviewed buildings need reviewer, review_evidence and review_method")
            if not isinstance(buildings.get("objects"), list):
                errors.append(prefix + "buildings objects must be list")
                continue
            object_ids = set()
            for obj in buildings.get("objects", []):
                if not isinstance(obj, dict):
                    errors.append(prefix + "building object must be object")
                    continue
                obj_id = obj.get("object_id")
                if not isinstance(obj_id, str) or not obj_id or obj_id in object_ids:
                    errors.append(prefix + "duplicate/missing object_id")
                object_ids.add(obj_id)
                if obj.get("class_name") not in known_buildings and not obj.get("ignore"):
                    errors.append(prefix + f"unknown building class {obj.get('class_name')!r}")
                if not _valid_box(obj.get("bbox"), row.get("width"), row.get("height")):
                    errors.append(prefix + f"invalid bbox for {obj_id}")
                if obj.get("ignore") and not obj.get("ignore_reason"):
                    errors.append(prefix + f"ignore without reason for {obj_id}")
                if not obj.get("ignore") and obj.get("review_status") != REVIEWED and buildings.get("review_status") == REVIEWED:
                    errors.append(prefix + f"unreviewed object {obj_id}")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "sample_count": len(manifest), "annotation_count": len(annotations)}


def _dhash(path: Path) -> int:
    gray = cv2.cvtColor(decode(path), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | bool(bit)
    return value


def leakage_groups(root: Path, manifest: list[dict], *, near_duplicates: bool = False) -> tuple[dict[str, str], list[dict]]:
    """Union transitive leakage links; near matches require separate human review."""
    samples = _ids(manifest, "sample_id")
    parent = {sample_id: sample_id for sample_id in samples}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for key in ("file_sha256", "pixel_sha256", "source_session", "battle_id", "layout_group_id", "leakage_group_id", "source_ref"):
        first = {}
        for sample_id, row in samples.items():
            value = row.get(key)
            if value:
                if value in first:
                    union(sample_id, first[value])
                else:
                    first[value] = sample_id
    for sample_id, row in samples.items():
        ancestor = row.get("parent_sample_id")
        if ancestor:
            if ancestor not in samples:
                raise DatasetError(f"{sample_id}: unknown parent {ancestor}")
            union(sample_id, ancestor)
    near = []
    if near_duplicates:
        hashes = [(sample_id, _dhash(image_path(root, row["image_path"]))) for sample_id, row in samples.items()]
        for index, (a, ahash) in enumerate(hashes):
            for b, bhash in hashes[index + 1:]:
                distance = (ahash ^ bhash).bit_count()
                if distance <= 5 and find(a) != find(b):
                    near.append({"sample_ids": [a, b], "dhash_distance": distance,
                                 "status": "needs_review"})
    groups = defaultdict(list)
    for sample_id in samples:
        groups[find(sample_id)].append(sample_id)
    result = {}
    for members in groups.values():
        group_id = "lg-" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:16]
        result.update({sample_id: group_id for sample_id in members})
    return result, near


def inspect(root: Path, manifest: list[dict], annotations: list[dict], *, near_duplicates: bool = False) -> dict:
    labels = _ids(annotations, "sample_id")
    groups, near = leakage_groups(root, manifest, near_duplicates=near_duplicates)
    return {
        "sample_count": len(manifest), "annotation_count": len(annotations),
        "leakage_groups": len(set(groups.values())), "near_duplicates_needing_review": near,
        "review_status": dict(Counter(row.get("review_status") or "missing" for row in manifest)),
        "client_version": dict(Counter(row.get("client_version") or "unknown" for row in manifest)),
        "scene_labels": dict(Counter((labels.get(row["sample_id"], {}).get("scene") or {}).get("label", "missing") for row in manifest)),
        "building_reviewed": sum((labels.get(row["sample_id"], {}).get("buildings") or {}).get("review_status") == REVIEWED for row in manifest),
        "missing_layout_group": sum(not row.get("layout_group_id") for row in manifest),
        "missing_source_group": sum(not any(row.get(k) for k in ("source_session", "battle_id", "layout_group_id")) for row in manifest),
    }


def _eligible(row: dict, annotation: dict | None, task: str, *, prototype_only: bool = False,
              building_classes: list[str] | None = None) -> bool:
    if (row.get("review_status") != REVIEWED or (not prototype_only and not row.get("client_version"))
            or not isinstance(annotation, dict) or not row.get("reviewer") or not row.get("review_evidence")):
        return False
    if row.get("review_method") not in ({"human", "agent_visual"} if prototype_only else {"human"}):
        return False
    if not any(row.get(key) for key in ("source_session", "battle_id", "layout_group_id")):
        return False
    # Historical scout images often lack a verified layout identifier. A
    # prototype can conservatively group them by source session; formal data
    # still needs the separately reviewed layout identity.
    layout_group = row.get("layout_group_id")
    if task == "building" and not prototype_only and (
            not isinstance(layout_group, str) or not layout_group or
            layout_group.startswith("provisional:")):
        return False
    item = annotation.get("scene" if task == "scene" else "buildings")
    if (not isinstance(item, dict) or item.get("review_status") != REVIEWED
            or not item.get("reviewer") or not item.get("review_evidence")):
        return False
    if item.get("complete") is not True:
        if (task != "building" or item.get("complete") is not False or building_classes is None
                or not isinstance(item.get("complete_classes"), list)
                or any(not isinstance(name, str) for name in item["complete_classes"])
                or len(set(item["complete_classes"])) != len(item["complete_classes"])
                or not set(building_classes) <= set(item["complete_classes"])):
            return False
    if item.get("review_method") not in ({"human", "agent_visual"} if prototype_only else {"human"}):
        return False
    if task == "building" and any(obj.get("ignore") or obj.get("review_status") != REVIEWED for obj in item.get("objects", [])):
        return False
    return True


def _verify_near_duplicate_reviews(near: list[dict], manifest: list[dict],
                                   reviews: list[dict] | None, *, prototype_only: bool) -> list[dict]:
    """Only explicit, source-bound visual verdicts can dismiss dHash candidates."""
    samples = _ids(manifest, "sample_id")
    expected = {tuple(sorted(item["sample_ids"])): item for item in near}
    supplied = {}
    if reviews is not None and not isinstance(reviews, list):
        raise DatasetError("Near-duplicate reviews must be a JSON list")
    for review in reviews or []:
        if not isinstance(review, dict) or not isinstance(review.get("sample_ids"), list) or any(
                not isinstance(item, str) for item in review["sample_ids"]):
            raise DatasetError("Near-duplicate review requires a sample_ids string pair")
        pair = tuple(sorted(review.get("sample_ids", [])))
        if len(pair) != 2 or pair[0] == pair[1] or pair in supplied or pair not in expected:
            raise DatasetError(f"Invalid or stale near-duplicate review pair: {pair}")
        if (review.get("disposition") != "distinct" or
                review.get("dhash_distance") != expected[pair]["dhash_distance"] or
                review.get("source_file_sha256") != {sample_id: samples[sample_id]["file_sha256"] for sample_id in pair} or
                not review.get("reviewer") or not review.get("review_evidence") or
                review.get("review_method") not in ({"human", "agent_visual"} if prototype_only else {"human"})):
            raise DatasetError(f"Near-duplicate review lacks a valid source-bound distinct verdict: {pair}")
        supplied[pair] = review
    missing = expected.keys() - supplied.keys()
    if missing:
        raise DatasetError(f"{len(missing)} near-duplicate pairs need review before split/export")
    return [supplied[pair] for pair in sorted(supplied)]


def split(root: Path, manifest: list[dict], annotations: list[dict], *, task: str,
          seed: int = 42, near_duplicates: bool = True, prototype_only: bool = False,
          near_duplicate_reviews: list[dict] | None = None,
          classes: list[str] | None = None, taxonomy: dict | None = None) -> dict:
    if task == "scene" and classes is not None:
        raise DatasetError("Scene class order comes from taxonomy")
    scope = _building_scope(taxonomy or load_taxonomy(), classes) if classes is not None else None
    labels = _ids(annotations, "sample_id")
    eligible = [row for row in manifest if _eligible(row, labels.get(row["sample_id"]), task,
                prototype_only=prototype_only, building_classes=classes)]
    if not eligible:
        raise DatasetError(f"No reviewed, versioned, fully annotated {task} samples with stable grouping")
    groups, near = leakage_groups(root, manifest, near_duplicates=near_duplicates)
    eligible_ids = {row["sample_id"] for row in eligible}
    relevant = [item for item in near if any(sample_id in eligible_ids for sample_id in item["sample_ids"])]
    reviews = _verify_near_duplicate_reviews(relevant, manifest, near_duplicate_reviews,
                                             prototype_only=prototype_only)
    by_group = defaultdict(list)
    for row in eligible:
        by_group[groups[row["sample_id"]]].append(row["sample_id"])
    if len(by_group) < 3:
        raise DatasetError(f"Only {len(by_group)} independent groups; need at least 3")
    group_ids = sorted(by_group)
    random.Random(seed).shuffle(group_ids)
    count = len(group_ids)
    # Keep related images together while balancing sample counts. Randomising
    # only group counts can put most images into test when one UI group is large.
    group_ids.sort(key=lambda group: len(by_group[group]), reverse=True)
    targets = dict(zip(SPLITS, (len(eligible) * .7, len(eligible) * .15, len(eligible) * .15)))
    allocated = Counter({name: 0 for name in SPLITS})
    assignments = {}
    for index, group_id in enumerate(group_ids):
        empty = [name for name in SPLITS if allocated[name] == 0]
        candidates = empty if len(group_ids) - index == len(empty) else list(SPLITS)
        name = max(candidates, key=lambda split_name: targets[split_name] - allocated[split_name])
        assignments.update({sample_id: name for sample_id in by_group[group_id]})
        allocated[name] += len(by_group[group_id])
    result = {"schema_version": 1, "task": task, "seed": seed, "prototype_only": prototype_only,
            "algorithm": "group_size_greedy_v2", "target_sample_ratios": [0.7, 0.15, 0.15],
            "near_duplicate_reviews": reviews,
            "group_assignments": {group_id: assignments[members[0]] for group_id, members in by_group.items()},
            "sample_assignments": dict(sorted(assignments.items())),
            "sample_groups": {sample_id: groups[sample_id] for sample_id in assignments},
            "group_count": count, "counts": dict(Counter(assignments.values())),
            "created_at_utc": datetime.now(timezone.utc).isoformat()}
    if scope is not None:
        result["building_scope"] = scope
    return result


def _mapping(taxonomy: dict, classes: list[str]) -> dict[str, int]:
    registered = {item["name"]: item["id"] for item in taxonomy["building_classes"]}
    if (not isinstance(classes, list) or not classes or
            any(not isinstance(name, str) or name not in registered for name in classes) or
            len(set(classes)) != len(classes)):
        raise DatasetError("Classes must be unique registered building names")
    # Caller order is intentional and frozen in class_mapping.json.
    return {name: index for index, name in enumerate(classes)}


def _building_scope(taxonomy: dict, classes: list[str]) -> dict:
    indices = _mapping(taxonomy, classes)
    registered = {item["name"]: item["id"] for item in taxonomy["building_classes"]}
    return {"classes": list(classes),
            "stable_ids": {name: registered[name] for name in classes},
            "output_indices": indices}


def export(root: Path, manifest: list[dict], annotations: list[dict], split_doc: dict,
           output: Path, *, task: str, data_version: str, classes: list[str] | None = None,
           taxonomy: dict | None = None, prototype_only: bool = False) -> dict:
    """Create a portable, immutable snapshot from reviewed and validated truth."""
    taxonomy = taxonomy or load_taxonomy()
    if output.exists():
        raise DatasetError(f"Refusing to overwrite snapshot: {output}")
    if not data_version or any(char in data_version for char in "/\\:"):
        raise DatasetError("data_version must be a simple name")
    if split_doc.get("schema_version") != 1 or split_doc.get("task") != task:
        raise DatasetError("Split schema/task mismatch")
    if split_doc.get("prototype_only", False) != prototype_only:
        raise DatasetError("Split prototype_only mode mismatch")
    split_scope = split_doc.get("building_scope")
    if task == "scene" and (classes is not None or split_scope is not None):
        raise DatasetError("Scene class order comes from taxonomy")
    mapping = _mapping(taxonomy, classes or []) if task == "building" else None
    if task == "building" and split_scope is not None:
        if classes is None or split_scope != _building_scope(taxonomy, classes):
            raise DatasetError("Building split/export class scope or taxonomy mapping changed")
    assignments = split_doc.get("sample_assignments", {})
    if not assignments or set(assignments.values()) != set(SPLITS):
        raise DatasetError("Split must contain train/val/test")
    labels, samples = _ids(annotations, "sample_id"), _ids(manifest, "sample_id")
    if not set(assignments) <= set(samples):
        raise DatasetError("Split references unknown sample")
    groups, near = leakage_groups(root, manifest, near_duplicates=True)
    relevant = [item for item in near if any(sample_id in assignments for sample_id in item["sample_ids"])]
    _verify_near_duplicate_reviews(relevant, manifest, split_doc.get("near_duplicate_reviews"),
                                   prototype_only=prototype_only)
    for sample_id, name in assignments.items():
        if name not in SPLITS or not _eligible(samples[sample_id], labels.get(sample_id), task,
                                               prototype_only=prototype_only,
                                               building_classes=classes if split_scope is not None else None):
            raise DatasetError(f"{sample_id}: not eligible for formal {task} export")
        if split_doc.get("sample_groups", {}).get(sample_id) != groups[sample_id]:
            raise DatasetError(f"{sample_id}: grouping changed since split")
    for group_id in set(groups.values()):
        member_splits = {assignments[sample_id] for sample_id in assignments if groups[sample_id] == group_id}
        if len(member_splits) > 1:
            raise DatasetError(f"Leakage group {group_id} crosses splits")
    selected = [samples[sample_id] for sample_id in sorted(assignments)]
    # Complete annotation means all present registered buildings are labeled;
    # exporting a subset must fail if omitted classes would become background.
    if task == "building" and split_scope is None:
        for row in selected:
            objects = labels[row["sample_id"]]["buildings"].get("objects", [])
            omitted = {obj["class_name"] for obj in objects} - set(mapping)
            if omitted:
                raise DatasetError(f"{row['sample_id']}: enabled classes omit labeled objects: {sorted(omitted)}")
    output.mkdir(parents=True)
    try:
        (output / "images").mkdir()
        (output / "annotations").mkdir()
        if task == "building":
            (output / "labels").mkdir()
        split_files = {name: [] for name in SPLITS}
        reviewed = []
        reviewed_manifest = []
        for row in selected:
            sample_id = row["sample_id"]
            source = image_path(root, row["image_path"])
            if sha256(source) != row["file_sha256"]:
                raise DatasetError(f"{sample_id}: source hash changed")
            image = decode(source)
            if [image.shape[1], image.shape[0]] != [row["width"], row["height"]]:
                raise DatasetError(f"{sample_id}: dimensions changed")
            dest = output / "images" / f"{sample_id}{source.suffix.lower()}"
            shutil.copyfile(source, dest)
            split_files[assignments[sample_id]].append(f"images/{dest.name}")
            reviewed_manifest.append({**row, "snapshot_image_path": f"images/{dest.name}",
                "snapshot_split": assignments[sample_id],
                "computed_leakage_group_id": groups[sample_id]})
            annotation = labels[sample_id]
            write_jsonl(output / "annotations" / f"{sample_id}.jsonl", [annotation])
            if task == "scene":
                reviewed.append({"image": f"images/{dest.name}", "label": annotation["scene"]["label"],
                                 "sample_id": sample_id, "group_id": groups[sample_id]})
            else:
                objects = annotation["buildings"].get("objects", [])
                if split_scope is not None:
                    objects = [obj for obj in objects if obj["class_name"] in mapping]
                boxes, lines = [], []
                for obj in objects:
                    name, (x0, y0, x1, y1) = obj["class_name"], obj["bbox"]
                    boxes.append({"class": name, "bbox": [x0, y0, x1, y1]})
                    width, height = row["width"], row["height"]
                    lines.append(f"{mapping[name]} {(x0+x1)/(2*width):.8f} {(y0+y1)/(2*height):.8f} {(x1-x0)/width:.8f} {(y1-y0)/height:.8f}")
                (output / "labels" / f"{sample_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                reviewed.append({"image": f"images/{dest.name}", "boxes": boxes,
                                 "sample_id": sample_id, "group_id": groups[sample_id]})
        for name, paths in split_files.items():
            (output / f"{name}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        write_jsonl(output / f"reviewed-{'scenes' if task == 'scene' else 'buildings'}.jsonl", reviewed)
        write_jsonl(output / "reviewed-manifest.jsonl", reviewed_manifest)
        (output / "split.json").write_text(json.dumps(split_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        snapshot_metadata = {"schema_version": 1, "task": task,
            "data_version": data_version, "prototype_only": prototype_only,
            "review_policy": "agent_or_human_reviewed_prototype" if prototype_only else "human_reviewed",
            "review_methods": sorted({samples[sample_id]["review_method"] for sample_id in assignments}),
            "client_versions": sorted({row["client_version"] or "unknown" for row in selected})}
        if split_scope is not None:
            snapshot_metadata["building_scope"] = split_scope
        (output / "snapshot.json").write_text(json.dumps(snapshot_metadata, indent=2), encoding="utf-8")
        (output / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
        if mapping is not None:
            (output / "class_mapping.json").write_text(json.dumps({"stable_ids": {item["name"]: item["id"] for item in taxonomy["building_classes"] if item["name"] in mapping},
                "output_indices": mapping, **({"building_scope": split_scope} if split_scope is not None else {})}, indent=2), encoding="utf-8")
            (output / "data.yaml").write_text("path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n" +
                "".join(f"  {index}: {name}\n" for name, index in mapping.items()), encoding="utf-8")
        card = (f"# Vision dataset {data_version}\n\nTask: {task}\nStatus: {'prototype_only; no client-version compatibility claim' if prototype_only else 'reviewed formal snapshot; model acceptance pending'}\n"
                f"Samples: {len(selected)}; leakage groups: {len({groups[row['sample_id']] for row in selected})}\n"
                f"Split counts: {dict(Counter(assignments.values()))}\n"
                f"Client versions: {sorted({row['client_version'] or 'unknown' for row in selected})}\n"
                "Source images copied with hash verification. "
                + ("Unknown client versions may be present for offline prototype smoke tests; no current-client compatibility or formal review is claimed. "
                   if prototype_only else "Unknown client versions are excluded. ")
                + (f"Building class scope: {split_scope['classes']}; other annotated classes are excluded from training labels. "
                   if split_scope is not None else "")
                + "Incomplete labels and unresolved near duplicates are excluded.\n")
        (output / "DATASET_CARD.md").write_text(card, encoding="utf-8")
        sums = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
        (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    except Exception:
        if output.resolve().is_relative_to(output.parent.resolve()) and output.name:
            shutil.rmtree(output)
        raise
    return {"output": str(output), "sample_count": len(selected), "sha256s": len(sums)}
