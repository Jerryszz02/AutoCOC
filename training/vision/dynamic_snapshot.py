"""Portable, hash-frozen reviewed R/U windows for offline training and evaluation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .data import DatasetError, image_path, read_jsonl, sha256
from .dynamic_data import STATE_BY_KIND, verify_dynamic_windows


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _classes(windows: list[dict], kind: str) -> list[str]:
    if kind in STATE_BY_KIND:
        return list(STATE_BY_KIND[kind])
    labels = {box["label"] for row in windows for box in row["truth_boxes"]}
    return sorted(labels)


def freeze_dynamic_snapshot(source_root: Path, windows: list[dict], output_root: Path, *,
                            kind: str, prototype_only: bool) -> dict:
    """Copy verified original pixels and review rows into a new immutable-by-hash directory.

    Existing output is never overwritten. Source rows/images are read-only.
    """
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    inventory = verify_dynamic_windows(source_root, windows, kind=kind,
                                       prototype_only=prototype_only)
    if output_root.exists():
        raise DatasetError(f"Dynamic snapshot output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dynamic-snapshot-", dir=output_root.parent) as temporary:
        stage = Path(temporary)
        (stage / "images").mkdir()
        reviewed = deepcopy(sorted(windows, key=lambda row: row["window_id"]))
        for row in reviewed:
            for frame in row["frames"]:
                for key, digest_key in (("image", "sha256"), ("parent_image", "parent_image_sha256")):
                    if key not in frame:
                        continue
                    original = image_path(source_root, frame[key])
                    digest = frame[digest_key]
                    if not original.is_file() or sha256(original) != digest:
                        raise DatasetError("Dynamic source changed during snapshot copy")
                    relative = f"images/{digest}.png"
                    destination = stage / relative
                    if not destination.exists():
                        shutil.copyfile(original, destination)
                    if sha256(destination) != digest:
                        raise DatasetError("Dynamic snapshot image hash mismatch after copy")
                    frame[key] = relative
        (stage / "windows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in reviewed),
            encoding="utf-8")
        assignments = {row["window_id"]: row["split"] for row in reviewed}
        group_assignments = {inventory["window_components"][row["window_id"]]: row["split"]
                             for row in reviewed}
        split = {"schema_version": 1, "kind": kind, "prototype_only": prototype_only,
                 "window_assignments": assignments,
                 "window_components": inventory["window_components"],
                 "component_assignments": group_assignments}
        _write_json(stage / "split.json", split)
        versions = sorted({frame["client_version"] for row in reviewed for frame in row["frames"]})
        metadata = {"schema_version": 1, "kind": kind, "prototype_only": prototype_only,
                    "windows": inventory["windows"], "groups": inventory["groups"],
                    "groups_by_split": inventory["groups_by_split"],
                    "client_versions": versions, "classes": _classes(reviewed, kind),
                    "review_policy": "agent_visual_or_human_prototype" if prototype_only else "human_only_formal",
                    "source_schema": "dynamic_windows_v1"}
        _write_json(stage / "snapshot.json", metadata)
        paths = sorted(path for path in stage.rglob("*") if path.is_file())
        (stage / "SHA256SUMS").write_text("".join(
            f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in paths),
            encoding="utf-8")
        verify_dynamic_snapshot(stage, kind=kind, prototype_only=prototype_only)
        stage.rename(output_root)
    return verify_dynamic_snapshot(output_root, kind=kind, prototype_only=prototype_only)


def verify_dynamic_snapshot(root: Path, *, kind: str,
                            prototype_only: bool | None = None) -> dict:
    """Recompute every file, review source, graph component and frozen assignment."""
    root = Path(root).resolve()
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise DatasetError("Dynamic snapshot SHA256SUMS missing")
    entries = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = (root / relative).resolve()
        if (not separator or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)
                or relative in entries or not path.is_relative_to(root)
                or relative != path.relative_to(root).as_posix()
                or not path.is_file() or path.is_symlink()
                or sha256(path) != digest):
            raise DatasetError(f"Dynamic snapshot hash entry invalid/mismatched: {relative}")
        entries[relative] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(entries) | {"SHA256SUMS"} or not {"windows.jsonl", "split.json", "snapshot.json"} <= entries.keys():
        raise DatasetError("Dynamic snapshot has unhashed/missing files")
    rows = read_jsonl(root / "windows.jsonl")
    metadata = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    policy = metadata.get("prototype_only")
    if (type(policy) is not bool or prototype_only is not None and policy is not prototype_only
            or metadata.get("schema_version") != 1 or metadata.get("kind") != kind
            or metadata.get("source_schema") != "dynamic_windows_v1"
            or split.get("schema_version") != 1 or split.get("kind") != kind
            or split.get("prototype_only") is not policy):
        raise DatasetError("Dynamic snapshot metadata/task/review policy mismatch")
    inventory = verify_dynamic_windows(root, rows, kind=kind, prototype_only=policy)
    expected_images = {frame[key] for row in rows for frame in row["frames"]
                       for key in ("image", "parent_image") if key in frame}
    if (set(entries) - {"windows.jsonl", "split.json", "snapshot.json"} != expected_images
            or any(not relative.startswith("images/") for relative in expected_images)):
        raise DatasetError("Dynamic snapshot image inventory differs from reviewed windows")
    if (metadata.get("windows") != inventory["windows"] or metadata.get("groups") != inventory["groups"]
            or metadata.get("groups_by_split") != inventory["groups_by_split"]
            or metadata.get("classes") != _classes(rows, kind)
            or metadata.get("client_versions") != sorted({frame["client_version"]
                for row in rows for frame in row["frames"]})
            or metadata.get("review_policy") != ("agent_visual_or_human_prototype" if policy else "human_only_formal")):
        raise DatasetError("Dynamic snapshot provenance/classes/counts differ from reviewed rows")
    assignments = {row["window_id"]: row["split"] for row in rows}
    components = inventory["window_components"]
    group_assignments = {components[row["window_id"]]: row["split"] for row in rows}
    if (split.get("window_assignments") != assignments or split.get("window_components") != components
            or split.get("component_assignments") != group_assignments):
        raise DatasetError("Dynamic frozen split differs from reviewed source graph")
    return {"root": str(root), "windows": rows, "metadata": metadata,
            "split_sha256": entries["split.json"], "snapshot_sha256": sha256(sums),
            "inventory": inventory}
