"""Promote manually reviewed real battle-frame pairs into validated templates.

Input JSON declares client_version and an alive and destroyed crop from each
of two independent battles, plus a known non-target crop. Coordinates are in
1280x720 space. The
script verifies stable cameras and held-out template scores before writing any
asset. The reviewer, not this tool, identifies building type and level.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from autococ.building_vision import BUILDING_TYPES, _read, building_manifest
from autococ.terrain import measure_camera_motion


def _frame_path(spec_dir: Path, item: dict) -> Path:
    value = item.get("frame")
    if not isinstance(value, str) or not value:
        raise ValueError("Each sample needs a frame path")
    path = Path(value)
    path = (spec_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"Sample frame is missing: {path}")
    return path


def _crop(spec_dir: Path, item: dict) -> tuple[Path, np.ndarray]:
    path = _frame_path(spec_dir, item)
    box = item.get("bbox")
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(value) is not int for value in box)):
        raise ValueError("Sample bbox must be four 1280x720 integer coordinates")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720 and
            20 <= x1 - x0 <= 160 and 20 <= y1 - y0 <= 160):
        raise ValueError(f"Invalid building sample bbox: {box}")
    source = _read(path)
    if source is None:
        raise ValueError(f"Cannot decode sample: {path}")
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    return path, image[y0:y1, x0:x1].copy()


def _score(template: np.ndarray, crop: np.ndarray) -> float:
    if (template.shape[0] > crop.shape[0] or template.shape[1] > crop.shape[1] or
            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).std() < 12):
        return -1.0
    return float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())


def _stable(first: Path, second: Path) -> dict:
    evidence = measure_camera_motion(first, second)
    if evidence.get("stationary") is not True:
        raise ValueError(f"Camera is not independently verified stable: {first} -> {second}")
    return evidence


def validate_variant(raw: dict, spec_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return manifest entry and crops; does not write files."""
    if not isinstance(raw, dict) or raw.get("reviewed_label") is not True:
        raise ValueError("A reviewer must explicitly confirm the building label")
    kind, level = raw.get("type"), raw.get("level")
    if kind not in BUILDING_TYPES or not (type(level) is int and level > 0 or level == "unknown_variant"):
        raise ValueError("Unsupported building type or level")
    if raw.get("zoom") != "minimum":
        raise ValueError("Only minimum-zoom battle samples are supported")
    samples = {}
    for name in ("source_alive", "source_destroyed", "heldout_alive", "heldout_destroyed", "negative"):
        item = raw.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Missing {name} sample")
        samples[name] = _crop(spec_dir, item)
    if samples["source_alive"][0].parent.parent == samples["heldout_alive"][0].parent.parent:
        raise ValueError("Held-out alive frame must come from a different battle report")
    if samples["source_destroyed"][0].parent.parent == samples["heldout_destroyed"][0].parent.parent:
        raise ValueError("Held-out destroyed frame must come from a different battle report")
    source_motion = _stable(samples["source_alive"][0], samples["source_destroyed"][0])
    heldout_motion = _stable(samples["heldout_alive"][0], samples["heldout_destroyed"][0])
    alive, destroyed = samples["source_alive"][1], samples["source_destroyed"][1]
    if alive.shape != destroyed.shape:
        raise ValueError("Alive and destroyed crops must use the same map footprint")
    scores = {"alive_holdout": _score(alive, samples["heldout_alive"][1]),
              "destroyed_holdout": _score(destroyed, samples["heldout_destroyed"][1]),
              "alive_on_destroyed": _score(alive, samples["heldout_destroyed"][1]),
              "destroyed_on_alive": _score(destroyed, samples["heldout_alive"][1]),
              "alive_negative": _score(alive, samples["negative"][1]),
              "destroyed_negative": _score(destroyed, samples["negative"][1])}
    if (scores["alive_holdout"] < .94 or scores["destroyed_holdout"] < .94 or
            max(scores[name] for name in ("alive_on_destroyed", "destroyed_on_alive",
                                           "alive_negative", "destroyed_negative")) >= .80):
        raise ValueError(f"Building sample failed independent discrimination: {scores}")
    slug = f"{kind}-{level}"
    entry = {"type": kind, "level": level, "zoom": "minimum",
             "alive": f"buildings/{slug}-alive.png",
             "destroyed": f"buildings/{slug}-destroyed.png",
             "validation": {"status": "validated",
                 **{name: str(samples[name][0]) for name in samples},
                 "scores": scores, "source_camera_motion": source_motion,
                 "heldout_camera_motion": heldout_motion}}
    return entry, alive, destroyed


def prepare(spec_path: Path, output_root: Path) -> list[dict]:
    source = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("variants"), list) or not source["variants"]:
        raise ValueError("Input JSON needs a nonempty variants list")
    client_version = source.get("client_version")
    if not isinstance(client_version, str) or client_version in {"", "unknown"}:
        raise ValueError("Input JSON needs the independently observed client_version")
    existing = building_manifest(output_root)
    if existing["levels"] and existing.get("client_version") != client_version:
        raise ValueError("Cannot combine building samples from different client versions")
    prepared = [validate_variant(raw, spec_path.parent) for raw in source["variants"]]
    entries = list(existing["levels"])
    keys = {(entry.get("type"), entry.get("level")) for entry in entries}
    for entry, alive, destroyed in prepared:
        key = (entry["type"], entry["level"])
        if key in keys:
            raise ValueError(f"Building variant already exists: {key}")
        keys.add(key)
        for relative, image in ((entry["alive"], alive), (entry["destroyed"], destroyed)):
            path = output_root / relative
            if path.exists():
                raise ValueError(f"Template already exists: {path}")
    # All checks pass before the first write. Never modify source battle frames.
    for entry, alive, destroyed in prepared:
        for relative, image in ((entry["alive"], alive), (entry["destroyed"], destroyed)):
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", image)[1].tofile(path)
        entries.append(entry)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "building_templates.json").write_text(json.dumps({
        "client": source.get("client", existing.get("client", "unknown")),
        "client_version": client_version,
        "coordinate_resolution": [1280, 720], "levels": entries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [entry for entry, _, _ in prepared]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Reviewed JSON annotations")
    parser.add_argument("--output-root", type=Path, default=Path("assets/catalogs"))
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.spec.resolve(), arguments.output_root.resolve()),
                     ensure_ascii=False, indent=2))
