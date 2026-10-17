"""Build a read-only, source-hashed review queue from current-client live frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roi(box: object, source: object, baseline: object) -> list[int] | None:
    if (not isinstance(box, list) or len(box) != 4 or
            not isinstance(source, list) or len(source) != 2 or
            not isinstance(baseline, list) or len(baseline) != 2 or
            any(type(value) is not int for value in [*box, *source, *baseline]) or
            min(*source, *baseline) <= 0):
        return None
    scaled = [round(box[0] * source[0] / baseline[0]),
              round(box[1] * source[1] / baseline[1]),
              round(box[2] * source[0] / baseline[0]),
              round(box[3] * source[1] / baseline[1])]
    return scaled if 0 <= scaled[0] < scaled[2] <= source[0] and 0 <= scaled[1] < scaled[3] <= source[1] else None


def review_queue(session_dirs: list[Path], collection_manifests: list[Path], *,
                 client_version: str, annotation_files: list[Path] | None = None) -> dict:
    """Emit candidates only; every R state/event and S/B annotation remains pending."""
    if not client_version or client_version == "unknown":
        raise ValueError("Review queue needs an explicit known client version")
    taxonomy = json.loads((Path(__file__).resolve().parents[1] / "training" / "vision" /
                           "taxonomy.json").read_text(encoding="utf-8"))
    building_ids = {entry["name"]: entry["id"] for entry in taxonomy["building_classes"]}
    sessions, collections = [], []
    annotations, building_scopes = {}, {}
    for supplied in annotation_files or []:
        path = Path(supplied).resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in annotations:
                raise ValueError(f"Duplicate or missing reviewed sample ID: {sample_id}")
            item = row.get("buildings") or {}
            if (row.get("client_version") != client_version or row.get("prototype_only") is not True
                    or item.get("review_status") != "reviewed" or item.get("review_method") != "agent_visual"
                    or not item.get("reviewer")
                    or not item.get("review_evidence")):
                raise ValueError(f"Annotation is not a reviewed current-version prototype: {sample_id}")
            classes = item.get("complete_classes")
            if item.get("complete") is True:
                if classes is not None:
                    raise ValueError(f"Invalid reviewed building scope: {sample_id}")
            elif (item.get("complete") is not False or not isinstance(classes, list) or not classes
                  or any(not isinstance(name, str) or name not in building_ids for name in classes)
                  or len(set(classes)) != len(classes)):
                raise ValueError(f"Invalid reviewed building scope: {sample_id}")
            annotations[sample_id] = (row, path)
            building_scopes[sample_id] = ({"classes": list(classes),
                                          "stable_ids": {name: building_ids[name] for name in classes},
                                          "output_indices": {name: index for index, name in enumerate(classes)}}
                                         if classes is not None else None)
    seen_frames = set()
    for supplied in session_dirs:
        directory = Path(supplied).resolve()
        events = directory / "events.jsonl"
        if not events.is_file():
            raise ValueError(f"Live session events missing: {events}")
        rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
        connected = [row for row in rows if row.get("kind") == "session_connected"]
        if len(connected) != 1 or connected[0].get("client_version") != client_version:
            raise ValueError(f"Live session version is missing or differs: {directory}")
        frames = []
        for row in rows:
            if row.get("kind") != "observation" or row.get("scene") != "battle":
                continue
            path = Path(row.get("frame", "")).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or path in seen_frames:
                raise ValueError(f"Battle frame outside session, missing or duplicated: {path}")
            seen_frames.add(path)
            observations = row.get("observations") or {}
            capture_started_utc = None
            if observations.get("capture_timestamp_basis") == "capture_request_start":
                candidate = observations.get("captured_at_utc")
                try:
                    parsed = datetime.fromisoformat(candidate)
                    if parsed.tzinfo is not None and parsed <= datetime.now(timezone.utc):
                        capture_started_utc = parsed.isoformat()
                except (TypeError, ValueError):
                    pass
            slots = (observations.get("battle") or {}).get("slots") or []
            cards = []
            for slot in slots:
                if slot.get("kind") != "hero" or not slot.get("unit_id"):
                    continue
                box = _roi(slot.get("bbox"), observations.get("source_resolution"),
                           observations.get("baseline_resolution"))
                if box is not None:
                    cards.append({"target_id": f"hero-card:{slot['unit_id']}",
                                  "candidate_roi_bbox_source": box,
                                  "identity_confidence": slot.get("confidence"),
                                  "state_label": None, "state_review_status": "pending"})
            frames.append({"frame": str(path), "sha256": _hash(path),
                           "frame_id": observations.get("frame_id"),
                           "capture_started_monotonic": observations.get("requested_at_monotonic"),
                           "capture_started_utc": capture_started_utc,
                           "capture_timestamp_basis": ("capture_request_start" if capture_started_utc else None),
                           "event_logged_utc": datetime.fromtimestamp(row["time"], timezone.utc).isoformat(),
                           "camera_epoch": None, "camera_stable": None, "zoom_verified": None,
                           "source_resolution": observations.get("source_resolution"),
                           "battle_phase": (observations.get("battle") or {}).get("phase"),
                           "hero_card_candidates": cards,
                           "building_state_label": None, "event_truth": None,
                           "review_status": "pending"})
        sessions.append({"source_session": directory.name, "battle_group_id": f"provisional:{directory.name}",
                         "client_version": client_version, "battle_frames": frames,
                         "independent_group_count": int(bool(frames))})
    for supplied in collection_manifests:
        manifest = Path(supplied).resolve()
        if not manifest.is_file():
            raise ValueError(f"Collection manifest missing: {manifest}")
        candidates = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("client_version") != client_version:
                raise ValueError(f"Collection sample version differs: {row.get('sample_id')}")
            path = (manifest.parent / row["image_path"]).resolve()
            if not path.is_relative_to(manifest.parent) or not path.is_file() or _hash(path) != row["file_sha256"]:
                raise ValueError(f"Collection image missing/hash mismatch: {row.get('sample_id')}")
            existing = annotations.get(row["sample_id"])
            annotation, annotation_path = existing if existing else (None, None)
            if annotation and annotation.get("source_sha256") != row["file_sha256"]:
                raise ValueError(f"Annotation source hash differs: {row['sample_id']}")
            candidates.append({"sample_id": row["sample_id"], "image": str(path),
                               "sha256": row["file_sha256"], "source_session": row.get("source_session"),
                               "battle_id": row.get("battle_id"),
                               "leakage_group_id": row.get("leakage_group_id"),
                               "client_version": client_version,
                               "scene_label": None,
                               "building_review": ({"status": "agent_visual_reviewed_prototype",
                                                    "reviewer": annotation["buildings"]["reviewer"],
                                                    "annotation_file": str(annotation_path),
                                                    "complete": annotation["buildings"]["complete"],
                                                    "complete_classes": annotation["buildings"].get("complete_classes"),
                                                    "building_scope": building_scopes[row["sample_id"]]}
                                                   if annotation else None),
                               "human_review_status": "pending",
                               "review_status": "agent_visual_reviewed_prototype" if annotation else "pending"})
        collections.append({"manifest": str(manifest), "samples": candidates})
    return {"schema_version": 1, "client_version": client_version,
            "sessions": sessions, "collections": collections,
            "required_before_R_freeze": ["human or agent_visual per-frame state review on original pixels",
                "independent truth_event review; never infer from model or missing detection",
                "actual capture-start UTC, camera epoch and stable camera/zoom provenance",
                "reviewed source battle/layout grouping and fixed whole-component split"],
            "policy": "candidate_queue_only; no model training truth or formal human review is claimed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, action="append", default=[])
    parser.add_argument("--collection-manifest", type=Path, action="append", default=[])
    parser.add_argument("--annotation", type=Path, action="append", default=[],
                        help="Existing current-client prototype building review JSONL")
    parser.add_argument("--client-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = review_queue(args.session, args.collection_manifest,
                              client_version=args.client_version,
                              annotation_files=args.annotation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output),
                          "battle_frames": sum(len(row["battle_frames"]) for row in report["sessions"]),
                          "collection_samples": sum(len(row["samples"]) for row in report["collections"])},
                         ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"vision-review-queue: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
