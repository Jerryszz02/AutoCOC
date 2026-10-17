"""Derive reviewable camera-motion evidence from retained live frames, without game input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autococ.terrain import measure_camera_motion


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_evidence(directory: Path) -> dict:
    directory = directory.resolve()
    event_path = directory / "events.jsonl"
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    connected = [row for row in rows if row.get("kind") == "session_connected"]
    if len(connected) != 1 or not connected[0].get("client_version"):
        raise ValueError("Camera evidence needs exactly one versioned session")
    frames, seen = [], set()
    reference = None
    zoom_reference = None
    epoch = 0
    last_scene = None

    def source(value: str) -> Path:
        path = Path(value).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Camera source must be a retained frame inside the session")
        return path

    for row in rows:
        if row.get("kind") in {"line_minimum_zoom", "building_minimum_zoom"}:
            motion = row.get("motion") or {}
            path = source(row["frame"])
            # Bind calibration only to the latest observed frame. An old event
            # cannot silently reauthorize a new camera or another scene.
            if (reference == path and last_scene in {"enemy_village", "battle"}
                    and motion.get("stationary") is True
                    and (row.get("kind") != "line_minimum_zoom" or row.get("verified") is True)):
                zoom_reference = path
            continue
        if row.get("kind") != "observation":
            continue
        if row.get("scene") not in {"enemy_village", "battle"}:
            reference = zoom_reference = None
            last_scene = row.get("scene")
            epoch += 1
            continue
        path = source(row["frame"])
        if path in seen:
            raise ValueError("Duplicate camera frame")
        seen.add(path)
        observations = row.get("observations") or {}
        motion = measure_camera_motion(reference, path) if reference is not None else None
        stable = motion is not None and motion.get("stationary") is True
        # Compare against the calibration too; pairwise small shifts may drift.
        zoom_motion = (measure_camera_motion(zoom_reference, path)
                       if stable and zoom_reference is not None else None)
        zoom_verified = zoom_motion is not None and zoom_motion.get("stationary") is True
        if not stable:
            epoch += 1
            zoom_reference = None
        elif zoom_reference is not None and not zoom_verified:
            epoch += 1
            zoom_reference = None
        frames.append({
            "frame": str(path), "source_sha256": _sha(path),
            "frame_id": observations.get("frame_id"),
            "captured_at_monotonic": observations.get("requested_at_monotonic"),
            "captured_at_utc": (observations.get("captured_at_utc")
                                if observations.get("capture_timestamp_basis") == "capture_request_start" else None),
            "candidate_camera_epoch": f"{directory.name}:camera:{epoch}",
            "candidate_camera_stable": stable, "candidate_zoom_verified": zoom_verified,
            "reference_frame": str(reference) if reference is not None else None,
            "reference_sha256": _sha(reference) if reference is not None else None,
            "pairwise_motion": motion, "calibration_motion": zoom_motion,
            "review_status": "pending",
        })
        reference, last_scene = path, row["scene"]
    return {"schema_version": 1, "session_id": directory.name,
            "client_version": connected[0]["client_version"],
            "source_events_sha256": _sha(event_path), "frames": frames,
            "policy": "offline_camera_candidates_only; no state labels, action or human review claimed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = camera_evidence(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "frames": len(report["frames"])}))


if __name__ == "__main__":
    main()
