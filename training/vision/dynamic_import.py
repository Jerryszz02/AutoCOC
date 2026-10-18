"""Import explicitly reviewed R windows from a source-hashed candidate queue.

The queue supplies immutable capture facts, never state, deployment, or event truth.
Review rows use the existing dynamic window fields plus queue_session/queue_frame_id
to bind each frame. Only a verified friendly hero card can become hero_skill:<id>.
"""

from __future__ import annotations

import json
from pathlib import Path

from autococ.dynamic_vision import hero_skill_identity

from .data import DatasetError, sha256
from .dynamic_data import EVENT_BY_KIND, verify_dynamic_windows
from .dynamic_snapshot import freeze_dynamic_snapshot


WINDOW_KEYS = frozenset({
    "schema_version", "kind", "window_id", "event_group_id", "split",
    "review_status", "reviewer", "review_method", "review_evidence",
    "prototype_only", "action_kind", "truth_event", "event_at_monotonic",
    "frames",
})
FRAME_KEYS = frozenset({
    "queue_session", "queue_frame_id", "battle_id", "layout_id",
    "camera_epoch", "camera_stable", "zoom_verified", "roi_bbox",
    "state_label", "state_review_status", "state_review_method",
    "state_reviewer", "state_review_evidence", "state_target_id",
    "state_source_frame_sha256", "state_roi_bbox",
})
DEPLOYMENT_KEYS = frozenset({
    "hero_id", "card_identity_verified", "deployed", "review_status",
    "review_method", "reviewer", "review_evidence", "source_frame_sha256",
    "roi_bbox", "frame_id", "session_id", "camera_epoch",
    "captured_at_monotonic", "client_version",
})


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetError(f"Cannot read reviewed R input: {path}: {exc}") from exc


def _rows(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line, object_pairs_hook=_unique_pairs)
                for line in lines if line.strip()]
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetError(f"Cannot read reviewed R rows: {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise DatasetError("Reviewed R input requires nonempty JSONL objects")
    return rows


def _keys(value: object, allowed: frozenset[str], required: frozenset[str],
          context: str) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - allowed:
        raise DatasetError(f"{context}: missing or unrecognized review fields")
    return value


def _queue_index(queue: object, source_root: Path) -> tuple[str, dict[tuple[str, str], dict]]:
    if (not isinstance(queue, dict) or queue.get("schema_version") != 1 or
            queue.get("policy") != "candidate_queue_only; no model training truth or formal human review is claimed" or
            not isinstance(queue.get("client_version"), str) or
            not queue["client_version"] or queue["client_version"] == "unknown" or
            not isinstance(queue.get("sessions"), list)):
        raise DatasetError("Unsupported R candidate queue")
    version = queue["client_version"]
    index = {}
    seen_sessions = set()
    for session in queue["sessions"]:
        if (not isinstance(session, dict) or
                not isinstance(session.get("source_session"), str) or
                not session["source_session"] or
                session["source_session"] in seen_sessions or
                session.get("client_version") != version or
                not isinstance(session.get("battle_frames"), list)):
            raise DatasetError("Conflicting R queue session/version")
        session_id = session["source_session"]
        seen_sessions.add(session_id)
        for frame in session["battle_frames"]:
            if not isinstance(frame, dict) or not isinstance(frame.get("frame_id"), str):
                raise DatasetError("R queue frame identity missing")
            key = (session_id, frame["frame_id"])
            if key in index:
                raise DatasetError("Duplicate R queue frame identity")
            index[key] = frame
    return version, index


def _source_frame(source_root: Path, candidate: dict, session_id: str) -> tuple[str, str]:
    frame_path = candidate.get("frame")
    digest = candidate.get("sha256")
    if not isinstance(frame_path, str) or not isinstance(digest, str):
        raise DatasetError("R queue original frame/hash missing")
    path = Path(frame_path).resolve()
    if (not path.is_relative_to(source_root) or path.parent.name != "frames" or
            path.parent.parent.name != session_id or not path.is_file() or
            sha256(path) != digest):
        raise DatasetError("R queue original frame outside source root or hash mismatch")
    return path.relative_to(source_root).as_posix(), digest


def _reviewed_frame(review: dict, candidate: dict, session_id: str,
                    version: str, source_root: Path, kind: str) -> dict:
    allowed = FRAME_KEYS | ({"target_id"} if kind == "r_building" else
                            {"hero_deployment_truth"})
    _keys(review, allowed, allowed, "R frame")
    image, digest = _source_frame(source_root, candidate, session_id)
    if (review["state_source_frame_sha256"] != digest or
            review["state_roi_bbox"] != review["roi_bbox"] or
            candidate.get("capture_timestamp_basis") != "capture_request_start" or
            not isinstance(candidate.get("capture_started_utc"), str) or
            type(candidate.get("capture_started_monotonic")) not in (int, float)):
        raise DatasetError("R review is not bound to captured source frame/ROI")
    frame = {key: review[key] for key in FRAME_KEYS - {"queue_session", "queue_frame_id"}}
    frame.update(image=image, sha256=digest, parent_image_sha256=digest,
                 session_id=session_id, frame_id=candidate["frame_id"],
                 captured_at_monotonic=candidate["capture_started_monotonic"],
                 captured_at_utc=candidate["capture_started_utc"],
                 client_version=version)
    if kind == "r_building":
        frame["target_id"] = review["target_id"]
    else:
        deployment = _keys(review["hero_deployment_truth"], DEPLOYMENT_KEYS,
                           DEPLOYMENT_KEYS, "R hero deployment")
        hero_id = deployment["hero_id"]
        target = f"hero_skill:{hero_id}"
        if (hero_skill_identity(target) != hero_id or
                review["state_target_id"] != target or
                deployment["card_identity_verified"] is not True or
                deployment["source_frame_sha256"] != digest or
                deployment["roi_bbox"] != review["roi_bbox"] or
                deployment["frame_id"] != candidate["frame_id"] or
                deployment["session_id"] != session_id or
                deployment["camera_epoch"] != review["camera_epoch"] or
                deployment["captured_at_monotonic"] != candidate["capture_started_monotonic"] or
                deployment["client_version"] != version):
            raise DatasetError("R hero identity/deployment review is not source-bound")
        card_candidates = candidate.get("hero_card_candidates")
        if not isinstance(card_candidates, list):
            raise DatasetError("R queue hero-card candidates are missing")
        matches = [item for item in card_candidates
                   if isinstance(item, dict) and
                   item.get("target_id") == f"hero-card:{hero_id}" and
                   item.get("candidate_roi_bbox_source") == review["roi_bbox"]]
        if len(matches) != 1:
            raise DatasetError("R reviewed hero does not match one same-frame card candidate")
        frame["target_id"] = target
        frame["hero_deployment_truth"] = deployment
    return frame


def import_reviewed_r_windows(queue_path: Path, reviews_path: Path,
                              source_root: Path, output_root: Path, *, kind: str,
                              prototype_only: bool) -> dict:
    """Validate reviewed R rows against queue originals, then freeze a new snapshot.

    No queue field supplies a state, event, split, or review decision. The existing
    snapshot freezer validates into a sibling temp directory and renames only
    after successful source and snapshot verification.
    """
    if kind not in EVENT_BY_KIND or type(prototype_only) is not bool:
        raise DatasetError("R import requires kind and explicit prototype policy")
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise DatasetError("R snapshot output already exists")
    version, index = _queue_index(_json(Path(queue_path)), source_root)
    rows = []
    for reviewed in _rows(Path(reviews_path)):
        required = WINDOW_KEYS - {"event_at_monotonic"}
        _keys(reviewed, WINDOW_KEYS, required, "R window")
        if reviewed["kind"] != kind or reviewed["prototype_only"] is not prototype_only:
            raise DatasetError("R reviewed kind/prototype policy differs from import")
        source_frames = reviewed["frames"]
        if not isinstance(source_frames, list) or not source_frames:
            raise DatasetError("R review has no frames")
        frames = []
        building_target = None
        for item in source_frames:
            if not isinstance(item, dict):
                raise DatasetError("R reviewed frame must be an object")
            key = (item.get("queue_session"), item.get("queue_frame_id"))
            candidate = index.get(key)
            if candidate is None:
                raise DatasetError("R reviewed frame absent from candidate queue")
            frame = _reviewed_frame(item, candidate, key[0], version,
                                    source_root, kind)
            if kind == "r_building":
                if building_target is not None and frame["target_id"] != building_target:
                    raise DatasetError("R building window changes target identity")
                building_target = frame["target_id"]
            frames.append(frame)
        row = {key: value for key, value in reviewed.items() if key != "frames"}
        row["frames"] = frames
        rows.append(row)
    verify_dynamic_windows(source_root, rows, kind=kind, prototype_only=prototype_only)
    return freeze_dynamic_snapshot(source_root, rows, output_root, kind=kind,
                                   prototype_only=prototype_only)
