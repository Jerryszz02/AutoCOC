"""Reviewed dynamic event windows; adjacent frames never split independently."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from autococ.dynamic_vision import BUILDING_STATES, CARD_STATES

from .data import DatasetError, image_path, sha256


EVENT_BY_KIND = {"r_building": "building_destroyed", "r_card": "skill_available"}
STATE_BY_KIND = {"r_building": BUILDING_STATES, "r_card": CARD_STATES}


def verify_dynamic_windows(root: Path, windows: list[dict], *, kind: str,
                           prototype_only: bool) -> dict:
    if kind not in (*EVENT_BY_KIND, "u_position") or not windows:
        raise DatasetError("Dynamic windows require a supported kind and nonempty data")
    ids, seen_frames = set(), {}
    linked: dict[tuple[str, str], set[int]] = defaultdict(set)
    counts = {"positive": 0, "negative": 0, "frames": 0, "groups": 0}
    now_utc = datetime.now(timezone.utc)
    for index, row in enumerate(windows):
        if not isinstance(row, dict) or row.get("schema_version") != 1 or row.get("kind") != kind:
            raise DatasetError("Invalid dynamic window schema/kind")
        window_id = row.get("window_id")
        group = row.get("event_group_id")
        if (not isinstance(window_id, str) or not window_id or window_id in ids or
                not isinstance(group, str) or not group):
            raise DatasetError("Dynamic window/group ID missing or duplicated")
        ids.add(window_id)
        split = row.get("split")
        if split not in {"train", "val", "test"}:
            raise DatasetError("Dynamic window split must be frozen")
        linked[("event_group", group)].add(index)
        if row.get("review_status") != "reviewed" or not row.get("reviewer") or not row.get("review_evidence"):
            raise DatasetError(f"{window_id}: incomplete review provenance")
        method = row.get("review_method")
        if method not in ({"human", "agent_visual"} if prototype_only else {"human"}):
            raise DatasetError(f"{window_id}: review method incompatible with prototype/formal policy")
        if row.get("prototype_only") is not prototype_only:
            raise DatasetError(f"{window_id}: prototype policy mismatch")
        truth = row.get("truth_event")
        if kind in EVENT_BY_KIND:
            if type(truth) is not bool or row.get("action_kind") != EVENT_BY_KIND[kind]:
                raise DatasetError(f"{window_id}: event truth/action is missing")
            counts["positive" if truth else "negative"] += 1
            if truth:
                event_at = row.get("event_at_monotonic")
                if (type(event_at) not in (int, float) or not math.isfinite(event_at) or
                        event_at < 0):
                    raise DatasetError(f"{window_id}: positive event needs finite time evidence")
        elif not isinstance(row.get("truth_boxes"), list):
            raise DatasetError(f"{window_id}: U needs reviewed truth_boxes")
        frames = row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise DatasetError(f"{window_id}: no source frames")
        previous = None
        first_capture = None
        window_source = None
        frame_bounds = {}
        for frame in frames:
            if not isinstance(frame, dict):
                raise DatasetError(f"{window_id}: frame row must be object")
            key = (frame.get("session_id"), frame.get("camera_epoch"), frame.get("frame_id"))
            if any(not isinstance(value, str) or not value for value in key):
                raise DatasetError(f"{window_id}: frame identity missing")
            battle = frame.get("battle_id")
            layout = frame.get("layout_id")
            parent_hash = frame.get("parent_image_sha256")
            if (not isinstance(battle, str) or not battle or
                    not isinstance(layout, str) or not layout or
                    not isinstance(parent_hash, str) or len(parent_hash) != 64 or
                    any(char not in "0123456789abcdef" for char in parent_hash)):
                raise DatasetError(f"{window_id}: battle/layout/parent-image provenance is missing")
            if not prototype_only and (battle.startswith("provisional:") or layout.startswith("provisional:") or
                                       battle == "unknown" or layout == "unknown"):
                raise DatasetError(f"{window_id}: formal source battle/layout grouping is not reviewed")
            source = (key[0], key[1], battle, layout)
            if window_source is not None and source != window_source:
                raise DatasetError(f"{window_id}: one event window spans multiple source battles/cameras/layouts")
            window_source = source
            for source_kind, source_value in (("session", key[0]), ("battle", battle),
                                              ("layout", layout), ("source_hash", frame.get("sha256")),
                                              ("source_hash", parent_hash)):
                linked[(source_kind, source_value)].add(index)
            version = frame.get("client_version")
            if not isinstance(version, str) or not version or (not prototype_only and version == "unknown"):
                raise DatasetError(f"{window_id}: frame client version is missing/unknown for policy")
            if type(frame.get("camera_stable")) is not bool or type(frame.get("zoom_verified")) is not bool:
                raise DatasetError(f"{window_id}: camera/zoom evidence must be explicit booleans")
            captured_utc = frame.get("captured_at_utc")
            try:
                wall_time = datetime.fromisoformat(captured_utc)
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"{window_id}: source capture UTC timestamp is missing") from exc
            if wall_time.tzinfo is None or wall_time.astimezone(timezone.utc) > now_utc:
                raise DatasetError(f"{window_id}: source capture UTC timestamp is future/untrusted")
            started = frame.get("captured_at_monotonic")
            if (type(started) not in (int, float) or not math.isfinite(started) or
                    started < 0 or
                    previous is not None and started <= previous):
                raise DatasetError(f"{window_id}: capture-start times must be finite and increasing")
            if first_capture is None:
                first_capture = started
            previous = started
            identity = (frame.get("sha256"), parent_hash, started)
            if key in seen_frames and seen_frames[key] != identity:
                raise DatasetError(f"{window_id}: reused frame identity has conflicting source evidence")
            seen_frames[key] = identity
            path = image_path(root, frame.get("image"))
            if not path.is_file() or sha256(path) != frame.get("sha256"):
                raise DatasetError(f"{window_id}: source frame SHA-256 mismatch")
            if parent_hash != frame["sha256"]:
                if not isinstance(frame.get("parent_image"), str):
                    raise DatasetError(f"{window_id}: parent source image path is missing")
                parent_path = image_path(root, frame.get("parent_image"))
                if not parent_path.is_file() or sha256(parent_path) != parent_hash:
                    raise DatasetError(f"{window_id}: parent source image SHA-256 mismatch")
            frame_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = (frame_image if parent_hash == frame["sha256"] else
                     cv2.imdecode(np.fromfile(parent_path, dtype=np.uint8), cv2.IMREAD_COLOR))
            if frame_image is None or image is None:
                raise DatasetError(f"{window_id}: source frame cannot be decoded")
            box = frame.get("roi_bbox")
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or
                    not (0 <= box[0] < box[2] <= image.shape[1] and
                         0 <= box[1] < box[3] <= image.shape[0])):
                raise DatasetError(f"{window_id}: ROI outside decoded source image")
            frame_bounds[key[2]] = (image.shape[1], image.shape[0])
            if kind != "u_position":
                if not isinstance(frame.get("target_id"), str) or not frame["target_id"]:
                    raise DatasetError(f"{window_id}: R target identity missing")
                methods = {"human", "agent_visual"} if prototype_only else {"human"}
                if (frame.get("state_label") not in STATE_BY_KIND[kind] or
                        frame.get("state_review_status") != "reviewed" or
                        frame.get("state_review_method") not in methods or
                        not isinstance(frame.get("state_reviewer"), str) or not frame["state_reviewer"] or
                        not isinstance(frame.get("state_review_evidence"), str) or not frame["state_review_evidence"] or
                        frame.get("state_target_id") != frame["target_id"] or
                        frame.get("state_source_frame_sha256") != parent_hash or
                        frame.get("state_roi_bbox") != box):
                    raise DatasetError(f"{window_id}: R frame lacks independently reviewed source-bound state")
            counts["frames"] += 1
        if kind in EVENT_BY_KIND and row["truth_event"] and not first_capture <= row["event_at_monotonic"] <= previous:
            raise DatasetError(f"{window_id}: event time is outside reviewed source frames")
        if kind == "u_position":
            for truth_box in row["truth_boxes"]:
                if not isinstance(truth_box, dict) or truth_box.get("frame_id") not in frame_bounds:
                    raise DatasetError(f"{window_id}: U truth box has no source frame")
                label, box = truth_box.get("label"), truth_box.get("bbox")
                width, height = frame_bounds[truth_box["frame_id"]]
                if (not isinstance(label, str) or not (label == "troop_group" or label.startswith("hero:")) or
                        not isinstance(box, list) or len(box) != 4 or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
                        not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                    raise DatasetError(f"{window_id}: U truth box is invalid/outside source")
    parent = list(range(len(windows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for indices in linked.values():
        first = min(indices)
        for index in indices:
            parent[find(index)] = find(first)
    splits_by_component: dict[int, set[str]] = defaultdict(set)
    for index, row in enumerate(windows):
        splits_by_component[find(index)].add(row["split"])
    if any(len(values) != 1 for values in splits_by_component.values()):
        raise DatasetError("Dynamic source-linked component crosses train/val/test split")
    members: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(windows):
        members[find(index)].append(row["window_id"])
    component_ids = {root_index: "component-" + hashlib.sha256(
        "\n".join(sorted(window_ids)).encode("utf-8")).hexdigest()[:20]
        for root_index, window_ids in members.items()}
    counts["groups"] = len(splits_by_component)
    counts["groups_by_split"] = {split: sum(split in values for values in splits_by_component.values())
                                  for split in ("train", "val", "test")}
    counts["window_components"] = {row["window_id"]: component_ids[find(index)]
                                   for index, row in enumerate(windows)}
    counts["windows"] = len(windows)
    return counts
