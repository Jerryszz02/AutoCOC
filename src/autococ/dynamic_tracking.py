"""Conservative same-identity ROI tracking and two-distinct-frame events."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np

from .errors import SceneError
from .dynamic_vision import (BUILDING_STATES, CARD_STATES, DynamicFrame,
                             DynamicObservation, DynamicUnavailable, hero_skill_identity)


@dataclass(frozen=True)
class HeroDeploymentEvidence:
    hero_id: str
    deployed: bool
    confidence: float
    session_id: str
    camera_epoch: str
    frame_id: str
    captured_at_monotonic: float
    client_version: str
    source_sha256: str
    roi_bbox: tuple[int, int, int, int]
    method: str  # current-frame friendly battle-card portrait plus enclosed HP


def read_hero_deployment_evidence(source_path: str | Path, frame: DynamicFrame,
                                  roi_bbox: tuple[int, int, int, int],
                                  hero_id: str) -> HeroDeploymentEvidence | None:
    """Independently inspect the original friendly battle card in this frame."""
    if (hero_skill_identity(f"hero_skill:{hero_id}") != hero_id or
            not isinstance(frame.source_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", frame.source_sha256) is None):
        return None
    path = Path(source_path)
    try:
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != frame.source_sha256:
                return None
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (frame.height, frame.width):
            return None
        from .hero_state import recognize_hero_state
        result = recognize_hero_state(path, roi_bbox,
                                      baseline_resolution=(frame.width, frame.height),
                                      unit_id=hero_id, client_version=frame.client_version)
    except (OSError, ValueError, SceneError, cv2.error):
        return None
    identity = result.get("evidence", {}).get("portrait_identity")
    confidence = identity.get("confidence") if isinstance(identity, dict) else None
    if (result.get("deployed") is not True or
            result.get("frame") != str(path) or result.get("slot_bbox") != list(roi_bbox) or
            not isinstance(identity, dict) or identity.get("unit_id") != hero_id or
            identity.get("version_verified") is not True or
            type(confidence) not in (int, float) or not math.isfinite(confidence) or
            confidence < .95 or result.get("evidence", {}).get("health_bar") is None):
        return None
    return HeroDeploymentEvidence(hero_id, True, float(confidence), frame.session_id,
                                  frame.camera_epoch, frame.frame_id,
                                  frame.captured_at_monotonic, frame.client_version,
                                  frame.source_sha256, tuple(roi_bbox),
                                  "friendly_battle_card_portrait_and_enclosed_hp")


@dataclass(frozen=True)
class TrackedObservation:
    track_id: str
    observation: DynamicObservation
    consecutive_state_frames: int


@dataclass(frozen=True)
class DynamicEvent:
    kind: str  # building_destroyed / skill_available
    track_id: str
    target_id: str
    first_frame_id: str
    confirmed_frame_id: str
    first_captured_at_monotonic: float
    confirmed_captured_at_monotonic: float


@dataclass(frozen=True)
class DynamicStep:
    tracked: tuple[TrackedObservation, ...]
    events: tuple[DynamicEvent, ...]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0., right - left) * max(0., bottom - top)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    denominator = area_a + area_b - overlap
    return overlap / denominator if denominator > 0 else 0.


class DynamicTracker:
    def __init__(self, *, min_iou: float = .4, max_gap_seconds: float = 1.0,
                 max_missed_frames: int = 1, event_min_confidence: float = .95,
                 max_frame_age_seconds: float = .5) -> None:
        if (type(min_iou) not in (int, float) or not math.isfinite(min_iou) or not 0 < min_iou <= 1 or
                type(max_gap_seconds) not in (int, float) or not math.isfinite(max_gap_seconds) or not 0 < max_gap_seconds <= 5 or
                type(max_missed_frames) is not int or not 0 <= max_missed_frames <= 2 or
                type(event_min_confidence) not in (int, float) or not math.isfinite(event_min_confidence) or not .5 <= event_min_confidence <= 1 or
                type(max_frame_age_seconds) not in (int, float) or not math.isfinite(max_frame_age_seconds) or not 0 < max_frame_age_seconds <= 2):
            raise ValueError("Invalid frozen dynamic tracker thresholds")
        self.min_iou = float(min_iou)
        self.max_gap_seconds = float(max_gap_seconds)
        self.max_missed_frames = max_missed_frames
        self.event_min_confidence = float(event_min_confidence)
        self.max_frame_age_seconds = float(max_frame_age_seconds)
        self._scope: tuple[str, str, str] | None = None
        self._seen_frame_ids: set[str] = set()
        self._last_capture: float | None = None
        self._serial = 0
        self._tracks: dict[str, dict] = {}

    def update(self, frame: DynamicFrame, observations: list[DynamicObservation], *,
               now_monotonic: float | None = None, offline_replay: bool = False,
               hero_deployments: tuple[HeroDeploymentEvidence, ...] | list[HeroDeploymentEvidence] = ()) -> DynamicStep:
        def reject(reason: str) -> None:
            # An invalid or ambiguous intervening frame breaks two-frame
            # continuity. A later frame must establish a new track/streak.
            self._tracks.clear()
            raise DynamicUnavailable(reason)

        if (not isinstance(frame.session_id, str) or not frame.session_id or
                not isinstance(frame.camera_epoch, str) or not frame.camera_epoch or
                not isinstance(frame.frame_id, str) or not frame.frame_id or
                not isinstance(frame.client_version, str) or not frame.client_version or
                type(frame.width) is not int or type(frame.height) is not int or
                frame.width <= 0 or frame.height <= 0):
            reject("Dynamic tracking requires session/camera epoch/frame ID")
        if not isinstance(observations, (list, tuple)):
            reject("Dynamic observations must be a sequence")
        if not isinstance(hero_deployments, (list, tuple)):
            reject("Hero deployment evidence must be a sequence")
        capture = frame.captured_at_monotonic
        if type(capture) not in (int, float) or not math.isfinite(capture) or capture < 0:
            reject("Dynamic tracking requires capture-start timestamp")
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if (type(now) not in (int, float) or not math.isfinite(now) or
                not offline_replay and (capture > now or now - capture > self.max_frame_age_seconds)):
            reject("Dynamic source frame is stale or has invalid capture age")
        scope = (frame.session_id, frame.camera_epoch, frame.client_version)
        if scope != self._scope:
            self._scope, self._seen_frame_ids, self._last_capture = scope, set(), None
            self._tracks.clear()
            self._serial = 0
        if frame.frame_id in self._seen_frame_ids or (self._last_capture is not None and capture <= self._last_capture):
            reject("Duplicate or out-of-order dynamic frame")
        self._seen_frame_ids.add(frame.frame_id)
        if not frame.camera_stable:
            self._tracks.clear()
            self._last_capture = capture
            return DynamicStep((), ())
        for evidence in hero_deployments:
            if (not isinstance(evidence, HeroDeploymentEvidence) or
                    hero_skill_identity(f"hero_skill:{evidence.hero_id}") != evidence.hero_id or
                    evidence.deployed is not True or
                    type(evidence.confidence) not in (int, float) or
                    not math.isfinite(evidence.confidence) or evidence.confidence < .95 or
                    evidence.session_id != frame.session_id or
                    evidence.camera_epoch != frame.camera_epoch or
                    evidence.frame_id != frame.frame_id or
                    evidence.captured_at_monotonic != capture or
                    evidence.client_version != frame.client_version or
                    not isinstance(frame.source_sha256, str) or
                    evidence.source_sha256 != frame.source_sha256 or
                    re.fullmatch(r"[0-9a-f]{64}", evidence.source_sha256) is None or
                    evidence.method != "friendly_battle_card_portrait_and_enclosed_hp" or
                    not isinstance(evidence.roi_bbox, tuple) or len(evidence.roi_bbox) != 4 or
                    any(type(value) is not int for value in evidence.roi_bbox) or
                    not (0 <= evidence.roi_bbox[0] < evidence.roi_bbox[2] <= frame.width and
                         0 <= evidence.roi_bbox[1] < evidence.roi_bbox[3] <= frame.height)):
                reject("Hero deployment evidence is not bound to the current original frame")
        for obs in observations:
            if (obs.session_id != frame.session_id or obs.camera_epoch != frame.camera_epoch or
                    obs.frame_id != frame.frame_id or obs.captured_at_monotonic != capture or
                    obs.kind not in {"r_building", "r_card", "u_position"} or
                    not isinstance(obs.model_sha256, str) or
                    re.fullmatch(r"[0-9a-f]{64}", obs.model_sha256) is None or
                    not isinstance(obs.contract_sha256, str) or
                    re.fullmatch(r"[0-9a-f]{64}", obs.contract_sha256) is None or
                    not isinstance(obs.label, str) or
                    type(obs.confidence) not in (int, float) or not math.isfinite(obs.confidence) or
                    not 0 <= obs.confidence <= 1 or
                    obs.kind != "u_position" and not obs.target_id or
                    obs.kind == "r_building" and obs.label not in BUILDING_STATES or
                    obs.kind == "r_card" and obs.label not in CARD_STATES or
                    obs.kind == "r_card" and hero_skill_identity(obs.target_id) is None or
                    obs.kind == "r_card" and (not isinstance(frame.source_sha256, str) or
                        re.fullmatch(r"[0-9a-f]{64}", frame.source_sha256) is None or
                        obs.source_sha256 != frame.source_sha256) or
                    obs.kind == "u_position" and not (obs.label == "troop_group" or obs.label.startswith("hero:")) or
                    not isinstance(obs.roi_bbox, tuple) or len(obs.roi_bbox) != 4 or
                    not isinstance(obs.bbox, tuple) or len(obs.bbox) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in (*obs.roi_bbox, *obs.bbox)) or
                    not (0 <= obs.roi_bbox[0] < obs.roi_bbox[2] <= frame.width and
                         0 <= obs.roi_bbox[1] < obs.roi_bbox[3] <= frame.height and
                         obs.roi_bbox[0] <= obs.bbox[0] < obs.bbox[2] <= obs.roi_bbox[2] and
                         obs.roi_bbox[1] <= obs.bbox[1] < obs.bbox[3] <= obs.roi_bbox[3])):
                reject("Dynamic observation is not bound to valid current source coordinates/target")
        # Low-confidence states are unknown, not evidence of an action event.
        observations = [obs for obs in observations if obs.confidence >= self.event_min_confidence]
        # Reject any ambiguous many-to-one or one-to-many association rather
        # than selecting a convenient nearby actor or building.
        matches: dict[int, str] = {}
        for index, obs in enumerate(observations):
            key = (obs.kind, obs.target_id if obs.kind != "u_position" else obs.label,
                   obs.model_sha256, obs.contract_sha256)
            candidates = [track_id for track_id, track in self._tracks.items()
                          if track["key"] == key and track["missed"] <= self.max_missed_frames and
                          capture - track["capture"] <= self.max_gap_seconds and
                          _iou(track["bbox"], obs.bbox) >= self.min_iou]
            if len(candidates) > 1:
                reject("Ambiguous dynamic association: multiple tracks")
            if candidates:
                matches[index] = candidates[0]
        if len(set(matches.values())) != len(matches):
            reject("Ambiguous dynamic association: multiple detections")
        matched_ids = set(matches.values())
        for track_id in list(self._tracks):
            if track_id not in matched_ids:
                self._tracks[track_id]["missed"] += 1
                self._tracks[track_id]["streak"] = 0
                if self._tracks[track_id]["missed"] > self.max_missed_frames:
                    del self._tracks[track_id]
        tracked, events = [], []
        for index, obs in enumerate(observations):
            key = (obs.kind, obs.target_id if obs.kind != "u_position" else obs.label,
                   obs.model_sha256, obs.contract_sha256)
            track_id = matches.get(index)
            if track_id is None:
                self._serial += 1
                track_id = f"{frame.session_id}:{frame.camera_epoch}:{self._serial}"
                self._tracks[track_id] = {"key": key, "bbox": obs.bbox, "capture": capture,
                                          "state": None, "streak": 0, "first_frame": None,
                                          "first_capture": None, "seen_visible": False,
                                          "emitted": set(), "missed": 0}
            track = self._tracks[track_id]
            if obs.kind == "r_building" and obs.label in {"intact", "damaged"}:
                track["seen_visible"] = True
            hero_id = hero_skill_identity(obs.target_id) if obs.kind == "r_card" else None
            skill_context = (obs.kind != "r_card" or obs.label != "available" or
                             len([evidence for evidence in hero_deployments
                                  if evidence.hero_id == hero_id and evidence.roi_bbox == obs.roi_bbox]) == 1)
            if not skill_context:
                track["streak"] = 0
                track["first_frame"] = None
                track["first_capture"] = None
            elif track["state"] == obs.label and track["missed"] == 0 and track["streak"] > 0:
                track["streak"] += 1
            else:
                track["streak"] = 1
                track["first_frame"] = frame.frame_id
                track["first_capture"] = capture
            track["state"] = obs.label
            track["bbox"] = obs.bbox
            track["capture"] = capture
            track["missed"] = 0
            tracked.append(TrackedObservation(track_id, obs, track["streak"]))
            event_kind = ("building_destroyed" if obs.kind == "r_building" and obs.label == "rubble" and
                          track["seen_visible"] else
                          "skill_available" if obs.kind == "r_card" and obs.label == "available" else None)
            if event_kind and track["streak"] >= 2 and event_kind not in track["emitted"]:
                events.append(DynamicEvent(event_kind, track_id, obs.target_id or "",
                                           track["first_frame"], frame.frame_id,
                                           track["first_capture"], capture))
                track["emitted"].add(event_kind)
        self._last_capture = capture
        return DynamicStep(tuple(tracked), tuple(events))
