"""Offline-only bridge from a source-bound dynamic event to a configured wait.

This module never produces an input command or a runnable StrategyStep. R/U
packages have no production release gate yet, so even a matching candidate is
only evidence for a future strategy integration review. A package's report ID
is a metadata claim, not a release or model-quality proof.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import re

import cv2
import numpy as np

from .dynamic_tracking import DynamicStep, HeroDeploymentEvidence, TrackedObservation
from .dynamic_vision import DynamicFrame, DynamicObservation, hero_skill_identity
from .strategy_config import StrategyDefinition


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _original_matches(path: Path, frame: DynamicFrame) -> bool:
    """Bind an offline proposal to the bytes and dimensions of its own PNG."""
    if (not isinstance(path, Path) or not isinstance(frame, DynamicFrame) or
            not isinstance(frame.source_sha256, str) or
            _SHA256.fullmatch(frame.source_sha256) is None):
        return False
    try:
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != frame.source_sha256:
            return False
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return False
    return image is not None and image.shape == (frame.height, frame.width, 3)


@dataclass(frozen=True)
class OfflineDynamicCandidate:
    """A wait that *could* be satisfied; never an authorization to tap."""

    strategy_id: str
    wait_step_index: int
    following_step_index: int
    hero_id: str
    event_track_id: str
    first_frame_id: str
    confirmed_frame_id: str
    session_id: str
    camera_epoch: str
    client_version: str
    model_sha256: str
    contract_sha256: str
    first_source_sha256: str
    confirmed_source_sha256: str
    roi_bbox: tuple[int, int, int, int]
    source_paths: tuple[str, str] = ()


@dataclass(frozen=True)
class DynamicStrategyResult:
    candidate: OfflineDynamicCandidate | None
    reason: str

    @property
    def execution_authorized(self) -> bool:
        return False


def _bound_observation(frame: DynamicFrame, tracked: TrackedObservation,
                       hero_id: str) -> bool:
    obs = tracked.observation
    if not isinstance(obs, DynamicObservation):
        return False
    if (obs.kind != "r_card" or obs.label != "available" or
            obs.target_id != f"hero_skill:{hero_id}" or
            obs.session_id != frame.session_id or obs.camera_epoch != frame.camera_epoch or
            obs.frame_id != frame.frame_id or
            obs.captured_at_monotonic != frame.captured_at_monotonic or
            obs.source_sha256 != frame.source_sha256 or
            type(obs.confidence) not in (int, float) or not math.isfinite(obs.confidence) or
            obs.confidence < .95 or
            not isinstance(obs.roi_bbox, tuple) or len(obs.roi_bbox) != 4 or
            any(type(value) is not int for value in obs.roi_bbox) or
            not (0 <= obs.roi_bbox[0] < obs.roi_bbox[2] <= frame.width and
                 0 <= obs.roi_bbox[1] < obs.roi_bbox[3] <= frame.height) or
            not isinstance(obs.bbox, tuple) or len(obs.bbox) != 4 or
            any(type(value) not in (int, float) or not math.isfinite(value)
                for value in obs.bbox) or
            not (obs.roi_bbox[0] <= obs.bbox[0] < obs.bbox[2] <= obs.roi_bbox[2] and
                 obs.roi_bbox[1] <= obs.bbox[1] < obs.bbox[3] <= obs.roi_bbox[3])):
        return False
    return all(isinstance(value, str) and _SHA256.fullmatch(value)
               for value in (obs.model_sha256, obs.contract_sha256, obs.source_sha256))


def _bound_hero(frame: DynamicFrame, tracked: TrackedObservation,
                evidence: HeroDeploymentEvidence | None, hero_id: str) -> bool:
    obs = tracked.observation
    return (isinstance(evidence, HeroDeploymentEvidence) and
            evidence.hero_id == hero_id and evidence.deployed is True and
            type(evidence.confidence) in (int, float) and
            math.isfinite(evidence.confidence) and evidence.confidence >= .95 and
            evidence.method == "friendly_battle_card_portrait_and_enclosed_hp" and
            evidence.session_id == frame.session_id and
            evidence.camera_epoch == frame.camera_epoch and
            evidence.frame_id == frame.frame_id and
            evidence.captured_at_monotonic == frame.captured_at_monotonic and
            evidence.client_version == frame.client_version and
            evidence.source_sha256 == frame.source_sha256 and
            evidence.roi_bbox == obs.roi_bbox)


def _propose_bound_hero_ready_wait(
    *, definition: StrategyDefinition, wait_step_index: int,
    first_frame: DynamicFrame, first_step: DynamicStep,
    first_hero: HeroDeploymentEvidence | None,
    confirmed_frame: DynamicFrame, confirmed_step: DynamicStep,
    confirmed_hero: HeroDeploymentEvidence | None,
    model: object,
) -> DynamicStrategyResult:
    """Map already byte-checked R-card frames to a configured wait, purely.

    The two DynamicStep values are expected from one tracker, but their
    provenance fields cannot prove they came from the same tracker instance.
    This function checks their common track ID, hero, ROI and original frames.
    """
    reject = lambda reason: DynamicStrategyResult(None, reason)
    if (type(wait_step_index) is not int or wait_step_index < 0 or
            wait_step_index + 1 >= len(definition.steps)):
        return reject("configured_wait_missing")
    wait, following = definition.steps[wait_step_index:wait_step_index + 2]
    hero_id = wait.unit_id
    if (wait.action != "wait" or wait.target != "hero_ready" or
            hero_skill_identity(f"hero_skill:{hero_id}") != hero_id or
            following.action != "activate_ability" or following.unit_id != hero_id):
        return reject("configured_hero_ability_pair_missing")
    if not isinstance(first_frame, DynamicFrame) or not isinstance(confirmed_frame, DynamicFrame):
        return reject("frame_or_camera_binding_invalid")
    validation = getattr(model, "validation", None)
    rules = getattr(model, "event_rules", None)
    max_gap = rules.get("max_gap_seconds") if isinstance(rules, dict) else None
    if (getattr(model, "kind", None) != "r_card" or
            getattr(model, "client_version", None) != confirmed_frame.client_version or
            not isinstance(validation, dict) or validation.get("status") != "evaluated" or
            validation.get("prototype_only") is not False or
            not isinstance(validation.get("report_id"), str) or not validation["report_id"] or
            type(max_gap) not in (int, float) or not math.isfinite(max_gap) or max_gap <= 0):
        return reject("formal_r_card_evaluation_claim_missing")
    if (not isinstance(first_step, DynamicStep) or
            not isinstance(confirmed_step, DynamicStep) or
            not first_frame.camera_stable or not confirmed_frame.camera_stable or
            not first_frame.zoom_verified or not confirmed_frame.zoom_verified or
            (first_frame.session_id, first_frame.camera_epoch, first_frame.client_version,
             first_frame.width, first_frame.height) !=
            (confirmed_frame.session_id, confirmed_frame.camera_epoch,
             confirmed_frame.client_version, confirmed_frame.width, confirmed_frame.height) or
            first_frame.frame_id == confirmed_frame.frame_id or
            not first_frame.frame_id or not confirmed_frame.frame_id or
            type(first_frame.captured_at_monotonic) not in (int, float) or
            type(confirmed_frame.captured_at_monotonic) not in (int, float) or
            not math.isfinite(first_frame.captured_at_monotonic) or
            not math.isfinite(confirmed_frame.captured_at_monotonic) or
            first_frame.captured_at_monotonic >= confirmed_frame.captured_at_monotonic or
            confirmed_frame.captured_at_monotonic - first_frame.captured_at_monotonic > max_gap or
            not all(isinstance(value, str) and _SHA256.fullmatch(value) for value in
                    (first_frame.source_sha256, confirmed_frame.source_sha256))):
        return reject("frame_or_camera_binding_invalid")
    events = [event for event in confirmed_step.events
              if event.kind == "skill_available" and event.target_id == f"hero_skill:{hero_id}" and
              event.first_frame_id == first_frame.frame_id and
              event.confirmed_frame_id == confirmed_frame.frame_id and
              event.first_captured_at_monotonic == first_frame.captured_at_monotonic and
              event.confirmed_captured_at_monotonic == confirmed_frame.captured_at_monotonic]
    if len(events) != 1:
        return reject("unique_two_frame_skill_event_missing")
    event = events[0]
    first = [item for item in first_step.tracked if item.track_id == event.track_id]
    second = [item for item in confirmed_step.tracked if item.track_id == event.track_id]
    if (len(first) != 1 or len(second) != 1 or
            first[0].consecutive_state_frames != 1 or
            second[0].consecutive_state_frames < 2 or
            not _bound_observation(first_frame, first[0], hero_id) or
            not _bound_observation(confirmed_frame, second[0], hero_id) or
            first[0].observation.roi_bbox != second[0].observation.roi_bbox or
            first[0].observation.model_sha256 != second[0].observation.model_sha256 or
            first[0].observation.contract_sha256 != second[0].observation.contract_sha256 or
            first[0].observation.model_sha256 != getattr(model, "model_sha256", None) or
            first[0].observation.contract_sha256 != getattr(model, "contract_sha256", None)):
        return reject("tracked_card_or_model_binding_invalid")
    if (not _bound_hero(first_frame, first[0], first_hero, hero_id) or
            not _bound_hero(confirmed_frame, second[0], confirmed_hero, hero_id)):
        return reject("friendly_hero_deployment_unverified")
    candidate = OfflineDynamicCandidate(
        definition.id, wait_step_index, wait_step_index + 1, hero_id,
        event.track_id, first_frame.frame_id, confirmed_frame.frame_id,
        confirmed_frame.session_id, confirmed_frame.camera_epoch,
        confirmed_frame.client_version, first[0].observation.model_sha256,
        first[0].observation.contract_sha256, first_frame.source_sha256,
        confirmed_frame.source_sha256, first[0].observation.roi_bbox)
    return DynamicStrategyResult(candidate, "offline_candidate_only_release_unverified")


def propose_configured_hero_ready_wait(
    *, definition: StrategyDefinition, wait_step_index: int,
    first_source_path: Path, first_frame: DynamicFrame, first_step: DynamicStep,
    first_hero: HeroDeploymentEvidence | None,
    confirmed_source_path: Path, confirmed_frame: DynamicFrame,
    confirmed_step: DynamicStep, confirmed_hero: HeroDeploymentEvidence | None,
    model: object,
) -> DynamicStrategyResult:
    """Check both original files, then produce only a non-executable hint.

    File hashes and decoded sizes are checked here. The model's validation
    metadata is not a release certificate; no result can authorize an action.
    """
    if (not _original_matches(first_source_path, first_frame) or
            not _original_matches(confirmed_source_path, confirmed_frame)):
        return DynamicStrategyResult(None, "original_source_unverified")
    result = _propose_bound_hero_ready_wait(
        definition=definition, wait_step_index=wait_step_index,
        first_frame=first_frame, first_step=first_step, first_hero=first_hero,
        confirmed_frame=confirmed_frame, confirmed_step=confirmed_step,
        confirmed_hero=confirmed_hero, model=model)
    if result.candidate is None:
        return result
    return DynamicStrategyResult(
        replace(result.candidate, source_paths=(str(first_source_path.resolve()),
                                                str(confirmed_source_path.resolve()))),
        result.reason)
