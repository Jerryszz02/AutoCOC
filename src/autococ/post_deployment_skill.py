"""One bounded current-client Warden skill attempt after a verified deployment.

This prototype reads each ability phase independently from one screenshot. It
does not authorize an in-burst input or claim a validated dynamic vision model.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any

from .deployment import _minimum_zoom_motion
from .device import foreground_package
from .errors import FlowError, StopRequested
from .hero_state import locate_current_deployed_hero_card, recognize_hero_state
from .terrain import measure_camera_motion


CLIENT_VERSION = "18.647.2"
HERO_ID = "grand_warden"
FRESH_SECONDS = 8.0  # Existing manual sampler's capture-to-input bound.
MAX_AFTER_FRAMES = 2
MAX_EXTRA_BEFORE_FRAMES = 2
SETTLEMENT_RESERVE_SECONDS = 90.0
STABILIZATION_TRANSLATION_LIMIT = 4.0  # Read-only retry only; an input still needs the original <3 proof.


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _foreground(session: Any) -> None:
    package = foreground_package(session.capture.adb, session.capture.serial,
                                 timeout_sec=min(5, session.config.runtime.step_timeout_sec))
    if package != session.config.game.package_name:
        raise FlowError("Post-deployment skill game foreground is unverified")


def _account_conflict(session: Any, exc: BaseException) -> bool:
    return ("account_in_use" in str(exc)
            or getattr(getattr(session, "last_snapshot", None), "scene", None) == "account_in_use")


def _frame(session: Any, snapshot: Any, *, previous: Any | None = None,
           after_input: float | None = None) -> dict[str, object]:
    if snapshot.scene == "account_in_use":
        raise FlowError("Game interruption: account_in_use")
    if (snapshot is not session.last_snapshot or snapshot.scene not in {"battle", "settlement"}
            or type(snapshot.confidence) not in (int, float)
            or not math.isfinite(snapshot.confidence) or snapshot.confidence < .8):
        raise FlowError("Skill observation has no positive current battle or settlement scene")
    observations = snapshot.observations
    captured = observations.get("requested_at_monotonic")
    observed = observations.get("observed_at_monotonic")
    if (type(captured) not in (int, float) or not math.isfinite(captured)
            or type(observed) not in (int, float) or not math.isfinite(observed)
            or observations.get("capture_timestamp_basis") != "capture_request_start"
            or tuple(observations.get("source_resolution") or ()) != (2560, 1440)
            or tuple(observations.get("baseline_resolution") or ()) != (1280, 720)
            or tuple(session.config.game.baseline_resolution) != (1280, 720)
            or session.client_version != CLIENT_VERSION
            or (after_input is not None and captured <= after_input)
            or (previous is not None and
                (captured <= previous.observations.get("requested_at_monotonic", math.inf)
                 or observed <= previous.observations.get("observed_at_monotonic", math.inf)
                 or str(snapshot.screenshot_path) == str(previous.screenshot_path)))):
        raise FlowError("Skill frame source, version, display geometry or capture order is unverified")
    return {"frame": str(snapshot.screenshot_path), "sha256": _sha(snapshot.screenshot_path),
            "capture_request_monotonic": captured, "observed_at_monotonic": observed,
            "scene": snapshot.scene, "confidence": snapshot.confidence}


def _current_warden(session: Any, snapshot: Any) -> dict[str, object]:
    locator = locate_current_deployed_hero_card(snapshot.screenshot_path, HERO_ID,
                                                 client_version=session.client_version)
    if locator is None or locator.get("template_client_version") != CLIENT_VERSION:
        return {"locator": locator, "state": None, "verified": False}
    state = recognize_hero_state(snapshot.screenshot_path, locator["bbox"],
                                 unit_id=HERO_ID, client_version=session.client_version)
    identity = state.get("evidence", {}).get("portrait_identity") or {}
    evidence = state.get("evidence", {})
    used = evidence.get("ability_used") or {}
    ready_anchors = evidence.get("current_equipment_matches") or {}
    closed = evidence.get("closed_card") or {}
    confidence = identity.get("confidence")
    yellow_ready_hp = (evidence.get("yellow_health_bar") is not None
                       and state.get("ability_ready") is True
                       and state.get("ability_used") is False
                       and (evidence.get("ability_ready") or {}).get("yellow_hp_ready_contract") is not None
                       and ready_anchors.get("ready_book") is not None
                       and ready_anchors.get("ready_card") is not None
                       and ready_anchors.get("used_book") is None
                       and ready_anchors.get("used_card") is None
                       and type(confidence) in (int, float) and math.isfinite(confidence)
                       and confidence >= .95
                       and state.get("frame") == str(snapshot.screenshot_path)
                       and closed.get("frame") == str(snapshot.screenshot_path)
                       and tuple(closed.get("bbox") or ()) == tuple(locator["bbox"]))
    yellow_used_hp = (evidence.get("yellow_health_bar") is not None
                      and state.get("ability_used") is True
                      and state.get("ability_ready") is False
                      and used.get("gray_equipment") is not None
                      and used.get("gray_card") is not None)
    return {"locator": locator, "state": state,
            "verified": (identity.get("version_verified") is True
                         and identity.get("unit_id") == HERO_ID
                         and state.get("deployed") is True
                         and (evidence.get("health_bar") is not None
                              or evidence.get("red_health_bar") is not None
                              or yellow_ready_hp or yellow_used_hp)
                         and tuple(locator["bbox"]) == tuple(state.get("slot_bbox") or ())) }


def _complete_deployment(snapshot: Any, hero_id: str) -> bool:
    receipt = snapshot.observations.get("deployment")
    if not isinstance(receipt, dict) or receipt.get("verified") is not True or receipt.get("completed") is not True:
        return False
    if type(receipt.get("deployed_units")) is not int or receipt["deployed_units"] <= 0:
        return False
    heroes = receipt.get("heroes")
    if not isinstance(heroes, list) or len(heroes) != 4 or any(
            not isinstance(row, dict) or row.get("verified") is not True
            or not isinstance(row.get("unit_id"), str) for row in heroes):
        return False
    ids = [row["unit_id"] for row in heroes]
    return len(set(ids)) == 4 and hero_id in ids


def _bounded_stabilization_candidate(motion: dict[str, object]) -> bool:
    """Allow one more observation only when translation alone missed strict proof."""
    affine = motion.get("affine")
    if (not isinstance(affine, list) or len(affine) != 2
            or any(not isinstance(row, list) or len(row) != 3 for row in affine)):
        return False
    dx, dy = affine[0][2], affine[1][2]
    if (type(dx) not in (int, float) or type(dy) not in (int, float)
            or not math.isfinite(dx) or not math.isfinite(dy)
            or not 3 <= math.hypot(dx, dy) < STABILIZATION_TRANSLATION_LIMIT):
        return False
    # Reuse every strict inlier/scale/rotation/stationary gate without changing
    # its action threshold. Zero translation is used only for this decision to
    # take another image; the next real image must pass the unmodified guard.
    without_translation = {**motion, "affine": [[*affine[0][:2], 0.0],
                                              [*affine[1][:2], 0.0]]}
    return _minimum_zoom_motion(without_translation)


def execute_post_deployment_hero_skill(
    session: Any, deployed: Any, *, hero_id: str, delay_sec: float,
) -> tuple[Any, dict[str, object]]:
    """Attempt one current Warden card tap; ordinary uncertainty remains a receipt.

    The first input attempt is persisted on ``session`` *before* the tap. A
    transport exception or missing post-frame can therefore never replay it.
    """
    receipt: dict[str, object] = {"hero_id": hero_id, "verified": False, "completed": False,
                                  "status": "skipped", "reason": "unverified_prerequisite",
                                  "attempted": False, "issued": False, "evidence": []}
    previous = getattr(session, "post_deployment_skill_receipt", None)
    if isinstance(previous, dict) and previous.get("attempted") is True:
        return session.last_snapshot or deployed, previous
    session.post_deployment_skill_receipt = receipt
    latest = deployed

    def fail(reason: str, status: str = "skipped") -> tuple[Any, dict[str, object]]:
        receipt.update(status=status, reason=reason, verified=False, completed=False)
        return latest, receipt

    if (hero_id != HERO_ID or session.client_version != CLIENT_VERSION
            or type(delay_sec) not in (int, float) or not math.isfinite(delay_sec)
            or not 0 <= delay_sec <= 30 or deployed is not session.last_snapshot
            or not _complete_deployment(deployed, hero_id)):
        return fail("client_hero_delay_or_deployment_receipt_unverified")
    display_id = session.capture.input_display_id
    source_path = Path(deployed.screenshot_path)
    try:
        source_sha = _sha(source_path)
        receipt["deployment_source_sha256"] = source_sha
        receipt["input_display_id"] = display_id
        if deployed.scene != "battle" or tuple(deployed.observations.get("source_resolution") or ()) != (2560, 1440):
            return fail("deployment_source_frame_unverified")
        if time.monotonic() + delay_sec >= min(session.deadline, session.task_deadline) - SETTLEMENT_RESERVE_SECONDS:
            return fail("insufficient_settlement_reserve_for_skill")
        start = time.monotonic()
        while time.monotonic() - start < delay_sec:
            session.check_deadline()
            time.sleep(min(.1, max(0., delay_sec - (time.monotonic() - start))))
        session.check_deadline()
        if (session.capture.input_display_id != display_id or session.client_version != CLIENT_VERSION
                or _sha(source_path) != source_sha):
            return fail("deployment_source_or_display_changed")
        _foreground(session)
        fresh = session.observe("post-deployment-skill-before", purpose="full")
        latest = fresh
        before = _frame(session, fresh, previous=deployed)
        session.check_deadline()
        _foreground(session)
        receipt["evidence"].append(before["frame"])
        receipt["before"] = before
        if fresh.scene == "settlement":
            return fail("battle_already_settled")
        if (_sha(source_path) != source_sha or before["sha256"] == source_sha
                or session.capture.input_display_id != display_id or session.client_version != CLIENT_VERSION):
            return fail("deployment_source_or_display_changed")
        motion = measure_camera_motion(source_path, fresh.screenshot_path)
        receipt["camera_motion"] = motion
        first_strict = _minimum_zoom_motion(motion)
        attempts = [{"index": 1, "frame": before, "motion": motion,
                     "strict_motion_verified": first_strict}]
        receipt["before_attempts"] = attempts
        if not first_strict:
            if not _bounded_stabilization_candidate(motion):
                return fail("camera_not_strictly_stationary")
            receipt["stabilization"] = {"first": before, "first_motion": motion,
                                        "status": "one_read_only_observation_pending"}
            seen_paths = {str(source_path), before["frame"]}
            seen_hashes = {source_sha, before["sha256"]}
            for extra in range(1, MAX_EXTRA_BEFORE_FRAMES + 1):
                session.check_deadline()
                if time.monotonic() >= min(session.deadline, session.task_deadline) - SETTLEMENT_RESERVE_SECONDS:
                    return fail("insufficient_settlement_reserve_for_stabilization")
                if session.capture.input_display_id != display_id or session.client_version != CLIENT_VERSION:
                    return fail("stabilization_source_or_display_changed")
                if (_sha(source_path) != source_sha or _sha(fresh.screenshot_path) != before["sha256"]):
                    return fail("stabilization_source_or_display_changed")
                _foreground(session)
                session.check_deadline()
                label = ("post-deployment-skill-stabilized-before" if extra == 1
                         else "post-deployment-skill-stabilized-before-3")
                observed = session.observe(label, purpose="full")
                latest = observed
                current = _frame(session, observed, previous=fresh)
                session.check_deadline()
                _foreground(session)
                receipt["evidence"].append(current["frame"])
                receipt["stabilization"]["second" if extra == 1 else "third"] = current
                if observed.scene == "settlement":
                    return fail("battle_already_settled")
                if (_sha(source_path) != source_sha or _sha(fresh.screenshot_path) != before["sha256"]
                        or session.capture.input_display_id != display_id or session.client_version != CLIENT_VERSION):
                    return fail("stabilization_source_or_display_changed")
                if current["frame"] in seen_paths or current["sha256"] in seen_hashes:
                    return fail("stabilization_frame_not_new")
                seen_paths.add(current["frame"])
                seen_hashes.add(current["sha256"])
                adjacent_motion = measure_camera_motion(fresh.screenshot_path, observed.screenshot_path)
                receipt["stabilization"]["second_motion" if extra == 1 else "third_motion"] = adjacent_motion
                strict = _minimum_zoom_motion(adjacent_motion)
                attempts.append({"index": extra + 1, "frame": current, "motion": adjacent_motion,
                                 "strict_motion_verified": strict})
                fresh, before = observed, current
                receipt["before"] = before
                if strict:
                    receipt["stabilization"]["status"] = "strict_motion_verified"
                    break
                if not _bounded_stabilization_candidate(adjacent_motion) or extra == MAX_EXTRA_BEFORE_FRAMES:
                    receipt["stabilization"]["status"] = "bounded_observations_exhausted"
                    return fail("camera_not_strictly_stationary_after_one_observation" if extra == 1
                                else "camera_not_strictly_stationary_after_two_observations")
                receipt["stabilization"]["status"] = "second_read_only_observation_pending"
        receipt["selected_current_frame"] = before
        target = _current_warden(session, fresh)
        receipt["target_before"] = target
        state = target.get("state") or {}
        if not target["verified"] or state.get("ability_ready") is not True or state.get("ability_used") is not False:
            return fail("current_warden_ready_with_live_hp_unverified")
        left, top, right, bottom = target["locator"]["bbox"]
        point = ((left + right) // 2, (top + bottom) // 2)
        receipt["point"] = list(point)
        # Nothing from a previous screenshot may authorize this action.
        _foreground(session)
        session.check_deadline()
        age = time.monotonic() - before["observed_at_monotonic"]
        if (fresh is not session.last_snapshot or session.capture.input_display_id != display_id
                or session.client_version != CLIENT_VERSION or not 0 <= age <= FRESH_SECONDS
                or time.monotonic() >= min(session.deadline, session.task_deadline) - SETTLEMENT_RESERVE_SECONDS
                or _sha(source_path) != source_sha or _sha(fresh.screenshot_path) != before["sha256"]):
            return fail("fresh_skill_input_source_stale_or_changed")
        receipt.update(status="attempted", reason="input_result_pending", attempted=True)
        # Persist before the only SDK tap. An exception here is action-uncertain.
        session.post_deployment_skill_receipt = receipt
        try:
            session.tap(fresh, point, reason="post_deployment_grand_warden_skill_one_shot")
        except StopRequested:
            raise
        except Exception as exc:
            if _account_conflict(session, exc):
                raise FlowError("Game interruption: account_in_use") from exc
            return fail(f"skill_tap_uncertain:{type(exc).__name__}", "action_uncertain")
        receipt["issued"] = True
        input_done_at = time.monotonic()
        previous_post = fresh
        for index in range(MAX_AFTER_FRAMES):
            session.check_deadline()
            post = session.observe(f"post-deployment-skill-after-{index + 1}", purpose="full")
            latest = post
            seen = _frame(session, post, previous=previous_post, after_input=input_done_at)
            previous_post = post
            if session.capture.input_display_id != display_id:
                return fail("skill_post_display_changed", "attempted_unverified")
            _foreground(session)
            receipt["evidence"].append(seen["frame"])
            receipt.setdefault("after", []).append(seen)
            if post.scene == "settlement":
                return fail("settlement_before_skill_used_verified", "attempted_unverified")
            current = _current_warden(session, post)
            receipt.setdefault("after_targets", []).append(current)
            phase = current.get("state") or {}
            if current["verified"] and phase.get("ability_used") is True and phase.get("ability_ready") is False:
                receipt.update(verified=True, completed=True, status="verified", reason="current_warden_used_verified")
                return latest, receipt
        return fail("current_warden_used_not_verified_in_bounded_after_frames", "attempted_unverified")
    except StopRequested:
        raise
    except FlowError as exc:
        if _account_conflict(session, exc):
            raise
        return fail(f"skill_observation_or_guard_failed:{exc}",
                    "attempted_unverified" if receipt["attempted"] else "skipped")
    except Exception as exc:
        if _account_conflict(session, exc):
            raise FlowError("Game interruption: account_in_use") from exc
        return fail(f"skill_capture_or_read_failed:{type(exc).__name__}",
                    "attempted_unverified" if receipt["attempted"] else "skipped")
