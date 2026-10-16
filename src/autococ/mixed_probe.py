"""Explicit mixed-roster probes: one balloon and one dragon on different edges.

Both card selections and both single deployments require their own fresh proof.
No failed or uncertain input is repeated. The old two-balloon mode is separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable

from .errors import DeploymentError, FlowError, StopRequested
from .frozen_line import (_COUNT, _cards, _closed_card_border,
                          _closed_dragon_card_border_18_647_2, _count_proves,
                          _digest, _file_sha256, _plan_digest, _post_troop_reading,
                          _verify_frozen_line_post, freeze_model_free_line)
from .prepared_battle import PreparedInput
from .probed_anchor import (ProbedAnchorPlan, _current_card_edges, _current_frame,
                            _distance_to_segment, _foreground_now,
                            _spent_balloon_dark_border, remainder_burst_limit)


@dataclass(frozen=True)
class MixedProbePlan(ProbedAnchorPlan):
    dragon_point: tuple[int, int]
    dragon_bbox: tuple[int, int, int, int]


class ProbeEvidenceConflict(FlowError):
    """A strong contradictory count or identity cannot be cured by another frame."""


class ProbeSelectionPending(FlowError):
    """Only the selected outline is absent from an otherwise proved fresh card."""


def _mixed_digest(plan: MixedProbePlan) -> str:
    return _digest({"base": plan.base.plan_sha256, "anchors": plan.anchors,
                    "balloon_point": plan.balloon_point, "balloon_bbox": plan.balloon_bbox,
                    "dragon_point": plan.dragon_point, "dragon_bbox": plan.dragon_bbox,
                    "checkpoint_cards": plan.checkpoint_cards,
                    "remaining": [asdict(action) for action in plan.remaining]})


def freeze_mixed_probe_line(session: Any, selected: Any, *, terrain: list[dict],
                            **kwargs: Any) -> MixedProbePlan:
    if tuple(session.config.game.baseline_resolution) != (1280, 720):
        raise FlowError("Mixed probes require the sampled 1280×720 layout")
    base = freeze_model_free_line(session, selected, terrain=terrain,
                                  line_tactics="two_edge_furnace_mixed_probes", **kwargs)
    expected = {"electro_dragon": 10, "dragon": 1, "furnace": 1, "balloon": 1}
    if ({unit_id: count for unit_id, source, count in base.troop_budgets if source == "army"} != expected or
            len(base.troop_cards) != 4 or len(base.hero_ids) != 4):
        raise FlowError("Mixed probes require the exact current troop and hero roster")
    segments = []
    for edge in (0, 1):
        points = sorted((tuple(row["point"]) for row in terrain if row["edge"] == edge),
                        key=lambda point: point[1])
        if len(points) != 2:
            raise FlowError("Mixed probes require two independently verified edges")
        segments.append((points[0], points[1]))
    probes = {}
    for unit_id, edge in (("balloon", 0), ("dragon", 1)):
        placements = [action for action in base.actions
                      if action.kind == "deploy" and action.unit_id == unit_id]
        cards = [card for card in base.troop_cards if card[0] == unit_id]
        if len(placements) != 1 or len(cards) != 1 or cards[0][2] != 1 or \
                _distance_to_segment(placements[0].point, *segments[edge]) > 1:
            raise FlowError(f"Mixed {unit_id} probe lacks one frozen edge placement")
        probes[unit_id] = (placements[0].point, cards[0][3], cards[0][4])
    anchors = (probes["balloon"][0], probes["dragon"][0])
    if anchors[0] == anchors[1]:
        raise FlowError("Mixed probe edge points overlap")
    remaining = []
    for action in base.actions:
        if action.unit_id in {"balloon", "dragon"}:
            continue
        if action.kind == "deploy":
            distances = [_distance_to_segment(action.point, *segment) for segment in segments]
            if abs(distances[0] - distances[1]) < 16:
                raise FlowError("Frozen remainder lacks a unique proved edge")
            action = PreparedInput(action.action_id, action.kind,
                                   anchors[0 if distances[0] < distances[1] else 1],
                                   action.unit_id, action.source, action.delay_after_sec)
        remaining.append(action)
    if len([action for action in remaining if action.kind == "deploy"]) != \
            sum(expected.values()) - 2 + len(base.hero_ids):
        raise FlowError("Mixed remainder does not conserve the frozen army")
    before = kwargs.get("toolbar_before")
    if (before is None or str(before.screenshot_path) != base.toolbar_before_frame_id or
            _file_sha256(before.screenshot_path) != base.toolbar_before_sha256):
        raise FlowError("Mixed checkpoint toolbar source differs from frozen source")
    previous = {card["unit_id"]: card for card in _cards(
        before, client_version=base.client_version,
        expected={(unit_id, source): count for unit_id, source, count in base.troop_budgets})
        if card["kind"] == "troop"}
    checkpoint = []
    for unit_id, source, count, point, _bbox in base.troop_cards:
        if unit_id in {"balloon", "dragon"}:
            continue
        card = previous.get(unit_id)
        if (card is None or card["source"] != source or card["count"] != count or
                any(abs(card["point"][axis] - point[axis]) > 4 for axis in (0, 1))):
            raise FlowError("Mixed checkpoint ordinary card geometry is incomplete")
        checkpoint.append((unit_id, tuple(card["point"]), tuple(card["bbox"])))
    empty = MixedProbePlan(base, anchors, probes["balloon"][1], probes["balloon"][2],
                           tuple(checkpoint), tuple(remaining), "",
                           probes["dragon"][1], probes["dragon"][2])
    return MixedProbePlan(**{**empty.__dict__, "digest": _mixed_digest(empty)})


def _probe_card(plan: MixedProbePlan, unit_id: str) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    if unit_id == "balloon":
        return plan.balloon_point, plan.balloon_bbox
    if unit_id == "dragon":
        return plan.dragon_point, plan.dragon_bbox
    raise FlowError("Mixed probe has an unknown card")


def _probe_count(session: Any, plan: MixedProbePlan, snapshot: Any, unit_id: str,
                 expected: int, *, selected: bool = False, pending: int | None = None) -> dict:
    """Exact current OCR, current portrait, source identity, and current border."""
    import cv2
    from .battle_vision import selected_card_bbox
    from .images import read_frame
    from .unit_catalog import recognize_card_identity

    point, bbox = _probe_card(plan, unit_id)
    base = plan.base
    reading = _post_troop_reading(session, snapshot.screenshot_path,
                                  base.toolbar_before_frame_id, point, bbox, unit_id=unit_id)
    actual = reading.get("count")
    reads = reading.get("readings")
    if reading.get("conflict") is True:
        raise ProbeEvidenceConflict(f"Mixed {unit_id} strong_count_conflict")
    if type(actual) is int and actual not in ({expected, pending} if pending is not None else {expected}):
        raise ProbeEvidenceConflict(f"Mixed {unit_id} count differs from frozen {expected} count")
    strong_counts = {int(match[1]) for item in reads if isinstance(item, dict) and
                     type(item.get("confidence")) in (int, float) and
                     math.isfinite(item["confidence"]) and item["confidence"] >= .9 and
                     (match := _COUNT.fullmatch(str(item.get("text", "")).strip())) is not None} \
                    if isinstance(reads, list) else set()
    if type(actual) is int and strong_counts and strong_counts != {actual}:
        raise ProbeEvidenceConflict(f"Mixed {unit_id} strong_count_conflict")
    if (type(actual) is not int or
            type(reading.get("confidence")) not in (int, float) or
            not math.isfinite(reading["confidence"]) or reading["confidence"] < .9 or
            type(reading.get("portrait_score")) not in (int, float) or
            not math.isfinite(reading["portrait_score"]) or reading["portrait_score"] < .92 or
            reading.get("portrait_source") != "current_frame_saved_battle_portrait" or
            reading.get("frame") != str(snapshot.screenshot_path) or
            reading.get("slot_bbox") != list(bbox) or not isinstance(reads, list) or not reads or
            strong_counts != {actual}):
        raise FlowError(f"Mixed {unit_id} has no unique strong same-frame count/portrait")
    if snapshot.observations.get("recognition_scope") == "troop_count":
        fast = snapshot.observations.get("deployment_count_reads")
        if (not isinstance(fast, list) or len(fast) != 1 or
                fast[0].get("frame") != str(snapshot.screenshot_path) or
                fast[0].get("slot_bbox") != list(bbox) or fast[0].get("count") != actual or
                type(fast[0].get("confidence")) not in (int, float) or
                not math.isfinite(fast[0]["confidence"]) or fast[0]["confidence"] < .9):
            raise ProbeEvidenceConflict(f"Mixed {unit_id} fast count conflicts with local count")
    slots = (snapshot.observations.get("battle") or {}).get("slots") or []
    if not isinstance(slots, list):
        raise FlowError("Mixed probe current slots are unavailable")
    for row in slots:
        if not isinstance(row, dict) or row.get("unit_id") != unit_id and row.get("bbox") != list(bbox):
            continue
        confidence = row.get("confidence")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < .9:
            continue
        identity = ((row.get("evidence") or {}).get("identity") or {})
        if (type(row.get("count")) is int and row["count"] != actual or
                row.get("unit_id") not in (None, unit_id) or
                identity.get("unit_id") not in (None, unit_id) or
                identity.get("version_verified") is False or
                identity.get("template_client_version") not in (None, base.client_version)):
            raise ProbeEvidenceConflict(f"Mixed {unit_id} strong current-card conflict")
    if _file_sha256(base.frame.screenshot_path) != base.frame_sha256:
        raise FlowError("Mixed source card pixels changed")
    source = cv2.resize(read_frame(base.frame.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
    identity = recognize_card_identity(source, bbox, "troop", surface="battle",
                                       client_version=base.client_version)
    score = identity.get("confidence")
    if (identity.get("unit_id") != unit_id or identity.get("version_verified") is not True or
            identity.get("template_client_version") != base.client_version or
            type(score) not in (int, float) or not math.isfinite(score) or score < .94):
        raise FlowError(f"Mixed source {unit_id} identity is unverified")
    reading["source_identity"] = identity
    reading["source_identity_frame"] = base.frame.frame_id
    if actual > 0:
        if not selected:
            raise FlowError(f"Mixed {unit_id} is not independently selected")
        border = selected_card_bbox(snapshot.screenshot_path, bbox, base.frame.resolution)
        if border is None:
            raise ProbeSelectionPending(f"Mixed selected {unit_id} lacks a closed card border")
        reading["selected_card_border"] = list(border)
    else:
        image = cv2.resize(read_frame(snapshot.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
        sampled_89_balloon = unit_id == "balloon" and bbox[2] - bbox[0] == 89
        if sampled_89_balloon and base.client_version != "18.647.2":
            raise FlowError("Mixed 89px balloon has no matching versioned border")
        border = (_spent_balloon_dark_border(image, bbox) if sampled_89_balloon else
                  _closed_card_border(image, bbox))
        if border is None and base.client_version == "18.647.2" and not sampled_89_balloon:
            border = (_spent_balloon_dark_border(image, bbox) if unit_id == "balloon" else
                      _closed_dragon_card_border_18_647_2(image, bbox))
        if border is None:
            raise FlowError(f"Mixed spent {unit_id} lacks a closed card border")
        reading["current_card_border"] = border
    return reading


def _remaining_checkpoint(session: Any, plan: MixedProbePlan, snapshot: Any) -> dict:
    """One independent full frame proves both unspent troop cards and four heroes."""
    import cv2
    from .hero_state import recognize_hero_state
    from .images import read_frame
    from .unit_catalog import CATALOG_ROOT, recognize_card_identity, template_manifest

    base = plan.base
    image = cv2.resize(read_frame(snapshot.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
    slots = (snapshot.observations.get("battle") or {}).get("slots")
    if not isinstance(slots, list) or len(plan.checkpoint_cards) != 2:
        raise FlowError("Mixed checkpoint lacks complete ordinary cards")
    ordinary = {unit_id: (point, bbox) for unit_id, point, bbox in plan.checkpoint_cards}
    if set(ordinary) != {"electro_dragon", "furnace"}:
        raise FlowError("Mixed checkpoint ordinary identities are incomplete")
    positive = []
    for unit_id, source, initial, _point, _bbox in base.troop_cards:
        if unit_id in {"balloon", "dragon"}:
            continue
        point, bbox = ordinary[unit_id]
        border = _current_card_edges(image, bbox)
        identity = recognize_card_identity(image, bbox, "troop", surface="battle",
                                           client_version=base.client_version)
        score = identity.get("confidence")
        if (border is None or identity.get("unit_id") != unit_id or
                identity.get("version_verified") is not True or
                type(score) not in (int, float) or not math.isfinite(score) or score < .94):
            raise FlowError(f"Mixed checkpoint {unit_id} identity/border changed")
        if any(isinstance(row, dict) and
               (row.get("unit_id") == unit_id or row.get("bbox") == list(bbox)) and
               type(row.get("count")) is int and row["count"] != initial and
               type(row.get("confidence")) in (int, float) and row["confidence"] >= .9
               for row in slots):
            raise FlowError(f"Mixed checkpoint {unit_id} full-card count conflict")
        matches = [row for row in slots if isinstance(row, dict) and
                   row.get("unit_id") == unit_id and row.get("source") == source and
                   row.get("bbox") == list(bbox) and row.get("count") == initial and
                   type(row.get("confidence")) in (int, float) and row["confidence"] >= .9 and
                   ((row.get("evidence") or {}).get("identity") or {}).get("unit_id") == unit_id and
                   ((row.get("evidence") or {}).get("identity") or {}).get("version_verified") is True and
                   ((row.get("evidence") or {}).get("identity") or {}).get("template_client_version") == base.client_version and
                   _count_proves((row.get("evidence") or {}).get("count") or {}, initial,
                                 snapshot.screenshot_path, bbox, unit_id, base.client_version)]
        local = _post_troop_reading(session, snapshot.screenshot_path, base.toolbar_before_frame_id,
                                    point, bbox, unit_id=unit_id)
        if (local.get("conflict") is True or
                not matches and not _count_proves(local, initial, snapshot.screenshot_path,
                                                   bbox, unit_id, base.client_version) or
                type(local.get("count")) is int and local["count"] != initial and
                type(local.get("confidence")) in (int, float) and local["confidence"] >= .9):
            raise FlowError(f"Mixed checkpoint {unit_id} quantity changed or is unknown")
        positive.append({"unit_id": unit_id, "count": initial, "identity": identity, "border": border,
                         "count_source": "full_card" if matches else "portrait_bound_local"})
    initial_templates = {(entry.get("unit_id"), str((CATALOG_ROOT / entry["path"]).resolve()))
                         for entry in template_manifest(client_version=base.client_version)["templates"]
                         if entry.get("surface") == "battle" and entry.get("state") == "initial_scout"
                         and isinstance(entry.get("path"), str)}
    heroes = []
    for unit_id, bbox in base.hero_cards:
        border = _current_card_edges(image, bbox)
        identity = recognize_card_identity(image, bbox, "hero", surface="battle",
                                           client_version=base.client_version)
        top = (identity.get("evidence") or [{}])[0]
        state = recognize_hero_state(snapshot.screenshot_path, bbox,
                                     unit_id=unit_id, client_version=base.client_version)
        score = identity.get("confidence")
        if (border is None or identity.get("unit_id") != unit_id or
                identity.get("version_verified") is not True or
                type(score) not in (int, float) or not math.isfinite(score) or score < .94 or
                (unit_id, top.get("template")) not in initial_templates or
                state.get("deployed") is True or
                ((state.get("evidence") or {}).get("portrait_identity") or {}).get("unit_id") != unit_id):
            raise FlowError(f"Mixed checkpoint {unit_id} is not an undeployed hero")
        heroes.append({"unit_id": unit_id, "identity": identity, "border": border,
                       "state": state.get("state")})
    boxes = [bbox for _, _, bbox in plan.checkpoint_cards] + \
            [plan.balloon_bbox, plan.dragon_bbox] + [bbox for _, bbox in base.hero_cards]
    if any(left[0] < right[2] and right[0] < left[2]
           for index, left in enumerate(boxes) for right in boxes[index + 1:]):
        raise FlowError("Mixed checkpoint cards overlap")
    return {"frame": str(snapshot.screenshot_path), "sha256": _file_sha256(snapshot.screenshot_path),
            "troops": positive, "heroes": heroes}


def _guard_fast(snapshot: Any, purpose: str) -> None:
    if snapshot.scene == "account_in_use":
        raise FlowError("Game interruption: account_in_use")
    if snapshot.observations.get("deployment_count_conflict") is not None:
        raise FlowError("Mixed probe strong_count_conflict in fast frame")
    scope = snapshot.observations.get("recognition_scope")
    if scope == purpose:
        if (snapshot.observations.get("fast_fallback") is not False or
                snapshot.observations.get("matched_reasons") != ["two_current_hud_anchors"]):
            raise FlowError("Mixed probe fast frame lacks two current HUD anchors")
    elif not (scope == "full" and snapshot.observations.get("fast_fallback") is True and
              snapshot.observations.get("recognition_requested") == purpose):
        raise FlowError("Mixed probe frame lacks guarded fast or full scene recognition")


def execute_mixed_probe_line(session: Any, plan: MixedProbePlan, *,
                             clock: Callable[[], float] = time.monotonic,
                             sleep: Callable[[float], None] = time.sleep) -> Any:
    """Two independent 1→0 probes, then the already frozen remainder."""
    if not isinstance(plan, MixedProbePlan):
        raise FlowError("Mixed probe plan type is invalid")
    base = plan.base
    if (_mixed_digest(plan) != plan.digest or _plan_digest(base) != base.plan_sha256 or
            base.line_tactics != "two_edge_furnace_mixed_probes" or
            base.frame.resolution != (1280, 720) or
            session.last_snapshot is None or
            str(session.last_snapshot.screenshot_path) != base.frame.frame_id or
            clock() >= base.latest_start or session.client_version != base.client_version or
            session.capture.input_display_id != base.frame.display_id or
            tuple(session.context.screen_resolution) != base.frame.source_resolution or
            _file_sha256(base.frame.screenshot_path) != base.frame_sha256 or
            _file_sha256(base.toolbar_before_frame_id) != base.toolbar_before_sha256 or
            base.hero_manifest_frame_id is not None and
            _file_sha256(base.hero_manifest_frame_id) != base.hero_manifest_frame_sha256):
        raise FlowError("Mixed probe frozen source permission is invalid")
    used = getattr(session, "_used_frozen_frames", None)
    if used is None:
        used = set()
        session._used_frozen_frames = used
    if base.frame.frame_id in used:
        raise FlowError("Mixed probe source frame was already attempted")
    used.add(base.frame.frame_id)
    receipt = {"mode": "model_free_line_mixed_probes", "line_tactics": base.line_tactics,
               "plan_id": base.plan_id, "plan_sha256": plan.digest, "battle_id": base.battle_id,
               "frame": base.frame.frame_id, "scope": "troops_and_heroes",
               "excluded_kinds": ["spell", "siege"], "anchors": plan.anchors,
               "probe_limit_sec": 12., "probe_events": [], "actions": [],
               "input_count": len(plan.remaining) + 4, "attempted_placements": 0,
               "issued_placements": 0, "deployed_units": 0, "verified": False,
               "completed": False, "input_sent": False, "any_input_sent": False,
               "burst_pass": False, "consumption_events": [],
               "evidence": [base.frame.frame_id], "error": None}
    session.frozen_line_receipt = receipt
    started = clock()
    total_deadline = min(started + 12, session.deadline, session.task_deadline)
    receipt.update(probe_started_at=started, probe_elapsed_sec=None, probe_authorized_at=None)
    current = session.last_snapshot
    try:
        prior_proof = _current_frame(session, plan, current,
                                     after=base.frame.captured_at - 1, scene="enemy_village")
        for index, (unit_id, edge) in enumerate((("balloon", 0), ("dragon", 1))):
            point, bbox = _probe_card(plan, unit_id)
            anchor = plan.anchors[edge]
            if index == 0 and clock() >= base.latest_start:
                raise FlowError("Mixed first selection missed the preview deadline")
            if index == 1 and current.scene != "battle":
                raise FlowError("Mixed second card requires a confirmed battle frame")
            if (clock() >= total_deadline or current is not session.last_snapshot or
                    _file_sha256(current.screenshot_path) != prior_proof["sha256"] or
                    session.capture.input_display_id != base.frame.display_id or
                    session.client_version != base.client_version):
                raise FlowError("Mixed selection source frame/display changed")
            _foreground_now(session)
            if (_file_sha256(current.screenshot_path) != prior_proof["sha256"] or
                    clock() >= total_deadline or index == 0 and clock() >= base.latest_start):
                raise FlowError("Mixed selection proof expired")
            session.check_deadline()
            selection = {"action_id": f"select-probe-{unit_id}", "kind": "select",
                         "unit_id": unit_id, "point": point, "status": "attempted",
                         "before": prior_proof}
            receipt["actions"].append(selection)
            try:
                session.tap(current, point, reason=f"Select frozen {unit_id} mixed probe card")
            except BaseException:
                selection["status"] = "uncertain"
                raise
            selection.update(status="input_sent", done_at=clock())
            receipt["any_input_sent"] = True
            select_done = selection["done_at"]
            selection_deadline = min(select_done + 3, total_deadline,
                                     base.latest_start if index == 0 else total_deadline,
                                     session.deadline, session.task_deadline)
            selected_proof = None
            for observation in range(3):
                if clock() >= selection_deadline:
                    break
                current = session.observe(f"mixed-{unit_id}-selected-{observation}",
                                          purpose="troop_count",
                                          slot={"point": list(point), "bbox": list(bbox)})
                receipt["evidence"].append(str(current.screenshot_path))
                _guard_fast(current, "troop_count")
                proof = _current_frame(session, plan, current, after=select_done,
                                       scene="enemy_village" if index == 0 else "battle")
                try:
                    reading = _probe_count(session, plan, current, unit_id, 1, selected=True)
                except ProbeSelectionPending as exc:
                    selection.setdefault("pending_frames", []).append({
                        "frame": proof["frame"], "sha256": proof["sha256"],
                        "message": str(exc)})
                    continue  # No input is repeated; only another fresh frame may prove selection.
                if clock() >= selection_deadline:
                    raise FlowError("Mixed selected-card proof exceeded three seconds")
                selected_proof = proof
                selection["reading"] = reading
                break
            if selected_proof is None:
                raise FlowError(f"Mixed {unit_id} has no fresh selected-card proof")
            stage_started = clock()
            stage_deadline = min(stage_started + 6, total_deadline,
                                 session.deadline, session.task_deadline)
            if (stage_started >= stage_deadline or
                    index == 0 and stage_started >= base.latest_start):
                raise FlowError("Mixed probe placement deadline expired")
            if (current is not session.last_snapshot or
                    _file_sha256(current.screenshot_path) != selected_proof["sha256"] or
                    session.capture.input_display_id != base.frame.display_id or
                    session.client_version != base.client_version):
                raise FlowError("Mixed selected-card source changed before placement")
            _foreground_now(session)
            if (_file_sha256(current.screenshot_path) != selected_proof["sha256"] or
                    clock() >= stage_deadline or index == 0 and clock() >= base.latest_start):
                raise FlowError("Mixed placement proof expired")
            session.check_deadline()
            if clock() >= stage_deadline:
                raise FlowError("Mixed placement stopped at deadline")
            event = {"unit_id": unit_id, "edge": edge, "point": anchor,
                     "before_count": 1, "before": selected_proof,
                     "started_at": stage_started, "deadline": stage_deadline,
                     "status": "attempted"}
            receipt["probe_events"].append(event)
            receipt["attempted_placements"] += 1
            action = {"action_id": f"probe-{unit_id}", "kind": "deploy",
                      "unit_id": unit_id, "point": anchor, "status": "attempted"}
            receipt["actions"].append(action)
            try:
                session.tap(current, anchor, reason=f"One frozen {unit_id} mixed probe edge {edge}")
            except BaseException:
                action["status"] = event["status"] = "uncertain"
                raise
            done = clock()
            action.update(status="input_sent", done_at=done)
            event.update(status="input_sent", input_done_at=done)
            receipt["issued_placements"] += 1
            receipt["any_input_sent"] = True
            if done >= stage_deadline:
                raise FlowError("Mixed probe input exceeded six seconds")
            confirmed = None
            for observation in range(2):
                if clock() >= stage_deadline:
                    break
                current = session.observe(f"mixed-{unit_id}-post-{observation}",
                                          purpose="battle_hud")
                receipt["evidence"].append(str(current.screenshot_path))
                _guard_fast(current, "battle_hud")
                if current.scene not in {"enemy_village", "battle"}:
                    continue
                proof = _current_frame(session, plan, current, after=done, scene=current.scene)
                try:
                    reading = _probe_count(session, plan, current, unit_id, 0,
                                           selected=True, pending=1)
                except FlowError as exc:
                    event.setdefault("count_proof_rejections", []).append({
                        "frame": proof["frame"], "sha256": proof["sha256"],
                        "message": str(exc)})
                    if isinstance(exc, ProbeEvidenceConflict) or "source" in str(exc):
                        raise
                    continue  # A new observation is permitted; another input is not.
                if current.scene == "enemy_village":
                    if reading["count"] != 1:
                        raise FlowError("Mixed preview count changed without battle")
                    continue
                if reading["count"] == 1:
                    continue
                if reading["count"] != 0:
                    raise ProbeEvidenceConflict(f"Mixed {unit_id} post count is not zero")
                if clock() >= stage_deadline:
                    raise FlowError("Mixed complete post proof exceeded six seconds")
                confirmed = (proof, reading)
                break
            if confirmed is None:
                raise FlowError(f"Mixed {unit_id} lacks one fresh 1→0 battle proof")
            event.update(status="verified", after=confirmed[0], after_count=0,
                         reading=confirmed[1], verified_at=clock())
            receipt["deployed_units"] += 1
            prior_proof = confirmed[0]
        if clock() >= total_deadline:
            raise FlowError("Mixed final full checkpoint missed twelve seconds")
        current = session.observe("mixed-final-checkpoint")
        receipt["evidence"].append(str(current.screenshot_path))
        if (current.scene != "battle" or
                current.observations.get("recognition_scope") not in (None, "full") or
                current.observations.get("fast_fallback") is True):
            if current.scene == "account_in_use":
                raise FlowError("Game interruption: account_in_use")
            raise FlowError("Mixed final checkpoint is not an independent full battle frame")
        final_proof = _current_frame(session, plan, current,
                                     after=prior_proof["observed_at"], scene="battle")
        balloon = _probe_count(session, plan, current, "balloon", 0)
        dragon = _probe_count(session, plan, current, "dragon", 0)
        roster = _remaining_checkpoint(session, plan, current)
        receipt["checkpoint"] = roster
        receipt["final_checkpoint"] = {"frame": final_proof["frame"],
                                       "sha256": final_proof["sha256"],
                                       "observed_at": final_proof["observed_at"],
                                       "camera": final_proof["camera"],
                                       "balloon": balloon, "dragon": dragon,
                                       "roster_sha256": roster["sha256"], "verified_at": clock()}
        if (clock() >= total_deadline or current is not session.last_snapshot or
                _mixed_digest(plan) != plan.digest or _plan_digest(base) != base.plan_sha256 or
                _file_sha256(current.screenshot_path) != final_proof["sha256"] or
                _file_sha256(base.frame.screenshot_path) != base.frame_sha256 or
                _file_sha256(base.toolbar_before_frame_id) != base.toolbar_before_sha256 or
                base.hero_manifest_frame_id is not None and
                _file_sha256(base.hero_manifest_frame_id) != base.hero_manifest_frame_sha256 or
                session.capture.input_display_id != base.frame.display_id or
                session.client_version != base.client_version):
            raise FlowError("Mixed final authorization expired or source changed")
        for event in receipt["probe_events"]:
            for key in ("before", "after"):
                if _file_sha256(event[key]["frame"]) != event[key]["sha256"]:
                    raise FlowError("Mixed probe proof pixels changed")
        _foreground_now(session)
        session.check_deadline()
        if (clock() >= total_deadline or current is not session.last_snapshot or
                _mixed_digest(plan) != plan.digest or _plan_digest(base) != base.plan_sha256 or
                _file_sha256(current.screenshot_path) != final_proof["sha256"] or
                _file_sha256(base.frame.screenshot_path) != base.frame_sha256 or
                _file_sha256(base.toolbar_before_frame_id) != base.toolbar_before_sha256 or
                base.hero_manifest_frame_id is not None and
                _file_sha256(base.hero_manifest_frame_id) != base.hero_manifest_frame_sha256 or
                session.capture.input_display_id != base.frame.display_id or
                session.client_version != base.client_version):
            raise FlowError("Mixed final foreground/source proof expired")
        receipt["probe_authorized_at"] = clock()
        receipt["probe_elapsed_sec"] = receipt["probe_authorized_at"] - started
        if receipt["probe_authorized_at"] >= total_deadline:
            raise FlowError("Mixed authorization exceeded twelve seconds")
        burst_started = clock()
        limit = remainder_burst_limit(plan)
        deadline = min(burst_started + limit, session.deadline, session.task_deadline)
        receipt["burst_limit_sec"] = limit
        for action in plan.remaining:
            session.check_deadline()
            remaining = deadline - clock()
            if remaining <= 0:
                raise FlowError("Mixed remainder batch deadline exceeded")
            if action.kind == "deploy":
                receipt["attempted_placements"] += 1
            session._active_frozen_plan = (base.plan_id, deadline, action.point)
            try:
                session.frozen_battle_tap(action.point,
                    timeout_sec=min(remaining, session.config.runtime.step_timeout_sec),
                    reason=f"mixed probe {base.plan_id} {action.action_id}", plan_id=base.plan_id)
            except BaseException:
                receipt["actions"].append({"action_id": action.action_id, "kind": action.kind,
                                           "unit_id": action.unit_id, "point": action.point,
                                           "status": "uncertain"})
                raise
            finally:
                session._active_frozen_plan = None
            done = clock()
            receipt["actions"].append({"action_id": action.action_id, "kind": action.kind,
                                       "unit_id": action.unit_id, "point": action.point,
                                       "status": "input_sent", "done_at": done})
            receipt["issued_placements"] += int(action.kind == "deploy")
            if done >= deadline:
                raise FlowError("Mixed remainder exceeded its deadline")
            if action.delay_after_sec:
                if action.delay_after_sec >= deadline - clock():
                    raise FlowError("Mixed fixed delay exceeds batch deadline")
                sleep(action.delay_after_sec)
        receipt["burst_elapsed_sec"] = clock() - burst_started
        receipt["burst_pass"] = receipt["burst_elapsed_sec"] <= limit
        receipt["input_sent"] = receipt["burst_pass"] and \
                                len(receipt["actions"]) == receipt["input_count"]
        if not receipt["input_sent"]:
            raise FlowError("Mixed full action receipt is incomplete")
        return _verify_frozen_line_post(session, base, receipt,
                                        receipt["actions"][-1]["done_at"], sleep=sleep)
    except StopRequested:
        receipt["probe_elapsed_sec"] = receipt["probe_elapsed_sec"] or clock() - started
        raise
    except DeploymentError as exc:
        receipt["probe_elapsed_sec"] = receipt["probe_elapsed_sec"] or clock() - started
        receipt["error"] = str(exc)
        raise
    except Exception as exc:
        receipt["probe_elapsed_sec"] = receipt["probe_elapsed_sec"] or clock() - started
        receipt["error"] = str(exc)
        if "account_in_use" in str(exc):
            receipt["interruption"] = "account_in_use"
        raise DeploymentError(str(exc), partial_receipt=receipt) from exc
    finally:
        session._active_frozen_plan = None
