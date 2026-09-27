"""Bounded prebattle planning and observation-free deployment.

This module deliberately keeps the execution permission separate from the
ordinary, short-lived SceneSnapshot action permission.  A frozen plan is valid
only for the frame, display, toolbar, model and layout checked in preparation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

from .errors import CapabilityUnavailable, DeploymentError, FlowError
from .scene import SceneSnapshot
from .strategies import spread_on_edge
from .terrain import measure_cloud_cover


@dataclass(frozen=True)
class FramePacket:
    frame_id: str
    captured_at: float
    scene: str
    confidence: float
    screenshot_path: Path
    resolution: tuple[int, int]
    source_resolution: tuple[int, int]
    display_id: int | None

    @classmethod
    def from_snapshot(cls, frame: SceneSnapshot, session: Any) -> "FramePacket":
        observed = frame.observations.get("observed_at_monotonic")
        if not isinstance(observed, (int, float)) or not math.isfinite(observed):
            raise FlowError("Battle frame has no capture-start timestamp")
        return cls(str(frame.screenshot_path), float(observed), frame.scene,
                   frame.confidence, frame.screenshot_path,
                   tuple(session.config.game.baseline_resolution),
                   tuple(session.context.screen_resolution),
                   session.capture.input_display_id)


@dataclass(frozen=True)
class PreparedInput:
    action_id: str
    kind: str
    point: tuple[int, int]
    unit_id: str
    source: str
    delay_after_sec: float = 0.0


@dataclass(frozen=True)
class PreparedBattleState:
    frame: FramePacket
    battle_id: str
    preparation_deadline: float
    execution_latest_start: float
    core: tuple[float, float]
    core_confidence: float
    cards: tuple[dict, ...]
    terrain: tuple[dict, ...]
    profile_sha256: str
    model_sha256: str


@dataclass(frozen=True)
class FrozenBattlePlan:
    plan_id: str
    battle_id: str
    frame: FramePacket
    frozen_at: float
    latest_start: float
    burst_limit_sec: float
    mode: str
    candidate_id: str
    decision_source: str
    decision_reason: str
    decision_model: str | None
    decision_elapsed_sec: float
    decision_confidence: float | None
    profile_sha256: str
    model_sha256: str
    cards_sha256: str
    actions: tuple[PreparedInput, ...]
    budgets: tuple[tuple[str, str, str, int], ...]
    plan_sha256: str

    @property
    def input_count(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class ActionReceipt:
    action_id: str
    kind: str
    unit_id: str
    source: str
    point: tuple[int, int]
    attempts: int
    input_done_at: float | None
    issued_count: int
    verified_consumed: int | None
    status: str
    reason: str | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _profile(session: Any) -> tuple[dict, str]:
    setting = session.config.vision_agent.layout_profile
    if not setting:
        raise CapabilityUnavailable("Accepted stable battle layout profile is missing")
    path = Path(setting)
    if not path.is_absolute():
        path = session.config.source_path.resolve().parent / path
    try:
        raw = path.read_bytes()
        profile = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise CapabilityUnavailable(f"Battle layout profile cannot be loaded: {exc}") from exc
    transport = "mumu_native" if session.native is not None else "adb"
    if (not isinstance(profile, dict) or profile.get("accepted") is not True or
            profile.get("cards_stable_after_deployment") is not True or
            profile.get("baseline_resolution") != list(session.config.game.baseline_resolution) or
            profile.get("client_version") != session.client_version or
            profile.get("transport") != transport or
            profile.get("display_id") != session.capture.input_display_id or
            not isinstance(profile.get("zoom"), str) or not profile["zoom"] or
            not isinstance(profile.get("cards"), list) or not profile["cards"]):
        raise CapabilityUnavailable("Battle layout has no accepted stable-card verification")
    if session.client_version == "unknown":
        raise CapabilityUnavailable("Battle client version could not be verified")
    return profile, sha256(raw).hexdigest()


def validate_prepared_capability(session: Any, model: Any) -> tuple[dict, str]:
    """Reject absent/incompatible real-mode packages before any enemy search."""
    profile, profile_sha = _profile(session)
    model_sha = getattr(model, "model_sha256", None)
    if not isinstance(model_sha, str) or profile.get("model_sha256") != model_sha:
        raise CapabilityUnavailable("Layout profile is not accepted for this building model")
    metadata = getattr(model, "metadata", None)
    if (not isinstance(metadata, dict) or
            not isinstance(metadata.get("applicability"), dict) or
            metadata["applicability"].get("zoom") != profile["zoom"]):
        raise CapabilityUnavailable("Model zoom applicability differs from accepted layout")
    validation = metadata.get("validation")
    if (not isinstance(validation, dict) or validation.get("status") != "evaluated" or
            validation.get("report_id") != profile.get("model_validation_report_id")):
        raise CapabilityUnavailable("Accepted layout does not match evaluated building model report")
    return profile, profile_sha


def _card_rows(frame: SceneSnapshot, profile: dict) -> tuple[dict, ...]:
    battle = frame.observations.get("battle")
    slots = battle.get("slots") if isinstance(battle, dict) else None
    if not isinstance(slots, list):
        raise FlowError("Battle toolbar was not observed")
    cards: list[dict] = []
    excluded = profile.get("excluded_kinds", [])
    if not isinstance(excluded, list) or not set(excluded).issubset({"hero", "siege"}):
        raise CapabilityUnavailable("Unsupported excluded card kinds in layout profile")
    for slot in slots:
        if not isinstance(slot, dict) or slot.get("kind") not in {"troop", "spell", "hero", "siege"}:
            continue
        if slot["kind"] in excluded:
            continue
        count = slot.get("count")
        if count == 0:
            continue
        point = slot.get("point")
        unit_id = slot.get("unit_id")
        source = slot.get("source")
        score = slot.get("confidence")
        proof = slot.get("evidence") or {}
        reading = proof.get("count") if isinstance(proof, dict) else None
        if (not isinstance(unit_id, str) or not unit_id or
                source not in {"army", "event"} or
                type(count) is not int or count < 0 or
                type(score) not in (float, int) or not .9 <= score <= 1 or
                not isinstance(point, (list, tuple)) or len(point) != 2 or
                any(type(x) is not int for x in point) or
                not isinstance(reading, dict) or
                type(reading.get("confidence")) not in (float, int) or
                not math.isfinite(reading["confidence"]) or
                not .9 <= reading["confidence"] <= 1):
            raise FlowError("A required battle card has uncertain identity, source, position or count")
        if slot["kind"] in {"hero", "siege"} and count != 1:
            raise FlowError("Hero or siege card quantity is unsupported in continuous mode")
        cards.append({"kind": slot["kind"], "unit_id": unit_id, "source": source,
                      "count": count, "point": tuple(point)})
    if not cards or not any(card["kind"] == "troop" for card in cards):
        raise FlowError("No complete supported troop toolbar was observed")
    if len({(c["kind"], c["unit_id"], c["source"]) for c in cards}) != len(cards):
        raise FlowError("Battle toolbar contains ambiguous duplicate cards")
    accepted = profile["cards"]
    if _digest(cards) != _digest(accepted):
        raise CapabilityUnavailable("Observed card identities, quantities or positions differ from accepted layout")
    return tuple(cards)


def _remaining_seconds(frame: SceneSnapshot) -> int:
    battle = frame.observations.get("battle")
    seconds = battle.get("countdown_seconds") if isinstance(battle, dict) else None
    evidence = battle.get("countdown_evidence") if isinstance(battle, dict) else None
    confidence = evidence.get("confidence") if isinstance(evidence, dict) else None
    if (type(seconds) is not int or not 1 <= seconds <= 120 or
            not isinstance(evidence, dict) or evidence.get("frame") != str(frame.screenshot_path) or
            evidence.get("source") != "current_frame_ocr" or
            type(confidence) not in (int, float) or not math.isfinite(confidence) or
            not .9 <= confidence <= 1):
        raise FlowError("Current preparation countdown is unavailable")
    return seconds


def _check_army_manifest(frame: SceneSnapshot, cards: tuple[dict, ...]) -> None:
    manifest = frame.observations.get("expected_army_manifest")
    if not isinstance(manifest, dict) or manifest.get("complete") is not True or manifest.get("identity_complete") is not True:
        raise FlowError("Independent prebattle army identities and quantities are incomplete")
    expected = {}
    for group, kind in (("troops", "troop"), ("spells", "spell")):
        rows = manifest.get(group)
        if not isinstance(rows, list):
            raise FlowError("Independent prebattle army manifest is incomplete")
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) or
                    type(row.get("count")) is not int or row["count"] <= 0):
                raise FlowError("Independent prebattle army card is unreadable")
            key = kind, row["unit_id"]
            expected[key] = expected.get(key, 0) + row["count"]
    observed = {(c["kind"], c["unit_id"]): c["count"] for c in cards if c["source"] == "army" and
                c["kind"] in {"troop", "spell"}}
    if expected != observed:
        raise FlowError("Battle toolbar does not match the independently checked army")


def _candidate_actions(cards: tuple[dict, ...], terrain: tuple[dict, ...], core: tuple[float, float],
                       *, flank: int, funnel_flank: int, funnel_units: frozenset[str],
                       order: tuple[str, ...], spell_targets: dict) -> tuple[PreparedInput, ...]:
    actions: list[PreparedInput] = []
    # The nearest observed flank enters toward the current core. The other
    # flank is a second finite, fully compiled option for the local/Jev choice.
    lookup = {card["unit_id"]: card for card in cards}
    for unit_id in order:
        card = lookup[unit_id]
        kind, name, source, count = card["kind"], card["unit_id"], card["source"], card["count"]
        actions.append(PreparedInput(f"{name}-select", "select", card["point"], name, source))
        if kind == "spell":
            if spell_targets.get(name) != "core":
                raise CapabilityUnavailable(f"Spell {name} has no accepted fixed core target")
            points = [tuple(round(v) for v in core)] * count
        else:
            target_flank = funnel_flank if name in funnel_units else flank
            points = [tuple(p) for p in spread_on_edge(list(terrain), target_flank, count)]
        for index, point in enumerate(points):
            actions.append(PreparedInput(f"{name}-{index + 1}", "deploy", point, name, source))
    return tuple(actions)


def _plan_digest(plan: FrozenBattlePlan) -> str:
    return _digest({key: value for key, value in asdict(plan).items() if key != "plan_sha256"})


def _validate_actions(actions: tuple[PreparedInput, ...], cards: tuple[dict, ...],
                      resolution: tuple[int, int]) -> tuple[tuple[str, str, str, int], ...]:
    width, height = resolution
    if not actions or len(actions) > 500:
        raise FlowError("Continuous input sequence is empty or unbounded")
    sent: dict[tuple[str, str], int] = {}
    selected: tuple[str, str] | None = None
    for action in actions:
        x, y = action.point
        if not (type(x) is int and type(y) is int and 0 <= x < width and 0 <= y < height):
            raise FlowError("Prepared input coordinate is outside the current display")
        if not 0 <= action.delay_after_sec <= .25:
            raise FlowError("Prepared fixed delay exceeds continuous-mode allowance")
        key = action.unit_id, action.source
        if action.kind == "select":
            selected = key
        elif action.kind == "deploy" and selected == key:
            sent[key] = sent.get(key, 0) + 1
        else:
            raise FlowError("Prepared input sequence has an unselected card")
    for card in cards:
        key = card["unit_id"], card["source"]
        if sent.get(key) != card["count"]:
            raise FlowError("Prepared sequence does not match observed army quantities")
    return tuple((c["kind"], c["unit_id"], c["source"], c["count"]) for c in cards)


def prepare_battle(session: Any, scout: SceneSnapshot, *, model: Any, decider: Any = None,
                   guides: Sequence[dict] = (),
                   clock: Callable[[], float] = time.monotonic) -> FrozenBattlePlan:
    session._validate_snapshot(scout)
    if str(scout.screenshot_path) in getattr(session, "_used_frozen_frames", set()):
        raise FlowError("Current battle frame already started a frozen plan")
    minimum_zoom = scout.observations.get("minimum_zoom")
    if not isinstance(minimum_zoom, dict) or minimum_zoom.get("verified") is not True:
        from .deployment import _prepare_two_edge_view
        expected_army_manifest = scout.observations.get("expected_army_manifest")
        preparation_receipt: dict = {"evidence": []}
        scout, terrain = _prepare_two_edge_view(session, scout, preparation_receipt)
        scout.observations["terrain"] = terrain
        scout.observations["minimum_zoom"] = preparation_receipt["minimum_zoom"]
        scout.observations["expected_army_manifest"] = expected_army_manifest
    frame = FramePacket.from_snapshot(scout, session)
    if frame.scene != "enemy_village" or frame.confidence < .8:
        raise FlowError("Preparation requires a recognized enemy preview")
    profile, profile_sha = validate_prepared_capability(session, model)
    model_sha = model.model_sha256
    if measure_cloud_cover(frame.screenshot_path)["obscured"]:
        raise FlowError("Enemy preview is obscured")
    remaining = _remaining_seconds(scout)
    reserve = session.config.vision_agent.preparation_reserve_sec
    preparation_deadline = frame.captured_at + remaining - reserve
    latest_start = preparation_deadline
    if clock() >= latest_start:
        raise FlowError("Preparation budget expired before model inference")
    cards = _card_rows(scout, profile)
    _check_army_manifest(scout, cards)
    terrain = scout.observations.get("terrain")
    if not terrain or any(not isinstance(item, dict) or item.get("evidence", {}).get("frame") != frame.frame_id
                          for item in terrain):
        raise FlowError("No current-frame legal deployment boundary")
    # Every placement on this terrain comes from the observed red boundary.
    terrain = tuple(terrain)
    from .images import read_frame
    from .model_vision import ModelUnavailable, infer_core_geometry
    image = read_frame(frame.screenshot_path)
    if image is None:
        raise FlowError("Battle frame cannot be decoded for building inference")
    try:
        detections = model.detect(image, captured_at=frame.captured_at, now=clock())
        core = infer_core_geometry(detections, profile, source_resolution=(image.shape[1], image.shape[0]),
                                   baseline_resolution=frame.resolution)
    except (ModelUnavailable, ValueError) as exc:
        raise CapabilityUnavailable(f"Current battle building inference unavailable: {exc}") from exc
    core_threshold = profile.get("core_min_confidence", .9)
    if (type(core_threshold) not in (int, float) or not math.isfinite(core_threshold) or
            not 0 <= core_threshold <= 1):
        raise CapabilityUnavailable("Accepted core confidence threshold is invalid")
    if core is None or not .0 <= core.confidence <= 1 or core.confidence < core_threshold:
        raise FlowError("Core location is unknown or below the accepted confidence threshold")
    core_point = tuple(core.center)
    if clock() >= latest_start:
        raise FlowError("Preparation budget expired after model inference")
    centers = {edge: spread_on_edge(list(terrain), edge, 1)[0] for edge in (0, 1)}
    first = min(centers, key=lambda edge: (centers[edge][0] - core_point[0]) ** 2 +
                (centers[edge][1] - core_point[1]) ** 2)
    accepted_orders = profile.get("candidate_orders")
    names = {card["unit_id"] for card in cards}
    if (not isinstance(accepted_orders, list) or not 1 <= len(accepted_orders) <= 4 or
            any(not isinstance(order, list) or len(order) != len(cards) or
                any(not isinstance(unit, str) for unit in order) or set(order) != names
                for order in accepted_orders) or len(names) != len(cards)):
        raise CapabilityUnavailable("Accepted battle layout has no complete finite deployment orders")
    spell_targets = profile.get("spell_targets", {})
    if not isinstance(spell_targets, dict):
        raise CapabilityUnavailable("Accepted spell targets are invalid")
    funnel_units = profile.get("funnel_units", [])
    troop_names = {card["unit_id"] for card in cards if card["kind"] == "troop"}
    if (not isinstance(funnel_units, list) or any(not isinstance(unit, str) for unit in funnel_units) or
            len(set(funnel_units)) != len(funnel_units) or
            not set(funnel_units).issubset(troop_names)):
        raise CapabilityUnavailable("Funnel units must be distinct observed troop cards")
    funnel_set = frozenset(funnel_units)
    candidate_actions = {}
    candidate_details = {}
    for main_flank in (first, 1 - first):
        funnel_options = (1 - main_flank, main_flank) if funnel_set else (main_flank,)
        for funnel_flank in funnel_options:
            for index, order in enumerate(accepted_orders):
                candidate_id = f"main-{main_flank}-funnel-{funnel_flank}-order-{index}"
                candidate_actions[candidate_id] = _candidate_actions(
                    cards, terrain, core_point, flank=main_flank, funnel_flank=funnel_flank,
                    funnel_units=funnel_set, order=tuple(order), spell_targets=spell_targets)
                candidate_details[candidate_id] = (main_flank, funnel_flank, index)
    candidates = []
    for candidate_id, actions in candidate_actions.items():
        budgets = _validate_actions(actions, cards, frame.resolution)
        main_flank, funnel_flank, order_index = candidate_details[candidate_id]
        entry = centers[main_flank]
        candidates.append({"id": candidate_id, "entry_point": entry,
                           "funnel_entry_point": centers[funnel_flank],
                           "funnel_units": funnel_units,
                           "core_distance_pixels": round(math.dist(entry, core_point), 2),
                           "unit_order": accepted_orders[order_index],
                           "spell_targets": spell_targets,
                           "input_count": len(actions),
                           "burst_limit_sec": min(15., 1. + .25 * len(actions)),
                           "budgets": budgets})
    choice = candidates[0]["id"]
    decision_source = "local"
    decision_reason = "nearest_core_entry"
    decision_model = None
    decision_elapsed_sec = 0.
    decision_confidence = None
    if decider is not None:
        state = {"core": {"center": core_point,
                 "confidence": core.confidence}, "cards": cards,
                 "countdown_remaining_sec": max(0., frame.captured_at + remaining - clock())}
        from .jev import relevant_guides
        selected_guides = relevant_guides(guides, {card["unit_id"] for card in cards})
        try:
            decision = decider.select(state, candidates, deadline=latest_start, guides=selected_guides)
        except Exception:
            decision_reason = "jev_unavailable_local_candidate"
        else:
            selected = getattr(decision, "candidate_id", None)
            if selected in candidate_actions and clock() < latest_start:
                choice = selected
                decision_source = getattr(decision, "source", "local")
                decision_reason = getattr(decision, "reason", "validated_choice")
                decision_model = getattr(decision, "model", None)
                decision_elapsed_sec = getattr(decision, "elapsed_sec", 0.)
                decision_confidence = getattr(decision, "confidence", None)
            else:
                decision_reason = "jev_invalid_or_late_candidate"
    if clock() >= latest_start:
        raise FlowError("Preparation budget expired before plan freeze")
    budget = getattr(session, "evidence_budget", None)
    if budget is not None:
        budget.require_frames((image.shape[1], image.shape[0]), count=6)
    actions = candidate_actions[choice]
    budgets = _validate_actions(actions, cards, frame.resolution)
    state = PreparedBattleState(frame, uuid4().hex, preparation_deadline, latest_start,
                                core_point, core.confidence, cards, terrain, profile_sha, model_sha)
    plan = FrozenBattlePlan(uuid4().hex, state.battle_id, frame, clock(), latest_start,
                            min(15., 1. + .25 * len(actions)), "continuous", choice,
                            decision_source, decision_reason, decision_model,
                            decision_elapsed_sec, decision_confidence,
                            profile_sha, model_sha, _digest(cards), actions,
                            budgets, "")
    from dataclasses import replace
    plan = replace(plan, plan_sha256=_plan_digest(plan))
    session.event("prepared_battle_frozen", plan_id=plan.plan_id, battle_id=plan.battle_id,
                  frame=frame.frame_id, candidate_id=choice, decision_source=decision_source,
                  decision_reason=decision_reason, decision_model=decision_model,
                  decision_elapsed_sec=decision_elapsed_sec, decision_confidence=decision_confidence,
                  input_count=plan.input_count, burst_limit_sec=plan.burst_limit_sec,
                  preparation_deadline=preparation_deadline, plan_sha256=plan.plan_sha256)
    return plan


def _verify_consumption(frame: SceneSnapshot, plan: FrozenBattlePlan,
                        receipts: list[ActionReceipt]) -> dict[str, Any]:
    latest_input = max((r.input_done_at for r in receipts if r.input_done_at is not None),
                       default=plan.frozen_at)
    observed_at = frame.observations.get("observed_at_monotonic")
    fresh = (frame.scene in {"enemy_village", "battle"} and
             str(frame.screenshot_path) != plan.frame.frame_id and
             type(frame.confidence) in (int, float) and math.isfinite(frame.confidence) and
             frame.confidence >= .8 and
             type(observed_at) in (int, float) and math.isfinite(observed_at) and
             observed_at >= latest_input)
    observed = frame.observations.get("battle") if fresh else None
    slots = observed.get("slots") if isinstance(observed, dict) else None
    verified: dict[tuple[str, str], int | None] = {}
    if isinstance(slots, list):
        for kind, unit_id, source, initial in plan.budgets:
            matches = [item for item in slots if isinstance(item, dict) and item.get("kind") == kind and
                       item.get("unit_id") == unit_id and item.get("source") == source and
                       type(item.get("confidence")) in (int, float) and
                       math.isfinite(item["confidence"]) and .9 <= item["confidence"] <= 1 and
                       type(item.get("count")) is int and
                       isinstance((item.get("evidence") or {}).get("count"), dict) and
                       type((item.get("evidence") or {})["count"].get("confidence")) in (int, float) and
                       math.isfinite((item.get("evidence") or {})["count"]["confidence"]) and
                       .9 <= (item.get("evidence") or {})["count"]["confidence"] <= 1]
            verified[(unit_id, source)] = initial - matches[0]["count"] if len(matches) == 1 and 0 <= matches[0]["count"] <= initial else None
    else:
        verified = {(unit_id, source): None for _, unit_id, source, _ in plan.budgets}
    issued = {(unit_id, source): sum(r.issued_count for r in receipts if r.unit_id == unit_id and r.source == source)
              for _, unit_id, source, _ in plan.budgets}
    complete = all(verified[key] == count == issued[key] for _, unit_id, source, count in plan.budgets
                   for key in [(unit_id, source)])
    return {"completed": complete, "verified": complete,
            "post_frame_fresh": fresh,
            "deployed_units": sum(value for value in verified.values() if type(value) is int),
            "offensive_actions": sum(value for value in verified.values() if type(value) is int),
            "issued_placements": sum(r.issued_count for r in receipts),
            "counts": [{"kind": kind, "unit_id": unit_id, "source": source,
                        "expected": count, "issued": issued[(unit_id, source)],
                        "verified_consumed": verified[(unit_id, source)]}
                       for kind, unit_id, source, count in plan.budgets]}


def execute_prepared_battle(session: Any, plan: FrozenBattlePlan, *,
                            clock: Callable[[], float] = time.monotonic) -> SceneSnapshot:
    """Execute only frozen inputs, then take one fresh observation for evidence."""
    if plan.mode != "continuous" or not plan.actions or session.last_snapshot is None:
        raise FlowError("No active continuous battle plan")
    if (str(session.last_snapshot.screenshot_path) != plan.frame.frame_id or
            session.capture.input_display_id != plan.frame.display_id or
            tuple(session.config.game.baseline_resolution) != plan.frame.resolution or
            tuple(session.context.screen_resolution) != plan.frame.source_resolution or
            clock() >= plan.latest_start):
        raise FlowError("Frozen battle permission no longer matches the current frame or display")
    profile, profile_sha = _profile(session)
    if profile_sha != plan.profile_sha256 or _digest(_card_rows(session.last_snapshot, profile)) != plan.cards_sha256:
        raise FlowError("Frozen battle toolbar or profile changed")
    if _plan_digest(plan) != plan.plan_sha256:
        raise FlowError("Frozen battle plan was modified")
    used = getattr(session, "_used_battle_plans", None)
    if used is None:
        used = set()
        session._used_battle_plans = used
    if plan.plan_id in used:
        raise FlowError("Frozen battle plan was already attempted")
    used_frames = getattr(session, "_used_frozen_frames", None)
    if used_frames is None:
        used_frames = set()
        session._used_frozen_frames = used_frames
    if plan.frame.frame_id in used_frames:
        raise FlowError("Battle frame has already started a frozen plan")
    used.add(plan.plan_id)
    used_frames.add(plan.frame.frame_id)
    start = clock()
    deadline = min(start + plan.burst_limit_sec, session.deadline, session.task_deadline)
    session._active_frozen_plan = None
    receipts: list[ActionReceipt] = []
    error: str | None = None
    attempted = 0
    try:
        for action in plan.actions:
            session.check_deadline()
            remaining = deadline - clock()
            if remaining <= 0:
                raise FlowError("Continuous deployment deadline exceeded")
            # This method only sends input; it never calls capture, recognition,
            # a model or the ordinary snapshot-based tap path.
            try:
                if action.kind == "deploy":
                    attempted += 1
                session._active_frozen_plan = (plan.plan_id, deadline, action.point)
                session.frozen_battle_tap(action.point, timeout_sec=min(remaining, session.config.runtime.step_timeout_sec),
                                          reason=f"prepared battle {plan.plan_id} {action.action_id}",
                                          plan_id=plan.plan_id)
            except BaseException as exc:
                receipts.append(ActionReceipt(action.action_id, action.kind, action.unit_id, action.source,
                                              action.point, 1, None, 0, None, "uncertain", str(exc)))
                raise
            finally:
                session._active_frozen_plan = None
            completed_at = clock()
            receipts.append(ActionReceipt(action.action_id, action.kind, action.unit_id, action.source,
                                          action.point, 1, completed_at,
                                          1 if action.kind == "deploy" else 0, None, "input_sent"))
            if completed_at >= deadline:
                raise FlowError("Continuous deployment deadline exceeded after input")
            if action.delay_after_sec:
                wait = action.delay_after_sec
                if wait >= deadline - clock():
                    raise FlowError("Fixed delay exceeds continuous deployment deadline")
                time.sleep(wait)
    except BaseException as exc:
        error = str(exc)
        if isinstance(exc, KeyboardInterrupt):
            session.prepared_battle_receipt = {
                "mode": "continuous", "plan_id": plan.plan_id, "battle_id": plan.battle_id,
                "plan_sha256": plan.plan_sha256, "candidate_id": plan.candidate_id,
                "input_count": plan.input_count, "burst_limit_sec": plan.burst_limit_sec,
                "burst_elapsed_sec": clock() - start, "burst_pass": False,
                "actions": [asdict(r) for r in receipts], "attempted_placements": attempted,
                "issued_placements": sum(r.issued_count for r in receipts),
                "input_sent": False, "verified": False, "completed": False,
                "error": error, "evidence": [plan.frame.frame_id]}
            session._active_frozen_plan = None
            session.event("prepared_battle_interrupted", plan_id=plan.plan_id,
                          attempted_placements=attempted, error=error)
            raise
    finally:
        session._active_frozen_plan = None
    elapsed = clock() - start
    # The window is closed before observation. Preserve the queue even if the
    # user stops while capture is pending; never replay uncertain actions.
    post: SceneSnapshot | None = None
    deployment = {"mode": "continuous", "plan_id": plan.plan_id, "battle_id": plan.battle_id,
                  "candidate_id": plan.candidate_id, "decision_source": plan.decision_source,
                  "decision_reason": plan.decision_reason, "decision_model": plan.decision_model,
                  "decision_elapsed_sec": plan.decision_elapsed_sec,
                  "decision_confidence": plan.decision_confidence,
                  "plan_sha256": plan.plan_sha256, "input_count": plan.input_count,
                  "burst_limit_sec": plan.burst_limit_sec, "burst_elapsed_sec": elapsed,
                  "actions": [asdict(r) for r in receipts], "error": error,
                  "attempted_placements": attempted,
                  "completed": False, "verified": False,
                  "input_sent": error is None and len(receipts) == len(plan.actions) and elapsed <= plan.burst_limit_sec,
                  "burst_pass": error is None and len(receipts) == len(plan.actions) and elapsed <= plan.burst_limit_sec,
                  "evidence": [plan.frame.frame_id]}
    session.prepared_battle_receipt = deployment
    try:
        post = session.observe("prepared-battle-post")
    except Exception as exc:
        error = error or f"Post-deployment observation failed: {exc}"
        deployment["error"] = error
    if post is not None:
        deployment["evidence"].append(str(post.screenshot_path))
    if post is not None:
        deployment.update(_verify_consumption(post, plan, receipts))
        deployment["completed"] = deployment["completed"] and deployment["input_sent"]
        deployment["verified"] = deployment["verified"] and deployment["input_sent"]
        post.observations["deployment"] = deployment
    else:
        deployment.update(completed=False, verified=False,
                          issued_placements=sum(r.issued_count for r in receipts))
    session.event("prepared_battle_result", plan_id=plan.plan_id, completed=deployment["completed"],
                  verified=deployment["verified"], burst_elapsed_sec=elapsed, error=error)
    session.prepared_battle_receipt = deployment
    if post is None:
        raise DeploymentError(error or "Post-deployment evidence unavailable", partial_receipt=deployment)
    return post


def run_prepared_battle(session: Any, scout: SceneSnapshot, *, model: Any,
                        decider: Any = None, guides: Sequence[dict] = ()) -> SceneSnapshot:
    session.prepared_battle_receipt = None
    plan = prepare_battle(session, scout, model=model, decider=decider, guides=guides)
    callback = getattr(session, "progress_callback", None)
    if callable(callback):
        callback("计划就绪")
    result = execute_prepared_battle(session, plan)
    if callable(callback) and result.observations["deployment"]["input_sent"]:
        callback("输入已发出")
    if callable(callback):
        if result.observations["deployment"]["verified"]:
            callback("消费已核验")
    return result
