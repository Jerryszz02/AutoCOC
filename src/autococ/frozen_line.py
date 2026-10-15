"""Model-free, observation-free edrag_line burst from independently verified inputs.

This module does not discover terrain, prepare the army, release a model, or
enter battle. Callers must finish those observations before freezing a plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

from .errors import DeploymentError, FlowError
from .prepared_battle import FramePacket, PreparedInput
from .scene import SceneSnapshot
from .strategies import BattleContext, EdragLineStrategy


_CLIENT_VERSION = "18.647.2"
_COUNT = re.compile(r"[xX×]\s*(\d+)")


@dataclass(frozen=True)
class FrozenLinePlan:
    plan_id: str
    battle_id: str
    frame: FramePacket
    frame_sha256: str
    toolbar_before_frame_id: str
    toolbar_before_sha256: str
    client_version: str
    frozen_at: float
    latest_start: float
    burst_limit_sec: float
    toolbar_sha256: str
    terrain_sha256: str
    actions: tuple[PreparedInput, ...]
    troop_budgets: tuple[tuple[str, str, int], ...]
    troop_cards: tuple[tuple[str, str, int, tuple[int, int, int, int], tuple[int, int, int, int]], ...]
    hero_ids: tuple[str, ...]
    hero_cards: tuple[tuple[str, tuple[int, int, int, int]], ...]
    hero_manifest_sha256: str | None
    hero_manifest_frame_id: str | None
    hero_manifest_frame_sha256: str | None
    minimum_zoom_sha256: str
    plan_sha256: str


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             default=str).encode("utf-8")).hexdigest()


def _file_sha256(path: Any) -> str:
    digest = sha256()
    try:
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FlowError(f"Frozen line source frame is unavailable: {path}") from exc
    return digest.hexdigest()


def _plan_digest(plan: FrozenLinePlan) -> str:
    payload = asdict(plan)
    payload.pop("plan_sha256")
    return _digest(payload)


def _point(value: Any, width: int, height: int) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(item) is not int for item in value) or
            not (0 <= value[0] < width and 0 <= value[1] < height)):
        raise FlowError("Frozen line input point is outside the verified display")
    return tuple(value)


def _cards(snapshot: SceneSnapshot, *, client_version: str,
           expected: dict[tuple[str, str], int] | None = None) -> tuple[dict, ...]:
    battle = snapshot.observations.get("battle")
    slots = battle.get("slots") if isinstance(battle, dict) else None
    if not isinstance(slots, list) or any(not isinstance(item, dict) or item.get("kind") == "unknown"
                                               for item in slots):
        raise FlowError("Frozen line toolbar contains an unknown card")
    width, height = snapshot.observations.get("baseline_resolution", (1280, 720))
    if type(width) is not int or type(height) is not int:
        raise FlowError("Frozen line toolbar resolution is unknown")
    cards = []
    for slot in slots:
        if slot.get("kind") not in {"troop", "hero"}:
            continue
        kind, unit_id, source = slot["kind"], slot.get("unit_id"), slot.get("source")
        evidence = slot.get("evidence") if isinstance(slot.get("evidence"), dict) else {}
        identity = evidence.get("identity") if isinstance(evidence.get("identity"), dict) else {}
        confidence = slot.get("confidence")
        score = identity.get("confidence")
        bbox = slot.get("bbox")
        if (not isinstance(unit_id, str) or not unit_id or source != "army" or
                type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < .9 or
                type(score) not in (int, float) or not math.isfinite(score) or score < .9 or
                identity.get("unit_id") != unit_id or identity.get("version_verified") is not True or
                identity.get("template_client_version") != client_version or
                not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
                any(type(value) is not int for value in bbox) or
                not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height)):
            raise FlowError("Frozen line card lacks current-version identity or geometry")
        point = _point(slot.get("point"), width, height)
        count = slot.get("count")
        if kind == "troop":
            reading = evidence.get("count") if isinstance(evidence.get("count"), dict) else {}
            anchor = evidence.get("count_roi_anchor") if isinstance(evidence.get("count_roi_anchor"), dict) else None
            read_bbox = tuple(anchor.get("source_slot_bbox") or ()) if anchor else tuple(bbox)
            if (anchor is not None and (anchor.get("verified_slot_bbox") != list(bbox) or
                                        anchor.get("verified_frame") != str(snapshot.screenshot_path) or
                                        anchor.get("source_frame") != evidence.get("identity_reference") or
                                        anchor.get("unit_id") != unit_id)):
                raise FlowError("Frozen line selected card count anchor is unbound")
            if not _count_proves(reading, count, snapshot.screenshot_path, read_bbox, unit_id, client_version):
                raise FlowError("Frozen line troop count lacks same-card OCR evidence")
            if expected is not None and expected.get((unit_id, source)) != count:
                raise FlowError("Frozen line troop count differs from independent army manifest")
        cards.append({"kind": kind, "unit_id": unit_id, "source": source,
                      "count": count, "point": point, "bbox": tuple(bbox)})
    keys = [(row["kind"], row["unit_id"], row["source"]) for row in cards]
    if len(set(keys)) != len(keys):
        raise FlowError("Frozen line toolbar has duplicate card identities")
    return tuple(cards)


def _count_proves(reading: dict, expected: int, frame: Any,
                  bbox: tuple[int, ...], unit_id: str, client_version: str) -> bool:
    confidence = reading.get("confidence")
    if (type(expected) is not int or expected <= 0 or type(confidence) not in (int, float) or
            not math.isfinite(confidence) or confidence < .9):
        return False
    if "readings" not in reading:
        match = _COUNT.fullmatch(str(reading.get("text", "")).strip())
        return match is not None and int(match[1]) == expected
    if (reading.get("count") != expected or reading.get("frame") != str(frame) or
            tuple(reading.get("slot_bbox") or ()) != bbox):
        return False
    score = reading.get("portrait_score")
    source = reading.get("portrait_source")
    if (type(score) not in (int, float) or not math.isfinite(score) or score < .92 or
            source not in {"current_frame_saved_battle_portrait", "current_frame_versioned_battle_template"}):
        return False
    if source == "current_frame_versioned_battle_template":
        identity = reading.get("portrait_identity") or {}
        if (identity.get("unit_id") != unit_id or identity.get("version_verified") is not True or
                identity.get("template_client_version") != client_version):
            return False
    reads = reading.get("readings")
    if not isinstance(reads, list) or not reads:
        return False
    counts = set()
    for item in reads:
        if not isinstance(item, dict):
            return False
        score = item.get("confidence")
        if type(score) not in (int, float) or not math.isfinite(score):
            return False
        if score >= .9:
            match = _COUNT.fullmatch(str(item.get("text", "")).strip())
            if match is not None:
                counts.add(int(match[1]))
    return counts == {expected}


def _post_troop_reading(session: Any, post_path: Any, source_path: str,
                        point: tuple[int, int], bbox: tuple[int, int, int, int]) -> dict:
    """Use a frozen card portrait to bind current-frame count OCR to its identity."""
    import cv2
    from .battle_vision import BattleObserver
    from .images import read_frame

    source = cv2.resize(read_frame(source_path), (1280, 720), interpolation=cv2.INTER_AREA)
    current = cv2.resize(read_frame(post_path), (1280, 720), interpolation=cv2.INTER_AREA)
    left, top, right, bottom = bbox
    patch = source[top + 38:min(top + 78, bottom), left + 20:right - 15]
    if patch.size == 0:
        return {}
    observer = BattleObserver(session.recognizer)
    observer.cards[round(point[0])] = (bbox, cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).copy())
    return observer.count(post_path, current, {"point": list(point), "bbox": list(bbox)}) or {}


def freeze_model_free_line(session: Any, selected: SceneSnapshot, *,
                           toolbar_before: SceneSnapshot, terrain: Sequence[dict],
                           army_manifest: dict, minimum_zoom: dict,
                           battle_id: str, latest_start: float, burst_limit_sec: float,
                           hero_ids: Sequence[str] = (), hero_manifest: dict | None = None,
                           clock: Callable[[], float] = time.monotonic) -> FrozenLinePlan:
    """Freeze a one-shot four-troop line; all discovery stays before this call."""
    if getattr(getattr(session.config, "vision_agent", None), "enabled", False):
        raise FlowError("Model-free line cannot replace vision-mode model release")
    if getattr(session, "client_version", None) != _CLIENT_VERSION or not isinstance(battle_id, str) or not battle_id:
        raise FlowError("Frozen line requires a current-version battle identity")
    if selected is not session.last_snapshot:
        raise FlowError("Frozen line must use the last observed selected frame")
    frame = FramePacket.from_snapshot(selected, session)
    before = FramePacket.from_snapshot(toolbar_before, session)
    now = clock()
    if (frame.scene not in {"enemy_village", "battle"} or not math.isfinite(frame.confidence) or frame.confidence < .8 or
            before.scene not in {"enemy_village", "battle"} or not math.isfinite(before.confidence) or before.confidence < .8 or
            before.frame_id == frame.frame_id or before.captured_at >= frame.captured_at or
            frame.captured_at - before.captured_at > 30 or
            before.display_id != frame.display_id or before.resolution != frame.resolution or
            before.source_resolution != frame.source_resolution or
            type(latest_start) not in (int, float) or not math.isfinite(latest_start) or
            not now < latest_start <= frame.captured_at + 10 or
            type(burst_limit_sec) not in (int, float) or not math.isfinite(burst_limit_sec) or
            not 0 < burst_limit_sec <= 20):
        raise FlowError("Frozen line frame/display/time or burst budget is invalid")
    motion = minimum_zoom.get("motion") if isinstance(minimum_zoom, dict) else None
    if (not isinstance(minimum_zoom, dict) or minimum_zoom.get("verified") is not True or
            minimum_zoom.get("frame") != before.frame_id or not isinstance(motion, dict) or
            motion.get("stationary") is not True or type(motion.get("matched_points")) is not int or
            motion["matched_points"] < 40 or type(motion.get("inliers")) is not int or
            motion["inliers"] < 40 or motion["inliers"] / motion["matched_points"] < .8 or
            not isinstance(motion.get("affine"), list) or len(motion["affine"]) != 2 or
            any(not isinstance(row, list) or len(row) != 3 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in row)
                for row in motion["affine"])):
        raise FlowError("Frozen line minimum-zoom evidence is unverified")
    affine = motion["affine"]
    if (abs(affine[0][0] - 1) > .02 or abs(affine[0][1]) > .02 or
            math.hypot(affine[0][2], affine[1][2]) >= 8):
        raise FlowError("Frozen line minimum-zoom camera was moving")
    manifest = army_manifest
    troops = manifest.get("troops") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get("complete") is not True or
            manifest.get("identity_complete") is not True or manifest.get("unknowns") or
            not isinstance(troops, list) or len(troops) != 4 or
            any(not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) or
                row.get("source") != "army" or type(row.get("count")) is not int or row["count"] <= 0
                for row in troops)):
        raise FlowError("Frozen line requires an independently complete four-troop army manifest")
    expected = {(row["unit_id"], row["source"]): row["count"] for row in troops}
    if len(expected) != 4:
        raise FlowError("Frozen line army manifest repeats a troop identity")
    current_cards = _cards(selected, client_version=_CLIENT_VERSION, expected=expected)
    previous_cards = _cards(toolbar_before, client_version=_CLIENT_VERSION, expected=expected)
    current_troops = sorted((card for card in current_cards if card["kind"] == "troop"),
                            key=lambda card: card["point"][0])
    if len(current_troops) != 4 or {(card["unit_id"], card["source"]) for card in current_troops} != set(expected):
        raise FlowError("Frozen line toolbar does not contain exactly the manifest troops")
    # Stable card positions are established by two real, distinct source frames.
    previous = {(card["kind"], card["unit_id"], card["source"]): card for card in previous_cards}
    for card in current_troops:
        prior = previous.get((card["kind"], card["unit_id"], card["source"]))
        if (prior is None or prior["count"] != card["count"] or
                any(abs(prior["point"][axis] - card["point"][axis]) > 4 for axis in (0, 1))):
            raise FlowError("Frozen line toolbar geometry/count changed across source frames")
    if len([card for card in previous_cards if card["kind"] == "troop"]) != 4:
        raise FlowError("Frozen line previous toolbar lacks four manifest troops")
    from .battle_vision import selected_card_bbox
    if selected_card_bbox(selected.screenshot_path, current_troops[0]["bbox"], frame.resolution) is None:
        raise FlowError("Frozen line first troop is not selected in the current frame")
    if not isinstance(terrain, (list, tuple)) or len(terrain) != 4:
        raise FlowError("Frozen line needs four current red-boundary edge anchors")
    for edge in (0, 1):
        anchors = [row for row in terrain if isinstance(row, dict) and row.get("edge") == edge]
        if len(anchors) != 2 or anchors[0].get("point") == anchors[1].get("point"):
            raise FlowError("Frozen line lacks two distinct anchors per edge")
    for row in terrain:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        if (not isinstance(evidence, dict) or evidence.get("method") != "current_red_boundary_offset" or
                evidence.get("frame") != frame.frame_id or evidence.get("horizontal_clearance") != 32 or
                any(type(evidence.get(name)) not in (int, float) or
                    not math.isfinite(evidence[name]) for name in ("slope", "intercept")) or
                not .68 <= abs(evidence["slope"]) <= .82):
            raise FlowError("Frozen line terrain is not bound to the selected source frame")
        _point(row.get("point"), *frame.resolution)
    selected_heroes = []
    if len(set(hero_ids)) != len(hero_ids) or any(not isinstance(item, str) or not item for item in hero_ids):
        raise FlowError("Frozen line hero identities are ambiguous")
    hero_frame_hash = None
    if hero_ids or hero_manifest is not None:
        rows = hero_manifest.get("cards") if isinstance(hero_manifest, dict) else None
        if (not isinstance(hero_manifest, dict) or hero_manifest.get("complete") is not True or
                hero_manifest.get("unknowns") or not isinstance(rows, list) or
                {row.get("unit_id") for row in rows if isinstance(row, dict)} != set(hero_ids) or
                len(rows) != len(hero_ids) or not isinstance(hero_manifest.get("frame"), str) or
                Path(hero_manifest["frame"]).parent != Path(toolbar_before.screenshot_path).parent):
            raise FlowError("Frozen line heroes differ from independent complete hero manifest")
        hero_frame_hash = _file_sha256(hero_manifest["frame"])
        for row in rows:
            identity = (row.get("evidence") or {}).get("identity") or {}
            score = identity.get("confidence")
            row_score = row.get("confidence")
            if (row.get("kind") != "hero" or row.get("source") != "army" or
                    type(row_score) not in (int, float) or not math.isfinite(row_score) or row_score < .9 or
                    type(score) not in (int, float) or
                    not math.isfinite(score) or score < .9 or
                    identity.get("unit_id") != row.get("unit_id") or
                    identity.get("version_verified") is not True or
                    identity.get("template_client_version") != _CLIENT_VERSION):
                raise FlowError("Frozen line independent hero identity is unverified")
    # The selected fast path can have only refreshed troop slots. Check heroes
    # directly on its current pixels at the independently located prior cards.
    if hero_ids:
        import cv2
        from .images import read_frame
        from .unit_catalog import recognize_card_identity
        image = cv2.resize(read_frame(selected.screenshot_path), (1280, 720), interpolation=cv2.INTER_AREA)
    for hero_id in hero_ids:
        prior = previous.get(("hero", hero_id, "army"))
        if prior is None:
            raise FlowError("Frozen line hero lacks current-version card identity")
        identity = recognize_card_identity(image, prior["bbox"], "hero", surface="battle",
                                           client_version=_CLIENT_VERSION)
        score = identity.get("confidence")
        if (identity.get("unit_id") != hero_id or identity.get("version_verified") is not True or
                type(score) not in (int, float) or not math.isfinite(score) or score < .94):
            raise FlowError("Frozen line hero lacks current-frame identity")
        selected_heroes.append(prior)
    placement = EdragLineStrategy().build_plan(BattleContext(tuple(current_troops),
                                                              tuple(selected_heroes), tuple(terrain)))
    actions = []
    for index, step in enumerate(placement.troops):
        card = step.card
        if index:
            actions.append(PreparedInput(f"select-troop-{index}", "select", card["point"],
                                         card["unit_id"], card["source"]))
        for offset, point in enumerate(step.points):
            actions.append(PreparedInput(f"troop-{index}-{offset}", "deploy", _point(point, *frame.resolution),
                                         card["unit_id"], card["source"], .12))
    for index, step in enumerate(placement.heroes):
        card = step.card
        actions.append(PreparedInput(f"select-hero-{index}", "select", card["point"],
                                     card["unit_id"], card["source"]))
        actions.append(PreparedInput(f"hero-{index}", "deploy", _point(step.points[0], *frame.resolution),
                                     card["unit_id"], card["source"], .12))
    budgets = tuple((card["unit_id"], card["source"], card["count"]) for card in current_troops)
    troop_cards = tuple((card["unit_id"], card["source"], card["count"],
                         card["point"], card["bbox"]) for card in current_troops)
    hero_cards = tuple((card["unit_id"], card["bbox"]) for card in selected_heroes)
    # Use the same complete-batch speed gate as the model-driven path. A
    # caller's larger timeout must not silently relax the acceptance budget.
    burst_limit_sec = min(burst_limit_sec, 15., 1. + .25 * len(actions))
    if sum(action.delay_after_sec for action in actions) >= burst_limit_sec:
        raise FlowError("Frozen line fixed delays exceed the complete-batch budget")
    plan = FrozenLinePlan(uuid4().hex, battle_id, frame,
                          _file_sha256(frame.screenshot_path), before.frame_id,
                          _file_sha256(before.screenshot_path),
                          _CLIENT_VERSION, now,
                          latest_start, burst_limit_sec, _digest(current_cards), _digest(terrain),
                          tuple(actions), budgets, troop_cards, tuple(hero_ids), hero_cards,
                          _digest(hero_manifest) if hero_manifest is not None else None,
                          hero_manifest["frame"] if hero_manifest is not None else None, hero_frame_hash,
                          _digest(minimum_zoom), "")
    return FrozenLinePlan(**{**asdict(plan), "frame": frame, "actions": plan.actions,
                             "troop_budgets": plan.troop_budgets, "troop_cards": plan.troop_cards,
                             "hero_ids": plan.hero_ids, "hero_cards": plan.hero_cards,
                             "plan_sha256": _plan_digest(plan)})


def execute_frozen_line(session: Any, plan: FrozenLinePlan, *,
                        clock: Callable[[], float] = time.monotonic,
                        sleep: Callable[[float], None] = time.sleep) -> SceneSnapshot:
    """One attempt: no observation/recognition during input; no retry or slow fallback."""
    if getattr(getattr(session.config, "vision_agent", None), "enabled", False):
        raise FlowError("Model-free line cannot replace vision-mode model release")
    if not isinstance(plan, FrozenLinePlan) or _plan_digest(plan) != plan.plan_sha256:
        raise FlowError("Frozen line plan digest does not match")
    if (session.last_snapshot is None or str(session.last_snapshot.screenshot_path) != plan.frame.frame_id or
            session.client_version != plan.client_version or
            session.capture.input_display_id != plan.frame.display_id or
            tuple(session.config.game.baseline_resolution) != plan.frame.resolution or
            tuple(session.context.screen_resolution) != plan.frame.source_resolution or
            clock() >= plan.latest_start):
        raise FlowError("Frozen line permission expired or display/frame changed")
    if _file_sha256(plan.frame.screenshot_path) != plan.frame_sha256:
        raise FlowError("Frozen line source pixels changed after plan freeze")
    if _file_sha256(plan.toolbar_before_frame_id) != plan.toolbar_before_sha256:
        raise FlowError("Frozen line toolbar evidence changed after plan freeze")
    if (plan.hero_manifest_frame_id is not None and
            _file_sha256(plan.hero_manifest_frame_id) != plan.hero_manifest_frame_sha256):
        raise FlowError("Frozen line independent hero manifest frame changed after plan freeze")
    used = getattr(session, "_used_frozen_frames", None)
    if used is None:
        used = set()
        session._used_frozen_frames = used
    if plan.frame.frame_id in used:
        raise FlowError("Frozen line source frame was already attempted")
    used.add(plan.frame.frame_id)  # consumed before the first uncertain transport call
    start = clock()
    deadline = min(start + plan.burst_limit_sec, session.deadline, session.task_deadline)
    receipts = []
    receipt = {"mode": "model_free_line_continuous", "plan_id": plan.plan_id,
               "plan_sha256": plan.plan_sha256, "battle_id": plan.battle_id,
               "frame": plan.frame.frame_id, "input_count": len(plan.actions),
               "scope": "troops_and_heroes" if plan.hero_ids else "troops_only",
               "excluded_kinds": ["spell", "siege"] + ([] if plan.hero_ids else ["hero"]),
               "actions": receipts, "completed": False, "verified": False,
               "input_sent": False, "issued_placements": 0,
               "attempted_placements": 0, "deployed_units": 0,
               "burst_limit_sec": plan.burst_limit_sec, "burst_pass": False,
               "consumption_events": [],
               "evidence": [plan.frame.frame_id], "error": None}
    session.frozen_line_receipt = receipt
    try:
        for action in plan.actions:
            session.check_deadline()
            remaining = deadline - clock()
            if remaining <= 0:
                raise FlowError("Frozen line burst deadline exceeded")
            if action.kind == "deploy":
                receipt["attempted_placements"] += 1
            session._active_frozen_plan = (plan.plan_id, deadline, action.point)
            try:
                session.frozen_battle_tap(action.point,
                                          timeout_sec=min(remaining, session.config.runtime.step_timeout_sec),
                                          reason=f"model-free line {plan.plan_id} {action.action_id}",
                                          plan_id=plan.plan_id)
            except BaseException as exc:
                receipts.append({"action_id": action.action_id, "status": "uncertain", "reason": str(exc)})
                raise
            finally:
                session._active_frozen_plan = None
            receipts.append({"action_id": action.action_id, "status": "input_sent",
                             "done_at": clock(), "kind": action.kind,
                             "unit_id": action.unit_id, "point": action.point})
            receipt["issued_placements"] += int(action.kind == "deploy")
            if clock() >= deadline:
                raise FlowError("Frozen line deadline exceeded after input")
            if action.delay_after_sec:
                if action.delay_after_sec >= deadline - clock():
                    raise FlowError("Frozen line fixed delay exceeds deadline")
                sleep(action.delay_after_sec)
    except BaseException as exc:
        receipt["error"] = str(exc)
        session._active_frozen_plan = None
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise DeploymentError(str(exc), partial_receipt=receipt) from exc
    finally:
        session._active_frozen_plan = None
    receipt["burst_elapsed_sec"] = clock() - start
    receipt["input_sent"] = len(receipts) == len(plan.actions) and receipt["burst_elapsed_sec"] <= plan.burst_limit_sec
    receipt["burst_pass"] = receipt["input_sent"]
    try:
        post = session.observe("frozen-line-post")
    except Exception as exc:
        receipt["error"] = f"Post-burst observation failed: {exc}"
        raise DeploymentError(receipt["error"], partial_receipt=receipt) from exc
    receipt["evidence"].append(str(post.screenshot_path))
    observed_at = post.observations.get("observed_at_monotonic")
    post_source = post.observations.get("source_resolution")
    post_baseline = post.observations.get("baseline_resolution")
    if (post.scene not in {"enemy_village", "battle"} or not math.isfinite(post.confidence) or post.confidence < .8 or
            type(observed_at) not in (int, float) or not math.isfinite(observed_at) or
            observed_at < receipts[-1]["done_at"] or
            not isinstance(post_source, (list, tuple)) or tuple(post_source) != plan.frame.source_resolution or
            not isinstance(post_baseline, (list, tuple)) or tuple(post_baseline) != plan.frame.resolution):
        receipt["error"] = "Fresh post-burst battle frame unavailable"
        raise DeploymentError(receipt["error"], partial_receipt=receipt)
    battle = post.observations.get("battle")
    slots = battle.get("slots") if isinstance(battle, dict) else []
    if not isinstance(slots, list):
        slots = []
    troop_results = []
    for unit_id, source, initial, point, bbox in plan.troop_cards:
        # Full-frame recognition intentionally omits gray spent cards. Revisit the
        # frozen original box with its pre-burst portrait, then read this frame's x0.
        try:
            reading = _post_troop_reading(session, post.screenshot_path, plan.toolbar_before_frame_id,
                                          point, bbox)
        except Exception as exc:
            reading = {"error": str(exc)}
        conflicting = [row for row in slots if isinstance(row, dict) and
                       (row.get("unit_id") == unit_id or row.get("bbox") == list(bbox)) and
                       type(row.get("count")) is int and row["count"] > 0]
        score = reading.get("confidence")
        portrait = reading.get("portrait_score")
        reads = reading.get("readings")
        explicit = [item for item in reads if isinstance(item, dict) and
                    type(item.get("confidence")) in (int, float) and item["confidence"] >= .9 and
                    (match := _COUNT.fullmatch(str(item.get("text", "")).strip())) is not None and int(match[1]) == 0] if isinstance(reads, list) else []
        valid = (reading.get("count") == 0 and reading.get("frame") == str(post.screenshot_path) and
                 reading.get("slot_bbox") == list(bbox) and not conflicting and
                 type(score) in (int, float) and math.isfinite(score) and score >= .9 and
                 type(portrait) in (int, float) and math.isfinite(portrait) and portrait >= .92 and
                 reading.get("portrait_source") == "current_frame_saved_battle_portrait" and
                 bool(explicit))
        troop_results.append({"unit_id": unit_id, "initial": initial, "post_count": 0 if valid else None,
                              "verified": valid, "reading": reading})
    receipt["troops"] = troop_results
    receipt["consumption_events"] = [
        {"unit_id": item["unit_id"], "before_count": item["initial"],
         "after_count": item["post_count"], "after_frame": str(post.screenshot_path),
         "verified": item["verified"]} for item in troop_results]
    hero_results = []
    if plan.hero_ids:
        from .hero_state import recognize_hero_state
        for hero_id, bbox in plan.hero_cards:
            state = recognize_hero_state(post.screenshot_path, bbox,
                                         baseline_resolution=plan.frame.resolution,
                                         unit_id=hero_id, client_version=plan.client_version)
            identity = (state.get("evidence") or {}).get("portrait_identity") or {}
            verified = (state.get("deployed") is True and state.get("frame") == str(post.screenshot_path) and
                        identity.get("unit_id") == hero_id and identity.get("version_verified") is True and
                        identity.get("template_client_version") == plan.client_version)
            hero_results.append({"unit_id": hero_id, "verified": verified, "state": state})
    receipt["heroes"] = hero_results
    receipt["verified"] = receipt["completed"] = (receipt["input_sent"] and
        all(item["verified"] for item in troop_results + hero_results))
    if receipt["verified"]:
        receipt["deployed_units"] = sum(initial for _unit_id, _source, initial in plan.troop_budgets)
    post.observations["deployment"] = receipt
    if not receipt["completed"]:
        receipt["error"] = "Frozen line troop or hero consumption is unverified"
        raise DeploymentError(receipt["error"], partial_receipt=receipt)
    return post
