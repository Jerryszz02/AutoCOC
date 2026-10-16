"""Deploy a visible army from a verified map edge and retain consumption evidence."""
from __future__ import annotations

import math
import re
import time

from .errors import DeploymentError, FlowError
from .scene import SceneSnapshot
from .session import GameSession
from .vision import ScreenshotRecognizer
from .strategies import BattleContext, DeploymentPlan, STRATEGIES, is_line_strategy, spread_on_edge


def _battle_frame(snapshot: SceneSnapshot) -> None:
    if snapshot.scene not in {"enemy_village", "battle"} or not math.isfinite(snapshot.confidence) or snapshot.confidence < 0.8:
        raise FlowError(f"Deployment requires a recognized battle view, got {snapshot.scene}")


def _trusted_count_roi_anchor(slot: dict) -> tuple[int, int, int, int] | None:
    """Use a prior glyph box only after this same card was reidentified selected."""
    evidence = slot.get("evidence") or {}
    anchor, current = evidence.get("count_roi_anchor"), evidence.get("count") or {}
    if not isinstance(anchor, dict) or not isinstance(current, dict):
        return None
    count, unit_id = slot.get("count"), slot.get("unit_id")
    source_text = anchor.get("source_text")
    match = re.fullmatch(r"[xX×]\s*([0-9]+)", source_text.strip()) if isinstance(source_text, str) else None
    def valid_score(value, minimum):
        return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= 1
    if (type(count) is not int or count <= 0 or not isinstance(unit_id, str) or not unit_id
            or match is None or int(match[1]) != count
            or anchor.get("unit_id") != unit_id or anchor.get("verified_count") != count
            or current.get("count") != count
            or not valid_score(anchor.get("source_confidence"), .9)
            or not valid_score(anchor.get("verified_portrait_score"), .92)
            or not valid_score(current.get("portrait_score"), .92)
            or anchor.get("verified_slot_bbox") != list(slot["bbox"])):
        return None
    source_frame, verified_frame = anchor.get("source_frame"), anchor.get("verified_frame")
    source_box, glyph_box = anchor.get("source_slot_bbox"), anchor.get("bbox")
    if (not isinstance(source_frame, str) or not source_frame
            or not isinstance(verified_frame, str) or not verified_frame or source_frame == verified_frame
            or current.get("frame") != verified_frame or current.get("slot_bbox") != source_box
            or not isinstance(source_box, list) or len(source_box) != 4
            or not isinstance(glyph_box, list) or len(glyph_box) != 4
            or any(type(value) is not int for value in (*source_box, *glyph_box))):
        return None
    left, top, right, bottom = glyph_box
    if not left < right or not top < bottom:
        return None
    source_left, source_top, source_right, _ = source_box
    if not (source_left - 4 <= left < right <= source_right + 4
            and source_top - 16 <= top < bottom <= source_top + 40):
        return None
    current_left, current_top, current_right, _ = slot["bbox"]
    if not (current_left <= left < right <= current_right
            and current_top - 16 <= top < bottom <= current_top + 40):
        return None
    return tuple(glyph_box)


def _remaining(snapshot: SceneSnapshot, slot: dict, *, recognizer: ScreenshotRecognizer | None = None) -> int | None:
    """Read even an exhausted gray card from its original, still visible header."""
    left, top, right, bottom = slot["bbox"]
    matches = []
    for item in snapshot.observations.get("ocr", []):
        box = item.get("bbox")
        match = re.fullmatch(r"[xX×]\s*(\d+)", item["text"].strip())
        if box and match and math.isfinite(item["confidence"]) and item["confidence"] >= 0.9:
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if left - 4 <= x <= right + 4 and top - 16 <= y <= top + 40:
                matches.append(int(match[1]))
    if matches:
        # Contradictory/duplicate full-frame detections are not resolved by
        # choosing a more convenient local reading.
        return matches[0] if len(matches) == 1 else None
    if recognizer is None:
        return None
    reads = snapshot.observations.setdefault("deployment_count_reads", [])
    reading = next((item for item in reads if item["slot_bbox"] == list(slot["bbox"])), None)
    if reading is None:
        evidence = slot.get("evidence", {}).get("count") or {}
        count_bbox = evidence.get("bbox")
        confidence = evidence.get("confidence", 0.0)
        if (count_bbox is not None and math.isfinite(confidence) and .9 <= confidence <= 1
                and re.fullmatch(r"[xX×]\s*[0-9]+", evidence.get("text", "").strip())):
            reading = recognizer.recognize_slot_count(snapshot.screenshot_path, tuple(slot["bbox"]), count_bbox=tuple(count_bbox))
        else:
            anchor = _trusted_count_roi_anchor(slot)
            reading = recognizer.recognize_slot_count(snapshot.screenshot_path, tuple(slot["bbox"]),
                                                       **({"count_bbox": anchor} if anchor is not None else {}))
        reads.append(reading)
    count, confidence = reading.get("count"), reading.get("confidence", 0.0)
    if (type(count) is int and count >= 0 and math.isfinite(confidence) and .9 <= confidence <= 1
            and reading.get("frame") == str(snapshot.screenshot_path)
            and reading.get("slot_bbox") == list(slot["bbox"])):
        return count
    return None


def _verify_visible_quantities(snapshot: SceneSnapshot, slots: list[dict]) -> None:
    """A missed card must not silently remove a visible positive stack from the plan."""
    width, height = snapshot.observations.get("baseline_resolution", (1280, 720))
    for item in snapshot.observations.get("ocr", []):
        box = item.get("bbox")
        match = re.fullmatch(r"[xX×]\s*([0-9]+)", item["text"].strip())
        confidence = item.get("confidence", 0)
        if not box or not match or int(match[1]) == 0 or not math.isfinite(confidence) or not .9 <= confidence <= 1:
            continue
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not 20 * width / 1280 <= x <= 1260 * width / 1280 or not 580 * height / 720 <= y <= 630 * height / 720:
            continue
        matches = [slot for slot in slots if slot["kind"] in {"troop", "spell"}
                   and slot["bbox"][0] - 4 <= x <= slot["bbox"][2] + 4
                   and slot["bbox"][1] - 16 <= y <= slot["bbox"][1] + 40]
        if (len(matches) != 1 or type(matches[0].get("count")) is not int
                or matches[0]["count"] != int(match[1])):
            raise FlowError(f"Visible army quantity {item['text']} has no unique numeric card: {box}")


def deploy_army(session: GameSession, scout: SceneSnapshot) -> SceneSnapshot:
    receipt: dict[str, object] = {
        "completed": False, "verified": False, "deployed_units": 0,
        "evidence": [str(scout.screenshot_path)], "consumption_events": [],
        "terrain_evidence": [], "support_attempts": [], "count_reads": [],
    }
    try:
        return _deploy_army(session, scout, receipt)
    except Exception as exc:
        # KeyboardInterrupt deliberately propagates to the runner unchanged.
        if isinstance(exc, DeploymentError):
            frames = receipt["evidence"]
            receipt.update(exc.partial_receipt)
            receipt["evidence"] = list(dict.fromkeys(frames + receipt.get("evidence", [])))
        last = session.last_snapshot
        if last is not None and str(last.screenshot_path) not in receipt["evidence"]:
            receipt["evidence"].append(str(last.screenshot_path))
        raise DeploymentError(str(exc), partial_receipt=receipt) from exc


def _deploy_army(session: GameSession, scout: SceneSnapshot, receipt: dict[str, object]) -> SceneSnapshot:
    from .terrain import find_clear_ground_probes, find_west_deployment_points, measure_camera_motion

    strategy = getattr(session.config.battle, "strategy", "verified")
    line_mode = is_line_strategy(strategy)
    receipt["strategy"] = strategy
    _battle_frame(scout)
    session._validate_snapshot(scout)
    manifest = scout.observations.get("expected_army_manifest")
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise FlowError("Deployment has no independent prebattle army manifest")
    expected = []
    for group, kind in (("troops", "troop"), ("spells", "spell")):
        cards = manifest.get(group)
        if (not isinstance(cards, list) or any(not isinstance(card, dict)
                or type(card.get("count")) is not int or card["count"] <= 0 for card in cards)):
            raise FlowError("Prebattle army manifest contains an unreadable quantity")
        if not line_mode or kind == "troop":
            expected.extend((kind, card["count"]) for card in cards)
    if not manifest["troops"]:
        raise FlowError("Prebattle army manifest has no troops")
    receipt["expected_army_manifest"] = manifest
    deadline = time.monotonic() + session.config.battle.deploy_timeout_sec
    frames: list[str] = [str(scout.screenshot_path)]
    events: list[dict] = []
    support_attempts: list[dict] = []
    count_reads: list[dict] = []
    receipt.update(evidence=frames, consumption_events=events, support_attempts=support_attempts, count_reads=count_reads)
    current = scout

    def check_deadline() -> None:
        session.check_deadline()
        if time.monotonic() >= deadline:
            raise FlowError("Army deployment deadline exceeded")

    def tap(point: list[int], *, reason: str) -> None:
        check_deadline()
        _battle_frame(current)
        session.tap(current, point, reason=reason)

    def observe(label: str, *, allow_settlement: bool = False) -> SceneSnapshot:
        nonlocal current
        check_deadline()
        current = session.observe(label)
        frames.append(str(current.screenshot_path))
        check_deadline()
        if not (allow_settlement and current.scene == "settlement"
                and math.isfinite(current.confidence) and current.confidence >= .8):
            _battle_frame(current)
        return current

    def visible_card(slot: dict) -> dict:
        for attempt in range(3):
            matches = [item for item in current.observations.get("battle", {}).get("slots", [])
                       if item["kind"] == slot["kind"]
                       and all(abs(item["point"][axis] - slot["point"][axis]) < 25 for axis in (0, 1))]
            if len(matches) == 1:
                return matches[0]
            if matches or attempt == 2:
                break
            # Selection/scene transitions can briefly merge card borders with
            # the map. Reacquire identity without sending any further input.
            observe("deploy-card-reacquire")
        raise FlowError("Selected army card position is no longer verified")

    def remaining(snapshot: SceneSnapshot, slot: dict) -> int | None:
        check_deadline()
        # Selection resizes a card. A consumption read already verified in this
        # exact frame must not be cropped again using its older, narrower box.
        confirmed = [event for event in events if event["after_frame"] == str(snapshot.screenshot_path)
                     and event["unit_kind"] == slot["kind"]
                     and all(abs(event["slot"][axis] - slot["point"][axis]) < 25 for axis in (0, 1))]
        if len(confirmed) == 1:
            return confirmed[0]["after_count"]
        count = _remaining(snapshot, slot, recognizer=getattr(session, "recognizer", None))
        for reading in snapshot.observations.get("deployment_count_reads", []):
            if reading not in count_reads:
                count_reads.append(reading)
                session.event("deployment_count_read", **reading)
        check_deadline()
        return count

    line_context = {}
    continuous_line = getattr(session.config.battle, "line_execution_mode", "verified") == "continuous"
    if line_mode:
        probed_boundary_scan = (continuous_line and
                                getattr(session.config.battle, "line_tactics", "single_edge") ==
                                "two_edge_probed_anchors")
        if probed_boundary_scan:
            # This phase contains zoom and card selection, but no field tap or
            # probe. These explicit zeroes are scoped to this phase only.
            receipt.update(phase="boundary_scan", attempted_placements=0,
                           issued_placements=0, offensive_actions=0)
        if continuous_line:
            if strategy != "edrag_line" or session.config.vision_agent.enabled:
                raise FlowError("Continuous line requires model-free edrag_line")
            current, terrain = _prepare_two_edge_view(session, current, receipt, capture_context=line_context)
        else:
            current, terrain = _prepare_two_edge_view(session, current, receipt)
        if probed_boundary_scan:
            receipt["phase"] = "boundary_verified"
            for field in ("attempted_placements", "issued_placements", "offensive_actions"):
                receipt.pop(field)
    else:
        terrain = find_west_deployment_points(current.screenshot_path,
                                              baseline_resolution=session.config.game.baseline_resolution)
    for duration in (600, 1200):
        if terrain:
            break
        check_deadline()
        ground = find_clear_ground_probes(current.screenshot_path,
                                         baseline_resolution=session.config.game.baseline_resolution, max_points=1)
        width, height = session.config.game.baseline_resolution
        start = [round(320 * width / 1280), round(350 * height / 720)]
        end = [round(1050 * width / 1280), start[1]]
        if ground:
            start = list(ground[0]["point"])
            end = [min(start[0] + round(400 * width / 1280), round(1050 * width / 1280)), start[1]]
            duration = 1200
        session._validate_snapshot(current)
        before_pan = current
        session.event("camera_pan", frame=str(current.screenshot_path), start=start, end=end,
                      duration_ms=duration, start_ground=ground[0] if ground else None)
        session.context.swipe(*start, *end, duration)
        session.action_count += 1
        # Input completion can precede the camera animation. Reobserve this pan
        # before giving up; do not pan again on the still-unmoved first frame.
        for attempt in range(3):
            if attempt:
                check_deadline()
                time.sleep(min(session.config.runtime.poll_interval_sec, 1, max(0, deadline - time.monotonic())))
            observe("deploy-west-edge")
            terrain = find_west_deployment_points(current.screenshot_path,
                                                  baseline_resolution=session.config.game.baseline_resolution)
            check_deadline()
            if terrain:
                break
        if terrain:
            break
        motion = measure_camera_motion(before_pan.screenshot_path, current.screenshot_path)
        check_deadline()
        pan = {"before_frame": str(before_pan.screenshot_path), "after_frame": str(current.screenshot_path),
               "duration_ms": duration, "motion": motion}
        receipt.setdefault("camera_pans", []).append(pan)
        session.event("camera_pan_motion", **pan)
        if motion["stationary"] is not True:
            break
    if not terrain and not line_mode:
        terrain = find_clear_ground_probes(current.screenshot_path,
                                          baseline_resolution=session.config.game.baseline_resolution)
        check_deadline()
        if terrain:
            session.event("ground_probe_candidates", frame=str(current.screenshot_path), candidates=terrain)
    if not terrain:
        raise FlowError("Two visible deployment edges could not be verified" if line_mode else
                        "No visible west deployment boundary or clear ground probe could be verified")
    receipt["terrain_evidence"] = terrain
    points = [item["point"] for item in terrain]
    slots = current.observations.get("battle", {}).get("slots", [])
    _verify_visible_quantities(current, slots)
    if not line_mode and any(slot["kind"] == "unknown" or slot["kind"] in {"troop", "spell"}
           and type(slot.get("count")) is not int for slot in slots):
        raise FlowError("An army card has no verified type or quantity")
    planned_cards = [slot for slot in slots if slot["kind"] in {"troop", "spell"}
                     and type(slot.get("count")) is int and slot["count"] > 0
                     and (not line_mode or slot["kind"] == "troop")]
    own_cards = []
    for slot in planned_cards:
        if slot.get("source") == "event":
            portrait = slot.get("evidence", {}).get("event_portrait") or {}
            if slot["kind"] != "troop" or not .95 <= portrait.get("confidence", 0) <= 1:
                raise FlowError("Additional event troop has no verified portrait")
        elif slot.get("source", "army") == "clan_reinforcement":
            badge = slot.get("evidence", {}).get("clan_badge") or {}
            confidence = badge.get("confidence", 0)
            if not math.isfinite(confidence) or not .95 <= confidence <= 1:
                raise FlowError("Additional clan card has no verified reinforcement badge")
        else:
            own_cards.append(slot)
    if sorted(expected) != sorted((slot["kind"], slot["count"]) for slot in own_cards):
        raise FlowError("Battle cards do not match the independent prebattle army manifest")
    receipt["army_manifest_matched"] = True
    receipt["planned_numeric_cards"] = [
        {"kind": slot["kind"], "source": slot.get("source", "army"),
         "bbox": list(slot["bbox"]), "before_count": slot["count"]}
        for slot in planned_cards]
    if line_mode:
        if continuous_line:
            return _deploy_continuous_line(session, current, receipt, line_context,
                                           scout.observations.get("expected_hero_manifest"), deadline)
        planner = STRATEGIES[strategy].planner
        plan = planner.build_plan(BattleContext(tuple(planned_cards),
            tuple(slot for slot in slots if slot["kind"] == "hero"), tuple(terrain)))
        return _deploy_line_cards(session, current, receipt, plan, deadline)
    troops = [slot for slot in slots if slot["kind"] == "troop" and type(slot.get("count")) is int and slot["count"] > 0]
    if not troops:
        raise FlowError("No positive troop count was recognized")
    troops.sort(key=lambda slot: slot["count"], reverse=True)
    deployed_units = 0
    verified_point = None

    def consume(slot: dict, point: list[int], count: int) -> int:
        nonlocal deployed_units
        slot = visible_card(slot)
        old = remaining(current, slot)
        if old is None or not 0 < count <= old:
            raise FlowError("Troop or spell quantity became unreadable before deployment")
        tap(slot["point"], reason=f"Select recognized {slot['kind']} card with {old} remaining")
        observe("deploy-selected")
        slot = visible_card(slot)
        before = current
        if remaining(before, slot) != old:
            raise FlowError("Army quantity changed before field placement")
        # Keep the camera fixed and bound each burst; screenshots take several
        # seconds, so capturing between individual placements would miss battle time.
        batch_deadline = time.monotonic() + 4
        issued = 0
        for _ in range(min(count, 8)):
            if time.monotonic() >= batch_deadline:
                break
            tap(point, reason=f"Deploy selected {slot['kind']} into verified battle area")
            issued += 1
            time.sleep(0.12)
        pending = None
        for _ in range(3):
            after = observe("deploy-consumption")
            after_count = remaining(after, slot)
            if after_count is not None and 0 <= after_count < old:
                if old - after_count > issued:
                    raise FlowError("Recognized consumption exceeds issued placements")
                if pending is not None and after_count > pending["after_count"]:
                    raise FlowError("Army quantity increased while verifying a deployment batch")
                event = {"unit_kind": slot["kind"], "slot": slot["point"], "point": point,
                         "before_count": old, "after_count": after_count,
                         "consumed": old - after_count, "before_frame": str(before.screenshot_path),
                         "after_frame": str(after.screenshot_path), "issued_placements": issued}
                already_verified = pending["consumed"] if pending is not None else 0
                if pending is None:
                    pending = event
                    events.append(pending)
                else:
                    pending.update(event)
                if slot["kind"] == "troop":
                    deployed_units += old - after_count - already_verified
                    receipt["deployed_units"] = deployed_units
                    receipt["verified"] = True
                # A first decrement can be an intermediate render while other
                # clicks in this same burst are still taking effect. Observe
                # the entire issued batch before selecting or placing again.
                if old - after_count == issued:
                    session.event("deployment_consumed", **pending)
                    return old - after_count
            time.sleep(min(session.config.runtime.poll_interval_sec, 1))
        if pending is not None:
            session.event("deployment_partial_batch", **pending)
            raise FlowError("Not all issued placements were verified before continuing")
        return 0

    # Prove one placement before committing the rest of the army to that edge.
    for point in points[:4]:
        consumed = consume(troops[0], point, 1)
        if consumed:
            verified_point = point
            break
    if verified_point is None:
        raise FlowError("No attempted edge placement consumed a troop")

    for slot in troops:
        for _ in range(30):
            count = remaining(current, slot)
            if count == 0:
                break
            if count is None:
                raise FlowError("Cannot verify remaining troop count")
            consumed = consume(slot, verified_point, min(count, 8))
            if not consumed:
                raise FlowError("Troop deployment stopped making progress")
        else:
            raise FlowError("Could not deploy the full visible troop card")

    # Hero/siege availability differs from numeric troop cards. Keep attempted
    # actions distinct from the verified numeric consumption receipt.
    for slot in slots:
        if slot["kind"] not in {"hero", "siege"}:
            continue
        # The previous consumption/state check already produced the current frame;
        # tap() still enforces its age and identity before issuing the selection.
        matches = [item for item in current.observations.get("battle", {}).get("slots", [])
                   if item["kind"] == slot["kind"] and abs(item["point"][0] - slot["point"][0]) < 25]
        if len(matches) != 1:
            continue
        if slot["kind"] == "hero" and (
            session.client_version not in {"18.600.7", "18.647.2"}
            or not isinstance(slot.get("unit_id"), str)
            or not slot["unit_id"]
            or matches[0].get("unit_id") != slot["unit_id"]
        ):
            support_attempts.append({"kind": "hero", "point": matches[0]["point"],
                                     "status": "identity_unverified", "verified": False,
                                     "selection_frame": str(current.screenshot_path)})
            continue
        attempt = {"kind": slot["kind"], "point": matches[0]["point"], "status": "attempted", "verified": False,
                   "selection_frame": str(current.screenshot_path), "placement_frame": None}
        support_attempts.append(attempt)
        tap(matches[0]["point"], reason=f"Select visible {slot['kind']} card")
        observe("deploy-support-selected")
        visible_card(matches[0])
        attempt["placement_frame"] = str(current.screenshot_path)
        tap(verified_point, reason=f"Place selected {slot['kind']} at troop deployment edge")
        if slot["kind"] == "hero":
            from .hero_state import recognize_hero_state

            states = []
            attempt["states"] = states

            def read_hero(label: str) -> dict:
                snapshot = observe(label)
                state = recognize_hero_state(snapshot.screenshot_path, slot["bbox"],
                                             baseline_resolution=session.config.game.baseline_resolution,
                                             unit_id=matches[0]["unit_id"], client_version=session.client_version)
                states.append(state)
                session.event("hero_state", **state)
                return state

            state = read_hero("deploy-hero-verify")
            if state["deployed"] is not True:
                state = read_hero("deploy-hero-verify")
            attempt.update(verified=state["deployed"] is True,
                           status="deployed" if state["deployed"] is True else "deployment_unverified")
            attempt["ability_verified"] = False
            if state["deployed"] is True and state["ability_ready"] is True:
                attempt["ability_before_frame"] = str(current.screenshot_path)
                tap(slot["point"], reason="Activate hero ability with verified ready-state evidence")
                attempt["ability_clicks"] = 1
                # An activation flash is not a consumed ability. Reobserve the
                # same action; never press again to resolve an uncertain result.
                for _ in range(3):
                    state = read_hero("hero-ability-verify")
                    if state["ability_used"] is True:
                        attempt["ability_verified"] = True
                        attempt["ability_after_frame"] = str(current.screenshot_path)
                        break
        else:
            from .hero_state import recognize_siege_state

            states = []
            attempt["states"] = states
            for _ in range(2):
                snapshot = observe("deploy-siege-verify")
                state = recognize_siege_state(snapshot.screenshot_path, slot["bbox"],
                                               baseline_resolution=session.config.game.baseline_resolution,
                                               client_version=session.client_version)
                states.append(state)
                session.event("siege_state", **state)
                if state["deployed"] is True:
                    attempt.update(verified=True, status="deployed")
                    break

    for slot in slots:
        if slot["kind"] != "spell":
            continue
        target = [min(1050, verified_point[0] + 260), verified_point[1]]
        for _ in range(30):
            count = remaining(current, slot)
            if count == 0:
                break
            if count is None:
                raise FlowError("Cannot verify remaining spell count")
            if not consume(slot, target, min(count, 8)):
                raise FlowError("Spell deployment stopped making progress")
        else:
            raise FlowError("Could not deploy the full visible spell card")

    last_battle_frame = current
    observe("deployment-complete", allow_settlement=True)
    quantity_frame = last_battle_frame if current.scene == "settlement" else current
    for slot in planned_cards:
        if remaining(quantity_frame, slot) != 0:
            raise FlowError("A planned troop or spell card is not verified empty")
    _verify_visible_quantities(quantity_frame, planned_cards)
    receipt["numeric_completion_frame"] = str(quantity_frame.screenshot_path)
    hero_attempts = [attempt for attempt in support_attempts if attempt["kind"] == "hero"]
    receipt["hero_abilities_verified"] = len(hero_attempts) == sum(slot["kind"] == "hero" for slot in slots) and all(
        attempt.get("verified") is True and attempt.get("ability_verified") is True
        for attempt in hero_attempts)
    receipt["numeric_completed"] = True
    receipt["support_deployment_verified"] = len(support_attempts) == sum(
        slot["kind"] in {"hero", "siege"} for slot in slots) and all(
            attempt.get("verified") is True for attempt in support_attempts)
    receipt["completed"] = receipt["support_deployment_verified"] and receipt["hero_abilities_verified"]
    current.observations["deployment"] = receipt
    return current


def _minimum_zoom_motion(motion: dict) -> bool:
    """Require near-identity map motion after a completed native zoom gesture."""
    affine = motion.get("affine")
    matches, inliers = motion.get("matched_points"), motion.get("inliers")
    return bool(
        motion.get("stationary") is True
        and type(matches) is int and matches >= 40
        and type(inliers) is int and inliers >= 40 and inliers / matches >= .8
        and isinstance(affine, list) and len(affine) == 2
        and all(isinstance(row, list) and len(row) == 3 for row in affine)
        and all(type(value) in (int, float) and math.isfinite(value)
                for row in affine for value in row)
        and abs(affine[0][0] - 1) <= .002
        and abs(affine[0][1]) <= .002
        and abs(affine[1][0]) <= .002
        and abs(affine[1][1] - 1) <= .002
        and math.hypot(affine[0][2], affine[1][2]) < 3
    )


def _prepare_two_edge_view(session: GameSession, current: SceneSnapshot, receipt: dict, *,
                           capture_context: dict | None = None) -> tuple[SceneSnapshot, list[dict]]:
    from .battle_vision import selected_card_bbox
    from .terrain import find_line_deployment_edges, measure_camera_motion

    if session.native is None:
        raise FlowError("Two-edge minimum zoom requires the MuMu native transport")
    if capture_context is not None:
        # The continuous batch must be frozen inside the preparation countdown.
        # Each native pinch takes about 1.7 s on the observed client. Observe
        # after each pinch so a camera already at minimum scale costs one pinch,
        # rather than spending the whole preview on four blind gestures.
        from .prepared_battle import _remaining_seconds

        if current.scene != "enemy_village":
            raise FlowError("Minimum zoom requires the current enemy preview")
        captured_at = current.observations.get("observed_at_monotonic")
        if type(captured_at) not in (int, float) or not math.isfinite(captured_at):
            raise FlowError("Minimum zoom requires a current preview capture time")
        preview_end = captured_at + _remaining_seconds(current) - 1
        before = current
        motion = None
        probes = receipt.setdefault("zoom_probes", [])
        for attempt in range(4):
            # Reserve time for one pinch, a full roster read, selected-border
            # verification and freezing. The final freeze still applies its own
            # stricter fresh-frame deadline.
            if time.monotonic() + 7 >= preview_end:
                raise FlowError("Preview time is insufficient to verify minimum zoom")
            session._validate_snapshot(before)
            gesture_started = time.monotonic()
            session.native.zoom_out()
            session.action_count += 1
            gesture_elapsed = time.monotonic() - gesture_started
            time.sleep(.2)
            observation_started = time.monotonic()
            after = session.observe("line-zoom-probe", purpose="settlement")
            observation_elapsed = time.monotonic() - observation_started
            receipt["evidence"].append(str(after.screenshot_path))
            if (after.scene != "enemy_village" or not math.isfinite(after.confidence)
                    or after.confidence < .8):
                raise FlowError("Enemy preview was lost while verifying minimum zoom")
            motion_started = time.monotonic()
            motion = measure_camera_motion(before.screenshot_path, after.screenshot_path)
            motion_elapsed = time.monotonic() - motion_started
            # The general camera detector tolerates 2% scale drift. That is
            # suitable for scene stability, but not proof of minimum zoom:
            # the real 00012->00013 pair drifted 0.42% and the subsequent
            # 00013->00014 pinch was still needed.
            converged = _minimum_zoom_motion(motion)
            probe = {"attempt": attempt + 1, "frame": str(after.screenshot_path),
                     "gesture_elapsed_sec": gesture_elapsed,
                     "observation_elapsed_sec": observation_elapsed,
                     "motion_elapsed_sec": motion_elapsed,
                     "converged": converged, "motion": motion}
            probes.append(probe)
            session.event("line_zoom_probe", **probe)
            if converged:
                break
            before = after
        else:
            raise FlowError("Minimum zoom did not stabilize before line deployment")
        if time.monotonic() + 5 >= preview_end:
            raise FlowError("Preview time is insufficient for the selected boundary")
        full_started = time.monotonic()
        current = session.observe("line-zoom-verified")
        full_elapsed = time.monotonic() - full_started
        receipt["evidence"].append(str(current.screenshot_path))
        if (current.scene != "enemy_village" or not math.isfinite(current.confidence)
                or current.confidence < .8):
            raise FlowError("Enemy preview was lost before line deployment")
        confirmation = measure_camera_motion(after.screenshot_path, current.screenshot_path)
        if not _minimum_zoom_motion(confirmation):
            raise FlowError("Minimum zoom changed before the full roster read")
        _remaining_seconds(current)
    else:
        for _ in range(3):
            session._validate_snapshot(current)
            session.native.zoom_out()
            session.action_count += 1
            time.sleep(.5)
        time.sleep(1)
        before = session.observe("line-zoom-out")
        _battle_frame(before)
        session._validate_snapshot(before)
        session.native.zoom_out()
        session.action_count += 1
        time.sleep(1)
        current = session.observe("line-zoom-verified")
        _battle_frame(current)
        motion = measure_camera_motion(before.screenshot_path, current.screenshot_path)
        receipt["evidence"].extend([str(before.screenshot_path), str(current.screenshot_path)])
    receipt["minimum_zoom"] = {"verified": motion["stationary"] is True, "motion": motion,
                               "frame": str(current.screenshot_path)}
    if capture_context is not None:
        receipt["minimum_zoom"].update(confirmation_motion=confirmation,
                                       gesture_count=attempt + 1,
                                       gesture_frame=str(after.screenshot_path),
                                       full_roster_elapsed_sec=full_elapsed)
    session.event("line_minimum_zoom", **receipt["minimum_zoom"])
    if motion["stationary"] is not True:
        raise FlowError("Minimum zoom did not stabilize before line deployment")
    if capture_context is not None:
        capture_context["toolbar_before"] = current
    # Minimum scale alone does not fix camera translation or the build boundary.
    # Selecting a troop reveals the current red border without placing anything.
    troops = [slot for slot in current.observations.get("battle", {}).get("slots", [])
              if slot["kind"] == "troop" and type(slot.get("count")) is int and slot["count"] > 0]
    if not troops:
        raise FlowError("No troop card available to reveal deployment boundaries")
    session.recognizer.battle_observer.remember(current)
    identity_frame = str(current.screenshot_path)
    session.tap(current, troops[0]["point"], reason="Reveal current deployment boundaries")
    failure_reason = "Selected troop card could not be verified while revealing deployment boundaries"
    for _ in range(3):
        current = session.observe("line-boundary")
        receipt["evidence"].append(str(current.screenshot_path))
        _battle_frame(current)
        selected_box = selected_card_bbox(current.screenshot_path, troops[0]["bbox"],
                                          session.config.game.baseline_resolution)
        if selected_box is None:
            continue
        # The selected white header can erase the hue used to classify a troop.
        # Its outline can also join a neighbor. Verify each original portrait and
        # current count independently before replacing any missing border detection.
        refreshed = []
        for index, troop in enumerate(troops):
            reading = _verified_line_card_reading(session, current, troop)
            portrait_score = reading.get("portrait_score")
            if (not isinstance(portrait_score, (int, float)) or not math.isfinite(portrait_score)
                    or portrait_score < .92 or reading.get("count") != troop["count"]):
                failure_reason = (f"Current troop card portrait or count unverified before boundary scan: "
                                  f"{troop.get('unit_id') or 'unknown'}")
                break
            box = selected_box if index == 0 else troop["bbox"]
            new_evidence = {**troop.get("evidence", {}), "count": reading,
                            "identity_reference": identity_frame,
                            "portrait_score": reading["portrait_score"]}
            prior = troop.get("evidence", {}).get("count") or {}
            if isinstance(prior, dict):
                prior_box = prior.get("bbox")
                anchor = {"bbox": list(prior_box) if isinstance(prior_box, (list, tuple)) else [],
                          "source_frame": identity_frame,
                          "source_slot_bbox": list(troop["bbox"]), "source_text": prior.get("text"),
                          "source_confidence": prior.get("confidence"),
                          "verified_frame": str(current.screenshot_path), "verified_count": reading["count"],
                          "verified_portrait_score": reading["portrait_score"],
                          "verified_slot_bbox": list(box), "unit_id": troop.get("unit_id")}
                if _trusted_count_roi_anchor({**troop, "bbox": list(box),
                                              "evidence": {**new_evidence, "count_roi_anchor": anchor}}) is not None:
                    new_evidence["count_roi_anchor"] = anchor
            refreshed.append({**troop, "bbox": list(box),
                "point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                "count": reading["count"], "evidence": new_evidence})
        if len(refreshed) != len(troops):
            continue
        slots = current.observations["battle"]["slots"]
        current.observations["battle"]["slots"] = sorted(
            [slot for slot in slots if all(abs(slot["point"][0] - troop["point"][0]) >= 25 for troop in troops)] + refreshed,
            key=lambda slot: slot["point"][0])
        terrain = find_line_deployment_edges(current.screenshot_path,
                                            baseline_resolution=session.config.game.baseline_resolution)
        if terrain:
            return current, terrain
        failure_reason = "Two current deployment boundaries could not be verified"
    raise FlowError(failure_reason)


def _deploy_continuous_line(session, current, receipt, context, hero_manifest, deadline):
    from uuid import uuid4
    from .frozen_line import execute_frozen_line, freeze_model_free_line
    from .probed_anchor import execute_probed_anchor_line, freeze_probed_anchor_line, remainder_burst_limit
    from .prepared_battle import _remaining_seconds

    # Starting the batch must remain inside the observed preview countdown.
    # Its separate execution budget covers the resulting active battle.
    if current.scene != "enemy_village":
        raise FlowError("Continuous line preparation did not finish inside the preview")
    captured_at = current.observations.get("observed_at_monotonic")
    if type(captured_at) not in (int, float) or not math.isfinite(captured_at):
        raise FlowError("Continuous line preview has no capture timestamp")
    latest_start = min(deadline, captured_at + min(10, _remaining_seconds(current) - 1))
    if not isinstance(hero_manifest, dict) or hero_manifest.get("complete") is not True:
        raise FlowError("Continuous line needs the independently verified hero roster")
    rows = hero_manifest.get("cards")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise FlowError("Continuous line hero roster is unavailable")
    tactics = getattr(session.config.battle, "line_tactics", "single_edge")
    freeze = freeze_probed_anchor_line if tactics == "two_edge_probed_anchors" else freeze_model_free_line
    plan = freeze(session, current, toolbar_before=context["toolbar_before"],
        terrain=receipt["terrain_evidence"], army_manifest=receipt["expected_army_manifest"],
        minimum_zoom=receipt["minimum_zoom"], battle_id=uuid4().hex,
        latest_start=latest_start, burst_limit_sec=10,
        **({} if tactics == "two_edge_probed_anchors" else {"line_tactics": tactics}),
        hero_ids=[row.get("unit_id") for row in rows], hero_manifest=hero_manifest)
    frozen = plan.base if tactics == "two_edge_probed_anchors" else plan
    probed = tactics == "two_edge_probed_anchors"
    session.event("model_free_line_plan", plan_id=frozen.plan_id,
                  plan_sha256=plan.digest if tactics == "two_edge_probed_anchors" else frozen.plan_sha256,
                  frame=str(current.screenshot_path), latest_start=latest_start,
                  input_count=len(plan.remaining) + 3 if tactics == "two_edge_probed_anchors" else len(frozen.actions),
                  burst_limit_sec=remainder_burst_limit(plan) if probed else frozen.burst_limit_sec,
                  **({"probe_limit_sec": 12., "remainder_limit_sec": remainder_burst_limit(plan)} if probed else {}),
                  line_tactics=tactics)
    post = (execute_probed_anchor_line(session, plan) if tactics == "two_edge_probed_anchors"
            else execute_frozen_line(session, plan))
    frames = receipt["evidence"]
    receipt.update(post.observations["deployment"])
    receipt["evidence"] = list(dict.fromkeys(frames + receipt["evidence"]))
    post.observations["deployment"] = receipt
    return post


def _current_full_header_count(snapshot: SceneSnapshot, troop: dict,
                               client_version: str) -> dict | None:
    """Use the full recognizer's OCR only beside its own closed current card.

    ``ScreenshotRecognizer.recognize`` builds ``ocr`` and ``battle.slots`` from
    one screenshot path in the same call. Neither the earlier army count nor a
    remembered BattleObserver card supplies this quantity.
    """
    if (client_version != "18.647.2" or snapshot.scene not in {"enemy_village", "battle"}
            or type(snapshot.confidence) not in (int, float)
            or not math.isfinite(snapshot.confidence) or snapshot.confidence < .8
            or snapshot.observations.get("recognition_scope", "full") != "full"
            or snapshot.observations.get("baseline_resolution") not in ([1280, 720], (1280, 720))):
        return None
    box = troop.get("bbox")
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) is not int for value in box)):
        return None
    left, top, right, bottom = box
    if not (0 <= left < right <= 1280 and 575 <= top < bottom <= 720):
        return None
    battle = snapshot.observations.get("battle")
    slots = battle.get("slots") if isinstance(battle, dict) else None
    if not isinstance(slots, list):
        return None
    closed = [slot for slot in slots if isinstance(slot, dict)
              and slot.get("kind") == "troop" and slot.get("unit_id") == troop.get("unit_id")
              and slot.get("bbox") == list(box)
              and isinstance(slot.get("evidence"), dict)
              and slot["evidence"].get("method") == "independent_card_border"]
    if len(closed) != 1:
        return None
    card = closed[0]
    identity = card["evidence"].get("identity")
    card_score = card.get("confidence")
    identity_score = identity.get("confidence") if isinstance(identity, dict) else None
    if (not isinstance(identity, dict) or identity.get("unit_id") != troop.get("unit_id")
            or identity.get("version_verified") is not True
            or identity.get("template_client_version") != client_version
            or type(card_score) not in (int, float) or not math.isfinite(card_score)
            or card_score < .94 or type(identity_score) not in (int, float)
            or not math.isfinite(identity_score) or identity_score < .94):
        return None
    counts = _snapshot_header_numbers(snapshot, box)
    if len(counts) != 1:
        return None
    count, item = counts[0]
    # The detector's count is a consistency check, never a source of the
    # recovered quantity. It may be absent, but may not contradict the OCR.
    if type(card.get("count")) is int and card["count"] != count:
        return None
    return {"count": count, "text": item["text"], "confidence": item["confidence"],
            "bbox": list(item["bbox"]), "frame": str(snapshot.screenshot_path),
            "source": "current_snapshot_full_ocr"}


def _snapshot_header_numbers(snapshot: SceneSnapshot, bbox: list[int] | tuple[int, ...]) -> list[tuple[int, dict]]:
    texts = snapshot.observations.get("ocr")
    if not isinstance(texts, list):
        return []
    left, top, right, _ = bbox
    numbers = []
    for item in texts:
        if not isinstance(item, dict):
            continue
        box, confidence = item.get("bbox"), item.get("confidence")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or not (left <= box[0] < box[2] <= right and
                        max(0, top - 16) <= box[1] < box[3] <= min(720, top + 40))
                or type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not .95 <= confidence <= 1):
            continue
        match = re.fullmatch(r"[xX×]\s*([0-9]+)", str(item.get("text", "")).strip())
        if match is not None:
            numbers.append((int(match[1]), item))
    return numbers


def _verified_line_card_reading(session: GameSession, snapshot: SceneSnapshot, troop: dict) -> dict:
    """Bind an OCR fallback to this frame's versioned troop portrait.

    The fast count reader normally proves both portrait and quantity. A small
    `x1` may fail local OCR even when the current full screenshot OCR reads it.
    Fallbacks still need independent same-frame card geometry and identity;
    neither a previous count nor a card position supplies the quantity.
    """
    reading = session.recognizer.recognize_slot_count(snapshot.screenshot_path, tuple(troop["bbox"]))
    count, confidence = reading.get("count"), reading.get("confidence")
    recovered_header = False
    full_header = None
    if reading.get("reason") == "header_count_unverified" and count is None:
        # The thresholded line OCR can hallucinate text over a card outline.
        # Try its unthresholded original first; only then consider the full
        # OCR and independently closed card of this very same snapshot.
        readings = reading.get("readings")
        originals = [item for item in readings if item.get("source") == "header_region_ocr"] if isinstance(readings, list) else []
        if len(originals) == 1:
            original = originals[0]
            match = re.fullmatch(r"[xX×]\s*([0-9]+)", str(original.get("text", "")).strip())
            original_confidence = original.get("confidence")
            if (match is not None and type(original_confidence) in (int, float)
                    and math.isfinite(original_confidence) and original_confidence >= .95):
                count, confidence = int(match[1]), original_confidence
                recovered_header = True
        if not recovered_header:
            full_header = _current_full_header_count(snapshot, troop, session.client_version)
            if full_header is None:
                return reading
            count, confidence = full_header["count"], full_header["confidence"]
            recovered_header = True
        if not isinstance(readings, list) or any(not isinstance(item, dict) for item in readings):
            return reading
        for item in readings:
            if item.get("source") not in {"header_region_ocr", "header_threshold_line"}:
                return reading
            other = re.fullmatch(r"[xX×]\s*([0-9]+)", str(item.get("text", "")).strip())
            if other is not None and int(other[1]) != count:
                return reading
        if full_header is None and any(item is not originals[0] and
                                       item.get("source") != "header_threshold_line"
                                       for item in readings):
            return reading
    if reading.get("reason") == "header_count_unverified" and not recovered_header:
        return reading
    if (reading.get("reason") not in {"explicit_header_count", "header_count_unverified"} or
            count != troop.get("count") or
            reading.get("frame") != str(snapshot.screenshot_path) or
            reading.get("slot_bbox") != list(troop["bbox"]) or
            not isinstance(troop.get("unit_id"), str)):
        return reading
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < .9:
        return reading
    if any(value != count for value, _ in _snapshot_header_numbers(snapshot, troop["bbox"])):
        return {**reading, "count": None, "confidence": 0.0,
                "portrait_score": None, "reason": "strong_count_conflict"}
    from .images import read_frame
    from .unit_catalog import recognize_card_identity
    import cv2

    source = read_frame(snapshot.screenshot_path, cv2.IMREAD_COLOR)
    if source is None:
        return reading
    image = cv2.resize(source, (1280, 720), interpolation=cv2.INTER_AREA)
    identity = recognize_card_identity(image, tuple(troop["bbox"]), "troop", surface="battle",
                                       client_version=session.client_version)
    if (identity["version_verified"] and identity["unit_id"] == troop["unit_id"]
            and isinstance(identity["confidence"], (int, float)) and identity["confidence"] >= .94):
        return {**reading, "count": count, "confidence": confidence,
                "reason": ("portrait_bound_current_full_header_count" if full_header is not None else
                           "portrait_bound_full_header_count" if recovered_header else reading["reason"]),
                "portrait_score": identity["confidence"],
                "portrait_source": "current_frame_versioned_battle_template", "portrait_identity": identity,
                **({"full_header": full_header} if full_header is not None else {})}
    return reading


def spread_along_edges(terrain: list[dict], count: int) -> list[list[int]]:
    """Split a stack evenly and interpolate along each observed flank."""
    if type(count) is not int or count <= 0:
        raise FlowError("Line deployment requires a positive troop count")
    return [list(point) for point in (spread_on_edge(terrain, 0, (count + 1) // 2)
                                      + spread_on_edge(terrain, 1, count // 2))]


def _deploy_line_cards(session: GameSession, current: SceneSnapshot, receipt: dict,
                       plan: DeploymentPlan, deadline: float) -> SceneSnapshot:
    started = time.monotonic()
    receipt.update(scope="troops_and_heroes" if plan.heroes else "troops_only",
                   excluded_kinds=["siege", "spell"] + ([] if plan.heroes else ["hero"]), issued_placements=0)

    def check() -> None:
        session.check_deadline()
        if time.monotonic() >= deadline:
            raise FlowError("Army deployment deadline exceeded")

    def record_count_reads(frame: SceneSnapshot) -> None:
        for reading in frame.observations.get("deployment_count_reads", []):
            if reading not in receipt["count_reads"]:
                receipt["count_reads"].append(reading)
                session.event("deployment_count_read", **reading)

    def observe(label: str, *, purpose: str = "troop_count", slot: dict | None = None) -> SceneSnapshot:
        check()
        frame = session.observe(label, purpose=purpose, slot=slot)
        receipt["evidence"].append(str(frame.screenshot_path))
        record_count_reads(frame)
        check()
        _battle_frame(frame)
        return frame

    for step in plan.troops:
        slot = step.card
        check()
        old = _remaining(current, slot, recognizer=getattr(session, "recognizer", None))
        record_count_reads(current)
        check()
        if old != slot["count"]:
            raise FlowError("Troop quantity changed before line deployment")
        session.tap(current, slot["point"], reason=f"Select troop stack for line: {old}")
        current = observe("line-selected", slot=slot)
        selected_count = _remaining(current, slot, recognizer=getattr(session, "recognizer", None))
        record_count_reads(current)
        if selected_count != old:
            raise FlowError("Troop quantity changed before line placement")
        planned = [list(point) for point in step.points]
        before = current
        issued = 0
        event = {"unit_kind": "troop", "slot": slot["point"], "points": planned,
                 "before_count": old, "issued_placements": 0, "before_frame": str(before.screenshot_path),
                 "after_count": None, "consumed": 0}
        receipt["consumption_events"].append(event)
        # All placements use one selected card and a fixed camera. Refresh only
        # if a very large stack outlives the snapshot, without selecting it again.
        for point in planned:
            check()
            if time.monotonic() - current.observations["observed_at_monotonic"] > 20:
                current = observe("line-refresh", slot=slot)
            session.tap(current, point, reason="Place troop along planned deployment line")
            issued += 1
            event["issued_placements"] = issued
            receipt["issued_placements"] += 1
            time.sleep(.12)
        for attempt in range(3):
            if attempt:
                time.sleep(min(session.config.runtime.poll_interval_sec, 1))
            current = observe("line-consumption", slot=slot)
            after = _remaining(current, slot, recognizer=getattr(session, "recognizer", None))
            record_count_reads(current)
            check()
            if after is None:
                continue
            if not 0 <= after <= old - event["consumed"]:
                raise FlowError("Troop quantity increased while verifying line deployment")
            consumed = old - after
            receipt["deployed_units"] += consumed - event["consumed"]
            receipt["verified"] = receipt["deployed_units"] > 0
            event.update(after_count=after, consumed=consumed, after_frame=str(current.screenshot_path))
            if after == 0:
                break
        session.event("line_deployment_consumed", **event)
        if event["after_count"] != 0:
            raise FlowError("Line deployment left troops unverified; placements will not be replayed")
    receipt["numeric_completed"] = True
    for step in plan.heroes:
        from .hero_state import recognize_hero_state

        slot = step.card
        if current.observations.get("recognition_scope") != "full":
            # Fast count and hero-state frames omit all battle slots. Refresh
            # the full roster before each selection, binding that hero's
            # identity to the same fresh frame used for the tap.
            current = observe("line-hero-roster", purpose="full")
        visible_heroes = [item for item in current.observations.get("battle", {}).get("slots", [])
                          if item.get("kind") == "hero"
                          and abs(item["point"][0] - slot["point"][0]) < 25]
        if (session.client_version not in {"18.600.7", "18.647.2"}
                or not isinstance(slot.get("unit_id"), str)
                or not slot["unit_id"]
                or len(visible_heroes) != 1
                or visible_heroes[0].get("unit_id") != slot["unit_id"]):
            raise FlowError("Hero identity is unverified; selection will not be attempted")
        check()
        attempt = {"kind": "hero", "point": slot["point"], "status": "attempted", "verified": False,
                   "ability_clicks": 0, "selection_frame": str(current.screenshot_path),
                   "placement_frame": None, "states": []}
        receipt["support_attempts"].append(attempt)
        session.tap(current, slot["point"], reason="Select hero to follow the line")
        for _ in range(3):
            current = observe("line-hero-selected", purpose="hero")
            selected = recognize_hero_state(current.screenshot_path, slot["bbox"],
                                           baseline_resolution=session.config.game.baseline_resolution,
                                           unit_id=slot.get("unit_id"), client_version=session.client_version)
            check()
            attempt["states"].append(selected)
            if selected["selected"] is True or selected["deployed"] is True:
                break
        if selected["selected"] is not True or selected["deployed"] is True:
            raise FlowError("Hero selection is unverified; placement will not be attempted")
        attempt["placement_frame"] = str(current.screenshot_path)
        session.tap(current, list(step.points[0]), reason="Place hero at the line midpoint")
        for _ in range(3):
            current = observe("line-hero-deployed", purpose="hero")
            state = recognize_hero_state(current.screenshot_path, slot["bbox"],
                                        baseline_resolution=session.config.game.baseline_resolution,
                                        unit_id=slot.get("unit_id"), client_version=session.client_version)
            check()
            attempt["states"].append(state)
            session.event("hero_state", **state)
            if state["deployed"] is True:
                attempt.update(verified=True, status="deployed")
                break
        if not attempt["verified"]:
            raise FlowError("Hero deployment is unverified; placement will not be replayed")
    receipt.update(completed=True, support_deployment_verified=True,
                   deployment_elapsed_sec=time.monotonic() - started)
    current.observations["deployment"] = receipt
    session.event("line_deployment_complete", deployed_units=receipt["deployed_units"],
                  elapsed_sec=receipt["deployment_elapsed_sec"], scope=receipt["scope"])
    return current
