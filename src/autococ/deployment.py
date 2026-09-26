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
            reading = recognizer.recognize_slot_count(snapshot.screenshot_path, tuple(slot["bbox"]))
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

    if line_mode:
        current, terrain = _prepare_two_edge_view(session, current, receipt)
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
                                             baseline_resolution=session.config.game.baseline_resolution)
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
                                               baseline_resolution=session.config.game.baseline_resolution)
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


def _prepare_two_edge_view(session: GameSession, current: SceneSnapshot, receipt: dict) -> tuple[SceneSnapshot, list[dict]]:
    from .battle_vision import selected_card_bbox
    from .terrain import find_line_deployment_edges, measure_camera_motion

    if session.native is None:
        raise FlowError("Two-edge minimum zoom requires the MuMu native transport")
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
    session.event("line_minimum_zoom", **receipt["minimum_zoom"])
    if motion["stationary"] is not True:
        raise FlowError("Minimum zoom did not stabilize before line deployment")
    # Minimum scale alone does not fix camera translation or the build boundary.
    # Selecting a troop reveals the current red border without placing anything.
    troops = [slot for slot in current.observations.get("battle", {}).get("slots", [])
              if slot["kind"] == "troop" and type(slot.get("count")) is int and slot["count"] > 0]
    if not troops:
        raise FlowError("No troop card available to reveal deployment boundaries")
    session.recognizer.battle_observer.remember(current)
    identity_frame = str(current.screenshot_path)
    session.tap(current, troops[0]["point"], reason="Reveal current deployment boundaries")
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
            reading = session.recognizer.recognize_slot_count(current.screenshot_path, tuple(troop["bbox"]))
            if reading.get("portrait_score", 0) < .92 or reading.get("count") != troop["count"]:
                break
            box = selected_box if index == 0 else troop["bbox"]
            refreshed.append({**troop, "bbox": list(box),
                "point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                "count": reading["count"], "evidence": {**troop.get("evidence", {}),
                "count": reading, "identity_reference": identity_frame,
                "portrait_score": reading["portrait_score"]}})
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
    raise FlowError("Two current deployment boundaries could not be verified")


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

    def observe(label: str, *, purpose: str = "troop_count", slot: dict | None = None) -> SceneSnapshot:
        check()
        frame = session.observe(label, purpose=purpose, slot=slot)
        receipt["evidence"].append(str(frame.screenshot_path))
        for reading in frame.observations.get("deployment_count_reads", []):
            receipt["count_reads"].append(reading)
            session.event("deployment_count_read", **reading)
        check()
        _battle_frame(frame)
        return frame

    for step in plan.troops:
        slot = step.card
        check()
        old = _remaining(current, slot, recognizer=getattr(session, "recognizer", None))
        check()
        if old != slot["count"]:
            raise FlowError("Troop quantity changed before line deployment")
        session.tap(current, slot["point"], reason=f"Select troop stack for line: {old}")
        current = observe("line-selected", slot=slot)
        if _remaining(current, slot, recognizer=getattr(session, "recognizer", None)) != old:
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
        check()
        attempt = {"kind": "hero", "point": slot["point"], "status": "attempted", "verified": False,
                   "ability_clicks": 0, "selection_frame": str(current.screenshot_path),
                   "placement_frame": None, "states": []}
        receipt["support_attempts"].append(attempt)
        session.tap(current, slot["point"], reason="Select hero to follow the line")
        for _ in range(3):
            current = observe("line-hero-selected", purpose="hero")
            selected = recognize_hero_state(current.screenshot_path, slot["bbox"],
                                           baseline_resolution=session.config.game.baseline_resolution)
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
                                        baseline_resolution=session.config.game.baseline_resolution)
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
