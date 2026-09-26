"""Execute validated strategy actions against fresh battle observations."""
from __future__ import annotations

import math
import time
from dataclasses import asdict

from .deployment import _battle_frame, _remaining, _prepare_two_edge_view
from .errors import DeploymentError, FlowError, StopRequested
from .scene import SceneSnapshot
from .strategies import BattleContext, spread_on_edge
from .strategy_config import StrategyDefinition, StrategyStep, planned_steps


def _cards(frame: SceneSnapshot) -> list[dict]:
    return frame.observations.get("battle", {}).get("slots", [])


def _match_card(frame: SceneSnapshot, unit_id: str, kind: str, *, source: str = "auto",
                prior: dict | None = None) -> dict:
    cards = _cards(frame)
    if unit_id == "*":
        matches = [c for c in cards if c.get("kind") == kind and
                   (c.get("source", "army") in {"army", "event"} if source == "auto"
                    else c.get("source", "army") == source) and
                   type(c.get("count")) is int and c["count"] > 0]
        if prior is not None:
            original_source = prior.get("source", "army")
            original_id = prior.get("unit_id")
            original_count = prior.get("count")
            original_point = prior.get("point")
            if (type(original_count) is not int or original_count <= 0 or
                    not isinstance(original_point, (list, tuple)) or len(original_point) != 2):
                raise FlowError("Captured troop has no original identity and count anchor")
            bound = []
            for card in matches:
                point = card.get("point")
                if (card.get("source", "army") != original_source or
                        card["count"] != original_count or
                        not isinstance(point, (list, tuple)) or len(point) != 2):
                    continue
                if original_id:
                    confidence = card.get("confidence")
                    if (card.get("unit_id") != original_id or
                            not isinstance(confidence, (int, float)) or
                            not math.isfinite(confidence) or confidence < .9 or
                            any(abs(point[axis] - original_point[axis]) > 8 for axis in (0, 1))):
                        continue
                else:
                    # An unidentified legacy card may move by a couple of pixels,
                    # but position alone cannot establish which portrait it is.
                    evidence = card.get("evidence") or {}
                    reading = evidence.get("count") or {}
                    score = evidence.get("portrait_score")
                    if (card.get("unit_id") is not None or
                            any(abs(point[axis] - original_point[axis]) > 3 for axis in (0, 1)) or
                            not isinstance(score, (int, float)) or
                            not math.isfinite(score) or score < .92 or
                            not isinstance(reading, dict) or
                            reading.get("frame") != str(frame.screenshot_path) or
                            reading.get("count") != original_count or
                            not isinstance(reading.get("confidence"), (int, float)) or
                            not math.isfinite(reading["confidence"]) or reading["confidence"] < .9):
                        continue
                bound.append(card)
            matches = bound
    else:
        matches = [c for c in cards if c.get("kind") == kind and c.get("unit_id") == unit_id and
                   (c.get("source", "army") in {"army", "event"} if source == "auto"
                    else c.get("source", "army") == source)]
    if len(matches) != 1:
        raise FlowError(f"Expected one identified {kind} card for {unit_id}, found {len(matches)}")
    card = matches[0]
    confidence = card.get("confidence")
    if ((kind in {"troop", "spell"} and (type(card.get("count")) is not int or card["count"] < 0)) or
            (unit_id != "*" and (not isinstance(confidence, (int, float)) or
                                  not math.isfinite(confidence) or confidence < .9))):
        raise FlowError(f"Card quantity or identity for {unit_id} is unverified")
    return card


def _building(frame: SceneSnapshot, target: str, *, building_id: str | None = None,
              state: str = "alive") -> dict:
    buildings = frame.observations.get("buildings", frame.observations.get("battle", {}).get("buildings", []))
    if not isinstance(buildings, list):
        raise FlowError("Building recognition is unavailable")
    candidates = [b for b in buildings if isinstance(b, dict) and
                  (b.get("type") == target or b.get("building_id") == target) and
                  (building_id is None or b.get("building_id") == building_id) and
                  b.get("state", b.get("status")) == state and
                  b.get("frame") == str(frame.screenshot_path) and
                  isinstance(b.get("confidence"), (int, float)) and
                  math.isfinite(b["confidence"]) and b["confidence"] >= .9 and
                  b.get("evidence")]
    if not candidates or building_id is not None and len(candidates) != 1:
        raise FlowError(f"No unique fresh {state} building evidence for {target}")
    if building_id is None:
        # A village may contain several copies of one building type. Each has
        # its own stable observed ID; choose one reproducibly, then bind all
        # later checks to that ID rather than drifting to a nearby building.
        candidates.sort(key=lambda item: (item["point"][0], item["point"][1], item["building_id"]))
    return candidates[0]


def _scene(frame: SceneSnapshot) -> None:
    _battle_frame(frame)


def _check(session, deadline: float) -> None:
    # The session stop signal takes precedence over a strategy timeout.
    session.check_deadline()
    if time.monotonic() >= deadline:
        raise FlowError("Strategy deployment deadline exceeded")


class _DeadlineSession:
    """Guard input in the shared terrain preparer without changing its API."""

    def __init__(self, session, deadline: float):
        self._session = session
        self._deadline = deadline

    def __getattr__(self, name):
        return getattr(self._session, name)

    @property
    def action_count(self):
        return self._session.action_count

    @action_count.setter
    def action_count(self, value):
        self._session.action_count = value

    @property
    def native(self):
        native = self._session.native
        if native is None:
            return None

        class _Native:
            def zoom_out(inner):
                _check(self._session, self._deadline)
                return native.zoom_out()

        return _Native()

    def _validate_snapshot(self, frame):
        _check(self._session, self._deadline)
        return self._session._validate_snapshot(frame)

    def tap(self, frame, point, *, reason):
        _check(self._session, self._deadline)
        return self._session.tap(frame, point, reason=reason)

    def observe(self, label, **kwargs):
        _check(self._session, self._deadline)
        frame = self._session.observe(label, **kwargs)
        _check(self._session, self._deadline)
        return frame


def _fresh(session, label: str, receipt: dict, deadline: float, *, purpose: str = "full",
           slot: dict | None = None) -> SceneSnapshot:
    _check(session, deadline)
    frame = session.observe(label, purpose=purpose, slot=slot) if purpose != "full" else session.observe(label)
    receipt["evidence"].append(str(frame.screenshot_path))
    _check(session, deadline)
    _scene(frame)
    return frame


def _battle_bar_viewport(frame: SceneSnapshot) -> tuple:
    """Recognized card identities, not screenshot names or old click positions."""
    slots = _cards(frame)
    return tuple((card.get("kind"), card.get("unit_id"), card.get("source", "army"),
                  card.get("count"), tuple(round(value / 8) for value in card.get("point", ())))
                 for card in slots if isinstance(card, dict))


def _observe_swiped_bar(session, before_view: tuple, direction: str,
                        receipt: dict, deadline: float) -> tuple[SceneSnapshot, tuple, bool]:
    """Observe an animated swipe; never swipe again on its first old frame."""
    previous_new = None
    old_reads = 0
    for attempt in range(3):
        _check(session, deadline)
        frame = session.observe(f"strategy-bar-{direction}")
        receipt["evidence"].append(str(frame.screenshot_path))
        _check(session, deadline)
        if frame.scene in {"enemy_village", "battle"} and math.isfinite(frame.confidence) and frame.confidence >= .8:
            view = _battle_bar_viewport(frame)
            if view and view == before_view:
                old_reads += 1
                previous_new = None
                if old_reads == 3:
                    return frame, view, True
            elif view and view == previous_new:
                return frame, view, False
            else:
                previous_new = view or None
                old_reads = 0
        elif frame.scene != "unknown":
            raise FlowError(f"Battle bar swipe entered unexpected scene {frame.scene}")
        if attempt < 2:
            _check(session, deadline)
            time.sleep(.1)
    raise FlowError("Battle bar viewport did not stabilize after one swipe")


def _seek_battle_bar(session, frame: SceneSnapshot, choose, receipt: dict,
                     deadline: float, label: str) -> tuple[SceneSnapshot, dict]:
    """Find a currently visible verified card within bounded bar-only swipes."""
    _check(session, deadline)
    _scene(frame)
    if frame.observations.get("recognition_scope", "full") != "full":
        frame = _fresh(session, "strategy-bar-full", receipt, deadline)
    card = choose(frame)
    if card is not None:
        return frame, card
    previous_view = _battle_bar_viewport(frame)
    if not previous_view:
        raise FlowError(f"{label} has no current battle bar evidence")
    for direction in ("left", "right"):
        for _ in range(3):
            _check(session, deadline)
            session.swipe_battle_bar(frame, direction=direction,
                                     reason=f"Find current battle card: {label}")
            frame, view, repeated = _observe_swiped_bar(session, previous_view, direction,
                                                        receipt, deadline)
            if repeated:
                break
            previous_view = view
            card = choose(frame)
            if card is not None:
                return frame, card
    raise FlowError(f"{label} was not identified in bounded battle bar search")


def _find_named_card(session, frame: SceneSnapshot, unit_id: str, kind: str,
                     source: str, receipt: dict, deadline: float) -> tuple[SceneSnapshot, dict]:
    def choose(current):
        matches = [card for card in _cards(current) if card.get("kind") == kind and
                   card.get("unit_id") == unit_id and
                   (card.get("source", "army") in {"army", "event"} if source == "auto"
                    else card.get("source", "army") == source)]
        # A visible but ambiguous/low-confidence card is a real failure, not
        # permission to keep scrolling until a more convenient match appears.
        return _match_card(current, unit_id, kind, source=source) if matches else None

    return _seek_battle_bar(session, frame, choose, receipt, deadline,
                            f"{kind} {unit_id} ({source})")


def _find_reveal_troop(session, frame: SceneSnapshot, receipt: dict,
                       deadline: float) -> SceneSnapshot:
    def choose(current):
        troops = [card for card in _cards(current) if card.get("kind") == "troop" and
                  card.get("source", "army") in {"army", "event"} and
                  type(card.get("count")) is int and card["count"] > 0]
        return troops[0] if troops else None

    return _seek_battle_bar(session, frame, choose, receipt, deadline,
                            "troop for boundary reveal")[0]


def _count(session, frame: SceneSnapshot, card: dict, deadline: float) -> int:
    _check(session, deadline)
    count = _remaining(frame, card, recognizer=getattr(session, "recognizer", None))
    _check(session, deadline)
    if type(count) is not int or count < 0:
        raise FlowError("Selected unit count is unreadable")
    return count


def _terrain(session, frame: SceneSnapshot, receipt: dict,
             needed: set[int], deadline: float) -> tuple[SceneSnapshot, list[dict]]:
    _check(session, deadline)
    cached = frame.observations.get("terrain")
    if _has_current_terrain(frame, needed):
        terrain = cached
    else:
        frame, terrain = _prepare_two_edge_view(_DeadlineSession(session, deadline), frame, receipt)
    _check(session, deadline)
    if not terrain or not _has_edges(terrain, needed):
        raise FlowError("No verified deployment boundary")
    return frame, terrain


def _needed_edges(steps: tuple[StrategyStep, ...]) -> set[int]:
    needed = set()
    for step in steps:
        if step.action in {"deploy_troop", "deploy_hero", "deploy_siege"}:
            needed.update({0, 1} if step.edge == "two" else
                          {1} if step.edge == "second" else {0})
        elif step.action == "cast_spell" and step.target.startswith("relative:"):
            needed.update({0} if step.target == "relative:first_midpoint" else
                          {1} if step.target == "relative:second_midpoint" else {0, 1})
    return needed


def _has_edges(terrain: list[dict], needed: set[int]) -> bool:
    return all(len({tuple(item["point"]) for item in terrain if item.get("edge") == edge}) >= 2
               for edge in needed)


def _has_current_terrain(frame: SceneSnapshot, needed: set[int]) -> bool:
    cached = frame.observations.get("terrain")
    return (isinstance(cached, list) and bool(cached) and
            all(isinstance(item, dict) and item.get("frame") == str(frame.screenshot_path)
                and item.get("verified") is True for item in cached) and
            _has_edges(cached, needed))


def _points(terrain: list[dict], edge: str, count: int) -> list[list[int]]:
    if edge == "midpoint":
        return [list(p) for p in spread_on_edge(terrain, 0, 1)] * count
    if edge == "two":
        return [list(p) for p in (spread_on_edge(terrain, 0, (count + 1) // 2) +
                                  spread_on_edge(terrain, 1, count // 2))]
    return [list(p) for p in spread_on_edge(terrain, 0 if edge in {"first", "west"} else 1, count)]


def _relative_spell_point(terrain: list[dict], target: str) -> list[int]:
    if target == "relative:first_midpoint":
        return list(spread_on_edge(terrain, 0, 1)[0])
    if target == "relative:second_midpoint":
        return list(spread_on_edge(terrain, 1, 1)[0])
    first = spread_on_edge(terrain, 0, 1)[0]
    second = spread_on_edge(terrain, 1, 1)[0]
    return [round((first[axis] + second[axis]) / 2) for axis in (0, 1)]


def _consume(session, frame: SceneSnapshot, card: dict, points: list[list[int]],
             receipt: dict, deadline: float, *, kind: str) -> SceneSnapshot:
    old = _count(session, frame, card, deadline)
    if not 0 < len(points) <= old:
        raise FlowError(f"{kind} action exceeds visible unit count")
    _check(session, deadline)
    session.tap(frame, card["point"], reason=f"Select identified {kind} {card.get('unit_id')}")
    frame = _fresh(session, "strategy-selected", receipt, deadline, purpose="troop_count", slot=card)
    if _count(session, frame, card, deadline) != old:
        raise FlowError("Unit count changed before placement")
    before = str(frame.screenshot_path)
    event = {"action": "cast_spell" if kind == "spell" else "deploy_troop",
             "unit_id": card.get("unit_id"), "before_count": old, "after_count": None,
             "issued_placements": 0, "consumed": 0, "points": points, "before_frame": before}
    receipt["actions"].append(event)
    # A fresh count after each tap bounds uncertainty to one placement. Never
    # replay an uncertain click; preserve the receipt for diagnosis instead.
    for point in points:
        _check(session, deadline)
        session.tap(frame, point, reason=f"Place selected {kind} at verified target")
        event["issued_placements"] += 1
        receipt["issued_placements"] += 1
        expected = old - event["issued_placements"]
        observed = None
        for _ in range(3):
            frame = _fresh(session, "strategy-consumption", receipt, deadline,
                           purpose="troop_count", slot=card)
            observed = _remaining(frame, card, recognizer=getattr(session, "recognizer", None))
            if observed == expected:
                event.update(after_count=observed, consumed=old - observed,
                             after_frame=str(frame.screenshot_path))
                if kind == "spell":
                    receipt["spells_used"] = receipt.get("spells_used", 0) + 1
                else:
                    receipt["deployed_units"] = receipt.get("deployed_units", 0) + 1
                receipt["offensive_actions"] += 1
                receipt["verified"] = True
                _check(session, deadline)
                break
            _check(session, deadline)
            if observed is not None and observed < expected:
                raise FlowError("Consumption exceeds issued placements")
        if observed != expected:
            raise FlowError("Unit placement consumption could not be verified")
    return _fresh(session, "strategy-action-complete", receipt, deadline)


def _hero(session, frame: SceneSnapshot, card: dict, receipt: dict, point: list[int],
          deadline: float, *, ability: bool) -> SceneSnapshot:
    from .hero_state import recognize_hero_state

    def read(current: SceneSnapshot) -> dict:
        _check(session, deadline)
        state = recognize_hero_state(current.screenshot_path, card["bbox"],
                                     baseline_resolution=session.config.game.baseline_resolution)
        receipt["hero_states"].append(state)
        _check(session, deadline)
        return state

    if ability:
        before = read(frame)
        if before.get("deployed") is not True or before.get("ability_ready") is not True:
            raise FlowError("Hero ability has no verified ready state")
        event = {"action": "activate_ability", "unit_id": card.get("unit_id"),
                 "before_frame": str(frame.screenshot_path), "verified": False, "clicks": 0}
        receipt["actions"].append(event)
        _check(session, deadline)
        session.tap(frame, card["point"], reason="Activate verified ready hero ability")
        event["clicks"] = 1
        for _ in range(3):
            frame = _fresh(session, "strategy-hero-ability", receipt, deadline, purpose="hero")
            if read(frame).get("ability_used") is True:
                event.update(verified=True, after_frame=str(frame.screenshot_path))
                receipt["hero_abilities"] += 1
                return _fresh(session, "strategy-ability-complete", receipt, deadline)
        raise FlowError("Hero ability state remained uncertain; no second activation issued")

    event = {"action": "deploy_hero", "unit_id": card.get("unit_id"),
             "before_frame": str(frame.screenshot_path), "verified": False, "clicks": 0}
    receipt["actions"].append(event)
    _check(session, deadline)
    session.tap(frame, card["point"], reason="Select identified hero")
    event["clicks"] += 1
    for _ in range(3):
        frame = _fresh(session, "strategy-hero-selected", receipt, deadline, purpose="hero")
        if read(frame).get("selected") is True:
            break
    else:
        raise FlowError("Hero selection unverified")
    _check(session, deadline)
    session.tap(frame, point, reason="Deploy selected hero at verified boundary")
    event["clicks"] += 1
    for _ in range(3):
        frame = _fresh(session, "strategy-hero-deployed", receipt, deadline, purpose="hero")
        if read(frame).get("deployed") is True:
            event.update(verified=True, after_frame=str(frame.screenshot_path))
            receipt["offensive_actions"] += 1
            receipt["verified"] = True
            return _fresh(session, "strategy-hero-complete", receipt, deadline)
    raise FlowError("Hero deployment unverified; no placement replayed")


def _siege(session, frame: SceneSnapshot, card: dict, receipt: dict, point: list[int],
           deadline: float) -> SceneSnapshot:
    from .battle_vision import selected_card_bbox
    from .hero_state import recognize_siege_state

    event = {"action": "deploy_siege", "unit_id": card.get("unit_id"),
             "before_frame": str(frame.screenshot_path), "verified": False, "clicks": 0,
             "states": []}
    receipt["actions"].append(event)
    _check(session, deadline)
    session.tap(frame, card["point"], reason="Select identified siege machine")
    event["clicks"] = 1
    selected = False
    for _ in range(3):
        frame = _fresh(session, "strategy-siege-selected", receipt, deadline)
        selected = selected_card_bbox(frame.screenshot_path, card["bbox"],
                                      session.config.game.baseline_resolution) is not None
        if selected:
            break
    if not selected:
        raise FlowError("Siege selection is unverified")
    _check(session, deadline)
    session.tap(frame, point, reason="Deploy selected siege machine at verified boundary")
    event["clicks"] = 2
    for _ in range(3):
        frame = _fresh(session, "strategy-siege-deployed", receipt, deadline)
        state = recognize_siege_state(frame.screenshot_path, card["bbox"],
                                      baseline_resolution=session.config.game.baseline_resolution)
        event["states"].append(state)
        _check(session, deadline)
        if state.get("deployed") is True:
            event.update(verified=True, after_frame=str(frame.screenshot_path))
            receipt["siege_deployed"] += 1
            receipt["offensive_actions"] += 1
            receipt["verified"] = True
            return frame
    raise FlowError("Siege deployment unverified; no placement replayed")


def _wait_step(session, frame: SceneSnapshot, step: StrategyStep, receipt: dict,
               deployment_deadline: float = math.inf) -> SceneSnapshot:
    begun = time.monotonic()
    deadline = begun + step.timeout_sec
    while time.monotonic() - begun < step.delay_sec:
        _check(session, deployment_deadline)
        if time.monotonic() >= deadline:
            raise FlowError("Strategy delay exceeded timeout")
        time.sleep(min(.1, step.delay_sec - (time.monotonic() - begun),
                       max(0, deployment_deadline - time.monotonic())))
    if step.delay_sec:
        _check(session, deployment_deadline)
        frame = session.observe("strategy-delay")
        receipt["evidence"].append(str(frame.screenshot_path))
        _check(session, deployment_deadline)
        if (not math.isfinite(frame.confidence) or frame.confidence < .8 or
                frame.scene not in {"battle", "enemy_village", step.target}):
            raise FlowError("Unexpected scene after strategy delay")
    if step.target == "hero_ready":
        from .hero_state import recognize_hero_state
        while True:
            _check(session, deployment_deadline)
            card = _match_card(frame, step.unit_id, "hero", source=step.source)
            state = recognize_hero_state(frame.screenshot_path, card["bbox"],
                                         baseline_resolution=session.config.game.baseline_resolution)
            receipt["hero_states"].append(state)
            _check(session, deployment_deadline)
            if state.get("deployed") is True and state.get("ability_ready") is True:
                break
            if time.monotonic() >= deadline:
                raise FlowError("Hero ability did not become verified ready before timeout")
            frame = _fresh(session, "strategy-hero-ready", receipt, deployment_deadline)
    elif step.target and frame.scene != step.target:
        _check(session, deployment_deadline)
        remaining = min(deadline, deployment_deadline) - time.monotonic()
        if remaining <= 0:
            raise FlowError("Strategy wait deadline exceeded")
        frame = session.wait_for({step.target}, timeout_sec=remaining,
                                 label="strategy-wait")
        receipt["evidence"].append(str(frame.screenshot_path))
        _check(session, deployment_deadline)
    _check(session, deployment_deadline)
    receipt["actions"].append({"action": "wait", "state": step.target or "delay",
                                "delay_sec": step.delay_sec, "frame": str(frame.screenshot_path),
                                "verified": True})
    return frame


def _target(step: StrategyStep, session) -> str:
    return (getattr(session.config.battle, "target_building", "") if step.target == "objective"
            else step.target)


def _preflight(session, scout: SceneSnapshot, steps: tuple[StrategyStep, ...],
               definition: StrategyDefinition, receipt: dict,
               deadline: float) -> SceneSnapshot:
    """Reject a missing card, building or boundary capability before game input."""
    needed = _needed_edges(steps)
    if needed:
        cached = scout.observations.get("terrain", [])
        has_terrain = (isinstance(cached, list) and cached and
                       all(isinstance(item, dict) and item.get("verified") is True and
                           item.get("frame") == str(scout.screenshot_path) for item in cached) and
                       _has_edges(cached, needed))
        has_reveal = getattr(session, "native", None) is not None
        if not has_terrain and not has_reveal:
            raise FlowError("Strategy has no verified boundary capability for all planned actions")
    # Check targets before any bar navigation, so an unsupported building
    # cannot be hidden behind a subsequent successful card search.
    for step in steps:
        if step.action == "cast_spell":
            target = _target(step, session)
            if not target:
                raise FlowError("Building objective is not configured")
            if target.startswith("relative:"):
                pass
            else:
                _building(scout, target)
    frame = scout
    seen = set()
    for step in steps:
        if step.action in {"deploy_troop", "deploy_hero", "deploy_siege", "activate_ability", "cast_spell"}:
            kind = {"deploy_troop": "troop", "deploy_hero": "hero", "deploy_siege": "siege",
                    "activate_ability": "hero", "cast_spell": "spell"}[step.action]
            if step.unit_id == "*":
                if definition.army_mode != "captured":
                    raise FlowError("Wildcard troop deployment requires captured-army mode")
                continue
            key = (step.unit_id, kind, step.source)
        elif step.action == "wait" and step.target == "hero_ready":
            key = (step.unit_id, "hero", step.source)
        else:
            continue
        if key in seen:
            continue
        frame, _ = _find_named_card(session, frame, *key, receipt, deadline)
        seen.add(key)
    return frame


def _verify_captured_stacks(frame: SceneSnapshot, original: tuple[dict, ...],
                            *, after_reveal: bool) -> None:
    """A terrain reveal must not silently shrink an independently counted army."""
    current = [card for card in _cards(frame) if card.get("kind") == "troop" and
               card.get("source", "army") in {"army", "event"} and
               type(card.get("count")) is int and card["count"] > 0]
    if len(current) != len(original):
        raise FlowError("Captured army stack missing after boundary reveal")
    matched = set()
    for before in original:
        candidates = [(index, card) for index, card in enumerate(current)
                      if index not in matched and card["count"] == before["count"] and
                      card.get("source", "army") == before.get("source", "army") and
                      abs(card["point"][0] - before["point"][0]) < 25 and
                      abs(card["point"][1] - before["point"][1]) < 25]
        if len(candidates) != 1:
            raise FlowError("Captured army stack identity or count changed after boundary reveal")
        index, card = candidates[0]
        matched.add(index)
        if after_reveal:
            evidence = card.get("evidence") or {}
            reading = evidence.get("count") or {}
            score = evidence.get("portrait_score")
            if (not isinstance(score, (int, float)) or not math.isfinite(score) or score < .92 or
                    reading.get("frame") != str(frame.screenshot_path) or
                    reading.get("count") != before["count"] or
                    not isinstance(reading.get("confidence"), (int, float)) or
                    reading["confidence"] < .9):
                raise FlowError("Captured army lacks fresh independent portrait and count proof")


def _end_battle_confirmation(frame: SceneSnapshot) -> tuple[int, int, int, int]:
    """A generic popup/OK button is never authority to surrender a battle."""
    if frame.scene != "popup" or not math.isfinite(frame.confidence) or frame.confidence < .8:
        raise FlowError("Expected a recognized end-battle confirmation dialog")
    text = []
    for item in frame.observations.get("ocr", []):
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        confidence = item.get("confidence")
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                not isinstance(confidence, (int, float)) or
                not math.isfinite(confidence) or confidence < .9):
            continue
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if 300 <= x <= 980 and 120 <= y <= 540:
            text.append(str(item.get("text", "")).replace(" ", "").casefold())
    joined = "".join(text)
    subject = any(phrase in joined for phrase in ("结束战斗", "放弃战斗", "投降",
                                                 "surrender", "endbattle", "endthebattle"))
    question = any(phrase in joined for phrase in ("确定", "确认", "是否", "吗", "?", "？",
                                                  "sure", "confirm"))
    if not subject or not question:
        raise FlowError("Popup has no positive surrender prompt evidence")
    buttons = [button for button in frame.observations.get("buttons", [])
               if button.get("name") == "confirm" and
               isinstance(button.get("point"), (tuple, list)) and
               len(button["point"]) == 2 and
               300 <= button["point"][0] <= 980 and 200 <= button["point"][1] <= 650]
    if len(buttons) != 1:
        raise FlowError("Surrender dialog has no unique central confirm button")
    return (300, 200, 980, 650)


def execute_strategy(session, scout: SceneSnapshot, definition: StrategyDefinition) -> SceneSnapshot:
    """Run ordered actions and attach a receipt understood by the battle flow.

    Input is issued only after current-frame scene, card, boundary, and target
    checks. All ambiguous action results fail with a partial receipt.
    """
    receipt = {"strategy": definition.id, "completed": False, "verified": False,
               "deployed_units": 0, "spells_used": 0, "offensive_actions": 0,
               "issued_placements": 0, "hero_abilities": 0, "siege_deployed": 0, "actions": [],
               "hero_states": [], "evidence": [str(scout.screenshot_path)]}
    frame = scout
    captured_stacks: tuple[dict, ...] = ()
    try:
        _scene(frame)
        session._validate_snapshot(frame)
        if definition.army_mode == "captured":
            manifest = frame.observations.get("expected_army_manifest")
            if not isinstance(manifest, dict) or manifest.get("complete") is not True:
                raise FlowError("Captured army has no independent complete manifest")
            observed = sorted(c["count"] for c in _cards(frame)
                              if c.get("kind") == "troop" and c.get("source", "army") in {"army", "event"}
                              and type(c.get("count")) is int)
            expected = sorted(c["count"] for c in manifest.get("troops", [])
                              if type(c.get("count")) is int and c["count"] > 0)
            if not expected or observed != expected:
                raise FlowError("Captured battle cards do not match the independent army manifest")
            captured_stacks = tuple(dict(card) for card in _cards(frame)
                                    if card.get("kind") == "troop" and
                                    card.get("source", "army") in {"army", "event"} and
                                    type(card.get("count")) is int and card["count"] > 0)
        slots = _cards(frame)
        context = BattleContext(
            troops=tuple(item for item in slots if item.get("kind") == "troop"),
            heroes=tuple(item for item in slots if item.get("kind") == "hero"),
            terrain=tuple(frame.observations.get("terrain", ())),
            spells=tuple(item for item in slots if item.get("kind") == "spell"),
            buildings=tuple(frame.observations.get("buildings", ())),
            hero_states=tuple(frame.observations.get("hero_states", ())),
            target_progress=dict(frame.observations.get("target_progress") or {}),
        )
        steps = planned_steps(definition, {"strategy_id": definition.id, **asdict(context)},
                              objective_target=getattr(session.config.battle, "target_building", ""))
        deadline = time.monotonic() + session.config.battle.deploy_timeout_sec
        frame = _preflight(session, frame, steps, definition, receipt, deadline)
        terrain = None
        needed_edges = _needed_edges(steps)
        destroyed = set()
        for index, step in enumerate(steps):
            _check(session, deadline)
            if step.action in {"deploy_troop", "deploy_hero", "deploy_siege"}:
                if terrain is None:
                    if definition.army_mode != "captured" and not _has_current_terrain(frame, needed_edges):
                        if step.action == "deploy_troop" and step.unit_id != "*":
                            frame, _ = _find_named_card(session, frame, step.unit_id, "troop",
                                                        step.source, receipt, deadline)
                        else:
                            frame = _find_reveal_troop(session, frame, receipt, deadline)
                    frame, terrain = _terrain(session, frame, receipt, needed_edges, deadline)
                    if definition.army_mode == "captured":
                        _verify_captured_stacks(frame, captured_stacks,
                                                after_reveal=frame.screenshot_path != scout.screenshot_path)
                kind = {"deploy_troop": "troop", "deploy_hero": "hero",
                        "deploy_siege": "siege"}[step.action]
                if step.unit_id == "*":
                    if definition.army_mode != "captured" or kind != "troop":
                        raise FlowError("Wildcard troop card requires captured-army mode")
                    cards = [c for c in _cards(frame) if c.get("kind") == "troop" and
                             (step.source == "auto" and c.get("source", "army") in {"army", "event"} or
                              c.get("source", "army") == step.source) and
                             type(c.get("count")) is int and c["count"] > 0]
                    if not cards:
                        raise FlowError("No visible troop cards for captured-army deployment")
                    # Bind each position and numeric stack independently. A
                    # missing selected card stops rather than shifting indices.
                    for initial in cards:
                        card = _match_card(frame, "*", "troop", source=step.source, prior=initial)
                        count = card["count"] if step.count == "all" else step.count
                        frame = _consume(session, frame, card, _points(terrain, step.edge, count),
                                         receipt, deadline, kind="troop")
                    continue
                frame, card = _find_named_card(session, frame, step.unit_id, kind,
                                               step.source, receipt, deadline)
                count = card["count"] if step.count == "all" else step.count
                points = _points(terrain, step.edge, count)
                if kind == "troop":
                    frame = _consume(session, frame, card, points, receipt, deadline, kind=kind)
                elif kind == "hero":
                    frame = _hero(session, frame, card, receipt, points[0], deadline, ability=False)
                else:
                    frame = _siege(session, frame, card, receipt, points[0], deadline)
            elif step.action == "activate_ability":
                frame, card = _find_named_card(session, frame, step.unit_id, "hero",
                                               step.source, receipt, deadline)
                frame = _hero(session, frame, card,
                              receipt, [], deadline, ability=True)
            elif step.action == "cast_spell":
                target = _target(step, session)
                if not target:
                    raise FlowError("Building objective is not configured")
                if target.startswith("relative:"):
                    if terrain is None:
                        if definition.army_mode != "captured" and not _has_current_terrain(frame, needed_edges):
                            frame = _find_reveal_troop(session, frame, receipt, deadline)
                        frame, terrain = _terrain(session, frame, receipt, needed_edges, deadline)
                        if definition.army_mode == "captured":
                            _verify_captured_stacks(frame, captured_stacks,
                                                    after_reveal=frame.screenshot_path != scout.screenshot_path)
                    for _ in range(step.count):
                        frame, card = _find_named_card(session, frame, step.unit_id, "spell",
                                                       step.source, receipt, deadline)
                        point = _relative_spell_point(terrain, target)
                        frame = _consume(session, frame, card, [point], receipt, deadline, kind="spell")
                    continue
                building = _building(frame, target)
                target_id = building["building_id"]
                event = {"action": "target_building", "target": target, "building_id": target_id,
                         "frame": str(frame.screenshot_path), "point": building["point"]}
                receipt["actions"].append(event)
                for _ in range(step.count):
                    # Target coordinates must remain positively identified in
                    # the current frame; smoke or absence is not destruction.
                    try:
                        building = _building(frame, target, building_id=target_id, state="destroyed")
                    except FlowError:
                        building = _building(frame, target, building_id=target_id)
                    if building["state"] == "destroyed":
                        destroyed.add(target_id)
                        break
                    frame, card = _find_named_card(session, frame, step.unit_id, "spell",
                                                   step.source, receipt, deadline)
                    # A swipe produced a new screenshot. Reconfirm the target
                    # and its point on that frame before selecting the spell.
                    building = _building(frame, target, building_id=target_id)
                    frame = _consume(session, frame, card, [building["point"]], receipt, deadline,
                                     kind="spell")
                    # A count decrement proves the spell cast, not damage. Require
                    # a fresh positive building state before another cast.
                    for _ in range(3):
                        try:
                            building = _building(frame, target, building_id=target_id, state="destroyed")
                            destroyed.add(target_id)
                            break
                        except FlowError:
                            try:
                                _building(frame, target, building_id=target_id)
                                break
                            except FlowError:
                                frame = _fresh(session, "strategy-building-state", receipt, deadline)
                    else:
                        raise FlowError("Building state unknown after spell; no extra cast issued")
                    if target_id in destroyed:
                        break
                receipt["target_building_id"] = target_id
                receipt["target_building_destroyed"] = target_id in destroyed
            elif step.action == "wait":
                if step.target == "hero_ready":
                    frame, _ = _find_named_card(session, frame, step.unit_id, "hero",
                                                step.source, receipt, deadline)
                frame = _wait_step(session, frame, step, receipt, deadline)
            elif step.action == "end_battle":
                if step.target == "target_destroyed" and not destroyed:
                    raise FlowError("Cannot end battle before verified building destruction")
                _scene(frame)
                _check(session, deadline)
                session.click(frame, "end_battle")
                receipt["actions"].append({"action": "end_battle", "frame": str(frame.screenshot_path),
                                            "target_destroyed": bool(destroyed)})
                _check(session, deadline)
                frame = session.wait_for({"popup", "settlement"},
                                         timeout_sec=min(step.timeout_sec, deadline - time.monotonic()),
                                         label="strategy-end-battle")
                receipt["evidence"].append(str(frame.screenshot_path))
                _check(session, deadline)
                if frame.scene == "popup":
                    region = _end_battle_confirmation(frame)
                    _check(session, deadline)
                    session.click(frame, "confirm", region=region)
                    _check(session, deadline)
                    frame = session.wait_for({"settlement"},
                                             timeout_sec=min(step.timeout_sec, deadline - time.monotonic()),
                                             label="strategy-end-settlement")
                    receipt["evidence"].append(str(frame.screenshot_path))
                    _check(session, deadline)
                if frame.scene != "settlement" or frame.confidence < .8:
                    raise FlowError("Early battle ending did not reach verified settlement")
                receipt["actions"][-1]["verified"] = True
            if frame.scene == "settlement" and index < len(steps) - 1:
                raise FlowError("Battle settled before remaining strategy actions")
        _check(session, deadline)
        if receipt["offensive_actions"] < 1 or not receipt["verified"]:
            raise FlowError("No offensive action has a verified receipt")
        receipt["completed"] = True
        frame.observations["deployment"] = receipt
        return frame
    except StopRequested:
        raise
    except Exception as exc:
        last = getattr(session, "last_snapshot", None)
        if last is not None and str(last.screenshot_path) not in receipt["evidence"]:
            receipt["evidence"].append(str(last.screenshot_path))
        raise DeploymentError(str(exc), partial_receipt=receipt) from exc
