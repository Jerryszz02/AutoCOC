"""Evidence-gated army recipe application through current or saved editors.

The army UI changes the entire lineup when a saved plan is used. A recipe only
specifies which *kinds* to replace; every other kind is copied from the fresh
current-army observation. No mutation starts without a complete manifest of
all four kinds and editor coverage for every unit it would need to touch.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import time
from typing import Callable

from .army_editor import visual_fingerprint_matches
from .errors import FlowError
from .reporting import TaskResult
from .unit_catalog import ArmyRecipe, ArmyRequirement, get_unit

_KINDS = ("troop", "spell", "hero", "siege")


def _result(status: str, reason: str, evidence: list[Path], **metrics) -> TaskResult:
    return TaskResult("prepare_army", status, reason, evidence=evidence, metrics=metrics)


def _read_current(snapshot, *, full: bool, relevant: set[str] | None = None) -> tuple[dict[str, dict[str, int]], str | None]:
    army = snapshot.observations.get("army")
    if not isinstance(army, dict):
        return {}, "army_observation_missing"
    manifest = army.get("manifest")
    if not isinstance(manifest, dict):
        return {}, "army_manifest_incomplete"
    needed = {"troop", "spell"} if full else ({"troop", "spell"} & (relevant or set()))
    complete_kinds = manifest.get("complete_kinds")
    if any(manifest.get("complete") is not True and
           (not isinstance(complete_kinds, dict) or complete_kinds.get(kind) is not True)
           for kind in needed):
        return {}, "army_manifest_incomplete"
    groups = {"troop": manifest.get("troops"), "spell": manifest.get("spells")}
    identities = army.get("identity_cards")
    coverage = army.get("identity_coverage")
    identity_kinds = {"hero", "siege"} if full else ({"hero", "siege"} & (relevant or set()))
    if identity_kinds and (not isinstance(identities, list) or not isinstance(coverage, dict)
                           or not all(coverage.get(kind) is True for kind in identity_kinds)):
        return {}, "hero_or_siege_identity_coverage_missing"
    if isinstance(identities, list):
        for kind in ("hero", "siege"):
            groups[kind] = [card for card in identities if card.get("kind") == kind
                            and card.get("source", "army") == "army"]
    result: dict[str, dict[str, int]] = {}
    for kind, cards in groups.items():
        if not full and relevant is not None and kind not in relevant:
            continue
        if not isinstance(cards, list):
            continue
        counts: dict[str, int] = {}
        for card in cards:
            unit_id, count = card.get("unit_id"), card.get("count")
            unit = get_unit(unit_id) if isinstance(unit_id, str) else None
            if unit is None or unit.kind != kind or type(count) is not int or count < 0:
                return {}, f"{kind}_identity_or_count_unknown"
            if unit_id in counts:
                return {}, f"duplicate_{kind}_identity"
            counts[unit_id] = count
        result[kind] = counts
    if full and set(result) != set(_KINDS):
        return {}, "full_army_manifest_incomplete"
    return result, None


def _desired(current: dict[str, dict[str, int]], recipe: ArmyRecipe) -> tuple[dict[str, dict[str, int]], str | None]:
    desired = {kind: dict(cards) for kind, cards in current.items()}
    by_kind: dict[str, list[ArmyRequirement]] = {}
    for item in recipe.units:
        by_kind.setdefault(get_unit(item.unit_id).kind, []).append(item)
    for kind, requirements in by_kind.items():
        if kind not in current:
            return {}, f"{kind}_observation_missing"
        desired[kind] = {item.unit_id: item.count for item in requirements
                         if not item.optional or item.unit_id in current[kind]}
    return desired, None


def _matches(observed: dict[str, dict[str, int]], desired: dict[str, dict[str, int]],
             *, kinds: set[str] | None = None) -> bool:
    for kind in kinds or set(_KINDS):
        if kind not in observed or kind not in desired:
            return False
        if {key: value for key, value in observed[kind].items() if value} != {
            key: value for key, value in desired[kind].items() if value}:
            return False
    return True


def _control(editor: dict, action: str, *, unit_id: str | None = None,
             kind: str | None = None, allow_capacity_blocked: bool = False) -> dict | None:
    matches = [item for item in editor.get("controls", []) if item.get("action") == action
               and (unit_id is None or item.get("unit_id") == unit_id)
               and (kind is None or item.get("kind") == kind)
               and (item.get("enabled") is True or (allow_capacity_blocked
                    and action == "increment" and item.get("enabled") is False
                    and item.get("capacity_blocked") is True))
               and item.get("cost_free") is True
               and isinstance(item.get("confidence"), (int, float))
               and item["confidence"] >= .9 and isinstance(item.get("point"), (list, tuple))
               and len(item["point"]) == 2 and all(type(n) is int for n in item["point"])]
    return matches[0] if len(matches) == 1 else None


def _observe_surface(session, wanted: str, evidence: list[Path], *, timeout_sec: float = 35,
                     require_ready: bool = True):
    deadline = time.monotonic() + timeout_sec
    for index in range(5):
        session.check_deadline()
        snapshot = session.observe(f"army-{wanted}-{index}")
        evidence.append(snapshot.screenshot_path)
        editor = snapshot.observations.get("army_editor")
        if (isinstance(editor, dict) and editor.get("surface") == wanted
                and (not require_ready or editor.get("ready") is True)):
            return snapshot
        if snapshot.scene in {"disconnected", "maintenance", "battle", "enemy_village"}:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return None


def _scroll_save_dialog(session, snapshot) -> None:
    """Reveal an empty save row inside an independently titled dialog."""
    editor = snapshot.observations.get("army_editor")
    if (not isinstance(editor, dict) or editor.get("surface") != "save_current"
            or snapshot is not session.last_snapshot):
        raise RuntimeError("Save-current dialog is not the fresh observed surface")
    session.check_deadline()
    session.event("swipe_save_current_dialog", frame=str(snapshot.screenshot_path),
                  start=[500, 620], end=[500, 180], reason="Reveal an unoccupied saved-plan slot")
    session.context.swipe(500, 620, 500, 180, 800)
    session.action_count += 1
    session.last_snapshot = None


def _observed_group(editor: dict, kind: str) -> dict[str, int] | None:
    complete = editor.get("complete_kinds")
    if not isinstance(complete, dict) or complete.get(kind) is not True:
        return None
    counts: dict[str, int] = {}
    if not isinstance(editor.get("cards"), list):
        return None
    for card in editor["cards"]:
        if not isinstance(card, dict):
            return None
        if card.get("kind") != kind:
            continue
        unit_id, count = card.get("unit_id"), card.get("count")
        unit = get_unit(unit_id) if isinstance(unit_id, str) else None
        if unit is None or unit.kind != kind or type(count) is not int or count < 0 or unit_id in counts:
            return None
        counts[unit_id] = count
    return counts


def _preset_manifest(preset: dict) -> dict[str, dict[str, int]] | None:
    if preset.get("complete") is not True:
        return None
    cards = preset.get("cards")
    if not isinstance(cards, list):
        return None
    observed = {kind: {} for kind in _KINDS}
    for card in cards:
        kind, unit_id, count = card.get("kind"), card.get("unit_id"), card.get("count")
        unit = get_unit(unit_id) if isinstance(unit_id, str) else None
        if kind not in observed or unit is None or unit.kind != kind or type(count) is not int or count < 0:
            return None
        if unit_id in observed[kind]:
            return None
        observed[kind][unit_id] = count
    return observed if preset.get("complete_kinds") == {kind: True for kind in _KINDS} else None


def _hero_loadout(army: dict, hero_ids: set[str]) -> dict | None:
    """Read pets and both equipment slots by hero ID, independent of portrait order."""
    if army.get("hero_loadout_complete") is not True:
        return None
    loadout = army.get("hero_loadout")
    if not isinstance(loadout, dict) or set(loadout) != hero_ids:
        return None
    for hero_id, state in loadout.items():
        if not isinstance(state, dict):
            return None
        semantic = (isinstance(state.get("pet_id"), str) and
                    isinstance(state.get("equipment_ids"), list) and
                    len(state["equipment_ids"]) == 2 and all(
                        isinstance(unit_id, str) and unit_id for unit_id in state["equipment_ids"]))
        visual = all(isinstance(state.get(name), dict) and
                     isinstance(state[name].get("phash"), str) for name in
                     ("pet_visual", "equipment_1_visual", "equipment_2_visual"))
        if not (semantic or visual):
            return None
    return loadout


def _army_capacity_counts(army: dict) -> dict[str, tuple[int, int]] | None:
    """Compare displayed capacities without volatile OCR confidence or boxes."""
    counts = {}
    for kind in ("troops", "spells", "heroes", "siege"):
        entry = army.get(kind)
        if (not isinstance(entry, dict) or type(entry.get("used")) is not int
                or type(entry.get("capacity")) is not int):
            return None
        counts[kind] = (entry["used"], entry["capacity"])
    return counts


def _reobserve_missing_loadout(session, snapshot, observed: dict[str, dict[str, int]],
                               evidence: list[Path]):
    """Recover a transient portrait frame without borrowing identity or icons."""
    original_army = snapshot.observations["army"]
    initial_capacities = _army_capacity_counts(original_army)
    if initial_capacities is None:
        return snapshot, observed, None, "army_capacity_unverified_during_loadout_recheck"
    last_path = snapshot.screenshot_path
    for attempt in range(1, 3):
        session.check_deadline()
        candidate = session.observe(f"army-loadout-recheck-{attempt}")
        evidence.append(candidate.screenshot_path)
        if candidate.screenshot_path == last_path:
            return snapshot, observed, None, "army_frame_not_fresh_during_loadout_recheck"
        last_path = candidate.screenshot_path
        editor = candidate.observations.get("army_editor")
        if (candidate.scene != "training" or not isinstance(editor, dict)
                or editor.get("surface") != "current" or editor.get("ready") is not True):
            return snapshot, observed, None, "army_page_unverified_during_loadout_recheck"
        counts, issue = _read_current(candidate, full=True)
        if issue is not None:
            return snapshot, observed, None, issue
        army = candidate.observations["army"]
        if counts != observed or _army_capacity_counts(army) != initial_capacities:
            return snapshot, observed, None, "army_changed_during_loadout_recheck"
        loadout = _hero_loadout(army, set(counts["hero"]))
        if loadout is not None:
            return candidate, counts, loadout, None
    return snapshot, observed, None, "hero_equipment_or_pet_observation_missing"


def _same_loadout(first: dict | None, second: dict | None) -> bool:
    if first is None or second is None or set(first) != set(second):
        return False
    for hero_id in first:
        a, b = first[hero_id], second[hero_id]
        if "pet_id" in a and "pet_id" in b:
            if a["pet_id"] != b["pet_id"] or a["equipment_ids"] != b["equipment_ids"]:
                return False
        elif all(name in a and name in b and visual_fingerprint_matches(a[name], b[name])
                 for name in ("pet_visual", "equipment_1_visual", "equipment_2_visual")):
            continue
        else:
            return False
    return True


def _known_no_opportunity(reason: str) -> bool:
    return reason.startswith(("event_unit_expired:", "unit_locked:")) or "_capacity_exceeded:" in reason


def _explicit_unavailable(editor: dict, unit_id: str) -> str | None:
    entries = editor.get("unit_availability")
    entry = entries.get(unit_id) if isinstance(entries, dict) else None
    if not isinstance(entry, dict) or entry.get("available") is not False:
        return None
    if entry.get("reason") == "expired" and get_unit(unit_id).source == "event":
        return f"event_unit_expired:{unit_id}"
    if entry.get("reason") == "locked":
        return f"unit_locked:{unit_id}"
    return None


def _preflight_change(snapshot, editor: dict, observed: dict[str, dict[str, int]],
                      desired: dict[str, dict[str, int]], *, known_only: bool = False,
                      deferred_availability: set[str] | None = None) -> str | None:
    """Reject a known impossible edit; require evidence for additions and space.

    Existing, already-matching troops need no unlock or housing observations.
    For a changed lineup, an absent unit must be visibly available in the
    picker. A current-picker capacity block may defer the availability check
    until space has been released; it never establishes availability itself.
    Unknown costs are not treated as zero.
    """
    army = snapshot.observations.get("army", {})
    availability = editor.get("unit_availability", {})
    costs = editor.get("unit_housing_space", {})
    for kind in _KINDS:
        before, after = observed[kind], desired[kind]
        delta = {unit_id: after.get(unit_id, 0) - before.get(unit_id, 0)
                 for unit_id in before.keys() | after.keys()}
        if not any(delta.values()):
            continue
        for unit_id, change in sorted(delta.items()):
            if change <= 0:
                continue
            blocked = _explicit_unavailable(editor, unit_id)
            if blocked is not None:
                return blocked
            entry = availability.get(unit_id) if isinstance(availability, dict) else None
            deferred = (unit_id in (deferred_availability or set()) and
                        (not isinstance(entry, dict) or entry.get("available") is not False))
            if (not isinstance(entry, dict) or entry.get("available") is not True) and not deferred:
                if not known_only:
                    return f"unit_availability_unverified:{unit_id}"
        capacity_kind = {"troop": "troops", "spell": "spells", "hero": "heroes", "siege": "siege"}[kind]
        capacity = army.get(capacity_kind) if isinstance(army, dict) else None
        if (not isinstance(capacity, dict) or type(capacity.get("used")) is not int
                or type(capacity.get("capacity")) is not int):
            if known_only:
                continue
            return f"{kind}_capacity_unverified"
        used, total = capacity["used"], capacity["capacity"]
        if not 0 <= used <= total:
            return f"{kind}_capacity_unverified"
        if kind in {"hero", "siege"}:
            projected = used + sum(delta.values())
        else:
            if sum(value for value in delta.values() if value > 0) == 0:
                continue  # Only removing units cannot overflow a confirmed capacity.
            projected = used
            for unit_id, change in sorted(delta.items()):
                if change == 0:
                    continue
                unit = get_unit(unit_id)
                cost = costs.get(unit_id) if isinstance(costs, dict) else None
                if type(cost) is not int:
                    cost = unit.housing_space
                if type(cost) is not int or cost <= 0:
                    if known_only:
                        projected = None
                        break
                    return f"{kind}_housing_unverified:{unit_id}"
                projected += change * cost
            if projected is None:
                continue
        if projected < 0:
            return f"{kind}_capacity_unverified"
        if projected > total:
            return f"{kind}_capacity_exceeded:{projected}>{total}"
    return None


def _positive_counts(group: dict[str, int]) -> dict[str, int]:
    return {unit_id: count for unit_id, count in group.items() if count}


def _direct_picker(snapshot, kind: str) -> dict | None:
    editor = snapshot.observations.get("army_editor")
    if (snapshot.scene != "training" or not isinstance(editor, dict)
            or editor.get("surface") != "current_picker" or editor.get("ready") is not True
            or editor.get("editing_kind") != kind):
        return None
    return editor


def _observe_direct_current_return(session, counts: dict[str, dict[str, int]],
                                   loadout: dict, evidence: list[Path]):
    """Allow animation to settle; never retry a clearly different full lineup."""
    deadline = time.monotonic() + 15
    last, state, actual = None, "current_unverified", None
    for attempt in range(3):
        session.check_deadline()
        if attempt and time.monotonic() >= deadline:
            break
        shot = session.observe(f"army-current-return-{attempt}")
        evidence.append(shot.screenshot_path)
        last = shot
        editor = shot.observations.get("army_editor")
        if (shot.scene == "training" and isinstance(editor, dict)
                and editor.get("surface") == "current" and editor.get("ready") is True):
            actual, issue = _read_current(shot, full=True)
            if issue is None:
                if not _matches(actual, counts):
                    return shot, "counts_mismatch", actual
                observed_loadout = _hero_loadout(shot.observations["army"], set(actual["hero"]))
                if _same_loadout(loadout, observed_loadout):
                    return shot, "matched", actual
                state = "loadout_unverified"
            else:
                state = issue
        if shot.scene in {"disconnected", "maintenance", "battle", "enemy_village"}:
            break
        if attempt < 2:
            session.check_deadline()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(.35, remaining))
    return last, state, actual


def _observe_direct_edit(session, kind: str, before: dict[str, int],
                         expected: dict[str, int], evidence: list[Path]):
    """Confirm one edit from at most three new frames without repeating its tap."""
    deadline = time.monotonic() + 15
    last = None
    for attempt in range(3):
        session.check_deadline()
        if attempt and time.monotonic() >= deadline:
            break
        shot = session.observe(f"army-current-picker-edit-{attempt}")
        evidence.append(shot.screenshot_path)
        editor = _direct_picker(shot, kind)
        actual = None if editor is None else _observed_group(editor, kind)
        last = (shot, editor, actual)
        if actual is not None:
            counts = _positive_counts(actual)
            if counts == _positive_counts(expected):
                return (*last, "matched")
            if counts != _positive_counts(before):
                return (*last, "mismatch")
        if shot.scene in {"disconnected", "maintenance", "battle", "enemy_village"}:
            break
        if attempt < 2:
            session.check_deadline()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(.35, remaining))
    return (*last, "uncertain") if last is not None else (None, None, None, "uncertain")


def _apply_direct(session, snapshot, observed: dict[str, dict[str, int]],
                  desired: dict[str, dict[str, int]], original_loadout: dict,
                  evidence: list[Path], *,
                  before_mutation_guard: Callable[[object], str | None] | None = None) -> TaskResult:
    """Preflight every changed picker, then mutate one observed count per tap."""
    changed = [kind for kind in ("troop", "spell")
               if not _matches(observed, desired, kinds={kind})]
    expected = {kind: dict(group) for kind, group in observed.items()}
    receipt: list[dict[str, object]] = []
    actions = 0
    total_edits = sum(abs(desired[kind].get(unit_id, 0) - observed[kind].get(unit_id, 0))
                      for kind in changed for unit_id in observed[kind].keys() | desired[kind].keys())
    if total_edits > 100:
        return _result("not_supported", "direct_edit_action_limit", evidence, actions=0,
                       recipe_mutations=0, required_mutations=total_edits)

    def outcome(status: str, reason: str, **extra) -> TaskResult:
        return _result(status, reason, evidence, actions=actions,
                       recipe_mutations=len(receipt), before=observed, desired=desired,
                       confirmed_counts=expected, mutations=receipt, **extra)

    # Current picker kind and completeness are anchored by selected cards.
    # Removing every original card leaves no independent kind anchor, so the
    # next count cannot be verified even when a target card is waiting below.
    if any(not any(count > 0 and desired[kind].get(unit_id, 0) > 0
                   for unit_id, count in observed[kind].items()) for kind in changed):
        return outcome("not_supported", "direct_empty_picker_unverified")

    # Navigation is read-only. Check all target kinds before making the first
    # decrement, so a missing target card cannot strand a partly edited army.
    for kind in changed:
        editor = snapshot.observations["army_editor"]
        control = _control(editor, "open_picker", kind=kind)
        if control is None:
            return outcome("not_supported", f"direct_open_picker_unavailable:{kind}")
        session.check_deadline()
        session.tap(snapshot, control["point"], reason=f"Inspect current {kind} picker")
        actions += 1
        picker = _observe_surface(session, "current_picker", evidence, timeout_sec=15)
        picker_editor = None if picker is None else _direct_picker(picker, kind)
        if picker_editor is None:
            return outcome("failed", f"direct_{kind}_picker_unverified")
        close = _control(picker_editor, "close_picker")

        def abort_preflight(status: str, reason: str) -> TaskResult:
            nonlocal actions, snapshot
            if close is None:
                return outcome("failed", f"direct_{kind}_close_unavailable",
                               preflight_reason=reason)
            session.check_deadline()
            session.tap(picker, close["point"], reason=f"Leave uneditable {kind} picker")
            actions += 1
            snapshot, return_state, actual = _observe_direct_current_return(
                session, expected, original_loadout, evidence)
            if return_state != "matched":
                return outcome("failed", "direct_preflight_return_unverified",
                               preflight_reason=reason, return_state=return_state,
                               return_counts=actual)
            return outcome(status, reason)

        group = _observed_group(picker_editor, kind)
        if group is None or _positive_counts(group) != _positive_counts(observed[kind]):
            return abort_preflight("not_supported", f"direct_{kind}_current_counts_unverified")
        partial = {item: dict(cards) for item, cards in observed.items()}
        partial[kind] = desired[kind]
        deferred_availability = set()
        for unit_id in desired[kind]:
            if desired[kind].get(unit_id, 0) <= observed[kind].get(unit_id, 0):
                continue
            candidate = _control(picker_editor, "increment", unit_id=unit_id,
                                 allow_capacity_blocked=True)
            if (observed[kind].get(unit_id, 0) > 0 and candidate is not None
                    and candidate.get("enabled") is False
                    and candidate.get("capacity_blocked") is True):
                deferred_availability.add(unit_id)
        preflight = _preflight_change(snapshot, picker_editor, observed, partial,
                                      deferred_availability=deferred_availability)
        if preflight is not None:
            return abort_preflight("skipped" if _known_no_opportunity(preflight) else "not_supported",
                                   preflight)
        for unit_id in observed[kind].keys() | desired[kind].keys():
            difference = desired[kind].get(unit_id, 0) - observed[kind].get(unit_id, 0)
            if difference and _control(picker_editor, "increment" if difference > 0 else "decrement",
                                       unit_id=unit_id, allow_capacity_blocked=difference > 0) is None:
                return abort_preflight("not_supported", f"direct_{kind}_control_unavailable:{unit_id}")
        if close is None:
            return outcome("failed", f"direct_{kind}_close_unavailable")
        session.check_deadline()
        session.tap(picker, close["point"], reason=f"Close inspected {kind} picker")
        actions += 1
        snapshot, return_state, actual = _observe_direct_current_return(
            session, expected, original_loadout, evidence)
        if return_state != "matched":
            return outcome("failed", "direct_preflight_return_unverified",
                           return_state=return_state, return_counts=actual)

    if before_mutation_guard is not None:
        issue = before_mutation_guard(snapshot)
        if issue is not None:
            return outcome("failed", f"direct_pre_mutation_guard_failed:{issue}")

    for kind in changed:
        opener = _control(snapshot.observations["army_editor"], "open_picker", kind=kind)
        if opener is None:
            return outcome("failed", f"direct_open_picker_lost:{kind}")
        session.check_deadline()
        session.tap(snapshot, opener["point"], reason=f"Edit current {kind} lineup")
        actions += 1
        picker = _observe_surface(session, "current_picker", evidence, timeout_sec=15)
        editor = None if picker is None else _direct_picker(picker, kind)
        group = None if editor is None else _observed_group(editor, kind)
        if group is None or _positive_counts(group) != _positive_counts(expected[kind]):
            return outcome("failed", f"direct_{kind}_reopen_unverified")
        for unit_id in sorted(expected[kind].keys() | desired[kind].keys(),
                              key=lambda uid: (desired[kind].get(uid, 0) >= expected[kind].get(uid, 0), uid)):
            target = desired[kind].get(unit_id, 0)
            while group.get(unit_id, 0) != target:
                direction = "decrement" if group.get(unit_id, 0) > target else "increment"
                if direction == "increment":
                    unavailable = _explicit_unavailable(editor, unit_id)
                    if unavailable is not None:
                        return outcome("failed", unavailable)
                control = _control(editor, direction, unit_id=unit_id)
                if control is None:
                    return outcome("failed", f"direct_{direction}_control_lost:{unit_id}")
                after_group = dict(group)
                after_group[unit_id] = group.get(unit_id, 0) + (-1 if direction == "decrement" else 1)
                session.check_deadline()
                session.tap(picker, control["point"], reason=f"Current army {unit_id} {direction}")
                actions += 1
                after, after_editor, actual, confirmation = _observe_direct_edit(
                    session, kind, group, after_group, evidence)
                if confirmation != "matched":
                    return outcome("failed", "direct_edit_result_uncertain", unit_id=unit_id,
                                   direction=direction, expected_count=after_group[unit_id],
                                   observed_count=None if actual is None else actual.get(unit_id, 0),
                                   observation_state=confirmation)
                receipt.append({"kind": kind, "unit_id": unit_id, "direction": direction,
                                "count": after_group[unit_id], "frame": str(after.screenshot_path)})
                expected[kind] = dict(actual)
                picker, editor, group = after, after_editor, actual
        if _positive_counts(group) != _positive_counts(desired[kind]):
            return outcome("failed", f"direct_{kind}_target_unverified")
        close = _control(editor, "close_picker")
        if close is None:
            return outcome("failed", f"direct_{kind}_close_lost")
        session.check_deadline()
        session.tap(picker, close["point"], reason=f"Close edited {kind} picker")
        actions += 1
        snapshot, return_state, actual = _observe_direct_current_return(
            session, expected, original_loadout, evidence)
        if return_state != "matched":
            return outcome("failed", f"direct_{kind}_return_mismatch",
                           return_state=return_state, return_counts=actual)
    if not _matches(expected, desired):
        return outcome("failed", "direct_full_recipe_unverified")
    final, _ = _read_current(snapshot, full=True)
    return outcome("succeeded", "Current army recipe applied and fully verified",
                   observed=final)


def _mixed_troop_capacity_evidence(snapshot, used: int, total: int) -> bool:
    """Bind the numeric ratio to this sampled current army observation."""
    if snapshot.observations.get("baseline_resolution") != [1280, 720]:
        return False
    army = snapshot.observations.get("army")
    capacity = army.get("troops") if isinstance(army, dict) else None
    if not isinstance(capacity, dict) or capacity.get("meaning") != "displayed_capacity_ratio":
        return False
    evidence = capacity.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("source") != "full_ocr":
        return False
    text = evidence.get("text")
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text) if isinstance(text, str) else None
    confidence, bbox = evidence.get("confidence"), evidence.get("bbox")
    if (match is None or (int(match[1]), int(match[2])) != (used, total)
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not .9 <= confidence <= 1
            or not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(type(n) is not int for n in bbox)
            or not 577 <= (bbox[0] + bbox[2]) / 2 <= 667
            or not 152 <= (bbox[1] + bbox[3]) / 2 <= 182):
        return False
    ocr = snapshot.observations.get("ocr")
    if not isinstance(ocr, list):
        return False
    strong = [item for item in ocr if isinstance(item, dict)
              and type(item.get("confidence")) in (int, float)
              and math.isfinite(item["confidence"]) and item["confidence"] >= .9
              and isinstance(item.get("bbox"), (list, tuple)) and len(item["bbox"]) == 4
              and all(type(n) is int for n in item["bbox"])
              and 577 <= (item["bbox"][0] + item["bbox"][2]) / 2 <= 667
              and 152 <= (item["bbox"][1] + item["bbox"][3]) / 2 <= 182
              and isinstance(item.get("text"), str)
              and re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", item["text"])]
    return (len(strong) == 1 and strong[0]["text"] == text
            and strong[0]["bbox"] == bbox and strong[0]["confidence"] == confidence)


def _inspect_mixed_archer_current(shot, client_version):
    """Validate the current four-card core, capacity evidence and loadout."""
    from .combat import _verified_army_identities
    from .mixed_roster import mixed_archer_count

    if client_version != "18.647.2":
        return None, "mixed_army_client_version_unverified"
    if shot.scene != "training":
        return None, "mixed_army_page_unverified"
    editor = shot.observations.get("army_editor")
    if (not isinstance(editor, dict) or editor.get("surface") != "current"
            or editor.get("ready") is not True):
        return None, "mixed_army_current_editor_unverified"
    observed, issue = _read_current(shot, full=True)
    if issue is not None:
        return None, issue
    army = shot.observations["army"]
    manifest = army.get("manifest")
    if (not isinstance(manifest, dict) or manifest.get("complete") is not True
            or manifest.get("supported_layout") is not True
            or manifest.get("frame") != str(shot.screenshot_path)):
        return None, "mixed_army_manifest_unverified"
    rows = manifest.get("troops")
    if not _verified_army_identities(rows, "troop", client_version):
        return None, "mixed_army_troop_identity_unverified"
    try:
        archer_count = mixed_archer_count(rows)
    except FlowError as exc:
        return None, f"mixed_army_roster_invalid:{exc}"
    capacity = army.get("troops")
    if (not isinstance(capacity, dict) or type(capacity.get("used")) is not int
            or type(capacity.get("capacity")) is not int):
        return None, "mixed_army_capacity_unverified"
    used, total = capacity["used"], capacity["capacity"]
    target = total - 343
    if (not 0 <= archer_count <= 9 or not 0 <= target <= 9
            or archer_count > target or used != 343 + archer_count
            or not 0 <= used <= total):
        return None, "mixed_army_capacity_or_count_mismatch"
    if not _mixed_troop_capacity_evidence(shot, used, total):
        return None, "mixed_army_capacity_evidence_unverified"
    loadout = _hero_loadout(army, set(observed["hero"]))
    if loadout is None:
        return None, "mixed_army_hero_loadout_unverified"
    return (observed, editor, loadout, archer_count, target), None


def _mixed_initial_identity_retry_signature(shot, client_version):
    """Identify a single transient hero gap without using an older hero label."""
    from .combat import _verified_army_identities
    from .mixed_roster import mixed_archer_count

    if (client_version != "18.647.2" or shot.scene != "training" or
            not math.isfinite(shot.confidence) or shot.confidence < .8):
        return None
    o = shot.observations
    editor, army = o.get("army_editor"), o.get("army")
    source_resolution = o.get("source_resolution")
    if (not isinstance(source_resolution, list) or len(source_resolution) != 2 or
            any(type(n) is not int or n <= 0 for n in source_resolution) or
            not isinstance(editor, dict) or editor.get("surface") != "current" or
            editor.get("ready") is not True or not isinstance(army, dict) or
            o.get("baseline_resolution") != [1280, 720] or
            o.get("account_in_use") not in (None, False)):
        return None
    manifest = army.get("manifest")
    if (not isinstance(manifest, dict) or manifest.get("complete") is not True or
            manifest.get("supported_layout") is not True or
            manifest.get("identity_complete") is not True or
            manifest.get("layout") != "my_army_rows_v1" or
            manifest.get("unknowns") != [] or
            manifest.get("frame") != str(shot.screenshot_path) or
            manifest.get("source_resolution") != source_resolution or
            manifest.get("baseline_resolution") != [1280, 720]):
        return None
    partial, issue = _read_current(shot, full=False, relevant={"troop", "spell"})
    if (issue is not None or set(partial) != {"troop", "spell"} or
            not _verified_army_identities(manifest.get("troops"), "troop", client_version) or
            not _verified_army_identities(manifest.get("spells"), "spell", client_version)):
        return None
    try:
        archers = mixed_archer_count(manifest["troops"])
    except FlowError:
        return None
    capacity = army.get("troops")
    if not isinstance(capacity, dict):
        return None
    used, total = capacity.get("used"), capacity.get("capacity")
    if (type(used) is not int or type(total) is not int or
            not 0 <= archers <= 9 or not 0 <= total - 343 <= 9 or
            archers > total - 343 or used != 343 + archers or
            not _mixed_troop_capacity_evidence(shot, used, total)):
        return None
    cards, coverage = army.get("identity_cards"), army.get("identity_coverage")
    hero_evidence, siege_evidence = army.get("hero_evidence"), army.get("siege_evidence")
    if (not isinstance(cards, list) or not isinstance(coverage, dict) or
            coverage.get("siege") is not True or
            not isinstance(hero_evidence, dict) or
            hero_evidence.get("frame") != str(shot.screenshot_path) or
            hero_evidence.get("layout") != "my_army_large_heroes_v1" or
            not isinstance(siege_evidence, dict) or
            siege_evidence.get("frame") != str(shot.screenshot_path) or
            siege_evidence.get("complete") is not True or
            siege_evidence.get("unknowns") != []):
        return None
    heroes = [card for card in cards if isinstance(card, dict) and card.get("kind") == "hero"]
    sieges = [card for card in cards if isinstance(card, dict) and card.get("kind") == "siege"]
    if (len(heroes) != 4 or hero_evidence.get("cards") != heroes or
            siege_evidence.get("cards") != sieges or
            not _verified_army_identities(sieges, "siege", client_version)
            or any(card.get("source") != "army" or card.get("count") != 1 or
                   not isinstance(card.get("bbox"), list) or len(card["bbox"]) != 4 or
                   any(type(n) is not int for n in card["bbox"]) for card in heroes)):
        return None
    slots = [tuple(card["bbox"]) for card in heroes]
    if len(set(slots)) != 4:
        return None
    known = [card for card in heroes if isinstance(card.get("unit_id"), str)]
    missing = [card for card in heroes if card.get("unit_id") is None]
    if (len(known) not in {3, 4} or len(missing) != 4 - len(known) or
            not _verified_army_identities(known, "hero", client_version)):
        return None
    if missing:
        if (coverage.get("hero") is not False or hero_evidence.get("complete") is not False or
                hero_evidence.get("unknowns") != ["hero_identity_unavailable"]):
            return None
    elif (coverage.get("hero") is not True or hero_evidence.get("complete") is not True or
          hero_evidence.get("unknowns") != []):
        return None
    return {"roster": partial, "sieges": tuple(sorted((card["unit_id"], card["count"])
                                                      for card in sieges)),
            "capacity": (used, total), "layout": (manifest["layout"],
                                                   tuple(o.get("source_resolution") or ())),
            "slots": tuple(sorted(slots)),
            "known": {tuple(card["bbox"]): card["unit_id"] for card in known},
            "missing": tuple(tuple(card["bbox"]) for card in missing)}


def mixed_archer_attack_guard(snapshot, prepared: TaskResult, client_version: str) -> str | None:
    """Require a fresh, unchanged and full army immediately before match input."""
    if (prepared.status != "succeeded" or
            str(snapshot.screenshot_path) == prepared.metrics.get("final_frame")):
        return "mixed_army_attack_frame_not_fresh"
    state, issue = _inspect_mixed_archer_current(snapshot, client_version)
    if issue is not None:
        return issue
    observed, _, loadout, archers, target = state
    reference = prepared.metrics
    if (archers != target or archers != reference.get("archer_target")
            or observed != reference.get("observed")
            or not _same_loadout(reference.get("verified_loadout"), loadout)):
        return "mixed_army_changed_before_attack"
    return None


def reobserve_mixed_archer_missing_loadout(session, snapshot, prepared: TaskResult,
                                            evidence: list[Path]):
    """Retry only a missing after-fill loadout with new, independently verified frames."""
    from .combat import _read_army, _verified_army_identities
    from .device import foreground_package, resolve_game_display

    issue = mixed_archer_attack_guard(snapshot, prepared, session.client_version)
    if issue != "mixed_army_hero_loadout_unverified" or prepared.metrics.get("recipe_mutations") != 0:
        return snapshot, issue
    observed, roster_issue = _read_current(snapshot, full=True)
    if roster_issue is not None or observed != prepared.metrics.get("observed"):
        return snapshot, "mixed_army_changed_before_attack"
    capacities = _army_capacity_counts(snapshot.observations["army"])
    if capacities is None:
        return snapshot, "mixed_army_loadout_retry_capacity_unverified"
    actions = getattr(session, "action_count", None)
    inputs = getattr(session, "_input_attempt_generation", None)
    captured = snapshot.observations.get("observed_at_monotonic")
    frame_id = snapshot.observations.get("frame_id")
    if (type(actions) is not int or type(inputs) is not int or
            type(captured) not in (int, float) or not math.isfinite(captured) or
            not 0 <= time.monotonic() - captured <= 8 or
            not isinstance(frame_id, str) or not frame_id or
            snapshot is not session.last_snapshot):
        return snapshot, "mixed_army_loadout_retry_initial_frame_unverified"
    try:
        with Path(snapshot.screenshot_path).open("rb") as source:
            original_sha = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return snapshot, "mixed_army_loadout_retry_initial_frame_unverified"
    paths, ids, hashes = {str(snapshot.screenshot_path)}, {frame_id}, {original_sha}
    capture = session.capture
    timeout = min(5, session.config.runtime.step_timeout_sec)
    source_path = Path(session.config.source_path).resolve()
    try:
        with source_path.open("rb") as source:
            source_sha = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError:
        return snapshot, "mixed_army_loadout_retry_source_unverified"

    def environment_ok():
        try:
            if Path(session.config.source_path).resolve() != source_path:
                return False
            with source_path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != source_sha:
                    return False
        except OSError:
            return False
        if (session.client_version != "18.647.2" or session.action_count != actions or
                session._input_attempt_generation != inputs or
                foreground_package(capture.adb, capture.serial, timeout_sec=timeout) !=
                session.config.game.package_name):
            return False
        display = resolve_game_display(capture.adb, capture.serial,
                                       session.config.game.package_name,
                                       display_id=session.config.game.display_id,
                                       timeout_sec=timeout)
        return (display.logical_id == capture.input_display_id and
                display.physical_id == capture.screenshot_display_id)

    for attempt in range(2):
        session.check_deadline()
        if not environment_ok():
            return snapshot, "mixed_army_loadout_retry_environment_changed"
        fresh = session.observe(f"battle-army-loadout-reread-{attempt + 1}")
        evidence.append(fresh.screenshot_path)
        session.check_deadline()
        next_captured = fresh.observations.get("observed_at_monotonic")
        next_id = fresh.observations.get("frame_id")
        if (fresh.scene == "account_in_use" or fresh is not session.last_snapshot or
                str(fresh.screenshot_path) in paths or
                not isinstance(next_id, str) or not next_id or next_id in ids or
                type(next_captured) not in (int, float) or not math.isfinite(next_captured) or
                next_captured <= captured or not 0 <= time.monotonic() - next_captured <= 8):
            return fresh, "mixed_army_loadout_retry_frame_not_fresh_or_interrupted"
        try:
            with Path(fresh.screenshot_path).open("rb") as source:
                frame_sha = hashlib.file_digest(source, "sha256").hexdigest()
        except OSError:
            return fresh, "mixed_army_loadout_retry_frame_not_fresh_or_interrupted"
        if frame_sha in hashes or not environment_ok():
            return fresh, "mixed_army_loadout_retry_frame_or_environment_changed"
        paths.add(str(fresh.screenshot_path))
        ids.add(next_id)
        hashes.add(frame_sha)
        captured = next_captured
        snapshot = fresh
        try:
            full_army = _read_army(fresh)
        except FlowError:
            return fresh, "mixed_army_loadout_retry_roster_unverified"
        manifest = full_army["recipe"]["manifest"]
        heroes = full_army["recipe"].get("hero_evidence")
        if (not full_army["troops_full"] or not isinstance(heroes, dict) or
                heroes.get("complete") is not True or heroes.get("unknowns") != [] or
                heroes.get("frame") != str(fresh.screenshot_path) or
                not _verified_army_identities(manifest.get("troops"), "troop", session.client_version) or
                not _verified_army_identities(heroes.get("cards"), "hero", session.client_version)):
            return fresh, "mixed_army_loadout_retry_roster_unverified"
        observed, roster_issue = _read_current(fresh, full=True)
        if (roster_issue is not None or observed != prepared.metrics.get("observed") or
                _army_capacity_counts(full_army["recipe"]) != capacities):
            return fresh, "mixed_army_changed_before_attack"
        issue = mixed_archer_attack_guard(fresh, prepared, session.client_version)
        if issue != "mixed_army_hero_loadout_unverified":
            return fresh, issue
    return snapshot, issue


def ensure_mixed_archer_army(session) -> TaskResult:
    """Fill only the observed free troop spaces of the sampled Furnace army.

    This route never opens saved plans and cannot replace another troop. The
    target count is derived from the current capacity, not an old screenshot.
    """
    evidence: list[Path] = []
    if getattr(session, "client_version", None) != "18.647.2":
        return _result("not_supported", "mixed_army_client_version_unverified", evidence,
                       recipe_mutations=0)
    session.check_deadline()
    snapshot = session.observe("mixed-army-preparation")
    evidence.append(snapshot.screenshot_path)

    state, issue = _inspect_mixed_archer_current(snapshot, session.client_version)
    if issue == "hero_or_siege_identity_coverage_missing":
        from .device import foreground_package

        baseline = _mixed_initial_identity_retry_signature(snapshot, session.client_version)
        captured = snapshot.observations.get("observed_at_monotonic")
        frame_id = snapshot.observations.get("frame_id")
        actions = getattr(session, "action_count", None)
        input_attempts = getattr(session, "_input_attempt_generation", None)
        if (baseline is not None and len(baseline["missing"]) == 1 and
                type(captured) in (int, float) and math.isfinite(captured) and
                0 <= time.monotonic() - captured <= 8 and
                isinstance(frame_id, str) and frame_id and type(actions) is int and
                type(input_attempts) is int and
                snapshot is session.last_snapshot):
            seen_paths = {str(snapshot.screenshot_path)}
            seen_ids = {frame_id}
            for attempt in range(2):  # Initial observation plus at most two fresh rereads.
                session.check_deadline()
                if (session.client_version != "18.647.2" or
                        session.action_count != actions or
                        session._input_attempt_generation != input_attempts or
                        foreground_package(session.capture.adb, session.capture.serial,
                            timeout_sec=min(5, session.config.runtime.step_timeout_sec)) !=
                        session.config.game.package_name):
                    return _result("not_supported", "mixed_army_retry_foreground_or_input_changed",
                                   evidence, recipe_mutations=0)
                fresh = session.observe(f"mixed-army-identity-reread-{attempt + 1}")
                evidence.append(fresh.screenshot_path)
                session.check_deadline()
                next_captured = fresh.observations.get("observed_at_monotonic")
                next_id = fresh.observations.get("frame_id")
                if (fresh.scene == "account_in_use" or
                        session.client_version != "18.647.2" or session.action_count != actions or
                        session._input_attempt_generation != input_attempts or
                        fresh is not session.last_snapshot or
                        str(fresh.screenshot_path) in seen_paths or
                        not isinstance(next_id, str) or not next_id or next_id in seen_ids or
                        type(next_captured) not in (int, float) or
                        not math.isfinite(next_captured) or next_captured <= captured or
                        not 0 <= time.monotonic() - next_captured <= 8):
                    return _result("not_supported", "mixed_army_retry_frame_not_fresh_or_interrupted",
                                   evidence, recipe_mutations=0)
                if foreground_package(session.capture.adb, session.capture.serial,
                                      timeout_sec=min(5, session.config.runtime.step_timeout_sec)) != \
                        session.config.game.package_name:
                    return _result("not_supported", "mixed_army_retry_foreground_changed",
                                   evidence, recipe_mutations=0)
                fresh_signature = _mixed_initial_identity_retry_signature(fresh, session.client_version)
                if (fresh_signature is None or
                        any(fresh_signature[key] != baseline[key] for key in
                            ("roster", "sieges", "capacity", "layout", "slots")) or
                        any(fresh_signature["known"].get(box) != unit_id for box, unit_id in
                            baseline["known"].items()) or
                        not set(fresh_signature["missing"]).issubset(baseline["missing"])):
                    return _result("not_supported", "mixed_army_retry_current_changed",
                                   evidence, recipe_mutations=0)
                seen_paths.add(str(fresh.screenshot_path))
                seen_ids.add(next_id)
                snapshot, captured, frame_id = fresh, next_captured, next_id
                state, issue = _inspect_mixed_archer_current(snapshot, session.client_version)
                if issue is None:
                    break
                if issue != "hero_or_siege_identity_coverage_missing":
                    return _result("not_supported", issue, evidence, recipe_mutations=0)
    if issue is not None:
        return _result("not_supported", issue, evidence, recipe_mutations=0)
    observed, editor, loadout, archer_count, target = state
    if archer_count == target:
        return _result("succeeded", "Mixed army already full", evidence,
                       observed=observed, archer_target=target, recipe_mutations=0,
                       verified_loadout=loadout, final_frame=str(snapshot.screenshot_path))
    if _control(editor, "open_picker", kind="troop") is None:
        return _result("not_supported", "mixed_army_direct_picker_unavailable", evidence,
                       recipe_mutations=0)
    desired = {kind: dict(group) for kind, group in observed.items()}
    desired["troop"]["archer"] = target
    recipe = ArmyRecipe(tuple(ArmyRequirement(unit_id, count)
                             for unit_id, count in desired["troop"].items()))
    built, issue = _desired(observed, recipe)
    if issue is not None or built != desired:
        return _result("not_supported", issue or "mixed_army_desired_mismatch", evidence,
                       recipe_mutations=0)
    def guard_before_mutation(current):
        if current.screenshot_path == snapshot.screenshot_path:
            return "mixed_army_pre_mutation_frame_not_fresh"
        refreshed, issue = _inspect_mixed_archer_current(current, session.client_version)
        if issue is not None:
            return issue
        if (refreshed[0] != observed or refreshed[3:] != (archer_count, target)
                or not _same_loadout(loadout, refreshed[2])):
            return "mixed_army_changed_during_picker_preflight"
        return None

    result = _apply_direct(session, snapshot, observed, desired, loadout, evidence,
                           before_mutation_guard=guard_before_mutation)
    if result.status != "succeeded":
        return result
    previous_frame = evidence[-1]
    session.check_deadline()
    final = session.observe("mixed-army-final-verification")
    evidence.append(final.screenshot_path)
    final_state, issue = _inspect_mixed_archer_current(final, session.client_version)
    if (final.screenshot_path in {snapshot.screenshot_path, previous_frame} or issue is not None
            or final_state is None or final_state[0] != desired
            or final_state[3:] != (target, target)
            or not _same_loadout(loadout, final_state[2])):
        return _result("failed", issue or "mixed_army_final_verification_failed", evidence,
                       **result.metrics)
    return _result("succeeded", "Mixed army filled and fully verified", evidence,
                   **result.metrics, archer_target=target, final_frame=str(final.screenshot_path),
                   verified_loadout=final_state[2])


def ensure_army(session, recipe: ArmyRecipe) -> TaskResult:
    """Match, then optionally use or create one fully observed saved plan.

    Any uncertainty *after* a mutation is a failure, never a skippable lack of
    support. A post-action observation may be repeated, but the action is not.
    """
    evidence: list[Path] = []
    session.check_deadline()
    snapshot = session.observe("army-preparation")
    evidence.append(snapshot.screenshot_path)
    if snapshot.scene == "village":
        if getattr(session, "client_version", None) == "18.647.2":
            from .army_navigation import open_current_army
            snapshot = open_current_army(session, snapshot, evidence)
            evidence.append(snapshot.screenshot_path)
        else:
            session.click_template(snapshot, "hud_army", roi=(0, 460, 230, 720))
        snapshot = _observe_surface(session, "current", evidence)
        if snapshot is None:
            return _result("failed", "army_page_unverified_after_navigation", evidence)
    if snapshot.scene != "training":
        return _result("failed", f"cannot_inspect_army_from:{snapshot.scene}", evidence)
    relevant = {get_unit(item.unit_id).kind for item in recipe.units}
    observed, issue = _read_current(snapshot, full=False, relevant=relevant)
    if issue:
        return _result("not_supported", issue, evidence)
    desired, issue = _desired(observed, recipe)
    if issue:
        return _result("not_supported", issue, evidence)
    current_editor = snapshot.observations.get("army_editor")
    current_army = snapshot.observations.get("army")
    for item in recipe.units:
        if item.optional and item.unit_id not in desired[get_unit(item.unit_id).kind]:
            continue
        for source in (current_editor, current_army):
            unavailable = _explicit_unavailable(source, item.unit_id) if isinstance(source, dict) else None
            if unavailable is not None:
                return _result("skipped", unavailable, evidence, actions=0, recipe_mutations=0)
    if _matches(observed, desired, kinds=relevant):
        return _result("succeeded", "Army recipe matched a fresh observation", evidence,
                       observed=observed, actions=0)

    # A saved plan replaces every category. Require a complete, named current
    # lineup before entering any mutating UI path, including a scratch preset.
    observed, issue = _read_current(snapshot, full=True)
    if issue:
        return _result("not_supported", issue, evidence, actions=0)
    desired, issue = _desired(observed, recipe)
    if issue:
        return _result("not_supported", issue, evidence, actions=0)
    editor = snapshot.observations.get("army_editor")
    if not isinstance(editor, dict) or editor.get("surface") != "current" or editor.get("ready") is not True:
        return _result("not_supported", "current_army_navigation_unavailable", evidence, actions=0)
    known_issue = _preflight_change(snapshot, editor, observed, desired, known_only=True)
    if known_issue is not None and _known_no_opportunity(known_issue):
        return _result("skipped", known_issue, evidence, actions=0, recipe_mutations=0)
    army = snapshot.observations["army"]
    original_loadout = _hero_loadout(army, set(observed["hero"]))
    if original_loadout is None:
        snapshot, observed, original_loadout, issue = _reobserve_missing_loadout(
            session, snapshot, observed, evidence)
        if issue is not None:
            return _result("not_supported", issue, evidence, actions=0)
        editor = snapshot.observations["army_editor"]
        desired, issue = _desired(observed, recipe)
        if issue is not None:
            return _result("not_supported", issue, evidence, actions=0)
        known_issue = _preflight_change(snapshot, editor, observed, desired, known_only=True)
        if known_issue is not None and _known_no_opportunity(known_issue):
            return _result("skipped", known_issue, evidence, actions=0, recipe_mutations=0)
    changed_kinds = {kind for kind in _KINDS if not _matches(observed, desired, kinds={kind})}
    if (changed_kinds <= {"troop", "spell"} and all(
            _control(editor, "open_picker", kind=kind) is not None for kind in changed_kinds)):
        return _apply_direct(session, snapshot, observed, desired, original_loadout, evidence)
    control = _control(editor, "open_saved")
    if control is None:
        return _result("not_supported", "saved_plan_tab_unavailable", evidence, actions=0)
    session.tap(snapshot, control["point"], reason="Open observed saved army plans")
    saved = _observe_surface(session, "saved", evidence)
    if saved is None:
        return _result("failed", "saved_plan_page_unverified", evidence, actions=1)
    saved_editor = saved.observations["army_editor"]
    preflight = _preflight_change(snapshot, saved_editor, observed, desired)
    if preflight is not None:
        if _known_no_opportunity(preflight):
            return _result("skipped", preflight, evidence, actions=1, recipe_mutations=0)
        return _result("not_supported", preflight, evidence, actions=1)
    for preset in saved_editor.get("presets", []):
        manifest = _preset_manifest(preset)
        if (manifest is None or not _matches(manifest, desired) or
                not _same_loadout(original_loadout, _hero_loadout(preset, set(desired["hero"])))):
            continue
        point = preset.get("use_point")
        if (preset.get("use_cost_free") is not True or not isinstance(point, (list, tuple))
                or len(point) != 2 or any(type(n) is not int for n in point)):
            continue
        session.tap(saved, point, reason="Use fully verified saved army recipe")
        verified = _observe_surface(session, "current", evidence)
        if verified is None:
            return _result("failed", "army_use_result_uncertain", evidence, actions=2)
        final, issue = _read_current(verified, full=True)
        final_loadout = None if issue else _hero_loadout(verified.observations["army"], set(final["hero"]))
        if issue or not _matches(final, desired) or not _same_loadout(original_loadout, final_loadout):
            return _result("failed", "army_use_result_mismatch", evidence, actions=2,
                           issue=issue, observed=final,
                           hero_loadout_verified=_same_loadout(original_loadout, final_loadout))
        return _result("succeeded", "Saved army recipe applied and fully verified", evidence,
                       actions=2, observed=final)

    # A new plan starts as an exact copy of the observed current army. This is
    # the verified game UI's Save current army action, so untouched troop kinds
    # and hero loadouts need not be reconstructed from empty picker slots.
    # Picker/edit coverage is required only for kinds the recipe changes.
    capabilities = saved_editor.get("editor_capabilities")
    changed_kinds = {kind for kind in _KINDS if observed[kind] != desired[kind]}
    needed = {unit_id for cards in (observed, desired) for kind in changed_kinds
              for unit_id in cards[kind]}
    if not isinstance(capabilities, dict) or any(capabilities.get(unit_id) is not True for unit_id in needed):
        return _result("not_supported", "saved_plan_or_editor_samples_unavailable", evidence,
                       actions=1, needed=sorted(needed))
    if saved_editor.get("occupied_preset_ids_complete") is not True:
        return _result("not_supported", "saved_plan_inventory_unverified", evidence, actions=1)
    before_ids = {item.get("preset_id") for item in saved_editor.get("presets", [])}
    if None in before_ids or not before_ids:
        return _result("not_supported", "saved_plan_inventory_unverified", evidence, actions=1)
    open_current = _control(saved_editor, "open_current")
    if open_current is None:
        return _result("not_supported", "copy_current_navigation_unavailable", evidence, actions=1)
    session.tap(saved, open_current["point"], reason="Open current army before copying it")
    copy_source = _observe_surface(session, "current", evidence)
    if copy_source is None:
        return _result("failed", "copy_current_source_unverified", evidence, actions=2)
    source_counts, issue = _read_current(copy_source, full=True)
    source_loadout = None if issue else _hero_loadout(
        copy_source.observations["army"], set(source_counts["hero"]))
    if issue or not _matches(source_counts, observed) or not _same_loadout(original_loadout, source_loadout):
        return _result("failed", "copy_current_source_changed", evidence, actions=2,
                       issue=issue)
    copy_control = _control(copy_source.observations.get("army_editor", {}), "save_current")
    if copy_control is None:
        return _result("not_supported", "copy_current_control_unavailable", evidence, actions=2)
    session.tap(copy_source, copy_control["point"], reason="Open observed Save current army dialog")
    dialog = _observe_surface(session, "save_current", evidence, require_ready=False)
    if dialog is None:
        return _result("failed", "copy_current_dialog_unverified", evidence, actions=3)
    if dialog.observations["army_editor"].get("ready") is not True:
        _scroll_save_dialog(session, dialog)
        dialog = _observe_surface(session, "save_current", evidence, timeout_sec=20)
    if dialog is None:
        return _result("not_supported", "empty_saved_plan_slot_unavailable", evidence, actions=4)
    save_copy = _control(dialog.observations["army_editor"], "save_to_empty")
    if dialog.scene != "training" or save_copy is None:
        return _result("not_supported", "empty_saved_plan_slot_unverified", evidence, actions=4)
    session.tap(dialog, save_copy["point"], reason="Copy current army into uniquely verified empty plan")
    copied = _observe_surface(session, "saved", evidence)
    if copied is None:
        return _result("failed", "copied_plan_result_uncertain", evidence, actions=5)
    copied_candidates = []
    for item in copied.observations["army_editor"].get("presets", []):
        if item.get("preset_id") in before_ids:
            continue
        manifest = _preset_manifest(item)
        loadout = _hero_loadout(item, set(observed["hero"]))
        if manifest is not None and _matches(manifest, observed) and _same_loadout(original_loadout, loadout):
            copied_candidates.append(item)
    if len(copied_candidates) != 1 or not copied_candidates[0].get("preset_id"):
        return _result("failed", "copied_plan_full_manifest_unverified", evidence, actions=5)
    copied_preset = copied_candidates[0]
    edit_point = copied_preset.get("edit_point")
    if not isinstance(edit_point, (list, tuple)) or len(edit_point) != 2:
        return _result("failed", "copied_plan_edit_control_unverified", evidence, actions=5)
    session.tap(copied, edit_point, reason="Edit fully verified copy of current army")
    editing = _observe_surface(session, "edit", evidence)
    if editing is None:
        return _result("failed", "copied_plan_editor_unverified", evidence, actions=6)
    if editing.observations["army_editor"].get("preset_id") != copied_preset["preset_id"]:
        return _result("failed", "copied_plan_editor_id_mismatch", evidence, actions=6)
    actions = 6
    # Editor controls are generated from identity templates, exact counts, and
    # the observed plus/minus icons. Removes precede adds to free capacity.
    for kind in changed_kinds:
        target = desired[kind]
        for _ in range(100):
            current_group = _observed_group(editing.observations["army_editor"], kind)
            if current_group is None:
                return _result("failed", f"scratch_{kind}_manifest_unverified", evidence, actions=actions)
            differences = {unit_id: target.get(unit_id, 0) - current_group.get(unit_id, 0)
                           for unit_id in set(target) | set(current_group)}
            pending = [(unit_id, delta) for unit_id, delta in differences.items() if delta]
            if not pending:
                break
            unit_id, delta = sorted(pending, key=lambda entry: (entry[1] > 0, entry[0]))[0]
            direction = "increment" if delta > 0 else "decrement"
            edit_control = _control(editing.observations["army_editor"], direction, unit_id=unit_id)
            if edit_control is None:
                return _result("failed", f"scratch_{direction}_control_unavailable:{unit_id}", evidence,
                               actions=actions)
            session.tap(editing, edit_control["point"], reason=f"Edit observed {unit_id} {direction}")
            actions += 1
            after = _observe_surface(session, "edit", evidence, timeout_sec=15)
            if after is None:
                return _result("failed", "scratch_edit_result_uncertain", evidence, actions=actions)
            after_group = _observed_group(after.observations["army_editor"], kind)
            expected = current_group.get(unit_id, 0) + (1 if delta > 0 else -1)
            if after_group is None or after_group.get(unit_id, 0) != expected:
                return _result("failed", "scratch_edit_result_unverified", evidence,
                               actions=actions, unit_id=unit_id, expected=expected)
            editing = after
        else:
            return _result("failed", "scratch_edit_action_limit", evidence, actions=actions)
    edit_observation = editing.observations["army_editor"]
    if any(_observed_group(edit_observation, kind) != desired[kind] for kind in _KINDS):
        return _result("failed", "scratch_full_recipe_unverified", evidence, actions=actions)
    if not _same_loadout(original_loadout, _hero_loadout(edit_observation, set(desired["hero"]))):
        return _result("failed", "scratch_hero_loadout_unverified", evidence, actions=actions)
    save = _control(edit_observation, "save_preset")
    if save is None:
        return _result("failed", "scratch_save_control_unavailable", evidence, actions=actions)
    session.tap(editing, save["point"], reason="Save fully checked scratch army plan")
    actions += 1
    saved = _observe_surface(session, "saved", evidence)
    if saved is None:
        return _result("failed", "scratch_save_result_uncertain", evidence, actions=actions)
    for preset in saved.observations["army_editor"].get("presets", []):
        if preset.get("preset_id") != copied_preset["preset_id"]:
            continue
        manifest = _preset_manifest(preset)
        if (manifest is None or not _matches(manifest, desired) or preset.get("use_cost_free") is not True
                or not _same_loadout(original_loadout, _hero_loadout(preset, set(desired["hero"])))):
            continue
        point = preset.get("use_point")
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(n) is not int for n in point):
            continue
        session.tap(saved, point, reason="Apply fully checked scratch army plan")
        actions += 1
        verified = _observe_surface(session, "current", evidence)
        if verified is None:
            return _result("failed", "scratch_use_result_uncertain", evidence, actions=actions)
        final, issue = _read_current(verified, full=True)
        final_loadout = None if issue else _hero_loadout(verified.observations["army"], set(final["hero"]))
        if issue or not _matches(final, desired) or not _same_loadout(original_loadout, final_loadout):
            return _result("failed", "scratch_use_result_mismatch", evidence, actions=actions,
                           issue=issue, observed=final,
                           hero_loadout_verified=_same_loadout(original_loadout, final_loadout))
        return _result("succeeded", "Army recipe applied and fully verified", evidence,
                       actions=actions, observed=final)
    return _result("failed", "saved_scratch_plan_unverified", evidence, actions=actions)
