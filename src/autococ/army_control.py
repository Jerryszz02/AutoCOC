"""Evidence-gated army recipe application through the saved-plan editor.

The army UI changes the entire lineup when a saved plan is used. A recipe only
specifies which *kinds* to replace; every other kind is copied from the fresh
current-army observation. No mutation starts without a complete manifest of
all four kinds and editor coverage for every unit it would need to touch.
"""

from __future__ import annotations

from pathlib import Path
import time

from .army_editor import visual_fingerprint_matches
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
    if not isinstance(manifest, dict) or not manifest.get("complete"):
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


def _control(editor: dict, action: str, *, unit_id: str | None = None) -> dict | None:
    matches = [item for item in editor.get("controls", []) if item.get("action") == action
               and (unit_id is None or item.get("unit_id") == unit_id)
               and item.get("enabled") is True and item.get("cost_free") is True
               and isinstance(item.get("confidence"), (int, float))
               and item["confidence"] >= .9 and isinstance(item.get("point"), (list, tuple))
               and len(item["point"]) == 2 and all(type(n) is int for n in item["point"])]
    return matches[0] if len(matches) == 1 else None


def _observe_surface(session, wanted: str, evidence: list[Path], *, timeout_sec: float = 35):
    deadline = time.monotonic() + timeout_sec
    for index in range(5):
        session.check_deadline()
        snapshot = session.observe(f"army-{wanted}-{index}")
        evidence.append(snapshot.screenshot_path)
        editor = snapshot.observations.get("army_editor")
        if isinstance(editor, dict) and editor.get("surface") == wanted and editor.get("ready") is True:
            return snapshot
        if snapshot.scene in {"disconnected", "maintenance", "battle", "enemy_village"}:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    return None


def _observed_group(editor: dict, kind: str) -> dict[str, int] | None:
    complete = editor.get("complete_kinds")
    if not isinstance(complete, dict) or complete.get(kind) is not True:
        return None
    counts: dict[str, int] = {}
    for card in editor.get("cards", []):
        if card.get("kind") != kind:
            continue
        unit_id, count = card.get("unit_id"), card.get("count")
        if get_unit(unit_id) is None or type(count) is not int or count < 0 or unit_id in counts:
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
    army = snapshot.observations["army"]
    original_loadout = _hero_loadout(army, set(observed["hero"]))
    if original_loadout is None:
        return _result("not_supported", "hero_equipment_or_pet_observation_missing", evidence, actions=0)
    editor = snapshot.observations.get("army_editor")
    if not isinstance(editor, dict) or editor.get("surface") != "current" or editor.get("ready") is not True:
        return _result("not_supported", "current_army_navigation_unavailable", evidence, actions=0)
    control = _control(editor, "open_saved")
    if control is None:
        return _result("not_supported", "saved_plan_tab_unavailable", evidence, actions=0)
    session.tap(snapshot, control["point"], reason="Open observed saved army plans")
    saved = _observe_surface(session, "saved", evidence)
    if saved is None:
        return _result("failed", "saved_plan_page_unverified", evidence, actions=1)
    saved_editor = saved.observations["army_editor"]
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

    # We may only create a scratch plan when every desired unit has observed
    # picker/edit coverage. The current production samples do not yet cover
    # heroes/siege or all picker cards, so this correctly stops before creation.
    capabilities = saved_editor.get("editor_capabilities")
    needed = {unit_id for cards in (observed, desired) for group in cards.values()
              for unit_id in group}
    if not isinstance(capabilities, dict) or any(capabilities.get(unit_id) is not True for unit_id in needed):
        return _result("not_supported", "saved_plan_or_editor_samples_unavailable", evidence,
                       actions=1, needed=sorted(needed))
    if saved_editor.get("editor_preserves_hero_loadout") is not True:
        return _result("not_supported", "scratch_hero_loadout_preservation_unavailable", evidence, actions=1)
    create = _control(saved_editor, "create_preset")
    if create is None:
        return _result("not_supported", "empty_saved_plan_slot_unavailable", evidence, actions=1)
    session.tap(saved, create["point"], reason="Create empty scratch army plan")
    editing = _observe_surface(session, "edit", evidence)
    if editing is None:
        return _result("failed", "scratch_plan_creation_uncertain", evidence, actions=2)
    actions = 2
    # Editor controls are generated from identity templates, exact counts, and
    # the observed plus/minus icons. Removes precede adds to free capacity.
    for kind in _KINDS:
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
