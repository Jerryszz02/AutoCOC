from pathlib import Path
import unittest
from unittest.mock import Mock

from autococ.army_control import ArmyRecipe, ArmyRequirement, ensure_army
from autococ.errors import StopRequested
from autococ.scene import SceneSnapshot


def current(name="current.png", *, troops=None, spells=None, heroes=None, siege=None,
            complete=True, editor=True):
    troops = [{"unit_id": "meteor_golem", "kind": "troop", "count": 8}] if troops is None else troops
    spells = [{"unit_id": "freeze_spell", "kind": "spell", "count": 2}] if spells is None else spells
    heroes = [{"unit_id": "barbarian_king", "kind": "hero", "count": 1}] if heroes is None else heroes
    siege = [{"unit_id": "wall_wrecker", "kind": "siege", "count": 1}] if siege is None else siege
    observations = {"army": {"manifest": {"complete": complete, "troops": troops, "spells": spells},
                            "troops": {"used": 320, "capacity": 335},
                            "spells": {"used": 2, "capacity": 11},
                            "heroes": {"used": 1, "capacity": 4},
                            "siege": {"used": 1, "capacity": 3},
                            "identity_cards": heroes + siege,
                            "identity_coverage": {"hero": True, "siege": True},
                            "hero_loadout_complete": True,
                            "hero_loadout": {"barbarian_king": {
                                "pet_id": "unicorn", "equipment_ids": ["gauntlet", "vial"]}}}}
    if editor:
        observations["army_editor"] = {"surface": "current", "ready": True, "controls": [
            {"action": "open_saved", "point": [640, 75], "enabled": True,
             "cost_free": True, "confidence": .99},
            {"action": "save_current", "point": [414, 74], "enabled": True,
             "cost_free": True, "confidence": .99}]}
    return SceneSnapshot("training", .99, Path(name), observations)


def saved(*presets, capabilities=None, availability=None, housing=None, inventory_complete=False):
    return SceneSnapshot("training", .99, Path("saved.png"), {"army_editor": {
        "surface": "saved", "ready": True, "presets": list(presets),
        "editor_capabilities": capabilities, "unit_availability": availability,
        "unit_housing_space": housing,
        "occupied_preset_ids_complete": inventory_complete,
        "controls": [{"action": "open_current", "point": [260, 75],
                      "enabled": True, "cost_free": True, "confidence": .99}]}})


def verified_saved(*presets):
    return saved(*presets, availability={"lightning_spell": {"available": True}},
                 housing={"freeze_spell": 1, "lightning_spell": 1})


def preset(troop="meteor_golem", spell="lightning_spell", *, complete=True, use_free=True):
    return {"complete": complete, "complete_kinds": {k: True for k in
            ("troop", "spell", "hero", "siege")}, "use_cost_free": use_free,
            "hero_loadout_complete": True,
            "hero_loadout": {"barbarian_king": {"pet_id": "unicorn",
                                               "equipment_ids": ["gauntlet", "vial"]}},
            "use_point": [1160, 185], "cards": [
                {"kind": "troop", "unit_id": troop, "count": 8},
                {"kind": "spell", "unit_id": spell, "count": 4},
                {"kind": "hero", "unit_id": "barbarian_king", "count": 1},
                {"kind": "siege", "unit_id": "wall_wrecker", "count": 1}]}


class DirectArmySession:
    def __init__(self, *, missing=None, uncertain_after=None, extra_change_after=None,
                 lightning_housing=1, full_capacity=False, untrusted_grey=False,
                 never_unblocks=False, grey_availability=None, stale_frames_after_tap=0,
                 stop_after_stale=False, stale_preflight_return_frames=0,
                 stale_post_return_frames=0, wrong_post_return_counts=False,
                 preserve_healing=False):
        self.counts = {"freeze_spell": 2, "lightning_spell": 1}
        if preserve_healing:
            self.counts["lightning_spell"] = 0
            self.counts["healing_spell"] = 1
        self.surface = "current"
        self.missing = missing
        self.uncertain_after = uncertain_after
        self.extra_change_after = extra_change_after
        self.lightning_housing = lightning_housing
        self.full_capacity = full_capacity
        self.untrusted_grey = untrusted_grey
        self.never_unblocks = never_unblocks
        self.grey_availability = grey_availability
        self.stale_frames_after_tap = stale_frames_after_tap
        self.stale_left = 0
        self.before_mutation = None
        self.stop_after_stale = stop_after_stale
        self.stop_requested = False
        self.stale_preflight_return_frames = stale_preflight_return_frames
        self.stale_post_return_frames = stale_post_return_frames
        self.wrong_post_return_counts = wrong_post_return_counts
        self.close_count = 0
        self.return_frames = 0
        self.mutations = 0
        self.taps = []
        self.frames = 0

    def check_deadline(self):
        if self.stop_requested:
            raise StopRequested("stopped in test")

    def observe(self, label):
        self.frames += 1
        if self.surface == "current":
            if self.close_count:
                self.return_frames += 1
            shown_counts = dict(self.counts)
            if self.wrong_post_return_counts and self.close_count == 2:
                shown_counts["lightning_spell"] += 1
            shot = current(f"direct-current-{self.frames}.png", spells=[
                {"unit_id": unit_id, "kind": "spell", "count": count}
                for unit_id, count in shown_counts.items() if count])
            if self.stale_preflight_return_frames or self.stale_post_return_frames:
                def icon(value):
                    return {"phash": value, "mean_bgr": [80, 100, 120],
                            "std_bgr": [25, 30, 35]}
                stale = ((self.close_count == 1 and
                          self.return_frames <= self.stale_preflight_return_frames) or
                         (self.close_count == 2 and
                          self.return_frames <= self.stale_post_return_frames))
                shot.observations["army"]["hero_loadout"] = {
                    "barbarian_king": {
                        "pet_visual": icon("ffffffffffffffff" if stale else "0000000000000000"),
                        "equipment_1_visual": icon("1111111111111111"),
                        "equipment_2_visual": icon("2222222222222222")}}
            shot.observations["army_editor"]["controls"].append({
                "action": "open_picker", "kind": "spell", "point": [100, 100],
                "enabled": True, "cost_free": True, "confidence": .99})
            if self.full_capacity:
                shot.observations["army"]["spells"] = {
                    "used": (self.counts["freeze_spell"] * 4 + self.counts["lightning_spell"]
                             + self.counts.get("healing_spell", 0)),
                    "capacity": 9}
            else:
                shot.observations["army"]["spells"]["used"] = sum(self.counts.values())
            return shot
        if self.uncertain_after == self.mutations and self.mutations:
            return SceneSnapshot("unknown", 0, Path("uncertain.png"), {})
        shown_counts = self.counts
        if self.stale_left:
            shown_counts = self.before_mutation
            self.stale_left -= 1
            if self.stop_after_stale:
                self.stop_requested = True
        controls = [{"action": action, "unit_id": unit_id, "point": point,
                     "enabled": True, "cost_free": True, "confidence": .99}
                    for action, unit_id, point in (
                        ("decrement", "freeze_spell", [300, 600]),
                        ("increment", "lightning_spell", [400, 600]))
                    if (action, unit_id) != self.missing]
        if self.full_capacity:
            blocked = (self.never_unblocks or
                       shown_counts["freeze_spell"] * 4 + shown_counts["lightning_spell"]
                       + shown_counts.get("healing_spell", 0) >= 9)
            for control in controls:
                if control["action"] == "increment":
                    control["enabled"] = not blocked
                    if blocked and not self.untrusted_grey:
                        control["capacity_blocked"] = True
        controls.append({"action": "close_picker", "point": [1200, 100],
                         "enabled": True, "cost_free": True, "confidence": .99})
        return SceneSnapshot("training", .99, Path(f"direct-picker-{self.frames}.png"), {
            "army_editor": {"surface": "current_picker", "ready": True,
                            "editing_kind": "spell", "complete_kinds": {"spell": True},
                            "cards": [{"kind": "spell", "unit_id": unit_id, "count": count}
                                      for unit_id, count in shown_counts.items()],
                            "unit_availability": (({"lightning_spell": self.grey_availability}
                                                  if self.grey_availability is not None else {})
                                                  if self.full_capacity else
                                                  {"lightning_spell": {"available": True}}),
                            "unit_housing_space": {"freeze_spell": 4 if self.full_capacity else 1,
                                                   "lightning_spell": self.lightning_housing},
                            "controls": controls}})

    def tap(self, snapshot, point, *, reason):
        self.taps.append(tuple(point))
        if point == [100, 100]:
            self.surface = "picker"
        elif point == [1200, 100]:
            self.surface = "current"
            self.close_count += 1
            self.return_frames = 0
        elif point == [300, 600]:
            self.before_mutation = dict(self.counts)
            self.counts["freeze_spell"] -= 1
            self.mutations += 1
            self.stale_left = self.stale_frames_after_tap
        elif point == [400, 600]:
            self.before_mutation = dict(self.counts)
            self.counts["lightning_spell"] += 1
            self.mutations += 1
            self.stale_left = self.stale_frames_after_tap
        if self.mutations and self.mutations == self.extra_change_after:
            self.counts["lightning_spell"] += 1


class ArmyControlTests(unittest.TestCase):
    def test_direct_edit_refuses_to_remove_the_last_selected_card(self):
        shot = current()
        shot.observations["army_editor"]["controls"].append({
            "action": "open_picker", "kind": "spell", "point": [100, 100],
            "enabled": True, "cost_free": True, "confidence": .99})
        session = Mock()
        session.observe.return_value = shot
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "direct_empty_picker_unverified"))
        self.assertEqual(result.metrics["recipe_mutations"], 0)
        session.tap.assert_not_called()

    def test_direct_edit_preflights_then_decrements_before_increments(self):
        session = DirectArmySession()
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.metrics["recipe_mutations"]), ("succeeded", 5), result.reason)
        self.assertEqual(session.taps, [(100, 100), (1200, 100), (100, 100),
                                        (300, 600), (300, 600),
                                        (400, 600), (400, 600), (400, 600),
                                        (1200, 100)])
        self.assertEqual(result.metrics["observed"]["spell"], {"lightning_spell": 4})

    def test_direct_preflight_return_retries_animated_hero_icon(self):
        session = DirectArmySession(stale_preflight_return_frames=1)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(session.close_count, 2)
        self.assertEqual(session.mutations, 5)

    def test_direct_postmutation_return_retries_animated_hero_icon(self):
        session = DirectArmySession(stale_post_return_frames=1)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(session.close_count, 2)
        self.assertEqual(session.return_frames, 2)
        self.assertEqual(session.mutations, 5)

    def test_direct_return_fails_after_three_unmatched_loadout_frames(self):
        session = DirectArmySession(stale_post_return_frames=3)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("failed", "direct_spell_return_mismatch"))
        self.assertEqual(result.metrics["return_state"], "loadout_unverified")
        self.assertEqual(session.return_frames, 3)
        self.assertEqual(session.mutations, 5)

    def test_direct_return_fails_immediately_on_complete_count_mismatch(self):
        session = DirectArmySession(wrong_post_return_counts=True)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("failed", "direct_spell_return_mismatch"))
        self.assertEqual(result.metrics["return_state"], "counts_mismatch")
        self.assertEqual(session.return_frames, 1)
        self.assertEqual(session.mutations, 5)

    def test_direct_edit_missing_target_control_never_mutates(self):
        session = DirectArmySession(missing=("increment", "lightning_spell"))
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "direct_spell_control_unavailable:lightning_spell"))
        self.assertEqual(session.mutations, 0)
        self.assertEqual(session.taps, [(100, 100), (1200, 100)])

    def test_uncertain_direct_edit_is_never_replayed(self):
        session = DirectArmySession(uncertain_after=1)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("failed", "direct_edit_result_uncertain"))
        self.assertEqual(session.mutations, 1)
        self.assertEqual(session.taps.count((300, 600)), 1)

    def test_direct_edit_waits_for_a_fresh_count_after_stale_frame(self):
        session = DirectArmySession(stale_frames_after_tap=1)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(session.mutations, 5)
        self.assertEqual(session.taps.count((300, 600)), 2)

    def test_direct_edit_stops_after_three_unchanged_frames(self):
        session = DirectArmySession(stale_frames_after_tap=10)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("failed", "direct_edit_result_uncertain"))
        self.assertEqual(result.metrics["observation_state"], "uncertain")
        self.assertEqual(session.mutations, 1)
        self.assertEqual(session.taps.count((300, 600)), 1)

    def test_direct_edit_honors_stop_during_reobservation(self):
        session = DirectArmySession(stale_frames_after_tap=1, stop_after_stale=True)
        with self.assertRaises(StopRequested):
            ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(session.mutations, 1)
        self.assertEqual(session.taps.count((300, 600)), 1)

    def test_direct_edit_rejects_other_count_change_after_tap(self):
        session = DirectArmySession(extra_change_after=1)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("failed", "direct_edit_result_uncertain"))
        self.assertEqual(session.mutations, 1)
        self.assertEqual(session.taps.count((300, 600)), 1)

    def test_direct_edit_checks_capacity_before_first_decrement(self):
        session = DirectArmySession(lightning_housing=4)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("skipped", "spell_capacity_exceeded:13>11"))
        self.assertEqual(session.mutations, 0)
        self.assertEqual(session.taps, [(100, 100), (1200, 100)])

    def test_full_capacity_grey_card_is_preflight_candidate_only(self):
        session = DirectArmySession(full_capacity=True)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(session.taps[:4], [(100, 100), (1200, 100),
                                            (100, 100), (300, 600)])
        self.assertEqual(session.mutations, 5)

    def test_untrusted_grey_card_cannot_start_removing_units(self):
        session = DirectArmySession(full_capacity=True, untrusted_grey=True)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "unit_availability_unverified:lightning_spell"))
        self.assertEqual(session.mutations, 0)

    def test_grey_target_absent_from_current_army_cannot_start_removing_units(self):
        session = DirectArmySession(full_capacity=True, preserve_healing=True)
        recipe = ArmyRecipe((ArmyRequirement("healing_spell", 1),
                             ArmyRequirement("lightning_spell", 4)))
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "unit_availability_unverified:lightning_spell"))
        self.assertEqual(result.metrics["recipe_mutations"], 0)
        self.assertEqual(session.taps, [(100, 100), (1200, 100)])

    def test_grey_card_must_become_enabled_before_increment(self):
        session = DirectArmySession(full_capacity=True, never_unblocks=True)
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason),
                         ("failed", "direct_increment_control_lost:lightning_spell"))
        self.assertEqual(session.mutations, 2)
        self.assertNotIn((400, 600), session.taps)

    def test_grey_card_does_not_override_explicit_locked_unit(self):
        session = DirectArmySession(full_capacity=True,
                                    grey_availability={"available": False, "reason": "locked"})
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("skipped", "unit_locked:lightning_spell"))
        self.assertEqual(session.mutations, 0)

    def test_relevant_manifest_coverage_allows_matching_recipe(self):
        shot = current(troops=[{"unit_id": None, "count": 8}])
        shot.observations["army"]["manifest"].update(
            complete=False, complete_kinds={"troop": False, "spell": True})
        session = Mock()
        session.observe.return_value = shot
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("freeze_spell", 2),)))
        self.assertEqual(result.status, "succeeded")
        session.tap.assert_not_called()

    def test_absent_optional_locked_unit_does_not_block_the_matching_recipe(self):
        session = Mock()
        shot = current()
        shot.observations["army_editor"]["unit_availability"] = {
            "lightning_spell": {"available": False, "reason": "locked"}}
        session.observe.return_value = shot
        recipe = ArmyRecipe((ArmyRequirement("freeze_spell", 2),
                             ArmyRequirement("lightning_spell", 4, optional=True)))
        result = ensure_army(session, recipe)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(result.metrics["observed"]["spell"], {"freeze_spell": 2})
        session.tap.assert_not_called()

    def test_matching_declared_kind_skips_switch_and_ignores_unknown_other_kind(self):
        session = Mock()
        session.observe.return_value = current(troops=[{"unit_id": None, "count": 8}])
        recipe = ArmyRecipe((ArmyRequirement("freeze_spell", 2),))
        result = ensure_army(session, recipe)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["actions"], 0)
        session.tap.assert_not_called()

    def test_unknown_declared_kind_never_mutates(self):
        session = Mock()
        session.observe.return_value = current(spells=[{"unit_id": None, "count": 2}])
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "not_supported")
        session.tap.assert_not_called()

    def test_partial_hero_coverage_cannot_satisfy_exact_hero_recipe(self):
        session = Mock()
        shot = current()
        shot.observations["army"]["identity_coverage"]["hero"] = False
        session.observe.return_value = shot
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("barbarian_king", 1),)))
        self.assertEqual(result.status, "not_supported")
        self.assertEqual(result.reason, "hero_or_siege_identity_coverage_missing")
        session.tap.assert_not_called()

    def test_changed_kind_requires_full_preserved_lineup_before_opening_saved(self):
        session = Mock()
        shot = current(heroes=[])
        shot.observations["army"]["identity_coverage"]["hero"] = False
        session.observe.return_value = shot
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.reason, "hero_or_siege_identity_coverage_missing")
        session.tap.assert_not_called()

    def test_missing_hero_equipment_or_pet_snapshot_prevents_saved_use(self):
        session = Mock()
        shot = current()
        shot.observations["army"]["hero_loadout_complete"] = False
        session.observe.return_value = shot
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.reason, "hero_equipment_or_pet_observation_missing")
        session.tap.assert_not_called()

    def test_saved_plan_must_preserve_unmentioned_troops_heroes_and_siege(self):
        session = Mock()
        session.observe.side_effect = [current(), verified_saved(preset(troop="dragon"))]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "not_supported")
        self.assertEqual(result.reason, "saved_plan_or_editor_samples_unavailable")
        self.assertEqual(session.tap.call_count, 1)  # Read-only saved-tab navigation only.

    def test_saved_plan_cannot_change_pet_or_equipment(self):
        session = Mock()
        changed = preset()
        changed["hero_loadout"]["barbarian_king"]["pet_id"] = "phoenix"
        session.observe.side_effect = [current(), saved(changed)]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "not_supported")
        self.assertEqual(session.tap.call_count, 1)

    def test_visual_loadout_fingerprint_can_verify_without_item_names(self):
        def icon(hash_value):
            return {"phash": hash_value, "mean_bgr": [80, 100, 120],
                    "std_bgr": [25, 30, 35]}

        loadout = {"barbarian_king": {"pet_visual": icon("1234567890abcdef"),
                                      "equipment_1_visual": icon("234567890abcdef1"),
                                      "equipment_2_visual": icon("34567890abcdef12")}}
        before = current()
        before.observations["army"]["hero_loadout"] = loadout
        after = current("after.png", spells=[
            {"unit_id": "lightning_spell", "kind": "spell", "count": 4}])
        after.observations["army"]["hero_loadout"] = loadout
        matching = preset()
        matching["hero_loadout"] = {"barbarian_king": {
            **loadout["barbarian_king"], "pet_visual": icon("1234567890abcdee")}}
        session = Mock()
        session.observe.side_effect = [before, verified_saved(matching), after]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(session.tap.call_count, 2)

        changed = preset()
        changed["hero_loadout"] = {"barbarian_king": {
            **loadout["barbarian_king"], "pet_visual": icon("ffffffffffffffff")}}
        session = Mock()
        session.observe.side_effect = [before, saved(changed)]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "not_supported")
        self.assertEqual(session.tap.call_count, 1)

    def test_fully_matching_saved_plan_is_used_and_checked(self):
        session = Mock()
        verified = current("verified.png", spells=[
            {"unit_id": "lightning_spell", "kind": "spell", "count": 4}])
        session.observe.side_effect = [current(), verified_saved(preset()), verified]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["observed"]["troop"], {"meteor_golem": 8})
        self.assertEqual(result.metrics["observed"]["hero"], {"barbarian_king": 1})
        self.assertEqual(result.metrics["observed"]["siege"], {"wall_wrecker": 1})
        self.assertEqual(session.tap.call_count, 2)

    def test_uncertain_use_is_not_replayed(self):
        session = Mock()
        session.observe.side_effect = [current(), verified_saved(preset())] + [
            SceneSnapshot("unknown", 0, Path(f"transition-{i}.png"), {}) for i in range(5)]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "army_use_result_uncertain")
        self.assertEqual(session.tap.call_count, 2)

    def test_matching_saved_plan_is_not_used_without_preflight_evidence(self):
        session = Mock()
        session.observe.side_effect = [current(), saved(preset())]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "unit_availability_unverified:lightning_spell"))
        self.assertEqual(session.tap.call_count, 1)

        session = Mock()
        session.observe.side_effect = [current(), saved(preset(),
            availability={"lightning_spell": {"available": False, "reason": "locked"}})]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual((result.status, result.reason), ("skipped", "unit_locked:lightning_spell"))
        self.assertEqual(result.metrics["recipe_mutations"], 0)
        self.assertEqual(result.metrics["actions"], 1)
        self.assertEqual(session.tap.call_count, 1)

    def test_current_exact_match_respects_explicit_unavailability(self):
        shot = current()
        shot.observations["army_editor"]["unit_availability"] = {
            "freeze_spell": {"available": False, "reason": "locked"}}
        session = Mock()
        session.observe.return_value = shot
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("freeze_spell", 2),)))
        self.assertEqual((result.status, result.reason), ("skipped", "unit_locked:freeze_spell"))
        self.assertEqual(result.metrics["recipe_mutations"], 0)
        session.tap.assert_not_called()

    def test_known_locked_and_capacity_shortage_skip_before_mutation(self):
        recipe = ArmyRecipe((ArmyRequirement("lightning_spell", 4),))
        capabilities = {"freeze_spell": True, "lightning_spell": True}
        session = Mock()
        session.observe.side_effect = [current(), saved(capabilities=capabilities,
            availability={"lightning_spell": {"available": False, "reason": "locked"}})]
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason), ("skipped", "unit_locked:lightning_spell"))
        self.assertEqual(session.tap.call_count, 1)  # Only opened the saved-plan page.

        session = Mock()
        session.observe.side_effect = [current(), saved(capabilities=capabilities,
            availability={"lightning_spell": {"available": True}},
            housing={"freeze_spell": 1, "lightning_spell": 3})]
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason), ("skipped", "spell_capacity_exceeded:12>11"))
        self.assertEqual(session.tap.call_count, 1)

    def test_unknown_availability_or_housing_blocks_copy(self):
        recipe = ArmyRecipe((ArmyRequirement("lightning_spell", 2),))
        capabilities = {"freeze_spell": True, "lightning_spell": True}
        session = Mock()
        session.observe.side_effect = [current(), saved(capabilities=capabilities)]
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "unit_availability_unverified:lightning_spell"))
        session = Mock()
        session.observe.side_effect = [current(), saved(capabilities=capabilities,
            availability={"lightning_spell": {"available": True}})]
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "spell_housing_unverified:freeze_spell"))

    def test_expired_event_troop_is_known_no_opportunity_before_copy(self):
        session = Mock()
        session.observe.side_effect = [current(), saved(capabilities={
            "meteor_golem": True, "event_super_pekka": True},
            availability={"event_super_pekka": {"available": False, "reason": "expired"}})]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("event_super_pekka", 1),)))
        self.assertEqual((result.status, result.reason),
                         ("skipped", "event_unit_expired:event_super_pekka"))
        self.assertEqual(session.tap.call_count, 1)

    def test_copy_current_never_uses_a_plan_without_full_postcopy_manifest(self):
        recipe = ArmyRecipe((ArmyRequirement("lightning_spell", 2),))
        capabilities = {"freeze_spell": True, "lightning_spell": True}
        dialog = SceneSnapshot("training", .99, Path("save-dialog.png"), {"army_editor": {
            "surface": "save_current", "ready": True,
            "controls": [{"action": "save_to_empty", "point": [1179, 372],
                          "enabled": True, "cost_free": True, "confidence": .99}]}})
        sequence = [current(), saved({"preset_id": "1", "complete": False},
                    capabilities=capabilities, inventory_complete=True,
                    availability={"lightning_spell": {"available": True}},
                    housing={"freeze_spell": 1, "lightning_spell": 1}),
                    current("source.png"), dialog, saved({"preset_id": "3", "complete": False})]
        session = Mock()
        session.observe.side_effect = sequence
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason),
                         ("failed", "copied_plan_full_manifest_unverified"))
        self.assertEqual(session.tap.call_count, 4)

    def test_copy_never_claims_an_old_matching_plan_as_the_new_slot(self):
        recipe = ArmyRecipe((ArmyRequirement("lightning_spell", 2),))
        dialog = SceneSnapshot("training", .99, Path("save-dialog.png"), {"army_editor": {
            "surface": "save_current", "ready": True,
            "controls": [{"action": "save_to_empty", "point": [1179, 372],
                          "enabled": True, "cost_free": True, "confidence": .99}]}})
        old = preset(spell="freeze_spell")
        old["preset_id"] = "1"
        next(card for card in old["cards"] if card["kind"] == "spell")["count"] = 2
        session = Mock()
        session.observe.side_effect = [
            current(), saved({"preset_id": "1", "complete": False},
                             capabilities={"freeze_spell": True, "lightning_spell": True},
                             availability={"lightning_spell": {"available": True}},
                             housing={"freeze_spell": 1, "lightning_spell": 1},
                             inventory_complete=True),
            current("source.png"), dialog, saved(old)]
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason),
                         ("failed", "copied_plan_full_manifest_unverified"))
        self.assertEqual(session.tap.call_count, 4)

    def test_copy_requires_complete_preexisting_plan_id_inventory(self):
        recipe = ArmyRecipe((ArmyRequirement("lightning_spell", 2),))
        session = Mock()
        session.observe.side_effect = [current(), saved({"preset_id": "1", "complete": False},
            capabilities={"freeze_spell": True, "lightning_spell": True},
            availability={"lightning_spell": {"available": True}},
            housing={"freeze_spell": 1, "lightning_spell": 1})]
        result = ensure_army(session, recipe)
        self.assertEqual((result.status, result.reason),
                         ("not_supported", "saved_plan_inventory_unverified"))
        self.assertEqual(session.tap.call_count, 1)


if __name__ == "__main__":
    unittest.main()
