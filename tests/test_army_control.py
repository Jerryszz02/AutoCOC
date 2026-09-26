from pathlib import Path
import unittest
from unittest.mock import Mock

from autococ.army_control import ArmyRecipe, ArmyRequirement, ensure_army
from autococ.scene import SceneSnapshot


def current(name="current.png", *, troops=None, spells=None, heroes=None, siege=None,
            complete=True, editor=True):
    troops = [{"unit_id": "meteor_golem", "kind": "troop", "count": 8}] if troops is None else troops
    spells = [{"unit_id": "freeze_spell", "kind": "spell", "count": 2}] if spells is None else spells
    heroes = [{"unit_id": "barbarian_king", "kind": "hero", "count": 1}] if heroes is None else heroes
    siege = [{"unit_id": "wall_wrecker", "kind": "siege", "count": 1}] if siege is None else siege
    observations = {"army": {"manifest": {"complete": complete, "troops": troops, "spells": spells},
                            "identity_cards": heroes + siege,
                            "identity_coverage": {"hero": True, "siege": True},
                            "hero_loadout_complete": True,
                            "hero_loadout": {"barbarian_king": {
                                "pet_id": "unicorn", "equipment_ids": ["gauntlet", "vial"]}}}}
    if editor:
        observations["army_editor"] = {"surface": "current", "ready": True, "controls": [
            {"action": "open_saved", "point": [640, 75], "enabled": True,
             "cost_free": True, "confidence": .99}]}
    return SceneSnapshot("training", .99, Path(name), observations)


def saved(*presets, capabilities=None):
    return SceneSnapshot("training", .99, Path("saved.png"), {"army_editor": {
        "surface": "saved", "ready": True, "presets": list(presets),
        "editor_capabilities": capabilities, "controls": []}})


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


class ArmyControlTests(unittest.TestCase):
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
        session.observe.side_effect = [current(), saved(preset(troop="dragon"))]
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
        session.observe.side_effect = [before, saved(matching), after]
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
        session.observe.side_effect = [current(), saved(preset()), verified]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["observed"]["troop"], {"meteor_golem": 8})
        self.assertEqual(result.metrics["observed"]["hero"], {"barbarian_king": 1})
        self.assertEqual(result.metrics["observed"]["siege"], {"wall_wrecker": 1})
        self.assertEqual(session.tap.call_count, 2)

    def test_uncertain_use_is_not_replayed(self):
        session = Mock()
        session.observe.side_effect = [current(), saved(preset())] + [
            SceneSnapshot("unknown", 0, Path(f"transition-{i}.png"), {}) for i in range(5)]
        result = ensure_army(session, ArmyRecipe((ArmyRequirement("lightning_spell", 4),)))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "army_use_result_uncertain")
        self.assertEqual(session.tap.call_count, 2)


if __name__ == "__main__":
    unittest.main()
