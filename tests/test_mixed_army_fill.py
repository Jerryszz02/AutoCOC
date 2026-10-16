"""The Furnace filler may only add visible Archers before paid search."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from autococ.army_control import ensure_mixed_archer_army, _read_current
from autococ.combat import run_battle, _read_army
from autococ.config import BattleConfig
from autococ.errors import FlowError
from autococ.reporting import TaskResult
from autococ.scene import SceneSnapshot


CORE = {"electro_dragon": 10, "dragon": 1, "furnace": 1, "balloon": 1}


def current(name, *, archers=0, used=None, capacity=346, core=None, source="army"):
    troops = dict(CORE if core is None else core)
    if archers:
        troops["archer"] = archers
    rows = [{"kind": "troop", "source": source, "unit_id": unit_id, "count": count,
             "evidence": {"identity": {"unit_id": unit_id, "confidence": .99,
                                       "version_verified": True,
                                       "template_client_version": "18.647.2"}}}
            for unit_id, count in troops.items()]
    path = Path(name)
    ratio = f"{343 + archers if used is None else used}/{capacity}"
    capacity_read = {"text": ratio, "confidence": .999,
                     "bbox": [578, 156, 663, 180], "source": "full_ocr"}
    army = {
        "manifest": {"complete": True, "supported_layout": True,
                     "frame": str(path), "troops": rows,
                     "spells": [{"kind": "spell", "unit_id": "lightning_spell", "count": 6}]},
        "troops": {"used": 343 + archers if used is None else used, "capacity": capacity,
                   "meaning": "displayed_capacity_ratio", "evidence": capacity_read},
        "spells": {"used": 6, "capacity": 11},
        "heroes": {"used": 1, "capacity": 4}, "siege": {"used": 1, "capacity": 3},
        "identity_cards": [{"kind": "hero", "unit_id": "grand_warden", "count": 1},
                           {"kind": "siege", "unit_id": "siege_barracks", "count": 1}],
        "identity_coverage": {"hero": True, "siege": True},
        "hero_loadout_complete": True,
        "hero_loadout": {"grand_warden": {"pet_id": "owl",
                                          "equipment_ids": ["book", "gem"]}},
    }
    return SceneSnapshot("training", .99, path, {"army": army,
        "baseline_resolution": [1280, 720],
        "ocr": [{key: value for key, value in capacity_read.items() if key != "source"}],
        "buttons": [{"name": "attack", "point": [100, 500], "text": "attack"}],
        "army_editor": {"surface": "current", "ready": True,
                        "controls": [{"action": "open_picker", "kind": "troop",
                                      "point": [700, 150], "enabled": True,
                                      "cost_free": True, "confidence": .99}]}})


class DirectMixedSession:
    """Exercise the real picker preflight and one-input-at-a-time edit loop."""

    client_version = "18.647.2"

    def __init__(self, *, drift=None):
        self.surface = "current"
        self.archers = 0
        self.drift = drift
        self.closes = 0
        self.frames = 0
        self.increment_taps = 0
        self.taps = []

    def check_deadline(self):
        pass

    def observe(self, label):
        self.frames += 1
        capacity = 347 if self.drift == "capacity" and self.closes else 346
        shot = current(f"mixed-{self.frames}.png", archers=self.archers,
                       capacity=capacity)
        if self.drift == "low_ocr" and self.closes:
            shot.observations["army"]["troops"]["evidence"]["confidence"] = .8
        if self.drift == "mismatched_ocr" and self.closes:
            shot.observations["army"]["troops"]["evidence"]["text"] = "342/346"
        if self.surface == "picker":
            rows = shot.observations["army"]["manifest"]["troops"]
            shot.observations["army_editor"] = {
                "surface": "current_picker", "ready": True,
                "editing_kind": "troop", "complete_kinds": {"troop": True},
                "cards": [{"kind": "troop", "unit_id": row["unit_id"],
                           "count": row["count"]} for row in rows],
                "unit_availability": {"archer": {"available": True}},
                "unit_housing_space": {"electro_dragon": 30, "dragon": 20,
                                       "furnace": 18, "balloon": 5, "archer": 1},
                "controls": [
                    {"action": "increment", "unit_id": "archer", "point": [350, 620],
                     "enabled": True, "cost_free": True, "confidence": .99},
                    {"action": "close_picker", "point": [800, 320],
                     "enabled": True, "cost_free": True, "confidence": .99},
                ],
            }
        return shot

    def tap(self, shot, point, *, reason):
        self.taps.append((tuple(point), reason))
        if point == [700, 150]:
            self.surface = "picker"
        elif point == [800, 320]:
            self.surface = "current"
            self.closes += 1
        elif point == [350, 620]:
            self.archers += 1
            self.increment_taps += 1
        else:
            raise AssertionError(f"Unexpected tap: {point}")


class MixedArmyFillTests(unittest.TestCase):
    def session(self, *shots):
        session = Mock(client_version="18.647.2")
        session.observe.side_effect = shots
        return session

    def test_three_archers_are_the_only_desired_change_and_final_frame_is_full(self):
        before = current("before.png")
        final = current("final.png", archers=3)
        session = self.session(before, final)

        def direct(active, shot, observed, desired, loadout, evidence, *, before_mutation_guard):
            self.assertIs(active, session)
            self.assertIs(shot, before)
            self.assertEqual(desired["troop"], {**CORE, "archer": 3})
            self.assertEqual(desired["spell"], observed["spell"])
            self.assertEqual(desired["hero"], observed["hero"])
            self.assertEqual(desired["siege"], observed["siege"])
            self.assertEqual(loadout, before.observations["army"]["hero_loadout"])
            self.assertTrue(callable(before_mutation_guard))
            evidence.append(Path("edited.png"))
            return TaskResult("prepare_army", "succeeded", "edited", evidence=evidence,
                              metrics={"recipe_mutations": 3, "observed": desired})

        with patch("autococ.army_control._apply_direct", side_effect=direct) as apply:
            result = ensure_mixed_archer_army(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["archer_target"], 3)
        self.assertEqual(result.metrics["recipe_mutations"], 3)
        apply.assert_called_once()
        session.tap.assert_not_called()  # The direct editor owns each verified input.

    def test_unverified_or_mismatched_preflight_never_enters_editor(self):
        cases = [
            current("wrong-core.png", core={**CORE, "dragon": 2}),
            current("extra-unit.png", core={**CORE, "barbarian": 1}),
            current("wrong-source.png", source="clan"),
            current("bad-used.png", used=344),
            current("over-target.png", archers=4),
            current("over-nine.png", capacity=353),
        ]
        for shot in cases:
            with self.subTest(frame=shot.screenshot_path):
                session = self.session(shot)
                with patch("autococ.army_control._apply_direct") as apply:
                    result = ensure_mixed_archer_army(session)
                self.assertEqual(result.status, "not_supported")
                self.assertEqual(result.metrics["recipe_mutations"], 0)
                apply.assert_not_called()
                session.tap.assert_not_called()

    def test_full_core_army_keeps_zero_mutation_path(self):
        session = self.session(current("full.png", capacity=343))
        with patch("autococ.army_control._apply_direct") as apply:
            result = ensure_mixed_archer_army(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["recipe_mutations"], 0)
        apply.assert_not_called()

    def test_after_edit_stale_or_changed_frame_is_failure(self):
        for final in (current("edited.png", archers=3),
                      current("final.png", archers=2, used=345),
                      current("final.png", archers=3, capacity=347)):
            with self.subTest(final=final):
                session = self.session(current("before.png"), final)

                def direct(*args, **kwargs):
                    evidence = args[-1]
                    evidence.append(Path("edited.png"))
                    return TaskResult("prepare_army", "succeeded", "edited", evidence=evidence,
                                      metrics={"recipe_mutations": 3})

                with patch("autococ.army_control._apply_direct", side_effect=direct):
                    result = ensure_mixed_archer_army(session)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.metrics["recipe_mutations"], 3)

    def test_real_direct_editor_fills_only_three_archers(self):
        session = DirectMixedSession()
        result = ensure_mixed_archer_army(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(session.increment_taps, 3)
        self.assertEqual(result.metrics["recipe_mutations"], 3)
        self.assertEqual(result.metrics["archer_target"], 3)
        self.assertTrue(all(unit["unit_id"] == "archer"
                            for unit in result.metrics["mutations"]))

    def test_real_direct_editor_rechecks_capacity_evidence_before_any_increment(self):
        for defect in ("capacity", "low_ocr", "mismatched_ocr"):
            with self.subTest(defect=defect):
                session = DirectMixedSession(drift=defect)
                result = ensure_mixed_archer_army(session)
                self.assertEqual(result.status, "failed")
                self.assertIn("direct_pre_mutation_guard_failed", result.reason)
                self.assertEqual(session.increment_taps, 0)
                self.assertEqual(result.metrics["recipe_mutations"], 0)

    def test_battle_calls_fill_only_for_partial_opt_in_furnace_line(self):
        from test_combat import FakeSession, frame

        for tactic, capacity, expected_calls in (
                ("two_edge_furnace_mixed_probes", 346, 1),
                ("two_edge_furnace_mixed_probes", 343, 1),
                ("two_edge_probed_anchors", 346, 0)):
            with self.subTest(tactic=tactic, capacity=capacity):
                initial = current("combat-2.png", capacity=capacity)
                after = current("combat-3.png", archers=3 if capacity == 346 else 0,
                                capacity=capacity)
                shots = [frame(0, "village", resources={"gold": 1000, "elixir": 2000,
                                                         "dark_elixir": 300}),
                         frame(1, "search"), initial]
                if expected_calls:
                    shots.append(after)
                session = FakeSession(shots)
                session.client_version = "18.647.2"
                session.config.battle = BattleConfig(strategy="edrag_line",
                    line_execution_mode="continuous", line_tactics=tactic)
                observed, _ = _read_current(after, full=True)
                receipt = TaskResult("prepare_army", "succeeded", "filled",
                    evidence=[Path("fill.png")], metrics={
                        "archer_target": 3 if capacity == 346 else 0,
                        "observed": observed,
                        "verified_loadout": after.observations["army"]["hero_loadout"],
                        "final_frame": "fill.png"})
                with patch("autococ.combat._read_continuous_line_army",
                           side_effect=lambda active, shot, evidence: (shot, _read_army(shot))), \
                     patch("autococ.army_control.ensure_mixed_archer_army",
                           return_value=receipt) as fill, \
                     patch("autococ.combat._next_candidate",
                           side_effect=FlowError("bounded stop")):
                    result = run_battle(session)
                self.assertEqual(fill.call_count, expected_calls)
                self.assertEqual(result.metrics["search_count"], 0)
                if expected_calls:
                    self.assertIn((after.screenshot_path, "Recognized attack: attack"), session.taps)

    def test_false_full_weak_or_line_roi_capacity_never_attacks(self):
        from test_combat import FakeSession, frame

        for defect in ("weak", "line_roi"):
            with self.subTest(defect=defect):
                initial = current("combat-2.png", capacity=343)
                strict = current("combat-3.png", capacity=343)
                evidence = strict.observations["army"]["troops"]["evidence"]
                if defect == "weak":
                    evidence["confidence"] = .7
                else:
                    evidence["source"] = "line_roi"
                session = FakeSession([frame(0, "village", resources={"gold": 1000,
                    "elixir": 2000, "dark_elixir": 300}), frame(1, "search"), initial, strict])
                session.client_version = "18.647.2"
                session.config.battle = BattleConfig(strategy="edrag_line",
                    line_execution_mode="continuous", line_tactics="two_edge_furnace_mixed_probes")
                with patch("autococ.combat._read_continuous_line_army",
                           side_effect=lambda active, shot, evidence: (shot, _read_army(shot))):
                    result = run_battle(session)
                self.assertEqual(result.status, "failed")
                self.assertNotIn((initial.screenshot_path, "Recognized attack: attack"), session.taps)
                self.assertFalse(any("attack" in reason for path, reason in session.taps
                                     if path == strict.screenshot_path))

    def test_final_attack_frame_rejects_weak_ratio_and_full_group_drift(self):
        from test_combat import FakeSession, frame

        for defect in ("weak", "mismatch", "spell_drift", "loadout_drift"):
            with self.subTest(defect=defect):
                initial = current("combat-2.png", capacity=343)
                after = current("combat-3.png", capacity=343)
                if defect == "weak":
                    after.observations["army"]["troops"]["evidence"]["confidence"] = .7
                elif defect == "mismatch":
                    after.observations["army"]["troops"]["evidence"]["text"] = "342/343"
                elif defect == "spell_drift":
                    after.observations["army"]["manifest"]["spells"][0]["count"] = 5
                else:
                    after.observations["army"]["hero_loadout"]["grand_warden"]["pet_id"] = "fox"
                session = FakeSession([frame(0, "village", resources={"gold": 1000,
                    "elixir": 2000, "dark_elixir": 300}), frame(1, "search"), initial, after])
                session.client_version = "18.647.2"
                session.config.battle = BattleConfig(strategy="edrag_line",
                    line_execution_mode="continuous", line_tactics="two_edge_furnace_mixed_probes")
                observed, _ = _read_current(initial, full=True)
                receipt = TaskResult("prepare_army", "succeeded", "already full",
                    evidence=[Path("fill.png")], metrics={"archer_target": 0,
                        "observed": observed,
                        "verified_loadout": initial.observations["army"]["hero_loadout"],
                        "final_frame": "fill.png"})
                with patch("autococ.combat._read_continuous_line_army",
                           side_effect=lambda active, shot, evidence: (shot, _read_army(shot))), \
                     patch("autococ.army_control.ensure_mixed_archer_army",
                           return_value=receipt):
                    result = run_battle(session)
                self.assertEqual(result.status, "failed")
                self.assertFalse(any(path == after.screenshot_path and "attack" in reason
                                     for path, reason in session.taps))


if __name__ == "__main__":
    unittest.main()
