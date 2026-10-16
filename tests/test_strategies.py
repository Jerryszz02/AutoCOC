from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autococ.config import BattleConfig, load_config
from autococ.desktop import RunOptions, desktop_config, load_options, save_options
from autococ.errors import FlowError
from autococ.strategies import BattleContext, STRATEGIES, TwoEdgeSplitStrategy


def troop(x, count, source="army"):
    return {"point": [x, 650], "kind": "troop", "count": count, "source": source}


class StrategyTests(unittest.TestCase):
    def context(self):
        return BattleContext((troop(200, 3), troop(400, 40, "event"), troop(100, 1)),
                             ({"point": [500, 650], "kind": "hero"},),
                             tuple({"edge": edge, "point": point} for edge, points in
                                   ((0, ((500, 100), (300, 260))), (1, ((270, 440), (400, 550)))) for point in points))

    def test_event_first_then_left_to_right_even_split_and_no_heroes(self):
        plan = STRATEGIES["two_edge"].planner.build_plan(self.context())
        self.assertEqual([step.card["count"] for step in plan.troops], [40, 1, 3])
        self.assertEqual(sum(y < 300 for x, y in plan.troops[0].points), 20)
        self.assertEqual(sum(y < 300 for x, y in plan.troops[2].points), 2)
        self.assertEqual(plan.heroes, ())

    def test_single_edge_uses_bar_order_then_heroes_at_midpoint(self):
        plan = STRATEGIES["edrag_line"].planner.build_plan(self.context())
        self.assertEqual([step.card["count"] for step in plan.troops], [1, 3, 40])
        self.assertTrue(all(y <= 260 for step in plan.troops for x, y in step.points))
        self.assertEqual(plan.heroes[0].points, ((400, 180),))

    def test_explicit_two_edge_split_preserves_counts_and_uses_both_verified_edges(self):
        counts = {"electro_dragon": 10, "dragon": 1, "baby_dragon": 1,
                  "balloon": 2, "archer": 3}
        troops = tuple({"unit_id": unit_id, "count": count, "point": [100 + index * 90, 650]}
                       for index, (unit_id, count) in enumerate(counts.items()))
        hero_ids = ("grand_warden", "dragon_duke", "minion_prince", "archer_queen")
        heroes = tuple({"unit_id": unit_id, "point": [650 + index * 90, 650]}
                      for index, unit_id in enumerate(hero_ids))
        terrain = tuple({"edge": edge, "point": point} for edge, points in (
            (0, ((454, 194), (404, 232))), (1, ((150, 447), (198, 480)))) for point in points)
        plan = TwoEdgeSplitStrategy().build_plan(BattleContext(troops, heroes, terrain))
        self.assertEqual([item.card["unit_id"] for item in plan.troops],
                         ["baby_dragon", "dragon", "balloon", "electro_dragon", "archer"])
        self.assertEqual({item.card["unit_id"]: len(item.points) for item in plan.troops}, counts)
        by_id = {item.card["unit_id"]: item.points for item in plan.troops}
        self.assertEqual(by_id["baby_dragon"], ((429, 213),))
        self.assertEqual(by_id["dragon"], ((174, 464),))
        self.assertEqual(by_id["balloon"], ((429, 213), (174, 464)))
        self.assertEqual(sum(y < 300 for _, y in by_id["electro_dragon"]), 7)
        self.assertEqual(sum(y > 300 for _, y in by_id["electro_dragon"]), 3)
        self.assertEqual(sum(y < 300 for _, y in by_id["archer"]), 2)
        self.assertEqual(sum(y > 300 for _, y in by_id["archer"]), 1)
        self.assertEqual([(step.card["unit_id"], step.points[0][1] < 300)
                          for step in plan.heroes],
                         [("grand_warden", True), ("archer_queen", True),
                          ("dragon_duke", False), ("minion_prince", False)])
        four_card = TwoEdgeSplitStrategy().build_plan(BattleContext(troops[:-1], heroes, terrain))
        self.assertEqual([step.card["unit_id"] for step in four_card.troops],
                         ["baby_dragon", "dragon", "balloon", "electro_dragon"])
        self.assertEqual(sum(len(step.points) for step in four_card.troops), 14)
        with self.assertRaises(FlowError):
            TwoEdgeSplitStrategy().build_plan(BattleContext(troops[1:], heroes, terrain))
        with self.assertRaises(FlowError):
            TwoEdgeSplitStrategy().build_plan(BattleContext(troops, heroes[:-1], terrain))
        with self.assertRaises(FlowError):
            TwoEdgeSplitStrategy().build_plan(BattleContext(troops, heroes, terrain[:2]))

    def test_battle_context_extends_old_planners_with_objective_evidence(self):
        original = self.context()
        self.assertEqual(original.spells, ())
        self.assertEqual(original.buildings, ())
        extended = BattleContext(original.troops, original.heroes, original.terrain,
                                 spells=({"unit_id": "lightning_spell", "count": 4},),
                                 buildings=({"type": "air_defense", "state": "alive"},),
                                 hero_states=({"ability_ready": True},),
                                 target_progress={"destroyed": 1, "required": 3})
        self.assertEqual(extended.target_progress["required"], 3)
        self.assertEqual(len(STRATEGIES["two_edge"].planner.build_plan(extended).troops), 3)

    def test_new_strategy_survives_config_and_saved_gui_options(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[battle]\nstrategy="edrag_line"\n', encoding="utf-8")
            config = load_config(path)
            options = RunOptions.from_config(config)
            settings = path.with_suffix(".json")
            save_options(settings, options)
            restored = load_options(settings, config)
            self.assertEqual(desktop_config(path, restored).battle.strategy, "edrag_line")
            self.assertTrue(restored.dry_run)
        self.assertEqual(BattleConfig().strategy, "verified")
