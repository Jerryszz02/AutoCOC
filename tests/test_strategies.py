from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autococ.config import BattleConfig, load_config
from autococ.desktop import RunOptions, desktop_config, load_options, save_options
from autococ.strategies import BattleContext, STRATEGIES


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
