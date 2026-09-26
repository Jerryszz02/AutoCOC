from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from autococ.combat import _candidate_changed, _next_candidate, _wait_for_clear_scout, inspect_army, run_battle
from autococ.config import BattleConfig
from autococ.errors import CaptureError, DeploymentError, FlowError
from autococ.reporting import RunStats, resource_metrics
from autococ.scene import SceneSnapshot
from autococ.session import GameSession
from autococ.routine_config import ResourceFilter


def frame(index: int, scene: str, *, resources: dict | None = None, army: dict | None = None,
          price: int | None = 10, deployment: dict | None = None, settlement: dict | None = None) -> SceneSnapshot:
    if army is not None:
        army.setdefault("manifest", {"frame": f"combat-{index}.png", "complete": True,
                                     "supported_layout": True, "troops": [{"count": 10}], "spells": [{"count": 2}]})
    names = {"village": ["attack", "shop"], "search": ["find_match_regular"], "training": ["attack"],
             "enemy_village": ["next", "end_battle"], "settlement": ["return_home"]}.get(scene, [])
    buttons = [{"name": name, "point": [100 + number * 100, 500], "text": name} for number, name in enumerate(names)]
    return SceneSnapshot(scene, 0.95, Path(f"combat-{index}.png"), {
        "resources": resources or {}, "resource_source": "village_inventory" if scene == "village" else "enemy_available",
        "buttons": buttons, "army": army, "search_cost_gold": price, "deployment": deployment,
        "settlement": settlement, "ocr": [],
    })


BEFORE = {"gold": 1000, "elixir": 2000, "dark_elixir": 300, "gems": 50}
AFTER = {"gold": 1110, "elixir": 2055, "dark_elixir": 310, "gems": 50}
ARMY = {"troops": {"used": 100, "capacity": 100}, "spells": {"used": 2, "capacity": 2},
        "clan_troops": {"used": 0, "capacity": 50}, "heroes_available": None, "clan_received_unknown": True}
DEPLOYMENT = {"completed": True, "verified": True, "deployed_units": 10,
              "evidence": ["units-before.png", "units-after.png"]}
SETTLEMENT = {"loot": {"gold": 100, "elixir": 50, "dark_elixir": 10},
              "bonus": {"gold": 20, "elixir": 5, "dark_elixir": 0},
              "percentage": 67, "stars": 2, "evidence": {"source": "settlement_fixture",
                  "stars": {"source": "explicit_victory_star_shapes", "result": {"confidence": .99},
                            "shapes": [{"verified": True}, {"verified": True}]}}}


def complete_frames() -> list[SceneSnapshot]:
    return [
        frame(0, "village", resources=deepcopy(BEFORE)), frame(1, "search"),
        frame(2, "training", army=deepcopy(ARMY)),
        frame(3, "enemy_village", resources={"gold": 150, "elixir": 100, "dark_elixir": 10}),
        frame(4, "battle", deployment=deepcopy(DEPLOYMENT)),
        frame(5, "settlement", settlement=deepcopy(SETTLEMENT)), frame(6, "village", resources=deepcopy(AFTER)),
    ]


class FakeSession:
    buttons = staticmethod(GameSession.buttons)
    click = GameSession.click

    def __init__(self, frames: list[SceneSnapshot], *, max_searches: int = 2) -> None:
        self.frames = iter(frames)
        self.config = SimpleNamespace(battle=BattleConfig(min_expected_resources=100, max_searches=max_searches),
                                      game=SimpleNamespace(startup_timeout_sec=10),
                                      runtime=SimpleNamespace(poll_interval_sec=0))
        self.last_snapshot = None
        self.native = object()
        self.taps, self.back_actions, self.templates, self.events = [], [], [], []
        self.observe_labels = []
        self.deploy_calls = 0

    def check_deadline(self) -> None:
        pass

    def observe(self, label: str = "observe", **options) -> SceneSnapshot:
        self.observe_labels.append(label)
        try:
            self.last_snapshot = next(self.frames)
        except StopIteration:
            raise FlowError("Observation deadline exceeded")
        return self.last_snapshot

    def wait_for(self, scenes: set[str], *, timeout_sec: float = 30, label: str = "wait", **options) -> SceneSnapshot:
        for _ in range(20):
            snapshot = self.observe(label)
            if snapshot.scene in scenes:
                return snapshot
        raise FlowError("Scene deadline exceeded")

    def tap(self, snapshot: SceneSnapshot, point: list[int], *, reason: str) -> None:
        if snapshot is not self.last_snapshot:
            raise AssertionError("Action uses stale screenshot")
        self.taps.append((snapshot.screenshot_path, reason))

    def back(self, snapshot: SceneSnapshot, *, reason: str) -> None:
        self.back_actions.append(snapshot.scene)

    def click_template(self, snapshot: SceneSnapshot, name: str, *, roi: tuple) -> None:
        self.templates.append((name, roi))

    def event(self, kind: str, **data: object) -> None:
        self.events.append((kind, data))

    def deploy(self, session: "FakeSession", scout: SceneSnapshot) -> SceneSnapshot:
        self.deploy_calls += 1
        return self.observe("deployment-confirmed")


def battle(session: FakeSession):
    with patch.dict("sys.modules", {"autococ.deployment": SimpleNamespace(deploy_army=session.deploy)}):
        return run_battle(session)


class CombatTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("autococ.combat._wait_for_clear_scout", side_effect=lambda session, snapshot, deadline: snapshot)
        self.scout_ready = patcher.start()
        self.addCleanup(patcher.stop)

    def test_complete_battle_requires_deployment_settlement_and_matching_inventory(self) -> None:
        session = FakeSession(complete_frames())
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["search_count"], 1)
        self.assertEqual(result.metrics["search_cost_gold"], 10)
        self.assertEqual(result.metrics["loot_gold"], 100)
        self.assertEqual(result.metrics["bonus_elixir"], 5)
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(session.deploy_calls, 1)
        self.assertEqual(session.last_snapshot.scene, "village")
        self.assertTrue(any("find_match_regular" in reason for _, reason in session.taps))
        self.assertTrue(result.metrics["army"]["clan_received_unknown"])
        self.assertEqual(len(result.metrics["settlement_reads"]), 1)
        self.assertTrue(result.metrics["settlement_reads"][0]["valid"])
        self.assertNotIn("battle-settlement-reread", session.observe_labels)

    def test_two_edge_round_accepts_loss_capped_inventory_and_first_low_loot_opponent(self):
        frames = complete_frames()
        frames[3].observations["resources"] = {"gold": 0, "elixir": 0}
        frames[5].observations["settlement"].update(stars=0, percentage=20)
        frames[5].observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        frames[6].observations["resources"] = deepcopy(BEFORE)
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertFalse(result.metrics["victory"])
        self.assertFalse(result.metrics["inventory_reconciled"])
        self.assertTrue(result.metrics["searches"][0]["decision"]["should_attack"])
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_line_missing_star_evidence_keeps_victory_unknown(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"] = {}
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertIsNone(result.metrics["victory"])
        self.assertIsNone(result.metrics["stars"])
        self.assertIsNone(result.metrics["loot_gold"])

    def test_unverified_star_shapes_never_claim_victory(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"]["stars"]["shapes"][0]["verified"] = False
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertIsNone(result.metrics["victory"])

    def test_two_edge_requires_native_zoom_before_any_game_input(self):
        session = FakeSession(complete_frames())
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        session.native = None
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("MuMu native", result.reason)
        self.assertEqual(session.taps, [])

    def test_shared_filter_skips_low_resource_then_deploys_same_line_strategy(self):
        frames = complete_frames()
        low = frame(30, "enemy_village", resources={"gold": 10, "elixir": 10, "dark_elixir": 0})
        frames.insert(3, low)
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge",
            resource_filter=ResourceFilter(min_total=200))
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual([s["decision"]["should_attack"] for s in result.metrics["searches"]], [False, True])
        self.assertEqual(session.deploy_calls, 1)

    def test_unknown_resource_at_search_limit_returns_home_without_deploying(self):
        frames = complete_frames()[:4]
        frames[3].observations["resources"] = {"gold": None, "elixir": 100}
        frames.extend([frame(40, "village", resources=BEFORE), frame(41, "village", resources=BEFORE)])
        session = FakeSession(frames, max_searches=1)
        session.config.battle = replace(session.config.battle, strategy="two_edge", resource_filter=ResourceFilter(min_total=300000))
        result = battle(session)
        self.assertEqual(result.status, "limited")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(session.deploy_calls, 0)
        self.assertFalse(any("next" in reason for _, reason in session.taps))

    def test_dark_elixir_only_filter_can_accept_next_target_with_unknown_gold_elixir(self):
        frames = complete_frames()
        low = frame(30, "enemy_village", resources={"gold": None, "elixir": None, "dark_elixir": 8000})
        unchanged = frame(31, "enemy_village", resources={"gold": None, "elixir": None, "dark_elixir": 8000})
        high = frame(32, "enemy_village", resources={"gold": None, "elixir": None, "dark_elixir": 10000})
        session = FakeSession(frames[:3] + [low, unchanged, high] + frames[4:])
        session.config.battle = replace(session.config.battle, strategy="two_edge",
                                       resource_filter=ResourceFilter(min_dark_elixir=9000))
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["search_count"], 2)
        self.assertEqual(result.metrics["selected_target"]["dark_elixir"], 10000)
        self.assertIsNone(result.metrics["selected_target"]["gold"])
        self.assertEqual(sum("next:" in reason for _, reason in session.taps), 1)
        self.assertEqual(session.deploy_calls, 1)

    def test_verified_spell_only_receipt_can_complete_a_battle(self):
        from autococ.strategy_config import StrategyDefinition, StrategyStep
        frames = complete_frames()
        frames[2].observations["army"]["troops"]["used"] = 0
        frames[2].observations["army"]["manifest"]["troops"] = []
        frames[4].observations["deployment"].update(deployed_units=0, offensive_actions=2, spells_used=2)
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy_file=str(Path("spells.toml").resolve()),
                                       resource_filter=ResourceFilter(enabled=False))
        strategy = StrategyDefinition("spells", "法术", None,
            (StrategyStep("cast_spell", "lightning_spell", 2, target="relative:center"),), "captured")
        with patch("autococ.strategy_config.load_strategy", return_value=strategy), \
                patch("autococ.strategy_execution.execute_strategy", side_effect=lambda s, scout, definition, **kwargs: s.deploy(s, scout)):
            result = run_battle(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["deployment"]["deployed_units"], 0)
        self.assertEqual(result.metrics["deployment"]["spells_used"], 2)

    def test_absent_optional_building_spell_does_not_require_building_samples(self):
        from autococ.reporting import TaskResult
        from autococ.strategy_config import StrategyDefinition, StrategyStep
        from autococ.unit_catalog import ArmyRecipe, ArmyRequirement
        frames = complete_frames()
        frames.insert(1, frame(7, "village", resources=deepcopy(BEFORE)))
        frames[5].observations["deployment"]["offensive_actions"] = 1
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle,
            strategy_file=str(Path("optional-spell.toml").resolve()),
            resource_filter=ResourceFilter(enabled=False))
        strategy = StrategyDefinition("optional", "optional", ArmyRecipe((
            ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1, optional=True))),
            (StrategyStep("cast_spell", "lightning_spell", target="spell_factory"),
             StrategyStep("deploy_troop", "barbarian")))
        prepared = TaskResult("prepare_army", "succeeded", "verified",
            evidence=[Path("army-verified.png")],
            metrics={"observed": {"troop": {"barbarian": 1}, "spell": {}}})
        passed = []

        def execute(s, scout, definition, **kwargs):
            passed.append(kwargs["prepared_army"])
            return s.deploy(s, scout)

        with patch("autococ.strategy_config.load_strategy", return_value=strategy), \
                patch("autococ.army_control.ensure_army", return_value=prepared), \
                patch("autococ.building_vision.building_coverage", return_value={}), \
                patch("autococ.strategy_execution.execute_strategy", side_effect=execute):
            result = run_battle(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(passed, [{"status": "succeeded", "frame": "army-verified.png",
                                   "observed": {"troop": {"barbarian": 1}, "spell": {}}}])
        self.assertFalse(any(label.startswith("building-zoom") for label in session.observe_labels))

    def test_task_building_target_is_not_exempted_by_optional_spell_absence(self):
        from autococ.strategy_config import StrategyDefinition, StrategyStep
        from autococ.unit_catalog import ArmyRecipe, ArmyRequirement
        session = FakeSession(complete_frames())
        session.config.battle = replace(session.config.battle,
            strategy_file=str(Path("optional-spell.toml").resolve()), target_building="spell_factory")
        strategy = StrategyDefinition("optional", "optional", ArmyRecipe((
            ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1, optional=True))),
            (StrategyStep("cast_spell", "lightning_spell", target="spell_factory"),
             StrategyStep("deploy_troop", "barbarian")))
        with patch("autococ.strategy_config.load_strategy", return_value=strategy), \
                patch("autococ.building_vision.building_coverage", return_value={}) as coverage:
            result = run_battle(session)
        self.assertEqual(result.status, "not_supported")
        self.assertIn("spell_factory", result.reason)
        self.assertEqual(session.observe_labels, [])
        coverage.assert_called()

    def test_contrary_battle_spell_requires_building_coverage_and_safe_return(self):
        from autococ.reporting import TaskResult
        from autococ.strategy_config import StrategyDefinition, StrategyStep
        from autococ.unit_catalog import ArmyRecipe, ArmyRequirement
        frames = complete_frames()[:4]
        frames.insert(1, frame(7, "village", resources=deepcopy(BEFORE)))
        frames[4].observations["battle"] = {"slots": [
            {"unit_id": "lightning_spell", "kind": "spell", "source": "army",
             "count": 1, "confidence": .99, "point": [200, 650]}]}
        frames.extend([frame(8, "village", resources=deepcopy(BEFORE)),
                       frame(9, "village", resources=deepcopy(BEFORE))])
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle,
            strategy_file=str(Path("optional-spell.toml").resolve()),
            resource_filter=ResourceFilter(enabled=False))
        strategy = StrategyDefinition("optional", "optional", ArmyRecipe((
            ArmyRequirement("barbarian", 1), ArmyRequirement("lightning_spell", 1, optional=True))),
            (StrategyStep("cast_spell", "lightning_spell", target="spell_factory"),
             StrategyStep("deploy_troop", "barbarian")))
        prepared = TaskResult("prepare_army", "succeeded", "verified",
            evidence=[Path("army-verified.png")],
            metrics={"observed": {"troop": {"barbarian": 1}, "spell": {}}})
        with patch("autococ.strategy_config.load_strategy", return_value=strategy), \
                patch("autococ.army_control.ensure_army", return_value=prepared), \
                patch("autococ.building_vision.building_coverage", return_value={}):
            result = run_battle(session)
        self.assertEqual(result.status, "not_supported", result.reason)
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(session.deploy_calls, 0)
        self.assertTrue(any("end_battle" in reason for _, reason in session.taps))

    def test_star_bonus_receipt_is_confirmed_once_before_verifying_home(self):
        for strategy in ("verified", "edrag_line"):
            with self.subTest(strategy=strategy):
                frames = complete_frames()
                popup = frame(7, "popup")
                popup.observations.update(ocr=[{"text": "已收到胜利之星奖励！", "confidence": .99}],
                    buttons=[{"name": "confirm", "text": "确定", "point": [645, 616]}])
                frames[6:6] = [popup, deepcopy(popup)]
                session = FakeSession(frames)
                session.config.battle = replace(session.config.battle, strategy=strategy)
                result = battle(session)
                self.assertEqual(result.status, "succeeded")
                self.assertEqual(result.metrics["star_bonus_popup_frame"], "combat-7.png")
                self.assertEqual(sum("Recognized confirm" in reason for _, reason in session.taps), 1)
                self.assertEqual(session.last_snapshot.scene, "village")

    def test_unknown_or_uncertain_return_popup_is_never_confirmed(self):
        for text, confidence in (("购买宝石", .99), ("已收到胜利之星奖励", .8),
                                 ("已收到胜利之星奖励", float("nan"))):
            with self.subTest(text=text, confidence=confidence):
                frames = complete_frames()
                popup = frame(7, "popup")
                popup.observations.update(ocr=[{"text": text, "confidence": confidence}],
                    buttons=[{"name": "confirm", "text": "确定", "point": [645, 616]}])
                frames[6:6] = [popup]
                session = FakeSession(frames)
                session.config.battle = replace(session.config.battle, strategy="edrag_line")
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertIn("Unrecognized popup", result.reason)
                self.assertFalse(any("Recognized confirm" in reason for _, reason in session.taps))

    def test_edrag_round_returns_home_without_inventory_or_ability_requirements(self):
        frames = complete_frames()
        frames[3].observations["resources"] = {"gold": 0, "elixir": 0}
        frames[5].observations["settlement"].update(stars=0, percentage=20)
        frames[5].observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        frames[6].observations["resources"] = deepcopy(BEFORE)
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.reason, "edrag_line_troops_settlement_and_return_verified")
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_settlement_wait_can_be_longer_without_relaxing_deployment_deadline(self):
        for configured, expected in ((None, 180), (1200, 1200)):
            with self.subTest(configured=configured):
                session = FakeSession(complete_frames())
                session.config.battle = replace(session.config.battle, settlement_timeout_sec=configured)
                with patch.object(session, "wait_for", wraps=session.wait_for) as wait:
                    result = battle(session)
                self.assertEqual(result.status, "succeeded")
                settlement = [call for call in wait.call_args_list if call.kwargs.get("label") == "battle-settlement"]
                self.assertEqual(settlement[0].kwargs["timeout_sec"], expected)
                self.assertEqual(session.config.battle.deploy_timeout_sec, 180)

    def test_completion_event_write_failure_retains_settled_battle_receipt(self):
        for strategy, event_name in (("verified", "battle_verified"), ("two_edge", "battle_round_verified")):
            with self.subTest(strategy=strategy):
                session = FakeSession(complete_frames())
                session.config.battle = replace(session.config.battle, strategy=strategy)
                original = session.event

                def event(kind, **data):
                    if kind == event_name:
                        raise OSError("event storage unavailable")
                    return original(kind, **data)

                session.event = event
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertIn("event storage unavailable", result.reason)
                self.assertIs(result.metrics["returned_home"], True)
                self.assertEqual(result.metrics["rounds_completed"], 1)
                self.assertIs(result.metrics["victory"], True)
                self.assertEqual(session.last_snapshot.scene, "village")

    def test_edrag_incomplete_hero_receipt_is_failure_even_after_return_home(self):
        frames = complete_frames()
        frames[4].observations["deployment"]["completed"] = False
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_line_home_counter_animation_is_reobserved_without_input(self):
        frames = complete_frames()
        frames[6].observations["resources"]["gems"] = 38
        frames.append(frame(7, "village", resources=deepcopy(AFTER)))
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["resources_after"]["gems"], 50)
        self.assertEqual(len(result.metrics["home_resource_reads"]), 2)
        self.assertEqual(session.taps[-1][1], "Recognized return_home: return_home")

    def test_line_persistent_or_unreadable_gem_change_is_not_accepted(self):
        for gems in (49, None):
            with self.subTest(gems=gems):
                frames = complete_frames()
                frames[6].observations["resources"]["gems"] = 38
                frames.extend(frame(i, "village", resources={**AFTER, "gems": gems}) for i in range(7, 10))
                session = FakeSession(frames)
                session.config.battle = replace(session.config.battle, strategy="two_edge")
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertIn("Gem balance", result.reason)
                self.assertEqual(len(result.metrics["home_resource_reads"]), 4)

    def test_failed_line_with_zero_consumption_still_waits_and_returns_without_replay(self):
        session = FakeSession(complete_frames())
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        def fail_deployment(session, scout):
            session.observe("failed-deployment")
            raise DeploymentError("No troop was consumed", partial_receipt={
                "completed": False, "verified": False, "deployed_units": 0,
                "issued_placements": 10, "evidence": ["combat-4.png"]})
        session.deploy = fail_deployment
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["deployment"]["deployed_units"], 0)
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_two_edge_does_not_claim_provisional_zero_rewards_as_verified(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"]["bonus"] = {"requires_inventory_reconciliation": True}
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertIsNone(result.metrics["bonus_gold"])

    def test_two_edge_result_keeps_unknown_loot_unknown_without_blocking_return(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertIsNone(result.metrics["loot_gold"])
        self.assertIsNotNone(result.metrics["settlement_validation_error"])
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_natural_countdown_without_deployment_cannot_count_as_success(self) -> None:
        for receipt in (None, {"verified": False, "deployed_units": 10, "evidence": ["frame.png"]},
                        {"verified": True, "deployed_units": 0, "evidence": ["frame.png"]},
                        {"verified": True, "deployed_units": 10, "evidence": []}):
            with self.subTest(receipt=receipt):
                frames = complete_frames()
                frames[4].observations["deployment"] = receipt
                self.assertEqual(battle(FakeSession(frames)).status, "failed")

    def test_partial_recipe_never_starts_search_for_enemy(self) -> None:
        frames = complete_frames()
        frames[2].observations["army"]["troops"]["used"] = 90
        session = FakeSession(frames)
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["search_cost_gold"], 0)
        self.assertEqual(session.deploy_calls, 0)
        self.assertFalse(any(path == Path("combat-2.png") for path, _ in session.taps))

    def test_incomplete_or_stale_manifest_stops_before_paid_search(self) -> None:
        for manifest in (None, {"complete": False}, {"complete": True, "supported_layout": True, "frame": "old.png"}):
            with self.subTest(manifest=manifest):
                frames = complete_frames()
                frames[2].observations["army"]["manifest"] = manifest
                session = FakeSession(frames)
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.metrics["search_cost_gold"], 0)
                self.assertEqual(session.deploy_calls, 0)
                self.assertFalse(any(path == Path("combat-2.png") for path, _ in session.taps))

    def test_partial_deployment_failure_keeps_evidence_without_claiming_settlement(self) -> None:
        session = FakeSession(complete_frames())
        partial = {"completed": False, "verified": True, "deployed_units": 1,
                   "evidence": ["combat-3.png", "selected.png", "one-consumed.png"],
                   "consumption_events": [{"unit_kind": "troop", "consumed": 1}],
                   "support_attempts": [{"kind": "hero", "status": "attempted", "verified": False}]}

        def fail_deployment(session: FakeSession, scout: SceneSnapshot) -> SceneSnapshot:
            session.last_snapshot = None
            raise DeploymentError("Screenshot capture failed", partial_receipt=partial)

        session.deploy = fail_deployment
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Screenshot capture failed")
        self.assertEqual(result.metrics["deployment"], partial)
        self.assertIsNone(result.metrics["settlement_observed"])
        self.assertFalse(result.metrics["inventory_reconciled"])
        self.assertEqual(result.metrics["search_cost_gold"], 10)
        self.assertNotIn("loot_gold", result.metrics)
        self.assertTrue(all(Path(path) in result.evidence for path in partial["evidence"]))
        self.assertEqual(result.evidence.count(Path("combat-3.png")), 1)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))
        stats = RunStats(task_results=[result])
        ledger = resource_metrics(stats, elapsed_sec=60)
        self.assertIsNone(ledger["battle_gold_elixir_per_hour"])
        for value in ledger["resources"].values():
            self.assertIsNone(value["battle_loot"])
            self.assertIsNone(value["battle_bonus"])

    def test_incomplete_deployment_returns_home_but_never_counts_as_success(self) -> None:
        frames = complete_frames()
        frames[4].observations["deployment"]["completed"] = False
        session = FakeSession(frames)
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("incomplete", result.reason)
        self.assertEqual(result.metrics["settlement_observed"], SETTLEMENT)
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(session.last_snapshot.scene, "village")
        self.assertTrue(any("return_home" in reason for _, reason in session.taps))
        ledger = resource_metrics(RunStats(task_results=[result]), elapsed_sec=60)
        self.assertIsNone(ledger["battle_gold_elixir_per_hour"])

    def test_partial_deployment_error_can_collect_result_without_repeating_attack(self) -> None:
        session = FakeSession(complete_frames())
        partial = {**deepcopy(DEPLOYMENT), "completed": False}
        def fail_after_start(active: FakeSession, scout: SceneSnapshot) -> SceneSnapshot:
            active.deploy_calls += 1
            active.observe("partial-deployment")
            raise DeploymentError("Final quantity unreadable", partial_receipt=partial)
        session.deploy = fail_after_start
        phases = []
        session.progress_callback = phases.append
        result = battle(session)
        self.assertIn("投放核验失败，等待结算", phases)
        self.assertNotIn("等待结算", phases)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Final quantity unreadable")
        self.assertEqual(session.deploy_calls, 1)
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(result.metrics["deployment_error"], "Final quantity unreadable")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_unavailable_recipe_skips_only_after_confirming_home_without_search(self):
        from autococ.reporting import TaskResult
        from autococ.strategy_config import StrategyDefinition, StrategyStep
        from autococ.unit_catalog import ArmyRecipe, ArmyRequirement
        strategy = StrategyDefinition("recipe", "recipe", ArmyRecipe((ArmyRequirement("barbarian", 1),)),
                                      (StrategyStep("deploy_troop", "barbarian"),))
        for safe_home, mutations, expected in ((True, 0, "skipped"), (False, 0, "failed"),
                                               (True, 1, "failed"), (True, None, "failed")):
            with self.subTest(safe_home=safe_home, mutations=mutations):
                session = FakeSession([])
                session.config.battle = replace(session.config.battle,
                                                strategy_file=str(Path("recipe.toml").resolve()))
                home = frame(0, "village", resources=deepcopy(BEFORE))
                prepared = TaskResult("prepare_army", "skipped", "unit_locked:barbarian",
                                     evidence=[Path("army-lock.png")],
                                     metrics={"actions": 1, "recipe_mutations": mutations})
                returns = [home, home if safe_home else FlowError("Cannot verify home")]
                with patch("autococ.strategy_config.load_strategy", return_value=strategy), \
                        patch("autococ.army_control.ensure_army", return_value=prepared), \
                        patch("autococ.combat.return_to_village", side_effect=returns):
                    result = run_battle(session)
                self.assertEqual(result.status, expected, result.reason)
                self.assertEqual(result.metrics["search_count"], 0)
                self.assertNotIn("rounds_completed", result.metrics)
                self.assertEqual(session.taps, [])
                self.assertEqual(session.deploy_calls, 0)
                if expected == "skipped":
                    self.assertTrue(result.metrics["returned_home"])
                    self.assertEqual(result.reason, "unit_locked:barbarian")

    def test_scout_to_battle_transition_requires_positive_consumption_before_settlement_recovery(self) -> None:
        for scene, verified, units, recover in (("enemy_village", True, 1, True),
                                                ("enemy_village", False, 0, False),
                                                ("enemy_village", True, 0, False),
                                                ("unknown", True, 1, False)):
            with self.subTest(scene=scene, verified=verified, units=units):
                frames = complete_frames()
                frames[4] = replace(frames[4], scene=scene)
                session = FakeSession(frames)
                partial = {**deepcopy(DEPLOYMENT), "completed": False, "verified": verified, "deployed_units": units}

                def fail_after_start(active, scout):
                    active.deploy_calls += 1
                    active.observe("partial-deployment")
                    raise DeploymentError("Card identity lost during transition", partial_receipt=partial)

                session.deploy = fail_after_start
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertEqual(session.deploy_calls, 1)
                self.assertEqual(result.metrics["inventory_reconciled"], recover)
                self.assertEqual(any("return_home" in reason for _, reason in session.taps), recover)
                self.assertEqual(session.last_snapshot.scene == "village", recover)

    def test_deployment_keyboard_interrupt_propagates_without_return_home_action(self) -> None:
        session = FakeSession(complete_frames())

        def interrupt(session: FakeSession, scout: SceneSnapshot) -> SceneSnapshot:
            raise KeyboardInterrupt()

        session.deploy = interrupt
        with self.assertRaises(KeyboardInterrupt):
            battle(session)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_missing_spell_recipe_is_failure(self) -> None:
        frames = complete_frames()
        frames[2].observations["army"]["spells"] = None
        self.assertEqual(battle(FakeSession(frames)).status, "failed")

    def test_unknown_search_price_does_not_start_paid_search(self) -> None:
        frames = complete_frames()
        frames[1].observations["search_cost_gold"] = None
        session = FakeSession(frames)
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["search_cost_gold"], 0)
        self.assertFalse(any("find_match_regular" in reason for _, reason in session.taps))

    def test_failed_search_transition_keeps_potential_cost_unknown(self) -> None:
        session = FakeSession(complete_frames()[:3])
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.metrics["search_cost_gold"])
        self.assertEqual(result.metrics["search_count"], 0)

    def test_budget_exhaustion_never_deploys_below_threshold(self) -> None:
        frames = complete_frames()[:3] + [
            frame(3, "enemy_village", resources={"gold": 20, "elixir": 30, "dark_elixir": 100000}, price=12),
            frame(4, "enemy_village", resources={"gold": 25, "elixir": 30, "dark_elixir": 100000}),
            frame(5, "village", resources=BEFORE),
            frame(6, "village", resources=BEFORE),
        ]
        session = FakeSession(frames)
        result = battle(session)
        self.assertEqual(result.status, "limited")
        self.assertEqual(result.reason, "search_limit_reached_without_target")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(session.last_snapshot.scene, "village")
        self.assertEqual(result.metrics["search_cost_gold"], 22)
        self.assertEqual(result.metrics["search_count"], 2)
        self.assertEqual(session.deploy_calls, 0)

    def test_old_candidate_frame_after_next_is_not_a_new_paid_candidate(self) -> None:
        frames = complete_frames()
        first = frame(3, "enemy_village", resources={"gold": 20, "elixir": 30, "dark_elixir": 10}, price=12)
        stale = frame(30, "enemy_village", resources={"gold": 20, "elixir": 30, "dark_elixir": 10}, price=12)
        rich = frame(31, "enemy_village", resources={"gold": 150, "elixir": 100, "dark_elixir": 10})
        frames[6].observations["resources"]["gold"] = 1098
        session = FakeSession(frames[:3] + [first, stale, rich] + frames[4:])
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["search_count"], 2)
        self.assertEqual(result.metrics["search_cost_gold"], 22)
        self.assertEqual(sum("next:" in reason for _, reason in session.taps), 1)

    def test_delayed_target_is_bounded_by_search_and_outer_deadlines(self) -> None:
        for arrival, outer_limit, succeeds in ((36, 300, True), (72, 300, False), (36, 30, False)):
            with self.subTest(arrival=arrival, outer_limit=outer_limit):
                clock = [100.0]
                target = frame(30, "enemy_village", resources={"gold": 500, "elixir": 1200, "dark_elixir": 5})
                session = FakeSession([frame(20, "unknown"), frame(21, "unknown"), target])
                session.deadline, session.task_deadline = 400.0, 100.0 + outer_limit
                session.wait_for = lambda scenes, **kwargs: GameSession.wait_for(session, scenes, **kwargs)
                observe = session.observe

                def slow_observe(label):
                    clock[0] += arrival / 3
                    return observe(label)

                session.observe = slow_observe
                with patch("autococ.combat.time.monotonic", side_effect=lambda: clock[0]), \
                        patch("autococ.combat.time.sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
                    if succeeds:
                        self.assertIs(_next_candidate(session), target)
                    else:
                        with self.assertRaisesRegex(FlowError, "deadline"):
                            _next_candidate(session)
                self.assertEqual(session.taps, [])
                self.assertEqual(session.deploy_calls, 0)

    def test_incomplete_resource_read_is_reobserved_without_another_paid_search(self) -> None:
        frames = complete_frames()
        incomplete = replace(deepcopy(frames[3]), screenshot_path=Path("incomplete-target.png"))
        incomplete.observations["resources"]["elixir"] = None
        session = FakeSession(frames[:3] + [incomplete] + frames[3:])
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["search_count"], 1)
        self.assertEqual(result.metrics["search_cost_gold"], 10)
        self.assertFalse(any("next:" in reason for _, reason in session.taps))

    def test_name_case_spacing_and_recovered_fields_do_not_prove_a_new_target(self) -> None:
        previous = frame(1, "enemy_village", resources={"gold": 500, "elixir": None, "dark_elixir": 5})
        previous.observations["ocr"] = [{"text": "Sheik ReHMan", "confidence": .89, "bbox": [64, 15, 246, 40]}]
        same = frame(2, "enemy_village", resources={"gold": 500, "elixir": 1200, "dark_elixir": 5})
        same.observations["ocr"] = [{"text": "SHEIK  rehman", "confidence": .9, "bbox": [64, 14, 246, 42]},
                                    {"text": "Clan", "confidence": .99, "bbox": [64, 38, 152, 60]}]
        self.assertFalse(_candidate_changed(previous, same))
        distinct = replace(deepcopy(same), screenshot_path=Path("different-target.png"))
        distinct.observations["resources"]["gold"] = 600
        session = FakeSession([same, distinct])
        self.assertIs(_next_candidate(session, previous), distinct)
        self.assertEqual(session.taps, [])

    def test_distinct_confident_player_name_can_disambiguate_equal_resources(self) -> None:
        previous = frame(1, "enemy_village", resources={"gold": 500, "elixir": 1200, "dark_elixir": 5})
        previous.observations["ocr"] = [{"text": "First", "confidence": .99, "bbox": [64, 15, 246, 40]}]
        current = deepcopy(previous)
        current.observations["ocr"][0]["text"] = "Second"
        self.assertTrue(_candidate_changed(previous, current))
        for confidence in (.8, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                current.observations["ocr"][0]["confidence"] = confidence
                self.assertFalse(_candidate_changed(previous, current))

    def test_live_sheik_ocr_change_is_not_a_completed_next_search(self) -> None:
        path = Path(__file__).resolve().parents[1] / "reports/20260924-000258-418877-80d0380b/events.jsonl"
        if not path.is_file():
            self.skipTest("Optional delayed-search fixture is absent")
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        frames = []
        for number in (10, 11):
            event = next(item for item in events if item.get("kind") == "observation"
                         and Path(item["frame"]).name.startswith(f"{number:05d}-"))
            frames.append(SceneSnapshot(event["scene"], event["confidence"], Path(event["frame"]), event["observations"]))
        self.assertFalse(_candidate_changed(*frames))

    def test_missing_bonus_is_unknown_and_battle_still_returns_home(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"]["bonus"]["gold"] = None
        fresh = [frame(50 + index, "settlement", settlement=deepcopy(frames[5].observations["settlement"])) for index in range(3)]
        session = FakeSession(frames[:6] + fresh + frames[6:])
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Settlement bonus resources are incomplete")
        self.assertEqual(session.last_snapshot.scene, "village")
        self.assertEqual(result.metrics["settlement_observed"]["loot"]["gold"], 100)
        self.assertEqual(session.observe_labels.count("battle-settlement-reread"), 3)
        self.assertEqual(len(result.metrics["settlement_reads"]), 4)
        self.assertFalse(result.metrics["inventory_reconciled"])
        self.assertNotIn("bonus_gold", result.metrics)
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason], [fresh[-1].screenshot_path])
        self.assertTrue(all(item.screenshot_path in result.evidence for item in [frames[5], *fresh]))

    def test_transient_incomplete_settlement_is_read_again_before_return_home(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        fresh = frame(50, "settlement", settlement=deepcopy(SETTLEMENT))
        session = FakeSession(frames[:6] + [fresh] + frames[6:])
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["loot_gold"], 100)
        reads = result.metrics["settlement_reads"]
        self.assertEqual([reading["valid"] for reading in reads], [False, True])
        self.assertIsNone(reads[0]["settlement"]["loot"]["gold"])
        self.assertEqual(reads[1]["settlement"]["loot"]["gold"], 100)
        self.assertIsNone(result.metrics["settlement_validation_error"])
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason], [fresh.screenshot_path])
        self.assertTrue(all(item.screenshot_path in result.evidence for item in (frames[5], fresh)))

    def test_unsupported_victory_returns_home_without_inventing_stars_or_rewards(self) -> None:
        frames = complete_frames()
        unsupported = deepcopy(SETTLEMENT)
        unsupported["stars"] = None
        unsupported["bonus"] = {name: None for name in ("gold", "elixir", "dark_elixir")}
        settlements = [frame(50 + index, "settlement", settlement=deepcopy(unsupported)) for index in range(4)]
        session = FakeSession(frames[:5] + settlements + frames[6:])
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(session.last_snapshot.scene, "village")
        self.assertEqual(len(result.metrics["settlement_reads"]), 4)
        self.assertIsNone(result.metrics["settlement_observed"]["stars"])
        self.assertNotIn("bonus_gold", result.metrics)
        self.assertIsNone(result.metrics.get("victory"))
        self.assertEqual(result.metrics["rounds_completed"], 1)

    def test_stars_and_recognition_evidence_must_also_be_valid_before_return(self) -> None:
        for field, missing in (("stars", None), ("evidence", {})):
            with self.subTest(field=field):
                frames = complete_frames()
                frames[5].observations["settlement"][field] = missing
                fresh = frame(50, "settlement", settlement=deepcopy(SETTLEMENT))
                session = FakeSession(frames[:6] + [fresh] + frames[6:])
                result = battle(session)
                self.assertEqual(result.status, "succeeded")
                self.assertEqual([item["valid"] for item in result.metrics["settlement_reads"]], [False, True])
                self.assertEqual([path for path, reason in session.taps if "return_home" in reason], [fresh.screenshot_path])

    def test_complete_reread_after_ten_second_window_cannot_turn_failure_into_success(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        fresh = frame(50, "settlement", settlement=deepcopy(SETTLEMENT))
        session = FakeSession(frames[:6] + [fresh] + frames[6:])
        clock = [100.0]
        observe = session.observe

        def slow_read(label="observe"):
            if label == "battle-settlement-reread":
                clock[0] += 11
            return observe(label)

        session.observe = slow_read
        with patch("autococ.combat.time.monotonic", side_effect=lambda: clock[0]):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Settlement loot resources are incomplete")
        self.assertEqual(session.observe_labels.count("battle-settlement-reread"), 1)
        self.assertEqual(result.metrics["settlement_reads"][-1]["error"], "Settlement reread window exceeded")
        self.assertEqual(session.last_snapshot.scene, "village")
        self.assertNotIn("loot_gold", result.metrics)

    def test_task_deadline_during_reread_preserves_error_and_prevents_return_click(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        fresh = frame(50, "settlement", settlement=deepcopy(SETTLEMENT))
        session = FakeSession(frames[:6] + [fresh] + frames[6:])
        observe = session.observe
        expired = [False]

        def expire_on_read(label="observe"):
            if label == "battle-settlement-reread":
                expired[0] = True
            return observe(label)

        def check_deadline():
            if expired[0]:
                raise FlowError("Session or task deadline exceeded")

        session.observe = expire_on_read
        session.check_deadline = check_deadline
        result = battle(session)
        self.assertEqual(result.reason, "Settlement loot resources are incomplete")
        self.assertIn("deadline exceeded", result.metrics["settlement_followup_error"])
        self.assertIn(fresh.screenshot_path, result.evidence)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_failed_fresh_capture_never_uses_previous_settlement_to_click(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        session = FakeSession(frames)
        observe = session.observe

        def failed_read(label="observe"):
            if label == "battle-settlement-reread":
                session.last_snapshot = None
                raise CaptureError("fresh screenshot unavailable")
            return observe(label)

        session.observe = failed_read
        result = battle(session)
        self.assertEqual(result.reason, "Settlement loot resources are incomplete")
        self.assertEqual(result.metrics["settlement_followup_error"], "fresh screenshot unavailable")
        self.assertIn(frames[5].screenshot_path, result.evidence)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_final_unknown_reread_cannot_use_older_return_button(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        unknown = [frame(50 + index, "unknown") for index in range(3)]
        session = FakeSession(frames[:6] + unknown)
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Settlement loot resources are incomplete")
        self.assertTrue(all(item.screenshot_path in result.evidence for item in unknown))
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_inventory_mismatch_is_not_hidden_by_valid_settlement(self) -> None:
        frames = complete_frames()
        frames[6].observations["resources"]["gold"] = 1000
        result = battle(FakeSession(frames))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["loot_gold"], 100)
        self.assertFalse(result.metrics["inventory_reconciled"])

    def test_gem_change_prevents_success(self) -> None:
        frames = complete_frames()
        frames[6].observations["resources"]["gems"] = 49
        result = battle(FakeSession(frames))
        self.assertEqual(result.status, "failed")
        self.assertIn("Gem balance", result.reason)

    def test_invalid_stars_or_percentage_prevents_success(self) -> None:
        for key, value in (("stars", 4), ("percentage", -1), ("stars", True)):
            with self.subTest(key=key, value=value):
                frames = complete_frames()
                frames[5].observations["settlement"][key] = value
                self.assertEqual(battle(FakeSession(frames)).status, "failed")

    def test_loss_and_task_completion_are_separate(self) -> None:
        frames = complete_frames()
        frames[5].observations["settlement"].update({"stars": 0, "percentage": 20})
        frames[5].observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        result = battle(FakeSession(frames))
        self.assertEqual(result.status, "succeeded")
        self.assertFalse(result.metrics["victory"])

    def test_reading_enemy_resources_cannot_substitute_for_home_inventory(self) -> None:
        frames = complete_frames()
        frames[0].observations["resource_source"] = "enemy_available"
        session = FakeSession(frames)
        self.assertEqual(battle(session).status, "failed")
        self.assertEqual(session.taps, [])

    def test_inspect_army_reports_recipe_without_claiming_received_reinforcements(self) -> None:
        army = deepcopy(ARMY)
        army["troops"]["used"] = 90
        session = FakeSession([
            frame(0, "village", resources=BEFORE), frame(1, "training", army=army),
            frame(2, "training", army=army), frame(3, "village", resources=BEFORE),
        ])
        result = inspect_army(session)
        self.assertEqual(result.status, "succeeded")
        self.assertFalse(result.metrics["troops_full"])
        self.assertTrue(result.metrics["clan_received_unknown"])
        self.assertIsNone(result.metrics["heroes_available"])
        self.assertEqual(session.back_actions, ["training"])
        self.assertEqual(session.taps, [])
        self.assertEqual(session.templates[0][0], "hud_army")


class ScoutReadinessTests(unittest.TestCase):
    def test_battle_selects_only_the_latest_clear_stable_candidate(self):
        frames = complete_frames()
        clear = frame(30, "enemy_village", resources={"gold": 500, "elixir": 250, "dark_elixir": 10})
        stable = replace(deepcopy(clear), screenshot_path=Path("stable-target.png"))
        session = FakeSession(frames[:4] + [clear, stable] + frames[4:])
        with patch("autococ.terrain.measure_cloud_cover", side_effect=[
                {"obscured": True}, {"obscured": False}, {"obscured": False}]), \
                patch("autococ.terrain.measure_camera_motion", return_value={"stationary": True}):
            result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["selected_target"]["gold"], 500)
        candidates = [data for kind, data in session.events if kind == "battle_candidate"]
        self.assertEqual([data["frame"] for data in candidates], [str(stable.screenshot_path)])
        self.assertEqual(result.metrics["search_count"], 1)

    def test_clouds_and_initial_zoom_are_observed_without_any_input(self):
        initial = frame(0, "enemy_village")
        frames = [frame(i, "enemy_village") for i in range(1, 4)]
        session = FakeSession(frames)
        with patch("autococ.terrain.measure_cloud_cover", side_effect=[
                {"obscured": True}, {"obscured": False}, {"obscured": False}, {"obscured": False}]), \
                patch("autococ.terrain.measure_camera_motion", side_effect=[{"stationary": False}, {"stationary": True}]) as motion:
            result = _wait_for_clear_scout(session, initial, float("inf"))
        self.assertIs(result, frames[-1])
        self.assertEqual(motion.call_args_list[0].args, (frames[0].screenshot_path, frames[1].screenshot_path))
        self.assertEqual([data["ready"] for kind, data in session.events], [False, False, False, True])
        self.assertEqual(session.taps, [])
        self.assertEqual(session.deploy_calls, 0)

    def test_another_cloud_frame_discards_the_previous_clear_reference(self):
        frames = [frame(i, "enemy_village") for i in range(4)]
        session = FakeSession(frames[1:])
        with patch("autococ.terrain.measure_cloud_cover", side_effect=[
                {"obscured": False}, {"obscured": True}, {"obscured": False}, {"obscured": False}]), \
                patch("autococ.terrain.measure_camera_motion", return_value={"stationary": True}) as motion:
            self.assertIs(_wait_for_clear_scout(session, frames[0], float("inf")), frames[3])
        motion.assert_called_once_with(frames[2].screenshot_path, frames[3].screenshot_path)
        self.assertEqual(session.taps, [])

    def test_stable_result_arriving_after_deadline_cannot_be_accepted(self):
        clock = [0.0]
        frames = [frame(i, "enemy_village") for i in range(3)]
        session = FakeSession(frames[1:])
        observe = session.wait_for

        def delayed(*args, **kwargs):
            clock[0] += 2
            return observe(*args, **kwargs)

        session.wait_for = delayed
        with patch("autococ.combat.time.monotonic", side_effect=lambda: clock[0]), \
                patch("autococ.terrain.measure_cloud_cover", return_value={"obscured": False}), \
                patch("autococ.terrain.measure_camera_motion", side_effect=[{"stationary": None}, {"stationary": True}]):
            with self.assertRaisesRegex(FlowError, "camera deadline"):
                _wait_for_clear_scout(session, frames[0], 3)
        self.assertEqual(session.taps, [])
        self.assertFalse(any(data["ready"] for _, data in session.events))


if __name__ == "__main__":
    unittest.main()
