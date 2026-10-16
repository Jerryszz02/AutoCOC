from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autococ.combat import (_candidate_changed, _fresh_boundary_retry_preview, _next_candidate,
                            _read_army, _retryable_boundary_preflight, _wait_for_clear_scout,
                            inspect_army, run_battle)
from autococ.config import BattleConfig, VisionAgentConfig
from autococ.errors import CaptureError, DeploymentError, FlowError, StopRequested
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
        self.frames = frames
        self.frame_index = 0
        self.synthetic_settlement_read = False
        self.config = SimpleNamespace(battle=BattleConfig(min_expected_resources=100, max_searches=max_searches),
                                      game=SimpleNamespace(startup_timeout_sec=10),
                                      runtime=SimpleNamespace(poll_interval_sec=0))
        self.last_snapshot = None
        self.evidence_budget = None
        self.native = object()
        self.taps, self.back_actions, self.templates, self.events = [], [], [], []
        self.observe_labels = []
        self.deploy_calls = 0

    def check_deadline(self) -> None:
        pass

    def observe(self, label: str = "observe", **options) -> SceneSnapshot:
        self.observe_labels.append(label)
        # Existing line fixtures describe one steady result screen. A reread
        # captures a distinct screenshot before returning home; dedicated
        # animation tests below supply their own sequence instead.
        line_mode = (self.config.battle.strategy in {"edrag_line", "two_edge"}
                     or bool(self.config.battle.strategy_file)
                     or bool(getattr(getattr(self.config, "vision_agent", None), "enabled", False)))
        if (line_mode and label == "battle-settlement-reread" and not self.synthetic_settlement_read
                and self.last_snapshot is not None and self.last_snapshot.scene == "settlement"
                and self.frame_index < len(self.frames)
                and self.frames[self.frame_index].scene in {"village", "popup"}):
            self.synthetic_settlement_read = True
            self.last_snapshot = replace(deepcopy(self.last_snapshot),
                                         screenshot_path=Path("combat-stable-settlement.png"))
            return self.last_snapshot
        if self.frame_index >= len(self.frames):
            raise FlowError("Observation deadline exceeded")
        self.last_snapshot = self.frames[self.frame_index]
        self.frame_index += 1
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

    def _skill_session(self, *, enabled=True):
        session = FakeSession(complete_frames())
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous",
                                        post_deployment_hero_skill="grand_warden" if enabled else "",
                                        hero_skill_delay_sec=20)
        return session

    def _run_with_skill(self, session, handler):
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))), \
             patch.dict("sys.modules", {"autococ.post_deployment_skill": SimpleNamespace(
                 execute_post_deployment_hero_skill=handler)}):
            return battle(session)

    def test_skill_disabled_leaves_existing_battle_untouched(self):
        session = self._skill_session(enabled=False)
        skill = Mock(side_effect=AssertionError("Disabled skill must not observe or act"))
        result = self._run_with_skill(session, skill)
        self.assertEqual(result.status, "succeeded")
        skill.assert_not_called()
        self.assertNotIn("hero_skill", result.metrics)
        self.assertNotIn("hero_skill_plan", result.metrics)

    def test_skill_called_once_after_deployment_with_frozen_choice(self):
        session = self._skill_session()
        def deploy(active, scout):
            active.config.battle = replace(active.config.battle, hero_skill_delay_sec=0)
            return active.observe("deployment-confirmed")
        session.deploy = deploy
        receipt = {"hero_id": "grand_warden", "verified": True, "completed": True,
                   "evidence": ["skill-before.png", "skill-after.png"]}
        skill = Mock(side_effect=lambda active, deployed, **options: (deployed, receipt))
        result = self._run_with_skill(session, skill)
        self.assertEqual(result.status, "succeeded")
        skill.assert_called_once_with(session, session.frames[4], hero_id="grand_warden", delay_sec=20)
        self.assertEqual(result.metrics["hero_skill_plan"], {"hero_id": "grand_warden", "delay_sec": 20})
        self.assertEqual(result.metrics["hero_skill"], receipt)
        self.assertIn(Path("skill-after.png"), result.evidence)
        self.assertTrue(result.metrics["inventory_reconciled"])

    def test_unverified_skill_still_settles_returns_and_keeps_actual_win(self):
        for reason in ("ability_not_ready", "input_uncertain", "used_state_unverified"):
            with self.subTest(reason=reason):
                session = self._skill_session()
                receipt = {"hero_id": "grand_warden", "verified": False, "completed": False,
                           "reason": reason, "evidence": ["skill-check.png"]}
                skill = Mock(side_effect=lambda active, deployed, **options: (deployed, receipt))
                result = self._run_with_skill(session, skill)
                self.assertEqual(result.status, "failed")
                self.assertIn(reason, result.reason)
                self.assertTrue(result.metrics["victory"])
                self.assertTrue(result.metrics["inventory_reconciled"])
                self.assertTrue(result.metrics["returned_home"])
                self.assertEqual(result.metrics["resources_after"], AFTER)
                self.assertEqual(skill.call_count, 1)

    def test_skill_can_return_fresh_settlement_without_discarding_it(self):
        session = self._skill_session()
        receipt = {"hero_id": "grand_warden", "verified": False, "completed": False,
                   "reason": "battle_already_settled", "evidence": ["combat-5.png"]}
        skill = Mock(side_effect=lambda active, deployed, **options:
                     (active.observe("skill-already-settled"), receipt))
        result = self._run_with_skill(session, skill)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertNotIn("battle-settlement", session.observe_labels)

    def test_unexpected_skill_error_keeps_attempt_and_finishes_battle_from_fresh_frames(self):
        for error in (RuntimeError("template decode failed"), ValueError("invalid skill result"),
                      CaptureError("skill capture failed")):
            with self.subTest(error=type(error).__name__):
                session = self._skill_session()
                def fail(active, deployed, **options):
                    active.post_deployment_skill_receipt = {"attempted": True, "input_count": 1,
                                                           "evidence": ["skill-before.png"]}
                    active.last_snapshot = None
                    raise error
                result = self._run_with_skill(session, fail)
                self.assertEqual(result.status, "failed")
                self.assertIn("unexpected_skill_error", result.reason)
                self.assertTrue(result.metrics["hero_skill"]["attempted"])
                self.assertEqual(result.metrics["hero_skill"]["input_count"], 1)
                self.assertIn(type(error).__name__, result.metrics["hero_skill"]["error"])
                self.assertTrue(result.metrics["returned_home"])
                self.assertTrue(result.metrics["victory"])
                self.assertTrue(result.metrics["inventory_reconciled"])
                self.assertIn("battle-settlement", session.observe_labels)
                self.assertEqual(session.deploy_calls, 1)

    def test_incomplete_deployment_never_runs_skill(self):
        session = self._skill_session()
        session.frames[4].observations["deployment"]["completed"] = False
        skill = Mock(side_effect=AssertionError("Incomplete deployment must not use skill"))
        result = self._run_with_skill(session, skill)
        skill.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["hero_skill"]["reason"], "deployment_incomplete")
        self.assertTrue(result.metrics["inventory_reconciled"])

    def test_skill_stop_and_account_conflict_do_not_continue_inputs(self):
        for error in (StopRequested("user stopped"), FlowError("Game interruption: account_in_use")):
            with self.subTest(error=type(error).__name__):
                session = self._skill_session()
                skill = Mock(side_effect=error)
                if isinstance(error, StopRequested):
                    with self.assertRaises(StopRequested):
                        self._run_with_skill(session, skill)
                else:
                    result = self._run_with_skill(session, skill)
                    self.assertEqual(result.status, "failed")
                    self.assertIn("account_in_use", result.reason)
                self.assertFalse(any("return_home" in reason for _, reason in session.taps))
                self.assertEqual(session.last_snapshot, session.frames[4])

    def test_skill_rejects_vision_combination_before_any_game_input(self):
        session = self._skill_session()
        session.config.vision_agent = VisionAgentConfig(enabled=True)
        skill = Mock()
        result = self._run_with_skill(session, skill)
        self.assertEqual(result.status, "not_supported")
        self.assertIn("model-free", result.reason)
        self.assertEqual(session.taps, [])
        self.assertEqual(session.observe_labels, [])
        skill.assert_not_called()

    def test_unexpected_skill_error_with_current_account_conflict_stops_inputs(self):
        session = self._skill_session()
        conflict = frame(12, "account_in_use")
        def fail(active, deployed, **options):
            active.post_deployment_skill_receipt = {"attempted": True, "input_count": 1,
                                                   "evidence": ["skill-before.png"]}
            active.last_snapshot = conflict
            raise ValueError("skill recognition failed")
        result = self._run_with_skill(session, fail)
        self.assertEqual(result.status, "failed")
        self.assertIn("account_in_use", result.reason)
        self.assertTrue(result.metrics["hero_skill"]["attempted"])
        self.assertEqual(result.metrics["hero_skill"]["input_count"], 1)
        self.assertIs(session.last_snapshot, conflict)
        self.assertNotIn("battle-settlement", session.observe_labels)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_unexpected_skill_error_at_deadline_keeps_attempt_without_more_inputs(self):
        session = self._skill_session()
        def fail(active, deployed, **options):
            active.post_deployment_skill_receipt = {"attempted": True, "input_count": 1,
                                                   "evidence": ["skill-before.png"]}
            active.check_deadline = Mock(side_effect=FlowError("Session or task deadline exceeded"))
            raise RuntimeError("post-skill observation failed")
        result = self._run_with_skill(session, fail)
        self.assertEqual(result.status, "failed")
        self.assertIn("deadline exceeded", result.reason)
        self.assertTrue(result.metrics["hero_skill"]["attempted"])
        self.assertEqual(result.metrics["hero_skill"]["input_count"], 1)
        self.assertNotIn("battle-settlement", session.observe_labels)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    @staticmethod
    def _probed_boundary_error(**updates):
        receipt = {"phase": "boundary_scan", "deployed_units": 0,
                   "attempted_placements": 0, "issued_placements": 0,
                   "offensive_actions": 0, "consumption_events": [],
                   "terrain_evidence": [], "evidence": ["boundary.png"]}
        receipt.update(updates)
        return DeploymentError("Two current deployment boundaries could not be verified",
                               partial_receipt=receipt)

    @staticmethod
    def _preview(index, *, count=15):
        preview = frame(index, "enemy_village", resources={"gold": 150, "elixir": 100,
                                                            "dark_elixir": 10})
        preview.observations["battle"] = {"phase": "scout", "countdown_seconds": count,
                                          "countdown_evidence": {
                                              "frame": str(preview.screenshot_path), "confidence": .99,
                                              "source": "current_frame_ocr"}}
        preview.observations["observed_at_monotonic"] = time.monotonic()
        return preview

    def test_probed_boundary_rejection_searches_next_once_and_keeps_both_costs(self):
        frames = complete_frames()[:3] + [self._preview(3), self._preview(4),
                  frame(5, "enemy_village", resources={"gold": 170, "elixir": 100,
                                                       "dark_elixir": 10}),
                  frame(6, "battle", deployment=deepcopy(DEPLOYMENT)),
                  frame(7, "settlement", settlement=deepcopy(SETTLEMENT)),
                  frame(8, "village", resources={**BEFORE, "gold": 1100,
                                                  "elixir": 2055, "dark_elixir": 310})]
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous",
                                        line_tactics="two_edge_probed_anchors", max_searches=2)
        def deploy(active, scout):
            active.deploy_calls += 1
            if active.deploy_calls == 1:
                raise self._probed_boundary_error()
            return active.observe("deployment-confirmed")
        session.deploy = deploy
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(session.deploy_calls, 2)
        self.assertEqual(result.metrics["search_count"], 2)
        self.assertEqual(result.metrics["search_cost_gold"], 20)
        self.assertEqual(len(result.metrics["preflight_rejections"]), 1)
        self.assertEqual(result.metrics["preflight_rejections"][0]["phase"], "boundary_scan")
        self.assertEqual(sum("Recognized next" in reason for _, reason in session.taps), 1)
        self.assertEqual([path for path, reason in session.taps if "Recognized next" in reason],
                         [frames[4].screenshot_path])
        self.assertTrue(result.metrics["inventory_reconciled"])

    def test_only_explicit_zero_action_boundary_phase_can_rescout(self):
        active = SimpleNamespace(frozen_line_receipt=None)
        self.assertTrue(_retryable_boundary_preflight(active, self._probed_boundary_error()))
        for changes in ({"phase": "boundary_verified"}, {"attempted_placements": 1},
                        {"issued_placements": 1}, {"offensive_actions": 1},
                        {"deployed_units": 1}, {"terrain_evidence": [{"edge": 0}]},
                        {"consumption_events": [{}]}, {"issued_placements": None}):
            with self.subTest(changes=changes):
                self.assertFalse(_retryable_boundary_preflight(
                    active, self._probed_boundary_error(**changes)))
        active.frozen_line_receipt = {"attempted_placements": 1}
        self.assertFalse(_retryable_boundary_preflight(active, self._probed_boundary_error()))

    def test_mixed_probe_boundary_rejection_can_rescout_without_input(self):
        from autococ.reporting import TaskResult

        # This test isolates scout-boundary retry. Its synthetic 100/100 army
        # predates the strict mixed-roster preflight, so supply its read-only
        # verified result and a distinct post-preflight army frame.
        frames = complete_frames()[:3] + [frame(82, "training", army=deepcopy(ARMY)),
                  self._preview(83), self._preview(84),
                  frame(85, "enemy_village", resources={"gold": 170, "elixir": 100,
                                                         "dark_elixir": 10}),
                  frame(86, "battle", deployment=deepcopy(DEPLOYMENT)),
                  frame(87, "settlement", settlement=deepcopy(SETTLEMENT)),
                  frame(88, "village", resources={**BEFORE, "gold": 1100,
                                                  "elixir": 2055, "dark_elixir": 310})]
        session = FakeSession(frames)
        session.client_version = "18.647.2"
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous",
                                        line_tactics="two_edge_furnace_mixed_probes", max_searches=2)

        def deploy(active, _scout):
            active.deploy_calls += 1
            if active.deploy_calls == 1:
                raise self._probed_boundary_error()
            return active.observe("deployment-confirmed")

        session.deploy = deploy
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))), \
             patch("autococ.army_control.ensure_mixed_archer_army", return_value=TaskResult(
                 "prepare_army", "succeeded", "preflight fixture",
                 evidence=[Path("mixed-army-preflight.png")])) as fill, \
             patch("autococ.army_control.mixed_archer_attack_guard", return_value=None):
            result = battle(session)
        fill.assert_called_once_with(session)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(session.deploy_calls, 2)
        self.assertEqual(result.metrics["search_count"], 2)
        self.assertEqual(result.metrics["search_cost_gold"], 20)
        self.assertEqual(len(result.metrics["preflight_rejections"]), 1)
        self.assertEqual(sum("Recognized next" in reason for _, reason in session.taps), 1)

    def test_rescout_requires_fresh_positive_preview_timer_and_navigation(self):
        good = self._preview(31)
        variations = [frame(32, "unknown"), frame(33, "battle"),
                      replace(self._preview(34), confidence=.7)]
        for change in (lambda p: p.observations["battle"].update(phase="active"),
                       lambda p: p.observations["battle"].update(countdown_seconds=0),
                       lambda p: p.observations["battle"]["countdown_evidence"].update(frame="old.png"),
                       lambda p: p.observations["battle"]["countdown_evidence"].update(confidence=.5),
                       lambda p: p.observations["battle"]["countdown_evidence"].update(source="cached"),
                       lambda p: p.observations["battle"].update(countdown_seconds=121),
                       lambda p: p.observations.update(observed_at_monotonic=time.monotonic() - 20),
                       lambda p: p.observations["buttons"].pop(0)):
            bad = deepcopy(good)
            change(bad)
            variations.append(bad)
        for bad in variations:
            with self.subTest(scene=bad.scene, observations=bad.observations):
                session = FakeSession([bad])
                self.assertIsNone(_fresh_boundary_retry_preview(session, []))
                self.assertEqual(session.taps, [])
        session = FakeSession([good])
        evidence = []
        self.assertIs(_fresh_boundary_retry_preview(session, evidence), good)
        self.assertEqual(evidence, [good.screenshot_path])

    def test_account_conflict_during_rescout_does_not_authorize_navigation(self):
        conflict = frame(39, "account_in_use")
        session = FakeSession([conflict])
        with self.assertRaisesRegex(FlowError, "account_in_use"):
            _fresh_boundary_retry_preview(session, [])
        self.assertEqual(session.taps, [])

    def test_probed_boundary_at_search_limit_preserves_original_failure_without_next(self):
        frames = complete_frames()[:4] + [self._preview(7), frame(8, "village"),
                                            frame(9, "village", resources=deepcopy(BEFORE))]
        session = FakeSession(frames, max_searches=1)
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous",
                                        line_tactics="two_edge_probed_anchors")
        def fail(active, scout):
            active.deploy_calls += 1
            raise self._probed_boundary_error()
        session.deploy = fail
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Two current deployment boundaries could not be verified")
        self.assertEqual(result.metrics["search_count"], 1)
        self.assertEqual(result.metrics["confirmed_search_cost_gold"], 10)
        self.assertEqual(len(result.metrics["preflight_rejections"]), 1)
        self.assertEqual(session.deploy_calls, 1)
        self.assertFalse(any("Recognized next" in reason for _, reason in session.taps))

    def test_unknown_after_boundary_rejection_never_authorizes_next(self):
        frames = complete_frames()[:4] + [frame(7, "unknown"), self._preview(8),
                  frame(9, "village"), frame(10, "village", resources=deepcopy(BEFORE))]
        session = FakeSession(frames, max_searches=2)
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous",
                                        line_tactics="two_edge_probed_anchors")
        session.deploy = lambda active, scout: (_ for _ in ()).throw(self._probed_boundary_error())
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Two current deployment boundaries could not be verified")
        self.assertEqual(result.metrics["confirmed_search_cost_gold"], 10)
        self.assertEqual(len(result.metrics["preflight_rejections"]), 1)
        self.assertFalse(any("Recognized next" in reason for _, reason in session.taps))

    def test_line_round_records_verified_storage_overflow_without_inventing_bonus(self):
        frames = complete_frames()
        frames[6].observations["resources"]["gold"] = 1100
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        session.client_version = "18.647.2"
        proof = {"capacity": 1100, "resource": "gold", "frame": "gold-detail.png"}
        with patch("autococ.combat.read_resource_capacity",
                   return_value=(proof, frames[6], [Path("gold-detail.png")])) as read:
            result = battle(session)
        read.assert_called_once()
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(result.metrics["inventory_reconciliation_method"], "capacity_capped")
        self.assertEqual(result.metrics["battle_credited_gold"], 110)
        self.assertEqual(result.metrics["overflow_gold"], 10)
        self.assertEqual(result.metrics["settlement_observed"]["bonus"]["gold"], 20)

    def test_unreleased_vision_model_is_rejected_before_enemy_search(self):
        session = FakeSession(complete_frames())
        session.config.vision_agent = VisionAgentConfig(enabled=True, model_dir="models")
        session.config.source_path = Path("config.toml").resolve()
        session.client_version = "18.1"
        model = SimpleNamespace(model_sha256="unreleased", detect=Mock(),
                                metadata={"validation": {"status": "evaluated", "report_id": "old"}})
        with patch("autococ.model_vision.BuildingModel", return_value=model):
            result = run_battle(session)
        self.assertEqual(result.metrics["search_count"], 0)
        self.assertEqual(session.observe_labels, [])
        self.assertEqual(session.taps, [])
        model.detect.assert_not_called()
        self.assertIn("release", result.reason.lower())

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

    def test_vision_mode_simulates_preload_plan_burst_post_settlement_and_home(self):
        frames = complete_frames()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scout = frames[3]
            army = frames[2].observations["army"]
            army["troops"] = {"used": 2, "capacity": 2}
            army["spells"] = {"used": 0, "capacity": 0}
            army["manifest"].update(identity_complete=True,
                                    troops=[{"unit_id": "archer", "count": 2}], spells=[])
            card = {"kind": "troop", "unit_id": "archer", "source": "army", "count": 2,
                    "point": [110, 630]}
            profile = {"accepted": True, "cards_stable_after_deployment": True,
                       "baseline_resolution": [1280, 720], "client_version": "18.1",
                       "transport": "mumu_native", "display_id": 2, "model_sha256": "modelhash",
                       "zoom": "minimum_stable",
                       "model_validation_report_id": "offline-1",
                       "cards": [card], "candidate_orders": [["archer"]],
                       "core_classes": {"town_hall": .9}}
            layout = root / "layout.json"
            layout.write_text(json.dumps(profile), encoding="utf-8")
            scout.observations.update(observed_at_monotonic=__import__("time").monotonic(),
                minimum_zoom={"verified": True},
                terrain=[{"edge": edge, "point": point,
                          "evidence": {"frame": str(scout.screenshot_path)}}
                         for edge, pair in ((0, ([250, 240], [350, 160])),
                                            (1, ([250, 420], [350, 500]))) for point in pair],
                battle={"countdown_seconds": 30, "countdown_evidence": {
                    "frame": str(scout.screenshot_path), "confidence": .99,
                    "source": "current_frame_ocr"},
                        "slots": [{**card, "confidence": .99,
                                   "evidence": {"count": {"confidence": .99}}}]})
            frames[4].observations["battle"] = {"slots": [{**card, "count": 0,
                "confidence": .99,
                "evidence": {"count": {"confidence": .99}}}]}
            session = FakeSession(frames)
            session.config.vision_agent = VisionAgentConfig(enabled=True, model_dir=str(root),
                layout_profile=str(layout))
            session.config.source_path = root / "config.toml"
            session.config.game.baseline_resolution = (1280, 720)
            session.config.runtime.step_timeout_sec = 2.
            session.capture = SimpleNamespace(input_display_id=2)
            session.context = SimpleNamespace(screen_resolution=(1280, 720))
            session.client_version = "18.1"
            session.recognizer = SimpleNamespace(building_model=None)
            session.deadline = session.task_deadline = __import__("time").monotonic() + 600
            session._validate_snapshot = lambda snapshot: None if snapshot is session.last_snapshot else (_ for _ in ()).throw(FlowError("stale"))
            original_observe = session.observe
            def observed(label="observe", **options):
                result = original_observe(label, **options)
                if result is frames[4]:
                    result.observations["observed_at_monotonic"] = __import__("time").monotonic()
                return result
            session.observe = observed
            input_calls = []
            session.frozen_battle_tap = lambda point, **kw: input_calls.append(point)
            model = SimpleNamespace(model_sha256="modelhash", detect=Mock(return_value=()),
                                    metadata={"classes": ["town_hall"], "applicability": {"zoom": "minimum_stable",
                                               "source_width_range": [1280, 2560],
                                               "source_height_range": [720, 1440]},
                                              "validation": {"status": "evaluated", "report_id": "offline-1"}})
            core = SimpleNamespace(center=(600., 260.), confidence=.99)
            with patch("autococ.model_vision.BuildingModel", return_value=model), \
                    patch("autococ.model_release.require_released_model", return_value={}), \
                    patch("autococ.prepared_battle.measure_cloud_cover", return_value={"obscured": False}), \
                    patch("autococ.images.read_frame", return_value=np.zeros((720, 1280, 3), dtype=np.uint8)), \
                    patch("autococ.model_vision.infer_core_geometry", return_value=core):
                result = run_battle(session)
            self.assertEqual(result.status, "succeeded", result.reason)
            self.assertEqual(len(input_calls), 3)
            self.assertEqual(model.detect.call_count, 2)  # preload and this battle
            self.assertTrue(result.metrics["vision_agent"]["plan_ready"])
            self.assertTrue(result.metrics["vision_agent"]["input_sent"])
            self.assertTrue(result.metrics["vision_agent"]["consumption_verified"])
            self.assertTrue(result.metrics["returned_home"])

    def _vision_failure(self, session, failure):
        session.config.vision_agent = VisionAgentConfig(enabled=True, model_dir="models")
        session.config.source_path = Path("config.toml").resolve()
        session.client_version = "18.1"
        model = SimpleNamespace(detect=Mock(return_value=()), metadata={
            "validation": {"status": "evaluated", "report_id": "offline-1"},
            "applicability": {"source_width_range": [1280, 1280], "source_height_range": [720, 720]}})
        with patch("autococ.model_vision.BuildingModel", return_value=model), \
                patch("autococ.model_release.require_released_model", return_value={}), \
                patch("autococ.prepared_battle.validate_prepared_capability"), \
                patch("autococ.prepared_battle.run_prepared_battle", side_effect=failure):
            return run_battle(session)

    def test_vision_preparation_failure_exits_fresh_preview_and_verifies_home(self):
        for error in ("Current preparation countdown is unavailable", "Core confidence too low",
                      "Observed cards differ from accepted layout"):
            with self.subTest(error=error):
                frames = complete_frames()[:4] + [frame(7, "enemy_village"),
                    frame(8, "village"), frame(9, "village")]
                session = FakeSession(frames)
                result = self._vision_failure(session, FlowError(error))
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, error)
                self.assertTrue(result.metrics.get("returned_home"))
                self.assertIn(frames[-1].screenshot_path, result.evidence)
                exits = [(path, reason) for path, reason in session.taps if "end_battle" in reason]
                self.assertEqual(len(exits), 1)
                self.assertEqual(exits[0][0], frames[4].screenshot_path)
                self.assertEqual(session.deploy_calls, 0)
                self.assertFalse(result.metrics["vision_agent"]["input_sent"])
                self.assertNotIn("rounds_completed", result.metrics)

    def test_vision_preparation_failure_cannot_exit_unverified_or_started_battle(self):
        for scene, confidence in (("unknown", .99), ("enemy_village", .7)):
            with self.subTest(scene=scene, confidence=confidence):
                current = replace(frame(7, scene), confidence=confidence)
                session = FakeSession(complete_frames()[:4] + [current])
                result = self._vision_failure(session, FlowError("preparation failed"))
                self.assertEqual(result.reason, "preparation failed")
                self.assertIn("preparation_recovery_error", result.metrics)
                self.assertFalse(result.metrics.get("returned_home", False))
                self.assertFalse(any("end_battle" in reason for _, reason in session.taps))

    def test_vision_preparation_failure_in_battle_waits_for_settlement_and_returns_failed(self):
        settled = frame(8, "settlement", settlement=deepcopy(SETTLEMENT))
        settled.observations["settlement"].update(stars=0, percentage=0)
        settled.observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        home = frame(9, "village", resources=deepcopy(AFTER))
        session = FakeSession(complete_frames()[:4] + [frame(7, "battle"), settled, home])
        result = self._vision_failure(session, FlowError("preparation expired"))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "preparation expired")
        self.assertIsNone(session.prepared_battle_receipt)
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertFalse(result.metrics["victory"])
        self.assertIn(settled.screenshot_path, result.evidence)
        self.assertIn(home.screenshot_path, result.evidence)
        self.assertEqual(session.observe_labels.count("preparation-failure-settlement"), 1)
        self.assertEqual([reason for _, reason in session.taps if "return_home" in reason],
                         ["Recognized return_home: return_home"])
        self.assertFalse(any("end_battle" in reason for _, reason in session.taps))
        self.assertIsNone(result.metrics["deployment"])

    def test_vision_preparation_failure_does_not_click_unknown_or_weak_settlement(self):
        cases = ([frame(7, "battle"), frame(8, "unknown")],
                 [frame(7, "battle"), replace(frame(8, "settlement"), confidence=.7)])
        for suffix in cases:
            with self.subTest(scene=suffix[-1].scene):
                session = FakeSession(complete_frames()[:4] + suffix)
                result = self._vision_failure(session, FlowError("preparation expired"))
                self.assertEqual(result.status, "failed")
                self.assertIn("preparation_recovery_error", result.metrics)
                self.assertFalse(result.metrics.get("returned_home", False))
                self.assertFalse(any("return_home" in reason or "end_battle" in reason
                                     for _, reason in session.taps))

    def test_vision_stop_or_existing_input_receipt_does_not_trigger_preview_recovery(self):
        session = FakeSession(complete_frames()[:4])
        with self.assertRaises(StopRequested):
            self._vision_failure(session, StopRequested("stopped"))
        self.assertNotIn("preparation-failure-check", session.observe_labels)
        def after_input(active, scout, **kwargs):
            active.prepared_battle_receipt = {"attempted_placements": 1}
            raise FlowError("after input")
        session = FakeSession(complete_frames()[:4])
        result = self._vision_failure(session, after_input)
        self.assertEqual(result.reason, "after input")
        self.assertNotIn("preparation-failure-check", session.observe_labels)

    def test_two_edge_round_records_loss_but_rejects_unreconciled_inventory(self):
        frames = complete_frames()
        frames[3].observations["resources"] = {"gold": 0, "elixir": 0}
        frames[5].observations["settlement"].update(stars=0, percentage=20)
        frames[5].observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        frames[6].observations["resources"] = deepcopy(BEFORE)
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertEqual(result.metrics["settlement_observed"]["stars"], 0)
        self.assertIsNone(result.metrics["victory"])
        self.assertFalse(result.metrics["inventory_reconciled"])
        self.assertNotIn("loot_gold", result.metrics)
        self.assertTrue(result.metrics["searches"][0]["decision"]["should_attack"])
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_line_missing_star_evidence_keeps_victory_unknown(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"] = {}
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.metrics.get("victory"))
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_unverified_star_shapes_never_claim_victory(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"]["stars"]["shapes"][0]["verified"] = False
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.metrics.get("victory"))
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

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
        frames[-1].observations["resources"]["gold"] -= 10  # Two verified searches cost 20 in total.
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
        frames[-1].observations["resources"]["gold"] -= 10  # Two verified searches cost 20 in total.
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

    def test_edrag_round_returns_home_but_rejects_unreconciled_inventory(self):
        frames = complete_frames()
        frames[3].observations["resources"] = {"gold": 0, "elixir": 0}
        frames[5].observations["settlement"].update(stars=0, percentage=20)
        frames[5].observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        frames[6].observations["resources"] = deepcopy(BEFORE)
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["settlement_observed"]["stars"], 0)
        self.assertIsNone(result.metrics["victory"])
        self.assertEqual(session.last_snapshot.scene, "village")

    def test_line_result_waits_for_two_distinct_stable_frames_after_count_up(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["bonus"]["gold"] = None
        final = frame(50, "settlement", settlement=deepcopy(SETTLEMENT))
        repeated = frame(51, "settlement", settlement=deepcopy(SETTLEMENT))
        session = FakeSession(frames[:6] + [final, repeated] + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual([read["valid"] for read in result.metrics["settlement_reads"]], [False, True, True])
        self.assertEqual([read["frame"] for read in result.metrics["settlement_reads"]],
                         ["combat-5.png", "combat-50.png", "combat-51.png"])
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason],
                         [repeated.screenshot_path])
        self.assertTrue(all(path in result.evidence for path in
                            (frames[5].screenshot_path, final.screenshot_path, repeated.screenshot_path)))

    def test_line_result_can_use_remaining_window_after_six_intermittent_star_reads(self):
        frames = complete_frames()
        readings = []
        for index, valid in enumerate((False, True, False, False, False, True, True)):
            observed = deepcopy(SETTLEMENT)
            if not valid:
                observed["stars"] = None
            readings.append(frame(50 + index, "settlement", settlement=observed))
        session = FakeSession(frames[:5] + readings + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual([read["valid"] for read in result.metrics["settlement_reads"]],
                         [False, True, False, False, False, True, True])
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason],
                         [readings[-1].screenshot_path])

    def test_line_result_conflicting_valid_value_resets_adjacent_confirmation(self):
        frames = complete_frames()
        readings = []
        for index, gold in enumerate((100, 101, 100, 100)):
            observed = deepcopy(SETTLEMENT)
            observed["loot"]["gold"] = gold
            readings.append(frame(50 + index, "settlement", settlement=observed))
        session = FakeSession(frames[:5] + readings + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertEqual(len(result.metrics["settlement_reads"]), 4)
        self.assertEqual(result.metrics["settlement_observed"]["loot"]["gold"], 100)
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason],
                         [readings[-1].screenshot_path])

    def test_line_result_still_obeys_thirty_five_second_reread_deadline(self):
        frames = complete_frames()
        invalid = deepcopy(SETTLEMENT)
        invalid["stars"] = None
        readings = [frame(50 + index, "settlement", settlement=deepcopy(invalid)) for index in range(12)]
        session = FakeSession(frames[:5] + readings + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        clock = [100.0]
        observe = session.observe

        def slow_observe(label="observe", **options):
            if label == "battle-settlement-reread":
                clock[0] += 10
            return observe(label, **options)

        session.observe = slow_observe
        with patch("autococ.combat.time.monotonic", side_effect=lambda: clock[0]):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["settlement_reads"][-1]["error"], "Settlement reread window exceeded")
        self.assertLess(len(result.metrics["settlement_reads"]), 12)
        self.assertTrue(result.metrics["returned_home"])

    def test_line_extra_reread_still_stops_on_user_interrupt(self):
        frames = complete_frames()
        invalid = deepcopy(SETTLEMENT)
        invalid["stars"] = None
        readings = [frame(50 + index, "settlement", settlement=deepcopy(invalid)) for index in range(12)]
        session = FakeSession(frames[:5] + readings + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        observe = session.observe

        def stop_at_seventh(label="observe", **options):
            if label == "battle-settlement-reread" and session.observe_labels.count(label) == 6:
                raise StopRequested("stop during extra reread")
            return observe(label, **options)

        session.observe = stop_at_seventh
        with self.assertRaises(StopRequested):
            battle(session)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_line_extra_reread_account_conflict_never_uses_old_return_control(self):
        frames = complete_frames()
        invalid = deepcopy(SETTLEMENT)
        invalid["stars"] = None
        readings = [frame(50 + index, "settlement", settlement=deepcopy(invalid)) for index in range(6)]
        conflict = frame(56, "account_in_use")
        session = FakeSession(frames[:5] + readings + [conflict])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIs(session.last_snapshot, conflict)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_line_result_does_not_click_while_valid_values_keep_changing(self):
        frames = complete_frames()
        changing = []
        for index in range(12):
            result = deepcopy(SETTLEMENT)
            result["loot"]["gold"] += index
            changing.append(frame(50 + index, "settlement", settlement=result))
        session = FakeSession(frames[:5] + changing + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("not stabilized", result.reason)
        self.assertEqual(len(result.metrics["settlement_reads"]), 12)
        self.assertTrue(result.metrics["returned_home"])
        self.assertNotIn("loot_gold", result.metrics)
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason],
                         [changing[-1].screenshot_path])

    def test_line_result_same_frame_replay_cannot_prove_stability(self):
        frames = complete_frames()
        repeated = [frame(50, "settlement", settlement=deepcopy(SETTLEMENT)) for _ in range(12)]
        session = FakeSession(frames[:5] + repeated + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("not stabilized", result.reason)
        self.assertTrue(result.metrics["returned_home"])
        self.assertNotIn("loot_gold", result.metrics)

    def test_line_incomplete_result_can_exit_only_from_latest_settlement(self):
        frames = complete_frames()
        partial = deepcopy(SETTLEMENT)
        partial["loot"]["dark_elixir"] = None
        readings = [frame(50 + index, "settlement", settlement=deepcopy(partial)) for index in range(12)]
        session = FakeSession(frames[:5] + readings + frames[6:])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(len(result.metrics["settlement_reads"]), 12)
        self.assertIsNone(result.metrics.get("victory"))
        self.assertNotIn("loot_gold", result.metrics)
        self.assertFalse(result.metrics["inventory_reconciled"])
        self.assertEqual([path for path, reason in session.taps if "return_home" in reason],
                         [readings[-1].screenshot_path])

    def test_line_unknown_latest_frame_never_uses_older_return_button(self):
        frames = complete_frames()
        partial = deepcopy(SETTLEMENT)
        partial["loot"]["dark_elixir"] = None
        readings = [frame(50 + index, "settlement", settlement=deepcopy(partial)) for index in range(5)]
        session = FakeSession(frames[:5] + readings + [frame(55, "unknown")])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.metrics.get("returned_home", False))
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_failed_line_reconciles_exact_settlement_and_home_without_claiming_success(self):
        frames = complete_frames()
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        def fail_deployment(session, scout):
            session.observe("partially-deployed")
            raise DeploymentError("Troop quantity unverified", partial_receipt={
                "completed": False, "verified": True, "deployed_units": 1,
                "issued_placements": 1, "evidence": ["combat-4.png"]})
        session.deploy = fail_deployment
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Troop quantity unverified")
        self.assertTrue(result.metrics["returned_home"])
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(result.metrics["inventory_delta"],
                         {"gold": 110, "elixir": 55, "dark_elixir": 10, "gems": 0})

    def test_attempted_line_capture_failure_clears_snapshot_then_recovers_only_from_new_battle(self):
        frames = complete_frames()
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous")
        session.deadline = session.task_deadline = __import__("time").monotonic() + 600
        original_observe = session.observe
        failed_once = False
        def observe(label="observe", **options):
            nonlocal failed_once
            if label == "partial-deployment-recovery" and not failed_once:
                failed_once = True
                session.last_snapshot = None  # Actual GameSession invalidates before capture.
                raise CaptureError("capture transport unavailable")
            return original_observe(label, **options)
        session.observe = observe
        def fail_deployment(active, scout):
            active.last_snapshot = None
            raise DeploymentError("Probe post capture failed", partial_receipt={
                "completed": False, "verified": False, "deployed_units": 0,
                "attempted_placements": 1, "issued_placements": 1,
                "evidence": ["probe-before.png"]})
        session.deploy = fail_deployment
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Probe post capture failed")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["deployment"]["attempted_placements"], 1)
        self.assertEqual(sum(name == "partial_deployment_recovery" for name, _ in session.events), 2)
        self.assertFalse(any("probe" in reason for _, reason in session.taps))

    def test_attempted_line_unknown_recovery_never_clicks_or_replays(self):
        frames = complete_frames()[:4] + [frame(40, "unknown"), frame(41, "unknown")]
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous")
        session.deadline = session.task_deadline = __import__("time").monotonic() + 600
        def fail_deployment(active, scout):
            active.last_snapshot = None
            raise DeploymentError("Probe transport uncertain", partial_receipt={
                "attempted_placements": 1, "issued_placements": 0,
                "deployed_units": 0, "evidence": ["probe-before.png"]})
        session.deploy = fail_deployment
        taps_before = len(session.taps)
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Probe transport uncertain")
        self.assertEqual(session.observe_labels.count("partial-deployment-recovery"), 2)
        self.assertFalse(result.metrics.get("returned_home", False))
        self.assertFalse(any("end_battle" in reason or "return_home" in reason
                             for _, reason in session.taps[taps_before:]))

    def test_account_conflict_partial_receipt_stops_without_recovery_observation(self):
        session = FakeSession(complete_frames()[:4])
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous")
        def fail_deployment(active, scout):
            active.last_snapshot = None
            raise DeploymentError("Game interruption: account_in_use", partial_receipt={
                "interruption": "account_in_use", "attempted_placements": 1,
                "issued_placements": 1, "evidence": ["probe-before.png"]})
        session.deploy = fail_deployment
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["deployment"]["interruption"], "account_in_use")
        self.assertNotIn("partial-deployment-recovery", session.observe_labels)
        self.assertFalse(result.metrics.get("returned_home", False))

    def test_account_conflict_discovered_during_recovery_retains_partial_receipt(self):
        session = FakeSession(complete_frames()[:4] + [frame(40, "account_in_use")])
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous")
        session.deadline = session.task_deadline = __import__("time").monotonic() + 600
        def fail_deployment(active, scout):
            active.last_snapshot = None
            raise DeploymentError("Probe post capture failed", partial_receipt={
                "mode": "model_free_line_probed_anchors", "attempted_placements": 1,
                "issued_placements": 1, "deployed_units": 0,
                "probe_events": [{"edge": 0, "status": "input_sent"}],
                "evidence": ["probe-before.png"]})
        session.deploy = fail_deployment
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Probe post capture failed")
        self.assertEqual(result.metrics["deployment"]["probe_events"][0]["status"], "input_sent")
        self.assertEqual(result.metrics["deployment_recovery_error"], "account_in_use")
        self.assertEqual(session.observe_labels.count("partial-deployment-recovery"), 1)
        self.assertFalse(result.metrics.get("returned_home", False))
        self.assertFalse(any("return_home" in reason or "end_battle" in reason
                             for _, reason in session.taps))

    def test_line_never_reconciles_unverified_start_inventory(self):
        for source, resources in (("village_inventory", {**BEFORE, "dark_elixir": None}),
                                  ("enemy_available", deepcopy(BEFORE)),
                                  ("village_inventory", None)):
            with self.subTest(source=source, resources=resources):
                frames = complete_frames()
                frames[0].observations["resource_source"] = source
                frames[0].observations["resources"] = resources
                session = FakeSession(frames)
                session.config.battle = replace(session.config.battle, strategy="edrag_line")
                result = battle(session)
                self.assertEqual(result.status, "failed", result.reason)
                self.assertTrue(result.metrics["returned_home"])
                self.assertIsNotNone(result.metrics["settlement_observed"])
                self.assertFalse(result.metrics["inventory_reconciled"])
                self.assertNotIn("inventory_delta", result.metrics)
                self.assertIsNone(result.metrics.get("victory"))
                self.assertIsNone(result.metrics.get("stars"))
                self.assertNotIn("loot_gold", result.metrics)
                self.assertNotIn("bonus_gold", result.metrics)
                self.assertFalse(any(name == "battle_round_verified" for name, _ in session.events))

    def test_line_mismatched_return_inventory_does_not_publish_win_or_rewards(self):
        frames = complete_frames()
        frames[6].observations["resources"]["gold"] += 1
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertEqual(result.metrics["settlement_observed"]["stars"], 2)
        self.assertFalse(result.metrics["inventory_reconciled"])
        self.assertIsNone(result.metrics.get("victory"))
        self.assertIsNone(result.metrics.get("stars"))
        self.assertNotIn("loot_gold", result.metrics)
        self.assertNotIn("bonus_gold", result.metrics)
        self.assertFalse(any(name == "battle_round_verified" for name, _ in session.events))
        ledger = resource_metrics(RunStats(task_results=[result]), elapsed_sec=60)
        self.assertIsNone(ledger["resources"]["gold"]["battle_loot"])

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

    def test_line_ledger_uses_fresh_home_after_navigation_controls_settle(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"]["bonus"] = {
            "requires_inventory_reconciliation": True}
        frames[6].observations["buttons"] = [
            button for button in frames[6].observations["buttons"] if button["name"] == "shop"]
        frames[6].observations["resources"]["gold"] = 0
        fresh = frame(7, "village", resources=deepcopy(AFTER))
        session = FakeSession(frames + [fresh])
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        result = battle(session)
        self.assertEqual(result.status, "succeeded", result.reason)
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(result.metrics["resources_after"], AFTER)
        self.assertIn(fresh.screenshot_path, result.evidence)
        self.assertIn("home-controls-reread", session.observe_labels)

    def test_new_battle_clears_old_frozen_line_receipt_before_preflight(self):
        session = FakeSession(complete_frames())
        session.frozen_line_receipt = {"old_battle": True}
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        session.native = None
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(session.frozen_line_receipt)

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

    def test_expired_preview_without_placements_waits_for_result_but_stays_failed(self):
        frames = complete_frames()
        frames[5].observations["settlement"].update(stars=0, percentage=0)
        frames[5].observations["settlement"]["evidence"]["stars"] = {
            "source": "explicit_defeat_result", "result": {"confidence": .99}}
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        def fail_deployment(session, scout):
            session.observe("expired-preview")
            raise DeploymentError("Deployment boundaries unverified", partial_receipt={
                "completed": False, "verified": False, "deployed_units": 0,
                "issued_placements": 0, "evidence": ["combat-4.png"]})
        session.deploy = fail_deployment
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIn("boundaries unverified", result.reason)
        self.assertTrue(result.metrics["returned_home"])
        self.assertFalse(result.metrics["victory"])
        self.assertEqual(result.metrics["deployment"]["issued_placements"], 0)
        self.assertEqual(session.taps[-1][1], "Recognized return_home: return_home")

    def _continuous_preparation_failure(self, frames, *, attempted=0, issued=0,
                                        failure=None, continuous=True):
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                        line_execution_mode="continuous" if continuous else "verified")
        partial = {"completed": False, "verified": False, "deployed_units": 0,
                   "attempted_placements": attempted, "issued_placements": issued,
                   "evidence": [str(frames[3].screenshot_path)]}

        def fail_before_placement(active, scout):
            active.deploy_calls += 1
            if failure is not None:
                raise failure
            raise DeploymentError("Continuous plan could not be frozen", partial_receipt=partial)

        session.deploy = fail_before_placement
        # This unit test exercises failure cleanup after the separate army
        # identity gate; no model-free plan or game input is constructed.
        with patch("autococ.combat._read_continuous_line_army",
                   side_effect=lambda active, current, evidence: (current, _read_army(current))):
            return session, battle(session)

    def test_unstarted_continuous_line_exits_only_from_fresh_enemy_preview(self):
        frames = complete_frames()[:4] + [frame(7, "enemy_village"),
                 frame(8, "village"), frame(9, "village", resources=deepcopy(BEFORE))]
        session, result = self._continuous_preparation_failure(frames)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Continuous plan could not be frozen")
        self.assertEqual(result.metrics["deployment"]["attempted_placements"], 0)
        self.assertTrue(result.metrics["returned_home"])
        self.assertNotIn("rounds_completed", result.metrics)
        self.assertEqual(session.deploy_calls, 1)
        self.assertIn("continuous-line-failure-check", session.observe_labels)
        exits = [(path, reason) for path, reason in session.taps if "end_battle" in reason]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0][0], frames[4].screenshot_path)
        self.assertIn(frames[4].screenshot_path, result.evidence)
        self.assertIn(frames[-1].screenshot_path, result.evidence)

    def test_unstarted_continuous_line_expired_preview_waits_for_settlement(self):
        settled = frame(8, "settlement", settlement=deepcopy(SETTLEMENT))
        home = frame(9, "village", resources=deepcopy(AFTER))
        session, result = self._continuous_preparation_failure(
            complete_frames()[:4] + [frame(7, "battle"), settled, home])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "Continuous plan could not be frozen")
        self.assertTrue(result.metrics["returned_home"])
        self.assertEqual(result.metrics["rounds_completed"], 1)
        self.assertEqual(result.metrics["deployment"]["deployed_units"], 0)
        self.assertIn("battle-settlement", session.observe_labels)
        self.assertFalse(any("end_battle" in reason for _, reason in session.taps))

    def test_unstarted_continuous_line_unknown_or_weak_scene_cannot_authorize_input(self):
        for current in (frame(7, "unknown"), frame(7, "settlement"),
                        replace(frame(7, "enemy_village"), confidence=.7),
                        replace(frame(7, "battle"), confidence=.7)):
            with self.subTest(scene=current.scene, confidence=current.confidence):
                session, result = self._continuous_preparation_failure(
                    complete_frames()[:4] + [current])
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "Continuous plan could not be frozen")
                self.assertFalse(result.metrics.get("returned_home", False))
                self.assertIn("deployment_recovery_error", result.metrics)
                self.assertEqual(session.deploy_calls, 1)
                self.assertEqual([reason for _, reason in session.taps], [
                    "Recognized attack: attack",
                    "Recognized find_match_regular: find_match_regular",
                    "Recognized attack: attack"])
                self.assertFalse(any("end_battle" in reason or "return_home" in reason
                                     for _, reason in session.taps))

    def test_continuous_line_attempted_placement_does_not_use_unstarted_exit(self):
        for attempted, issued, continuous in ((1, 0, True), (0, 1, True), (0, 0, False)):
            with self.subTest(attempted=attempted, issued=issued, continuous=continuous):
                session, result = self._continuous_preparation_failure(
                    complete_frames()[:4] + [frame(7, "enemy_village")],
                    attempted=attempted, issued=issued, continuous=continuous)
                self.assertEqual(result.status, "failed")
                self.assertNotIn("continuous-line-failure-check", session.observe_labels)
                self.assertFalse(any("end_battle" in reason for _, reason in session.taps))

    def test_continuous_line_stop_bypasses_failure_cleanup(self):
        frames = complete_frames()[:4] + [frame(7, "enemy_village")]
        with self.assertRaises(StopRequested):
            self._continuous_preparation_failure(frames, failure=StopRequested("stop"))

    def test_unrecognized_expired_preview_does_not_authorize_recovery_input(self):
        frames = complete_frames()
        frames[4] = frame(4, "unknown")
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="edrag_line")
        def fail_deployment(session, scout):
            session.observe("unknown-after-preparation")
            raise DeploymentError("Deployment boundaries unverified", partial_receipt={
                "completed": False, "verified": False, "deployed_units": 0,
                "issued_placements": 0, "evidence": ["combat-4.png"]})
        session.deploy = fail_deployment
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertNotIn("returned_home", result.metrics)
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

    def test_two_edge_does_not_claim_provisional_zero_rewards_as_verified(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["evidence"]["bonus"] = {"requires_inventory_reconciliation": True}
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.metrics["inventory_reconciled"])
        self.assertEqual(result.metrics["bonus_gold"], 20)
        self.assertTrue(result.metrics["victory"])
        self.assertEqual(result.metrics["stars"], 2)

    def test_provisional_two_row_result_is_not_published_without_exact_ledger(self):
        for before, source, after in (
            (deepcopy(BEFORE), "village_inventory", {**AFTER, "gold": AFTER["gold"] + 1}),
            ({**BEFORE, "dark_elixir": None}, "village_inventory", deepcopy(AFTER)),
            (deepcopy(BEFORE), "enemy_available", deepcopy(AFTER)),
        ):
            with self.subTest(before=before, source=source, after=after):
                frames = complete_frames()
                frames[0].observations.update(resources=before, resource_source=source)
                frames[5].observations["settlement"]["evidence"].setdefault("loot", {})["dark_elixir"] = {
                    "source": "verified_absent_resource_row_layout",
                    "requires_inventory_reconciliation": True}
                frames[5].observations["settlement"]["evidence"]["bonus"] = {
                    "source": "verified_no_bonus_layout",
                    "requires_inventory_reconciliation": True}
                frames[6].observations["resources"] = after
                session = FakeSession(frames)
                session.config.battle = replace(session.config.battle, strategy="edrag_line")
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertTrue(result.metrics["returned_home"])
                self.assertFalse(result.metrics["inventory_reconciled"])
                self.assertIsNone(result.metrics.get("victory"))
                self.assertIsNone(result.metrics.get("stars"))
                self.assertNotIn("loot_gold", result.metrics)
                self.assertNotIn("bonus_gold", result.metrics)
                self.assertIn("exact inventory reconciliation", result.reason)
                ledger = resource_metrics(RunStats(task_results=[result]), elapsed_sec=60)
                for resource in ("gold", "elixir", "dark_elixir"):
                    self.assertIsNone(ledger["resources"][resource]["battle_loot"])
                    self.assertIsNone(ledger["resources"][resource]["battle_bonus"])

    def test_two_edge_unknown_loot_prevents_unverified_return(self):
        frames = complete_frames()
        frames[5].observations["settlement"]["loot"]["gold"] = None
        session = FakeSession(frames)
        session.config.battle = replace(session.config.battle, strategy="two_edge")
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.metrics.get("loot_gold"))
        self.assertIsNotNone(result.metrics["settlement_validation_error"])
        self.assertFalse(any("return_home" in reason for _, reason in session.taps))

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

    def test_continuous_missing_troop_identity_never_pays_for_search(self):
        for defect in ("missing_id", "wrong_version", "unverified_version", "unknown_version",
                       "wrong_identity", "weak_identity", "invalid_evidence", "wrong_source"):
            with self.subTest(defect=defect):
                frames = complete_frames()[:3]
                army = frames[2].observations["army"]
                troop = {"kind": "troop", "source": "army", "unit_id": "archer", "count": 3,
                         "evidence": {"identity": {"unit_id": "archer", "confidence": .99,
                             "version_verified": True, "template_client_version": "18.647.2"}}}
                identity = troop["evidence"]["identity"]
                if defect == "missing_id":
                    troop["unit_id"] = None
                elif defect == "wrong_version":
                    identity["template_client_version"] = "18.600.7"
                elif defect == "unverified_version":
                    identity["version_verified"] = False
                elif defect == "unknown_version":
                    identity["template_client_version"] = "unknown"
                elif defect == "wrong_identity":
                    identity["unit_id"] = "barbarian"
                elif defect == "weak_identity":
                    identity["confidence"] = .89
                elif defect == "invalid_evidence":
                    troop["evidence"] = "unverified"
                else:
                    troop["source"] = "clan"
                army["manifest"]["troops"] = [troop]
                army["hero_evidence"] = {"complete": True, "frame": str(frames[2].screenshot_path),
                    "cards": [{"kind": "hero", "source": "army", "unit_id": "archer_queen",
                        "evidence": {"identity": {"unit_id": "archer_queen", "confidence": .99,
                            "version_verified": True, "template_client_version": "18.647.2"}}}]}
                for index in (7, 8):
                    repeated = frame(index, "training", army=deepcopy(army))
                    repeated.observations["army"]["manifest"]["frame"] = str(repeated.screenshot_path)
                    repeated.observations["army"]["hero_evidence"]["frame"] = str(repeated.screenshot_path)
                    frames.append(repeated)
                session = FakeSession(frames)
                session.client_version = "unknown" if defect == "unknown_version" else "18.647.2"
                session.config.battle = replace(session.config.battle, strategy="edrag_line",
                                                line_execution_mode="continuous")
                home = frame(9, "village", resources=deepcopy(BEFORE))
                frames.append(home)
                result = battle(session)
                self.assertEqual(result.status, "failed")
                self.assertIn("no paid search", result.reason)
                self.assertEqual(result.metrics["search_cost_gold"], 0)
                self.assertEqual(session.deploy_calls, 0)
                self.assertFalse(any(path in {Path("combat-2.png"), Path("combat-7.png"),
                                             Path("combat-8.png")} for path, _ in session.taps))
                self.assertEqual(session.observe_labels.count("battle-army-roster-reread"), 2)

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


class BattleRewardIntegrationTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("autococ.combat._wait_for_clear_scout",
                        side_effect=lambda session, snapshot, deadline: snapshot)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_claim_branch_keeps_battle_ledger_and_separate_potion_receipt(self):
        for completed in (True, False):
            with self.subTest(completed=completed):
                frames = complete_frames()
                frames[4].observations["deployment"]["completed"] = completed
                frames[5].observations["buttons"] = [{"name": "claim_battle_reward", "point": [613, 621],
                                                       "text": "领取奖励"}]
                frames[5].observations["battle_reward_claim"] = {"surface": "battle_reward_claim_settlement"}
                session = FakeSession(frames)
                potion = {"kind": "battle_chest_ui", "status": "ui_complete_home_verified",
                          "reward_item": "hero_potion", "reward_amount": None,
                          "reward_inventory_verified": False, "returned_home": True}

                def finish(active, settlement, evidence):
                    self.assertIs(settlement, frames[5])
                    return active.observe("battle-chest-home"), potion

                with patch("autococ.battle_rewards.finish_claimed_battle_reward", side_effect=finish) as handler:
                    result = battle(session)
                handler.assert_called_once()
                self.assertEqual(result.metrics["battle_reward"], potion)
                self.assertEqual(result.metrics["inventory_delta"], {"gold": 110, "elixir": 55,
                                                                      "dark_elixir": 10, "gems": 0})
                self.assertTrue(result.metrics["inventory_reconciled"])
                self.assertTrue(result.metrics["returned_home"])
                self.assertFalse(any("return_home" in reason for _, reason in session.taps))
                self.assertEqual(result.status, "succeeded" if completed else "failed")

    def test_old_chest_receipt_does_not_leak_into_later_non_chest_failure(self):
        session = FakeSession([])
        old = {"kind": "battle_chest_ui", "status": "ui_complete_home_verified", "returned_home": True}
        session.battle_reward_receipt = old
        result = battle(session)
        self.assertEqual(result.status, "failed")
        self.assertNotIn("battle_reward", result.metrics)
        self.assertIs(session.battle_reward_receipt, old)


if __name__ == "__main__":
    unittest.main()
