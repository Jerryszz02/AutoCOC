"""The frozen queue is checked with simulated time and transport, never a game."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autococ.errors import FlowError, StopRequested
from autococ.prepared_battle import execute_prepared_battle, prepare_battle
from autococ.scene import SceneSnapshot
from autococ.session import GameSession


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


class PreparedBattleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.scout_path = self.root / "scout.png"
        self.post_path = self.root / "post.png"
        self.cards = [{"kind": "troop", "unit_id": "archer", "source": "army", "count": 2,
                       "point": [110, 630]}]
        self.profile = {"accepted": True, "cards_stable_after_deployment": True,
                        "baseline_resolution": [1280, 720], "client_version": "18.1",
                        "transport": "mumu_native", "display_id": 2, "model_sha256": "modelhash",
                        "zoom": "minimum_stable",
                        "model_validation_report_id": "offline-1",
                        "cards": self.cards, "candidate_orders": [["archer"]],
                        "core_classes": {"town_hall": .9}}
        (self.root / "layout.json").write_text(json.dumps(self.profile), encoding="utf-8")
        self.terrain = [
            {"edge": edge, "point": point, "evidence": {"frame": str(self.scout_path)}}
            for edge, pair in ((0, ([250, 240], [350, 160])), (1, ([250, 420], [350, 500])))
            for point in pair]
        battle_slot = {**self.cards[0], "confidence": .99,
                       "evidence": {"count": {"confidence": .99}}}
        self.scout = SceneSnapshot("enemy_village", .99, self.scout_path, {
            "observed_at_monotonic": self.clock(),
            "minimum_zoom": {"verified": True}, "terrain": self.terrain,
            "battle": {"countdown_seconds": 30, "countdown_evidence": {
                "frame": str(self.scout_path), "confidence": .99, "source": "current_frame_ocr"},
                       "slots": [battle_slot]},
            "expected_army_manifest": {"complete": True, "identity_complete": True,
                "troops": [{"unit_id": "archer", "count": 2}], "spells": []}})
        post_slot = {**battle_slot, "count": 0}
        self.post = SceneSnapshot("battle", .99, self.post_path, {
            "battle": {"slots": [post_slot]}, "observed_at_monotonic": self.clock()})
        self.session = SimpleNamespace(
            config=SimpleNamespace(
                source_path=self.root / "config.toml",
                game=SimpleNamespace(baseline_resolution=(1280, 720)),
                runtime=SimpleNamespace(step_timeout_sec=2.),
                vision_agent=SimpleNamespace(layout_profile=str(self.root / "layout.json"),
                                             preparation_reserve_sec=5.)),
            capture=SimpleNamespace(input_display_id=2), native=object(), client_version="18.1",
            context=SimpleNamespace(screen_resolution=(1280, 720)),
            last_snapshot=self.scout, deadline=999., task_deadline=999., evidence_budget=None)
        self.session._validate_snapshot = lambda frame: None if frame is self.session.last_snapshot else (_ for _ in ()).throw(FlowError("stale"))
        self.session.check_deadline = lambda: None
        self.events = []
        self.session.event = lambda kind, **data: self.events.append((kind, data))
        self.inputs = []
        self.observations = []

        def tap(point, *, timeout_sec, reason, plan_id):
            self.inputs.append((point, timeout_sec, reason))
            self.clock.now += .05

        def observe(label):
            self.observations.append(label)
            self.post.observations["observed_at_monotonic"] = self.clock()
            self.session.last_snapshot = self.post
            return self.post

        self.session.frozen_battle_tap = tap
        self.session.observe = observe
        self.model = SimpleNamespace(model_sha256="modelhash", detect=Mock(return_value=()),
                                     metadata={"applicability": {"zoom": "minimum_stable"},
                                               "validation": {"status": "evaluated", "report_id": "offline-1"}})
        self.core = SimpleNamespace(center=(600., 260.), confidence=.99)
        self.patches = [patch("autococ.prepared_battle.measure_cloud_cover", return_value={"obscured": False}),
                        patch("autococ.images.read_frame", return_value=np.zeros((720, 1280, 3), dtype=np.uint8)),
                        patch("autococ.model_vision.infer_core_geometry", return_value=self.core)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def prepare(self):
        return prepare_battle(self.session, self.scout, model=self.model, clock=self.clock)

    def test_prepare_execute_verify_and_single_use_without_observation_in_queue(self):
        plan = self.prepare()
        self.assertEqual(plan.input_count, 3)
        self.assertEqual(plan.burst_limit_sec, 1.75)
        self.assertEqual(self.observations, [])
        result = execute_prepared_battle(self.session, plan, clock=self.clock)
        self.assertEqual(len(self.inputs), 3)
        self.assertEqual(self.observations, ["prepared-battle-post"])
        self.assertEqual(self.model.detect.call_count, 1)
        self.assertTrue(result.observations["deployment"]["verified"])
        self.assertTrue(result.observations["deployment"]["burst_pass"])
        with self.assertRaises(FlowError):
            execute_prepared_battle(self.session, plan, clock=self.clock)
        self.assertEqual(len(self.inputs), 3)

    def test_tampered_execution_window_and_unknown_card_fail_before_input(self):
        plan = self.prepare()
        with self.assertRaisesRegex(FlowError, "modified"):
            execute_prepared_battle(self.session, replace(plan, burst_limit_sec=50), clock=self.clock)
        self.assertEqual(self.inputs, [])
        self.scout.observations["battle"]["slots"].append({"kind": "spell", "count": None})
        with self.assertRaisesRegex(FlowError, "uncertain"):
            self.prepare()

    def test_slow_input_stops_queue_and_never_claims_completed(self):
        plan = self.prepare()

        def slow(point, *, timeout_sec, reason, plan_id):
            self.inputs.append(point)
            self.clock.now += plan.burst_limit_sec + .01

        self.session.frozen_battle_tap = slow
        result = execute_prepared_battle(self.session, plan, clock=self.clock)
        receipt = result.observations["deployment"]
        self.assertEqual(len(self.inputs), 1)
        self.assertFalse(receipt["completed"])
        self.assertFalse(receipt["burst_pass"])
        self.assertEqual(self.observations, ["prepared-battle-post"])

    def test_uncertain_transport_stops_without_replay(self):
        plan = self.prepare()

        def uncertain(point, *, timeout_sec, reason, plan_id):
            self.inputs.append(point)
            if len(self.inputs) == 2:
                raise FlowError("transport result uncertain")

        self.session.frozen_battle_tap = uncertain
        result = execute_prepared_battle(self.session, plan, clock=self.clock)
        receipt = result.observations["deployment"]
        self.assertEqual(len(self.inputs), 2)
        self.assertFalse(receipt["completed"])
        self.assertEqual(receipt["attempted_placements"], 1)
        self.assertEqual(receipt["actions"][-1]["status"], "uncertain")
        with self.assertRaises(FlowError):
            execute_prepared_battle(self.session, plan, clock=self.clock)

    def test_new_plan_from_already_used_frame_and_direct_input_are_rejected(self):
        plan = self.prepare()
        execute_prepared_battle(self.session, plan, clock=self.clock)
        self.session.last_snapshot = self.scout
        with self.assertRaisesRegex(FlowError, "already started"):
            self.prepare()
        with self.assertRaisesRegex(FlowError, "no active plan"):
            GameSession.frozen_battle_tap(self.session, (100, 200), timeout_sec=1,
                                          reason="bypass", plan_id="forged")

    def test_stop_preserves_partial_receipt_without_post_capture(self):
        plan = self.prepare()

        def stop(point, **kwargs):
            self.inputs.append(point)
            if len(self.inputs) == 2:
                raise StopRequested("user stopped")

        self.session.frozen_battle_tap = stop
        with self.assertRaises(StopRequested):
            execute_prepared_battle(self.session, plan, clock=self.clock)
        self.assertEqual(self.observations, [])
        self.assertEqual(self.session.prepared_battle_receipt["attempted_placements"], 1)
        self.assertFalse(self.session.prepared_battle_receipt["completed"])
        self.assertEqual(self.session.prepared_battle_receipt["input_count"], plan.input_count)
        self.assertEqual(self.session.prepared_battle_receipt["plan_sha256"], plan.plan_sha256)
        self.assertFalse(self.session.prepared_battle_receipt["burst_pass"])

    def test_refreshed_zoom_frame_keeps_independent_manifest(self):
        self.scout.observations.pop("minimum_zoom")
        current_path = self.root / "after-zoom.png"
        updated = dict(self.scout.observations)
        updated.pop("expected_army_manifest")
        updated["battle"] = {**updated["battle"], "countdown_evidence": {
            "frame": str(current_path), "confidence": .99, "source": "current_frame_ocr"}}
        zoomed = SceneSnapshot("enemy_village", .99, current_path, updated)
        terrain = [{**item, "evidence": {"frame": str(current_path)}} for item in self.terrain]
        def refresh(session, scout, receipt):
            receipt["minimum_zoom"] = {"verified": True}
            self.session.last_snapshot = zoomed
            return zoomed, terrain

        with patch("autococ.deployment._prepare_two_edge_view", side_effect=refresh):
            plan = self.prepare()
        self.assertEqual(plan.frame.frame_id, str(current_path))
        self.assertEqual(zoomed.observations["expected_army_manifest"]["troops"][0]["count"], 2)

    def test_unknown_hero_requires_explicit_exclusion_and_countdown_frame_is_bound(self):
        self.scout.observations["battle"]["slots"].append({"kind": "hero", "count": None})
        with self.assertRaisesRegex(FlowError, "uncertain"):
            self.prepare()
        self.profile["excluded_kinds"] = ["hero"]
        (self.root / "layout.json").write_text(json.dumps(self.profile), encoding="utf-8")
        self.prepare()
        self.scout.observations["battle"]["countdown_evidence"]["frame"] = "old.png"
        with self.assertRaisesRegex(FlowError, "countdown"):
            self.prepare()

    def test_core_driven_funnel_candidates_and_jev_order_are_finite(self):
        wizard = {"kind": "troop", "unit_id": "wizard", "source": "army", "count": 1,
                  "point": [220, 630]}
        self.scout.observations["battle"]["slots"].append({**wizard, "confidence": .99,
            "evidence": {"count": {"confidence": .99}}})
        self.scout.observations["expected_army_manifest"]["troops"].append(
            {"unit_id": "wizard", "count": 1})
        self.profile.update(cards=self.cards + [wizard],
                            candidate_orders=[["archer", "wizard"], ["wizard", "archer"]],
                            funnel_units=["archer"])
        (self.root / "layout.json").write_text(json.dumps(self.profile), encoding="utf-8")
        decider = Mock()
        decider.select.return_value = SimpleNamespace(candidate_id="main-0-funnel-1-order-1",
            source="jev", reason="validated_choice", model="jev-1.13.0",
            elapsed_sec=.05, confidence=.95)
        plan = prepare_battle(self.session, self.scout, model=self.model,
                              decider=decider, clock=self.clock)
        self.assertEqual(len(decider.select.call_args.args[1]), 8)
        self.assertEqual([action.unit_id for action in plan.actions if action.kind == "select"],
                         ["wizard", "archer"])
        self.assertEqual([action.point for action in plan.actions if action.unit_id == "archer"
                          and action.kind == "deploy"], [(250, 420), (350, 500)])
        self.assertEqual(plan.decision_source, "jev")
        self.assertEqual(plan.decision_model, "jev-1.13.0")


if __name__ == "__main__":
    unittest.main()
