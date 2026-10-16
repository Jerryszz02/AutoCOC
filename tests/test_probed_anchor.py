"""Probe safety contracts and lower-toolbar replay; no live deployment proof."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.errors import DeploymentError, FlowError, StopRequested
from autococ.frozen_line import FrozenLinePlan, _file_sha256, _plan_digest
from autococ.prepared_battle import FramePacket, PreparedInput
from autococ.probed_anchor import (ExactCameraError, _balloon_count, _current_frame, _frame_envelope,
                                   _remaining_roster_checkpoint,
                                   _spent_balloon_dark_border,
                                   _current_card_edges,
                                   execute_probed_anchor_line,
                                   freeze_probed_anchor_line, remainder_burst_limit,
                                   verify_exact_anchor_camera)
from autococ.scene import SceneSnapshot
from autococ.strategies import BattleContext, TwoEdgeSplitStrategy


TERRAIN = ({"edge": 0, "point": (400, 200)}, {"edge": 0, "point": (350, 300)},
           {"edge": 1, "point": (500, 400)}, {"edge": 1, "point": (550, 500)})
TROOPS = (("electro_dragon", 10), ("dragon", 1), ("baby_dragon", 1),
          ("balloon", 2), ("archer", 3))
HEROES = ("grand_warden", "archer_queen", "dragon_duke", "minion_prince")
FIXTURES = Path(__file__).parent / "fixtures"


def image_frame(tmp_path, name, scene, observed, *, fast=False):
    path = tmp_path / f"{name}.png"
    path.write_bytes(f"frame {name}".encode())
    return SceneSnapshot(scene, .99, path, {"observed_at_monotonic": observed,
        "source_resolution": (2560, 1440), "baseline_resolution": (1280, 720),
        **({"recognition_scope": ("battle_hud" if fast == "battle_hud" else "troop_count"), "fast_fallback": False,
            "matched_reasons": ["two_current_hud_anchors"]} if fast else {}),
        "battle": {"slots": []}})


def frame_proof(snapshot):
    return {"frame": str(snapshot.screenshot_path),
            "sha256": _file_sha256(snapshot.screenshot_path),
            "observed_at": snapshot.observations["observed_at_monotonic"],
            "camera": {"stationary": True}}


def bound_camera_error(snapshot, message):
    error = ExactCameraError(message)
    error.frame_proof = {key: value for key, value in frame_proof(snapshot).items() if key != "camera"}
    return error


def frozen_base(tmp_path):
    source = image_frame(tmp_path, "source", "enemy_village", 9.)
    before = image_frame(tmp_path, "before", "enemy_village", 8.)
    troop_cards = tuple((unit_id, "army", count, (150 + index * 90, 650),
                         ((108, 589, 193, 708) if index == 0 else
                          (110 + index * 90, 595, 190 + index * 90, 711)))
                        for index, (unit_id, count) in enumerate(TROOPS))
    heroes = tuple((hero_id, (600 + index * 80, 595, 680 + index * 80, 710))
                  for index, hero_id in enumerate(HEROES))
    troops = tuple({"unit_id": unit_id, "source": "army", "count": count,
                    "point": card[3]} for card, (unit_id, count) in zip(troop_cards, TROOPS))
    hero_rows = tuple({"unit_id": hero_id, "source": "army", "point": (640 + index * 80, 650)}
                      for index, hero_id in enumerate(HEROES))
    placement = TwoEdgeSplitStrategy().build_plan(BattleContext(troops, hero_rows, TERRAIN))
    actions = []
    selected = "electro_dragon"
    for index, step in enumerate(placement.troops):
        if selected != step.card["unit_id"]:
            actions.append(PreparedInput(f"select-{index}", "select", step.card["point"],
                                         step.card["unit_id"], "army"))
        selected = step.card["unit_id"]
        for offset, point in enumerate(step.points):
            actions.append(PreparedInput(f"troop-{index}-{offset}", "deploy", point,
                                         step.card["unit_id"], "army", .12))
    for index, step in enumerate(placement.heroes):
        actions.append(PreparedInput(f"select-hero-{index}", "select", step.card["point"],
                                     step.card["unit_id"], "army"))
        actions.append(PreparedInput(f"hero-{index}", "deploy", step.points[0],
                                     step.card["unit_id"], "army", .12))
    packet = FramePacket(str(source.screenshot_path), 9., "enemy_village", .99,
                         source.screenshot_path, (1280, 720), (2560, 1440), 0)
    base = FrozenLinePlan("plan", "battle", packet, _file_sha256(source.screenshot_path),
                          str(before.screenshot_path), _file_sha256(before.screenshot_path),
                          "18.647.2", 9., 11., 10., "cards", "terrain", tuple(actions),
                          tuple((name, "army", count) for name, count in TROOPS), troop_cards,
                          HEROES, heroes, None, None, None, "zoom", "two_edge_split", "")
    return replace(base, plan_sha256=_plan_digest(base)), source


class Session:
    client_version = "18.647.2"
    deadline = 100.
    task_deadline = 100.

    def __init__(self, source, observations):
        self.last_snapshot = source
        self.observations = iter(observations)
        self.inputs = []
        self.observe_calls = []
        self.capture = SimpleNamespace(input_display_id=0)
        self.context = SimpleNamespace(screen_resolution=(2560, 1440))
        self.config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720)),
                                      runtime=SimpleNamespace(step_timeout_sec=5))

    def check_deadline(self):
        pass

    def tap(self, snapshot, point, *, reason):
        assert snapshot is self.last_snapshot
        self.inputs.append(("tap", tuple(point)))

    def frozen_battle_tap(self, point, *, timeout_sec, reason, plan_id):
        assert self._active_frozen_plan == (plan_id, self._active_frozen_plan[1], point)
        self.inputs.append(("frozen", tuple(point)))

    def observe(self, label, **_kwargs):
        self.observe_calls.append((label, _kwargs.get("purpose")))
        self.last_snapshot = None  # GameSession invalidates old frame before capture.
        next_frame = next(self.observations)
        if isinstance(next_frame, BaseException):
            raise next_frame
        if _kwargs.get("purpose") in {"troop_count", "battle_hud"} and "recognition_scope" not in next_frame.observations:
            next_frame.observations.update(recognition_scope="full", fast_fallback=True,
                                           recognition_requested=_kwargs["purpose"])
        self.last_snapshot = next_frame
        return next_frame


def make_plan(monkeypatch, tmp_path):
    base, source = frozen_base(tmp_path)
    monkeypatch.setattr("autococ.probed_anchor.freeze_model_free_line", lambda *_a, **_kw: base)
    monkeypatch.setattr("autococ.probed_anchor._cards", lambda *_a, **_kw: tuple(
        {"kind": "troop", "unit_id": unit_id, "source": source_kind,
         "count": count, "point": point,
         "bbox": (110, 595, 190, 711) if index == 0 else bbox}
        for index, (unit_id, source_kind, count, point, bbox) in enumerate(base.troop_cards)))
    monkeypatch.setattr("autococ.probed_anchor._foreground_now", lambda _session: None)
    monkeypatch.setattr("autococ.probed_anchor._remaining_roster_checkpoint",
                        lambda _session, _plan, snapshot: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    session = Session(source, [])
    before = SceneSnapshot("enemy_village", .99, Path(base.toolbar_before_frame_id),
                           {"observed_at_monotonic": 8., "source_resolution": (2560, 1440),
                            "baseline_resolution": (1280, 720), "battle": {"slots": []}})
    plan = freeze_probed_anchor_line(session, source, terrain=list(TERRAIN), toolbar_before=before)
    return plan, source


def test_frozen_plan_uses_only_two_precompiled_points_and_preserves_21_deployments(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    assert plan.anchors == ((375, 250), (525, 450))
    assert len([a for a in plan.remaining if a.kind == "deploy"]) == 19
    assert {a.point for a in plan.remaining if a.kind == "deploy"} == set(plan.anchors)
    assert sum(row[2] for row in plan.base.troop_budgets) == 17
    assert [a.unit_id for a in plan.remaining if a.kind == "deploy"].count("balloon") == 0
    assert len(plan.base.actions) == 30  # Existing two-edge split order is unchanged.
    assert remainder_burst_limit(plan) == 7.75
    assert plan.base.troop_cards[0][4] != plan.checkpoint_cards[0][2]
    assert plan.checkpoint_cards[0][2] == (110, 595, 190, 711)
    with pytest.raises(FlowError, match="permission"):
        execute_probed_anchor_line(Session(source, []),
                                   replace(plan, checkpoint_cards=()), clock=lambda: 10.)


def test_probed_plan_rejects_unsupported_baseline_before_freezing(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    session = Session(source, [])
    session.config.game.baseline_resolution = (1920, 1080)
    with pytest.raises(FlowError, match="1280"):
        freeze_probed_anchor_line(session, source, terrain=list(TERRAIN))


def test_checkpoint_geometry_must_come_from_the_frozen_before_frame(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    other = image_frame(tmp_path, "unrelated-toolbar", "enemy_village", 8.)
    with pytest.raises(FlowError, match="frozen toolbar frame"):
        freeze_probed_anchor_line(Session(source, []), source, terrain=list(TERRAIN),
                                  toolbar_before=other)
    assert plan.checkpoint_cards[0][2] != plan.base.troop_cards[0][4]


def test_two_probes_once_then_frozen_remainder_and_complete_receipt(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    final = image_frame(tmp_path, "final", "battle", 10.4)
    session = Session(source, [selected, first, second, final])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, _frame, expected, **_kw: {"count": expected})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def verify(_session, _base, receipt, _when, *, sleep):
        receipt["verified"] = receipt["completed"] = True
        receipt["deployed_units"] = 17
        return final
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post", verify)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is final
    receipt = session.frozen_line_receipt
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 21
    assert receipt["deployed_units"] == 17
    assert receipt["burst_limit_sec"] == remainder_burst_limit(plan)
    assert [item["after_count"] for item in receipt["probe_events"]] == [1, 0]
    assert receipt["probe_events"][1]["after"]["frame"] == str(second.screenshot_path)
    assert receipt["final_checkpoint"]["frame"] == str(final.screenshot_path)
    assert receipt["probe_started_at"] <= receipt["probe_authorized_at"]
    assert receipt["probe_elapsed_sec"] < receipt["probe_limit_sec"]
    assert all(event["started_at"] <= event["input_done_at"] <=
               event["verified_at"] < event["deadline"] for event in receipt["probe_events"])
    assert [point for _, point in session.inputs if point in plan.anchors][:2] == list(plan.anchors)
    assert len(session.inputs) == receipt["input_count"]


def test_second_balloon_exact_camera_recovery_waits_for_two_full_old_point_frames(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    pending = image_frame(tmp_path, "pending", "battle", 10.3)
    spent = image_frame(tmp_path, "spent-shifted", "battle", 10.4)
    moving = image_frame(tmp_path, "moving", "battle", 10.5)
    restored = image_frame(tmp_path, "restored", "battle", 10.6)
    stable = image_frame(tmp_path, "stable", "battle", 10.7)
    session = Session(source, [selected, first, pending, spent, moving, restored, stable])
    session.deadline = session.task_deadline = 200.
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)

    def current_frame(_session, _plan, snapshot, **_kw):
        if snapshot is spent:
            raise bound_camera_error(snapshot, "Probe camera has insufficient stationary map inliers")
        return frame_proof(snapshot)

    monkeypatch.setattr("autococ.probed_anchor._current_frame", current_frame)
    monkeypatch.setattr("autococ.probed_anchor._frame_envelope",
                        lambda _s, _p, snapshot, **_kw: {
                            key: value for key, value in frame_proof(snapshot).items() if key != "camera"})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, snapshot, expected, **_kw: {
                            "count": 1 if snapshot is pending else expected})

    def camera(a, b, _anchors):
        if b == moving.screenshot_path:
            raise ExactCameraError("Probe camera moved an exact deployment anchor")
        return {"matches": 100, "inliers": 95, "residual_p95": .01, "anchor_offsets": [.01, .01]}

    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera", camera)
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post",
                        lambda _s, _b, _r, _when, *, sleep: stable)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is stable
    receipt = session.frozen_line_receipt
    assert receipt["probe_events"][1]["status"] == "verified_after_recovery"
    assert receipt["probe_events"][1]["camera_rejection"] == {
        "observation_index": 1, "frame": str(spent.screenshot_path),
        "sha256": _file_sha256(spent.screenshot_path),
        "message": "Probe camera has insufficient stationary map inliers", "same_frame_count": 0}
    assert [row["status"] for row in receipt["recovery"]["frames"]] == [
        "camera_rejected", "first_exact_frame", "second_exact_frame"]
    assert receipt["recovery"]["status"] == "authorized"
    assert receipt["recovery"]["input_count"] == 0
    assert receipt["final_checkpoint"]["frame"] == str(stable.screenshot_path)
    assert ("probed-anchor-final-checkpoint", None) not in session.observe_calls
    assert len(session.inputs) == receipt["input_count"]
    assert [point for kind, point in session.inputs if kind == "tap" and point in plan.anchors] == list(plan.anchors)


@pytest.mark.parametrize("failure", ["still_moving", "count_conflict", "account_in_use",
                                      "wrong_version", "wrong_display", "wrong_foreground",
                                      "source_mutation", "source_mutation_during_camera",
                                      "stop", "not_battle", "fast_scope",
                                      "reserve"])
def test_second_balloon_recovery_fails_closed_without_more_input(monkeypatch, tmp_path, failure):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    pending = image_frame(tmp_path, "pending", "battle", 10.3)
    spent = image_frame(tmp_path, "spent-shifted", "battle", 10.4)
    recovery = [image_frame(tmp_path, f"recovery-{i}",
                            "account_in_use" if failure == "account_in_use" and i == 0 else "battle",
                            10.5 + i * .1) for i in range(3)]
    session = Session(source, [selected, first, pending, spent, *recovery])
    session.deadline = session.task_deadline = 100. if failure == "reserve" else 200.
    session.capture.adb = object()
    session.capture.serial = "offline"
    session.config.game.package_name = "com.supercell.clashofclans"
    if failure == "not_battle":
        recovery[0] = replace(recovery[0], scene="home")
        session.observations = iter([selected, first, pending, spent, *recovery])
    if failure == "fast_scope":
        recovery[0].observations["recognition_scope"] = "battle_hud"
    if failure == "stop":
        session.check_deadline = lambda: (_ for _ in ()).throw(StopRequested("stop")) if session.last_snapshot is recovery[0] else None
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_a, **_kw: (
        "other.package" if failure == "wrong_foreground" and session.last_snapshot is recovery[0]
        else session.config.game.package_name))

    def current_frame(_session, _plan, snapshot, **_kw):
        if snapshot is spent:
            raise bound_camera_error(snapshot, "Probe camera moved an exact deployment anchor")
        return frame_proof(snapshot)

    def envelope(_session, _plan, snapshot, **kwargs):
        if failure == "wrong_version" and snapshot is recovery[0]:
            session.client_version = "18.600.7"
        if failure == "wrong_display" and snapshot is recovery[0]:
            session.capture.input_display_id = 1
        if failure == "source_mutation" and snapshot is recovery[0]:
            Path(plan.base.frame.screenshot_path).write_bytes(b"changed camera source")
        return _frame_envelope(_session, _plan, snapshot, **kwargs)

    def count(_session, _plan, snapshot, expected, **_kw):
        if failure == "count_conflict" and snapshot is recovery[0]:
            raise FlowError("Probe strong_count_conflict on the current balloon card")
        return {"count": 1 if snapshot is pending else expected}

    monkeypatch.setattr("autococ.probed_anchor._current_frame", current_frame)
    monkeypatch.setattr("autococ.probed_anchor._frame_envelope", envelope)
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    def camera(_source, current, _anchors):
        if failure == "source_mutation_during_camera" and str(current) == str(recovery[0].screenshot_path):
            Path(plan.base.frame.screenshot_path).write_bytes(b"changed during camera proof")
        raise ExactCameraError("Probe camera moved an exact deployment anchor")
    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera", camera)
    with pytest.raises(StopRequested if failure == "stop" else DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None)
    receipt = session.frozen_line_receipt if failure == "stop" else raised.value.partial_receipt
    assert receipt["issued_placements"] == 2
    assert receipt["probe_events"][1]["status"] == "input_sent"
    assert receipt["recovery"]["status"] == "failed"
    assert receipt["input_sent"] is False
    assert len(session.inputs) == 3  # One card select, two balloon taps, no main army.
    assert (len(receipt["recovery"]["frames"]) == (0 if failure in ("reserve", "stop") else
            3 if failure == "still_moving" else 1))
    if failure == "source_mutation_during_camera":
        assert "frozen camera source pixels changed" in receipt["recovery"]["reason"]


def test_second_balloon_recovery_requires_positive_zero_count_before_any_new_frame(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    pending = image_frame(tmp_path, "pending", "battle", 10.3)
    shifted_pending = image_frame(tmp_path, "shifted-pending", "battle", 10.4)
    session = Session(source, [selected, first, pending, shifted_pending])
    session.deadline = session.task_deadline = 200.
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: (
                            (_ for _ in ()).throw(bound_camera_error(
                                snapshot, "Probe camera moved an exact deployment anchor"))
                            if snapshot is shifted_pending else frame_proof(snapshot)))
    monkeypatch.setattr("autococ.probed_anchor._frame_envelope",
                        lambda _s, _p, snapshot, **_kw: {
                            key: value for key, value in frame_proof(snapshot).items() if key != "camera"})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, snapshot, expected, **_kw: {
                            "count": 1 if snapshot in (pending, shifted_pending) else expected})
    with pytest.raises(DeploymentError, match="camera moved an exact deployment anchor") as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None)
    assert "recovery" not in raised.value.partial_receipt
    assert raised.value.partial_receipt["issued_placements"] == 2
    assert len(session.inputs) == 3


@pytest.mark.parametrize("failure", ["deadline_after_hash", "deadline_after_foreground",
                                      "source_after_foreground", "display_after_foreground",
                                      "version_after_foreground", "frame_after_foreground",
                                      "stop_after_foreground"])
def test_recovered_final_proof_rechecks_time_pixels_and_live_gates_before_remainder(monkeypatch, tmp_path, failure):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    pending = image_frame(tmp_path, "pending", "battle", 10.3)
    spent = image_frame(tmp_path, "shifted", "battle", 10.4)
    restored = image_frame(tmp_path, "restored", "battle", 10.5)
    stable = image_frame(tmp_path, "stable", "battle", 10.6)
    session = Session(source, [selected, first, pending, spent, restored, stable])
    session.deadline = session.task_deadline = 200.
    now = {"value": 10.}
    stopped = {"value": False}
    session.check_deadline = lambda: (_ for _ in ()).throw(StopRequested("stop")) if stopped["value"] else None
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: (
                            (_ for _ in ()).throw(bound_camera_error(snapshot, "Probe camera moved"))
                            if snapshot is spent else frame_proof(snapshot)))
    monkeypatch.setattr("autococ.probed_anchor._frame_envelope",
                        lambda _s, _p, snapshot, **_kw: {
                            key: value for key, value in frame_proof(snapshot).items() if key != "camera"})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, snapshot, expected, **_kw: {
                            "count": 1 if snapshot is pending else expected})
    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera",
                        lambda *_a: {"matches": 100, "inliers": 95,
                                     "residual_p95": .01, "anchor_offsets": [.01, .01]})
    foreground_calls = {"value": 0}
    def foreground(_session):
        foreground_calls["value"] += 1
        if foreground_calls["value"] != 4:
            return
        if failure == "deadline_after_foreground":
            now["value"] = 22.
        elif failure == "source_after_foreground":
            Path(plan.base.frame.screenshot_path).write_bytes(b"source changed")
        elif failure == "display_after_foreground":
            session.capture.input_display_id = 1
        elif failure == "version_after_foreground":
            session.client_version = "18.600.7"
        elif failure == "frame_after_foreground":
            stable.screenshot_path.write_bytes(b"frame changed")
        elif failure == "stop_after_foreground":
            stopped["value"] = True
    monkeypatch.setattr("autococ.probed_anchor._foreground_now", foreground)
    if failure == "deadline_after_hash":
        original_sha = _file_sha256
        final_hashes = {"value": 0}
        def slow_hash(path):
            if ("final_checkpoint" in getattr(session, "frozen_line_receipt", {}) and
                    str(path) == str(stable.screenshot_path)):
                final_hashes["value"] += 1
                if final_hashes["value"] == 2:
                    now["value"] = 22.
            return original_sha(path)
        monkeypatch.setattr("autococ.probed_anchor._file_sha256", slow_hash)
    with pytest.raises(StopRequested if failure == "stop_after_foreground" else DeploymentError):
        execute_probed_anchor_line(session, plan, clock=lambda: now["value"], sleep=lambda _: None)
    assert session.frozen_line_receipt["probe_events"][1]["status"] == "verified_after_recovery"
    assert session.frozen_line_receipt["input_sent"] is False
    assert len(session.inputs) == 3
    assert all(kind != "frozen" for kind, _point in session.inputs)


def test_selection_waits_for_fresh_border_without_repeating_select(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    unselected = image_frame(tmp_path, "unselected", "enemy_village", 10.1, fast=True)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.2, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.3)
    second = image_frame(tmp_path, "second", "battle", 10.4)
    final = image_frame(tmp_path, "final", "battle", 10.5)
    session = Session(source, [unselected, selected, first, second, final])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox",
                        lambda path, *_a: None if path == unselected.screenshot_path else plan.balloon_bbox)
    readings = []
    def count(_session, _plan, snapshot, expected, **kwargs):
        readings.append((snapshot.screenshot_path.name, expected, kwargs))
        return {"count": expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post",
                        lambda _s, _p, receipt, _when, *, sleep: final)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is final
    assert [point for kind, point in session.inputs if kind == "tap" and point == plan.balloon_point] == [plan.balloon_point]
    assert [item[:2] for item in readings] == [("unselected.png", 2), ("selected.png", 2),
                                                ("first.png", 1), ("second.png", 0),
                                                ("final.png", 0)]
    assert readings[0][2]["selected"] is False and readings[1][2]["selected"] is True


@pytest.mark.parametrize("failure", ("three_unselected", "fast_scene", "fast_conflict", "three_seconds"))
def test_selected_fast_gate_fails_with_one_select_and_no_probe(monkeypatch, tmp_path, failure):
    plan, source = make_plan(monkeypatch, tmp_path)
    frames = [image_frame(tmp_path, f"wait-{index}", "enemy_village", 10.1 + index * .1,
                          fast=True) for index in range(3)]
    if failure == "fast_scene":
        frames[0].observations["matched_reasons"] = ["inherited_preview"]
    if failure == "fast_conflict":
        frames[0].observations["deployment_count_conflict"] = {"counts": [1, 2]}
        frames[0] = replace(frames[0], scene="unknown")
    session = Session(source, frames)
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda *_a, **_kw: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: None)
    now = [10.]
    if failure == "three_seconds":
        old_observe = session.observe
        def slow_observe(label, **kwargs):
            snapshot = old_observe(label, **kwargs)
            now[0] = 13.1
            return snapshot
        session.observe = slow_observe
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: now[0])
    assert raised.value.partial_receipt["issued_placements"] == 0
    assert session.inputs == [("tap", plan.balloon_point)]
    assert len(raised.value.partial_receipt["evidence"]) == (4 if failure == "three_unselected" else 2)
    if failure == "fast_conflict":
        assert "strong_count_conflict" in str(raised.value)


def test_first_probe_pending_count_uses_second_frame_without_replaying(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    pending = image_frame(tmp_path, "pending", "battle", 10.2)
    consumed = image_frame(tmp_path, "consumed", "battle", 10.3)
    second = image_frame(tmp_path, "second", "battle", 10.4)
    final = image_frame(tmp_path, "final", "battle", 10.5)
    session = Session(source, [selected, pending, consumed, second, final])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def count(_session, _plan, snapshot, expected, **kwargs):
        return {"count": 2 if snapshot is pending else expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post",
                        lambda _s, _p, receipt, _when, *, sleep: final)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is final
    receipt = session.frozen_line_receipt
    assert receipt["probe_events"][0]["after"]["frame"] == str(consumed.screenshot_path)
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 21
    assert [point for kind, point in session.inputs if kind == "tap" and point in plan.anchors] == list(plan.anchors)


def test_transient_count_rejection_is_recorded_before_later_probe_success(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    rejected = image_frame(tmp_path, "rejected", "battle", 10.2)
    accepted = image_frame(tmp_path, "accepted", "battle", 10.3)
    second = image_frame(tmp_path, "second", "battle", 10.4)
    final = image_frame(tmp_path, "final", "battle", 10.5)
    session = Session(source, [selected, rejected, accepted, second, final])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def count(_session, _plan, snapshot, expected, **_kw):
        if snapshot is rejected:
            raise FlowError("Probe current balloon count is unreadable")
        return {"count": expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post",
                        lambda _s, _p, _r, _when, *, sleep: final)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is final
    receipt = session.frozen_line_receipt
    assert receipt["probe_events"][0]["count_proof_rejections"] == [{
        "observation_index": 0, "frame": str(rejected.screenshot_path),
        "sha256": _file_sha256(rejected.screenshot_path),
        "message": "Probe current balloon count is unreadable"}]
    assert receipt["probe_events"][0]["after"]["frame"] == str(accepted.screenshot_path)
    assert [point for kind, point in session.inputs if kind == "tap" and point in plan.anchors] == list(plan.anchors)


def test_two_count_rejections_remain_in_failed_probe_receipt_without_replay(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    rejected = [image_frame(tmp_path, f"rejected-{index}", "battle", 10.2 + index * .1)
                for index in range(2)]
    session = Session(source, [selected, *rejected])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def count(_session, _plan, snapshot, _expected, **_kw):
        if snapshot is selected:
            return {"count": 2}
        raise FlowError(f"unreadable {snapshot.screenshot_path.stem}")
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    with pytest.raises(DeploymentError, match="complete battle/count/camera proof") as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    receipt = raised.value.partial_receipt
    assert receipt["probe_events"][0]["count_proof_rejections"] == [
        {"observation_index": index, "frame": str(snapshot.screenshot_path),
         "sha256": _file_sha256(snapshot.screenshot_path),
         "message": f"unreadable {snapshot.screenshot_path.stem}"}
        for index, snapshot in enumerate(rejected)]
    assert receipt["issued_placements"] == 1
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]


def test_post_fast_preview_waits_then_full_battle_fallback_and_independent_final_full(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    preview = image_frame(tmp_path, "preview", "enemy_village", 10.2, fast="battle_hud")
    consumed = image_frame(tmp_path, "consumed", "battle", 10.3)
    second = image_frame(tmp_path, "second", "battle", 10.4, fast="battle_hud")
    final = image_frame(tmp_path, "final", "battle", 10.5)
    session = Session(source, [selected, preview, consumed, second, final])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    counts = {"selected": 2, "preview": 2, "consumed": 1, "second": 0, "final": 0}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, snapshot, _expected, **_kw:
                            {"count": counts[snapshot.screenshot_path.stem]})
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post",
                        lambda _s, _p, _r, _when, *, sleep: final)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is final
    receipt = session.frozen_line_receipt
    assert [item["after"]["frame"] for item in receipt["probe_events"]] == [
        str(consumed.screenshot_path), str(second.screenshot_path)]
    assert receipt["final_checkpoint"]["frame"] == str(final.screenshot_path)
    assert receipt["checkpoint"]["sha256"] == _file_sha256(final.screenshot_path)
    assert session.observe_calls == [
        ("probed-anchor-balloon-selected", "troop_count"),
        ("probed-anchor-0-post-0", "battle_hud"),
        ("probed-anchor-0-post-1", "battle_hud"),
        ("probed-anchor-1-post-0", "battle_hud"),
        ("probed-anchor-final-checkpoint", None)]
    assert [point for kind, point in session.inputs if kind == "tap" and point in plan.anchors] == list(plan.anchors)


@pytest.mark.parametrize("failure", ("fast_guard", "strong_conflict", "identity_conflict"))
def test_post_fast_scene_or_strong_conflict_never_replays_or_sends_second_probe(monkeypatch, tmp_path, failure):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2, fast="battle_hud")
    later = image_frame(tmp_path, "later", "battle", 10.3, fast="battle_hud")
    if failure == "fast_guard":
        first.observations["matched_reasons"] = ["cached_preview"]
    elif failure == "strong_conflict":
        first.observations["deployment_count_conflict"] = {"counts": [1, 2]}
    session = Session(source, [selected, first, later])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def count(_session, _plan, snapshot, expected, **_kw):
        if failure == "identity_conflict" and snapshot is first:
            raise FlowError("Probe strong current-card identity conflicts with frozen balloon")
        return {"count": expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert raised.value.partial_receipt["issued_placements"] == 1
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]
    assert session.last_snapshot is first


@pytest.mark.parametrize("scene,scope", (("unknown", "full"), ("battle", "troop_count")))
def test_final_checkpoint_requires_new_positive_full_battle_frame(monkeypatch, tmp_path, scene, scope):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    final = image_frame(tmp_path, "final", scene, 10.4, fast=scope == "troop_count")
    final.observations["recognition_scope"] = scope
    session = Session(source, [selected, first, second, final])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, _snapshot, expected, **_kw: {"count": expected})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(DeploymentError, match="independent full battle frame") as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert raised.value.partial_receipt["issued_placements"] == 2
    assert session.observe_calls[-1] == ("probed-anchor-final-checkpoint", None)
    assert [point for kind, point in session.inputs if kind == "tap" and point in plan.anchors] == list(plan.anchors)


@pytest.mark.parametrize("failure", ("stale", "count_conflict"))
def test_final_full_cannot_reuse_second_probe_or_ignore_balloon_conflict(monkeypatch, tmp_path, failure):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    final = image_frame(tmp_path, "final", "battle", 10.3 if failure == "stale" else 10.4)
    session = Session(source, [selected, first, second, final])
    now = [10.]
    def current_frame(_session, _plan, snapshot, *, after, **_kw):
        if snapshot.observations["observed_at_monotonic"] <= after:
            raise FlowError("Probe frame is not newer than its input proof")
        return frame_proof(snapshot)
    monkeypatch.setattr("autococ.probed_anchor._current_frame", current_frame)
    def count(_session, _plan, _snapshot, expected, **_kw):
        if failure == "count_conflict" and _snapshot is final:
            raise FlowError("Probe strong_count_conflict on final balloon card")
        if failure == "stale" and _snapshot is second:
            now[0] = 10.35
        return {"count": expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", count)
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: now[0])
    assert raised.value.partial_receipt["issued_placements"] == 2
    assert raised.value.partial_receipt["input_sent"] is False
    assert len(session.inputs) == 3  # Select and two probes only.
    assert session.last_snapshot is final


def test_two_pending_probe_frames_never_authorize_second_input(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    pending1 = image_frame(tmp_path, "pending-1", "battle", 10.2)
    pending2 = image_frame(tmp_path, "pending-2", "battle", 10.3)
    session = Session(source, [selected, pending1, pending2])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, _snapshot, expected, **_kw: {"count": 2})
    with pytest.raises(DeploymentError, match="complete battle/count/camera proof") as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert raised.value.partial_receipt["issued_placements"] == 1
    assert raised.value.partial_receipt["probe_events"][0]["status"] == "input_sent"
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]


@pytest.mark.parametrize("when", ("before_select", "before_first_probe"))
def test_stop_arriving_during_foreground_proof_prevents_next_input(monkeypatch, tmp_path, when):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    session = Session(source, [selected])
    stopped = [False]
    checks = [0]
    def foreground(_session):
        checks[0] += 1
        if when == "before_select" and checks[0] == 1 or when == "before_first_probe" and checks[0] == 2:
            stopped[0] = True
    def check_deadline():
        if stopped[0]:
            raise StopRequested("stopped after foreground observation")
    session.check_deadline = check_deadline
    monkeypatch.setattr("autococ.probed_anchor._foreground_now", foreground)
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a, **_kw: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(StopRequested):
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert session.inputs == ([] if when == "before_select" else [("tap", plan.balloon_point)])
    assert session.frozen_line_receipt["issued_placements"] == 0


def test_failed_probe_capture_has_partial_receipt_and_never_replays(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    session = Session(source, [selected, FlowError("capture failed")])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a, **_kw: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    receipt = raised.value.partial_receipt
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 1
    assert receipt["deployed_units"] == 0
    assert [point for kind, point in session.inputs if kind == "tap"] == [plan.balloon_point, plan.anchors[0]]
    assert session.last_snapshot is None


def test_selected_card_capture_failure_records_selection_input_not_full_batch(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    session = Session(source, [FlowError("selected capture failed")])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    receipt = raised.value.partial_receipt
    assert receipt["any_input_sent"] is True and receipt["input_sent"] is False
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 0
    assert receipt["actions"][0]["status"] == "input_sent"
    assert session.last_snapshot is None
    assert session.inputs == [("tap", plan.balloon_point)]


@pytest.mark.parametrize("mutation", ("before_select", "before_probe"))
def test_foreground_check_cannot_authorize_pixels_changed_before_input(monkeypatch, tmp_path, mutation):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    session = Session(source, [selected])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a, **_kw: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    checks = [0]
    def foreground(_session):
        checks[0] += 1
        if mutation == "before_select" and checks[0] == 1:
            source.screenshot_path.write_bytes(b"changed while checking foreground")
        if mutation == "before_probe" and checks[0] == 2:
            selected.screenshot_path.write_bytes(b"changed while checking foreground")
    monkeypatch.setattr("autococ.probed_anchor._foreground_now", foreground)
    with pytest.raises(DeploymentError):
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert session.inputs == ([] if mutation == "before_select" else [("tap", plan.balloon_point)])


def test_strong_count_conflict_stops_probe_without_next_frame_or_replay(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    session = Session(source, [selected, first, second])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    def conflicting_count(_session, _plan, snapshot, expected, **_kw):
        if snapshot is first:
            raise FlowError("Probe strong_count_conflict between full card and local OCR")
        return {"count": expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", conflicting_count)
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(DeploymentError, match="strong_count_conflict") as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    receipt = raised.value.partial_receipt
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 1
    assert receipt["probe_events"][0]["status"] == "input_sent"
    assert receipt["probe_events"][0]["count_proof_rejections"] == [{
        "observation_index": 0, "frame": str(first.screenshot_path),
        "sha256": _file_sha256(first.screenshot_path),
        "message": "Probe strong_count_conflict between full card and local OCR"}]
    assert session.last_snapshot is first  # The second frame cannot wash out strong conflict.
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]


def test_completed_probe_proof_after_six_second_budget_cannot_send_next_input(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    session = Session(source, [selected, first])
    now = [10.]
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    def slow_count(_session, _plan, snapshot, expected, **_kw):
        if snapshot is first:
            now[0] = 16.1
        return {"count": expected}
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", slow_count)
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(DeploymentError, match="exceeded its deadline") as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: now[0])
    assert raised.value.partial_receipt["issued_placements"] == 1
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]


@pytest.mark.parametrize("changed", ("budget", "source_pixels", "probe_pixels"))
def test_final_checkpoint_rechecks_total_budget_and_all_source_pixels(monkeypatch, tmp_path, changed):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    final = image_frame(tmp_path, "final", "battle", 10.4)
    session = Session(source, [selected, first, second, final])
    now = [10.]
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: frame_proof(snapshot))
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, _frame, expected, **_kw: {"count": expected})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def checkpoint(_session, _plan, snapshot):
        if changed == "budget":
            now[0] = 22.1
        elif changed == "source_pixels":
            source.screenshot_path.write_bytes(b"source changed after both probes")
        else:
            first.screenshot_path.write_bytes(b"first probe proof changed")
        return {"frame": str(snapshot.screenshot_path),
                "sha256": _file_sha256(snapshot.screenshot_path)}
    monkeypatch.setattr("autococ.probed_anchor._remaining_roster_checkpoint", checkpoint)
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: now[0])
    receipt = raised.value.partial_receipt
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 2
    assert receipt["input_sent"] is False
    assert [point for _, point in session.inputs] == [plan.balloon_point, *plan.anchors]


def test_probe_stop_propagates_without_replay(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    session = Session(source, [selected, StopRequested("stop")])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a, **_kw: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(StopRequested):
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert len(session.inputs) == 2


def test_uncertain_probe_transport_records_exact_attempt_without_retry(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1, fast=True)
    session = Session(source, [selected])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a, **_kw: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    original_tap = session.tap
    def uncertain_tap(snapshot, point, *, reason):
        original_tap(snapshot, point, reason=reason)
        if tuple(point) == plan.anchors[0]:
            raise TimeoutError("unknown input result")
    session.tap = uncertain_tap
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    receipt = raised.value.partial_receipt
    assert receipt["attempted_placements"] == 1
    assert receipt["issued_placements"] == 0
    assert receipt["probe_events"][0]["status"] == "uncertain"
    assert receipt["actions"][-1]["status"] == "uncertain"
    assert [point for _, point in session.inputs].count(plan.anchors[0]) == 1


def test_current_count_refuses_conflict_wrong_version_and_missing_border(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    image = image_frame(tmp_path, "count", "battle", 10.2)
    image.observations["battle"]["slots"] = [{"unit_id": "balloon", "source": "army",
        "bbox": list(plan.balloon_bbox), "count": 1, "confidence": .99,
        "evidence": {"identity": {"unit_id": "balloon", "version_verified": True,
                                  "template_client_version": "18.647.2"}}}]
    session = Session(image, [])
    reading = {"count": 1, "confidence": .99, "portrait_score": .99,
               "portrait_source": "current_frame_saved_battle_portrait",
               "frame": str(image.screenshot_path), "slot_bbox": list(plan.balloon_bbox),
               "readings": [{"text": "x1", "confidence": .99}]}
    monkeypatch.setattr("autococ.probed_anchor._post_troop_reading", lambda *_a, **_kw: reading)
    assert _balloon_count(session, plan, image, 1) is reading
    image.observations["recognition_scope"] = "troop_count"
    image.observations["deployment_count_reads"] = [{"frame": str(image.screenshot_path),
        "slot_bbox": list(plan.balloon_bbox), "count": 2, "confidence": .99}]
    with pytest.raises(FlowError, match="strong_count_conflict between fast"):
        _balloon_count(session, plan, image, 1)
    image.observations["deployment_count_reads"] = []
    with pytest.raises(FlowError, match="fast current-card count lacks"):
        _balloon_count(session, plan, image, 1)
    del image.observations["recognition_scope"]
    del image.observations["deployment_count_reads"]
    reading["readings"].append({"text": "x0", "confidence": .99})
    with pytest.raises(FlowError):
        _balloon_count(session, plan, image, 1)
    reading["readings"].pop()
    image.observations["battle"]["slots"][0]["evidence"]["identity"]["template_client_version"] = "18.600.7"
    with pytest.raises(FlowError):
        _balloon_count(session, plan, image, 1)
    reading.update(count=0, readings=[{"text": "x0", "confidence": .99}])
    image.observations["battle"]["slots"] = []
    monkeypatch.setattr("autococ.images.read_frame",
                        lambda _path: np.zeros((720, 1280, 3), dtype=np.uint8))
    with pytest.raises(FlowError):
        _balloon_count(session, plan, image, 0)
    monkeypatch.setattr("autococ.probed_anchor._closed_card_border",
                        lambda _image, _box: {"left_edge": 300, "right_edge": -200, "bottom_contrast": 50})
    assert _balloon_count(session, plan, image, 0) is reading


def test_selected_positive_count_rebinds_frozen_source_and_current_border(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    image = image_frame(tmp_path, "selected-balloon", "enemy_village", 10.2)
    session = Session(image, [])
    reading = {"count": 2, "confidence": .99, "portrait_score": .935,
               "portrait_source": "current_frame_saved_battle_portrait",
               "frame": str(image.screenshot_path), "slot_bbox": list(plan.balloon_bbox),
               "readings": [{"text": "x2", "confidence": .99}]}
    monkeypatch.setattr("autococ.probed_anchor._post_troop_reading", lambda *_a, **_kw: reading)
    monkeypatch.setattr("autococ.images.read_frame", lambda _path: np.zeros((720, 1280, 3), dtype=np.uint8))
    identity = {"unit_id": "balloon", "confidence": .967, "version_verified": True,
                "template_client_version": "18.647.2"}
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity", lambda *_a, **_kw: identity)
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    assert _balloon_count(session, plan, image, 2, source_bound=True, selected=True)["selected_card_border"] == list(plan.balloon_bbox)
    assert reading["source_identity_frame"] == str(source.screenshot_path)
    image.observations["battle"]["slots"] = [{"unit_id": "electro_dragon",
        "bbox": list(plan.balloon_bbox), "count": 2, "confidence": .99}]
    with pytest.raises(FlowError, match="current-card identity conflicts"):
        _balloon_count(session, plan, image, 2, source_bound=True, selected=True)
    image.observations["battle"]["slots"] = []
    identity["template_client_version"] = "18.600.7"
    with pytest.raises(FlowError, match="source balloon versioned identity"):
        _balloon_count(session, plan, image, 2, source_bound=True, selected=True)
    identity["template_client_version"] = "18.647.2"
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: None)
    with pytest.raises(FlowError, match="closed current card border"):
        _balloon_count(session, plan, image, 2, source_bound=True, selected=True)
    source.screenshot_path.write_bytes(b"source pixels changed")
    with pytest.raises(FlowError, match="source pixels changed"):
        _balloon_count(session, plan, image, 2, source_bound=True, selected=False)


def test_real_current_balloon_selected_fixture_binds_source_and_ordinary_ed_geometry(tmp_path):
    """Real 021021 lower toolbar pixels; no simulated x1 or probe success."""
    from autococ.battle_vision import selected_card_bbox
    from autococ.images import read_frame
    from autococ.vision import ScreenshotRecognizer

    paths = {name: FIXTURES / f"probed_balloon_021021_{name}.png"
             for name in ("before", "source", "unselected", "selected")}
    for name, path in paths.items():
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert provenance["client_version"] == "18.647.2"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
        assert provenance["retained_native_xyxy"] == [0, 1090, 2560, 1440]
    image_before = cv2.resize(read_frame(paths["before"]), (1280, 720), interpolation=cv2.INTER_AREA)
    image_source = cv2.resize(read_frame(paths["source"]), (1280, 720), interpolation=cv2.INTER_AREA)
    image_selected = cv2.resize(read_frame(paths["selected"]), (1280, 720), interpolation=cv2.INTER_AREA)
    ordinary_ed = (45, 595, 133, 711)
    selected_ed = (44, 591, 135, 708)
    assert _current_card_edges(image_before, ordinary_ed) is not None
    assert _current_card_edges(image_source, selected_ed) is None
    assert _current_card_edges(image_selected, ordinary_ed) is not None

    balloon_bbox = (335, 595, 423, 711)
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(paths["source"]), screenshot_path=paths["source"])
    base = replace(base, frame=frame, frame_sha256=_file_sha256(paths["source"]),
                   toolbar_before_frame_id=str(paths["before"]),
                   toolbar_before_sha256=_file_sha256(paths["before"]))
    plan = SimpleNamespace(base=base, balloon_point=(379, 653), balloon_bbox=balloon_bbox)
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    for name, expected_selected in (("unselected", False), ("selected", True)):
        path = paths[name]
        snapshot = SceneSnapshot("enemy_village", .99, path,
                                 {"ocr": [], "battle": {"slots": []},
                                  "source_resolution": (2560, 1440),
                                  "baseline_resolution": (1280, 720)})
        session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                                  client_version="18.647.2")
        border = selected_card_bbox(path, balloon_bbox)
        assert (border is not None) is expected_selected
        reading = _balloon_count(session, plan, snapshot, 2, source_bound=True,
                                 selected=expected_selected)
        assert reading["count"] == 2 and reading["confidence"] >= .9
        assert reading["portrait_score"] >= .92
        assert reading["source_portrait_identity"]["unit_id"] == "balloon"
        assert ("selected_card_border" in reading) is expected_selected


def test_real_024858_spent_balloon_requires_independent_gray_border(tmp_path):
    """Real x0 toolbar pixels; full battle scene and tactical success are not fixture claims."""
    from autococ.images import read_frame
    from autococ.vision import ScreenshotRecognizer
    from autococ.frozen_line import _closed_card_border

    paths = {name: FIXTURES / f"probed_balloon_024858_{name}.png"
             for name in ("before", "source", "spent")}
    for path in paths.values():
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert provenance["client_version"] == "18.647.2"
        assert provenance["retained_native_xyxy"] == [0, 1090, 2560, 1440]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    bbox = (335, 595, 423, 711)
    image = cv2.resize(read_frame(paths["spent"]), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _closed_card_border(image, bbox) is None  # Reproduce the live gate.
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(paths["source"]), screenshot_path=paths["source"])
    base = replace(base, frame=frame, frame_sha256=_file_sha256(paths["source"]),
                   toolbar_before_frame_id=str(paths["before"]),
                   toolbar_before_sha256=_file_sha256(paths["before"]))
    plan = SimpleNamespace(base=base, balloon_point=(379, 653), balloon_bbox=bbox)
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    # A battle_hud snapshot supplies only a current two-HUD scene proof;
    # its empty OCR/slots cannot supply the quantity for this assertion.
    snapshot = SceneSnapshot("battle", .99, paths["spent"], {
        "recognition_scope": "battle_hud", "fast_fallback": False,
        "matched_reasons": ["two_current_hud_anchors"],
        "ocr": [], "battle": {"slots": []}})
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    reading = _balloon_count(session, plan, snapshot, 0)
    assert reading["count"] == 0 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["source_portrait_identity"]["unit_id"] == "balloon"
    assert reading["current_card_border"]["method"] == "spent_balloon_dark_outline"


def test_real_050614_dark_background_keeps_complete_spent_card(tmp_path):
    from autococ.images import read_frame
    from autococ.vision import ScreenshotRecognizer
    from autococ.frozen_line import _closed_card_border

    paths = {name: FIXTURES / f"probed_balloon_050614_{name}.png"
             for name in ("before", "source", "pending", "spent")}
    for path in paths.values():
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert provenance["client_version"] == "18.647.2"
        assert provenance["retained_native_xyxy"] == [0, 1090, 2560, 1440]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    bbox = (335, 595, 423, 711)
    pending = cv2.resize(read_frame(paths["pending"]), (1280, 720), interpolation=cv2.INTER_AREA)
    spent = cv2.resize(read_frame(paths["spent"]), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _spent_balloon_dark_border(pending, bbox) is None  # Real x1 is not a spent card.
    assert _closed_card_border(spent, bbox) is None  # Reproduce the failed live gate.
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(paths["source"]), screenshot_path=paths["source"])
    base = replace(base, frame=frame, frame_sha256=_file_sha256(paths["source"]),
                   toolbar_before_frame_id=str(paths["before"]),
                   toolbar_before_sha256=_file_sha256(paths["before"]))
    plan = SimpleNamespace(base=base, balloon_point=(379, 653), balloon_bbox=bbox)
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    snapshot = SceneSnapshot("battle", .99, paths["spent"], {
        "recognition_scope": "battle_hud", "fast_fallback": False,
        "matched_reasons": ["two_current_hud_anchors"],
        "ocr": [], "battle": {"slots": []}})
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    reading = _balloon_count(session, plan, snapshot, 0)
    assert reading["count"] == 0 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["source_portrait_identity"]["unit_id"] == "balloon"
    assert reading["current_card_border"]["method"] == "spent_balloon_versioned_bottom_shape"
    assert reading["current_card_border"]["core_max_error"] <= 4
    assert reading["current_card_border"]["lower_rebound_min"] >= 7
    for invalid_base in (replace(base, client_version="18.600.7"),
                         replace(base, frame_sha256="0" * 64)):
        with pytest.raises(FlowError):
            invalid_plan = SimpleNamespace(base=invalid_base, balloon_point=plan.balloon_point,
                                           balloon_bbox=bbox)
            _balloon_count(session, invalid_plan, snapshot, 0)


def test_real_084232_spent_card_allows_textured_background(tmp_path):
    """The card outline is exact even when the scene below it is not uniform."""
    from autococ.images import read_frame
    from autococ.vision import ScreenshotRecognizer

    paths = {name: FIXTURES / f"probed_balloon_084232_{name}.png"
             for name in ("before", "source", "pending", "spent")}
    for path in paths.values():
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert provenance["client_version"] == "18.647.2"
        assert provenance["retained_native_xyxy"] == [0, 1090, 2560, 1440]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    bbox = (335, 595, 423, 711)
    pending = cv2.resize(read_frame(paths["pending"]), (1280, 720), interpolation=cv2.INTER_AREA)
    assert _spent_balloon_dark_border(pending, bbox) is None
    spent = cv2.resize(read_frame(paths["spent"]), (1280, 720), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(spent, cv2.COLOR_BGR2GRAY)
    below = gray[713:716, 343:415].mean(axis=0)
    farther = gray[716:719, 343:415].mean(axis=0)
    assert np.percentile(np.abs(below - farther), 95) > 5  # Reproduce old rejection.
    assert np.max(np.abs(below - farther)) > 10
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(paths["source"]), screenshot_path=paths["source"])
    base = replace(base, frame=frame, frame_sha256=_file_sha256(paths["source"]),
                   toolbar_before_frame_id=str(paths["before"]),
                   toolbar_before_sha256=_file_sha256(paths["before"]))
    plan = SimpleNamespace(base=base, balloon_point=(379, 653), balloon_bbox=bbox)
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    snapshot = SceneSnapshot("battle", .99, paths["spent"], {
        "recognition_scope": "battle_hud", "fast_fallback": False,
        "matched_reasons": ["two_current_hud_anchors"],
        "ocr": [], "battle": {"slots": []}})
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    reading = _balloon_count(session, plan, snapshot, 0)
    assert reading["count"] == 0 and reading["confidence"] >= .9
    assert reading["portrait_score"] >= .92
    assert reading["source_portrait_identity"]["unit_id"] == "balloon"
    border = reading["current_card_border"]
    assert border["method"] == "spent_balloon_versioned_bottom_shape"
    assert border["core_max_error"] == 0
    assert border["lower_rebound_min"] >= 7
    assert border["background_jump_p95"] > 5


@pytest.mark.parametrize("group", ("050614", "084232"))
@pytest.mark.parametrize("damage", ("bottom_occluded", "bottom_truncated", "black_below",
                                     "left_corner", "right_corner", "left_side",
                                     "shifted_bbox"))
def test_real_spent_shape_rejects_damage(damage, group):
    from autococ.images import read_frame

    source = FIXTURES / f"probed_balloon_{group}_spent.png"
    image = cv2.resize(read_frame(source), (1280, 720), interpolation=cv2.INTER_AREA).copy()
    bbox = (335, 595, 423, 711)
    if damage == "bottom_occluded":
        image[711:713, 365:378] = (40, 40, 40)
    elif damage == "bottom_truncated":
        image[712:720, 343:415] = 0
    elif damage == "black_below":
        image[713:720, 343:415] = 0
    elif damage == "left_corner":
        image[704:713, 335:344] = (40, 40, 40)
    elif damage == "right_corner":
        image[704:713, 414:423] = (40, 40, 40)
    elif damage == "left_side":
        image[625:695, 335:337] = (40, 40, 40)
    else:
        bbox = (336, 595, 424, 711)
    assert _spent_balloon_dark_border(image, bbox) is None


@pytest.mark.parametrize("group", ("050614", "084232"))
def test_real_spent_count_proof_rejects_adjacent_card(tmp_path, group):
    """A neighboring card may have a valid border; it is still not balloon x0."""
    from autococ.vision import ScreenshotRecognizer

    paths = {name: FIXTURES / f"probed_balloon_{group}_{name}.png"
             for name in ("before", "source", "spent")}
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(paths["source"]), screenshot_path=paths["source"])
    base = replace(base, frame=frame, frame_sha256=_file_sha256(paths["source"]),
                   toolbar_before_frame_id=str(paths["before"]),
                   toolbar_before_sha256=_file_sha256(paths["before"]))
    plan = SimpleNamespace(base=base, balloon_point=(476, 653), balloon_bbox=(432, 595, 520, 711))
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    snapshot = SceneSnapshot("battle", .99, paths["spent"], {"ocr": [], "battle": {"slots": []}})
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer, client_version="18.647.2")
    with pytest.raises(FlowError):
        _balloon_count(session, plan, snapshot, 0)


def test_real_050614_spent_shape_rejects_changed_template(monkeypatch, tmp_path):
    from autococ.images import read_frame
    from autococ.unit_catalog import CATALOG_ROOT

    image = cv2.resize(read_frame(FIXTURES / "probed_balloon_050614_spent.png"),
                       (1280, 720), interpolation=cv2.INTER_AREA)
    template = CATALOG_ROOT / "units" / "balloon_spent_bottom_18_647_2.png"
    provenance = json.loads(template.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(template.read_bytes()).hexdigest() == provenance["template_sha256"]
    changed = tmp_path / "units"
    changed.mkdir()
    (changed / template.name).write_bytes(template.read_bytes() + b"changed")
    monkeypatch.setattr("autococ.unit_catalog.CATALOG_ROOT", tmp_path)
    assert _spent_balloon_dark_border(image, (335, 595, 423, 711)) is None


def test_versioned_spent_shape_thresholds_cover_three_real_backgrounds():
    from autococ.images import read_frame
    from autococ.unit_catalog import CATALOG_ROOT

    template_path = CATALOG_ROOT / "units" / "balloon_spent_bottom_18_647_2.png"
    provenance = json.loads(template_path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    assert provenance["source_sha256"] == "2d39842eb37f6875096180fa2d847e2a39b887b68fb139ffc3e20d111260b91e"
    assert hashlib.sha256(template_path.read_bytes()).hexdigest() == provenance["template_sha256"]
    assert provenance["baseline_roi_xyxy"] == [343, 706, 415, 713]
    for group in ("024858", "035742", "050614"):
        path = FIXTURES / f"probed_balloon_{group}_spent.png"
        source = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["fixture_sha256"]
        image = cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        core = gray[706:713, 343:415]
        if group == "024858":
            assert np.array_equal(core, template)  # Rebuild template from its anonymized source pixels.
        assert np.max(np.abs(core.astype(np.int16) - template.astype(np.int16))) <= 4
        trough = gray[711:713, 343:415].mean(axis=0)
        below = gray[713:716, 343:415].mean(axis=0)
        farther = gray[716:719, 343:415].mean(axis=0)
        assert np.all(below - trough >= 7)
        assert np.percentile(np.abs(below - farther), 95) <= 5
        assert np.all(np.abs(below - farther) <= 10)


@pytest.mark.parametrize("damage", ("bottom", "left_corner", "right_corner",
                                     "left_turn", "right_turn", "left_side",
                                     "shifted_bbox", "adjacent_card",
                                     "wrong_version", "changed_source", "wrong_identity"))
def test_gray_balloon_fallback_rejects_missing_outline_or_source_binding(monkeypatch, tmp_path, damage):
    from autococ.images import read_frame
    from autococ.vision import ScreenshotRecognizer

    fixture = lambda name: FIXTURES / f"probed_balloon_024858_{name}.png"
    source = fixture("source")
    before = fixture("before")
    spent = fixture("spent")
    bbox = (335, 595, 423, 711)
    point = (379, 653)
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(source), screenshot_path=source)
    base = replace(base, frame=frame, frame_sha256=_file_sha256(source),
                   toolbar_before_frame_id=str(before), toolbar_before_sha256=_file_sha256(before))
    if damage == "wrong_version":
        base = replace(base, client_version="18.600.7")
    elif damage == "changed_source":
        base = replace(base, frame_sha256="0" * 64)
    elif damage == "wrong_identity":
        monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                            lambda *_a, **_kw: {"unit_id": "archer", "confidence": .99,
                                                  "version_verified": True,
                                                  "template_client_version": "18.647.2"})
    elif damage in ("shifted_bbox", "adjacent_card"):
        bbox = ((336, 595, 424, 711) if damage == "shifted_bbox"
                else (432, 595, 520, 711))
        point = ((380, 653) if damage == "shifted_bbox" else (476, 653))
    else:
        image = read_frame(spent).copy()
        if damage == "bottom":
            image[1414:1432, 686:830] = (40, 40, 40)
        elif damage == "left_corner":
            image[1416:1432, 686:702] = (40, 40, 40)
        elif damage == "right_corner":
            image[1416:1432, 814:830] = (40, 40, 40)
        elif damage == "left_turn":
            image[1392:1422, 670:686] = (40, 40, 40)
        elif damage == "right_turn":
            image[1392:1422, 830:846] = (40, 40, 40)
        else:
            image[1250:1390, 666:676] = (40, 40, 40)
        spent = tmp_path / f"{damage}.png"
        assert cv2.imwrite(str(spent), image)
    plan = SimpleNamespace(base=base, balloon_point=point, balloon_bbox=bbox)
    snapshot = SceneSnapshot("battle", .99, spent, {"ocr": [], "battle": {"slots": []}})
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    with pytest.raises(FlowError):
        _balloon_count(session, plan, snapshot, 0)


def test_old_real_spent_balloon_keeps_original_border_path(tmp_path):
    from autococ.vision import ScreenshotRecognizer

    source = FIXTURES / "frozen-line-001038-00018-toolbar.png"
    spent = FIXTURES / "frozen-line-001038-00020-toolbar.png"
    base, _ = frozen_base(tmp_path)
    frame = replace(base.frame, frame_id=str(source), screenshot_path=source)
    base = replace(base, frame=frame, frame_sha256=_file_sha256(source),
                   toolbar_before_frame_id=str(source),
                   toolbar_before_sha256=_file_sha256(source))
    plan = SimpleNamespace(base=base, balloon_point=(379, 653),
                           balloon_bbox=(335, 595, 423, 711))
    snapshot = SceneSnapshot("battle", .99, spent, {"ocr": [], "battle": {"slots": []}})
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(last_snapshot=snapshot, recognizer=recognizer,
                              client_version="18.647.2")
    reading = _balloon_count(session, plan, snapshot, 0)
    assert reading["count"] == 0
    assert "method" not in reading["current_card_border"]


def test_real_060543_anonymous_return_frames_prove_only_offline_old_camera_and_roster():
    from autococ.vision import ScreenshotRecognizer

    provenance = json.loads((FIXTURES / "probed-anchor-return-060543.provenance.json").read_text(encoding="utf-8"))
    assert provenance["client_version"] == "18.647.2"
    assert provenance["truth_status"] == "agent_visual_prototype_only"
    assert provenance["formal_success"] is False
    fixture = {row["frame_number"]: FIXTURES / row["fixture"] for row in provenance["rows"]}
    for row in provenance["rows"]:
        assert _file_sha256(fixture[row["frame_number"]]) == row["fixture_sha256"]
    anchors = ((375, 250), (525, 450))
    with pytest.raises(ExactCameraError, match="stationary map inliers"):
        verify_exact_anchor_camera(fixture[26], fixture[32], anchors)
    for number in (34, 35, 36):
        proof = verify_exact_anchor_camera(fixture[26], fixture[number], anchors)
        assert proof["inliers"] >= 80 and proof["residual_p95"] <= .5
        assert max(proof["anchor_offsets"]) <= .25
    for before, after in ((34, 35), (35, 36)):
        assert max(verify_exact_anchor_camera(fixture[before], fixture[after], anchors)["anchor_offsets"]) <= .25

    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    source = recognizer.recognize(fixture[24])
    assert source.scene == "enemy_village"
    slots = source.observations["battle"]["slots"]
    troops = [item for item in slots if item.get("kind") == "troop"]
    heroes = [item for item in slots if item.get("kind") == "hero"]
    assert {item["unit_id"]: item["count"] for item in troops} == dict(TROOPS)
    assert {item["unit_id"] for item in heroes} == set(HEROES)
    balloon = next(item for item in troops if item["unit_id"] == "balloon")
    base = SimpleNamespace(
        troop_cards=tuple((item["unit_id"], "army", item["count"], tuple(item["point"]),
                           tuple(item["bbox"])) for item in troops),
        hero_cards=tuple((item["unit_id"], tuple(item["bbox"])) for item in heroes),
        client_version="18.647.2", toolbar_before_frame_id=str(fixture[24]),
        frame=SimpleNamespace(screenshot_path=fixture[26], frame_id=str(fixture[26]),
                              resolution=(1280, 720)), frame_sha256=_file_sha256(fixture[26]))
    plan = SimpleNamespace(base=base,
                           checkpoint_cards=tuple((item["unit_id"], tuple(item["point"]),
                                                   tuple(item["bbox"])) for item in troops
                                                  if item["unit_id"] != "balloon"),
                           balloon_point=tuple(balloon["point"]), balloon_bbox=tuple(balloon["bbox"]))
    for number in (34, 35, 36):
        frame = recognizer.recognize(fixture[number])
        assert frame.scene == "battle"
        offline = SimpleNamespace(recognizer=recognizer, last_snapshot=frame)
        assert _balloon_count(offline, plan, frame, 0)["count"] == 0
        roster = _remaining_roster_checkpoint(offline, plan, frame)
        assert {item["unit_id"] for item in roster["troops"]} == set(dict(TROOPS)) - {"balloon"}
        assert {item["unit_id"] for item in roster["heroes"]} == set(HEROES)


def test_exact_camera_rejects_shift_and_occlusion(tmp_path):
    rng = np.random.default_rng(14)
    image = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    same = tmp_path / "same.png"
    shifted = tmp_path / "shifted.png"
    occluded = tmp_path / "occluded.png"
    cv2.imwrite(str(source), image)
    cv2.imwrite(str(same), image)
    cv2.imwrite(str(shifted), cv2.warpAffine(image, np.float32([[1, 0, 1], [0, 1, 0]]), (1280, 720)))
    blocked = image.copy()
    blocked[150:490, 240:1050] = 0
    cv2.imwrite(str(occluded), blocked)
    anchors = ((375, 250), (525, 450))
    assert max(verify_exact_anchor_camera(source, same, anchors)["anchor_offsets"]) <= .25
    with pytest.raises(FlowError):
        verify_exact_anchor_camera(source, shifted, anchors)
    with pytest.raises(FlowError):
        verify_exact_anchor_camera(source, occluded, anchors)


def test_current_probe_frame_requires_fresh_source_version_display_and_foreground(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    current = image_frame(tmp_path, "fresh", "battle", 10.2)
    session = Session(current, [])
    session.capture.adb = object()
    session.capture.serial = "offline"
    session.config.game.package_name = "com.supercell.clashofclans"
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_a, **_kw: session.config.game.package_name)
    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera",
                        lambda *_a: {"inliers": 100, "residual_p95": 0.})
    assert _current_frame(session, plan, current, after=10., scene="battle")["frame"] == str(current.screenshot_path)
    with pytest.raises(FlowError):
        _current_frame(session, plan, current, after=10.2, scene="battle")
    session.client_version = "18.600.7"
    with pytest.raises(FlowError):
        _current_frame(session, plan, current, after=10., scene="battle")
    session.client_version = "18.647.2"
    session.capture.input_display_id = 1
    with pytest.raises(FlowError):
        _current_frame(session, plan, current, after=10., scene="battle")
    session.capture.input_display_id = 0
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_a, **_kw: "other.package")
    with pytest.raises(FlowError):
        _current_frame(session, plan, current, after=10., scene="battle")
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_a, **_kw: session.config.game.package_name)
    Path(plan.base.frame.screenshot_path).write_bytes(b"changed original pixels")
    with pytest.raises(FlowError):
        _current_frame(session, plan, current, after=10., scene="battle")


def test_checkpoint_requires_all_other_troops_and_initial_four_heroes_on_one_frame(monkeypatch, tmp_path):
    plan, _ = make_plan(monkeypatch, tmp_path)
    frame = image_frame(tmp_path, "checkpoint", "battle", 10.3)
    session = Session(frame, [])
    session.recognizer = object()
    monkeypatch.setattr("autococ.images.read_frame",
                        lambda _path: np.zeros((720, 1280, 3), dtype=np.uint8))
    inspected = []
    def edges(_image, bbox):
        inspected.append(bbox)
        return {"left_edge": 300, "right_edge": -300, "bottom_contrast": 50}
    monkeypatch.setattr("autococ.probed_anchor._current_card_edges", edges)
    from autococ.unit_catalog import CATALOG_ROOT
    cards = {tuple(bbox): unit_id for unit_id, _, bbox in plan.checkpoint_cards}
    cards.update({tuple(box): name for name, box in plan.base.hero_cards})
    wrong = {"unit_id": None, "deployed": False, "count": None}
    def identity(_image, box, kind, **_kwargs):
        unit_id = cards[tuple(box)]
        if wrong["unit_id"] == unit_id:
            unit_id = "other_hero"
        return {"unit_id": unit_id, "confidence": .99, "version_verified": True,
                "evidence": [{"template": str((CATALOG_ROOT / f"units/{unit_id}_battle_18_647_2.png").resolve())}]}
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity", identity)
    monkeypatch.setattr("autococ.unit_catalog.template_manifest", lambda **_kw: {"templates": [
        {"surface": "battle", "state": "initial_scout", "unit_id": unit_id,
         "path": f"units/{unit_id}_battle_18_647_2.png"} for unit_id in HEROES]})
    monkeypatch.setattr("autococ.hero_state.recognize_hero_state",
                        lambda _path, _box, *, unit_id, **_kw: {
                            "deployed": wrong["deployed"] if unit_id == "archer_queen" else None,
                            "state": "unknown", "evidence": {"portrait_identity": {"unit_id": unit_id}}})
    def count(_session, _post, _source, _point, bbox, *, unit_id):
        expected = next(row[2] for row in plan.base.troop_cards if row[0] == unit_id)
        actual = wrong["count"] if unit_id == "dragon" and wrong["count"] is not None else expected
        return {"count": actual, "confidence": .99, "frame": str(frame.screenshot_path),
                "slot_bbox": list(bbox), "portrait_score": .99,
                "portrait_source": "current_frame_saved_battle_portrait",
                "readings": [{"text": f"x{actual}", "confidence": .99}]}
    monkeypatch.setattr("autococ.probed_anchor._post_troop_reading", count)
    proof = _remaining_roster_checkpoint(session, plan, frame)
    assert len(proof["troops"]) == 4 and len(proof["heroes"]) == 4
    assert proof["troops"][0]["unit_id"] == "electro_dragon"
    assert inspected[0] == plan.checkpoint_cards[0][2]
    assert plan.base.troop_cards[0][4] not in inspected
    frame.observations["battle"]["slots"] = [{"unit_id": "electro_dragon",
        "bbox": list(plan.base.troop_cards[0][4]), "count": 9, "confidence": .99}]
    with pytest.raises(FlowError, match="strong full-card quantity conflict"):
        _remaining_roster_checkpoint(session, plan, frame)  # Local x10 cannot override full x9.
    frame.observations["battle"]["slots"] = []
    wrong["count"] = 0
    with pytest.raises(FlowError, match="dragon quantity"):
        _remaining_roster_checkpoint(session, plan, frame)
    wrong["count"] = None
    wrong["deployed"] = True
    with pytest.raises(FlowError, match="archer_queen"):
        _remaining_roster_checkpoint(session, plan, frame)
    wrong["deployed"] = False
    wrong["unit_id"] = "minion_prince"
    with pytest.raises(FlowError, match="minion_prince"):
        _remaining_roster_checkpoint(session, plan, frame)
