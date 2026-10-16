"""Synthetic executor contracts; no game connection or model readiness is implied."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autococ.errors import DeploymentError, FlowError, StopRequested
from autococ.frozen_line import (_cards, _closed_dragon_card_border_18_647_2, _count_proves,
                                 _post_troop_reading, execute_frozen_line,
                                 freeze_model_free_line)
from autococ.scene import SceneSnapshot


TROOPS = (("electro_dragon", 10), ("dragon", 1), ("baby_dragon", 1), ("balloon", 2))


def slot(unit_id, count, x, *, kind="troop"):
    evidence = {"identity": {"unit_id": unit_id, "confidence": .99,
                             "version_verified": True, "template_client_version": "18.647.2"}}
    if kind == "troop":
        evidence["count"] = {"text": f"x{count}", "confidence": .99}
    return {"kind": kind, "unit_id": unit_id, "source": "army", "count": count,
            "confidence": .99, "point": [x, 650], "bbox": [x - 40, 580, x + 40, 710],
            "evidence": evidence}


def frame(name, captured_at, *, root: Path, post=False, heroes=False, troops=TROOPS):
    cards = [slot(unit_id, 0 if post else count, 140 + index * 97)
             for index, (unit_id, count) in enumerate(troops)]
    if heroes:
        cards.append(slot("grand_warden", None, 650, kind="hero"))
    manifest = {"complete": True, "identity_complete": True, "unknowns": [],
                "troops": [{"unit_id": unit_id, "source": "army", "count": count}
                           for unit_id, count in troops]}
    path = root / f"{name}.png"
    path.write_bytes(f"immutable pixels: {name}".encode())
    return SceneSnapshot("battle", .99, path,
                         {"observed_at_monotonic": captured_at,
                          "baseline_resolution": (1280, 720),
                          "source_resolution": (2560, 1440),
                          "expected_army_manifest": manifest,
                          "battle": {"slots": cards}})


def terrain(selected):
    return [{"edge": edge, "point": [280 + edge * 30, y],
             "evidence": {"method": "current_red_boundary_offset",
                          "frame": str(selected.screenshot_path),
                          "horizontal_clearance": 32,
                          "slope": -.75 if edge == 0 else .75, "intercept": 300.}}
            for edge, ys in ((0, (160, 260)), (1, (370, 470))) for y in ys]


class Session:
    client_version = "18.647.2"
    deadline = 999
    task_deadline = 999

    def __init__(self, selected, post, *, fail_at=None):
        self.last_snapshot = selected
        self.post = post
        self.fail_at = fail_at
        self.inputs = []
        self.observes = []
        self.capture = SimpleNamespace(input_display_id=0)
        self.context = SimpleNamespace(screen_resolution=(2560, 1440))
        self.config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720)),
                                      runtime=SimpleNamespace(step_timeout_sec=5))

    def check_deadline(self):
        pass

    def frozen_battle_tap(self, point, *, timeout_sec, reason, plan_id):
        assert self._active_frozen_plan[0] == plan_id
        assert not self.observes
        self.inputs.append((point, reason))
        if len(self.inputs) == self.fail_at:
            raise TimeoutError("transport result unknown")

    def observe(self, purpose):
        self.observes.append(purpose)
        self.last_snapshot = self.post
        return self.post


def freeze(session, selected, before, *, heroes=(), line_tactics="single_edge"):
    hero_manifest = None
    if heroes:
        hero_manifest = {"complete": True, "unknowns": [], "frame": str(before.screenshot_path),
                         "cards": [{"unit_id": hero_id, "kind": "hero", "source": "army",
                                    "confidence": .99, "evidence": {"identity": {
                                        "unit_id": hero_id, "confidence": .99, "version_verified": True,
                                        "template_client_version": "18.647.2"}}} for hero_id in heroes]}
    return freeze_model_free_line(session, selected, toolbar_before=before,
                                  terrain=terrain(selected),
                                  army_manifest=selected.observations["expected_army_manifest"],
                                  minimum_zoom={"verified": True, "frame": str(before.screenshot_path),
                                                "motion": {"stationary": True, "matched_points": 100,
                                                           "inliers": 95, "affine": [[1, 0, 0], [0, 1, 0]]}},
                                  battle_id="battle-one",
                                  latest_start=105., burst_limit_sec=15.,
                                  line_tactics=line_tactics,
                                  hero_ids=heroes, hero_manifest=hero_manifest, clock=lambda: 101.)


@pytest.fixture(autouse=True)
def offline_post_reader(monkeypatch):
    monkeypatch.setattr("autococ.frozen_line._post_troop_reading",
                        lambda _session, post, _before, _point, bbox, **_kwargs: {
                            "count": 0, "confidence": .99, "frame": str(post),
                            "slot_bbox": list(bbox), "portrait_score": .99,
                            "portrait_source": "current_frame_saved_battle_portrait",
                            "readings": [{"text": "x0", "confidence": .99}]})


def test_four_troops_burst_has_zero_midstream_observations_and_fresh_counts(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected, post = (frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path),
                              frame("post", 103, root=tmp_path, post=True))
    session = Session(selected, post)
    plan = freeze(session, selected, before)
    assert len(plan.actions) == 17
    assert plan.burst_limit_sec == 5.25
    assert not any(action.kind in {"spell", "siege", "skill"} for action in plan.actions)
    assert len([action for action in plan.actions if action.kind == "deploy"]) == 14
    assert execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None) is post
    assert session.observes == ["frozen-line-post"]
    assert len(session.inputs) == len(plan.actions)
    assert session.frozen_line_receipt["completed"] is True
    assert session.frozen_line_receipt["deployed_units"] == 14
    assert session.frozen_line_receipt["burst_pass"] is True
    assert session.frozen_line_receipt["attempted_placements"] == 14
    assert all(item["verified"] for item in session.frozen_line_receipt["troops"])
    assert not hasattr(plan, "model_sha256") and not hasattr(plan, "core")


@pytest.mark.parametrize("archer_count", [1, 2, 3])
def test_bounded_archer_filler_is_last_and_all_counts_are_consumed(monkeypatch, tmp_path, archer_count):
    selected_boxes = []
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox",
                        lambda _path, bbox, *_args: selected_boxes.append(tuple(bbox)) or [100, 580, 180, 710])
    troops = (("archer", archer_count), *TROOPS)
    before, selected, post = (frame("archer-before", 99, root=tmp_path, troops=troops),
                              frame("archer-selected", 100, root=tmp_path, troops=troops),
                              frame("archer-post", 103, root=tmp_path, post=True, troops=troops))
    session = Session(selected, post)
    plan = freeze(session, selected, before)

    deployed = [action.unit_id for action in plan.actions if action.kind == "deploy"]
    expected_order = [unit_id for unit_id, count in TROOPS for _ in range(count)] + ["archer"] * archer_count
    assert deployed == expected_order
    assert selected_boxes == [tuple(selected.observations["battle"]["slots"][0]["bbox"])]
    assert [action.unit_id for action in plan.actions if action.kind == "select"] == [
        "electro_dragon", "dragon", "baby_dragon", "balloon", "archer"]
    assert len(plan.troop_budgets) == 5
    assert len(plan.troop_cards) == 5
    assert len(plan.actions) == 14 + archer_count + 5
    assert plan.burst_limit_sec == min(15., 1. + .25 * len(plan.actions))
    assert execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None) is post
    assert session.observes == ["frozen-line-post"]
    assert session.frozen_line_receipt["deployed_units"] == 14 + archer_count
    assert [row["unit_id"] for row in session.frozen_line_receipt["troops"]] == [
        "archer", *(unit_id for unit_id, _count in TROOPS)]
    assert all(row["verified"] for row in session.frozen_line_receipt["troops"])


@pytest.mark.parametrize("troops", [
    (*TROOPS, ("archer", 0)),
    (*TROOPS, ("archer", 4)),
    (*TROOPS, ("barbarian", 1)),
    (*TROOPS, ("archer", 1), ("wizard", 1)),
])
def test_frozen_line_rejects_unbounded_or_unsupported_filler_rosters(monkeypatch, tmp_path, troops):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = (frame("invalid-roster-before", 99, root=tmp_path, troops=troops),
                        frame("invalid-roster-selected", 100, root=tmp_path, troops=troops))
    session = Session(selected, frame("invalid-roster-post", 103, root=tmp_path, post=True, troops=troops))
    with pytest.raises(FlowError):
        freeze(session, selected, before)


def test_frozen_line_rejects_archer_count_or_identity_without_current_version_same_frame_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    troops = (("archer", 3), *TROOPS)
    before, selected = (frame("invalid-archer-before", 99, root=tmp_path, troops=troops),
                        frame("invalid-archer-selected", 100, root=tmp_path, troops=troops))
    archer = next(card for card in selected.observations["battle"]["slots"] if card["unit_id"] == "archer")
    archer["evidence"]["identity"]["template_client_version"] = "18.600.7"
    session = Session(selected, frame("invalid-archer-post", 103, root=tmp_path, post=True, troops=troops))
    with pytest.raises(FlowError, match="current-version identity"):
        freeze(session, selected, before)

    selected = frame("invalid-archer-count-selected", 100, root=tmp_path, troops=troops)
    archer = next(card for card in selected.observations["battle"]["slots"] if card["unit_id"] == "archer")
    archer["evidence"]["count"]["text"] = "x2"
    session = Session(selected, frame("invalid-archer-count-post", 103, root=tmp_path, post=True, troops=troops))
    with pytest.raises(FlowError, match="same-card OCR evidence"):
        freeze(session, selected, before)


def test_uncertain_transport_is_one_shot_and_never_uses_observe_or_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    session = Session(selected, frame("post", 103, root=tmp_path, post=True), fail_at=2)
    plan = freeze(session, selected, before)
    with pytest.raises(DeploymentError, match="transport result unknown") as error:
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert len(session.inputs) == 2 and not session.observes
    assert error.value.partial_receipt["actions"][-1]["status"] == "uncertain"
    assert error.value.partial_receipt["attempted_placements"] == 2
    with pytest.raises(FlowError, match="already attempted"):
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert len(session.inputs) == 2


def test_plan_tamper_stale_display_and_unbound_terrain_fail_before_input(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    session = Session(selected, frame("post", 103, root=tmp_path, post=True))
    plan = freeze(session, selected, before)
    with pytest.raises(FlowError, match="digest"):
        execute_frozen_line(session, replace(plan, latest_start=200), clock=lambda: 102.)
    session.capture.input_display_id = 1
    with pytest.raises(FlowError, match="display/frame changed"):
        execute_frozen_line(session, plan, clock=lambda: 102.)
    assert not session.inputs
    session.capture.input_display_id = 0
    selected.screenshot_path.write_bytes(b"changed after freeze")
    with pytest.raises(FlowError, match="source pixels changed"):
        execute_frozen_line(session, plan, clock=lambda: 102.)
    selected.screenshot_path.write_bytes(b"immutable pixels: selected")
    bad_terrain = terrain(selected)
    bad_terrain[0]["evidence"]["frame"] = "other.png"
    with pytest.raises(FlowError, match="bound to the selected source frame"):
        freeze_model_free_line(session, selected, toolbar_before=before, terrain=bad_terrain,
                               army_manifest=selected.observations["expected_army_manifest"],
                               minimum_zoom={"verified": True, "frame": str(before.screenshot_path),
                                             "motion": {"stationary": True, "matched_points": 100,
                                                        "inliers": 95, "affine": [[1, 0, 0], [0, 1, 0]]}},
                               battle_id="battle-one", latest_start=105., burst_limit_sec=15.,
                               clock=lambda: 101.)


def test_optional_identity_verified_hero_is_deployed_without_skill(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    monkeypatch.setattr("autococ.hero_state.locate_current_deployed_hero_card",
                        lambda path, _hero_id, **_kwargs: {
                            "bbox": [610, 595, 699, 711], "frame": str(path),
                            "template_client_version": "18.647.2"})
    monkeypatch.setattr("autococ.hero_state.recognize_hero_state",
                        lambda path, bbox, **_kwargs: {"deployed": True, "frame": str(path),
                            "evidence": {"portrait_identity": {"unit_id": "grand_warden",
                                "version_verified": True, "template_client_version": "18.647.2"}}})
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                        lambda *_args, **_kwargs: {"unit_id": "grand_warden", "confidence": .99,
                                                     "version_verified": True})
    monkeypatch.setattr("autococ.images.read_frame", lambda *_args: __import__("numpy").zeros((720, 1280, 3), dtype="uint8"))
    before, selected, post = (frame("before", 99, root=tmp_path, heroes=True),
                              frame("selected", 100, root=tmp_path, heroes=True),
                              frame("post", 103, root=tmp_path, post=True, heroes=True))
    session = Session(selected, post)
    plan = freeze(session, selected, before, heroes=("grand_warden",))
    assert len([action for action in plan.actions if action.unit_id == "grand_warden"]) == 2
    execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert session.frozen_line_receipt["heroes"][0]["verified"] is True


def test_post_hero_animation_reobserves_only_after_burst_with_fresh_frame(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                        lambda *_args, **_kwargs: {"unit_id": "grand_warden", "confidence": .99,
                                                     "version_verified": True})
    monkeypatch.setattr("autococ.images.read_frame", lambda *_args: __import__("numpy").zeros((720, 1280, 3), dtype="uint8"))
    before, selected = frame("before", 99, root=tmp_path, heroes=True), frame("selected", 100, root=tmp_path, heroes=True)
    first, stable = frame("post-faded", 103, root=tmp_path, post=True), frame("post-stable", 104, root=tmp_path, post=True)
    session = Session(selected, first)
    pending = iter((first, stable))
    def observe(purpose):
        session.observes.append(purpose)
        session.last_snapshot = next(pending)
        return session.last_snapshot
    session.observe = observe
    monkeypatch.setattr("autococ.hero_state.locate_current_deployed_hero_card",
                        lambda path, _hero_id, **_kwargs: None if str(path) == str(first.screenshot_path) else {
                            "bbox": [610, 595, 699, 711], "frame": str(path),
                            "template_client_version": "18.647.2"})
    monkeypatch.setattr("autococ.hero_state.recognize_hero_state",
                        lambda path, _bbox, **_kwargs: {"deployed": True, "frame": str(path),
                            "evidence": {"portrait_identity": {"unit_id": "grand_warden",
                                "version_verified": True, "template_client_version": "18.647.2"}}})
    plan = freeze(session, selected, before, heroes=("grand_warden",))
    assert execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None) is stable
    assert session.observes == ["frozen-line-post", "frozen-line-post-recheck"]
    assert len(session.inputs) == len(plan.actions)
    assert [row["heroes"] for row in session.frozen_line_receipt["verification_frames"]] == [[False], [True]]
    assert session.frozen_line_receipt["completed"] is True


def test_post_recheck_rejects_reused_frame_without_more_input(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                        lambda *_args, **_kwargs: {"unit_id": "grand_warden", "confidence": .99,
                                                     "version_verified": True})
    monkeypatch.setattr("autococ.images.read_frame", lambda *_args: __import__("numpy").zeros((720, 1280, 3), dtype="uint8"))
    monkeypatch.setattr("autococ.hero_state.locate_current_deployed_hero_card", lambda *_args, **_kwargs: None)
    before, selected, post = (frame("before", 99, root=tmp_path, heroes=True),
                              frame("selected", 100, root=tmp_path, heroes=True),
                              frame("post", 103, root=tmp_path, post=True))
    session = Session(selected, post)
    plan = freeze(session, selected, before, heroes=("grand_warden",))
    with pytest.raises(DeploymentError, match="Fresh post-burst battle frame unavailable"):
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert len(session.inputs) == len(plan.actions)
    assert session.observes == ["frozen-line-post", "frozen-line-post-recheck"]


def test_missing_post_count_evidence_uses_only_two_fresh_observations_without_replay(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    posts = [frame(f"post-{number}", 103 + number, root=tmp_path, post=True) for number in range(3)]
    monkeypatch.setattr("autococ.frozen_line._post_troop_reading",
                        lambda _session, post_path, _before, _point, bbox, **_kwargs: {
                            "count": None, "confidence": .4, "frame": str(post_path),
                            "slot_bbox": list(bbox)})
    session = Session(selected, posts[0])
    pending = iter(posts)
    def observe(purpose):
        session.observes.append(purpose)
        session.last_snapshot = next(pending)
        return session.last_snapshot
    session.observe = observe
    plan = freeze(session, selected, before)
    with pytest.raises(DeploymentError, match="consumption is unverified") as error:
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert len(session.inputs) == len(plan.actions)
    assert session.observes == ["frozen-line-post", "frozen-line-post-recheck",
                                "frozen-line-post-recheck"]
    assert len(error.value.partial_receipt["verification_frames"]) == 3
    assert error.value.partial_receipt["completed"] is False
    with pytest.raises(FlowError, match="permission expired or display/frame changed"):
        execute_frozen_line(session, plan, clock=lambda: 102.)


def test_unknown_post_troop_recovers_only_from_fresh_same_frame_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    first, second = (frame("post-unclear", 103, root=tmp_path, post=True),
                     frame("post-clear", 104, root=tmp_path, post=True))
    session = Session(selected, first)
    pending = iter((first, second))
    def observe(purpose):
        session.observes.append(purpose)
        session.last_snapshot = next(pending)
        return session.last_snapshot
    session.observe = observe
    def read(_session, post, _before, _point, bbox, **_kwargs):
        if str(post) == str(first.screenshot_path) and bbox[0] == 100:
            return {}  # no guessed x0 from the prebattle army
        return {"count": 0, "confidence": .99, "frame": str(post), "slot_bbox": list(bbox),
                "portrait_score": .99, "portrait_source": "current_frame_saved_battle_portrait",
                "readings": [{"text": "x0", "confidence": .99}]}
    monkeypatch.setattr("autococ.frozen_line._post_troop_reading", read)
    plan = freeze(session, selected, before)
    assert execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None) is second
    assert session.observes == ["frozen-line-post", "frozen-line-post-recheck"]
    assert len(session.inputs) == len(plan.actions)
    assert [row["troops"] for row in session.frozen_line_receipt["verification_frames"]] == [
        [False, True, True, True], [True, True, True, True]]


def test_positive_post_count_conflict_stops_without_rechecking_or_redeploying(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected, post = (frame("before", 99, root=tmp_path),
                              frame("selected", 100, root=tmp_path),
                              frame("post", 103, root=tmp_path, post=True))
    post.observations["battle"]["slots"][0]["count"] = 1
    session = Session(selected, post)
    plan = freeze(session, selected, before)
    with pytest.raises(DeploymentError, match="consumption is unverified"):
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert session.observes == ["frozen-line-post"]
    assert len(session.inputs) == len(plan.actions)


def test_strong_ocr_count_conflict_stops_without_recheck_or_redeployment(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected, post = (frame("conflict-before", 99, root=tmp_path),
                              frame("conflict-selected", 100, root=tmp_path),
                              frame("conflict-post", 103, root=tmp_path, post=True))
    session = Session(selected, post)
    plan = freeze(session, selected, before)
    def read(_session, path, _before, _point, bbox, **_kwargs):
        if bbox == plan.troop_cards[0][4]:
            return {"conflict": True, "reason": "strong_count_conflict", "counts": [0, 1],
                    "frame": str(path), "slot_bbox": list(bbox)}
        return {"count": 0, "confidence": .99, "frame": str(path), "slot_bbox": list(bbox),
                "portrait_score": .99, "portrait_source": "current_frame_saved_battle_portrait",
                "readings": [{"text": "x0", "confidence": .99}]}
    monkeypatch.setattr("autococ.frozen_line._post_troop_reading", read)
    with pytest.raises(DeploymentError, match="consumption is unverified") as error:
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert session.observes == ["frozen-line-post"]
    assert len(session.inputs) == len(plan.actions)
    receipt = error.value.partial_receipt
    assert len(receipt["verification_frames"]) == 1
    assert receipt["troops"][0]["reading"]["reason"] == "strong_count_conflict"
    assert receipt["troops"][0]["verified"] is False


def test_rechecks_never_combine_troops_and_hero_from_different_frames(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                        lambda *_args, **_kwargs: {"unit_id": "grand_warden", "confidence": .99,
                                                     "version_verified": True})
    monkeypatch.setattr("autococ.images.read_frame", lambda *_args: __import__("numpy").zeros((720, 1280, 3), dtype="uint8"))
    before, selected = frame("before", 99, root=tmp_path, heroes=True), frame("selected", 100, root=tmp_path, heroes=True)
    posts = [frame(f"post-{index}", 103 + index, root=tmp_path, post=True)
             for index in range(3)]
    session = Session(selected, posts[0])
    pending = iter(posts)
    def observe(purpose):
        session.observes.append(purpose)
        session.last_snapshot = next(pending)
        return session.last_snapshot
    session.observe = observe
    def read(_session, path, _before, _point, bbox, **_kwargs):
        if str(path) != str(posts[0].screenshot_path):
            return {}
        return {"count": 0, "confidence": .99, "frame": str(path), "slot_bbox": list(bbox),
                "portrait_score": .99, "portrait_source": "current_frame_saved_battle_portrait",
                "readings": [{"text": "x0", "confidence": .99}]}
    monkeypatch.setattr("autococ.frozen_line._post_troop_reading", read)
    monkeypatch.setattr("autococ.hero_state.locate_current_deployed_hero_card",
                        lambda path, _hero, **_kwargs: None if str(path) == str(posts[0].screenshot_path) else {
                            "bbox": [610, 595, 699, 711], "frame": str(path),
                            "template_client_version": "18.647.2"})
    monkeypatch.setattr("autococ.hero_state.recognize_hero_state",
                        lambda path, _bbox, **_kwargs: {"deployed": True, "frame": str(path),
                            "evidence": {"portrait_identity": {"unit_id": "grand_warden",
                                "version_verified": True, "template_client_version": "18.647.2"}}})
    plan = freeze(session, selected, before, heroes=("grand_warden",))
    with pytest.raises(DeploymentError, match="consumption is unverified") as error:
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    rows = error.value.partial_receipt["verification_frames"]
    assert len(rows) == 3 and rows[0]["troops"] == [True] * 4 and rows[0]["heroes"] == [False]
    assert all(row["troops"] == [False] * 4 and row["heroes"] == [True] for row in rows[1:])
    assert len(session.inputs) == len(plan.actions)


def test_recheck_unknown_scene_stops_without_any_new_game_input(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    first = frame("post-unclear", 103, root=tmp_path, post=True)
    unknown = replace(frame("post-unknown", 104, root=tmp_path, post=True), scene="unknown")
    session = Session(selected, first)
    pending = iter((first, unknown))
    def observe(purpose):
        session.observes.append(purpose)
        session.last_snapshot = next(pending)
        return session.last_snapshot
    session.observe = observe
    monkeypatch.setattr("autococ.frozen_line._post_troop_reading", lambda *_args, **_kwargs: {})
    plan = freeze(session, selected, before)
    with pytest.raises(DeploymentError, match="Fresh post-burst battle frame unavailable"):
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert session.observes == ["frozen-line-post", "frozen-line-post-recheck"]
    assert len(session.inputs) == len(plan.actions)


def test_vision_mode_never_enters_model_free_executor(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    session = Session(selected, frame("post", 103, root=tmp_path, post=True))
    plan = freeze(session, selected, before)
    session.config.vision_agent = SimpleNamespace(enabled=True)
    with pytest.raises(FlowError, match="cannot replace vision-mode"):
        freeze(session, selected, before)
    with pytest.raises(FlowError, match="cannot replace vision-mode"):
        execute_frozen_line(session, plan, clock=lambda: 102.)
    assert not session.inputs


def test_stop_propagates_and_shared_frame_is_consumed_without_replay(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path), frame("selected", 100, root=tmp_path)
    session = Session(selected, frame("post", 103, root=tmp_path, post=True))
    plan = freeze(session, selected, before)
    def stopped(*_args, **_kwargs):
        raise StopRequested()
    session.frozen_battle_tap = stopped
    with pytest.raises(StopRequested):
        execute_frozen_line(session, plan, clock=lambda: 102., sleep=lambda _seconds: None)
    assert session.observes == []
    assert session._active_frozen_plan is None
    assert plan.frame.frame_id in session._used_frozen_frames
    with pytest.raises(FlowError, match="already attempted"):
        execute_frozen_line(session, plan, clock=lambda: 102.)


def test_fast_count_shape_requires_current_frame_portrait_and_unambiguous_read(tmp_path):
    path = tmp_path / "current.png"
    bbox = (90, 589, 184, 709)
    reading = {"count": 10, "confidence": .985, "frame": str(path), "slot_bbox": list(bbox),
               "portrait_score": .99, "portrait_source": "current_frame_saved_battle_portrait",
               "readings": [{"text": "x10", "confidence": .985, "source": "fast_header_line"}]}
    assert _count_proves(reading, 10, path, bbox, "electro_dragon", "18.647.2")
    assert not _count_proves({**reading, "frame": "previous.png"}, 10, path, bbox,
                             "electro_dragon", "18.647.2")
    assert not _count_proves({**reading, "portrait_score": None}, 10, path, bbox,
                             "electro_dragon", "18.647.2")
    assert not _count_proves({**reading, "readings": reading["readings"] +
                             [{"text": "x1", "confidence": .95}]}, 10, path, bbox,
                             "electro_dragon", "18.647.2")


def test_hero_manifest_cannot_silently_omit_or_relabel_heroes(monkeypatch, tmp_path):
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_args: [100, 580, 180, 710])
    before, selected = frame("before", 99, root=tmp_path, heroes=True), frame("selected", 100, root=tmp_path, heroes=True)
    session = Session(selected, frame("post", 103, root=tmp_path, post=True))
    with pytest.raises(FlowError, match="hero manifest"):
        freeze_model_free_line(session, selected, toolbar_before=before, terrain=terrain(selected),
            army_manifest=selected.observations["expected_army_manifest"],
            minimum_zoom={"verified": True, "frame": str(before.screenshot_path),
                          "motion": {"stationary": True, "matched_points": 100, "inliers": 95,
                                     "affine": [[1, 0, 0], [0, 1, 0]]}},
            hero_manifest={"complete": True, "unknowns": [], "frame": str(before.screenshot_path),
                           "cards": [{"unit_id": "grand_warden"}]},
            hero_ids=(), battle_id="battle-one", latest_start=105., burst_limit_sec=15., clock=lambda: 101.)


@pytest.mark.parametrize("run,post_number", [
    ("20261015-192725-196937-2f0c35e9", "00016"),
    ("20261015-193505-038604-84c40a8d", "00015"),
    ("20261015-194523-915723-bff2f65f", "00015"),
])
def test_saved_real_frames_replay_gray_card_x0_with_current_portrait(run, post_number):
    # Optional local evidence from three completed battles; no emulator access.
    from autococ.vision import ScreenshotRecognizer

    events = Path(r"E:\projects\AutoCOC\reports\completion-20261015\live") / run / "events.jsonl"
    if not events.exists():
        pytest.skip("local real-battle evidence unavailable")
    rows = [json.loads(line) for line in events.open(encoding="utf-8")
            if '"kind": "observation"' in line]
    before = next(row for row in rows if str(row.get("frame", "")).endswith("00011-line-zoom-verified.png"))
    post = next(row for row in rows if str(row.get("frame", "")).endswith(f"{post_number}-line-consumption.png"))
    card = before["observations"]["battle"]["slots"][0]
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.647.2"
    result = _post_troop_reading(SimpleNamespace(recognizer=recognizer), Path(post["frame"]),
                                  before["frame"], tuple(card["point"]), tuple(card["bbox"]))
    assert result["count"] == 0
    assert result["confidence"] >= .9
    assert result["portrait_score"] >= .92
    assert result["portrait_source"] == "current_frame_saved_battle_portrait"
    assert result["frame"] == post["frame"]
    assert not any(slot.get("unit_id") == "electro_dragon" for slot in post["observations"]["battle"]["slots"])


@pytest.mark.parametrize("battle,post", [
    ("new", "post1"), ("new", "post2"), ("new", "post3"),
    ("prior_a", "post1"), ("prior_b", "post1"),
])
def test_anonymized_real_dragon_gray_card_requires_current_x0(battle, post):
    from autococ.ocr import RapidOCRProvider
    from autococ.vision import ScreenshotRecognizer

    root = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
    source, current = root / f"{battle}_source.png", root / f"{battle}_{post}.png"
    recognizer = ScreenshotRecognizer(provider=RapidOCRProvider())
    recognizer.client_version = "18.647.2"
    session = SimpleNamespace(recognizer=recognizer, client_version="18.647.2",
                              last_snapshot=SceneSnapshot("battle", .99, current, {"ocr": []}))
    result = _post_troop_reading(session, current, str(source), (185, 653), (141, 595, 230, 711),
                                  unit_id="dragon")
    assert result["count"] == 0
    assert result["confidence"] >= .9
    assert result["portrait_score"] >= .92
    assert result["locator"]["source"] == "current_frame_closed_card_header_geometry"
    assert result["current_card_border"]["method"] == "18.647.2_dragon_rounded_closed_card"


def test_anonymized_dragon_toolbar_provenance_preserves_exact_native_pixels():
    from autococ.images import read_frame

    root = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
    for path in root.glob("*.png"):
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        image = read_frame(path)
        x0, y0, x1, y1 = provenance["native_preserved_xyxy"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["output_sha256"]
        assert hashlib.sha256(image[y0:y1, x0:x1].tobytes()).hexdigest() == \
            provenance["native_preserved_pixel_sha256"]
        assert provenance["native_preserved_pixel_sha256"] == \
            provenance["output_preserved_pixel_sha256"]
        assert image[:y0].max() == 0


def test_dragon_specific_closed_shape_rejects_neighbor_shift_truncation_and_occlusion():
    import cv2
    from autococ.images import read_frame

    path = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2" / "new_post1.png"
    image = cv2.resize(read_frame(path), (1280, 720), interpolation=cv2.INTER_AREA)
    box = (141, 595, 230, 711)
    assert _closed_dragon_card_border_18_647_2(image, box)
    assert _closed_dragon_card_border_18_647_2(image, (149, 595, 238, 711)) is None
    for target in ("side", "bottom", "corner", "truncated"):
        changed = image.copy()
        if target == "side":
            changed[650:690, 139:145] = 0
        elif target == "bottom":
            changed[707:715, 175:198] = 0
        elif target == "corner":
            changed[699:708, 139:147] = 0
        else:
            changed[700:715, 139:232] = 0
        assert _closed_dragon_card_border_18_647_2(changed, box) is None, target


@pytest.mark.parametrize("variant", ["wrong_version", "neighbor", "missing_digit", "nonzero"])
def test_dragon_current_header_negative_evidence(tmp_path, variant):
    import cv2
    from autococ.images import read_frame
    from autococ.ocr import RapidOCRProvider
    from autococ.vision import ScreenshotRecognizer

    root = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
    source, post = root / "new_source.png", root / "new_post1.png"
    box = (141, 595, 230, 711)
    version = "18.600.7" if variant == "wrong_version" else "18.647.2"
    if variant == "neighbor":
        box = (238, 595, 327, 711)
    if variant in {"missing_digit", "nonzero"}:
        image = read_frame(post)
        if variant == "missing_digit":
            image[1184:1246, 370:456] = 0
        else:
            selected = read_frame(source)
            image[1184:1246, 370:456] = selected[1184:1246, 370:456]
        post = tmp_path / f"{variant}.png"
        post.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    recognizer = ScreenshotRecognizer(provider=RapidOCRProvider())
    recognizer.client_version = version
    session = SimpleNamespace(recognizer=recognizer, client_version=version,
                              last_snapshot=SceneSnapshot("battle", .99, post, {"ocr": []}))
    result = _post_troop_reading(session, post, str(source), (185, 653), box, unit_id="dragon")
    assert result.get("count") != 0


def test_dragon_current_x0_cannot_override_strong_same_frame_x1():
    from autococ.ocr import RapidOCRProvider
    from autococ.vision import ScreenshotRecognizer

    root = Path(__file__).parent / "fixtures" / "dragon_closed_card_18_647_2"
    source, post = root / "new_source.png", root / "new_post1.png"
    recognizer = ScreenshotRecognizer(provider=RapidOCRProvider())
    recognizer.client_version = "18.647.2"
    snapshot = SceneSnapshot("battle", .99, post, {"ocr": [
        {"text": "x1", "confidence": .99, "bbox": [188, 595, 225, 620]}]})
    session = SimpleNamespace(recognizer=recognizer, client_version="18.647.2",
                              last_snapshot=snapshot)
    result = _post_troop_reading(session, post, str(source), (185, 653), (141, 595, 230, 711),
                                  unit_id="dragon")
    assert result["reason"] == "strong_count_conflict"
    assert result["counts"] == [0, 1]


def test_saved_real_fast_selected_count_contract():
    events = Path(r"E:\projects\AutoCOC\reports\completion-20261015\live") / \
             "20261015-193505-038604-84c40a8d" / "events.jsonl"
    if not events.exists():
        pytest.skip("local real-battle evidence unavailable")
    rows = [json.loads(line) for line in events.open(encoding="utf-8")
            if '"kind": "observation"' in line]
    selected = next(row for row in rows if str(row.get("frame", "")).endswith("00014-line-selected.png"))
    assert selected["observations"]["battle"]["slots"] == []
    reading = selected["observations"]["deployment_count_reads"][0]
    assert "text" not in reading
    assert _count_proves(reading, 10, Path(selected["frame"]), tuple(reading["slot_bbox"]),
                         "electro_dragon", "18.647.2")


def test_saved_real_full_selected_toolbar_contract():
    events = Path(r"E:\projects\AutoCOC\reports\completion-20261015\live") / \
             "20261015-192725-196937-2f0c35e9" / "events.jsonl"
    if not events.exists():
        pytest.skip("local real-battle evidence unavailable")
    rows = [json.loads(line) for line in events.open(encoding="utf-8")
            if '"kind": "observation"' in line]
    selected = next(row for row in rows if str(row.get("frame", "")).endswith("00015-line-selected.png"))
    snapshot = SceneSnapshot(selected["scene"], selected["confidence"], Path(selected["frame"]),
                             selected["observations"])
    cards = _cards(snapshot, client_version="18.647.2")
    assert [card["unit_id"] for card in cards if card["kind"] == "troop"] == [unit_id for unit_id, _ in TROOPS]


def test_saved_real_full_plan_freezes_without_any_input():
    from autococ.terrain import find_line_deployment_edges

    events = Path(r"E:\projects\AutoCOC\reports\completion-20261015\live") / \
             "20261015-192725-196937-2f0c35e9" / "events.jsonl"
    if not events.exists():
        pytest.skip("local real-battle evidence unavailable")
    rows = [json.loads(line) for line in events.open(encoding="utf-8")]
    def snapshot(suffix):
        row = next(item for item in rows if item.get("kind") == "observation" and
                   str(item.get("frame", "")).endswith(suffix))
        return SceneSnapshot(row["scene"], row["confidence"], Path(row["frame"]), row["observations"])
    army = snapshot("00006-battle-army-confirmation.png").observations["army"]
    before = snapshot("00011-line-zoom-verified.png")
    selected = snapshot("00015-line-selected.png")
    zoom = next(item for item in rows if item.get("kind") == "line_minimum_zoom")
    minimum_zoom = {key: zoom[key] for key in ("verified", "frame", "motion")}
    session = SimpleNamespace(config=SimpleNamespace(vision_agent=SimpleNamespace(enabled=False),
                                                       game=SimpleNamespace(baseline_resolution=(1280, 720))),
                              client_version="18.647.2", last_snapshot=selected,
                              capture=SimpleNamespace(input_display_id=2),
                              context=SimpleNamespace(screen_resolution=(2560, 1440)))
    now = selected.observations["observed_at_monotonic"] + .05
    hero_ids = tuple(card["unit_id"] for card in army["hero_evidence"]["cards"])
    plan = freeze_model_free_line(session, selected, toolbar_before=before,
        terrain=find_line_deployment_edges(selected.screenshot_path),
        army_manifest=army["manifest"], minimum_zoom=minimum_zoom,
        battle_id="offline-replay", latest_start=now + 2, burst_limit_sec=15,
        hero_ids=hero_ids, hero_manifest=army["hero_evidence"], clock=lambda: now)
    assert len(plan.actions) == 25
    assert plan.burst_limit_sec == 7.25
    assert len(plan.hero_cards) == 4
    assert plan.troop_budgets == tuple((unit_id, "army", count) for unit_id, count in TROOPS)


@pytest.mark.parametrize("root,army_name,before_name,selected_name", (
    (Path(r"E:\projects\AutoCOC\reports\completion-20261015\probes\20261016-012515-manual-skill"),
     "00009-battle-army-roster-reread.png", "00014-line-zoom-verified.png", "00016-line-boundary.png"),
    (Path(r"E:\projects\AutoCOC\reports\completion-20261015\live\20261016-001038-401680-d5107440"),
     "00010-battle-army-roster-reread.png", "00017-line-zoom-verified.png", "00019-line-boundary.png"),
    (Path(r"E:\projects\AutoCOC\reports\completion-20261015\live\20261016-005304-767109-058a9bfa"),
     "00006-battle-army-confirmation.png", "00011-line-zoom-verified.png", "00013-line-boundary.png"),
))
def test_real_five_card_windows_freeze_explicit_two_edge_split_without_input(
        root, army_name, before_name, selected_name):
    from collections import Counter

    from autococ.terrain import find_line_deployment_edges

    events = root / "events.jsonl"
    if not events.exists():
        pytest.skip("local real-battle evidence unavailable")
    rows = [json.loads(line) for line in events.open(encoding="utf-8")]

    def snapshot(name):
        row = next(item for item in rows if item.get("kind") == "observation" and
                   Path(item.get("frame", "")).name == name)
        return SceneSnapshot(row["scene"], row["confidence"], Path(row["frame"]),
                             row["observations"])

    army = snapshot(army_name).observations["army"]
    before, selected = snapshot(before_name), snapshot(selected_name)
    zoom = next(item for item in rows if item.get("kind") == "line_minimum_zoom")
    minimum_zoom = {key: zoom[key] for key in ("verified", "frame", "motion")}
    current_terrain = find_line_deployment_edges(selected.screenshot_path)
    session = SimpleNamespace(
        config=SimpleNamespace(vision_agent=SimpleNamespace(enabled=False),
                               game=SimpleNamespace(baseline_resolution=(1280, 720))),
        client_version="18.647.2", last_snapshot=selected,
        capture=SimpleNamespace(input_display_id=2),
        context=SimpleNamespace(screen_resolution=(2560, 1440)))
    now = selected.observations["observed_at_monotonic"] + .05
    hero_manifest = army["hero_evidence"]
    plan = freeze_model_free_line(
        session, selected, toolbar_before=before, terrain=current_terrain,
        army_manifest=army["manifest"], minimum_zoom=minimum_zoom,
        battle_id="two-edge-offline-replay", latest_start=now + 2,
        burst_limit_sec=15, line_tactics="two_edge_split",
        hero_ids=[card["unit_id"] for card in hero_manifest["cards"]],
        hero_manifest=hero_manifest, clock=lambda: now)
    placements = [action for action in plan.actions if action.kind == "deploy"]
    counts = Counter(action.unit_id for action in placements)
    assert counts == {**{row["unit_id"]: row["count"] for row in army["manifest"]["troops"]},
                      **{row["unit_id"]: 1 for row in hero_manifest["cards"]}}
    assert plan.line_tactics == "two_edge_split"
    assert len(plan.actions) == 30 and len(placements) == 21
    assert plan.burst_limit_sec == 8.5
    assert sum(action.delay_after_sec for action in plan.actions) < plan.burst_limit_sec
    assert [action.unit_id for action in placements[:2]] == ["baby_dragon", "dragon"]
    assert [action.unit_id for action in placements[-4:]] == [
        "grand_warden", "archer_queen", "dragon_duke", "minion_prince"]
    y_bounds = {edge: (min(row["point"][1] for row in current_terrain if row["edge"] == edge),
                       max(row["point"][1] for row in current_terrain if row["edge"] == edge))
                for edge in (0, 1)}
    def on_edge(action, edge):
        return y_bounds[edge][0] <= action.point[1] <= y_bounds[edge][1]
    assert all(on_edge(action, 0) or on_edge(action, 1) for action in placements)
    assert sum(on_edge(action, 0) for action in placements if action.unit_id == "electro_dragon") == 7
    assert sum(on_edge(action, 1) for action in placements if action.unit_id == "electro_dragon") == 3
    assert sum(on_edge(action, 0) for action in placements if action.unit_id == "archer") == 2
    assert sum(on_edge(action, 1) for action in placements if action.unit_id == "archer") == 1
    assert not any(action.kind in {"spell", "siege", "skill"} for action in plan.actions)
