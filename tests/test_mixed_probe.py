"""Offline mixed probe contracts; these do not establish live battle readiness."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autococ.errors import DeploymentError, FlowError
from autococ.frozen_line import FrozenLinePlan, _file_sha256, _plan_digest
from autococ.mixed_probe import (execute_mixed_probe_line, freeze_mixed_probe_line,
                                 _probe_count, _remaining_checkpoint, remainder_burst_limit)
from autococ.ocr import OCRText
from autococ.prepared_battle import FramePacket, PreparedInput
from autococ.scene import SceneSnapshot
from autococ.strategies import BattleContext, TwoEdgeFurnaceStrategy
from autococ.vision import ScreenshotRecognizer


TERRAIN = ({"edge": 0, "point": (400, 200)}, {"edge": 0, "point": (350, 300)},
           {"edge": 1, "point": (500, 400)}, {"edge": 1, "point": (550, 500)})
TROOPS = (("electro_dragon", 10), ("dragon", 1), ("furnace", 1), ("balloon", 1))
HEROES = ("grand_warden", "archer_queen", "dragon_duke", "minion_prince")
ROOT = Path(__file__).resolve().parents[1]


def snapshot(tmp_path, name, scene, observed, *, fast=None):
    path = tmp_path / f"{name}.png"
    path.write_bytes(f"immutable {name}".encode())
    observations = {"observed_at_monotonic": observed, "source_resolution": (2560, 1440),
                    "baseline_resolution": (1280, 720), "battle": {"slots": []}}
    if fast:
        observations.update(recognition_scope=fast, fast_fallback=False,
                            matched_reasons=["two_current_hud_anchors"])
    return SceneSnapshot(scene, .99, path, observations)


def proof(frame):
    return {"frame": str(frame.screenshot_path), "sha256": _file_sha256(frame.screenshot_path),
            "observed_at": frame.observations["observed_at_monotonic"],
            "camera": {"stationary": True}}


def base_plan(tmp_path):
    source = snapshot(tmp_path, "source", "enemy_village", 9.)
    before = snapshot(tmp_path, "before", "enemy_village", 8.)
    troop_cards = tuple((unit_id, "army", count, (150 + i * 90, 650),
                         (110 + i * 90, 595, 190 + i * 90, 711))
                        for i, (unit_id, count) in enumerate(TROOPS))
    hero_cards = tuple((unit_id, (600 + i * 90, 595, 680 + i * 90, 711))
                       for i, unit_id in enumerate(HEROES))
    troops = tuple({"unit_id": unit_id, "count": count, "source": "army", "point": card[3]}
                   for card, (unit_id, count) in zip(troop_cards, TROOPS))
    heroes = tuple({"unit_id": unit_id, "source": "army", "point": (640 + i * 90, 650)}
                  for i, unit_id in enumerate(HEROES))
    placement = TwoEdgeFurnaceStrategy().build_plan(BattleContext(troops, heroes, TERRAIN))
    actions = []
    selected = "electro_dragon"
    for index, step in enumerate(placement.troops):
        unit_id = step.card["unit_id"]
        if selected != unit_id:
            actions.append(PreparedInput(f"select-{index}", "select", step.card["point"],
                                         unit_id, "army"))
        selected = unit_id
        for offset, point in enumerate(step.points):
            actions.append(PreparedInput(f"troop-{index}-{offset}", "deploy", point,
                                         unit_id, "army", .12))
    for index, step in enumerate(placement.heroes):
        unit_id = step.card["unit_id"]
        actions.append(PreparedInput(f"select-hero-{index}", "select", step.card["point"],
                                     unit_id, "army"))
        actions.append(PreparedInput(f"hero-{index}", "deploy", step.points[0],
                                     unit_id, "army", .12))
    packet = FramePacket(str(source.screenshot_path), 9., "enemy_village", .99,
                         source.screenshot_path, (1280, 720), (2560, 1440), 0)
    base = FrozenLinePlan("plan", "battle", packet, _file_sha256(source.screenshot_path),
                          str(before.screenshot_path), _file_sha256(before.screenshot_path),
                          "18.647.2", 9., 11., 10., "cards", "terrain", tuple(actions),
                          tuple((name, "army", count) for name, count in TROOPS), troop_cards,
                          HEROES, hero_cards, None, None, None, "zoom",
                          "two_edge_furnace_mixed_probes", "")
    return replace(base, plan_sha256=_plan_digest(base)), source, before


class Session:
    client_version = "18.647.2"
    deadline = 100.
    task_deadline = 100.

    def __init__(self, source, observations):
        self.last_snapshot = source
        self.observations = iter(observations)
        self.inputs = []
        self.capture = SimpleNamespace(input_display_id=0)
        self.context = SimpleNamespace(screen_resolution=(2560, 1440))
        self.config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720)),
                                      runtime=SimpleNamespace(step_timeout_sec=5))

    def check_deadline(self):
        pass

    def tap(self, frame, point, *, reason):
        assert frame is self.last_snapshot
        self.inputs.append(("tap", tuple(point)))

    def frozen_battle_tap(self, point, *, timeout_sec, reason, plan_id):
        assert self._active_frozen_plan[0] == plan_id
        self.inputs.append(("frozen", tuple(point)))

    def observe(self, _label, **_kwargs):
        self.last_snapshot = next(self.observations)
        return self.last_snapshot


def make_plan(monkeypatch, tmp_path):
    base, source, before = base_plan(tmp_path)
    monkeypatch.setattr("autococ.mixed_probe.freeze_model_free_line", lambda *_a, **_kw: base)
    monkeypatch.setattr("autococ.mixed_probe._cards", lambda *_a, **_kw: tuple(
        {"kind": "troop", "unit_id": unit_id, "source": source_kind,
         "count": count, "point": point, "bbox": bbox}
        for unit_id, source_kind, count, point, bbox in base.troop_cards))
    plan = freeze_mixed_probe_line(Session(source, []), source, terrain=list(TERRAIN),
                                   toolbar_before=before)
    return plan, source


def test_mixed_plan_preserves_exact_two_edges_and_remainder(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    assert plan.anchors == ((375, 250), (525, 450))
    assert [row[0] for row in plan.checkpoint_cards] == ["electro_dragon", "furnace"]
    assert {action.unit_id for action in plan.remaining if action.kind == "deploy"} == \
           {"electro_dragon", "furnace", *HEROES}
    assert len([action for action in plan.remaining if action.kind == "deploy"]) == 15
    assert {action.point for action in plan.remaining if action.kind == "deploy"} == set(plan.anchors)
    assert remainder_burst_limit(plan) <= 10
    with pytest.raises(FlowError, match="permission"):
        execute_mixed_probe_line(Session(source, []), replace(plan, dragon_point=(1, 2)),
                                 clock=lambda: 10.)


@pytest.mark.parametrize("failure", ["duplicate_number", "wrong_number", "conflict",
                                      "wrong_version", "open_selected_border"])
def test_mixed_card_count_requires_exact_current_evidence(monkeypatch, tmp_path, failure):
    import numpy as np

    plan, source = make_plan(monkeypatch, tmp_path)
    current = snapshot(tmp_path, "number-check", "battle", 10.)
    session = Session(source, [])
    reading = {"count": 1, "confidence": .99, "frame": str(current.screenshot_path),
               "slot_bbox": list(plan.dragon_bbox), "portrait_score": .99,
               "portrait_source": "current_frame_saved_battle_portrait",
               "readings": [{"text": "x1", "confidence": .99}]}
    if failure == "duplicate_number":
        reading["readings"].append({"text": "x2", "confidence": .99})
    elif failure == "wrong_number":
        reading["count"] = 2
    elif failure == "conflict":
        reading["conflict"] = True
    monkeypatch.setattr("autococ.mixed_probe._post_troop_reading", lambda *_a, **_kw: reading)
    monkeypatch.setattr("autococ.images.read_frame",
                        lambda _p: np.zeros((1440, 2560, 3), dtype=np.uint8))
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                        lambda *_a, **_kw: {"unit_id": "balloon" if failure == "wrong_version" else "dragon",
                                                "version_verified": failure != "wrong_version",
                                                "template_client_version": "18.647.2",
                                                "confidence": .99})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: None)
    with pytest.raises(FlowError):
        _probe_count(session, plan, current, "dragon", 1, selected=True)


def test_real_anonymous_furnace_toolbar_proves_unspent_checkpoint_without_new_ocr(
        monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    metadata = json.loads((ROOT / "tests/fixtures/furnace_battle_identity_18_647_2.json")
                          .read_text(encoding="utf-8"))
    row = metadata["frames"][1]
    path = ROOT / row["path"]
    assert _file_sha256(path) == row["sha256"]
    recognizer = ScreenshotRecognizer(provider=None)
    recognizer.client_version = "18.647.2"
    texts = [OCRText(item["text"], item["confidence"], tuple(item["bbox"]))
             for item in row["cached_ocr"]]
    battle = recognizer._battle_observation(path, texts, row["scene"])
    current = SceneSnapshot("battle", .99, path, {"battle": battle})
    troops = [item for item in battle["slots"] if item["kind"] == "troop"]
    heroes = [item for item in battle["slots"] if item["kind"] == "hero"]
    assert {item["unit_id"]: item["count"] for item in troops} == dict(TROOPS)
    assert {item["unit_id"] for item in heroes} == set(HEROES)
    base = replace(plan.base,
                   troop_cards=tuple((item["unit_id"], item["source"], item["count"],
                                      tuple(item["point"]), tuple(item["bbox"])) for item in troops),
                   hero_cards=tuple((item["unit_id"], tuple(item["bbox"])) for item in heroes))
    ordinary = tuple((item["unit_id"], tuple(item["point"]), tuple(item["bbox"]))
                     for item in troops if item["unit_id"] in {"electro_dragon", "furnace"})
    plan = replace(plan, base=base, checkpoint_cards=ordinary)
    monkeypatch.setattr("autococ.mixed_probe._post_troop_reading", lambda *_a, **_kw: {})
    result = _remaining_checkpoint(Session(source, []), plan, current)
    assert [item["unit_id"] for item in result["troops"]] == ["electro_dragon", "furnace"]
    assert {item["unit_id"] for item in result["heroes"]} == set(HEROES)
    assert all(item["count_source"] == "full_card" for item in result["troops"])


def test_mixed_two_independent_selections_and_consumptions(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected_balloon = snapshot(tmp_path, "balloon-selected", "enemy_village", 10.1, fast="troop_count")
    spent_balloon = snapshot(tmp_path, "balloon-spent", "battle", 10.2, fast="battle_hud")
    selected_dragon = snapshot(tmp_path, "dragon-selected", "battle", 10.3, fast="troop_count")
    spent_dragon = snapshot(tmp_path, "dragon-spent", "battle", 10.4, fast="battle_hud")
    final = snapshot(tmp_path, "full-final", "battle", 10.5)
    session = Session(source, [selected_balloon, spent_balloon, selected_dragon, spent_dragon, final])
    monkeypatch.setattr("autococ.mixed_probe._current_frame",
                        lambda _s, _p, frame, **_kw: proof(frame))
    monkeypatch.setattr("autococ.mixed_probe._foreground_now", lambda _s: None)
    monkeypatch.setattr("autococ.mixed_probe._probe_count",
                        lambda _s, _p, _f, unit_id, expected, **_kw: {"count": expected,
                                                                         "unit_id": unit_id})
    monkeypatch.setattr("autococ.mixed_probe._remaining_checkpoint",
                        lambda _s, _p, frame: {"frame": str(frame.screenshot_path),
                                                "sha256": _file_sha256(frame.screenshot_path),
                                                "troops": ["electro_dragon", "furnace"],
                                                "heroes": list(HEROES)})

    def verified(_s, _base, receipt, _when, *, sleep):
        receipt["verified"] = receipt["completed"] = True
        return final

    monkeypatch.setattr("autococ.mixed_probe._verify_frozen_line_post", verified)
    assert execute_mixed_probe_line(session, plan, clock=lambda: 10., sleep=lambda _s: None) is final
    receipt = session.frozen_line_receipt
    assert receipt["probe_events"][0]["before"]["frame"] == str(selected_balloon.screenshot_path)
    assert receipt["probe_events"][1]["before"]["frame"] == str(selected_dragon.screenshot_path)
    assert [event["unit_id"] for event in receipt["probe_events"]] == ["balloon", "dragon"]
    assert [event["after_count"] for event in receipt["probe_events"]] == [0, 0]
    assert [point for kind, point in session.inputs[:4]] == [
        plan.balloon_point, plan.anchors[0], plan.dragon_point, plan.anchors[1]]
    assert receipt["issued_placements"] == receipt["attempted_placements"] == 17
    assert len(session.inputs) == receipt["input_count"]


def test_first_post_strong_count_two_stops_before_new_frame_or_second_card(monkeypatch, tmp_path):
    from autococ import mixed_probe

    plan, source = make_plan(monkeypatch, tmp_path)
    selected = snapshot(tmp_path, "first-selected", "enemy_village", 10.1, fast="troop_count")
    wrong = snapshot(tmp_path, "first-wrong-x2", "battle", 10.2, fast="battle_hud")
    later_zero = snapshot(tmp_path, "later-x0", "battle", 10.3, fast="battle_hud")
    session = Session(source, [selected, wrong, later_zero])
    monkeypatch.setattr("autococ.mixed_probe._current_frame",
                        lambda _s, _p, frame, **_kw: proof(frame))
    monkeypatch.setattr("autococ.mixed_probe._foreground_now", lambda _s: None)
    actual_count = mixed_probe._probe_count

    def count(active, frozen, frame, unit_id, expected, **kwargs):
        if frame is not wrong:
            return {"count": expected}
        monkeypatch.setattr("autococ.mixed_probe._post_troop_reading",
                            lambda *_a, **_kw: {"count": 2})
        return actual_count(active, frozen, frame, unit_id, expected, **kwargs)

    monkeypatch.setattr("autococ.mixed_probe._probe_count", count)
    with pytest.raises(DeploymentError, match="differs from frozen"):
        execute_mixed_probe_line(session, plan, clock=lambda: 10., sleep=lambda _s: None)
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]
    assert session.last_snapshot is wrong
    assert session.frozen_line_receipt["probe_events"][0]["status"] == "input_sent"


@pytest.mark.parametrize("expired_at,elapsed,expected_inputs", [
    ("second-selected", 13.1, 3), ("first-post", 16.1, 2),
    ("final", 22.1, 4)])
def test_mixed_expired_proof_never_authorizes_another_input(monkeypatch, tmp_path,
                                                             expired_at, elapsed, expected_inputs):
    plan, source = make_plan(monkeypatch, tmp_path)
    frames = [snapshot(tmp_path, "timed-selected-a", "enemy_village", 10.1, fast="troop_count"),
              snapshot(tmp_path, "timed-spent-a", "battle", 10.2, fast="battle_hud"),
              snapshot(tmp_path, "timed-selected-b", "battle", 10.3, fast="troop_count"),
              snapshot(tmp_path, "timed-spent-b", "battle", 10.4, fast="battle_hud"),
              snapshot(tmp_path, "timed-final", "battle", 10.5)]
    session = Session(source, frames)
    now = {"value": 10.}
    original = session.observe

    def observe(label, **kwargs):
        frame = original(label, **kwargs)
        if ((expired_at == "second-selected" and label.startswith("mixed-dragon-selected")) or
                (expired_at == "first-post" and label.startswith("mixed-balloon-post")) or
                (expired_at == "final" and label == "mixed-final-checkpoint")):
            now["value"] = elapsed
        return frame

    session.observe = observe
    monkeypatch.setattr("autococ.mixed_probe._current_frame",
                        lambda _s, _p, frame, **_kw: proof(frame))
    monkeypatch.setattr("autococ.mixed_probe._foreground_now", lambda _s: None)
    monkeypatch.setattr("autococ.mixed_probe._probe_count",
                        lambda _s, _p, _f, _unit_id, expected, **_kw: {"count": expected})
    monkeypatch.setattr("autococ.mixed_probe._remaining_checkpoint",
                        lambda _s, _p, frame: {"frame": str(frame.screenshot_path),
                                                "sha256": _file_sha256(frame.screenshot_path)})
    with pytest.raises(DeploymentError):
        execute_mixed_probe_line(session, plan, clock=lambda: now["value"], sleep=lambda _s: None)
    assert len(session.inputs) == expected_inputs
    assert not any(kind == "frozen" for kind, _point in session.inputs)


@pytest.mark.parametrize("failure,expected_inputs", [
    ("second_selection_bad", 3), ("second_consumption_bad", 4),
    ("final_roster_bad", 4), ("second_selection_uncertain", 3)])
def test_mixed_failures_do_not_repeat_input_or_start_remainder(monkeypatch, tmp_path,
                                                                failure, expected_inputs):
    plan, source = make_plan(monkeypatch, tmp_path)
    frames = [snapshot(tmp_path, "selected-a", "enemy_village", 10.1, fast="troop_count"),
              snapshot(tmp_path, "spent-a", "battle", 10.2, fast="battle_hud"),
              snapshot(tmp_path, "selected-b", "battle", 10.3, fast="troop_count"),
              snapshot(tmp_path, "spent-b", "battle", 10.4, fast="battle_hud"),
              snapshot(tmp_path, "final-b", "battle", 10.5)]
    session = Session(source, frames)
    monkeypatch.setattr("autococ.mixed_probe._current_frame",
                        lambda _s, _p, frame, **_kw: proof(frame))
    monkeypatch.setattr("autococ.mixed_probe._foreground_now", lambda _s: None)

    def count(_session, _plan, frame, unit_id, expected, **_kw):
        if failure == "second_selection_bad" and frame is frames[2] or \
                failure == "second_consumption_bad" and frame is frames[3]:
            raise FlowError("Mixed strong_count_conflict")
        return {"count": expected}

    monkeypatch.setattr("autococ.mixed_probe._probe_count", count)
    monkeypatch.setattr("autococ.mixed_probe._remaining_checkpoint",
                        lambda *_a: (_ for _ in ()).throw(FlowError("roster unavailable"))
                        if failure == "final_roster_bad" else {"sha256": _file_sha256(frames[-1].screenshot_path)})
    if failure == "second_selection_uncertain":
        original = session.tap

        def uncertain(frame, point, *, reason):
            original(frame, point, reason=reason)
            if len(session.inputs) == 3:
                raise TimeoutError("transport uncertain")

        session.tap = uncertain
    with pytest.raises(DeploymentError):
        execute_mixed_probe_line(session, plan, clock=lambda: 10., sleep=lambda _s: None)
    assert len(session.inputs) == expected_inputs
    assert not any(kind == "frozen" for kind, _point in session.inputs)
