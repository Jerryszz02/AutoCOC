"""Offline mixed probe contracts; these do not establish live battle readiness."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import cv2
import numpy as np

from autococ.battle_vision import selected_card_bbox
from autococ.errors import DeploymentError, FlowError, StopRequested
from autococ.frozen_line import FrozenLinePlan, _file_sha256, _plan_digest
from autococ.mixed_probe import (execute_mixed_probe_line, freeze_mixed_probe_line,
                                 ProbeEvidenceConflict, ProbeSelectionPending,
                                 _probe_count, _remaining_checkpoint,
                                 remainder_burst_limit)
from autococ.ocr import OCRText
from autococ.probed_anchor import ExactCameraError
from autococ.prepared_battle import FramePacket, PreparedInput
from autococ.scene import SceneSnapshot
from autococ.strategies import BattleContext, TwoEdgeFurnaceStrategy
from autococ.vision import ScreenshotRecognizer


TERRAIN = ({"edge": 0, "point": (400, 200)}, {"edge": 0, "point": (350, 300)},
           {"edge": 1, "point": (500, 400)}, {"edge": 1, "point": (550, 500)})
TROOPS = (("electro_dragon", 10), ("dragon", 1), ("furnace", 1), ("balloon", 1))
HEROES = ("grand_warden", "archer_queen", "dragon_duke", "minion_prince")
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\projects\AutoCOC")
SELECTION_METADATA = ROOT / "tests/fixtures/mixed_balloon_selection_18_647_2_20261016.json"


def _selection_rows():
    return json.loads(SELECTION_METADATA.read_text(encoding="utf-8"))["frames"]


def _retained_ocr(items, rectangles):
    return [item for item in items if item.get("bbox") and
            any(left <= item["bbox"][0] < item["bbox"][2] <= right and
                top <= item["bbox"][1] < item["bbox"][3] <= bottom
                for left, top, right, bottom in rectangles)]


def snapshot(tmp_path, name, scene, observed, *, fast=None):
    path = tmp_path / f"{name}.png"
    path.write_bytes(f"immutable {name}".encode())
    observations = {"observed_at_monotonic": observed, "source_resolution": (2560, 1440),
                    "baseline_resolution": (1280, 720), "battle": {"slots": []}}
    if fast:
        observations.update(recognition_scope=fast, fast_fallback=False,
                            matched_reasons=["two_current_hud_anchors"])
        if fast == "mixed_selection_wait":
            observations["deployment_count_reads"] = [{"count": 1}]
    return SceneSnapshot(scene, .99, path, observations)


def proof(frame):
    return {"frame": str(frame.screenshot_path), "sha256": _file_sha256(frame.screenshot_path),
            "observed_at": frame.observations["observed_at_monotonic"],
            "camera": {"stationary": True}}


def base_plan(tmp_path, *, archers=0):
    source = snapshot(tmp_path, "source", "enemy_village", 9.)
    before = snapshot(tmp_path, "before", "enemy_village", 8.)
    troop_counts = TROOPS + (("archer", archers),) if archers else TROOPS
    troop_cards = tuple((unit_id, "army", count, (150 + i * 90, 650),
                         (110 + i * 90, 595, 190 + i * 90, 711))
                        for i, (unit_id, count) in enumerate(troop_counts))
    hero_cards = tuple((unit_id, (600 + i * 90, 595, 680 + i * 90, 711))
                       for i, unit_id in enumerate(HEROES))
    troops = tuple({"unit_id": unit_id, "count": count, "source": "army", "point": card[3]}
                   for card, (unit_id, count) in zip(troop_cards, troop_counts))
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
                          tuple((name, "army", count) for name, count in troop_counts), troop_cards,
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


def make_plan(monkeypatch, tmp_path, *, archers=0):
    base, source, before = base_plan(tmp_path, archers=archers)
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


@pytest.mark.parametrize("archers", [1, 3, 9])
def test_mixed_filler_keeps_both_probes_and_places_archers_last(monkeypatch, tmp_path, archers):
    plan, _source = make_plan(monkeypatch, tmp_path, archers=archers)
    assert plan.anchors == ((375, 250), (525, 450))
    assert [row[0] for row in plan.checkpoint_cards] == ["electro_dragon", "furnace", "archer"]
    deployments = [action for action in plan.remaining if action.kind == "deploy"]
    assert [action.unit_id for action in deployments[-4:]] == list(HEROES)
    assert [action.unit_id for action in deployments[-4 - archers:-4]] == ["archer"] * archers
    assert len(deployments) == 15 + archers
    assert sum(action.point == plan.anchors[0] for action in deployments if action.unit_id == "archer") == \
           (archers + 1) // 2
    assert sum(action.point == plan.anchors[1] for action in deployments if action.unit_id == "archer") == \
           archers // 2
    assert remainder_burst_limit(plan) <= 10


@pytest.mark.parametrize("fault", [None, "missing_archer", "archer_count_conflict", "archer_identity",
                                    "duplicate_checkpoint"])
def test_mixed_filler_checkpoint_requires_third_independent_card(monkeypatch, tmp_path, fault):
    # A synthetic plumbing check; real five-card toolbar and x0 OCR are not yet sampled.
    plan, source = make_plan(monkeypatch, tmp_path, archers=3)
    base = replace(plan.base, hero_cards=())
    plan = replace(plan, base=base)
    cards = {unit_id: (point, bbox, count) for unit_id, _source, count, point, bbox
             in base.troop_cards}
    current = snapshot(tmp_path, "synthetic-five-card-checkpoint", "battle", 10.5)
    current.observations["battle"]["slots"] = [
        {"unit_id": unit_id, "source": "army", "bbox": list(cards[unit_id][1]),
         "count": cards[unit_id][2], "confidence": .99,
         "evidence": {"identity": {"unit_id": unit_id, "version_verified": True,
                                   "template_client_version": "18.647.2"},
                      "count": {"text": f"x{cards[unit_id][2]}"}}}
        for unit_id in ("electro_dragon", "furnace", "archer")]
    if fault == "missing_archer":
        current.observations["battle"]["slots"].pop()
    elif fault == "archer_count_conflict":
        current.observations["battle"]["slots"][-1]["count"] = 2
    elif fault == "duplicate_checkpoint":
        plan = replace(plan, checkpoint_cards=plan.checkpoint_cards[:-1] +
                       (plan.checkpoint_cards[0],))
    # Use real synthetic pixels instead of replacing the shared image loader:
    # a lazy hero_state import could otherwise retain the temporary function.
    cv2.imencode(".png", np.zeros((1440, 2560, 3), dtype=np.uint8))[1].tofile(
        current.screenshot_path)
    monkeypatch.setattr("autococ.mixed_probe._current_card_edges", lambda *_a: {"closed": True})
    monkeypatch.setattr("autococ.mixed_probe._count_proves",
                        lambda reading, *_a: bool(reading.get("text")))
    monkeypatch.setattr("autococ.mixed_probe._post_troop_reading", lambda *_a, **_kw: {})
    by_bbox = {bbox: unit_id for unit_id, (_point, bbox, _count) in cards.items()}
    monkeypatch.setattr("autococ.unit_catalog.recognize_card_identity",
                        lambda _image, bbox, *_a, **_kw: {
                            "unit_id": ("balloon" if fault == "archer_identity" and
                                        by_bbox[tuple(bbox)] == "archer" else by_bbox[tuple(bbox)]),
                            "version_verified": True, "confidence": .99})
    if fault is None:
        result = _remaining_checkpoint(Session(source, []), plan, current)
        assert [row["unit_id"] for row in result["troops"]] == ["electro_dragon", "furnace", "archer"]
    else:
        with pytest.raises(FlowError):
            _remaining_checkpoint(Session(source, []), plan, current)


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
    with pytest.raises(FlowError) as failure_error:
        _probe_count(session, plan, current, "dragon", 1, selected=True)
    assert isinstance(failure_error.value, ProbeSelectionPending) is \
           (failure == "open_selected_border")


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
    selected_balloon = snapshot(tmp_path, "balloon-selected", "enemy_village", 10.1, fast="mixed_selection_wait")
    spent_balloon = snapshot(tmp_path, "balloon-spent", "battle", 10.2, fast="battle_hud")
    selected_dragon = snapshot(tmp_path, "dragon-selected", "battle", 10.3, fast="mixed_selection_wait")
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


def _mixed_camera_session(monkeypatch, tmp_path, *, dragon_frames):
    plan, source = make_plan(monkeypatch, tmp_path)
    balloon_selected = snapshot(tmp_path, "balloon-selected", "enemy_village", 10.1,
                                fast="mixed_selection_wait")
    balloon_spent = snapshot(tmp_path, "balloon-spent", "battle", 10.2,
                             fast="battle_hud")
    dragon_selected = [snapshot(tmp_path, f"dragon-selected-{index}", "battle",
                                10.3 + index * .1, fast="mixed_selection_wait")
                       for index in range(dragon_frames)]
    dragon_spent = snapshot(tmp_path, "dragon-spent", "battle", 10.7,
                            fast="battle_hud")
    final = snapshot(tmp_path, "full-final", "battle", 10.8)
    session = Session(source, [balloon_selected, balloon_spent, *dragon_selected,
                               dragon_spent, final])
    session.capture.adb = object()
    session.capture.serial = "offline"
    session.config.game.package_name = "game"
    monkeypatch.setattr("autococ.device.foreground_package", lambda *_a, **_kw: "game")
    monkeypatch.setattr("autococ.mixed_probe._remaining_checkpoint",
                        lambda _s, _p, frame: {"frame": str(frame.screenshot_path),
                                               "sha256": _file_sha256(frame.screenshot_path)})
    monkeypatch.setattr("autococ.mixed_probe._verify_frozen_line_post",
                        lambda _s, _p, receipt, _when, *, sleep: final)
    return plan, source, session, dragon_selected, final


def test_mixed_dragon_transient_exact_camera_requires_new_full_proof(monkeypatch, tmp_path):
    plan, source, session, selected, final = _mixed_camera_session(
        monkeypatch, tmp_path, dragon_frames=2)
    camera_calls = []

    def camera(_source, current, _anchors):
        camera_calls.append(current)
        if current == selected[0].screenshot_path:
            raise ExactCameraError("Probe camera has insufficient stationary map inliers")
        return {"matches": 100, "inliers": 100, "residual_p95": 0.,
                "anchor_offsets": [0., 0.]}

    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera", camera)
    count_calls = []

    def count(_s, _p, frame, unit_id, expected, **_kw):
        count_calls.append((frame, unit_id, expected))
        return {"count": expected}

    monkeypatch.setattr("autococ.mixed_probe._probe_count", count)
    assert execute_mixed_probe_line(session, plan, clock=lambda: 10.,
                                    sleep=lambda _s: None) is final
    receipt = session.frozen_line_receipt
    dragon = next(action for action in receipt["actions"]
                  if action["action_id"] == "select-probe-dragon")
    assert dragon["camera_rejections"] == [{
        "frame": str(selected[0].screenshot_path),
        "sha256": _file_sha256(selected[0].screenshot_path),
        "message": "Probe camera has insufficient stationary map inliers"}]
    assert receipt["probe_events"][1]["before"]["frame"] == str(selected[1].screenshot_path)
    assert selected[0].screenshot_path in camera_calls
    assert selected[1].screenshot_path in camera_calls
    assert not any(frame is selected[0] for frame, _unit, _count in count_calls)
    assert [point for kind, point in session.inputs[:4]] == [
        plan.balloon_point, plan.anchors[0], plan.dragon_point, plan.anchors[1]]
    assert session.inputs.count(("tap", plan.dragon_point)) == 1


def test_mixed_dragon_three_camera_rejections_exhaust_existing_observations(
        monkeypatch, tmp_path):
    plan, _source, session, selected, _final = _mixed_camera_session(
        monkeypatch, tmp_path, dragon_frames=3)

    def camera(_source, current, _anchors):
        if current in {frame.screenshot_path for frame in selected}:
            raise ExactCameraError("Probe camera has insufficient stationary map inliers")
        return {"matches": 100, "inliers": 100, "residual_p95": 0.,
                "anchor_offsets": [0., 0.]}

    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera", camera)
    monkeypatch.setattr("autococ.mixed_probe._probe_count",
                        lambda _s, _p, _f, _unit, expected, **_kw: {"count": expected})
    with pytest.raises(DeploymentError, match="no fresh selected-card proof"):
        execute_mixed_probe_line(session, plan, clock=lambda: 10., sleep=lambda _s: None)
    dragon = next(action for action in session.frozen_line_receipt["actions"]
                  if action["action_id"] == "select-probe-dragon")
    assert [row["frame"] for row in dragon["camera_rejections"]] == [
        str(frame.screenshot_path) for frame in selected]
    assert session.last_snapshot is selected[-1]
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0]),
                              ("tap", plan.dragon_point)]


@pytest.mark.parametrize("fast_count", [0, 2, "malformed_pending"])
def test_mixed_dragon_camera_rejection_cannot_hide_same_frame_fast_conflict(
        monkeypatch, tmp_path, fast_count):
    plan, _source, session, selected, _final = _mixed_camera_session(
        monkeypatch, tmp_path, dragon_frames=2)
    if fast_count == "malformed_pending":
        selected[0].observations["deployment_count_reads"] = []
        selected[0].observations["deployment_count_pending"] = {
            "frame": "stale-frame.png", "slot_bbox": list(plan.dragon_bbox),
            "purpose": "mixed_selection_wait", "reason": "current_frame_count_unavailable"}
    else:
        selected[0].observations["deployment_count_reads"][0]["count"] = fast_count
    camera_calls = []

    def camera(_source, current, _anchors):
        camera_calls.append(current)
        if current == selected[0].screenshot_path:
            raise ExactCameraError("Probe camera has insufficient stationary map inliers")
        return {"matches": 100, "inliers": 100, "residual_p95": 0.,
                "anchor_offsets": [0., 0.]}

    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera", camera)
    monkeypatch.setattr("autococ.mixed_probe._probe_count",
                        lambda _s, _p, _f, _unit, expected, **_kw: {"count": expected})
    with pytest.raises(DeploymentError, match=(
            "malformed count-pending" if fast_count == "malformed_pending"
            else "fast count conflicts")):
        execute_mixed_probe_line(session, plan, clock=lambda: 10., sleep=lambda _s: None)
    assert selected[0].screenshot_path not in camera_calls
    assert session.last_snapshot is selected[0]
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0]),
                              ("tap", plan.dragon_point)]
    dragon = next(action for action in session.frozen_line_receipt["actions"]
                  if action["action_id"] == "select-probe-dragon")
    assert "camera_rejections" not in dragon


@pytest.mark.parametrize("fault", ["expired", "stop", "account", "source_changed",
                                    "version_changed", "display_changed", "reused_frame",
                                    "stale_observed", "strong_count_conflict"])
def test_mixed_dragon_camera_wait_does_not_swallow_stronger_failures(
        monkeypatch, tmp_path, fault):
    plan, source, session, selected, _final = _mixed_camera_session(
        monkeypatch, tmp_path, dragon_frames=2)
    now = {"value": 10.}
    original_observe = session.observe

    def observe(label, **kwargs):
        if fault == "stop" and label == "mixed-dragon-selected-1":
            raise StopRequested("offline stop")
        frame = original_observe(label, **kwargs)
        if frame is selected[0] and fault == "expired":
            now["value"] = 13.1
        if frame is selected[1] and fault == "source_changed":
            source.screenshot_path.write_bytes(b"changed frozen source")
        if frame is selected[1] and fault == "version_changed":
            session.client_version = "18.600.7"
        if frame is selected[1] and fault == "display_changed":
            session.capture.input_display_id = 1
        if frame is selected[1] and fault == "account":
            frame = replace(frame, scene="account_in_use")
            session.last_snapshot = frame
        if frame is selected[1] and fault == "reused_frame":
            frame = selected[0]
            session.last_snapshot = frame
        if frame is selected[1] and fault == "stale_observed":
            frame.observations["observed_at_monotonic"] = selected[0].observations[
                "observed_at_monotonic"]
        return frame

    session.observe = observe

    def camera(_source, current, _anchors):
        if current == selected[0].screenshot_path:
            raise ExactCameraError("Probe camera has insufficient stationary map inliers")
        return {"matches": 100, "inliers": 100, "residual_p95": 0.,
                "anchor_offsets": [0., 0.]}

    monkeypatch.setattr("autococ.probed_anchor.verify_exact_anchor_camera", camera)

    def count(_s, _p, frame, unit_id, expected, **_kw):
        if fault == "strong_count_conflict" and unit_id == "dragon":
            raise ProbeEvidenceConflict("Mixed dragon strong_count_conflict")
        return {"count": expected}

    monkeypatch.setattr("autococ.mixed_probe._probe_count", count)
    expected = StopRequested if fault == "stop" else DeploymentError
    with pytest.raises(expected):
        execute_mixed_probe_line(session, plan, clock=lambda: now["value"],
                                 sleep=lambda _s: None)
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0]),
                              ("tap", plan.dragon_point)]
    dragon = next(action for action in session.frozen_line_receipt["actions"]
                  if action["action_id"] == "select-probe-dragon")
    assert len(dragon["camera_rejections"]) == 1
    assert not any(event["unit_id"] == "dragon"
                   for event in session.frozen_line_receipt["probe_events"])


def test_first_post_strong_count_two_stops_before_new_frame_or_second_card(monkeypatch, tmp_path):
    from autococ import mixed_probe

    plan, source = make_plan(monkeypatch, tmp_path)
    selected = snapshot(tmp_path, "first-selected", "enemy_village", 10.1, fast="mixed_selection_wait")
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
    frames = [snapshot(tmp_path, "timed-selected-a", "enemy_village", 10.1, fast="mixed_selection_wait"),
              snapshot(tmp_path, "timed-spent-a", "battle", 10.2, fast="battle_hud"),
              snapshot(tmp_path, "timed-selected-b", "battle", 10.3, fast="mixed_selection_wait"),
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
    frames = [snapshot(tmp_path, "selected-a", "enemy_village", 10.1, fast="mixed_selection_wait"),
              snapshot(tmp_path, "spent-a", "battle", 10.2, fast="battle_hud"),
              snapshot(tmp_path, "selected-b", "battle", 10.3, fast="mixed_selection_wait"),
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


def test_live_selected_transition_fixtures_are_anonymous_and_source_bound():
    metadata = json.loads(SELECTION_METADATA.read_text(encoding="utf-8"))
    assert metadata["client_version"] == "18.647.2"
    assert metadata["human_reviewed"] is False
    assert metadata["counts_toward_acceptance"] is False
    assert metadata["cached_ocr_filter"] == (
        "Retain only OCR items whose entire xyxy bbox lies inside one of the "
        "frame's retained_baseline_rectangles_xyxy; preserve original order and values.")
    assert [Path(row["source_frame"]).name[:5] for row in metadata["frames"]] == [
        "00015", "00016"]
    for row in metadata["frames"]:
        path = ROOT / row["path"]
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (720, 1280, 3)
        assert _file_sha256(path) == row["sha256"]
        mask = np.zeros(image.shape[:2], dtype=bool)
        for left, top, right, bottom in row["retained_baseline_rectangles_xyxy"]:
            mask[top:bottom, left:right] = True
        assert np.all(image[~mask] == 45)
        assert row["cached_ocr"] == _retained_ocr(
            row["cached_ocr"], row["retained_baseline_rectangles_xyxy"])
        source = SOURCE_ROOT / row["source_frame"]
        if source.is_file():
            assert _file_sha256(source) == row["source_sha256"]
            original = cv2.resize(cv2.imdecode(np.fromfile(source, dtype=np.uint8),
                                               cv2.IMREAD_COLOR), (1280, 720),
                                  interpolation=cv2.INTER_AREA)
            assert np.array_equal(image[mask], original[mask])
        events = SOURCE_ROOT / row["source_events"]
        if events.is_file():
            assert _file_sha256(events) == row["source_events_sha256"]
            records = [event for line in events.read_text(encoding="utf-8").splitlines()
                       if (event := json.loads(line)).get("kind") == "observation"
                       and Path(str(event.get("frame"))) == source]
            assert len(records) == 1
            observation = records[0]["observations"]
            assert row["cached_ocr"] == _retained_ocr(
                observation["ocr"], row["retained_baseline_rectangles_xyxy"])
            if "cached_fast_count" in row:
                assert len(observation["deployment_count_reads"]) == 1
                assert row["cached_fast_count"] == {
                    key: value for key, value in observation["deployment_count_reads"][0].items()
                    if key != "frame"}


def test_real_pixels_and_cached_count_prove_pending_then_selected(monkeypatch, tmp_path):
    from autococ.images import read_frame
    from autococ.unit_catalog import recognize_card_identity

    plan, source = make_plan(monkeypatch, tmp_path)
    first, second = _selection_rows()
    first_path, second_path = ROOT / first["path"], ROOT / second["path"]
    bbox = (383, 595, 472, 711)
    image = cv2.resize(read_frame(first_path), (1280, 720), interpolation=cv2.INTER_AREA)
    identity = recognize_card_identity(image, bbox, "troop", surface="battle",
                                       client_version="18.647.2")
    assert identity["unit_id"] == "balloon"
    assert identity["version_verified"] is True
    assert selected_card_bbox(first_path, bbox, (1280, 720)) is None
    assert selected_card_bbox(second_path, bbox, (1280, 720)) == (380, 589, 475, 709)
    assert first["cached_fast_count"]["count"] == 1
    assert first["cached_fast_count"]["portrait_score"] == 1.0
    full_count = [item for item in second["cached_ocr"] if item["text"] == "x1" and
                  380 <= item["bbox"][0] < 475]
    assert len(full_count) == 1 and full_count[0]["confidence"] >= .9

    base = replace(plan.base, frame=replace(plan.base.frame,
                                            frame_id=str(first_path),
                                            screenshot_path=first_path),
                   frame_sha256=_file_sha256(first_path),
                   toolbar_before_frame_id=str(first_path),
                   toolbar_before_sha256=_file_sha256(first_path))
    plan = replace(plan, base=base, balloon_bbox=bbox, balloon_point=(427, 653))
    # The OCR provider only replays each frame's recorded x1; actual source
    # and current portrait matching, source identity and outline stay live code.
    active_confidence = {"value": first["cached_fast_count"]["confidence"]}
    provider = SimpleNamespace(recognize_line_image=lambda _crop: [
        OCRText("x1", active_confidence["value"], (0, 0, 1, 1))])
    recognizer = ScreenshotRecognizer(provider=provider)
    recognizer.client_version = "18.647.2"
    session = Session(source, [])
    session.recognizer = recognizer
    first_read = {**first["cached_fast_count"], "frame": str(first_path)}
    first_snapshot = SceneSnapshot("enemy_village", .99, first_path, {
        "recognition_scope": "troop_count", "fast_fallback": False,
        "matched_reasons": ["two_current_hud_anchors"],
        "deployment_count_reads": [first_read], "battle": {"slots": []},
        "ocr": first["cached_ocr"]})
    session.last_snapshot = first_snapshot
    with pytest.raises(ProbeSelectionPending, match="closed card border"):
        _probe_count(session, plan, first_snapshot, "balloon", 1, selected=True)
    active_confidence["value"] = full_count[0]["confidence"]
    second_snapshot = SceneSnapshot("enemy_village", .99, second_path, {
        "recognition_scope": "full", "fast_fallback": True,
        "recognition_requested": "troop_count", "battle": {"slots": []},
        "ocr": second["cached_ocr"]})
    session.last_snapshot = second_snapshot
    reading = _probe_count(session, plan, second_snapshot, "balloon", 1, selected=True)
    assert reading["count"] == 1
    assert reading["portrait_score"] >= .92
    assert reading["portrait_source"] == "current_frame_saved_battle_portrait"
    assert reading["source_identity"]["unit_id"] == "balloon"
    assert all(reading["roi"][index] <= full_count[0]["bbox"][index] for index in (0, 1))
    assert all(full_count[0]["bbox"][index] <= reading["roi"][index] for index in (2, 3))
    assert reading["selected_card_border"] == [380, 589, 475, 709]


@pytest.mark.parametrize("outcome,selected_frames,expected_inputs", [
    ("resolved", 2, 2), ("always_pending", 3, 1),
    ("strong_conflict", 2, 1), ("source_conflict", 2, 1),
    ("camera_conflict", 2, 1), ("expired", 1, 1)])
def test_pending_selection_only_reobserves_within_existing_budget(
        monkeypatch, tmp_path, outcome, selected_frames, expected_inputs):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = [snapshot(tmp_path, f"selected-{index}", "enemy_village",
                         10.1 + index * .1, fast="mixed_selection_wait")
                for index in range(selected_frames)]
    posts = [snapshot(tmp_path, f"post-{index}", "battle", 10.5 + index * .1,
                      fast="battle_hud") for index in range(2)] if outcome == "resolved" else []
    session = Session(source, selected + posts)
    now = {"value": 10.}
    observe = session.observe

    def timed_observe(label, **kwargs):
        frame = observe(label, **kwargs)
        if outcome == "expired" and frame is selected[0]:
            now["value"] = 13.1
        return frame

    session.observe = timed_observe

    def current(_session, _plan, frame, **_kwargs):
        if outcome == "camera_conflict" and frame is selected[1]:
            raise FlowError("camera proof changed")
        return proof(frame)

    monkeypatch.setattr("autococ.mixed_probe._current_frame", current)
    monkeypatch.setattr("autococ.mixed_probe._foreground_now", lambda _s: None)

    def count(_session, _plan, frame, unit_id, expected, **kwargs):
        assert unit_id == "balloon"
        if frame in selected:
            index = selected.index(frame)
            if index == 0 or outcome == "always_pending":
                raise ProbeSelectionPending("closed card border pending")
            if outcome == "strong_conflict":
                raise ProbeEvidenceConflict("strong current count conflict")
            if outcome == "source_conflict":
                raise FlowError("source identity changed")
            return {"count": 1}
        assert frame in posts and expected == 0
        return {"count": 1}  # Never claims an unproved 1→0 placement.

    monkeypatch.setattr("autococ.mixed_probe._probe_count", count)
    with pytest.raises(DeploymentError):
        execute_mixed_probe_line(session, plan, clock=lambda: now["value"],
                                 sleep=lambda _s: None)
    assert len(session.inputs) == expected_inputs
    assert session.inputs[0] == ("tap", plan.balloon_point)
    assert session.inputs.count(("tap", plan.balloon_point)) == 1
    assert not any(kind == "frozen" for kind, _point in session.inputs)
    assert session.frozen_line_receipt["actions"][0]["pending_frames"][0] == {
        "frame": str(selected[0].screenshot_path),
        "sha256": _file_sha256(selected[0].screenshot_path),
        "status": "count_proven_border_pending",
        "message": "closed card border pending"}
    if outcome == "resolved":
        assert session.inputs[1] == ("tap", plan.anchors[0])
        assert session.last_snapshot is posts[-1]
    elif outcome == "always_pending":
        assert session.last_snapshot is selected[-1]
        assert len(session.frozen_line_receipt["actions"][0]["pending_frames"]) == 3
    elif outcome == "expired":
        assert session.last_snapshot is selected[0]
    else:
        assert session.last_snapshot is selected[1]


@pytest.mark.parametrize("outcome,expected_inputs", [
    ("pending_then_valid", 2), ("always_pending", 1), ("malformed_pending", 1),
    ("strong_wrong_count", 1), ("guard_failed", 1), ("camera_changed", 1),
    ("expired", 1), ("wrong_purpose", 1)])
def test_unproved_mixed_selection_never_reuses_a_count_or_repeats_select(
        monkeypatch, tmp_path, outcome, expected_inputs):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = [snapshot(tmp_path, f"unproved-{index}", "enemy_village",
                         10.1 + index * .1, fast="mixed_selection_wait")
                for index in range(3)]
    for frame in selected:
        frame.observations.pop("deployment_count_reads")
        frame.observations["deployment_count_pending"] = {
            "frame": str(frame.screenshot_path), "slot_bbox": list(plan.balloon_bbox),
            "purpose": "mixed_selection_wait", "reason": "current_frame_count_unavailable"}
    if outcome == "pending_then_valid":
        selected[1].observations.pop("deployment_count_pending")
        selected[1].observations["deployment_count_reads"] = [{"count": 1}]
    elif outcome == "malformed_pending":
        selected[0].observations["deployment_count_pending"]["frame"] = "stale.png"
    elif outcome == "strong_wrong_count":
        selected[0].observations.pop("deployment_count_pending")
        selected[0].observations["deployment_count_reads"] = [{"count": 2}]
    elif outcome == "guard_failed":
        selected[0].observations["matched_reasons"] = ["mixed_selection_guard_unverified"]
        selected[0] = replace(selected[0], scene="unknown")
    elif outcome == "wrong_purpose":
        selected[0].observations["recognition_scope"] = "troop_count"
    post = snapshot(tmp_path, "unproved-post", "battle", 10.5, fast="battle_hud")
    session = Session(source, selected + [post])
    now = {"value": 10.}
    requested_purposes = []
    original_observe = session.observe

    def observe(label, **kwargs):
        requested_purposes.append(kwargs.get("purpose"))
        frame = original_observe(label, **kwargs)
        if outcome == "expired":
            now["value"] = 13.1
        return frame

    session.observe = observe

    def current(_session, _plan, frame, **_kwargs):
        if outcome == "camera_changed" and frame is selected[0]:
            raise FlowError("camera proof changed")
        return proof(frame)

    monkeypatch.setattr("autococ.mixed_probe._current_frame", current)
    monkeypatch.setattr("autococ.mixed_probe._foreground_now", lambda _s: None)
    count_calls = []

    def count(_session, _plan, frame, _unit_id, expected, **_kwargs):
        count_calls.append(frame)
        if frame is selected[1]:
            return {"count": expected}
        raise FlowError("post proof unavailable")

    monkeypatch.setattr("autococ.mixed_probe._probe_count", count)
    with pytest.raises(DeploymentError):
        execute_mixed_probe_line(session, plan, clock=lambda: now["value"],
                                 sleep=lambda _s: None)
    assert len(session.inputs) == expected_inputs
    assert session.inputs.count(("tap", plan.balloon_point)) == 1
    assert not any(kind == "frozen" for kind, _point in session.inputs)
    assert requested_purposes[0] == "mixed_selection_wait"
    if outcome == "pending_then_valid":
        assert count_calls == [selected[1]]
        assert session.inputs[1] == ("tap", plan.anchors[0])
        assert session.frozen_line_receipt["actions"][0]["pending_frames"][0]["status"] == \
               "count_unproved_waiting"
    elif outcome == "always_pending":
        assert len(session.frozen_line_receipt["actions"][0]["pending_frames"]) == 3
        assert count_calls == []
    else:
        assert count_calls == []
