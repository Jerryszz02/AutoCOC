"""Synthetic safety contracts; these tests do not constitute live deployment proof."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from autococ.errors import DeploymentError, FlowError, StopRequested
from autococ.frozen_line import FrozenLinePlan, _file_sha256, _plan_digest
from autococ.prepared_battle import FramePacket, PreparedInput
from autococ.probed_anchor import (_balloon_count, _current_frame, _remaining_roster_checkpoint,
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


def image_frame(tmp_path, name, scene, observed):
    path = tmp_path / f"{name}.png"
    path.write_bytes(f"frame {name}".encode())
    return SceneSnapshot(scene, .99, path, {"observed_at_monotonic": observed,
        "source_resolution": (2560, 1440), "baseline_resolution": (1280, 720),
        "battle": {"slots": []}})


def frozen_base(tmp_path):
    source = image_frame(tmp_path, "source", "enemy_village", 9.)
    before = image_frame(tmp_path, "before", "enemy_village", 8.)
    troop_cards = tuple((unit_id, "army", count, (150 + index * 90, 650),
                         (110 + index * 90, 595, 190 + index * 90, 710))
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

    def observe(self, label):
        self.last_snapshot = None  # GameSession invalidates old frame before capture.
        next_frame = next(self.observations)
        if isinstance(next_frame, BaseException):
            raise next_frame
        self.last_snapshot = next_frame
        return next_frame


def make_plan(monkeypatch, tmp_path):
    base, source = frozen_base(tmp_path)
    monkeypatch.setattr("autococ.probed_anchor.freeze_model_free_line", lambda *_a, **_kw: base)
    monkeypatch.setattr("autococ.probed_anchor._foreground_now", lambda _session: None)
    monkeypatch.setattr("autococ.probed_anchor._remaining_roster_checkpoint",
                        lambda _session, _plan, snapshot: {"frame": str(snapshot.screenshot_path)})
    session = Session(source, [])
    plan = freeze_probed_anchor_line(session, source, terrain=list(TERRAIN))
    return plan, source


def test_frozen_plan_uses_only_two_precompiled_points_and_preserves_21_deployments(monkeypatch, tmp_path):
    plan, _ = make_plan(monkeypatch, tmp_path)
    assert plan.anchors == ((375, 250), (525, 450))
    assert len([a for a in plan.remaining if a.kind == "deploy"]) == 19
    assert {a.point for a in plan.remaining if a.kind == "deploy"} == set(plan.anchors)
    assert sum(row[2] for row in plan.base.troop_budgets) == 17
    assert [a.unit_id for a in plan.remaining if a.kind == "deploy"].count("balloon") == 0
    assert len(plan.base.actions) == 30  # Existing two-edge split order is unchanged.
    assert remainder_burst_limit(plan) == 7.75


def test_probed_plan_rejects_unsupported_baseline_before_freezing(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    session = Session(source, [])
    session.config.game.baseline_resolution = (1920, 1080)
    with pytest.raises(FlowError, match="1280"):
        freeze_probed_anchor_line(session, source, terrain=list(TERRAIN))


def test_two_probes_once_then_frozen_remainder_and_complete_receipt(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    session = Session(source, [selected, first, second])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, _frame, expected: {"count": expected})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def verify(_session, _base, receipt, _when, *, sleep):
        receipt["verified"] = receipt["completed"] = True
        receipt["deployed_units"] = 17
        return second
    monkeypatch.setattr("autococ.probed_anchor._verify_frozen_line_post", verify)
    assert execute_probed_anchor_line(session, plan, clock=lambda: 10., sleep=lambda _: None) is second
    receipt = session.frozen_line_receipt
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 21
    assert receipt["deployed_units"] == 17
    assert receipt["burst_limit_sec"] == remainder_burst_limit(plan)
    assert [item["after_count"] for item in receipt["probe_events"]] == [1, 0]
    assert receipt["probe_started_at"] <= receipt["probe_authorized_at"]
    assert receipt["probe_elapsed_sec"] < receipt["probe_limit_sec"]
    assert all(event["started_at"] <= event["input_done_at"] <=
               event["verified_at"] < event["deadline"] for event in receipt["probe_events"])
    assert [point for _, point in session.inputs if point in plan.anchors][:2] == list(plan.anchors)
    assert len(session.inputs) == receipt["input_count"]


def test_failed_probe_capture_has_partial_receipt_and_never_replays(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    session = Session(source, [selected, FlowError("capture failed")])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a: {"count": 2})
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
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    session = Session(source, [selected])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a: {"count": 2})
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
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    session = Session(source, [selected, first, second])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    def conflicting_count(_session, _plan, snapshot, expected):
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
    assert session.last_snapshot is first  # The second frame cannot wash out strong conflict.
    assert session.inputs == [("tap", plan.balloon_point), ("tap", plan.anchors[0])]


def test_completed_probe_proof_after_six_second_budget_cannot_send_next_input(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    session = Session(source, [selected, first])
    now = [10.]
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    def slow_count(_session, _plan, snapshot, expected):
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
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    first = image_frame(tmp_path, "first", "battle", 10.2)
    second = image_frame(tmp_path, "second", "battle", 10.3)
    session = Session(source, [selected, first, second])
    now = [10.]
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"frame": str(snapshot.screenshot_path),
                            "sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count",
                        lambda _s, _p, _frame, expected: {"count": expected})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    def checkpoint(_session, _plan, snapshot):
        if changed == "budget":
            now[0] = 22.1
        elif changed == "source_pixels":
            source.screenshot_path.write_bytes(b"source changed after both probes")
        else:
            first.screenshot_path.write_bytes(b"first probe proof changed")
        return {"frame": str(snapshot.screenshot_path)}
    monkeypatch.setattr("autococ.probed_anchor._remaining_roster_checkpoint", checkpoint)
    with pytest.raises(DeploymentError) as raised:
        execute_probed_anchor_line(session, plan, clock=lambda: now[0])
    receipt = raised.value.partial_receipt
    assert receipt["attempted_placements"] == receipt["issued_placements"] == 2
    assert receipt["input_sent"] is False
    assert [point for _, point in session.inputs] == [plan.balloon_point, *plan.anchors]


def test_probe_stop_propagates_without_replay(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    session = Session(source, [selected, StopRequested("stop")])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a: {"count": 2})
    monkeypatch.setattr("autococ.battle_vision.selected_card_bbox", lambda *_a: plan.balloon_bbox)
    with pytest.raises(StopRequested):
        execute_probed_anchor_line(session, plan, clock=lambda: 10.)
    assert len(session.inputs) == 2


def test_uncertain_probe_transport_records_exact_attempt_without_retry(monkeypatch, tmp_path):
    plan, source = make_plan(monkeypatch, tmp_path)
    selected = image_frame(tmp_path, "selected", "enemy_village", 10.1)
    session = Session(source, [selected])
    monkeypatch.setattr("autococ.probed_anchor._current_frame",
                        lambda _s, _p, snapshot, **_kw: {"sha256": _file_sha256(snapshot.screenshot_path)})
    monkeypatch.setattr("autococ.probed_anchor._balloon_count", lambda *_a: {"count": 2})
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
    monkeypatch.setattr("autococ.probed_anchor._current_card_edges",
                        lambda _image, _bbox: {"left_edge": 300, "right_edge": -300, "bottom_contrast": 50})
    from autococ.unit_catalog import CATALOG_ROOT
    cards = {tuple(row[4]): row[0] for row in plan.base.troop_cards}
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
