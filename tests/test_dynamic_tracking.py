from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from autococ.dynamic_tracking import DynamicTracker, HeroDeploymentEvidence, read_hero_deployment_evidence
from autococ.dynamic_vision import DynamicFrame, DynamicObservation, DynamicUnavailable


def frame(index, *, epoch="c1", stable=True):
    return DynamicFrame("session", epoch, f"frame-{index}", float(index), "18.647.2",
                        100, 100, stable, True, "c" * 64)


def obs(index, label, *, target="hall", box=(10, 10, 30, 30), kind="r_building", epoch="c1"):
    roi = (max(0, box[0] - 5), max(0, box[1] - 5),
           min(100, box[2] + 5), min(100, box[3] + 5))
    return DynamicObservation(kind, label, box, .99, "session", epoch, f"frame-{index}",
                              float(index), roi, target, "a" * 64, "b" * 64, "c" * 64)


def deployment(index, *, hero="archer_queen", epoch="c1", roi=(5, 5, 35, 35),
               confidence=.99):
    return HeroDeploymentEvidence(hero, True, confidence, "session", epoch,
                                  f"frame-{index}", float(index), "18.647.2",
                                  "c" * 64, roi,
                                  "friendly_battle_card_portrait_and_enclosed_hp")


def test_destroyed_requires_prior_visible_and_two_distinct_fresh_frames():
    tracker = DynamicTracker(max_gap_seconds=2)
    assert not tracker.update(frame(1), [obs(1, "rubble")], offline_replay=True).events
    assert not tracker.update(frame(2), [obs(2, "rubble")], offline_replay=True).events
    tracker = DynamicTracker(max_gap_seconds=2)
    visible = tracker.update(frame(1), [obs(1, "intact")], offline_replay=True)
    assert visible.tracked[0].track_id.startswith("session:c1:")
    assert not tracker.update(frame(2), [obs(2, "rubble")], offline_replay=True).events
    event = tracker.update(frame(3), [obs(3, "rubble")], offline_replay=True).events[0]
    assert event.kind == "building_destroyed"
    assert event.first_frame_id == "frame-2" and event.confirmed_frame_id == "frame-3"
    assert not tracker.update(frame(4), [obs(4, "rubble")], offline_replay=True).events


def test_duplicate_frame_miss_camera_epoch_and_unstable_view_cannot_confirm():
    tracker = DynamicTracker(max_gap_seconds=2)
    tracker.update(frame(1), [obs(1, "intact")], offline_replay=True)
    tracker.update(frame(2), [obs(2, "rubble")], offline_replay=True)
    with pytest.raises(DynamicUnavailable, match="Duplicate"):
        tracker.update(frame(2), [obs(2, "rubble")], offline_replay=True)
    tracker.update(frame(3), [], offline_replay=True)
    assert not tracker.update(frame(4), [obs(4, "rubble")], offline_replay=True).events
    assert not tracker.update(frame(5, epoch="c2"), [obs(5, "rubble", epoch="c2")], offline_replay=True).events
    assert not tracker.update(frame(6, epoch="c2", stable=False), [], offline_replay=True).events
    assert not tracker.update(frame(7, epoch="c2"), [obs(7, "rubble", epoch="c2")], offline_replay=True).events


def test_same_class_iou_and_ambiguity_refuse_identity_shortcut():
    tracker = DynamicTracker(max_gap_seconds=2)
    tracker.update(frame(1), [obs(1, "intact")], offline_replay=True)
    moved = tracker.update(frame(2), [obs(2, "rubble", box=(70, 70, 90, 90))], offline_replay=True)
    assert moved.tracked[0].track_id != "session:c1:1"
    assert not tracker.update(frame(3), [obs(3, "rubble", box=(70, 70, 90, 90))], offline_replay=True).events
    tracker = DynamicTracker(max_gap_seconds=2)
    tracker.update(frame(1), [obs(1, "intact")], offline_replay=True)
    with pytest.raises(DynamicUnavailable, match="multiple detections"):
        tracker.update(frame(2), [obs(2, "rubble"), obs(2, "rubble", box=(11, 11, 31, 31))], offline_replay=True)


def test_skill_available_also_needs_two_distinct_frames():
    tracker = DynamicTracker(max_gap_seconds=2)
    skill = lambda index: obs(index, "available", target="hero_skill:archer_queen", kind="r_card")
    assert not tracker.update(frame(1), [skill(1)], offline_replay=True).events
    assert not tracker.update(frame(2), [skill(2)], hero_deployments=[deployment(2)], offline_replay=True).events
    events = tracker.update(frame(3), [skill(3)], hero_deployments=[deployment(3)], offline_replay=True).events
    assert [event.kind for event in events] == ["skill_available"]
    assert events[0].first_frame_id == "frame-2"


def test_skill_rejects_troop_unknown_hero_and_wrong_deployment_provenance():
    for target in ("electro_dragon", "hero_skill:invented", "hero_skill:electro_dragon"):
        tracker = DynamicTracker(max_gap_seconds=2)
        with pytest.raises(DynamicUnavailable, match="valid current source"):
            tracker.update(frame(1), [obs(1, "available", target=target, kind="r_card")],
                           offline_replay=True)
    skill = lambda index: obs(index, "available", target="hero_skill:archer_queen", kind="r_card")
    tracker = DynamicTracker(max_gap_seconds=2)
    tracker.update(frame(1), [skill(1)], hero_deployments=[deployment(1)], offline_replay=True)
    for invalid in (replace(deployment(2), frame_id="frame-1"),
                    replace(deployment(2), source_sha256="d" * 64),
                    replace(deployment(2), confidence=.8),
                    replace(deployment(2), deployed=False)):
        with pytest.raises(DynamicUnavailable, match="Hero deployment evidence"):
            tracker.update(frame(2), [skill(2)], hero_deployments=[invalid], offline_replay=True)
        tracker = DynamicTracker(max_gap_seconds=2)
        tracker.update(frame(1), [skill(1)], hero_deployments=[deployment(1)], offline_replay=True)
    assert not tracker.update(frame(2), [skill(2)], hero_deployments=[
        replace(deployment(2), hero_id="grand_warden")], offline_replay=True).events
    assert not tracker.update(frame(3), [skill(3)], hero_deployments=[deployment(3)], offline_replay=True).events
    assert not tracker.update(frame(4), [skill(4)], hero_deployments=[
        replace(deployment(4), roi_bbox=(1, 1, 20, 20))], offline_replay=True).events


def test_original_frame_hero_reader_requires_portrait_and_hp(monkeypatch):
    with TemporaryDirectory() as temp:
        source = Path(temp) / "synthetic.png"
        source.write_bytes(cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))[1].tobytes())
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        current = replace(frame(1), source_sha256=digest)
        result = {"deployed": True, "frame": str(source), "slot_bbox": [5, 5, 35, 35],
                  "evidence": {"portrait_identity": {
            "unit_id": "archer_queen", "version_verified": True, "confidence": .99},
            "health_bar": {"method": "enclosed_hp"}}}
        monkeypatch.setattr("autococ.hero_state.recognize_hero_state", lambda *_args, **_kwargs: result)
        assert read_hero_deployment_evidence(source, current, (5, 5, 35, 35), "archer_queen") is not None
        assert read_hero_deployment_evidence(source, replace(current, source_sha256="0" * 64),
                                             (5, 5, 35, 35), "archer_queen") is None
        for bad in ({"deployed": None}, {"evidence": {"portrait_identity": {
                "unit_id": "barbarian_king", "version_verified": True, "confidence": .99},
                "health_bar": {"method": "enclosed_hp"}}},
                    {"evidence": {"portrait_identity": {
                        "unit_id": "archer_queen", "version_verified": True, "confidence": .94},
                        "health_bar": {"method": "enclosed_hp"}}},
                    {"evidence": {"portrait_identity": {
                        "unit_id": "archer_queen", "version_verified": True, "confidence": .99},
                        "health_bar": None}}):
            evidence = {**result, **bad}
            monkeypatch.setattr("autococ.hero_state.recognize_hero_state",
                                lambda *_args, evidence=evidence, **_kwargs: evidence)
            assert read_hero_deployment_evidence(source, current, (5, 5, 35, 35), "archer_queen") is None


@pytest.mark.parametrize("changed", ["client_version", "model_sha256", "contract_sha256"])
def test_skill_streak_never_combines_client_or_model_contracts(changed):
    tracker = DynamicTracker(max_gap_seconds=2)
    for index in (1, 2, 3):
        current = frame(index)
        reading = obs(index, "available", target="hero_skill:archer_queen", kind="r_card")
        deployed = deployment(index)
        if index >= 2:
            if changed == "client_version":
                current = replace(current, client_version="different-client")
                deployed = replace(deployed, client_version=current.client_version)
            else:
                reading = replace(reading, **{changed: "d" * 64})
        step = tracker.update(current, [reading], hero_deployments=[deployed], offline_replay=True)
        if index < 3:
            assert not step.events
        else:
            assert len(step.events) == 1
            assert step.events[0].first_frame_id == "frame-2"


def test_fresh_confident_source_bound_evidence_is_required_for_event():
    tracker = DynamicTracker(max_gap_seconds=2)
    with pytest.raises(DynamicUnavailable, match="stale"):
        tracker.update(frame(1), [obs(1, "intact")], now_monotonic=2)
    tracker.update(frame(2), [obs(2, "intact")], now_monotonic=2.1)
    assert not tracker.update(frame(3), [replace(obs(3, "rubble"), confidence=.7)],
                              offline_replay=True).events
    assert not tracker.update(frame(4), [obs(4, "rubble")], offline_replay=True).events
    assert tracker.update(frame(5), [obs(5, "rubble")], offline_replay=True).events
    for invalid in (replace(obs(6, "rubble"), confidence=float("nan")),
                    replace(obs(6, "rubble"), model_sha256=""),
                    replace(obs(6, "rubble"), contract_sha256="invalid"),
                    replace(obs(6, "rubble"), bbox=(10, 10, 200, 210)),
                    replace(obs(6, "rubble"), roi_bbox=(0, 0, 12, 12))):
        isolated = DynamicTracker()
        with pytest.raises(DynamicUnavailable, match="valid current source"):
            isolated.update(frame(6), [invalid], offline_replay=True)


def test_ambiguous_intervening_frame_clears_confirmation_streak():
    tracker = DynamicTracker(max_gap_seconds=2)
    tracker.update(frame(1), [obs(1, "intact")], offline_replay=True)
    tracker.update(frame(2), [obs(2, "rubble")], offline_replay=True)
    with pytest.raises(DynamicUnavailable, match="multiple detections"):
        tracker.update(frame(3), [obs(3, "rubble"), obs(3, "rubble", box=(11, 11, 31, 31))],
                       offline_replay=True)
    assert not tracker.update(frame(4), [obs(4, "rubble")], offline_replay=True).events
    assert not tracker.update(frame(5), [obs(5, "rubble")], offline_replay=True).events
