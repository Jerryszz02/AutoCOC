from dataclasses import replace

import pytest

from autococ.dynamic_tracking import DynamicTracker
from autococ.dynamic_vision import DynamicFrame, DynamicObservation, DynamicUnavailable


def frame(index, *, epoch="c1", stable=True):
    return DynamicFrame("session", epoch, f"frame-{index}", float(index), "18.647.2",
                        100, 100, stable, True)


def obs(index, label, *, target="hall", box=(10, 10, 30, 30), kind="r_building", epoch="c1"):
    roi = (max(0, box[0] - 5), max(0, box[1] - 5),
           min(100, box[2] + 5), min(100, box[3] + 5))
    return DynamicObservation(kind, label, box, .99, "session", epoch, f"frame-{index}",
                              float(index), roi, target, "a" * 64, "b" * 64)


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
    assert not tracker.update(frame(1), [obs(1, "available", target="queen-skill", kind="r_card")], offline_replay=True).events
    events = tracker.update(frame(2), [obs(2, "available", target="queen-skill", kind="r_card")], offline_replay=True).events
    assert [event.kind for event in events] == ["skill_available"]


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
