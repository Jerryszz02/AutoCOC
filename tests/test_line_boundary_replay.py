"""Read-only replay of the selected-card boundary from the third daily smoke."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autococ.battle_vision import selected_card_bbox
from autococ.deployment import _prepare_two_edge_view
from autococ.scene import SceneSnapshot
from autococ.strategy_execution import _verify_captured_stacks
from autococ.terrain import find_line_deployment_edges
from autococ.vision import ScreenshotRecognizer


REPORT = (Path(__file__).resolve().parents[1] / "reports" / "daily-smoke-3" /
          "20260926-162349-874443-41152517")
SANITIZED = Path(__file__).resolve().parent / "fixtures"


def test_tracked_sanitized_card_bars_confirm_selected_and_unselected_geometry():
    """Only the 565:720 card bar is retained; no account or opponent map is shipped."""
    from autococ.images import read_frame

    plain = SANITIZED / "smoke3_card_bar_unselected.png"
    selected = SANITIZED / "smoke3_card_bar_selected.png"
    for path in (plain, selected):
        assert path.is_file()
        assert not read_frame(path)[:565].any()
    boxes = {"first_troop": [16, 595, 104, 711], "other_troop": [306, 595, 394, 711],
             "siege": [420, 595, 509, 711], "hero": [521, 595, 610, 711],
             "spell": [946, 595, 1034, 711]}
    assert all(selected_card_bbox(plain, box) is None for box in boxes.values())
    assert selected_card_bbox(selected, boxes["first_troop"]) == (12, 588, 107, 711)
    assert all(selected_card_bbox(selected, boxes[kind]) is None for kind in
               ("other_troop", "siege", "hero", "spell"))


def _snapshots():
    if not (REPORT / "events.jsonl").is_file():
        pytest.skip("Optional smoke3 battle frames are absent")
    observed = {}
    for line in (REPORT / "events.jsonl").open(encoding="utf-8"):
        item = json.loads(line)
        if item.get("kind") == "observation":
            path = Path(item["frame"])
            if path.name.startswith(("00029-", "00030-", "00031-", "00032-", "00033-", "00034-")):
                observed[path.name] = SceneSnapshot(item["scene"], item["confidence"], path,
                                                     item["observations"])
    if len(observed) != 6 or any(not item.screenshot_path.is_file() for item in observed.values()):
        pytest.skip("Optional smoke3 battle frames are incomplete")
    return observed


def test_selected_outline_only_on_later_real_boundary_frames():
    frames = _snapshots()
    slot = frames["00031-line-zoom-verified.png"].observations["battle"]["slots"][0]
    for name in ("00031-line-zoom-verified.png", "00032-line-boundary.png"):
        assert selected_card_bbox(frames[name].screenshot_path, slot["bbox"]) is None
    for name in ("00033-line-boundary.png", "00034-line-boundary.png"):
        frame = frames[name]
        selected = selected_card_bbox(frame.screenshot_path, slot["bbox"])
        assert selected is not None
        assert abs((selected[0] + selected[2]) / 2 - slot["point"][0]) <= 7
        assert {item["edge"] for item in find_line_deployment_edges(frame.screenshot_path)} == {0, 1}
        other = next(card for card in frame.observations["battle"]["slots"]
                     if card["kind"] == "troop")
        assert selected_card_bbox(frame.screenshot_path, other["bbox"]) is None


def test_real_boundary_replay_keeps_independent_portrait_and_count_checks(monkeypatch):
    frames = _snapshots()
    monkeypatch.setattr("autococ.deployment.time", SimpleNamespace(sleep=lambda _: None))
    recognizer = ScreenshotRecognizer()
    recognizer.client_version = "18.600.7"
    # Archived event observations have the old incomplete two-card scan.
    # Re-run current recognition on the original frames for this optional replay.
    ordered = [recognizer.recognize(frames[f"{number:05d}-{label}.png"].screenshot_path)
               for number, label in (
        (30, "line-zoom-out"), (31, "line-zoom-verified"),
        (32, "line-boundary"), (33, "line-boundary"))]

    class ReplaySession:
        native = SimpleNamespace(zoom_out=lambda: None)
        config = SimpleNamespace(game=SimpleNamespace(baseline_resolution=(1280, 720)))
        action_count = 0

        def __init__(self):
            self.frames = iter(ordered)
            self.recognizer = recognizer
            self.taps = []

        def _validate_snapshot(self, frame):
            assert frame.scene == "enemy_village"

        def observe(self, label):
            return next(self.frames)

        def tap(self, frame, point, *, reason):
            self.taps.append((frame.screenshot_path.name, point, reason))

        def event(self, kind, **data):
            pass

    session = ReplaySession()
    scout = recognizer.recognize(frames["00029-scout-camera-ready.png"].screenshot_path)
    result, terrain = _prepare_two_edge_view(session, scout,
                                             {"evidence": []})
    assert result.screenshot_path.name == "00033-line-boundary.png"
    assert {item["edge"] for item in terrain} == {0, 1}
    assert len(session.taps) == 1  # Select only; no field placement.
    slots = [item for item in result.observations["battle"]["slots"] if item["kind"] == "troop"]
    assert sorted(item["count"] for item in slots) == [1, 1, 2, 8]
    for slot in slots:
        assert slot["evidence"]["portrait_score"] >= .92
        assert slot["evidence"]["count"]["confidence"] >= .9
    original = tuple(slot for slot in scout.observations["battle"]["slots"]
                     if slot["kind"] == "troop" and
                     slot.get("source", "army") in {"army", "event"} and
                     type(slot.get("count")) is int and slot["count"] > 0)
    assert sorted(slot["count"] for slot in original) == [1, 1, 2, 8]
    _verify_captured_stacks(result, original, after_reveal=True)
